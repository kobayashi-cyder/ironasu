"""Run with python -m unittest discover -s tests -v."""
import threading
import unittest

from ironasu.speculative_state import (
    BusySpeculationError, SpeculativeInitializer, StateStore,
)


class TestState(unittest.TestCase):
    def test_reset_snapshot_and_generation(self):
        s = StateStore()
        b = s.fork()
        b.write("x", b"hello")
        self.assertTrue(s.try_promote(b))
        old = s.snapshot()
        new = s.reset()
        self.assertEqual(new.generation, old.generation + 1)
        self.assertIsNone(new.get("x"))
        self.assertEqual(old.get("x"), b"hello")

    def test_reset_invalidates_speculation(self):
        s = StateStore()
        b = s.fork()
        b.write("x", b"old")
        s.reset()
        self.assertFalse(s.try_promote(b))
        self.assertIsNone(s.snapshot().get("x"))

    def test_one_winner_same_base(self):
        s = StateStore()
        a, b = s.fork(), s.fork()
        a.write("x", b"first")
        b.write("x", b"second")
        self.assertTrue(s.try_promote(a))
        self.assertFalse(s.try_promote(b))
        self.assertEqual(s.snapshot().get("x"), b"first")

    def test_race_during_verification(self):
        s = StateStore()
        b = s.fork()
        b.write("x", b"bad")
        def race(_):
            s.reset()
            return True
        self.assertFalse(s.try_promote(b, verify=race))

    def test_validator_rejects_and_seals(self):
        s = StateStore()
        b = s.fork()
        b.write("x", b"invalid")
        self.assertFalse(s.try_promote(b, verify=lambda x: x.get("x") == b"valid"))
        self.assertIsNone(s.snapshot().get("x"))
        with self.assertRaises(RuntimeError):
            b.write("x", b"retry")

    def test_byte_and_key_budgets(self):
        s = StateStore(max_delta_bytes=3, max_delta_keys=2)
        b = s.fork()
        b.write("a", b"123")
        with self.assertRaises(MemoryError):
            b.write("b", b"4")
        with self.assertRaises(TypeError):
            b.write("a", bytearray(b"a"))
        b.write("a", b"1")
        b.write("b", b"23")
        with self.assertRaises(MemoryError):
            b.delete("c")
        self.assertTrue(s.try_promote(b))

    def test_delete_override_and_materialize(self):
        s = StateStore()
        a = s.fork()
        a.write("x", b"x")
        a.write("y", b"y")
        self.assertTrue(s.try_promote(a))
        b = s.fork()
        b.delete("x")
        self.assertTrue(s.try_promote(b))
        self.assertEqual(s.snapshot().materialize(), {"y": b"y"})

    def test_depth_compacted(self):
        s = StateStore()
        for i in range(100):
            b = s.fork()
            b.write(str(i), b"x")
            self.assertTrue(s.try_promote(b))
        snapshot = s.snapshot()
        self.assertLess(snapshot._node.depth, 32)
        self.assertEqual(len(snapshot.materialize()), 100)
        self.assertEqual(snapshot.get("0"), b"x")


class TestSpeculation(unittest.TestCase):
    def test_only_verified_candidate_wins(self):
        s = StateStore()
        with SpeculativeInitializer(s) as init:
            result = init.run({
                "invalid": lambda b: b.write("x", b"bad"),
                "valid": lambda b: b.write("x", b"good"),
            }, verify=lambda _, snap: snap.get("x") == b"good")
        self.assertTrue(result.committed)
        self.assertEqual(result.selected, "valid")
        self.assertEqual(s.snapshot().get("x"), b"good")

    def test_all_rejected(self):
        s = StateStore()
        with SpeculativeInitializer(s) as init:
            result = init.run({"bad": lambda b: b.write("x", b"bad")},
                              verify=lambda _, snap: False)
        self.assertFalse(result.committed)
        self.assertIsNone(s.snapshot().get("x"))

    def test_exception_does_not_mask_valid(self):
        s = StateStore()
        def broken(_):
            raise ValueError("bad")
        with SpeculativeInitializer(s) as init:
            result = init.run({"broken": broken,
                               "valid": lambda b: b.write("x", b"ok")},
                              verify=lambda _, snap: snap.get("x") == b"ok")
        self.assertEqual(result.selected, "valid")

    def test_backpressure_from_running_loser(self):
        s = StateStore()
        entered, release = threading.Event(), threading.Event()
        def slow(b):
            entered.set()
            release.wait(timeout=2)
            b.write("x", b"slow")
        def fast(b):
            self.assertTrue(entered.wait(timeout=2))
            b.write("x", b"fast")
        init = SpeculativeInitializer(s)
        try:
            result = init.run({"slow": slow, "fast": fast},
                              verify=lambda _, snap: snap.get("x") == b"fast")
            self.assertEqual(result.selected, "fast")
            with self.assertRaises(BusySpeculationError):
                init.run({"next": lambda b: None}, verify=lambda _, snap: True)
        finally:
            release.set()
            init.close()

    def test_candidate_cap(self):
        with SpeculativeInitializer(StateStore(), max_candidates=1) as init:
            with self.assertRaises(ValueError):
                init.run({"a": lambda b: None, "b": lambda b: None},
                         verify=lambda _, snap: True)


if __name__ == "__main__":
    unittest.main()
