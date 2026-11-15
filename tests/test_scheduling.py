"""Tests for experimental scheduling primitives; no model hardware required."""
import threading
import time
import unittest

from ironasu.scheduling import (
    AdaptiveSpeculationScheduler,
    LayerSpec,
    NgramDraftIndex,
    OrderedLayerPrefetcher,
)


class DraftIndexTests(unittest.TestCase):
    def test_known_suffix_only(self):
        idx = NgramDraftIndex(order=2, max_entries=5, max_draft=4)
        idx.remember([1, 2, 3, 4, 5, 6])
        self.assertEqual(idx.propose([0, 1, 2], 3), (3, 4, 5))
        self.assertEqual(idx.propose([8, 9], 3), ())
        self.assertEqual(idx.propose([1], 3), ())
        self.assertEqual(idx.propose([1, 2], 0), ())

    def test_lru_bound(self):
        idx = NgramDraftIndex(order=1, max_entries=2)
        idx.remember([1, 2, 3, 4])
        self.assertEqual(len(idx), 2)
        self.assertEqual(idx.propose([1], 2), ())
        self.assertEqual(idx.propose([3], 2), (4,))


class AdaptiveTests(unittest.TestCase):
    def test_gating_and_baseline(self):
        s = AdaptiveSpeculationScheduler(draft_lengths=(2,), min_samples=2)
        self.assertEqual(s.choose(backend_supports_speculation=True).draft_tokens, 0)
        s.record_baseline(2, 1)
        self.assertEqual(s.choose(backend_supports_speculation=True).draft_tokens, 0)
        s.record_baseline(2, 1)
        self.assertEqual(s.choose(backend_supports_speculation=False).draft_tokens, 0)
        self.assertEqual(s.choose(backend_supports_speculation=True, queue_busy=True).draft_tokens, 0)
        self.assertEqual(s.choose(backend_supports_speculation=True, memory_headroom=False).draft_tokens, 0)
        self.assertEqual(s.choose(backend_supports_speculation=True).draft_tokens, 2)

    def test_measured_improvement(self):
        s = AdaptiveSpeculationScheduler(draft_lengths=(2,), min_samples=2)
        s.record_baseline(2, 1)
        s.record_baseline(2, 1)
        self.assertEqual(s.choose(backend_supports_speculation=True).draft_tokens, 2)
        s.record_speculative(2, 2, 3, 3)
        s.record_speculative(2, 2, 3, 3)
        choice = s.choose(backend_supports_speculation=True)
        self.assertEqual(choice.draft_tokens, 2)
        self.assertAlmostEqual(choice.estimated_speedup, 2.0)

    def test_regression_and_invalid_data(self):
        s = AdaptiveSpeculationScheduler(draft_lengths=(2,), min_samples=2)
        s.record_baseline(1, 1)
        s.record_baseline(1, 1)
        s.record_speculative(2, 0, 3, 1)
        s.record_speculative(2, 0, 3, 1)
        self.assertEqual(s.choose(backend_supports_speculation=True).draft_tokens, 0)
        with self.assertRaises(ValueError):
            s.record_speculative(2, 1, -1, 2)
        with self.assertRaises(ValueError):
            s.record_speculative(2, 1, 1, 3)


class PrefetchTests(unittest.TestCase):
    def test_layer_order_and_state(self):
        pipeline = OrderedLayerPrefetcher(max_inflight=2, max_staged_bytes=20)
        layers = [LayerSpec(i, 10) for i in range(5)]
        result = pipeline.run(layers, lambda x: x * 2,
                              lambda layer, weight, acc: acc + [(layer, weight)], [])
        self.assertEqual(result, [(i, i * 2) for i in range(5)])

    def test_next_load_starts_before_current_compute_finishes(self):
        ready = threading.Event()
        def load(i):
            if i == 1:
                ready.set()
            return i
        def apply(i, weight, state):
            if i == 0:
                self.assertTrue(ready.wait(1.0))
            return state + [i]
        p = OrderedLayerPrefetcher(max_inflight=2, max_staged_bytes=20)
        self.assertEqual(p.run([LayerSpec(0, 10), LayerSpec(1, 10)], load, apply, []), [0, 1])

    def test_staging_budget_and_oversized_layer(self):
        p = OrderedLayerPrefetcher(max_inflight=2, max_staged_bytes=10)
        started = [0]
        lock = threading.Lock()
        def load(i):
            with lock:
                started[0] += 1
            return i
        def apply(i, weight, state):
            if i == 0:
                time.sleep(0.02)
                with lock:
                    self.assertEqual(started[0], 1)
            return state + 1
        self.assertEqual(p.run([LayerSpec(0, 10), LayerSpec(1, 10)], load, apply, 0), 2)
        with self.assertRaises(ValueError):
            p.run([LayerSpec(0, 11)], load, apply, 0)


if __name__ == "__main__":
    unittest.main()
