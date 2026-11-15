import unittest
from ironasu.prefetch_policy import AdaptivePrefetchGate


class GateTests(unittest.TestCase):
    def test_healthy_overlap_keeps_prefetch(self):
        gate = AdaptivePrefetchGate(window=4, cooldown=3)
        for _ in range(4):
            self.assertTrue(gate.should_submit())
            gate.observe(used=True, read_seconds=.10, wait_seconds=.01)
        self.assertTrue(gate.should_submit())
        self.assertEqual(gate.snapshot()["pauses"], 0)

    def test_wasted_work_backoff_and_probe(self):
        gate = AdaptivePrefetchGate(window=4, cooldown=3)
        for _ in range(4):
            self.assertTrue(gate.should_submit())
            gate.observe(used=False, read_seconds=.10)
        self.assertEqual(gate.snapshot()["pauses"], 1)
        self.assertEqual([gate.should_submit() for _ in range(4)],
                         [False, False, False, True])
        self.assertEqual(gate.snapshot()["skipped"], 3)

    def test_cached_read_can_suspend_speculation(self):
        gate = AdaptivePrefetchGate(window=4, cooldown=2)
        for _ in range(4):
            gate.should_submit()
            gate.observe(used=True, read_seconds=0)
        self.assertFalse(gate.should_submit())

    def test_invalid_metrics_rejected(self):
        with self.assertRaises(ValueError):
            AdaptivePrefetchGate(window=1)
        with self.assertRaises(ValueError):
            AdaptivePrefetchGate(cooldown=0)
        with self.assertRaises(ValueError):
            AdaptivePrefetchGate(minimum_overlap=2)
        with self.assertRaises(ValueError):
            AdaptivePrefetchGate().observe(used=True, read_seconds=-.1)
