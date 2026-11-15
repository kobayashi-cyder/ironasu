"""Pure stdlib tests; all timings simulated for deterministic, offline CI."""
import unittest
from ironasu.tuner import (
    Candidate, Observation, candidates, choose_hot_stages,
    select_safe_candidate, tune,
)


def obs(text, seconds, trace=None):
    return Observation(text, seconds, trace or {"per_stage": {}})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.plans = [Candidate("baseline"), Candidate("prefetch", prefetch=True)]

    def test_only_verifiably_faster_candidate_is_selected(self):
        r = select_safe_candidate({"baseline": [obs("hello", 10), obs("hello", 10)],
             "prefetch": [obs("hello", 6), obs("hello", 6.5)]}, self.plans,
             baseline_after=[obs("hello", 10), obs("hello", 10)])
        self.assertEqual(r["selected"], "prefetch")
        self.assertGreater(r["measured_gain"], .3)

    def test_changed_output_gets_rejected_even_if_fast(self):
        r = select_safe_candidate({"baseline": [obs("right", 10)],
                 "prefetch": [obs("wrong", 1)]}, self.plans,
                 baseline_after=[obs("right", 10)])
        self.assertEqual(r["selected"], "baseline")
        self.assertEqual(r["trials"][1]["status"], "rejected_output_mismatch")

    def test_slow_candidate_is_rejected(self):
        r = select_safe_candidate({"baseline": [obs("same", 10)],
                 "prefetch": [obs("same", 9.5)]}, self.plans,
                 baseline_after=[obs("same", 10)], min_gain=.10)
        self.assertEqual(r["selected"], "baseline")

    def test_instability_rejects_all_optimizations(self):
        r = select_safe_candidate({"baseline": [obs("a", 10)],
                 "prefetch": [obs("a", 1)]}, self.plans,
                 baseline_after=[obs("b", 10)])
        self.assertFalse(r["baseline_stable"])
        self.assertEqual(r["selected"], "baseline")

    def test_missing_runs_rejected(self):
        r = select_safe_candidate({"baseline": [obs("x", 3), obs("x", 3)],
                 "prefetch": [obs("x", 1)]}, self.plans,
                 baseline_after=[obs("x", 3)])
        self.assertEqual(r["trials"][1]["status"], "rejected_incomplete_runs")

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            Candidate("cache", cache_mb=0, pin_stages=("layer.0",))
        with self.assertRaises(ValueError):
            select_safe_candidate({"baseline": [obs("x", 0)]}, self.plans)
        with self.assertRaises(ValueError):
            candidates(cache_mb=-1)

    def test_cache_is_opt_in(self):
        self.assertEqual([x.name for x in candidates()], ["baseline", "prefetch"])
        self.assertEqual(len(candidates(cache_mb=512)), 4)


class PlanningTests(unittest.TestCase):
    def test_hot_stage_plan_is_capacity_bounded(self):
        trace = {"per_stage": {
            "a": {"loads": 5, "disk_seconds": 4.0},
            "b": {"loads": 5, "disk_seconds": 10.0},
            "c": {"loads": 0, "disk_seconds": 20.0}}}
        self.assertEqual(choose_hot_stages(trace, {"a": 4, "b": 12, "c": 1}, 12), ("a",))
        self.assertEqual(choose_hot_stages(trace, {"a": 4, "b": 12}, 16), ("a", "b"))

    def test_tuner_rechecks_reference_and_rolls_back_on_mismatch(self):
        calls = {"baseline": 0}
        class Runner:
            def __init__(self, c):
                self.config = c
            def __call__(self):
                calls[self.config.name] = calls.get(self.config.name, 0) + 1
                if self.config.name == "baseline" and calls["baseline"] == 2:
                    return obs("different", 10)
                return obs("same", 2 if self.config.prefetch else 10)
            def close(self):
                pass
        report = tune(Runner, repeats=1)
        self.assertEqual(report["selected"], "baseline")
        self.assertFalse(report["baseline_stable"])

    def test_tuner_tolerates_failure_and_rejects_it(self):
        class Runner:
            def __init__(self, config):
                self.config = config
            def __call__(self):
                if self.config.prefetch:
                    raise OSError("synthetic read failure")
                return obs("same", 10)
            def close(self):
                pass
        report = tune(Runner, repeats=2)
        self.assertEqual(report["selected"], "baseline")
        self.assertIn("prefetch", report["errors"])

    def test_tuner_can_select_hot_cache_from_trace(self):
        trace = {"per_stage": {"model.layers.1": {"loads": 5, "disk_seconds": 2.0}}}
        class Runner:
            def __init__(self, config):
                self.config = config
            def __call__(self):
                return obs("same", 4 if self.config.name == "hot_cache" else 10, trace)
            def close(self):
                pass
        r = tune(Runner, cache_mb=1, stage_bytes={"model.layers.1": 1024}, repeats=2)
        self.assertEqual(r["selected"], "hot_cache")
        self.assertEqual(r["hot_stages"], ["model.layers.1"])


if __name__ == "__main__":
    unittest.main()
