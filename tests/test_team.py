"""Orchestrator tests: DAG order, capacity bounds, failure gating and isolation."""
from threading import Lock
import time
import unittest

from ironasu.team import ModelSlotGate, TeamScheduler, TeamTask


class SchedulerTests(unittest.TestCase):
    def test_dag_and_context_visibility(self):
        tasks = [
            TeamTask("final", "Synthesize", ("a", "b"), priority=10),
            TeamTask("a", "Analyze A", priority=0),
            TeamTask("b", "Analyze B", priority=1),
        ]
        observations = {}

        def worker(task, prompt):
            observations[task.task_id] = prompt
            return "report-" + task.task_id

        results = TeamScheduler(workers=2).run(tasks, worker, shared_context="shared-brief")
        self.assertEqual([x.status for x in results], ["succeeded"] * 3)
        self.assertIn("shared-brief", observations["a"])
        self.assertIn("[a] report-a", observations["final"])
        self.assertIn("[b] report-b", observations["final"])

    def test_failure_blocks_downstream_but_not_independent_tasks(self):
        tasks = [
            TeamTask("a", "bad"),
            TeamTask("b", "dependent", ("a",)),
            TeamTask("c", "independent"),
            TeamTask("d", "depends on blocked", ("b",)),
        ]
        def worker(task, prompt):
            if task.task_id == "a":
                raise RuntimeError("failure")
            return "fine"
        r = TeamScheduler(2).run(tasks, worker)
        self.assertEqual([x.status for x in r], ["failed", "blocked", "succeeded", "blocked"])

    def test_uncertain_and_paused_are_not_success(self):
        r = TeamScheduler().run(
            [TeamTask("a", "x"), TeamTask("b", "y", ("a",))],
            lambda task, prompt: "UNCERTAIN a tool call")
        self.assertEqual([x.status for x in r], ["failed", "blocked"])

    def test_cycle_and_duplicate_ids_rejected(self):
        cases = [
            [TeamTask("x", "x", ("y",)), TeamTask("y", "y", ("x",))],
            [TeamTask("x", "x"), TeamTask("x", "duplicate")],
            [TeamTask("x", "x", ("missing",))],
        ]
        for case in cases:
            with self.assertRaises(ValueError):
                TeamScheduler().run(case, lambda t, p: "ok")
        with self.assertRaises(ValueError):
            TeamScheduler(workers=17)

    def test_16_workers_capacity(self):
        lock = Lock()
        active = [0]
        max_active = [0]
        def work(task, prompt):
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1
            return task.task_id
        tasks = [TeamTask(f"task{i}", "ok") for i in range(20)]
        result = TeamScheduler(workers=16).run(tasks, work)
        self.assertEqual(len(result), 20)
        self.assertLessEqual(max_active[0], 16)
        self.assertGreater(max_active[0], 1)

    def test_llm_gate_two_slots(self):
        lock = Lock()
        active = [0]
        peak = [0]
        class FakeBackend:
            def complete(self, messages, tools):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.02)
                with lock:
                    active[0] -= 1
                return {"role": "assistant", "content": "ok"}
        gate = ModelSlotGate(max_parallel_model_calls=2)
        backend = gate.wrap(FakeBackend())
        TeamScheduler(16).run(
            [TeamTask(f"t{i}", "anything") for i in range(16)],
            lambda task, prompt: backend.complete([], [])["content"])
        self.assertEqual(peak[0], 2)


if __name__ == "__main__":
    unittest.main()
