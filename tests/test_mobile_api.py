"""Mobile API unit tests use a fake runner, no model download or shell execution."""
import time
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
import unittest

from ironasu.core import Config, InferenceError
from ironasu.mobile_api import MobileJobManager
from ironasu.team import TeamOutcome


class MobileJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.manager = MobileJobManager(
            Config(listen_port=8766, max_concurrency=2, server_key="key"),
            workspace=Path(self.temp.name),
            runner=lambda tasks, **kwargs: tuple(TeamOutcome(t.task_id, "succeeded", "ok") for t in tasks))

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def test_16_agents_bounded_to_configured_slots(self):
        job = self.manager.start({"tasks": [{"task_id": f"a{i}", "prompt": "analyze"}
                                              for i in range(16)], "workers": 16, "model_slots": 16})
        self.assertEqual(job["model_slots"], 2)
        for _ in range(100):
            finished = self.manager.get(job["id"])
            if finished["finished_at"] is not None:
                break
            time.sleep(.01)
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(len(finished["results"]), 16)

    def test_invalid_input_and_dependency_cycles(self):
        for p in ({"tasks": []}, {"tasks": [{"task_id": "a", "prompt": "x", "depends_on": ["a"]}]},
                  {"tasks": [{"task_id": "a", "prompt": "x"}], "workers": True},
                  {"tasks": [{"task_id": "a", "prompt": "x"}], "model_slots": 0}):
            with self.assertRaises(InferenceError):
                self.manager.start(p)

    def test_concurrent_job_rejected(self):
        gate = Event()
        self.manager.close()
        def blocking(tasks, **kw):
            gate.wait(timeout=2)
            return (TeamOutcome("a", "succeeded", "done"),)
        self.manager = MobileJobManager(Config(), workspace=Path(self.temp.name), runner=blocking)
        self.manager.start({"tasks": [{"task_id": "a", "prompt": "x"}]})
        with self.assertRaises(InferenceError) as exc:
            self.manager.start({"tasks": [{"task_id": "a", "prompt": "x"}]})
        self.assertEqual(exc.exception.status, 409)
        gate.set()

if __name__ == "__main__":
    unittest.main()