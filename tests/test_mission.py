"""Mission Controller: plan DAG, verification, safety, crash/restart and repairs."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
import time
import unittest

from ironasu.harness import Settings
from ironasu.mission import MissionController, _exclusive, parse_plan


class FakeBackend:
    def __init__(self, state):
        self.state = state

    def complete(self, messages, tools):
        with self.state["lock"]:
            self.state["active"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["active"])
            self.state["calls"] += 1
        try:
            time.sleep(0.003)
            text = str(messages[-1].get("content", ""))
            if not tools:
                return {"content": json.dumps({"tasks": [
                    {"task_id": "inspect", "prompt": "Inspect", "depends_on": []},
                    {"task_id": "implement", "prompt": "Implement",
                     "depends_on": ["inspect"]}]})}
            if messages[-1].get("role") == "tool":
                return {"content": "Done; see verification results."}
            if ("A requested acceptance check failed" in text or
                    "Create file" in text or "Implement" in text and self.state["edit"]):
                return {"content": "", "tool_calls": [{
                    "id": "create_result",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": json.dumps({
                        "path": "result.txt", "content": "verified deliverable"})}
                }]}
            return {"content": "Read-only research completed; verification is independent."}
        finally:
            with self.state["lock"]:
                self.state["active"] -= 1


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {"lock": Lock(), "active": 0, "peak": 0,
                      "calls": 0, "edit": False}
        self.controller = MissionController(
            self.root, endpoint="http://127.0.0.1:1234/v1",
            model="local-model", backend_factory=lambda settings: FakeBackend(self.state))

    def test_auto_plan_dependencies_and_persisted_resumption(self):
        (self.root / "existing.txt").write_text("existing", encoding="utf-8")
        result = self.controller.run(
            goal="Inspect the project", checks=("compileall",),
            accept_files=("existing.txt",), allow_tests=True,
            workers=4, model_slots=2)
        self.assertTrue(result.verified, result)
        self.assertEqual([t["task_id"] for t in result.tasks], ["inspect", "implement"])
        self.assertEqual({t["state"] for t in result.tasks}, {"succeeded"})
        calls = self.state["calls"]
        restored = MissionController(
            self.root, endpoint="http://127.0.0.1:1234/v1",
            model="local-model", backend_factory=lambda s: FakeBackend(self.state))
        second = restored.run(mission_id=result.mission_id, resume=True,
                              allow_tests=False)
        self.assertTrue(second.verified)
        self.assertEqual(self.state["calls"], calls)
        self.assertIn("verified", [e["kind"] for e in
                                  restored.journal.events(result.mission_id)])

    def test_no_permissions_does_not_write_or_verify_by_fiat(self):
        report = self.controller.run(
            goal="Create file", auto_plan=False, max_repairs=3,
            checks=("compileall",), accept_files=("result.txt",))
        self.assertEqual(report.state, "needs_permission")
        self.assertFalse(report.verified)
        self.assertFalse((self.root / "result.txt").exists())

    def test_explicit_permissions_can_repair_failed_acceptance(self):
        report = self.controller.run(
            goal="Produce result.txt", auto_plan=False, max_repairs=2,
            checks=("compileall",), accept_files=("result.txt",),
            allow_writes=True, allow_tests=True)
        self.assertTrue(report.verified, report)
        self.assertEqual(report.repairs, 1)
        self.assertEqual((self.root / "result.txt").read_text(),
                         "verified deliverable")
        self.assertTrue(any(e["kind"] == "repair_started" for e in
                            self.controller.journal.events(report.mission_id)))

    def test_no_success_without_observable_verification(self):
        report = self.controller.run(
            goal="Report current status", auto_plan=False,
            checks=(), accept_files=())
        self.assertFalse(report.verified)
        self.assertEqual(report.state, "unverified")

    def test_explicit_dag_with_max_16_agents_has_bounded_model_calls(self):
        (self.root / "result.txt").write_text("ready")
        tasks = [{"task_id": "t" + str(i), "prompt": "Research " + str(i),
                  "depends_on": []} for i in range(16)]
        report = self.controller.run(
            goal="Research 16 areas", plan=tasks,
            workers=16, model_slots=2, allow_tests=True,
            checks=("compileall",), accept_files=("result.txt",))
        self.assertTrue(report.verified, report)
        self.assertEqual(len(report.tasks), 16)
        self.assertLessEqual(self.state["peak"], 2)

    def test_running_side_effect_not_automatically_replayed_after_crash(self):
        from ironasu.harness import SYSTEM, Store
        from ironasu.team import TeamTask
        mission_id = "crash_case"
        self.controller.journal.create(
            mission_id, "Do work", self.controller.endpoint,
            self.controller.model, (TeamTask("work", "Do work"),),
            ("compileall",), ())
        store = Store(self.root)
        call = {"id": "call1", "type": "function", "function": {
            "name": "write_file", "arguments": '{"path":"result.txt","content":"ok"}'}}
        sid = "m_" + mission_id + "_work"
        store.save(sid, self.root, "ready",
                   [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Do work"},
                    {"role": "assistant", "content": "", "tool_calls": [call]}],
                   [call])
        store.start(sid, call)
        store.conn.close()
        result = self.controller.run(
            mission_id=mission_id, resume=True,
            allow_writes=True, allow_tests=True)
        self.assertEqual(result.state, "needs_review")
        self.assertEqual(result.tasks[0]["state"], "uncertain")
        self.assertFalse((self.root / "result.txt").exists())

    def test_cancelled_mission_does_no_model_work(self):
        stop = Event()
        stop.set()
        result = self.controller.run(
            goal="Something", auto_plan=False, cancel=stop)
        self.assertEqual(result.state, "cancelled")
        self.assertEqual(self.state["calls"], 0)

    def test_mission_lock_rejects_second_runner(self):
        with _exclusive(self.root, "same"):
            with self.assertRaises(RuntimeError):
                with _exclusive(self.root, "same"):
                    pass

    def test_invalid_plan_and_verification_paths(self):
        with self.assertRaises(ValueError):
            parse_plan([{"task_id": "a", "prompt": "a",
                         "depends_on": ["missing"]}])
        with self.assertRaises(ValueError):
            self.controller.run(goal="x", auto_plan=False,
                                accept_files=("../outside",))
        with self.assertRaises(ValueError):
            self.controller.run(goal="x", auto_plan=False, workers=17)


if __name__ == "__main__":
    unittest.main()
