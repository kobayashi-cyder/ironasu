"""Git worktree isolation and independent baseline/candidate verification."""
from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
from threading import Lock
import unittest

from ironasu.mission import MissionController
from ironasu.worktree_mission import GitWorktrees, IsolatedMission, GitConflict


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class WritingModel:
    def __init__(self, settings, stats):
        self.settings, self.stats = settings, stats

    def complete(self, messages, tools):
        with self.stats["lock"]:
            self.stats["calls"] += 1
        if messages[-1].get("role") == "tool":
            return {"content": "Task complete"}
        latest = messages[-1].get("content", "")
        filename = "combined.txt" if "Task C" in latest else (
            "alpha.txt" if "Task A" in latest else "beta.txt")
        if filename == "combined.txt":
            assert (self.settings.workspace / "alpha.txt").is_file()
            assert (self.settings.workspace / "beta.txt").is_file()
        return {"content": "", "tool_calls": [{
            "id": "edit_" + filename, "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({
                    "path": filename, "content": "verified: " + filename + "\n",
                })}}]}


class TestIsolatedMissions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        git(self.root, "config", "user.email", "test@example.invalid")
        git(self.root, "config", "user.name", "Test Agent")
        (self.root / ".gitignore").write_text(".ironasu/\n__pycache__/\n", encoding="utf-8")
        (self.root / "README.md").write_text("baseline\n", encoding="utf-8")
        git(self.root, "add", ".gitignore", "README.md")
        git(self.root, "commit", "-qm", "baseline")
        self.baseline = git(self.root, "rev-parse", "HEAD")
        self.stats = {"lock": Lock(), "calls": 0}
        self.ctrl = MissionController(
            self.root, endpoint="http://127.0.0.1:1234/v1", model="local-coder",
            backend_factory=lambda settings: WritingModel(settings, self.stats),
        )
        self.tasks = [
            {"task_id": "a", "prompt": "Task A", "depends_on": []},
            {"task_id": "b", "prompt": "Task B", "depends_on": []},
            {"task_id": "c", "prompt": "Task C", "depends_on": ["a", "b"]},
        ]

    def launch(self, **overrides):
        opts = dict(goal="Produce checked result", plan=self.tasks,
                    workers=3, model_slots=2, max_steps=4,
                    max_repairs=0, auto_plan=False,
                    checks=("compileall", "git_diff_check"),
                    accept_files=("alpha.txt", "beta.txt", "combined.txt"),
                    allow_writes=True, allow_tests=True, isolated_worktrees=True)
        opts.update(overrides)
        return self.ctrl.run(**opts)

    def test_parallel_dependency_graph_and_verified_candidate_do_not_touch_main(self):
        report = self.launch()
        self.assertTrue(report.verified, report)
        self.assertEqual(report.state, "verified_isolated")
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.baseline)
        for file in ("alpha.txt", "beta.txt", "combined.txt"):
            self.assertFalse((self.root / file).exists())
        self.assertEqual(len(report.tasks), 3)
        self.assertTrue(all(x["state"] == "succeeded" for x in report.tasks))
        info = report.isolation
        self.assertEqual(info["base"], self.baseline)
        self.assertTrue(info["candidate"])
        self.assertFalse(info["baseline"]["passed"])
        self.assertTrue(info["verification"]["candidate"]["passed"])
        stage = Path(info["worktrees"]) / "candidate"
        self.assertEqual((stage / "combined.txt").read_text(), "verified: combined.txt\n")
        self.assertEqual(git(stage, "rev-parse", "HEAD"), info["candidate"])
        self.assertGreater(self.stats["calls"], 0)

        before = self.stats["calls"]
        resumed = self.ctrl.run(mission_id=report.mission_id, resume=True,
                                isolated_worktrees=True, allow_writes=True,
                                allow_tests=True, checks=("compileall",))
        self.assertTrue(resumed.verified)
        self.assertEqual(before, self.stats["calls"])
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.baseline)
        with self.assertRaisesRegex(ValueError, "isolated worktrees"):
            self.ctrl.run(mission_id=report.mission_id, resume=True)

    def test_explicit_approval_fast_forwards_only_a_clean_unchanged_head(self):
        report = self.launch()
        assert report.verified
        result = IsolatedMission(self.ctrl).approve(report.mission_id)
        self.assertEqual(result["state"], "integrated")
        self.assertEqual(git(self.root, "rev-parse", "HEAD"),
                         report.isolation["candidate"])
        self.assertEqual((self.root / "alpha.txt").read_text(), "verified: alpha.txt\n")
        with self.assertRaises(ValueError):
            IsolatedMission(self.ctrl).approve(report.mission_id)

    def test_divergent_or_dirty_main_rejects_approval(self):
        report = self.launch()
        (self.root / "untracked.txt").write_text("do not overwrite")
        with self.assertRaises(ValueError):
            IsolatedMission(self.ctrl).approve(report.mission_id)
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.baseline)
        (self.root / "untracked.txt").unlink()
        (self.root / "new.md").write_text("user change\n")
        git(self.root, "add", "new.md")
        git(self.root, "commit", "-qm", "user change")
        with self.assertRaises(RuntimeError):
            IsolatedMission(self.ctrl).approve(report.mission_id)

    def test_no_write_and_test_permission_is_rejected_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "permissions"):
            self.launch(allow_writes=False)
        with self.assertRaisesRegex(ValueError, "permissions"):
            self.launch(allow_tests=False)
        self.assertEqual(self.stats["calls"], 0)
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.baseline)

    def test_dirty_source_rejected_before_parallel_agents(self):
        (self.root / "README.md").write_text("dirty\n")
        with self.assertRaisesRegex(ValueError, "dirty repository"):
            self.launch()
        self.assertEqual(self.stats["calls"], 0)

    def test_worktree_merge_conflicts_do_not_change_main(self):
        # Simulate two worktrees based on the same SHA, with conflicting edits.
        git_ops = GitWorktrees(self.root, "conflict")
        git_ops.clean_main()
        left = git_ops.checkout("left", self.baseline)
        right = git_ops.checkout("right", self.baseline)
        (left / "README.md").write_text("left\n")
        (right / "README.md").write_text("right\n")
        lsha = git_ops.commit(left, label="left")
        rsha = git_ops.commit(right, label="right")
        candidate = git_ops.checkout("candidate", self.baseline)
        git_ops.merge(candidate, lsha)
        with self.assertRaises(GitConflict):
            git_ops.merge(candidate, rsha)
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.baseline)
        self.assertEqual((self.root / "README.md").read_text(), "baseline\n")

    def test_protected_and_deleted_files_are_not_stageable(self):
        gt = GitWorktrees(self.root, "validate")
        task = gt.checkout("task", self.baseline)
        (task / "README.md").unlink()
        with self.assertRaises(ValueError):
            gt.commit(task, label="no-delete")


if __name__ == "__main__":
    unittest.main()
