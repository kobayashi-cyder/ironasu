"""Contracts shared by the formerly separate ironasu feature branches."""
import unittest
from pathlib import Path

from ironasu import (
    AdaptiveEngine, AdaptiveSpeculationScheduler, Config,
    NgramDraftIndex, StateStore, SpeculativeInitializer,
)
from ironasu.harness import Harness, Settings
from ironasu.server import serve


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def complete(self, payload):
        self.calls.append(payload)
        return {
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "t1", "type": "function",
                                "function": {"name": "list_files", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}
        }


class IntegrationContracts(unittest.TestCase):
    def test_public_imports(self):
        self.assertTrue(callable(serve))
        self.assertTrue(callable(Harness))
        self.assertTrue(callable(SpeculativeInitializer))
        self.assertTrue(callable(AdaptiveSpeculationScheduler))

    def test_packaging_provides_separate_commands(self):
        data = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        self.assertIn('ironasu = "ironasu.server:serve"', data)
        self.assertIn('ironasu-harness = "ironasu.harness:main"', data)

    def test_tool_calls_pass_through_gateway_uncached(self):
        backend = _FakeBackend()
        engine = AdaptiveEngine(Config(), backend=backend)
        request = {
            "messages": [{"role": "user", "content": "Inspect project files"}],
            "tools": [{"type": "function", "function": {"name": "list_files",
                         "parameters": {"type": "object", "properties": {}}}}],
            "temperature": 0,
        }
        first = engine.complete(request)
        second = engine.complete(request)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(first["choices"][0]["message"]["tool_calls"][0]["id"], "t1")
        self.assertEqual(second["choices"][0]["message"]["tool_calls"][0]["id"], "t1")

    def test_speculation_is_off_without_target_verifier(self):
        policy = AdaptiveSpeculationScheduler()
        for _ in range(3):
            policy.record_baseline(1.0, 1)
        choice = policy.choose(backend_supports_speculation=False)
        self.assertEqual(choice.draft_tokens, 0)
        index = NgramDraftIndex(order=2)
        index.remember([1, 2, 3])
        self.assertEqual(index.propose([1, 2], 1), (3,))

    def test_speculative_state_is_separate_from_model_state(self):
        store = StateStore()
        b = store.fork()
        b.write("hint", b"value")
        self.assertTrue(store.try_promote(b, verify=lambda x: x.get("hint") == b"value"))
        store.reset()
        self.assertIsNone(store.snapshot().get("hint"))


if __name__ == "__main__":
    unittest.main()
