import threading
import time
import unittest
from ironasu.core import AdaptiveEngine, Config, InferenceError


class FakeBackend:
    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()

    def complete(self, payload):
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(payload)
        return {"id": "fake", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class EngineTests(unittest.TestCase):
    def setup_engine(self, backend=None, **kwargs):
        return AdaptiveEngine(Config(**kwargs), backend or FakeBackend())

    def test_route_short_simple_and_complex(self):
        fake = FakeBackend()
        engine = self.setup_engine(fake)
        engine.complete({"messages": [{"role": "user", "content": "explain this function"}]})
        engine.complete({"messages": [{"role": "user", "content": "review this security architecture"}]})
        self.assertEqual([p["model"] for p in fake.calls], ["qwen2.5-coder-1.5b", "qwen2.5-coder-7b"])

    def test_exact_cache_skips_second_inference(self):
        fake = FakeBackend()
        engine = self.setup_engine(fake)
        q = {"messages": [{"role": "user", "content": "explain a variable"}], "temperature": 0}
        self.assertEqual(engine.complete(q)["_ironasu"]["source"], "upstream")
        self.assertEqual(engine.complete(q)["_ironasu"]["source"], "cache")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(engine.metrics.snapshot()["cache_hits"], 1)

    def test_different_prompts_do_not_collide(self):
        fake = FakeBackend()
        engine = self.setup_engine(fake)
        for prompt in ("explain variable a", "explain variable b"):
            engine.complete({"messages": [{"role": "user", "content": prompt}], "temperature": 0})
        self.assertEqual(len(fake.calls), 2)

    def test_nonzero_temperature_and_tool_calls_do_not_cache(self):
        fake = FakeBackend()
        engine = self.setup_engine(fake)
        p = {"messages": [{"role": "user", "content": "explain a"}], "temperature": 0.8}
        engine.complete(p)
        engine.complete(p)
        q = {**p, "temperature": 0, "tools": [{"type": "function", "function": {"name": "get_clock"}}]}
        engine.complete(q)
        engine.complete(q)
        self.assertEqual(len(fake.calls), 4)

    def test_pressure_uses_small_model_only_for_short_unknown_requests(self):
        fake = FakeBackend(delay=0.02)
        engine = self.setup_engine(fake, slow_request_s=0.005)
        engine.complete({"messages": [{"role": "user", "content": "First complex request"}]})
        self.assertTrue(engine.metrics.snapshot()["pressure"])
        engine.complete({"messages": [{"role": "user", "content": "Hello"}]})
        engine.complete({"messages": [{"role": "user", "content": "security review"}]})
        self.assertEqual([p["model"] for p in fake.calls], ["qwen2.5-coder-7b", "qwen2.5-coder-1.5b", "qwen2.5-coder-7b"])

    def test_forced_quality_wins_under_pressure(self):
        engine = self.setup_engine()
        engine.metrics.pressure = True
        ans = engine.complete({"messages": [{"role": "user", "content": "explain"}], "x_ironasu_mode": "quality"})
        self.assertEqual(ans["_ironasu"]["route"], "quality")

    def test_singleflight_reduces_simultaneous_requests(self):
        fake = FakeBackend(delay=0.1)
        engine = self.setup_engine(fake)
        q = {"messages": [{"role": "user", "content": "explain same"}], "temperature": 0}
        barrier = threading.Barrier(5)
        answers = []
        def worker():
            barrier.wait()
            answers.append(engine.complete(q))
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(answers), 5)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(engine.metrics.snapshot()["singleflight_hits"], 4)

    def test_streaming_rejected_clearly(self):
        engine = self.setup_engine()
        with self.assertRaisesRegex(InferenceError, "not yet supported"):
            engine.complete({"messages": [{"role": "user", "content": "Hi"}], "stream": True})

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            self.setup_engine(listen_host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
