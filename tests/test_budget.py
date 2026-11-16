"""Execution budget policy, gateway integration and evidence accounting tests."""
from __future__ import annotations

import threading
import time
import unittest

from ironasu.budget import RequestBudget
from ironasu.core import AdaptiveEngine, Config, InferenceError


class UsageBackend:
    def __init__(self, *, slow: bool = False):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.slow = slow
        self.lock = threading.Lock()

    def complete(self, request):
        self.entered.set()
        if self.slow and not self.release.wait(3):
            raise RuntimeError("timed out waiting for test release")
        with self.lock:
            self.calls += 1
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"completion_tokens": 7},
        }


def question(text, *, budget=None, cache=False, mode="auto"):
    result = {
        "messages": [{"role": "user", "content": text}],
        "x_ironasu_mode": mode,
        "temperature": 0 if cache else 0.3,
    }
    if budget is not None:
        result["x_ironasu_budget"] = budget
    return result


class RequestBudgetTests(unittest.TestCase):
    def test_no_budget_preserves_existing_policy(self):
        legacy = RequestBudget.from_request({})
        self.assertFalse(legacy.explicit)
        self.assertTrue(legacy.pressure_for_routing(True))
        self.assertEqual(legacy.queue_timeout_s(10), 10)
        self.assertTrue(legacy.allow_cache)

    def test_explicit_defaults_preserve_auto_quality(self):
        budget = RequestBudget.from_request({"x_ironasu_budget": {}})
        self.assertTrue(budget.explicit)
        self.assertFalse(budget.pressure_for_routing(True))
        self.assertTrue(budget.allow_cache)
        self.assertIsNone(budget.max_queue_ms)

    def test_queue_budget_never_extends_configured_limit(self):
        short = RequestBudget.from_request({"x_ironasu_budget": {"max_queue_ms": 250}})
        long = RequestBudget.from_request({"x_ironasu_budget": {"max_queue_ms": 10000}})
        self.assertEqual(short.queue_timeout_s(5), 0.25)
        self.assertEqual(long.queue_timeout_s(5), 5)

    def test_invalid_budget_rejected(self):
        for raw in [None, [], "5", {"max_queue_ms": True}, {"max_queue_ms": -1},
                    {"max_queue_ms": 120001}, {"preserve_auto_quality": 1},
                    {"allow_cache": "false"}, {"max_gpu_bytes": 1000}]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    RequestBudget.from_request({"x_ironasu_budget": raw})


class BudgetGatewayTests(unittest.TestCase):
    def engine(self, backend=None, **config):
        return AdaptiveEngine(Config(**config), backend or UsageBackend())

    def test_quality_guard_preserves_ambiguous_requests_under_pressure(self):
        engine = self.engine()
        engine.metrics.pressure = True
        legacy = engine.complete(question("Hello"))
        guarded = engine.complete(question("Hello", budget={}))
        explicit = engine.complete(question("Hello", budget={}, mode="fast"))
        self.assertEqual(legacy["_ironasu"]["route"], "fast")
        self.assertEqual(guarded["_ironasu"]["route"], "quality")
        self.assertEqual(explicit["_ironasu"]["route"], "fast")
        self.assertTrue(guarded["_ironasu"]["execution_budget"]["preserve_auto_quality"])
        self.assertEqual(engine.diagnostic_snapshot()["execution_budget"]["quality_guard_requested"], 2)

    def test_budget_can_disable_cache_without_changing_other_callers(self):
        backend = UsageBackend()
        engine = self.engine(backend)
        standard = question("explain example", cache=True)
        bypass = question("explain example", cache=True, budget={"allow_cache": False})
        engine.complete(standard)
        engine.complete(standard)
        engine.complete(bypass)
        engine.complete(standard)
        self.assertEqual(backend.calls, 2)
        snap = engine.diagnostic_snapshot()
        self.assertEqual(snap["efficiency"]["observed"]["avoided_upstream_requests"], 2)
        self.assertEqual(snap["efficiency"]["observed"]["reused_reported_output_tokens"], 14)
        self.assertEqual(snap["efficiency"]["derived"]["response_reuse_fraction"], 0.5)
        self.assertEqual(snap["execution_budget"]["cache_bypass_requested"], 1)
        self.assertEqual(snap["efficiency"]["observed"]["successful_upstream_responses"], 2)
        self.assertIn("GPU compute avoided", snap["efficiency"]["unmeasured"])

    def test_zero_queue_budget_does_not_consume_waiting_slot(self):
        backend = UsageBackend(slow=True)
        engine = self.engine(backend, max_concurrency=1, max_waiting_requests=2)
        errors = []
        def first():
            try:
                engine.complete(question("first"))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=first)
        thread.start()
        try:
            self.assertTrue(backend.entered.wait(2))
            started = time.monotonic()
            with self.assertRaises(InferenceError) as error:
                engine.complete(question("second", budget={"max_queue_ms": 0}))
            self.assertEqual(error.exception.status, 503)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(engine.diagnostic_snapshot()["waiting_requests"], 0)
        finally:
            backend.release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_singleflight_timeout_is_not_reported_as_avoided_work(self):
        backend = UsageBackend(slow=True)
        engine = self.engine(backend)
        prompt = question("explain same", cache=True)
        def owner():
            engine.complete(prompt)
        thread = threading.Thread(target=owner)
        thread.start()
        try:
            self.assertTrue(backend.entered.wait(2))
            with self.assertRaises(InferenceError) as error:
                engine.complete(question("explain same", cache=True,
                                         budget={"max_queue_ms": 10}))
            self.assertEqual(error.exception.status, 504)
            self.assertEqual(engine.diagnostic_snapshot()["efficiency"]["observed"]["avoided_upstream_requests"], 0)
        finally:
            backend.release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(backend.calls, 1)

    def test_failed_backend_call_not_counted_as_success_or_savings(self):
        class Fail:
            def complete(self, payload):
                raise RuntimeError("failure")
        engine = self.engine(Fail())
        with self.assertRaises(RuntimeError):
            engine.complete(question("fail", cache=True))
        obs = engine.diagnostic_snapshot()["efficiency"]["observed"]
        self.assertEqual(obs["successful_upstream_responses"], 0)
        self.assertEqual(obs["avoided_upstream_requests"], 0)

    def test_budget_validation_reaches_api_as_bad_request(self):
        engine = self.engine()
        for bad in [[], {"max_queue_ms": True}, {"allow_cache": 1}, {"gpu_bytes": 100}]:
            with self.subTest(bad=bad):
                with self.assertRaises(InferenceError) as ctx:
                    engine.complete(question("Hello", budget=bad))
                self.assertEqual(ctx.exception.status, 400)


if __name__ == "__main__":
    unittest.main()
