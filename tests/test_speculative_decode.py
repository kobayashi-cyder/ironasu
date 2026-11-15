"""Greedy target parity and safe speculative verification contracts."""
import unittest
import time
from types import SimpleNamespace

from ironasu.speculative_decode import (
    GreedySpeculativeDecoder, PagedTorchGreedyTarget, verify_greedy)
from ironasu.scheduling import AdaptiveSpeculationScheduler, NgramDraftIndex


def transition(last):
    return 2 if last == 1 else 1


def target(context, draft):
    sequence = context + draft
    return tuple(transition(x) for x in sequence[-len(draft)-1:])


def baseline(prompt, length):
    result = list(prompt)
    for _ in range(length):
        result.append(target(tuple(result), ())[0])
    return tuple(result[len(prompt):])


class FakeCache:
    def __init__(self, count, fail=False):
        self.count = count
        self.fail = fail

    def get_seq_length(self):
        return self.count

    def crop(self, delta):
        if self.fail:
            raise RuntimeError("crop unsupported")
        if delta >= 0:
            raise ValueError("expects relative negative crop")
        self.count += delta


class FakeTokens:
    def __init__(self, items):
        self.items = items


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values, **kwargs):
        return FakeTokens(values[0])

    @staticmethod
    def ones(shape, **kwargs):
        return FakeTokens([1] * shape[1])

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()


class FakeGreedy:
    def __init__(self, predictions):
        self.predictions = predictions

    def tolist(self):
        return list(self.predictions)


class FakeSlice:
    def __init__(self, predictions):
        self.predictions = predictions

    def argmax(self, dim=-1):
        return FakeGreedy(self.predictions)


class FakeLogits:
    ndim = 3

    def __init__(self, predictions):
        self.predictions = predictions
        self.shape = (1, len(predictions), 3)

    def __getitem__(self, key):
        return FakeSlice(self.predictions[key[1]])


class FakeModel:
    def __init__(self, fail_crop=False):
        self.calls = []
        self.fail_crop = fail_crop

    def __call__(self, **kwargs):
        suffix = tuple(kwargs["input_ids"].items)
        old_cache = kwargs.get("past_key_values")
        past = old_cache.get_seq_length() if old_cache is not None else 0
        # Fake target predicts deterministic alternation after each token.
        preds = tuple(transition(x) for x in suffix)
        self.calls.append((suffix, past))
        return SimpleNamespace(logits=FakeLogits(preds),
                               past_key_values=FakeCache(past + len(suffix), self.fail_crop))


class VerifiedSpeculationTests(unittest.TestCase):
    def test_verification_prefix_and_rejection(self):
        self.assertEqual(verify_greedy((4, 5), (4, 5, 6)).emitted, (4, 5, 6))
        self.assertEqual(verify_greedy((4, 5), (9, 5, 6)).emitted, (9,))
        self.assertEqual(verify_greedy((4, 5), (4, 8, 6)).emitted, (4, 8))
        with self.assertRaises(ValueError):
            verify_greedy((1,), (1,))
        with self.assertRaises(ValueError):
            verify_greedy((1,), (-1, 2))

    def test_speculative_matches_baseline_and_accepts(self):
        prompt = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
        s = AdaptiveSpeculationScheduler(draft_lengths=(2,), min_samples=2, explore_every=2)
        decoder = GreedySpeculativeDecoder(
            target, scheduler=s, index=NgramDraftIndex(order=2, max_draft=4))
        report = decoder.generate(prompt, max_new_tokens=20)
        self.assertEqual(report.new_tokens, baseline(prompt, 20))
        self.assertGreater(report.proposed_draft_tokens, 0)
        self.assertGreater(report.accepted_draft_tokens, 0)
        self.assertLessEqual(report.target_calls, 20)

    def test_untrusted_proposal_corrected_by_target(self):
        class BadIndex:
            order = 1
            max_draft = 1
            def remember(self, history):
                pass
            def propose(self, history, count):
                return (99,) if count else ()
        s = AdaptiveSpeculationScheduler(draft_lengths=(1,), min_samples=1, explore_every=1)
        decoder = GreedySpeculativeDecoder(target, scheduler=s, index=BadIndex())
        result = decoder.generate((1,), max_new_tokens=12)
        self.assertEqual(result.new_tokens, baseline((1,), 12))
        self.assertEqual(result.accepted_draft_tokens, 0)

    def test_eos_and_token_budget(self):
        s = AdaptiveSpeculationScheduler(draft_lengths=(2,), min_samples=1, explore_every=1)
        d = GreedySpeculativeDecoder(target, scheduler=s,
                                     index=NgramDraftIndex(order=2, max_draft=4))
        r = d.generate((1, 2, 1, 2), max_new_tokens=10, eos_token_ids=(2,))
        self.assertEqual(r.new_tokens, (1, 2))

    def test_speculative_cycle_includes_draft_lookup_cost(self):
        class SlowIndex:
            order = 1
            max_draft = 1
            def remember(self, history):
                pass
            def propose(self, history, count):
                time.sleep(0.03)
                return (transition(history[-1]),) if count else ()

        scheduler = AdaptiveSpeculationScheduler(
            draft_lengths=(1,), min_samples=1, explore_every=1)
        decoder = GreedySpeculativeDecoder(target, scheduler=scheduler, index=SlowIndex())
        result = decoder.generate((1,), max_new_tokens=3)
        self.assertEqual(result.new_tokens, baseline((1,), 3))
        self.assertEqual(result.proposed_draft_tokens, 1)
        self.assertGreater(result.speculative_policy["spec_seconds_per_token"][1], 0.01)

    def test_cache_reuse_across_verified_cycles(self):
        model = FakeModel()
        verifier = PagedTorchGreedyTarget(model, FakeTorch(), "cpu")
        prompt = (1, 2, 1, 2)
        self.assertEqual(verifier(prompt, (1, 2)), (1, 2, 1))
        self.assertEqual(verifier.processed_prefix, prompt + (1, 2))
        # Emitted bonus is 1, still unprocessed.
        self.assertEqual(verifier(prompt + (1, 2, 1), (2,)), (2, 1))
        self.assertEqual(model.calls[1], ((1, 2), 6))
        self.assertEqual(verifier.cache_reuse_count, 1)

    def test_cache_reject_crops_and_reuses(self):
        model = FakeModel()
        verifier = PagedTorchGreedyTarget(model, FakeTorch(), "cpu")
        prefix = (1, 2, 1)
        self.assertEqual(verifier(prefix, (2, 99)), (2, 1, 1))
        self.assertEqual(verifier.processed_prefix, prefix + (2,))
        self.assertEqual(verifier.cache.count, 4)
        self.assertEqual(verifier(prefix + (2, 1), ()), (2,))
        self.assertEqual(model.calls[-1], ((1,), 4))

    def test_cache_crop_failure_forces_full_replay(self):
        model = FakeModel(fail_crop=True)
        verifier = PagedTorchGreedyTarget(model, FakeTorch(), "cpu")
        verifier((1, 2, 1), (2, 99))
        self.assertIsNone(verifier.cache)
        verifier((1, 2, 1, 2, 1), ())
        self.assertEqual(model.calls[-1][0], (1, 2, 1, 2, 1))
        self.assertEqual(model.calls[-1][1], 0)


if __name__ == "__main__":
    unittest.main()
