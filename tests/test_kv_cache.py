import unittest
try:
    import torch
except ImportError:
    torch = None
from ironasu.kv_cache import PrefixKVCache, greedy_decode_with_exact_prefix


@unittest.skipIf(torch is None, "torch unavailable")
class PrefixCacheTests(unittest.TestCase):
    def data(self):
        return ((torch.arange(3).float(), torch.arange(3).float()),), torch.arange(5).float()

    def test_exact_match_clone_and_reset(self):
        c = PrefixKVCache(max_bytes=1024)
        past, logits = self.data()
        self.assertTrue(c.put((1, 2), (1, 1), past, logits))
        p, l = c.get((1, 2), (1, 1))
        p[0][0][0] = -99
        l[0] = -99
        newer = c.get((1, 2), (1, 1))
        self.assertEqual(newer[0][0][0][0].item(), 0)
        self.assertEqual(newer[1][0].item(), 0)
        self.assertIsNone(c.get((1, 3), (1, 1)))
        self.assertIsNone(c.get((1, 2), (1, 0)))
        self.assertEqual(c.snapshot()["saved_prefill_tokens"], 4)
        c.reset()
        self.assertIsNone(c.get((1, 2), (1, 1)))
        self.assertEqual(c.snapshot()["occupied_bytes"], 0)
        self.assertEqual(c.snapshot()["generation"], 1)

    def test_lru_and_budget(self):
        c = PrefixKVCache(max_bytes=1024, max_entries=2)
        p, l = self.data()
        c.put((1,), (1,), p, l)
        c.put((2,), (1,), p, l)
        c.get((1,), (1,))
        c.put((3,), (1,), p, l)
        self.assertIsNone(c.get((2,), (1,)))
        self.assertEqual(c.snapshot()["entries"], 2)
        self.assertFalse(PrefixKVCache(max_bytes=1).put((1,), (1,), p, l))
        self.assertFalse(c.put((4,), (1,), object(), l))
        self.assertLessEqual(c.snapshot()["occupied_bytes"], 1024)

    def test_settings(self):
        for opts in ({"max_bytes": -1}, {"max_bytes": True},
                     {"max_bytes": 10, "max_entries": 0}):
            with self.assertRaises(ValueError):
                PrefixKVCache(**opts)

    def test_greedy_hit_skips_prompt_forward(self):
        from types import SimpleNamespace
        class Fake:
            def __init__(self):
                self.prefills = 0
            def __call__(self, input_ids, attention_mask, use_cache, return_dict,
                         past_key_values=None):
                if past_key_values is None:
                    self.prefills += 1
                    past = ((torch.zeros(1), torch.zeros(1)),)
                else:
                    past = past_key_values
                    past[0][0].add_(1)  # mutate input cache in place
                logits = torch.full((1, input_ids.size(1), 10), -20.)
                next_token = (int(input_ids[0, -1]) + 1) % 10
                logits[0, -1, next_token] = 20.
                return SimpleNamespace(past_key_values=past, logits=logits)
        model = Fake()
        c = PrefixKVCache(max_bytes=1024)
        inputs = {"input_ids": torch.tensor([[2, 3, 4]]),
                  "attention_mask": torch.tensor([[1, 1, 1]])}
        results = [greedy_decode_with_exact_prefix(model, inputs, max_new_tokens=4,
                                                   kv_cache=c) for _ in range(3)]
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)
        self.assertEqual(model.prefills, 1)
        self.assertEqual(c.snapshot()["hits"], 2)
        self.assertEqual(c.snapshot()["saved_prefill_tokens"], 6)

    def test_eos_stop(self):
        from types import SimpleNamespace
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[[0., 1., 8.]]]),
                                       past_key_values=((torch.zeros(1), torch.zeros(1)),))
        actual = greedy_decode_with_exact_prefix(
            Model(), {"input_ids": torch.tensor([[1]])},
            max_new_tokens=10, kv_cache=PrefixKVCache(max_bytes=1024), eos_token_id=2)
        self.assertEqual(actual.tolist(), [[2]])
