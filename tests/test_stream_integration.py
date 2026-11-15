"""Optional real-Transformers forward parity; builds its own tiny checkpoint."""
import tempfile
import unittest

try:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, Qwen2Config
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

from ironasu.paging import LayerPager, SafetensorsIndex
from ironasu.q4 import Q4Index, quantize_checkpoint


@unittest.skipUnless(HAVE_DEPS, "optional transformers/accelerate dependencies are not installed")
class RealTransformersTests(unittest.TestCase):
    def test_gpt2_forward_matches_no_paging(self):
        config = GPT2Config(vocab_size=43, n_positions=32, n_ctx=32,
                            n_embd=16, n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2)
        torch.manual_seed(3)
        direct = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
        ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            direct.save_pretrained(tmp, safe_serialization=True)
            with init_empty_weights(include_buffers=False):
                empty = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(tmp, local_files_only=True),
                    attn_implementation="eager",
                ).eval()
            with LayerPager(empty, SafetensorsIndex(tmp)) as pager:
                with torch.inference_mode():
                    first = direct(ids, use_cache=True)
                    with pager.session() as model:
                        second = model(ids, use_cache=True)
                torch.testing.assert_close(first.logits, second.logits, atol=1e-5, rtol=1e-5)
                # Greedy autoregressive generation must also match, including
                # the KV-cache path over more than one decoding step.
                with torch.inference_mode():
                    normal_tokens = direct.generate(ids, max_new_tokens=2, do_sample=False,
                                                    eos_token_id=None, pad_token_id=2)
                    with pager.session() as model:
                        streamed_tokens = model.generate(ids, max_new_tokens=2,
                                                         do_sample=False, eos_token_id=None,
                                                         pad_token_id=2)
                torch.testing.assert_close(normal_tokens, streamed_tokens, rtol=0, atol=0)
                self.assertTrue(all(p.device.type == "meta" for p in empty.parameters()))

    def test_tiny_qwen2_parity_with_cpu_cache(self):
        """Exercise a locally generated Qwen2 architecture, no HF download."""
        config = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, tie_word_embeddings=False,
            bos_token_id=1, eos_token_id=2,
        )
        torch.manual_seed(17)
        direct = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
        ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            direct.save_pretrained(tmp, safe_serialization=True)
            with init_empty_weights(include_buffers=False):
                empty = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(tmp, local_files_only=True),
                    attn_implementation="eager",
                ).eval()
            with LayerPager(empty, SafetensorsIndex(tmp), cpu_cache_mb=1) as pager:
                with torch.inference_mode():
                    first = direct(ids, use_cache=True)
                    with pager.session() as model:
                        second = model(ids, use_cache=True)
                torch.testing.assert_close(first.logits, second.logits, atol=1e-5, rtol=1e-5)
                with torch.inference_mode():
                    normal_tokens = direct.generate(ids, max_new_tokens=3, do_sample=False,
                                                    eos_token_id=None, pad_token_id=2)
                    with pager.session() as model:
                        streamed_tokens = model.generate(ids, max_new_tokens=3,
                                                         do_sample=False, eos_token_id=None,
                                                         pad_token_id=2)
                torch.testing.assert_close(normal_tokens, streamed_tokens, rtol=0, atol=0)
                self.assertGreater(pager.snapshot()["cpu_cache"]["hits"], 0)
                self.assertTrue(all(p.device.type == "meta" for p in empty.parameters()))


    def test_qwen2_q4_quantized_layer_paging(self):
        """Real tiny Qwen2 architecture with per-group lossy Q4 on disk."""
        config = Qwen2Config(
            vocab_size=80, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, tie_word_embeddings=False,
            bos_token_id=1, eos_token_id=2,
        )
        torch.manual_seed(123)
        direct = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
        ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            raw = Path(tmp) / "original"
            q_dir = Path(tmp) / "q4"
            direct.save_pretrained(raw, safe_serialization=True)
            report = quantize_checkpoint(raw, q_dir, min_elements=128)
            self.assertLess(report["stored_payload_bytes"], report["original_payload_bytes"])
            with init_empty_weights(include_buffers=False):
                empty = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(raw, local_files_only=True),
                    attn_implementation="eager",
                ).eval()
            with LayerPager(empty, Q4Index(q_dir), cpu_cache_mb=1, prefetch=True) as pager:
                with torch.inference_mode():
                    with pager.session() as model:
                        result = model(ids, use_cache=True).logits
                    expected = direct(ids, use_cache=True).logits
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertEqual(result.shape, expected.shape)
                # Lossy Q4 is NOT expected to match original logits bitwise.
                self.assertLess((result - expected).abs().mean().item(), 1.0)
                with torch.inference_mode(), pager.session() as model:
                    again = model(ids, use_cache=True).logits
                torch.testing.assert_close(again, result, rtol=0, atol=0)
                self.assertEqual(pager.snapshot()["resident_stage_count"], 0)


    def test_exact_kv_prefill_reuse_matches_standard_generation(self):
        """Cold/warm prefills must match normal greedy tokens on GPT2/Qwen2."""
        from ironasu.kv_cache import PrefixKVCache, greedy_decode_with_exact_prefix
        for family in ("gpt2", "qwen2"):
            if family == "gpt2":
                config = GPT2Config(vocab_size=43, n_positions=32, n_ctx=32,
                                    n_embd=16, n_layer=2, n_head=2,
                                    bos_token_id=1, eos_token_id=2)
            else:
                config = Qwen2Config(
                    vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=64,
                    tie_word_embeddings=False, bos_token_id=1, eos_token_id=2)
            torch.manual_seed(30)
            direct = AutoModelForCausalLM.from_config(
                config, attn_implementation="eager").eval()
            ids = torch.tensor([[3, 4, 5, 6]])
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            with tempfile.TemporaryDirectory() as tmp:
                direct.save_pretrained(tmp, safe_serialization=True)
                with init_empty_weights(include_buffers=False):
                    empty = AutoModelForCausalLM.from_config(
                        AutoConfig.from_pretrained(tmp, local_files_only=True),
                        attn_implementation="eager").eval()
                cache = PrefixKVCache(max_bytes=4 * 1024**2)
                with LayerPager(empty, SafetensorsIndex(tmp)) as pager, torch.inference_mode():
                    expected = direct.generate(
                        **inputs, max_new_tokens=3, do_sample=False,
                        eos_token_id=None, pad_token_id=2
                    )[:, ids.shape[1]:]
                    with pager.session() as model:
                        cold = greedy_decode_with_exact_prefix(
                            model, inputs, max_new_tokens=3, kv_cache=cache,
                            eos_token_id=None)
                    with pager.session() as model:
                        warm = greedy_decode_with_exact_prefix(
                            model, inputs, max_new_tokens=3, kv_cache=cache,
                            eos_token_id=None)
                torch.testing.assert_close(cold, expected, rtol=0, atol=0)
                torch.testing.assert_close(warm, expected, rtol=0, atol=0)
                self.assertEqual(cache.snapshot()["hits"], 1)
                self.assertEqual(cache.snapshot()["saved_prefill_tokens"], 4)



if __name__ == "__main__":
    unittest.main()
