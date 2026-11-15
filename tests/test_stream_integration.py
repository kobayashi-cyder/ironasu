"""Optional real-Transformers forward parity; builds its own tiny checkpoint."""
import tempfile
import unittest

try:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

from ironasu.paging import LayerPager, SafetensorsIndex


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
                # Compare greedy generation including two KV-cache decoding steps.
                with torch.inference_mode():
                    normal_tokens = direct.generate(ids, max_new_tokens=2, do_sample=False,
                                                    eos_token_id=None, pad_token_id=2)
                    with pager.session() as model:
                        streamed_tokens = model.generate(ids, max_new_tokens=2,
                                                         do_sample=False, eos_token_id=None,
                                                         pad_token_id=2)
                torch.testing.assert_close(normal_tokens, streamed_tokens, rtol=0, atol=0)
                self.assertTrue(all(p.device.type == "meta" for p in empty.parameters()))


if __name__ == "__main__":
    unittest.main()
