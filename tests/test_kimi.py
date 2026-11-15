"""Kimi text-path checks: no network, no official giant weights or custom imports."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ironasu.kimi import (
    KimiCompatibilityError, TextWeightView, inspect_kimi_config,
    require_local_kimi_code,
)
from ironasu.paging import LayerPager, PagingError, SafetensorsIndex
from ironasu.stream import PagedCausalLM

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from safetensors.torch import save_file
except ImportError:
    torch = None


def mini_config(*, kda=None, mla=None) -> dict:
    return {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_kimi_k3.KimiK3Config",
            "AutoModel": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        },
        "text_config": {
            "model_type": "kimi_linear", "num_hidden_layers": 3,
            "num_experts": 4, "num_experts_per_token": 2,
            "hidden_size": 8, "attn_res_block_size": 2,
            "linear_attn_config": {"kda_layers": [1, 2] if kda is None else kda,
                                   "full_attn_layers": [3] if mla is None else mla},
            "auto_map": {"AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM"},
        },
    }


class KimiConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(json.dumps(mini_config()), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_inspection_never_imports_or_reads_weights(self):
        report = inspect_kimi_config(self.root)
        self.assertEqual((report["kda_layers"], report["mla_layers"]), (2, 1))
        self.assertFalse(report["code_executed"])
        self.assertFalse(report["weights_loaded"])
        self.assertEqual(report["expert_fraction_per_token"], .5)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": "missing-shard.safetensors",
                           "vision_tower.conv.weight": "also-missing.safetensors"}
        }))
        report = inspect_kimi_config(self.root)
        self.assertEqual(report["checkpoint"]["expert_tensor_keys"], 1)
        self.assertEqual(report["checkpoint"]["non_text_tensor_keys_ignored"], 1)
        self.assertEqual(report["checkpoint"]["text_tensor_keys"], 1)

    def test_rejects_ambiguous_layout_or_wrong_family(self):
        for c in (mini_config(kda=[1], mla=[3]), mini_config(kda=[1, 2], mla=[2, 3])):
            (self.root / "config.json").write_text(json.dumps(c), encoding="utf-8")
            with self.assertRaisesRegex(KimiCompatibilityError, "layout"):
                inspect_kimi_config(self.root)
        (self.root / "config.json").write_text(json.dumps({"model_type": "qwen"}), encoding="utf-8")
        with self.assertRaisesRegex(KimiCompatibilityError, "not an official"):
            inspect_kimi_config(self.root)

    def test_opt_in_local_python_not_silently_trusted(self):
        with self.assertRaisesRegex(KimiCompatibilityError, "Missing or unsafe"):
            require_local_kimi_code(self.root)
        for name in ("configuration_kimi_k3.py", "modeling_kimi_linear.py", "modeling_kimi_k3.py"):
            (self.root / name).write_text("# a placeholder for tests\n", encoding="utf-8")
        require_local_kimi_code(self.root)
        config = mini_config()
        config["text_config"]["auto_map"]["AutoModelForCausalLM"] = "remote/repo--unsafe.KimiLinearForCausalLM"
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(KimiCompatibilityError, "code mapping"):
            require_local_kimi_code(self.root)

    def test_kimi_mode_requires_explicit_consent_before_import(self):
        with self.assertRaisesRegex(PagingError, "explicit trust_local_code"):
            PagedCausalLM.from_local_dir(self.root, kimi_text_only=True)
        with self.assertRaisesRegex(PagingError, "only supported"):
            PagedCausalLM.from_local_dir(self.root, trust_local_code=True)


if torch is not None:
    class MiniKimiCore(nn.Module):
        """Fake Kimi-like structure where residual params are accessed directly."""
        def __init__(self):
            super().__init__()
            self.use_attn_residuals = True
            self.embed_tokens = nn.Embedding(20, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
            self.output_attn_res_proj = nn.Linear(8, 1, bias=False)
            self.output_attn_res_norm = nn.LayerNorm(8)
            self.norm = nn.LayerNorm(8)

        def forward(self, ids):
            x = self.embed_tokens(ids)
            for layer in self.layers:
                x = torch.tanh(layer(x))
            # Deliberately don't call norm/proj Module.forward; the Kimi
            # reference similarly uses their weights functionally.
            y = F.layer_norm(x, (8,), self.output_attn_res_norm.weight,
                             self.output_attn_res_norm.bias)
            gate = F.linear(y, self.output_attn_res_proj.weight).sigmoid()
            return self.norm(x + gate * y)


    class MiniKimiText(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MiniKimiCore()
            self.lm_head = nn.Linear(8, 20, bias=False)

        def forward(self, ids):
            return self.lm_head(self.model(ids))


    class MiniMultimodal(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = MiniKimiText()
            self.vision_tower = nn.Linear(8, 8)


    class CountingReader:
        def __init__(self, inner):
            self.inner = inner
            self.reads: list[tuple[str, ...]] = []

        def keys(self):
            return self.inner.keys()

        def read(self, names):
            self.reads.append(tuple(names))
            return self.inner.read(names)

        def tensor_byte_sizes(self, names):
            return self.inner.tensor_byte_sizes(names)


@unittest.skipIf(torch is None, "torch and safetensors required")
class KimiTextPagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.manual_seed(104)
        self.original = MiniMultimodal().eval()
        save_file({name: tensor.detach().contiguous() for name, tensor in self.original.state_dict().items()},
                  self.root / "model.safetensors")
        self.ids = torch.tensor([[1, 3, 5]])
        self.ref = self.original.language_model(self.ids)

    def tearDown(self):
        self.temp.cleanup()

    def test_text_only_remaps_names_ignoring_vision_weights(self):
        inner = CountingReader(SafetensorsIndex(self.root))
        reader = TextWeightView(inner)
        self.assertNotIn("vision_tower.weight", reader.keys())
        self.assertIn("model.embed_tokens.weight", reader.keys())
        self.assertEqual(reader.tensor_byte_sizes(["lm_head.weight"])["lm_head.weight"], 20*8*4)
        with torch.device("meta"):
            model = MiniKimiText().eval()
        with LayerPager(model, reader, prefetch=True, cpu_cache_mb=1) as pager:
            for _ in range(2):
                with pager.session() as stream:
                    torch.testing.assert_close(stream(self.ids), self.ref, atol=0, rtol=0)
                self.assertTrue(all(p.device.type == "meta" for p in stream.parameters()))
                self.assertEqual(pager.snapshot()["resident_stage_count"], 0)
            self.assertIn("model", pager.snapshot()["per_stage"])
            self.assertEqual(pager.snapshot()["per_stage"]["model"]["loads"], 2)
        self.assertTrue(inner.reads)
        self.assertTrue(all(all(key.startswith("language_model.") for key in call) for call in inner.reads))

    def test_fail_closed_if_attn_residual_tensor_missing(self):
        class MissingReader(CountingReader):
            def keys(self):
                return {k for k in self.inner.keys() if "output_attn_res_proj.weight" not in k}
        index = TextWeightView(MissingReader(SafetensorsIndex(self.root)))
        with torch.device("meta"):
            model = MiniKimiText().eval()
        with self.assertRaisesRegex(PagingError, "No checkpoint tensor"):
            LayerPager(model, index)

    def test_read_rejects_unscoped_weights(self):
        reader = TextWeightView(SafetensorsIndex(self.root))
        with self.assertRaisesRegex(KimiCompatibilityError, "Unknown text weight"):
            reader.read(["vision_tower.weight"])
        with self.assertRaisesRegex(KimiCompatibilityError, "Unknown text weight"):
            reader.read(["../../outside"])


if __name__ == "__main__":
    unittest.main()
