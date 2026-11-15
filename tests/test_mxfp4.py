"""Exact OCP FP4 decoder checks, plus a real MoE forward using native packed weights."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from torch import nn
    from safetensors.torch import save_file
except ImportError:
    torch = None

from ironasu.mxfp4 import MXFP4Index, MXFP4Tensor
from ironasu.paging import LayerPager, SafetensorsIndex


@unittest.skipIf(torch is None, "torch/safetensors optional dependency unavailable")
class MXFP4Tests(unittest.TestCase):
    def test_exact_e2m1_nibbles_and_e8m0_exponents(self):
        # Repeat every nibble once in a group; two rows with scales 1 and 2.
        nibs = torch.tensor(list(range(16))*2, dtype=torch.uint8)
        packed = nibs[0::2] | (nibs[1::2] << 4)
        p = packed.repeat(2, 1)
        scale = torch.tensor([[127], [128]], dtype=torch.uint8)
        decoded = MXFP4Tensor(p, scale, (2, 32), torch.float32).to(device="cpu")
        reference = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                                  -0.0, -.5, -1, -1.5, -2, -3, -4, -6]*2)
        torch.testing.assert_close(decoded[0], reference, atol=0, rtol=0)
        torch.testing.assert_close(decoded[1], reference * 2, atol=0, rtol=0)
        with_nan = MXFP4Tensor(p[:1], torch.tensor([[255]], dtype=torch.uint8),
                               (1, 32), torch.float32).to(device="cpu")
        self.assertTrue(torch.isnan(with_nan).all())

    def test_bad_mxfp4_layout_is_rejected(self):
        p = torch.zeros(2, 16, dtype=torch.uint8)
        s = torch.zeros(2, 1, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            MXFP4Tensor(p, s, (2, 31), torch.float32).to(device="cpu")
        with self.assertRaises(ValueError):
            MXFP4Tensor(p.float(), s, (2, 32), torch.float32).to(device="cpu")

    def test_native_pair_maps_to_logical_weight(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save_file({"model.layers.0.mlp.experts.0.weight_packed": torch.zeros(32,16,dtype=torch.uint8),
                       "model.layers.0.mlp.experts.0.weight_scale": torch.full((32,1),127,dtype=torch.uint8),
                       "model.norm.weight": torch.ones(32)}, p/"model.safetensors")
            idx = MXFP4Index(SafetensorsIndex(p), dtype=torch.float32)
            self.assertIn("model.layers.0.mlp.experts.0.weight", idx.keys())
            self.assertNotIn("model.layers.0.mlp.experts.0.weight_packed", idx.keys())
            self.assertEqual(idx.tensor_byte_sizes(["model.layers.0.mlp.experts.0.weight"])["model.layers.0.mlp.experts.0.weight"], 32*17)
            data = idx.read(["model.layers.0.mlp.experts.0.weight"])
            self.assertEqual(data["model.layers.0.mlp.experts.0.weight"].shape, (32,32))
            self.assertEqual(data["model.layers.0.mlp.experts.0.weight"].storage_bytes, 544)
            self.assertEqual(data["model.layers.0.mlp.experts.0.weight"].to(device="cpu").shape, (32,32))

    def test_missing_scale_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save_file({"layer.weight_packed": torch.zeros(2,16,dtype=torch.uint8)}, p/"model.safetensors")
            with self.assertRaisesRegex(ValueError, "scale tensor missing"):
                MXFP4Index(SafetensorsIndex(p))

    def test_orphan_scale_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save_file({"layer.weight_scale": torch.zeros(2,1,dtype=torch.uint8)}, p/"model.safetensors")
            with self.assertRaisesRegex(ValueError, "packed tensor missing"):
                MXFP4Index(SafetensorsIndex(p))

    def test_paged_moe_with_original_native_mxfp4_expert(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(32, 2)
                self.mlp = nn.Module()
                self.mlp.experts = nn.ModuleList([nn.Linear(32,32,bias=False) for _ in range(2)])
            def forward(self,x):
                expert = self.router(x).argmax(dim=-1)
                # Synthetic top-1 routing; chosen expert calls trigger the pager.
                out = torch.zeros_like(x)
                for i in expert.unique().tolist():
                    mask = expert == i
                    out[mask] = self.mlp.experts[i](x[mask])
                return out
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(16,32)
                self.model.layers = nn.ModuleList([Block()])
                self.model.norm = nn.LayerNorm(32)
                self.lm_head = nn.Linear(32,16)
            def forward(self, ids):
                x = self.model.embed_tokens(ids)
                x = self.model.layers[0](x)
                return self.lm_head(self.model.norm(x))
        torch.manual_seed(8)
        src = Model().eval()
        with torch.no_grad():
            src.model.layers[0].router.weight.zero_()
            src.model.layers[0].router.bias.copy_(torch.tensor([5., -5.]))
            # Exact FP4 weights: +/-1 are E2M1 2 or 10 at scale 127.
            for expert in src.model.layers[0].mlp.experts:
                expert.weight.copy_(torch.where(expert.weight >= 0, 1.0, -1.0))
        weights = {k:v.detach().contiguous() for k,v in src.state_dict().items()
                   if ".experts." not in k}
        for key, value in src.state_dict().items():
            if ".experts." not in key: continue
            codes = torch.where(value > 0, 2, 10).to(torch.uint8)
            packed = codes[:,::2] | (codes[:,1::2] << 4)
            weights[key[:-len("weight")] + "weight_packed"] = packed.contiguous()
            weights[key[:-len("weight")] + "weight_scale"] = torch.full((32,1), 127, dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as d:
            save_file(weights, Path(d)/"model.safetensors")
            with torch.device("meta"):
                model = Model().eval()
            idx = MXFP4Index(SafetensorsIndex(d), dtype=torch.float32)
            ids = torch.tensor([[1,2,3]])
            with LayerPager(model, idx, expert_paging=True, expert_cache_mb=1) as pager:
                with pager.session() as running:
                    actual = running(ids)
                torch.testing.assert_close(actual, src(ids), atol=0, rtol=0)
                with pager.session() as running:
                    again = running(ids)
                torch.testing.assert_close(actual, again, atol=0, rtol=0)
                cache = pager.snapshot()["expert_paging"]["cache"]
                self.assertEqual(cache["hits"], 1)
                self.assertEqual(cache["misses"], 1)


if __name__ == "__main__":
    unittest.main()
