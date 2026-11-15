"""Actual PyTorch forward tests with disk weights; no HF download needed."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

try:
    import torch
    from torch import nn
    from safetensors.torch import save_file
except ImportError:
    torch = None

from ironasu.paging import LayerPager, PagingError, SafetensorsIndex, decoder_stage_names


if torch is not None:
    class ToyDecoder(nn.Module):
        def __init__(self, tied=False):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(17, 8)
            self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
            self.model.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 17, bias=False)
            if tied:
                self.lm_head.weight = self.model.embed_tokens.weight

        def forward(self, ids):
            out = self.model.embed_tokens(ids)
            for layer in self.model.layers:
                out = torch.relu(layer(out))
            return self.lm_head(self.model.norm(out))


@unittest.skipIf(torch is None, "torch + safetensors optional dependency unavailable")
class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        torch.manual_seed(10)
        self.source = ToyDecoder().eval()
        save_file({k: v.detach().contiguous() for k, v in self.source.state_dict().items()}, self.path / "model.safetensors")
        with torch.device("meta"):
            self.meta = ToyDecoder()
        self.index = SafetensorsIndex(self.path)
        self.ids = torch.tensor([[3, 1, 5], [2, 4, 6]])

    def tearDown(self):
        self.tmp.cleanup()

    def test_forward_is_identical_and_layers_evicted(self):
        before = self.source(self.ids)
        with LayerPager(self.meta, self.index) as pager:
            for _ in range(2):
                with pager.session() as streaming:
                    actual = streaming(self.ids)
                torch.testing.assert_close(actual, before, rtol=0, atol=0)
                self.assertTrue(all(p.device.type == "meta" for p in streaming.parameters()))
            trace = pager.snapshot()
            self.assertEqual(trace["stage_count"], 5)
            self.assertEqual(trace["resident_stage_count"], 0)
            self.assertEqual(sum(v["loads"] for v in trace["per_stage"].values()), 10)
            self.assertEqual(max(v["peak_simultaneous_stages"] for v in trace["per_stage"].values()), 1)
            self.assertGreaterEqual(trace["per_stage"]["model.layers.0"]["disk_seconds"], 0)
        with self.assertRaises(PagingError):
            with pager.session():
                pass

    def test_no_model_forward_outside_session(self):
        with LayerPager(self.meta, self.index) as pager:
            with self.assertRaises(PagingError):
                pager.model(self.ids)

    def test_tied_embedding_head_uses_checkpoint_alias(self):
        src = ToyDecoder(tied=True).eval()
        torch.manual_seed(14)
        with torch.no_grad():
            src.model.embed_tokens.weight.uniform_(-0.1, 0.1)
        # safetensors cannot store shared tensors twice; omit output alias.
        weights = {k: v.detach().contiguous() for k, v in src.state_dict().items() if k != "lm_head.weight"}
        save_file(weights, self.path / "tied.safetensors")
        (self.path / "model.safetensors").unlink()
        (self.path / "tied.safetensors").rename(self.path / "model.safetensors")
        with torch.device("meta"):
            tied_meta = ToyDecoder(tied=True)
        with LayerPager(tied_meta, SafetensorsIndex(self.path)) as pager:
            with pager.session() as model:
                result = model(self.ids)
        torch.testing.assert_close(result, src(self.ids), rtol=0, atol=0)

    def test_missing_tensor_rejected_before_installing_hooks(self):
        self.meta.model.layers[0].weight = nn.Parameter(torch.empty((7, 8), device="meta"))
        with LayerPager(self.meta, self.index) as pager:
            with self.assertRaises(PagingError):
                with pager.session() as model:
                    model(self.ids)
            self.assertEqual(pager.snapshot()["resident_stage_count"], 0)
            self.assertTrue(all(p.device.type == "meta" for p in self.meta.parameters()))

    def test_unknown_architecture_rejected(self):
        with self.assertRaises(PagingError):
            decoder_stage_names(nn.Linear(2, 3))

    def test_prefetch_matches_normal_forward(self):
        with LayerPager(self.meta, self.index, prefetch=True) as pager:
            for _ in range(2):
                with pager.session() as model:
                    out = model(self.ids)
                torch.testing.assert_close(out, self.source(self.ids), rtol=0, atol=0)
            self.assertTrue(pager.snapshot()["prefetch"])
            self.assertEqual(pager.snapshot()["resident_stage_count"], 0)
            self.assertTrue(all(p.device.type == "meta" for p in self.meta.parameters()))

    def test_checkpoint_index_shards(self):
        weights = self.source.state_dict()
        a = {k: v.detach().contiguous() for k, v in weights.items() if "layers." in k}
        b = {k: v.detach().contiguous() for k, v in weights.items() if "layers." not in k}
        save_file(a, self.path / "a.safetensors")
        save_file(b, self.path / "b.safetensors")
        (self.path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {**{k:"a.safetensors" for k in a}, **{k:"b.safetensors" for k in b}}
        }))
        index = SafetensorsIndex(self.path)
        with LayerPager(self.meta, index) as pager:
            with pager.session() as model:
                torch.testing.assert_close(model(self.ids), self.source(self.ids), rtol=0, atol=0)

    def test_traversal_shards_rejected(self):
        (self.path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.layers.0.weight": "../outside.safetensors"}
        }))
        with self.assertRaises((PagingError, FileNotFoundError)):
            SafetensorsIndex(self.path)

    def test_failure_during_layer_forward_evicts_weights(self):
        class Boom(nn.Linear):
            def forward(self, x):
                raise RuntimeError("compute failed")
        original = self.meta.model.layers[1]
        replacement = Boom(8, 8, device="meta")
        self.meta.model.layers[1] = replacement
        with LayerPager(self.meta, self.index) as pager:
            with self.assertRaisesRegex(RuntimeError, "compute failed"):
                with pager.session() as model:
                    model(self.ids)
            self.assertEqual(pager.snapshot()["resident_stage_count"], 0)
            self.assertTrue(all(p.device.type == "meta" for p in self.meta.parameters()))
        self.meta.model.layers[1] = original

    def test_reject_non_meta_parameters(self):
        with self.assertRaises(PagingError):
            LayerPager(self.source, self.index)

    def test_cannot_open_nested_sessions(self):
        with LayerPager(self.meta, self.index) as pager:
            with pager.session():
                with self.assertRaises(PagingError):
                    with pager.session():
                        pass


if __name__ == "__main__":
    unittest.main()
