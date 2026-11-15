"""MoE integration tests with real torch and local safetensors, no model downloads."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from torch import nn
    from safetensors.torch import save_file
except ImportError:
    torch = None

from ironasu.moe import ExpertLRU, inspect_moe_weight_map
from ironasu.paging import LayerPager, PagingError, SafetensorsIndex


class InspectTests(unittest.TestCase):
    def test_kimi_style_packed_checkpoint_selection(self):
        prefix = "language_model.model.layers.1.block_sparse_moe.experts"
        index = {
            f"{prefix}.{i}.{part}": "a.safetensors" if i < 2 else "b.safetensors"
            for i in range(4) for part in ("w1.weight_packed", "w1.scale", "w2.weight_packed")
        }
        report = inspect_moe_weight_map(index, layer=prefix, selected=[1, 3])
        self.assertEqual(report["expert_sets"][prefix], 4)
        self.assertEqual(report["selection"]["selected_tensor_keys"], 6)
        self.assertEqual(report["selection"]["total_tensor_keys"], 12)
        self.assertEqual(report["selection"]["selected_indexed_shards"], 2)
        with self.assertRaisesRegex(ValueError, "Missing expert"):
            inspect_moe_weight_map(index, layer=prefix, selected=[100])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_moe_weight_map(index, layer=prefix, selected=[1, 1])
        self.assertEqual(inspect_moe_weight_map(index)["expert_containers"], 1)

    def test_expert_lru_is_bounded_and_recent_experts_survive(self):
        if torch is None:
            self.skipTest("torch unavailable")
        cache = ExpertLRU(16)
        cache.put("a", {"a": torch.ones(4)})
        self.assertEqual(cache.occupied_bytes, 16)
        cache.put("b", {"b": torch.ones(4)})
        self.assertEqual(cache.snapshot()["evictions"], 1)
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        cache.put("huge", {"huge": torch.ones(1024)})
        self.assertEqual(cache.snapshot()["oversize_bypasses"], 1)
        self.assertLessEqual(cache.occupied_bytes, cache.budget_bytes)
        cache.clear()
        self.assertEqual(cache.occupied_bytes, 0)


if torch is not None:
    class ToyMoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(8, 4)
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([nn.Sequential(nn.Linear(8, 16), nn.Tanh(),
                                                            nn.Linear(16, 8)) for _ in range(4)])

        def forward(self, x):
            picks = self.router(x).argmax(dim=-1).reshape(-1)
            x_flat = x.reshape(-1, x.shape[-1])
            y = torch.zeros_like(x_flat)
            for expert_id in picks.unique().tolist():
                mask = picks == expert_id
                # Crucial: call Module via __call__ so pager hooks run.
                y[mask] = self.mlp.experts[expert_id](x_flat[mask])
            return y.reshape(x.shape)


    class ToyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(10, 8)
            self.model.layers = nn.ModuleList([ToyMoEBlock()])
            self.model.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 10)

        def forward(self, ids):
            x = self.model.embed_tokens(ids)
            for block in self.model.layers:
                x = block(x)
            return self.lm_head(self.model.norm(x))


    class RecordingIndex:
        def __init__(self, index):
            self.index = index
            self.calls: list[tuple[str, ...]] = []
        def keys(self):
            return self.index.keys()
        def tensor_byte_sizes(self, names):
            return self.index.tensor_byte_sizes(names)
        def read(self, names):
            self.calls.append(tuple(names))
            return self.index.read(names)


@unittest.skipIf(torch is None, "torch or safetensors unavailable")
class MoEPagingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        torch.manual_seed(123)
        self.source = ToyMoE().eval()
        with torch.no_grad():
            self.source.model.layers[0].router.weight.zero_()
            self.source.model.layers[0].router.bias[:] = torch.tensor([8., -8., -8., -8.])
        self.ids = torch.tensor([[2, 5, 6], [3, 4, 9]])
        save_file({k: v.detach().contiguous() for k, v in self.source.state_dict().items()},
                  Path(self.directory.name) / "model.safetensors")
        self.reference = self.source(self.ids)
        self.index = RecordingIndex(SafetensorsIndex(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def _meta(self):
        with torch.device("meta"):
            return ToyMoE().eval()

    def test_loads_only_active_expert_with_identical_outputs(self):
        model = self._meta()
        with LayerPager(model, self.index, expert_paging=True) as pager:
            for _ in range(2):
                with pager.session() as stream:
                    actual = stream(self.ids)
                torch.testing.assert_close(actual, self.reference, atol=0, rtol=0)
                self.assertTrue(all(p.device.type == "meta" for p in stream.parameters()))
                self.assertEqual(pager.snapshot()["expert_paging"]["resident_experts"], 0)
            report = pager.snapshot()["expert_paging"]
            self.assertEqual(report["indexed_experts"], 4)
            self.assertEqual(report["per_expert"]["model.layers.0.mlp.experts.0"]["loads"], 2)
            self.assertEqual(report["cache"]["hits"], 0)
            self.assertEqual(report["cache"]["misses"], 2)
        expert_keys = [k for call in self.index.calls for k in call if ".experts." in k]
        self.assertEqual({k.split(".experts.")[1].split(".")[0] for k in expert_keys}, {"0"})
        self.assertEqual(len(expert_keys), 8)  # 2 linears × 2 params × 2 passes

    def test_expert_lru_reuses_reads_without_changing_outputs(self):
        model = self._meta()
        with LayerPager(model, self.index, expert_paging=True, expert_cache_mb=1,
                        cpu_cache_mb=1, prefetch=True) as pager:
            for _ in range(3):
                with pager.session() as m:
                    torch.testing.assert_close(m(self.ids), self.reference, atol=0, rtol=0)
            report = pager.snapshot()["expert_paging"]
            self.assertEqual(report["cache"]["hits"], 2)
            self.assertEqual(report["cache"]["misses"], 1)
            self.assertEqual(len(report["cache"]["cached_experts"]), 1)
            self.assertLessEqual(report["cache"]["occupied_bytes"], 1024 ** 2)
        expert_keys = [k for call in self.index.calls for k in call if ".experts." in k]
        self.assertEqual(len(expert_keys), 4)
        self.assertEqual(pager.snapshot()["expert_paging"]["cache"]["occupied_bytes"], 0)

    def test_all_experts_loaded_without_expert_paging(self):
        with LayerPager(self._meta(), self.index) as pager:
            with pager.session() as model:
                torch.testing.assert_close(model(self.ids), self.reference, atol=0, rtol=0)
        expert_keys = [k for call in self.index.calls for k in call if ".experts." in k]
        self.assertEqual(len(expert_keys), 16)

    def test_missing_expert_tensor_rejected_during_preflight(self):
        base = self.index
        class Missing(RecordingIndex):
            def keys(self):
                return {k for k in base.keys() if "experts.1.0.weight" not in k}
        with self.assertRaisesRegex(PagingError, "Missing expert tensor"):
            LayerPager(self._meta(), Missing(base.index), expert_paging=True)

    def test_expert_compute_error_always_evicts(self):
        class Boom(nn.Linear):
            def forward(self, x):
                raise RuntimeError("expert failed")
        meta = self._meta()
        meta.model.layers[0].mlp.experts[0] = nn.Sequential(
            Boom(8, 16, device="meta"), nn.Tanh(), nn.Linear(16, 8, device="meta")
        )
        with LayerPager(meta, self.index, expert_paging=True) as pager:
            with self.assertRaisesRegex(RuntimeError, "expert failed"):
                with pager.session() as model:
                    model(self.ids)
            self.assertEqual(pager.snapshot()["expert_paging"]["resident_experts"], 0)
            self.assertTrue(all(p.device.type == "meta" for p in meta.parameters()))

    def test_wrong_container_and_cache_arguments_rejected(self):
        with self.assertRaises(PagingError):
            LayerPager(self._meta(), self.index, expert_cache_mb=1)
        with self.assertRaises(PagingError):
            LayerPager(self._meta(), self.index, expert_paging=True, expert_cache_mb=-1)
        with self.assertRaises(PagingError):
            LayerPager(self._meta(), self.index, expert_paging=True, expert_cache_mb=True)


if __name__ == "__main__":
    unittest.main()
