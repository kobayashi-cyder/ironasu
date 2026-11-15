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
        def __init__(self, tied=False, vocab=17, width=8):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, width)
            self.model.layers = nn.ModuleList([nn.Linear(width, width), nn.Linear(width, width)])
            self.model.norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, vocab, bias=False)
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

    def test_stable_cpu_cache_reuses_exact_weights(self):
        before = self.source(self.ids)
        with LayerPager(self.meta, self.index, cpu_cache_mb=1) as pager:
            for _ in range(3):
                with pager.session() as model:
                    torch.testing.assert_close(model(self.ids), before, rtol=0, atol=0)
            trace = pager.snapshot()
            cache = trace["cpu_cache"]
            self.assertEqual(cache["policy"], "fixed_admission")
            self.assertEqual(cache["hits"], 10)
            self.assertEqual(cache["misses"], 5)
            self.assertEqual(len(cache["cached_stages"]), 5)
            self.assertGreater(cache["avoided_read_bytes"], 0)
            self.assertGreater(cache["disk_read_bytes"], 0)
            self.assertLessEqual(cache["occupied_bytes"], cache["budget_bytes"])
            self.assertEqual(cache["budget_bytes"], 1024 * 1024)
        self.assertEqual(pager.snapshot()["cpu_cache"]["occupied_bytes"], 0)

    def test_pin_stage_allowlist_does_not_cache_other_stages(self):
        with LayerPager(self.meta, self.index, cpu_cache_mb=1,
                        pin_stages=["lm_head"]) as pager:
            for _ in range(2):
                with pager.session() as model:
                    torch.testing.assert_close(model(self.ids), self.source(self.ids), rtol=0, atol=0)
            cache = pager.snapshot()["cpu_cache"]
            self.assertEqual(cache["planned_stages"], ["lm_head"])
            self.assertEqual(cache["cached_stages"], ["lm_head"])
            self.assertEqual(cache["hits"], 1)
            self.assertEqual(cache["misses"], 9)

    def test_cache_and_prefetch_together(self):
        with LayerPager(self.meta, self.index, cpu_cache_mb=1, prefetch=True) as pager:
            for _ in range(2):
                with pager.session() as model:
                    torch.testing.assert_close(model(self.ids), self.source(self.ids), rtol=0, atol=0)
            self.assertEqual(pager.snapshot()["cpu_cache"]["hits"], 5)

    def test_hash_table_reuses_tied_tensors_across_unpinned_stages(self):
        # Only lm_head is explicitly pinned. The input embedding uses the same
        # checkpoint tensor, and must profit from the hash table on later runs.
        source = ToyDecoder(tied=True).eval()
        save_file({k: v.detach().contiguous() for k, v in source.state_dict().items()
                   if k != "lm_head.weight"}, self.path / "shared.safetensors")
        (self.path / "model.safetensors").unlink()
        (self.path / "shared.safetensors").rename(self.path / "model.safetensors")
        with torch.device("meta"):
            meta = ToyDecoder(tied=True)
        index = SafetensorsIndex(self.path)
        read_calls = []
        original_read = index.read

        def counting_read(names):
            read_calls.append(list(names))
            return original_read(names)
        index.read = counting_read
        with LayerPager(meta, index, cpu_cache_mb=1,
                        pin_stages=["lm_head"]) as pager:
            for _ in range(3):
                with pager.session() as model:
                    torch.testing.assert_close(model(self.ids), source(self.ids), rtol=0, atol=0)
            snapshot = pager.snapshot()["cpu_cache"]
            self.assertEqual(snapshot["granularity"], "tensor_hash_table")
            self.assertEqual(snapshot["cached_tensors"], 1)
            self.assertEqual(snapshot["hits"], 5)  # head on first run, then both aliases
            self.assertEqual(snapshot["tensor_hits"], 5)
            self.assertIn("model.embed_tokens", snapshot["cached_stages"])
            self.assertIn("lm_head", snapshot["cached_stages"])
            self.assertLessEqual(snapshot["occupied_bytes"], snapshot["budget_bytes"])
            self.assertEqual(sum("model.embed_tokens.weight" in keys for keys in read_calls), 1)
            self.assertEqual(snapshot["avoided_read_bytes"], 5 * 17 * 8 * 4)
        # Independent baseline: without a hash cache, the physical tensor must
        # be fetched twice per forward, once per alias (6 reads across 3 runs).
        baseline_index = SafetensorsIndex(self.path)
        baseline_calls = []
        original_baseline_read = baseline_index.read
        def count_baseline(names):
            baseline_calls.append(list(names))
            return original_baseline_read(names)
        baseline_index.read = count_baseline
        with torch.device("meta"):
            baseline_meta = ToyDecoder(tied=True)
        with LayerPager(baseline_meta, baseline_index, cpu_cache_mb=0) as baseline_pager:
            for _ in range(3):
                with baseline_pager.session() as model:
                    torch.testing.assert_close(model(self.ids), source(self.ids), rtol=0, atol=0)
        self.assertEqual(sum("model.embed_tokens.weight" in keys for keys in baseline_calls), 6)

    def test_hash_table_admits_tied_weight_once_with_limited_ram(self):
        # Each alias consumes ~768KiB, so stage-wise caching cannot fit both
        # copies in 1MiB, whereas tensor-keyed caching fits the unique tensor.
        source = ToyDecoder(tied=True, vocab=12000, width=16).eval()
        save_file({k: v.detach().contiguous() for k, v in source.state_dict().items()
                   if k != "lm_head.weight"}, self.path / "shared.safetensors")
        (self.path / "model.safetensors").unlink()
        (self.path / "shared.safetensors").rename(self.path / "model.safetensors")
        with torch.device("meta"):
            meta = ToyDecoder(tied=True, vocab=12000, width=16)
        with LayerPager(meta, SafetensorsIndex(self.path), cpu_cache_mb=1) as pager:
            with pager.session() as model:
                torch.testing.assert_close(model(self.ids), source(self.ids), rtol=0, atol=0)
            snap = pager.snapshot()["cpu_cache"]
            self.assertIn("lm_head", snap["planned_stages"])
            self.assertIn("model.embed_tokens", snap["planned_stages"])
            self.assertLess(snap["occupied_bytes"], 1024 * 1024)
            self.assertLess(snap["cached_tensors"], len(pager.index.keys()) + 1)
            self.assertGreaterEqual(snap["deduplicated_planned_bytes"], 12000 * 16 * 4)

    def test_invalid_cache_settings_are_rejected(self):
        for budget in (-1, 1.1, True):
            with self.subTest(budget=budget), self.assertRaises(PagingError):
                LayerPager(self.meta, self.index, cpu_cache_mb=budget)
        with self.assertRaises(PagingError):
            LayerPager(self.meta, self.index, pin_stages=["lm_head"])
        with self.assertRaises(PagingError):
            LayerPager(self.meta, self.index, cpu_cache_mb=1, pin_stages=["unknown"])
        with self.assertRaises(PagingError):
            LayerPager(self.meta, self.index, cpu_cache_mb=1,
                       pin_stages=["lm_head", "lm_head"])

    def test_checkpoint_inventory_reports_payload_and_cache_plan(self):
        from ironasu.checkpoint import inspect_checkpoint
        report = inspect_checkpoint(self.path, cpu_cache_mb=1)
        self.assertEqual(report["shards"], 1)
        self.assertEqual(report["tensor_count"], len(self.index.keys()))
        self.assertEqual(report["unrecognized_tensor_bytes"], 0)
        self.assertEqual(report["tensor_payload_bytes"],
                         sum(self.index.tensor_byte_sizes(sorted(self.index.keys())).values()))
        self.assertIn("model.layers.0", report["stages_bytes"])
        self.assertIn("lm_head", report["cpu_cache_plan"]["estimated_pinned_stages"])
        self.assertLessEqual(report["cpu_cache_plan"]["estimated_pinned_bytes"], 1024 * 1024)

    def test_checkpoint_inventory_rejects_invalid_budget(self):
        from ironasu.checkpoint import inspect_checkpoint
        for bad in (-1, 1.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inspect_checkpoint(self.path, cpu_cache_mb=bad)

    def test_safetensors_metadata_only_byte_count(self):
        sizes = self.index.tensor_byte_sizes([
            "model.embed_tokens.weight", "model.layers.0.weight",
            "model.layers.0.weight",
        ])
        self.assertEqual(sizes["model.embed_tokens.weight"], 17 * 8 * 4)
        self.assertEqual(sizes["model.layers.0.weight"], 8 * 8 * 4)
        self.assertEqual(len(sizes), 2)

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
