"""Q4 format tests with small local safetensors files; no online model download."""
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

from ironasu.paging import LayerPager, PagingError
from ironasu.q4 import Q4Index, Q4Tensor, INDEX_FILE, pack_q4, quantize_checkpoint


@unittest.skipIf(torch is None, "torch+safetensors optional dependency unavailable")
class Q4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.original = self.path / "original"
        self.original.mkdir()
        self.quant = self.path / "q4"

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_nibbles_negative_zero_and_odd(self):
        x = torch.tensor([-7., -3., -1., 0., 1., 3., 7.] * 13)
        q = pack_q4(x, 32)
        self.assertEqual(q.shape, (91,))
        self.assertEqual(q.packed.dtype, torch.uint8)
        self.assertLessEqual((q.to(device="cpu") - x).abs().max().item(), 0.51)

    def test_float16_bfloat16_and_zeros(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for x in (torch.zeros(20, dtype=dtype), torch.randn(197, dtype=dtype)):
                q = pack_q4(x, 64)
                y = q.to(device="cpu")
                self.assertEqual(y.dtype, dtype)
                self.assertTrue(torch.isfinite(y).all().item())
                self.assertEqual(y.shape, x.shape)

    def test_reject_nan_inf_and_invalid_group(self):
        for x in (torch.tensor([float("nan")]), torch.tensor([float("inf")])):
            with self.assertRaises(ValueError):
                pack_q4(x)
        with self.assertRaises(ValueError):
            pack_q4(torch.zeros(8), 31)

    def test_quantize_checkpoint_reduces_bytes_and_reads_back(self):
        torch.manual_seed(23)
        weights = {"model.layers.0.weight": torch.randn(128, 128),
                   "model.norm.weight": torch.ones(16),
                   "some.counter": torch.arange(5, dtype=torch.int64)}
        save_file(weights, str(self.original / "model.safetensors"))
        report = quantize_checkpoint(self.original, self.quant)
        self.assertLess(report["stored_payload_bytes"], report["original_payload_bytes"] * .2)
        index = Q4Index(self.quant)
        self.assertEqual(index.keys(), set(weights))
        read = index.read(list(weights))
        self.assertIsInstance(read["model.layers.0.weight"], Q4Tensor)
        self.assertEqual(read["model.norm.weight"].dtype, torch.float32)
        self.assertTrue(torch.equal(read["some.counter"], weights["some.counter"]))
        delta = (read["model.layers.0.weight"].to(device="cpu") - weights["model.layers.0.weight"]).abs()
        self.assertLess(delta.mean().item(), .2)
        self.assertLess(index.tensor_byte_sizes(["model.layers.0.weight"])["model.layers.0.weight"],
                        weights["model.layers.0.weight"].numel())

    def test_refuse_overwrite(self):
        save_file({"a": torch.randn(256)}, str(self.original / "model.safetensors"))
        quantize_checkpoint(self.original, self.quant)
        with self.assertRaises(FileExistsError):
            quantize_checkpoint(self.original, self.quant)
        with self.assertRaises(ValueError):
            quantize_checkpoint(self.original, self.original / "new")

    def test_reject_index_path_traversal(self):
        save_file({"a": torch.randn(256)}, str(self.original / "model.safetensors"))
        quantize_checkpoint(self.original, self.quant)
        p = self.quant / INDEX_FILE
        content = json.loads(p.read_text())
        content["tensors"]["a"]["file"] = "../../untrusted.safetensors"
        p.write_text(json.dumps(content))
        with self.assertRaises(PagingError):
            Q4Index(self.quant)

    def test_corrupt_quantized_tensor_rejected(self):
        save_file({"a": torch.randn(256)}, str(self.original / "model.safetensors"))
        quantize_checkpoint(self.original, self.quant)
        idx = Q4Index(self.quant)
        path = idx.paths["a"]
        save_file({"packed": torch.zeros(2, dtype=torch.uint8),
                   "scales": torch.ones(4, dtype=torch.float32)}, str(path))
        with self.assertRaises(PagingError):
            idx.read(["a"])

    def test_toy_paged_forward_uses_compressed_weights(self):
        class ToyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(17, 8)
                self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
                self.model.norm = nn.LayerNorm(8)
                self.lm_head = nn.Linear(8, 17, bias=False)
            def forward(self, ids):
                out = self.model.embed_tokens(ids)
                for layer in self.model.layers:
                    out = torch.relu(layer(out))
                return self.lm_head(self.model.norm(out))
        torch.manual_seed(42)
        src = ToyDecoder().eval()
        save_file({key: val.detach().contiguous() for key, val in src.state_dict().items()},
                  str(self.original / "model.safetensors"))
        quantize_checkpoint(self.original, self.quant, min_elements=16)
        with torch.device("meta"):
            meta = ToyDecoder()
        ids = torch.tensor([[1, 2, 3]])
        index = Q4Index(self.quant)
        with LayerPager(meta, index, cpu_cache_mb=1) as pager:
            with pager.session() as model:
                y = model(ids)
            self.assertEqual(y.shape, src(ids).shape)
            self.assertTrue(torch.isfinite(y).all().item())
            # Compression alters values: enforce a bounded, measured regression.
            self.assertLess((y - src(ids)).abs().max().item(), .5)
            initial = pager.snapshot()["cpu_cache"]["disk_read_bytes"]
            with pager.session() as model:
                y2 = model(ids)
            torch.testing.assert_close(y2, y, atol=0, rtol=0)
            self.assertEqual(pager.snapshot()["cpu_cache"]["disk_read_bytes"], initial)
            self.assertEqual(pager.snapshot()["cpu_cache"]["cached_tensors"], len(index.keys()))


if __name__ == "__main__":
    unittest.main()
