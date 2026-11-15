"""Text-only Kimi K3 checkpoint adapter and non-executing compatibility preflight.

Do not import or vendor the upstream Kimi implementation. If explicitly
requested, ironasu delegates mathematical operations to the LOCAL official
implementation, and provides its out-of-core checkpoint reader. No model
weights are downloaded by this module.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


class KimiCompatibilityError(ValueError):
    """A checkpoint cannot safely be interpreted as Kimi K3 text weights."""


class TextWeightView:
    """Strip the K3 multimodal `language_model.` prefix without copying weights.

    K3 checkpoints contain both vision and text tensors. This adapter exposes
    ONLY language weights to the existing dense/sparse pager. Reads always
    translate back to the actual names in the original safetensors index.
    It can wrap SafetensorsIndex, MXFP4Index, or other same-interface indexes.
    """

    def __init__(self, source: Any, prefix: str = "language_model."):
        if not isinstance(prefix, str) or not prefix or not prefix.endswith("."):
            raise KimiCompatibilityError("Expected a nonempty module prefix ending in '.'")
        self.source = source
        self.prefix = prefix
        self._mapping = {
            name[len(prefix):]: name
            for name in source.keys() if name.startswith(prefix) and len(name) > len(prefix)
        }
        if not self._mapping:
            raise KimiCompatibilityError(f"No checkpoint weights with prefix {prefix}")
        # Do not silently mix image weights or accept nonsensical keys.
        required = ("model.embed_tokens.weight", "lm_head.weight")
        missing = [k for k in required if k not in self._mapping]
        if missing:
            raise KimiCompatibilityError(f"Missing required Kimi text weight(s): {missing}")

    def keys(self) -> set[str]:
        return set(self._mapping)

    def _resolve(self, names: list[str]) -> list[str]:
        resolved = []
        for name in dict.fromkeys(names):
            if name not in self._mapping:
                raise KimiCompatibilityError(f"Unknown text weight: {name}")
            resolved.append(self._mapping[name])
        return resolved

    def tensor_byte_sizes(self, names: list[str]) -> dict[str, int]:
        full = self.source.tensor_byte_sizes(self._resolve(names))
        return {name: full[self._mapping[name]] for name in dict.fromkeys(names)}

    def read(self, names: list[str]) -> dict[str, Any]:
        full = self.source.read(self._resolve(names))
        return {name: full[self._mapping[name]] for name in dict.fromkeys(names)}


def _read_config(directory: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise KimiCompatibilityError("Expected a local model directory")
    with (root / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "kimi_k3" or "KimiK3ForConditionalGeneration" not in config.get("architectures", []):
        raise KimiCompatibilityError("This is not an official-format Kimi K3 multimodal config")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "kimi_linear":
        raise KimiCompatibilityError("Missing Kimi Linear text configuration")
    return root, config, text


def require_local_kimi_code(directory: str | Path) -> None:
    """Ensure custom model code resolves only to explicit local files.

    This is a path validation, NOT a security audit of the Python code. Callers
    must still opt in explicitly before executing any local model Python.
    """
    root, outer, text = _read_config(directory)
    allowed = {
        ("AutoConfig", "configuration_kimi_k3.KimiK3Config"),
        ("AutoModel", "modeling_kimi_k3.KimiK3ForConditionalGeneration"),
        ("AutoModelForCausalLM", "modeling_kimi_k3.KimiK3ForConditionalGeneration"),
    }
    actual = outer.get("auto_map")
    if not isinstance(actual, dict) or any((key, value) not in allowed for key, value in actual.items()):
        raise KimiCompatibilityError("Unexpected Kimi outer code mapping")
    text_map = text.get("auto_map")
    if not isinstance(text_map, dict) or text_map.get("AutoModelForCausalLM") != "modeling_kimi_linear.KimiLinearForCausalLM":
        raise KimiCompatibilityError("Unexpected Kimi text model code mapping")
    for name in ("configuration_kimi_k3.py", "modeling_kimi_linear.py", "modeling_kimi_k3.py"):
        try:
            path = (root / name).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise KimiCompatibilityError(f"Missing or unsafe local code file: {name}") from exc
        if not path.is_file() or not path.is_relative_to(root):
            raise KimiCompatibilityError(f"Missing or unsafe local code file: {name}")


def inspect_kimi_config(directory: str | Path) -> dict[str, Any]:
    """Summarize K3 text requirements without loading or executing Python/weights."""
    root, config, text = _read_config(directory)
    integers = ("num_hidden_layers", "num_experts", "num_experts_per_token", "hidden_size")
    for name in integers:
        if type(text.get(name)) is not int or text[name] <= 0:
            raise KimiCompatibilityError(f"Invalid Kimi config field: {name}")
    if text["num_experts_per_token"] > text["num_experts"]:
        raise KimiCompatibilityError("Selected experts exceed expert pool")
    attn = text.get("linear_attn_config")
    if not isinstance(attn, dict):
        raise KimiCompatibilityError("Missing Kimi attention layout")
    kda = set(attn.get("kda_layers", []))
    mla = set(attn.get("full_attn_layers", []))
    expected = set(range(1, text["num_hidden_layers"] + 1))
    if (any(type(n) is not int for n in kda | mla) or kda & mla
            or kda | mla != expected):
        raise KimiCompatibilityError("KDA/MLA layer layout incomplete or overlapping")
    result: dict[str, Any] = {
        "model_type": config["model_type"],
        "text_model_type": text["model_type"],
        "num_hidden_layers": text["num_hidden_layers"],
        "kda_layers": len(kda),
        "mla_layers": len(mla),
        "num_experts": text["num_experts"],
        "experts_per_token": text["num_experts_per_token"],
        "expert_fraction_per_token": text["num_experts_per_token"] / text["num_experts"],
        "attn_res_block_size": text.get("attn_res_block_size"),
        "text_only": True,
        "code_executed": False,
        "weights_loaded": False,
        "architecture_supported": "requires locally reviewed Kimi upstream code and runtime dependencies",
    }
    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        with index_file.open("r", encoding="utf-8") as f:
            weights = json.load(f).get("weight_map")
        if not isinstance(weights, dict):
            raise KimiCompatibilityError("Invalid checkpoint weight_map")
        counts = Counter()
        for key in weights:
            if key.startswith("language_model."):
                counts["text"] += 1
                if ".experts." in key:
                    counts["expert"] += 1
            else:
                counts["non_text"] += 1
        result["checkpoint"] = {
            "total_tensor_keys": len(weights),
            "text_tensor_keys": counts["text"],
            "expert_tensor_keys": counts["expert"],
            "non_text_tensor_keys_ignored": counts["non_text"],
            "metadata_only": True,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely inspect Kimi K3 metadata without loading 2.8T weights")
    parser.add_argument("model_dir", help="local Kimi K3 model directory")
    args = parser.parse_args()
    print(json.dumps(inspect_kimi_config(args.model_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
