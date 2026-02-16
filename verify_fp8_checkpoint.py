#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

import torch
from safetensors import safe_open

EXPECTED_FORMAT = "float8_e4m3fn"
WEIGHT_SAMPLE_LIMIT = 16


def verify(path: str) -> bool:
    print(f"Verifying {path} ...")
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            keys = set(f.keys())
            print(f"[OK] Total tensor keys: {len(keys)}")

            if "_quantization_metadata" not in metadata:
                errors.append("Missing metadata key: _quantization_metadata")
                print_result(errors, warnings)
                return False

            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
            except json.JSONDecodeError as exc:
                errors.append(f"_quantization_metadata JSON parse failed: {exc}")
                print_result(errors, warnings)
                return False

            if not isinstance(quant_meta, dict):
                errors.append("_quantization_metadata must be a JSON object")
                print_result(errors, warnings)
                return False

            layers = quant_meta.get("layers")
            if not isinstance(layers, dict):
                errors.append("metadata.layers must be a JSON object")
                print_result(errors, warnings)
                return False

            if not layers:
                warnings.append("metadata.layers is empty")

            print(f"[OK] Declared quantized layers: {len(layers)}")

            checked_weight_samples = 0
            for layer_name, layer_conf in layers.items():
                if not isinstance(layer_conf, dict):
                    errors.append(f"Layer config must be object: {layer_name} -> {type(layer_conf).__name__}")
                    continue

                layer_format = layer_conf.get("format")
                if layer_format != EXPECTED_FORMAT:
                    errors.append(
                        f"Unexpected format at layer {layer_name}: {layer_format} (expected {EXPECTED_FORMAT})"
                    )

                weight_key = f"{layer_name}.weight"
                scale_key = f"{layer_name}.weight_scale"
                input_scale_key = f"{layer_name}.input_scale"

                if weight_key not in keys:
                    errors.append(f"Missing tensor: {weight_key}")
                if scale_key not in keys:
                    errors.append(f"Missing tensor: {scale_key}")
                    continue

                scale_t = f.get_tensor(scale_key)
                if scale_t.dtype != torch.float32:
                    errors.append(f"{scale_key} dtype is {scale_t.dtype}, expected torch.float32")
                if scale_t.numel() != 1:
                    errors.append(f"{scale_key} numel is {scale_t.numel()}, expected 1")
                elif scale_t.ndim != 0:
                    warnings.append(f"{scale_key} ndim is {scale_t.ndim}; recommended is 0-dim scalar")

                if input_scale_key in keys:
                    input_scale_t = f.get_tensor(input_scale_key)
                    if input_scale_t.dtype != torch.float32:
                        errors.append(f"{input_scale_key} dtype is {input_scale_t.dtype}, expected torch.float32")
                    if input_scale_t.numel() != 1:
                        errors.append(f"{input_scale_key} numel is {input_scale_t.numel()}, expected 1")
                    elif input_scale_t.ndim != 0:
                        warnings.append(
                            f"{input_scale_key} ndim is {input_scale_t.ndim}; recommended is 0-dim scalar"
                        )

                if checked_weight_samples < WEIGHT_SAMPLE_LIMIT and weight_key in keys:
                    weight_t = f.get_tensor(weight_key)
                    if weight_t.dtype != torch.float8_e4m3fn:
                        errors.append(f"{weight_key} dtype is {weight_t.dtype}, expected torch.float8_e4m3fn")
                    if weight_t.ndim != 2:
                        warnings.append(f"{weight_key} ndim is {weight_t.ndim}; expected 2 for linear weight")
                    checked_weight_samples += 1

            layers_from_weight_scale = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
            declared_layers = set(layers.keys())

            undeclared = sorted(layers_from_weight_scale - declared_layers)
            overdeclared = sorted(declared_layers - layers_from_weight_scale)
            if undeclared:
                errors.append(f"Layers present in state_dict but missing in metadata.layers: {len(undeclared)}")
            if overdeclared:
                errors.append(
                    f"Layers declared in metadata.layers but missing weight_scale tensor: {len(overdeclared)}"
                )

            print(f"[OK] Checked sampled weight tensors: {checked_weight_samples}")
            print(
                "[OK] Layers from weight_scale keys: "
                f"{len(layers_from_weight_scale)} (declared={len(declared_layers)})"
            )

    except Exception as exc:
        errors.append(f"Error opening or parsing file: {exc}")

    print_result(errors, warnings)
    return len(errors) == 0


def print_result(errors: list[str], warnings: list[str], max_lines: int = 40) -> None:
    if warnings:
        for msg in warnings[:max_lines]:
            print(f"[WARN] {msg}")
        if len(warnings) > max_lines:
            print(f"[WARN] ... {len(warnings) - max_lines} more warning(s) omitted")

    if errors:
        for msg in errors[:max_lines]:
            print(f"[ERR] {msg}")
        if len(errors) > max_lines:
            print(f"[ERR] ... {len(errors) - max_lines} more error(s) omitted")

    if errors:
        print(f"[FAIL] Verification failed with {len(errors)} error(s).")
    else:
        print("[PASS] Verification passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_fp8_checkpoint.py <safetensors_file>")
        sys.exit(1)

    success = verify(sys.argv[1])
    sys.exit(0 if success else 2)
