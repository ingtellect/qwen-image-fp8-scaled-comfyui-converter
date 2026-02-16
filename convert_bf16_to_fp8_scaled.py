#!/usr/bin/env python3
"""
Convert BF16/FP16/FP32 safetensors checkpoint into ComfyUI-compatible FP8 scaled checkpoint.

Output format (for each quantized linear layer):
  - {layer}.weight       : float8_e4m3fn
  - {layer}.weight_scale : float32 scalar (0-dim recommended)
  - {layer}.input_scale  : float32 scalar (optional)
  - metadata key         : _quantization_metadata (JSON)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """Quantize only 2D linear weights with enough elements."""
    if not name.endswith(".weight"):
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if tensor.ndim != 2:
        return False
    if tensor.numel() < 4096:
        return False
    return True


def quantize_per_tensor_fp8(w: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8 quantization using amax/FP8_MAX scaling."""
    w_dev = w.to(device=device, dtype=torch.float32)

    amax = torch.amax(torch.abs(w_dev))
    scale = amax / FP8_MAX
    if scale.item() == 0.0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    w_scaled = (w_dev / scale).clamp(min=-FP8_MAX, max=FP8_MAX)
    w_fp8 = w_scaled.to(dtype=FP8_DTYPE)

    scale_cpu = scale.to(dtype=torch.float32).cpu()
    return w_fp8.cpu(), scale_cpu


def convert(
    input_path: str,
    output_path: str,
    device_str: str = "cpu",
    write_input_scale: bool = False,
    full_precision_mm: bool = False,
) -> None:
    device = torch.device(device_str)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Write input_scale: {write_input_scale}")
    print(f"Full precision mm: {full_precision_mm}")
    print()

    out_state: dict[str, torch.Tensor] = {}
    quant_layers: dict[str, dict[str, object]] = {}
    skipped_count = 0
    quantized_count = 0

    t0 = time.time()

    with safe_open(input_path, framework="pt", device="cpu") as f:
        in_meta = f.metadata() or {}
        keys = list(f.keys())

        total = len(keys)
        print(f"Total tensors: {total}")
        print()

        for idx, key in enumerate(keys):
            tensor = f.get_tensor(key)

            if should_quantize(key, tensor):
                w_fp8, scale_t = quantize_per_tensor_fp8(tensor, device)
                out_state[key] = w_fp8

                scale_key = key.replace(".weight", ".weight_scale")
                out_state[scale_key] = scale_t

                if write_input_scale:
                    input_scale_key = key.replace(".weight", ".input_scale")
                    out_state[input_scale_key] = torch.tensor(1.0, dtype=torch.float32)

                layer_name = key[: -len(".weight")]
                layer_conf: dict[str, object] = {"format": "float8_e4m3fn"}
                if full_precision_mm:
                    layer_conf["full_precision_matrix_mult"] = True
                quant_layers[layer_name] = layer_conf

                quantized_count += 1
            else:
                out_state[key] = tensor
                skipped_count += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                elapsed = time.time() - t0
                pct = (idx + 1) / total * 100
                print(
                    f"\r  [{idx + 1}/{total}] {pct:5.1f}%  "
                    f"quantized={quantized_count}  skipped={skipped_count}  "
                    f"elapsed={elapsed:.1f}s",
                    end="",
                    flush=True,
                )

    print("\n")

    quant_meta = {
        "format_version": "1.0",
        "layers": quant_layers,
    }

    out_meta = dict(in_meta)
    out_meta["_quantization_metadata"] = json.dumps(quant_meta)

    print(f"Saving to {output_path} ...")
    print(f"  Quantized layers: {quantized_count}")
    print(f"  Skipped tensors:  {skipped_count}")
    print(f"  Total output keys: {len(out_state)}")

    save_file(out_state, output_path, metadata=out_meta)

    t_total = time.time() - t0
    out_size = Path(output_path).stat().st_size / (1024**3)
    print()
    print(f"Done! {t_total:.1f}s, output size: {out_size:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BF16 safetensors -> ComfyUI-compatible FP8 scaled safetensors converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input BF16/FP16/FP32 safetensors file")
    parser.add_argument("--output", "-o", required=True, help="Path to output FP8 scaled safetensors file")
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for quantization math (default: cpu)",
    )
    parser.add_argument(
        "--write-input-scale",
        action="store_true",
        help="Write input_scale=1.0 (float32 scalar) for each quantized layer",
    )
    parser.add_argument(
        "--full-precision-mm",
        action="store_true",
        help="Set full_precision_matrix_mult=true in per-layer metadata",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if Path(args.output).exists():
        resp = input(f"Output file already exists: {args.output}\nOverwrite? [y/N] ")
        if resp.lower() != "y":
            print("Canceled.")
            sys.exit(0)

    convert(
        args.input,
        args.output,
        args.device,
        write_input_scale=args.write_input_scale,
        full_precision_mm=args.full_precision_mm,
    )


if __name__ == "__main__":
    main()
