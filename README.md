# Qwen-Image BF16 to ComfyUI FP8 Scaled Converter

Convert a **Qwen-Image BF16** safetensors checkpoint into a ComfyUI-compatible FP8 (E4M3FN) scaled checkpoint.

This repository is focused on format compatibility and reproducibility:
- Per-layer `weight_scale` tensors are written as `float32` scalars.
- Quantization metadata is written to `_quantization_metadata`.
- A verification script validates the output format.

## Hugging Face Model
https://huggingface.co/ingtellect/qwen-image-2512-fp8-scaled-comfyui

## Scope
- Target use case: Qwen-Image BF16 checkpoint conversion.
- This is not positioned as a generic converter for all model families.

## What this repo claims
- Produces checkpoints that pass `verify_fp8_checkpoint.py`.
- Uses per-tensor FP8 scaling for 2D linear weights.

## What this repo does NOT claim
- It does **not** guarantee quality improvements for every LoRA/workflow.
- Runtime behavior can vary by ComfyUI version, model family, and workflow setup.

## Support policy
This project is published as a reference implementation.
No support, maintenance, or backward-compatibility guarantees are provided.

## Files
- `convert_bf16_to_fp8_scaled.py`: converter
- `verify_fp8_checkpoint.py`: format checker

## Recommended environment
Use the Python interpreter from your ComfyUI installation.

## Usage (ComfyUI Python)

### 1) Set your ComfyUI Python path
```powershell
$COMFY_PY = "<path-to-your-comfyui-python>"
```
Examples:
- Portable Windows: `<ComfyUI_root>\python_embeded\python.exe`
- venv install: `<ComfyUI_root>\.venv\Scripts\python.exe`

### 2) Convert
```powershell
& $COMFY_PY convert_bf16_to_fp8_scaled.py `
  --input merged_qwen_image_2512_bf16.safetensors `
  --output merged_qwen_image_2512_fp8_scaled_comfyui.safetensors
```

### Optional GPU quantization path
```powershell
& $COMFY_PY convert_bf16_to_fp8_scaled.py `
  --input merged_qwen_image_2512_bf16.safetensors `
  --output merged_qwen_image_2512_fp8_scaled_comfyui.safetensors `
  --device cuda
```

### Optional flags
- `--write-input-scale`: write `input_scale=1.0` (`float32` scalar) for quantized layers
- `--full-precision-mm`: add `full_precision_matrix_mult=true` into per-layer quantization metadata

## Verify output
```powershell
& $COMFY_PY verify_fp8_checkpoint.py `
  merged_qwen_image_2512_fp8_scaled_comfyui.safetensors
```

## Minimal A/B validation workflow
1. Generate with BF16 baseline (fixed prompt/seed/settings).
2. Generate with converted FP8 scaled checkpoint using the same settings.
3. Compare outputs and logs.

For public release, include:
- exact prompt/seed/settings
- ComfyUI version
- checkpoint names
- verification output

