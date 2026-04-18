# NVFP4 quantization recipe

How the `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` checkpoint was produced. Use this if you want to:
- Re-quantize with a different calibration set
- Quantize a different Qwen3.6 fine-tune
- Reproduce the artifact for audit

## Format

| | |
|---|---|
| Quant scheme | NVFP4 (FP4 E2M1 + per-block FP8 e4m3 scales + per-tensor FP32 scales) |
| Block size | 16 |
| Format | `nvfp4-pack-quantized` (compressed-tensors 0.14+) |
| Weights | per-tensor + per-block scales |
| Activations | dynamic per-block (`dynamic="local"`) |
| MoE expert layout | per-expert (NOT pre-fused — vLLM fuses on load) |
| Preserved fp16/bf16 | `lm_head`, `embed_tokens`, MoE routing gates, all norms, MTP/NextN heads |

## Hardware needs

- **GPU**: 1× A100 80GB or H100 80GB (RunPod $2-3/hr)
- **CPU RAM**: 96 GB minimum (BF16 model + calibration buffers)
- **Disk**: 200 GB
- **Time**: ~3 hours end-to-end (download + calibrate + save)

We used a RunPod `H100 PCIe` with 96 GB RAM, 250 GB disk.

## Recipe

```bash
# Pod startup (RunPod)
apt-get update && apt-get install -y python3-pip git
pip install -U \
  llmcompressor \
  compressed-tensors \
  torch \
  transformers \
  accelerate \
  huggingface_hub[hf_transfer] \
  datasets

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=hf_xxxxxxxxx
```

The quantization script:

```python
# qwen36_requant.py — see ../scripts/qwen36_requant.py for the full version
import os, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch

SOURCE = "tvall43/Qwen3.6-35B-A3B-heretic"
LOCAL = "/workspace/qwen36-bf16"
OUTPUT = "/workspace/qwen36-nvfp4"

# 1. Download (~70 GB BF16)
snapshot_download(SOURCE, local_dir=LOCAL,
                  ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"])

# 2. Load to CPU (avoids GPU OOM during MoE expert replacement)
tokenizer = AutoTokenizer.from_pretrained(LOCAL, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(LOCAL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 3. Quantization recipe — generous ignore list preserves accuracy
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*embed_tokens",
        # MoE routing — must stay high-precision
        "re:.*gate$",
        "re:.*router.*",
        "re:.*shared_expert_gate.*",
        # Norms
        "re:.*norm.*",
        # MTP / NextN speculative heads — preserve for native spec decode compat
        "re:.*mtp.*",
        "re:.*nextn.*",
        "re:.*next_n.*",
        "re:.*tok_proj.*",
    ],
)

# 4. Sequential calibration — moves one decoder layer to GPU at a time
# Critical for 256-expert MoE: full-model GPU placement OOMs at init
oneshot(
    model=model,
    dataset="open_platypus",        # general-purpose calibration corpus
    recipe=recipe,
    output_dir=OUTPUT,
    processor=processor,
    max_seq_length=2048,
    num_calibration_samples=256,
    pipeline="sequential",
    sequential_targets=["Qwen3_5MoeDecoderLayer"],   # one layer at a time
)

tokenizer.save_pretrained(OUTPUT)
processor.save_pretrained(OUTPUT)
```

Run it:

```bash
python3 qwen36_requant.py 2>&1 | tee quant.log
```

Expected output:
- 41 sequential calibration stages (40 decoder layers + final norm/lm_head)
- ~3 minutes per layer with 256 calibration samples
- Final NVFP4 size: ~20 GB safetensors
- 124,423 keys (40 layers × 256 experts × 12 quant components per expert)

## Verification

```python
# Check output keys are well-formed
from safetensors import safe_open

with safe_open(f"{OUTPUT}/model.safetensors", framework="pt") as f:
    keys = sorted(f.keys())

# Sanity checks
assert any("experts.0.gate_proj.weight_packed" in k for k in keys), "missing expert weights"
assert any("input_global_scale" in k for k in keys), "missing activation scales"
assert "lm_head.weight" in keys, "lm_head should be preserved unquantized"

# Scale magnitudes — sanity-check that scales aren't inverted
import torch
with safe_open(f"{OUTPUT}/model.safetensors", framework="pt") as f:
    s = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.input_global_scale")
    print(f"input_global_scale: {s.item():.2f}")   # healthy values: 50-150

print(f"Total keys: {len(keys)}")
print(f"Output size: {os.path.getsize(f'{OUTPUT}/model.safetensors') / 1e9:.1f} GB")
```

If `input_global_scale` is < 1 or > 1e4, your scales are inverted (a known modelopt bug — llmcompressor doesn't have it, so this should always pass for our recipe).

## Upload to Hugging Face

```bash
huggingface-cli upload AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 \
  /workspace/qwen36-nvfp4 . \
  --repo-type model
```

## Choosing a different calibration corpus

`open_platypus` is a general-purpose mix; for specialized models:

| Workload | Recommended dataset |
|---|---|
| General chat | `open_platypus` (default) |
| Math / reasoning | `gsm8k` or `MathX-5M` subset |
| Code | `bigcode/the-stack-v2` subset |
| Long-context | `pg19` or `proof-pile-2` |
| Multi-turn | `OpenAssistant/oasst1` |

256 samples × 2048 seq length = 524,288 calibration tokens. For long-context calibration, consider 128 samples × 4096 seq instead — same token budget, better long-range scale estimates.

## Troubleshooting

### OOM during model load

You're loading to GPU instead of CPU. Confirm `device_map="cpu"` and `low_cpu_mem_usage=True`.

### `RuntimeError: AutoConfig has no class for qwen3_5_moe_text`

Update transformers to ≥ 5.5.0:

```bash
pip install -U "transformers>=5.5.0"
```

### Calibration runs forever (one layer takes > 30 min)

You're not using sequential pipeline. Confirm `pipeline="sequential"` and `sequential_targets=["Qwen3_5MoeDecoderLayer"]`.

### Output file is huge (> 30 GB)

Some weights are being saved unquantized. Inspect the recipe `ignore` list — anything not in `ignore` AND matching `targets="Linear"` will be quantized to NVFP4 (1/8 the size of bf16).

## Cost estimate

| Step | Time | RunPod cost (H100) |
|---|---|---|
| Pod boot + setup | 5 min | $0.20 |
| Download BF16 model | 15 min | $0.60 |
| Calibration (40 layers × 3 min) | 120 min | $4.80 |
| Save + verify | 5 min | $0.20 |
| Upload to HF | 10 min | $0.40 |
| **Total** | **~2.5 hr** | **~$6** |

Cheaper on A100 (~$3 total) but ~50% slower (3.5 hr).
