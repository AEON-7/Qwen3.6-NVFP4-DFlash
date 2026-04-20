# NVFP4 quantization recipe (v2 — multimodal preserved)

How the `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` v2 checkpoint was produced (re-quantized 2026-04-19). Use this if you want to:
- Re-quantize with a different calibration set
- Quantize a different Qwen3.6 fine-tune
- Reproduce the artifact for audit

> **What changed from v1:** v1 used `AutoModelForCausalLM` and stripped the
> `language_model.` prefix to match a text-only Qwen3_5MoeForCausalLM registry hack.
> That layout caused intermittent NaN/crash in vLLM's prefix-strip codepath in production.
> v2 uses `AutoModelForImageTextToText` to preserve the full multimodal architecture
> (`Qwen3_5MoeForConditionalGeneration` + 27-block ViT vision encoder kept BF16).
> vLLM's canonical multimodal class loads it natively — no prefix-strip patches needed.

## Format

| | |
|---|---|
| Quant scheme | NVFP4 (FP4 E2M1 + per-block FP8 e4m3 scales + per-tensor FP32 scales) |
| Block size | 16 |
| Format | `nvfp4-pack-quantized` (compressed-tensors 0.14+) |
| Weights | per-tensor + per-block scales |
| Activations | dynamic per-block (`dynamic="local"`) |
| MoE expert layout | per-expert, all 256 experts × 40 layers calibrated (`moe_calibrate_all_experts=True`) |
| Preserved fp16/bf16 | `lm_head`, `embed_tokens`, MoE routing gates, all norms, **27-block ViT vision encoder**, **30 linear_attn (Mamba/GDN) layers** |
| Total NVFP4 keys | 122,880 (40 layers × 256 experts × 3 projs × 4 quant components) |
| Total preserved BF16 keys | 333 visual + 270 linear_attn + 241 norms + lm_head + embed_tokens |

## Hardware needs

- **GPU**: 1× **RTX PRO 6000 Blackwell** (96 GB) recommended — what v2 was actually built on. A100 80GB / H100 80GB also work.
- **CPU RAM**: 96 GB minimum (BF16 model offload during sequential calibration). Peaks around 77 GB used.
- **Disk**: 200 GB (BF16 source ~70 GB + NVFP4 output ~22 GB + scratch)
- **Time**: ~3 hours end-to-end on RTX PRO 6000 Blackwell

The v2 calibration ran on a RunPod RTX PRO 6000 Blackwell pod with 96 GB RAM, 250 GB disk.
Expect ~$7-9 in compute on RunPod.

## Recipe (v2 — exact script that produced the production checkpoint)

```bash
# Pod startup (RunPod)
apt-get update && apt-get install -y python3-pip git
pip install -U \
  llmcompressor \
  compressed-tensors \
  torch \
  accelerate \
  huggingface_hub[hf_transfer] \
  datasets

# CRITICAL ORDER: install transformers + huggingface_hub LAST and pinned, since llmcompressor
# may pull in older versions that don't recognize qwen3_5_moe
pip install -U --force-reinstall "transformers>=5.5.4" "huggingface_hub>=1.10"

# CRITICAL: torchvision must NOT be installed — it cascades into a
# `torchvision::nms operator does not exist` at quant time on cu130 nightly torch.
pip uninstall -y torchvision

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=hf_xxxxxxxxx
```

The actual v2 quantization script (also at `scripts/qwen36_requant_v2.py`):

```python
# qwen36_requant_v2.py — produces multimodal-preserved Qwen3.6 NVFP4
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch

SOURCE = "tvall43/Qwen3.6-35B-A3B-heretic"
LOCAL = "/workspace/qwen36-bf16"
OUTPUT = "/workspace/qwen36-nvfp4-v2"

# 1. Download BF16 source (~70 GB)
snapshot_download(SOURCE, local_dir=LOCAL,
                  ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"])

# 2. Load to CPU as a multimodal model (preserves vision encoder)
tokenizer = AutoTokenizer.from_pretrained(LOCAL, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(LOCAL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(   # ← KEY DIFFERENCE FROM v1
    LOCAL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 3. Recipe — wider ignore list to skip the vision tower + linear-attn layers
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*embed_tokens",
        # MoE routing — must stay high-precision
        "re:.*mlp\\.gate$",            # exact MoE router (NOT gate_proj)
        "re:.*shared_expert_gate.*",
        # Norms
        "re:.*norm.*",
        # Vision tower — keep BF16 (NVFP4 ViT path not validated)
        "re:.*visual\\..*",
        # Linear / Mamba / Gated-DeltaNet attention layers — keep BF16
        # (these are 30 of the 40 attention layers; quantizing them tanks accuracy
        #  and they're tiny relative to the MoE experts anyway)
        "re:.*linear_attn\\..*",
    ],
)

# 4. Sequential calibration — moves one decoder layer to GPU at a time
oneshot(
    model=model,
    dataset="open-platypus",         # ← registered alias; "open_platypus" also works depending on version
    recipe=recipe,
    output_dir=OUTPUT,
    processor=processor,
    max_seq_length=2048,
    num_calibration_samples=256,
    pipeline="sequential",
    sequential_targets=["Qwen3_5MoeDecoderLayer"],
)

tokenizer.save_pretrained(OUTPUT)
processor.save_pretrained(OUTPUT)
```

Run it:

```bash
python3 qwen36_requant.py 2>&1 | tee quant.log
```

Expected output:
- 40 sequential calibration stages (one per `Qwen3_5MoeDecoderLayer`)
- ~3 minutes per layer with 256 calibration samples on RTX PRO 6000 Blackwell
- Final NVFP4 size: ~22 GB across 9 safetensors shards
- 122,880 expert keys (40 layers × 256 experts × 3 projs × 4 quant components)
- Plus 333 visual + 270 linear_attn + 241 norms preserved BF16

## Verification

```python
# Check output keys are well-formed for v2 multimodal layout
from safetensors import safe_open
import glob, os

shards = sorted(glob.glob(f"{OUTPUT}/model-*.safetensors"))
all_keys = []
for shard in shards:
    with safe_open(shard, framework="pt") as f:
        all_keys.extend(f.keys())

# v2 sanity checks — multimodal layout
assert any("model.language_model.layers" in k for k in all_keys), \
    "v2 must preserve language_model. prefix (you may have used AutoModelForCausalLM by mistake)"
assert any("experts.0.gate_proj.weight_packed" in k for k in all_keys), \
    "missing per-expert NVFP4 weights"
assert any("input_global_scale" in k for k in all_keys), \
    "missing activation scales"
assert "lm_head.weight" in all_keys, "lm_head should be preserved unquantized"

# Vision tower preserved BF16
visual_keys = [k for k in all_keys if "visual" in k]
assert len(visual_keys) > 100, f"visual tower seems missing or quantized (only {len(visual_keys)} keys)"

# Scale magnitudes — sanity-check that scales aren't inverted
with safe_open(shards[0], framework="pt") as f:
    for k in f.keys():
        if "input_global_scale" in k:
            s = f.get_tensor(k)
            print(f"sample input_global_scale: {s.item():.2f}")   # healthy: 50-150
            break

total_size = sum(os.path.getsize(s) for s in shards)
print(f"Total keys: {len(all_keys)}")
print(f"Output size: {total_size / 1e9:.1f} GB across {len(shards)} shards")
print(f"Visual keys (BF16): {len(visual_keys)}")
print(f"Expert keys (NVFP4): {sum(1 for k in all_keys if 'experts.' in k and '_proj.' in k)}")
```

Expected on a healthy v2 build:
- Total keys: ~124,000+
- Visual keys: ~333
- Expert keys: ~122,880
- Total size: ~22 GB across 9 shards
- `input_global_scale`: 50-150 range

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
