#!/usr/bin/env python3
"""NVFP4 quantization of Qwen3.6-35B-A3B-heretic — v2 multimodal-preserved.

Source: tvall43/Qwen3.6-35B-A3B-heretic (BF16, ~70 GB, qwen3_5_moe architecture)
Output: compressed-tensors NVFP4 (~22 GB across 9 shards)
Tool:   llmcompressor w/ NVFP4 scheme

What changed from v1 (qwen36_requant.py):
  - Uses AutoModelForImageTextToText (preserves multimodal architecture)
    instead of AutoModelForCausalLM (which strips the vision tower)
  - Wider ignore list: visual.* (27-block ViT) and linear_attn.* (30 Mamba layers)
    are explicitly preserved as BF16
  - Output keys retain `model.language_model.layers.X.*` prefix
  - vLLM loads via canonical Qwen3_5MoeForConditionalGeneration class
    (no registry hack, no prefix-strip post-processing required)

Why: v1 weights were unstable in production with intermittent NaN/crash in
the prefix-strip codepath. v2 uses the canonical multimodal class throughout
and runs rock-solid under sustained chat load.

Hardware: tested on RunPod RTX PRO 6000 Blackwell, 96 GB RAM, 250 GB disk.
Time: ~3 hours end-to-end.
Cost: ~$7-9 on RunPod.

PREREQUISITES (read carefully):
  pip install -U llmcompressor compressed-tensors torch accelerate \
                  huggingface_hub[hf_transfer] datasets
  # Pin transformers + huggingface_hub LAST so llmcompressor doesn't pull older versions
  pip install -U --force-reinstall "transformers>=5.5.4" "huggingface_hub>=1.10"
  # Critical: torchvision must NOT be installed (cascades into torchvision::nms missing)
  pip uninstall -y torchvision

  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_TOKEN=hf_xxxxxxxxx
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SOURCE_MODEL = "tvall43/Qwen3.6-35B-A3B-heretic"
OUTPUT_DIR = "/workspace/qwen36-heretic-nvfp4-v2"
LOCAL_SRC = "/workspace/qwen36-heretic-bf16"
CALIB_SAMPLES = 256
CALIB_SEQ_LEN = 2048


def download_source():
    """Pull Qwen3.6 heretic BF16 to pod."""
    from huggingface_hub import snapshot_download
    if os.path.exists(f"{LOCAL_SRC}/config.json"):
        print(f"[download] source already at {LOCAL_SRC}")
        return LOCAL_SRC
    print(f"[download] {SOURCE_MODEL} -> {LOCAL_SRC}")
    snapshot_download(
        repo_id=SOURCE_MODEL,
        local_dir=LOCAL_SRC,
        max_workers=8,
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"],
    )
    import subprocess
    sz = subprocess.check_output(["du", "-sh", LOCAL_SRC]).decode().split()[0]
    print(f"[download] done: {sz}")
    return LOCAL_SRC


def inspect_architecture(source_dir):
    """Inspect the model structure for sanity-check."""
    import json
    with open(f"{source_dir}/config.json") as f:
        cfg = json.load(f)
    print(f"[cfg] model_type: {cfg.get('model_type')}")
    print(f"[cfg] architectures: {cfg.get('architectures')}")
    tc = cfg.get("text_config", cfg)
    print(f"[cfg] num_hidden_layers: {tc.get('num_hidden_layers')}")
    print(f"[cfg] hidden_size: {tc.get('hidden_size')}")
    print(f"[cfg] num_experts: {tc.get('num_experts', tc.get('num_local_experts'))}")
    print(f"[cfg] moe_intermediate_size: {tc.get('moe_intermediate_size')}")
    print(f"[cfg] vision_config present: {'vision_config' in cfg}")
    return cfg


def quantize(source_dir, output_dir, cfg):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    os.makedirs(output_dir, exist_ok=True)

    print(f"[load] tokenizer + processor from {source_dir}")
    tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(source_dir, trust_remote_code=True)
        print(f"[load] processor: {processor.__class__.__name__}")
    except Exception as e:
        print(f"[load] no processor ({e}); using tokenizer")
        processor = tokenizer

    # KEY DIFFERENCE FROM V1: AutoModelForImageTextToText preserves multimodal architecture
    # Result: keys retain `model.language_model.layers.X.*` prefix that vLLM's canonical
    # Qwen3_5MoeForConditionalGeneration class loads natively.
    print(f"[load] model from {source_dir} (CPU offload, multimodal preserved)")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        source_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[load] model loaded in {time.time()-t0:.0f}s (on CPU)")
    print(f"[load] model class: {model.__class__.__name__}")

    # Recipe — wider ignore list than v1:
    #   - visual.*       skip the 27-block ViT vision tower (NVFP4 ViT path not validated)
    #   - linear_attn.*  skip the 30 linear-attention (Mamba/GDN) layers
    #                    (quantizing them tanks accuracy + they're tiny vs MoE experts)
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
            # Vision tower — keep BF16
            "re:.*visual\\..*",
            # Linear / Mamba / Gated-DeltaNet attention layers — keep BF16
            "re:.*linear_attn\\..*",
        ],
    )

    print(f"[quant] running oneshot NVFP4 ({CALIB_SAMPLES} samples, sequential mode)...")
    t0 = time.time()
    oneshot(
        model=model,
        dataset="open-platypus",         # registered alias; "open_platypus" also works
        recipe=recipe,
        output_dir=output_dir,
        processor=processor,
        max_seq_length=CALIB_SEQ_LEN,
        num_calibration_samples=CALIB_SAMPLES,
        pipeline="sequential",
        sequential_targets=["Qwen3_5MoeDecoderLayer"],
    )
    print(f"[quant] done in {time.time()-t0:.0f}s")

    # Save tokenizer + processor + misc configs
    print(f"[save] tokenizer + processor -> {output_dir}")
    tokenizer.save_pretrained(output_dir)
    try:
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"[save] processor save skipped: {e}")

    # Copy any extra config files (chat template, preprocessor, etc.)
    import shutil
    for f in ["chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "generation_config.json"]:
        src = f"{source_dir}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{output_dir}/{f}"):
            shutil.copy2(src, f"{output_dir}/{f}")
            print(f"[copy] {f}")

    import subprocess
    sz = subprocess.check_output(["du", "-sh", output_dir]).decode().split()[0]
    n_files = len([f for f in os.listdir(output_dir) if os.path.isfile(f"{output_dir}/{f}")])
    print(f"\n[ok] NVFP4 v2 model at {output_dir} — {sz}, {n_files} files")


def verify(output_dir):
    """Sanity-check that v2 multimodal layout was preserved."""
    from safetensors import safe_open
    import glob

    shards = sorted(glob.glob(f"{output_dir}/model-*.safetensors"))
    print(f"[verify] {len(shards)} safetensors shards (expected 9)")

    all_keys = []
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            all_keys.extend(f.keys())

    has_lm_prefix = any("model.language_model.layers" in k for k in all_keys)
    expert_keys = [k for k in all_keys if "experts." in k and "_proj." in k]
    visual_keys = [k for k in all_keys if "visual" in k]

    print(f"[verify] total keys: {len(all_keys)}")
    print(f"[verify] multimodal layout (language_model. prefix): {'OK' if has_lm_prefix else 'MISSING'}")
    print(f"[verify] expert keys (NVFP4): {len(expert_keys)} (expected ~122,880)")
    print(f"[verify] visual keys (BF16): {len(visual_keys)} (expected ~333)")

    assert has_lm_prefix, "language_model. prefix missing — re-run with AutoModelForImageTextToText"
    assert len(expert_keys) > 100_000, f"too few expert keys ({len(expert_keys)}) — calibration may have failed"
    assert len(visual_keys) > 100, f"vision tower seems missing ({len(visual_keys)} keys)"

    print("[verify] all checks passed — v2 multimodal layout confirmed")


def main():
    source_dir = download_source()
    cfg = inspect_architecture(source_dir)
    quantize(source_dir, OUTPUT_DIR, cfg)
    verify(OUTPUT_DIR)
    print("\n[done] Upload to HF with:")
    print(f"  huggingface-cli upload AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 \\")
    print(f"    {OUTPUT_DIR} . --repo-type model")


if __name__ == "__main__":
    main()
