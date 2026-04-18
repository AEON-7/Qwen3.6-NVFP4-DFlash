#!/usr/bin/env python3
"""NVFP4 quantization of Qwen3.6-35B-A3B-heretic using llmcompressor.

Source: tvall43/Qwen3.6-35B-A3B-heretic (BF16, ~70 GB, qwen3_5_moe architecture)
Output: compressed-tensors NVFP4 (~18 GB expected)
Tool:   llmcompressor w/ NVFP4 scheme (proven working on Gemma4 pipeline)

Key architectural notes:
  - 256 experts (2× Gemma4's 128) — even more sparsity → more memory savings
  - Hybrid attention: Gated DeltaNet (linear) + Gated Attention
  - Has MTP / NextN heads for native speculative decoding — PRESERVE in quantization
  - 40 layers (30 in Gemma4)

Strategy:
  - device_map="cpu" to avoid GPU OOM during initialization
  - pipeline="sequential" with sequential_targets=["Qwen35MoeDecoderLayer"] (or similar)
    to calibrate one decoder layer at a time
  - Ignore vision tower, norms, routers, gates, embed/lm_head, AND MTP heads
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SOURCE_MODEL = "tvall43/Qwen3.6-35B-A3B-heretic"
OUTPUT_DIR = "/workspace/qwen36-heretic-nvfp4"
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
    """Inspect the model structure to identify decoder layer class name + MTP heads."""
    from transformers import AutoConfig
    import json
    with open(f"{source_dir}/config.json") as f:
        cfg = json.load(f)
    print(f"[cfg] model_type: {cfg.get('model_type')}")
    print(f"[cfg] architectures: {cfg.get('architectures')}")
    # Count MTP layers if advertised
    for k in ("num_nextn_predict_layers", "num_mtp_layers", "mtp_num_layers"):
        if k in cfg:
            print(f"[cfg] {k}: {cfg[k]}")
    tc = cfg.get("text_config", cfg)
    print(f"[cfg] num_hidden_layers: {tc.get('num_hidden_layers')}")
    print(f"[cfg] hidden_size: {tc.get('hidden_size')}")
    print(f"[cfg] num_experts: {tc.get('num_experts', tc.get('num_local_experts'))}")
    print(f"[cfg] moe_intermediate_size: {tc.get('moe_intermediate_size')}")
    return cfg


def quantize(source_dir, output_dir, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
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

    # Load model to CPU — sequential calibration moves layers to GPU one at a time
    print(f"[load] model from {source_dir} (CPU offload)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        source_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[load] model loaded in {time.time()-t0:.0f}s (on CPU)")

    # Identify the decoder layer class name for sequential calibration
    # Try common candidates for Qwen 3.5 / 3.6 MoE
    layer_class = None
    for name in ["Qwen3NextDecoderLayer", "Qwen3MoeDecoderLayer",
                 "Qwen35MoeDecoderLayer", "Qwen3_5MoeDecoderLayer",
                 "Qwen36MoeDecoderLayer", "Qwen2MoeDecoderLayer"]:
        if any(name in str(type(m)) for m in model.modules()):
            layer_class = name
            break
    if layer_class is None:
        # Fallback — find any "DecoderLayer" class
        for m in model.modules():
            t = str(type(m))
            if "DecoderLayer" in t and "Qwen" in t:
                layer_class = t.split("'")[1].split(".")[-1]
                break
    print(f"[arch] decoder layer class: {layer_class or 'UNKNOWN'}")

    # Recipe: NVFP4 quantization with generous ignore list
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            "re:.*embed_tokens",
            # Vision tower (if multimodal)
            "re:.*visual.*",
            "re:.*vision.*",
            "re:.*mm_projector.*",
            # MoE routing components (critical for accuracy)
            "re:.*gate$",
            "re:.*router.*",
            "re:.*shared_expert_gate.*",
            # Norms
            "re:.*norm.*",
            # MTP / NextN speculative heads — PRESERVE for native spec decode
            "re:.*mtp.*",
            "re:.*nextn.*",
            "re:.*next_n.*",
            "re:.*tok_proj.*",   # NextN token projection
        ],
    )

    print(f"[quant] running oneshot NVFP4 ({CALIB_SAMPLES} samples, sequential mode)...")
    t0 = time.time()
    kwargs = dict(
        model=model,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=output_dir,
        processor=processor,
        max_seq_length=CALIB_SEQ_LEN,
        num_calibration_samples=CALIB_SAMPLES,
        pipeline="sequential",
    )
    if layer_class:
        kwargs["sequential_targets"] = [layer_class]
    oneshot(**kwargs)
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
    print(f"\n[ok] NVFP4 model at {output_dir} — {sz}, {n_files} files")


def main():
    source_dir = download_source()
    cfg = inspect_architecture(source_dir)
    quantize(source_dir, OUTPUT_DIR, cfg)


if __name__ == "__main__":
    main()
