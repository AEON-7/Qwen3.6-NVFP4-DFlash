#!/usr/bin/env python3
"""Strip `language_model.` segment from safetensors keys.

Background:
  Source model `tvall43/Qwen3.6-35B-A3B-heretic` was structured as multimodal
  Qwen3.6 (transformers' `Qwen3_5MoeForConditionalGeneration` style with
  `self.language_model = ...` wrapping the text part). When we quantized via
  llmcompressor, the saved keys preserved that hierarchy:
      model.language_model.layers.X.*

  But our config declares text-only architecture:
      architectures: ["Qwen3_5MoeForCausalLM"]
      model_type: qwen3_5_moe_text

  vLLM's text-only Qwen3_5MoeForCausalLM has `self.model = Qwen3_5Model(...)`
  (no extra `language_model` wrapper), so its named_parameters yield:
      model.layers.X.*

  Mismatch → KeyError on every parameter lookup.

Fix:
  Rewrite the safetensors file with keys remapped:
      model.language_model.X.*   →   model.X.*
      model.language_model       →   model
  (lm_head.* stays unchanged.)

Usage:
  strip_language_model_prefix.py SRC_DIR DST_DIR
"""
import sys
import os
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def remap_key(k: str) -> str:
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model.") :]
    return k


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src_dir = Path(sys.argv[1]).resolve()
    dst_dir = Path(sys.argv[2]).resolve()

    if not (src_dir / "model.safetensors").exists():
        print(f"ERROR: {src_dir}/model.safetensors not found", file=sys.stderr)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy non-tensor files unchanged
    print(f"[copy] {src_dir} -> {dst_dir}")
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, dst_dir / f.name)
            print(f"  copied {f.name}")

    # 2. Read tensors + remap keys + save
    src_st = src_dir / "model.safetensors"
    dst_st = dst_dir / "model.safetensors"

    print(f"[remap] reading {src_st} ({src_st.stat().st_size / 1e9:.1f} GB)...")
    t0 = time.time()
    new_state = {}
    n_remapped = 0
    n_unchanged = 0
    with safe_open(src_st, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            new_k = remap_key(k)
            if new_k != k:
                n_remapped += 1
            else:
                n_unchanged += 1
            new_state[new_k] = f.get_tensor(k)
        # Try to preserve format metadata
        meta = f.metadata() or {}

    print(f"[remap] read {len(keys)} tensors in {time.time()-t0:.0f}s")
    print(f"[remap] remapped: {n_remapped}, unchanged: {n_unchanged}")
    if n_remapped == 0:
        print("[remap] WARN: nothing to remap — keys may already be in correct form")

    print(f"[save] writing {dst_st}...")
    t0 = time.time()
    save_file(new_state, str(dst_st), metadata=meta)
    print(f"[save] wrote in {time.time()-t0:.0f}s")
    print(f"[save] output size: {dst_st.stat().st_size / 1e9:.1f} GB")

    # 3. Sanity check
    print(f"[verify] reopening {dst_st}...")
    with safe_open(dst_st, framework="pt") as f:
        new_keys = sorted(f.keys())
    print(f"[verify] {len(new_keys)} keys in output")
    print("[verify] first 3 keys:")
    for k in new_keys[:3]:
        print(f"  {k}")
    print("[verify] sample expert key:")
    expert_keys = [k for k in new_keys if "experts.0.down_proj" in k]
    if expert_keys:
        print(f"  {expert_keys[0]}")

    bad = [k for k in new_keys if "language_model" in k]
    if bad:
        print(f"[verify] WARN: {len(bad)} keys still contain 'language_model':")
        for k in bad[:3]:
            print(f"  {k}")
    else:
        print("[verify] OK: no 'language_model' segment in any key")

    print(f"\n[done] new checkpoint at {dst_dir}")


if __name__ == "__main__":
    main()
