#!/usr/bin/env python3
"""Patch vLLM model registry to expose text-only Qwen3.6 (qwen3_5_moe) classes.

Background: Upstream PRs to register `Qwen3_5ForCausalLM` and
`Qwen3_5MoeForCausalLM` (text-only) were closed unmerged (#36289, #36607, #36850).
The classes exist in `vllm/model_executor/models/qwen3_5.py` but aren't wired
into `_TEXT_GENERATION_MODELS`, so vLLM falls through to the multimodal pathway
and crashes on `vision_config.spatial_merge_size` for text-only checkpoints.

This patch is idempotent — safe to run multiple times.
"""
import sys
from pathlib import Path

REGISTRY = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
)

src = REGISTRY.read_text()

# Idempotency guard
if '"Qwen3_5MoeForCausalLM"' in src:
    print("[register_qwen3_5_text] already applied — skipping")
    sys.exit(0)

# Anchor: insert immediately after the existing Qwen3MoeForCausalLM entry inside
# _TEXT_GENERATION_MODELS. The agent recon confirmed this exists on line ~196.
ANCHOR = '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),\n'
ADDITION = (
    '    # Manually registered: Qwen3.6 text-only classes\n'
    '    # Upstream PRs (#36289, #36607, #36850) to add these were closed unmerged.\n'
    '    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
    '    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
)

if ANCHOR not in src:
    raise RuntimeError(
        f"Anchor not found in {REGISTRY}.\n"
        "vLLM registry layout may have changed — re-inspect with:\n"
        "  grep -n 'Qwen3MoeForCausalLM' "
        f"{REGISTRY}"
    )

new_src = src.replace(ANCHOR, ANCHOR + ADDITION, 1)
REGISTRY.write_text(new_src)
print(f"[register_qwen3_5_text] inserted Qwen3_5*ForCausalLM entries into {REGISTRY}")
