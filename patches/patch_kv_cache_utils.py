#!/usr/bin/env python3
"""Patch vLLM hybrid-attention KV-cache handling for Qwen3.6 (+ DFlash).

Qwen3.6 has hybrid attention: 30 linear_attention layers (mamba-style state,
no KV block) + 10 full_attention layers (standard KV block).

Two distinct bugs are handled here:

1. None-handling. vLLM HEAD calls `min(block_size for group in groups)` in a
   few places, but linear_attention groups (and added padding groups) have
   block_size=None, which crashes with
   `TypeError: '<' not supported between NoneType and NoneType`.
   Fix: default mamba_block_size / filter None before min().

2. Page-size unification (surfaces with DFlash). Once the drafter's attention
   layers introduce a larger KV page size, unify_kv_cache_spec_page_size scales
   block_size to match --- which no-ops for MambaSpec (its page size doesn't
   depend on block_size), so the trailing page-size assert fails at boot.
   Fix: pad MambaSpec's physical page via page_size_padded instead. See
   patch_unify_page_size() below.

Idempotent — safe to run multiple times.
"""
import sys
from pathlib import Path


def patch_kv_cache_utils() -> None:
    target = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
    src = target.read_text()
    if "# kv_cache_utils_min_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "    min_block_size = min(\n"
        "        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]\n"
        "    )"
    )
    new = (
        "    # kv_cache_utils_min_none_safe\n"
        "    _block_sizes = [\n"
        "        group.kv_cache_spec.block_size\n"
        "        for group in kv_cache_config.kv_cache_groups\n"
        "        if group.kv_cache_spec.block_size is not None\n"
        "    ]\n"
        "    min_block_size = min(_block_sizes) if _block_sizes else 1"
    )
    if old not in src:
        raise RuntimeError(f"anchor not found in {target}")
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe min()")


def patch_engine_core() -> None:
    target = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py")
    src = target.read_text()
    if "# engine_core_block_size_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "            vllm_config.cache_config.block_size = min(\n"
        "                g.kv_cache_spec.block_size for g in kv_cache_groups\n"
        "            )"
    )
    new = (
        "            # engine_core_block_size_none_safe\n"
        "            _bs = [g.kv_cache_spec.block_size for g in kv_cache_groups if g.kv_cache_spec.block_size is not None]\n"
        "            if _bs:\n"
        "                vllm_config.cache_config.block_size = min(_bs)"
    )
    if old not in src:
        raise RuntimeError(f"anchor not found in {target}")
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe min()")


def patch_gpu_model_runner() -> None:
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
    )
    src = target.read_text()
    if "# gpu_model_runner_block_size_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "            block_size = kv_cache_group.kv_cache_spec.block_size\n"
        "            block_sizes.append(block_size)\n"
        "            max_num_blocks_per_req = cdiv(\n"
        "                max_model_len, block_size * get_total_cp_world_size()\n"
        "            )"
    )
    new = (
        "            block_size = kv_cache_group.kv_cache_spec.block_size\n"
        "            block_sizes.append(block_size)\n"
        "            # gpu_model_runner_block_size_none_safe\n"
        "            if block_size is None:\n"
        "                # MambaSpec / linear-attention groups: block-based KV doesn't apply.\n"
        "                # MambaSpec branch below overrides max_num_blocks_per_req anyway.\n"
        "                max_num_blocks_per_req = 0\n"
        "            else:\n"
        "                max_num_blocks_per_req = cdiv(\n"
        "                    max_model_len, block_size * get_total_cp_world_size()\n"
        "                )"
    )
    if old not in src:
        raise RuntimeError(f"anchor not found in {target}")
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe block_size handling")


def patch_mamba_abstract() -> None:
    """Root-cause fix: ensure MambaSpec is never constructed with block_size=None.
    Setting block_size=1 makes all downstream `block_size * X` and `X % block_size`
    arithmetic work as identity ops for Mamba/linear-attention groups."""
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/abstract.py"
    )
    src = target.read_text()
    if "# mamba_abstract_block_size_default" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "        mamba_block_size = vllm_config.cache_config.mamba_block_size\n"
        "        page_size_padded = vllm_config.cache_config.mamba_page_size_padded"
    )
    new = (
        "        mamba_block_size = vllm_config.cache_config.mamba_block_size\n"
        "        # mamba_abstract_block_size_default — None propagates through downstream\n"
        "        # `block_size * X` arithmetic and `block_size % hash_block_size` assertions.\n"
        "        # Default to the attention block_size (typically 16) so MoE/hybrid models\n"
        "        # work without padding the KV cache to 1-token granularity.\n"
        "        if mamba_block_size is None:\n"
        "            mamba_block_size = vllm_config.cache_config.block_size or 16\n"
        "        page_size_padded = vllm_config.cache_config.mamba_page_size_padded"
    )
    if old not in src:
        raise RuntimeError(f"anchor not found in {target}")
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied mamba_block_size=1 default")


def patch_unify_page_size() -> None:
    """Fix unify_kv_cache_spec_page_size for MambaSpec (GatedDeltaNet) layers.

    With DFlash, the drafter's full_attention layers introduce a larger KV page
    size than the model's linear_attention (Mamba/GDN) layers, so vLLM calls
    unify_kv_cache_spec_page_size to bring them into line. That function unifies
    by scaling block_size up by the page-size ratio --- correct for AttentionSpec,
    where page_size_bytes is linear in block_size. It is wrong for MambaSpec:
    MambaSpec.page_size_bytes is derived from a fixed per-sequence state shape
    (self.shapes / self.dtypes) and never references block_size, so multiplying
    block_size leaves page_size_bytes unchanged and the trailing
    `assert new_spec.page_size_bytes == max_page_size` fails every boot at
    kv_cache_utils.py (this is the AssertionError users hit after the None-safe
    patches above let boot get this far).

    Fix: for MambaSpec layers, pad the physical page via page_size_padded instead
    of scaling block_size. MambaSpec already carries page_size_padded for exactly
    this, and vLLM already uses the same padding path for the divisible-but-strided
    attention case a few lines down --- it's just never applied to Mamba here.
    Costs vLLM's own "may waste at most N% KV cache memory" padding warning at
    boot, which is expected. Related upstream: vllm-project/vllm#41560.
    """
    target = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
    src = target.read_text()
    if "# kv_cache_utils_mamba_page_size_padded" in src:
        print(f"[{target.name}] unify page-size already applied")
        return

    old = (
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            if max_page_size % layer_page_size == 0:"
    )
    new = (
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            # kv_cache_utils_mamba_page_size_padded\n"
        "            if isinstance(layer_spec, MambaSpec):\n"
        "                # MambaSpec.page_size_bytes is a fixed per-sequence state\n"
        "                # shape and does NOT scale with block_size, so the ratio\n"
        "                # branch below no-ops for it and the assert fails. Pad the\n"
        "                # physical page instead (page_size_padded already exists on\n"
        "                # MambaSpec for this, same as the strided-attention case).\n"
        "                new_spec = replace(layer_spec, page_size_padded=max_page_size)\n"
        "            elif max_page_size % layer_page_size == 0:"
    )
    if old not in src:
        raise RuntimeError(f"anchor not found in {target}")
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied MambaSpec page_size_padded unify fix")


def main() -> None:
    patch_mamba_abstract()
    patch_kv_cache_utils()
    patch_engine_core()
    patch_gpu_model_runner()
    patch_unify_page_size()


if __name__ == "__main__":
    main()
