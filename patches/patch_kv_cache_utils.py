#!/usr/bin/env python3
"""Patch vLLM hybrid-attention KV-cache None-handling in two files.

Background:
  Qwen3.6 has hybrid attention: 30 linear_attention layers (mamba-style state,
  no KV block) + 10 full_attention layers (standard KV block). vLLM HEAD calls
  `min(block_size for group in groups)` in two places, but linear_attention
  groups (and added padding groups) have block_size=None. This crashes with
  `TypeError: '<' not supported between NoneType and NoneType`.

  Fix: filter None values before min(), default to 1 if all None.

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


def main() -> None:
    patch_mamba_abstract()
    patch_kv_cache_utils()
    patch_engine_core()
    patch_gpu_model_runner()


if __name__ == "__main__":
    main()
