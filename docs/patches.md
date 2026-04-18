# Patches applied in this image

The image bakes in 7 patches plus 1 source-build modification. Each is a targeted fix for a real issue in vLLM HEAD running Qwen3.6 on sm_120/sm_121a (DGX Spark).

All patches live in [`patches/`](../patches/) (Python scripts that modify the installed vLLM dist-package files at image build time, idempotently).

---

## 1. `register_qwen3_5_text.py` — register text-only Qwen3.6 classes

**Problem:** vLLM HEAD's `_TEXT_GENERATION_MODELS` registry doesn't include `Qwen3_5ForCausalLM` or `Qwen3_5MoeForCausalLM` (the text-only Qwen3.6 classes). Only the multimodal `Qwen3_5MoeForConditionalGeneration` is registered. Loading a text-only checkpoint falls through to the multimodal path and crashes on `vision_config.spatial_merge_size`.

**Fix:** insert the two text-only entries into `_TEXT_GENERATION_MODELS` after the existing `Qwen3MoeForCausalLM` entry.

**Upstream:** PRs [#36289](https://github.com/vllm-project/vllm/pull/36289), [#36607](https://github.com/vllm-project/vllm/pull/36607), [#36850](https://github.com/vllm-project/vllm/pull/36850) all closed unmerged. Open PR [#39476](https://github.com/vllm-project/vllm/pull/39476) only adds `IsHybrid` mixin, not the registry entries.

**Code:** [`patches/register_qwen3_5_text.py`](../patches/register_qwen3_5_text.py)

---

## 2. `patch_cuda_optional_import.py` — RTLD_LAZY for `_C_stable_libtorch`

**Problem:** vLLM HEAD's `_C_stable_libtorch.abi3.so` references SM100-only kernels (`mxfp4_experts_quant`, `silu_and_mul_mxfp4_experts_quant`) used by gpt-oss MXFP4 MoE. These kernels are **not built for sm_120**, leaving undefined symbols. Default `dlopen` is `RTLD_NOW` which resolves all symbols at load → fails with `ImportError: undefined symbol`. Since `vllm/platforms/cuda.py` does an unconditional import at init time, this cascades into all of vLLM being unusable.

**Fix:** wrap the `_C_stable_libtorch` import in an `RTLD_LAZY | RTLD_GLOBAL` dlopen. The .so loads, all the symbols we DO need (e.g., `cutlass_scaled_mm_supports_fp8`) register cleanly, and the missing MXFP4 symbols stay unresolved harmlessly (we never call them on sm_120).

**Code:** [`patches/patch_cuda_optional_import.py`](../patches/patch_cuda_optional_import.py)

---

## 3-6. `patch_kv_cache_utils.py` — Mamba block_size None handling (4 sites)

**Problem:** Qwen3.6 has hybrid attention (30 linear_attention layers + 10 full_attention layers). vLLM HEAD's MambaSpec creation reads `vllm_config.cache_config.mamba_block_size` which is None on Spark, and many downstream sites do `block_size * X`, `X % block_size`, or `min(block_sizes)` — all crash on None.

**Fix:** four interlocking patches:

| Site | What it does |
|---|---|
| `model_executor/layers/mamba/abstract.py` (root cause) | Default `mamba_block_size = vllm_config.cache_config.block_size or 16` when None |
| `v1/core/kv_cache_utils.py:_report_kv_cache_config` | Filter None before `min(block_sizes)` |
| `v1/engine/core.py:_initialize_kv_caches` | Filter None before `min(g.kv_cache_spec.block_size for g in groups)` |
| `v1/worker/gpu_model_runner.py:may_reinitialize_input_batch` | Skip `cdiv(max_model_len, block_size * world_size)` for MambaSpec groups |

The root-cause fix at `mamba/abstract.py` makes most downstream sites work; the other 3 are defense-in-depth in case future code paths hit them too.

**Code:** [`patches/patch_kv_cache_utils.py`](../patches/patch_kv_cache_utils.py)

---

## 7. `patch_mrope_text_fallback.py` — M-RoPE text-only fallback

**Problem:** Qwen3.6 declares M-RoPE in its config (`mrope_interleaved=True`, `mrope_section=[11,11,10]`). vLLM HEAD's `_init_mrope_positions` calls `model.get_mrope_input_positions(prompt_token_ids, mm_features)` on the model class. **Neither `Qwen3_5MoeForCausalLM` nor `Qwen3_5MoeForConditionalGeneration` implements this method in vLLM HEAD as of 2026-04-18.** Hard assertion fail.

**Fix:** add a text-only fallback to `_init_mrope_positions` that constructs the canonical text-only positions inline:
```python
positions = arange(n).unsqueeze(0).expand(3, -1)   # T = H = W = arange(n)
delta = 0
```

**Critical detail:** the canonical text-only M-RoPE positions are **`T = H = W = arange(n)`**, NOT `T=arange, H=W=0`. The latter is what naive implementations write but it produces *different* RoPE math than standard 1D RoPE, breaking DFlash drafter agreement (since the drafter is plain Qwen3 with standard 1D RoPE). With T=H=W=arange, M-RoPE math becomes bit-identical to standard 1D RoPE — drafter and target agree → high acceptance.

Source for the canonical formula: vLLM `qwen2_5_vl.py:1072-1109`, transformers `Qwen2.5-VL/get_rope_index` text-only branch (lines 1127-1138).

**Code:** [`patches/patch_mrope_text_fallback.py`](../patches/patch_mrope_text_fallback.py)

---

## 8. Safetensors prefix strip (one-time, applied to weights not vLLM)

Not a vLLM patch but documented here for completeness.

**Problem:** The source model `tvall43/Qwen3.6-35B-A3B-heretic` was structured as multimodal Qwen3.6, so llmcompressor saved keys as `model.language_model.layers.X.*` (3 levels). But our config declares text-only architecture (`Qwen3_5MoeForCausalLM`), and vLLM's text-only loader expects `model.layers.X.*` (2 levels). Mismatch → `KeyError` on every parameter.

**Fix:** one-shot rewrite of `model.safetensors` stripping the `language_model.` segment from every key. Done by [`scripts/strip_language_model_prefix.py`](../patches/strip_language_model_prefix.py) (also in `qwen36-omni-build/`). The corrected weights are what's published at `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`.

---

## Why so many patches?

vLLM HEAD's text-only Qwen3.6 (`qwen3_5_moe_text` / `Qwen3_5MoeForCausalLM`) is **work-in-progress**. The architecture exists in the codebase but isn't fully wired:
- Registry entries missing
- Hybrid attention KV cache assumes uniform block_size
- M-RoPE methods unimplemented
- Reference RedHatAI checkpoints use the *multimodal* arch (which is more wired) but ours is text-only

These patches make text-only Qwen3.6 actually load and serve. The work will mostly be obviated as upstream PRs land — at which point we'd drop the corresponding patches. See each patch's docstring for the upstream PR/issue tracking.

---

## Patch ordering

Patches must apply in this order (encoded in the [Dockerfile](../Dockerfile)):

1. Source-clone vLLM
2. Build vLLM (`uv pip install --no-build-isolation .`)
3. Apply `register_qwen3_5_text.py`
4. Apply `patch_cuda_optional_import.py`
5. Apply `patch_kv_cache_utils.py` (covers 4 sites in one pass)
6. Apply `patch_mrope_text_fallback.py`
7. Verification step
