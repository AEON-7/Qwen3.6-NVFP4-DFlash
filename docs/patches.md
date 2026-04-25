# Patches applied in this image

The v1.2 image bakes in **5 vLLM source patches** + **1 build flag** + **1 dependency upgrade** + **1 env var** = 8 modifications total. Each is a targeted fix for a real issue running Qwen3.6 on sm_120/sm_121a (DGX Spark), but not every item is still required on newer vLLM/FlashInfer bases.

All patches live in [`patches/`](../patches/) (Python scripts that modify the installed vLLM dist-package files at image build time, idempotently).

> **v2 weights note:** v1 of these weights (text-only key layout) required patch #1
> below to be active in the registry. **v2 weights** (the production
> `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` published 2026-04-19) load via vLLM's
> canonical multimodal `Qwen3_5MoeForConditionalGeneration` class with no
> registry override needed. Patch #1 is still applied (idempotent, harmless) for
> backward-compat with anyone still on v1 weights.

> **Current-base note (2026-04-25):** newer vLLM snapshots around and after
> mid-April 2026 include more native Qwen3.5/Qwen3.6 hybrid handling. Treat
> patches #1, #2, and #3 as legacy/backport fixes unless your image reproduces
> the exact failure described in that section. The v2 GHCR image keeps the
> production behavior but does not require operators to re-apply these manually.

---

## 1. `register_qwen3_5_text.py` — register text-only Qwen3.6 classes (v1-only, harmless on v2)

**Problem:** vLLM HEAD's `_TEXT_GENERATION_MODELS` registry doesn't include `Qwen3_5ForCausalLM` or `Qwen3_5MoeForCausalLM` (the text-only Qwen3.6 classes). Only the multimodal `Qwen3_5MoeForConditionalGeneration` is registered.

**v1 use:** v1 weights had the `language_model.` key prefix stripped to match the text-only class, so the registry entry was mandatory.

**v2 use:** v2 weights preserve the multimodal layout, so vLLM picks `Qwen3_5MoeForConditionalGeneration` natively. The text-only registry entry is unused but kept for backward-compat (idempotent — re-applying does nothing).

**Current status:** legacy for multimodal v2 weights and newer vLLM bases that already register/load the canonical multimodal class. Keep it only if you are serving the old text-layout checkpoint.

**Fix:** insert the two text-only entries into `_TEXT_GENERATION_MODELS` after the existing `Qwen3MoeForCausalLM` entry.

**Upstream:** PRs [#36289](https://github.com/vllm-project/vllm/pull/36289), [#36607](https://github.com/vllm-project/vllm/pull/36607), [#36850](https://github.com/vllm-project/vllm/pull/36850) all closed unmerged. Open PR [#39476](https://github.com/vllm-project/vllm/pull/39476) only adds `IsHybrid` mixin, not the registry entries.

**Code:** [`patches/register_qwen3_5_text.py`](../patches/register_qwen3_5_text.py)

---

## 2. `patch_cuda_optional_import.py` — RTLD_LAZY for `_C_stable_libtorch`

**Problem:** vLLM HEAD's `_C_stable_libtorch.abi3.so` references SM100-only kernels (`mxfp4_experts_quant`, `silu_and_mul_mxfp4_experts_quant`) used by gpt-oss MXFP4 MoE. These kernels are **not built for sm_120**, leaving undefined symbols. Default `dlopen` is `RTLD_NOW` which resolves all symbols at load → fails with `ImportError: undefined symbol`. Since `vllm/platforms/cuda.py` does an unconditional import at init time, this cascades into all of vLLM being unusable.

**Fix:** wrap the `_C_stable_libtorch` import in an `RTLD_LAZY | RTLD_GLOBAL` dlopen. The .so loads, all the symbols we DO need (e.g., `cutlass_scaled_mm_supports_fp8`) register cleanly, and the missing MXFP4 symbols stay unresolved harmlessly (we never call them on sm_120).

**Current status:** legacy/backport for images whose `_C_stable_libtorch` import fails. Some newer 0.20.0-dev GB10 builds export the needed symbols or include stubs, so default `RTLD_NOW` import succeeds and this patch is unnecessary there.

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

**Current status:** legacy/backport for bases that do not derive `mamba_block_size` early. Newer vLLM commits add Qwen3.5/Qwen3.6 model files plus `validate_mamba_block_size` / platform derivation before the `min()` and `cdiv()` sites execute, so they may have the buggy-looking code but never pass `None` into it.

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

## 8. `patch_cudagraph_align.py` — CUDA graph capture-size alignment (SM121 stability)

**Problem:** vLLM's `compilation.py:1378` only applies the spec-decode capture-size alignment filter when the decode graph mode is `FULL`. Pure `PIECEWISE` mode silently skips it, so capture sizes can contain non-multiples of `(1 + num_speculative_tokens)`. With DFlash k=15 and default capture sizes `[1, 2, 4, 8, 16, 24, 32, 40, ...]`, partial-acceptance decode steps can land on graph slots that do not divide evenly, triggering `cudaErrorIllegalAddress` mid-decode on SM121.

**Fix:** remove the FULL-only gate so PIECEWISE mode also gets aligned capture sizes. Without this patch, users would need to pass `--compilation-config '{"cudagraph_capture_sizes":[16,32,48,...]}'` manually as a workaround.

**Mode caveat:** default spec-decode deployments commonly run `FULL_AND_PIECEWISE`; in that mode the decode path still captures FULL graphs, so long-soak reports have not reproduced this issue. This patch is primarily for operators forcing pure `PIECEWISE`.

This patch + the post-2026-04-19 DFlash drafter together eliminate the need for `--enforce-eager`, restoring ~30% throughput.

**Code:** [`patches/patch_cudagraph_align.py`](../patches/patch_cudagraph_align.py)

---

## (No longer applied) Safetensors prefix strip — superseded by v2 weights

**v1 only — not used in the v1.2 production image.** v1 of the published weights had `model.language_model.layers.X.*` keys rewritten to `model.layers.X.*` so they'd load via the text-only `Qwen3_5MoeForCausalLM` class. The rewrite codepath turned out to be unstable in vLLM's loader at scale.

v2 weights (re-quantized 2026-04-19 with `AutoModelForImageTextToText`) preserve the canonical multimodal key layout and load natively via `Qwen3_5MoeForConditionalGeneration`. **No prefix-strip is performed at any stage.**

If you have a v1 checkpoint locally, the simplest fix is: delete and re-pull `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (the same repo URL — v2 commits replaced v1).

---

## Build-time modifications (not patches but required)

| # | Item | What it does |
|---:|---|---|
| A | `TORCH_CUDA_ARCH_LIST="12.0+PTX"` | Single-arch sm_120 build with PTX → driver JITs to sm_121a on Spark |
| B | `flashinfer-python>=0.6.8` | sm_120 NVFP4 KV-cache decode kernels (PRs #2520, #2702) |
| C | `VLLM_TEST_FORCE_FP8_MARLIN` | v1/v1.2 baked `=1` as a defensive pin for older MoE/grouped NVFP4 backend selection. Current v2 images bake `=0`; FlashInfer CUTLASS NVFP4 linear GEMM is validated on GB10, and `VLLM_USE_FLASHINFER_MOE_FP4=0` prevents the unsupported MoE FP4 auto-probe path. |

---

## Why so many patches?

vLLM HEAD on Qwen3.6 + DFlash + sm_121a hits multiple work-in-progress edges:
- Hybrid attention KV cache assumes uniform block_size (no MambaSpec accommodation)
- M-RoPE methods unimplemented for either Qwen3.6 class
- gpt-oss MXFP4 kernels referenced in `_C_stable_libtorch.abi3.so` undefined on sm_120
- Older CUTLASS/grouped NVFP4 backend selection needed a Marlin guard on some bases; v2 validates the FlashInfer CUTLASS linear path with Marlin forcing disabled
- CUDA graph capture-size alignment gated to FULL mode, breaking pure PIECEWISE+spec-decode
- Text-only registry classes unregistered upstream (legacy v1 path; harmless on v2)

These patches collapse all of that into a single `docker compose up -d`. Each will be dropped as upstream PRs land — see each patch's docstring for the issue/PR tracking.

---

## Patch ordering

Patches apply in this order (encoded in the [Dockerfile](../Dockerfile)):

1. Source-clone vLLM HEAD
2. Build vLLM (`uv pip install --no-build-isolation .`)
3. Upgrade to flashinfer 0.6.8
4. Apply `register_qwen3_5_text.py`
5. Apply `patch_cuda_optional_import.py`
6. Apply `patch_kv_cache_utils.py` (covers 4 sites in one pass)
7. Apply `patch_mrope_text_fallback.py`
8. Apply `patch_cudagraph_align.py`
9. Verification step (must pass; image is unusable otherwise)
