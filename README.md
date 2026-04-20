# Qwen3.6-35B-A3B-heretic NVFP4 + DFlash on DGX Spark

[![Image](https://img.shields.io/badge/ghcr.io-aeon--7%2Fvllm--spark--omni--q36-blue)](https://ghcr.io/aeon-7/vllm-spark-omni-q36)
[![Model](https://img.shields.io/badge/HuggingFace-AEON--7%2FQwen3.6--35B--A3B--heretic--NVFP4-yellow)](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)
[![Drafter](https://img.shields.io/badge/Drafter-z--lab%2FQwen3.6--35B--A3B--DFlash-orange)](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

A production-stable deployment of **[`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)** with **[DFlash](https://github.com/z-lab/dflash)** speculative decoding on **NVIDIA DGX Spark** (GB10 / sm_121a).

> ⚠️ **READ THE REQUIREMENTS SECTION FIRST.** This image and its weights are tuned specifically for the DGX Spark (GB10 / sm_120-121 Blackwell) with PyTorch nightly cu130. It will NOT work on Hopper, Ampere, B200, or other Blackwell variants without rebuilding.

| | |
|---|---|
| **Model** | `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (~22 GB, multimodal preserved) |
| **Drafter** | `z-lab/Qwen3.6-35B-A3B-DFlash` (~905 MB, gated — request access on HF) |
| **Hardware** | DGX Spark (NVIDIA GB10, 128 GB unified memory, sm_121a) |
| **Image** | `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` (~9 GB compressed) |

---

## Headline performance (measured)

DGX Spark, greedy decoding (T=0), 512-token outputs, full 256K context budget:

| Concurrency | Aggregate tok/s | Per-request tok/s | TTFT |
|---:|---:|---:|---:|
| 1   | 116.8  | 116.8 | 72 ms |
| 4   | 218.3  | 54.6 | 146 ms |
| 16  | 410.1  | 25.6 | 217 ms |
| 64  | 578.4  | 9.0 | 589 ms |
| **128** | **785.3** | 6.1 | 801 ms |

DFlash spec-decode acceptance: **62-78% position-0**, **2.7-4.4 mean accepted tokens per target step**.

Stress-tested with 22K-token prompts + multi-hour soak: **zero crashes**.

---

## ⚠️ Hard Requirements (read FIRST)

### Hardware (mandatory — image is purpose-built for this only)
| Component | Required | Notes |
|---|---|---|
| GPU | **NVIDIA GB10** (DGX Spark only) | sm_120 / sm_121a Blackwell. Other GPUs WILL NOT WORK with the published image. |
| Unified memory | 128 GB | Spark default |
| Disk | 35 GB free | Image (~22 GB) + weights (~22 GB) + drafter (~1 GB) + headroom |

**Image will NOT work on:**
- H100/H200 (sm_90 — Hopper)
- A100/A40 (sm_80 — Ampere)
- B200/GB200 (sm_100 — different Blackwell variant; rebuild from source)
- L40S/RTX 4090/RTX PRO 6000 (sm_89/sm_120 desktop variants — see `docs/build.md`)

### Software (mandatory)
| Component | Version | Notes |
|---|---|---|
| NVIDIA driver | ≥ **580.x** | `nvidia-smi` should print "NVIDIA GB10" |
| Docker | ≥ 25.x | with `nvidia-container-toolkit` |
| OS | Ubuntu 24.04 LTS confirmed | other Linux distros likely fine |

### Hugging Face access (mandatory for drafter)
The DFlash drafter `z-lab/Qwen3.6-35B-A3B-DFlash` is a **gated** HF repo:
1. Visit https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
2. Click **"Request access"** — usually granted within hours
3. Run `huggingface-cli login` or set `HF_TOKEN` env var

> ⚠️ If you cloned the drafter before **2026-04-19**, you MUST re-pull. The earlier
> drafter had a long-context bug that caused `cudaErrorIllegalAddress` crashes
> after ~16K tokens. The fixed version is now on HF.

---

## Quick start (5 commands)

```bash
# 1. Pre-flight check — confirm anonymous pull works
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2

# 2. Pull both models into the canonical layout
sudo mkdir -p /opt/qwen36 && sudo chown $USER:$USER /opt/qwen36
cd /opt/qwen36
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir ./qwen36-nvfp4 &
hf download z-lab/Qwen3.6-35B-A3B-DFlash         --local-dir ./qwen36-dflash &
wait

# 3. Get the compose file
curl -fsSL https://raw.githubusercontent.com/AEON-7/Qwen3.6-NVFP4-DFlash/main/examples/docker-compose.yml \
  -o docker-compose.yml

# 4. Start the server (3-5 min to first "Application startup complete")
docker compose up -d
docker compose logs -f

# 5. Smoke test (use temperature=0 for greedy → max DFlash speedup)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-fast",
    "messages": [{"role":"user","content":"What is 17 × 23?"}],
    "max_tokens": 2048,
    "temperature": 0
  }'
```

If `max_tokens` < ~1500 your response may show `content: null` with `finish_reason: "length"` — that's the model hitting max-tokens during reasoning, not a crash. See [docs/troubleshooting.md](docs/troubleshooting.md). Use ≥ 2048 for thinking-enabled requests.

For the full step-by-step (with pre-flight + post-deploy verification), see [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md).

---

## What this image actually is

vLLM HEAD source-built for **CUDA 13.0 / sm_120 + PTX** (DGX Spark / GB10 / sm_121a) with the following **8 patches baked in** — every one solves a real upstream bug we hit during deployment:

| # | Patch | What it fixes |
|---:|---|---|
| 1 | `register_qwen3_5_text.py` | Adds text-only `Qwen3_5MoeForCausalLM` to vLLM model registry. Upstream PRs (#36289, #36607, #36850) closed unmerged. **Not strictly required for v2 multimodal weights but harmless.** |
| 2 | `patch_cuda_optional_import.py` | Wraps `import vllm._C_stable_libtorch` in `RTLD_LAZY` dlopen. The .so depends on SM100-only MXFP4 kernels (`mxfp4_experts_quant`) that don't exist on sm_120. |
| 3 | `patch_kv_cache_utils.py` (×4 sites) | Mamba/linear-attention groups have `block_size=None`. Multiple downstream sites in vLLM HEAD do `block_size * X` arithmetic on this. Defaults to `cache_config.block_size or 16` at MambaSpec creation, plus None-safe guards at min()/cdiv() sites. |
| 4 | `patch_mrope_text_fallback.py` | Qwen3.6 declares M-RoPE in config but no model class implements `get_mrope_input_positions` in vLLM HEAD. Adds inline fallback for the canonical text-only positions (T=H=W=arange). |
| 5 | `patch_cudagraph_align.py` | `compilation.py:1378` only applies the spec-decode cudagraph capture-size alignment for `cudagraph_mode=FULL`; PIECEWISE silently skipped it. Result: capture sizes contained non-multiples of (1+spec_tokens), causing `cudaErrorIllegalAddress` on partial-acceptance decode steps. Patch removes the FULL-only gate. |
| 6 | ENV `VLLM_TEST_FORCE_FP8_MARLIN=1` | Forces Marlin GEMM. FlashInfer's CUTLASS NVFP4 path is broken on SM121 (101 KB SMEM vs 228 KB on SM100; autotuner SMEM-overflows on every tile shape larger than 128×128×64B). |
| 7 | ENV `TORCH_CUDA_ARCH_LIST="12.0+PTX"` | Build target for sm_120, runtime JITs to sm_121a on Spark. |
| 8 | flashinfer 0.6.8 | sm_120 NVFP4 KV-cache decode kernels (PRs #2520, #2702). |

All patches live in [`patches/`](patches/) and run automatically at image build time (idempotent). The [`Dockerfile`](Dockerfile) is reproducible — see [`docs/build.md`](docs/build.md).

---

## What changed in v2 (this release)

Previous v1 weights had `language_model.` prefix stripped from safetensors keys to match a text-only model class — required vLLM registry + key-rename patches and was unstable in production (intermittent `cudaErrorIllegalAddress` crashes during real chat sessions).

v2 (current) re-quantized from `tvall43/Qwen3.6-35B-A3B-heretic` directly with `AutoModelForImageTextToText`, preserving:
- Full multimodal architecture (`Qwen3_5MoeForConditionalGeneration`)
- 27-block ViT vision encoder (BF16, NVFP4-skipped)
- Original `model.language_model.layers.X.*` key layout — vLLM's multimodal class loads natively, no prefix-strip patch needed
- 30 linear_attention (Mamba/GDN, fp32) + 10 full_attention layers
- 256 routed experts × 8 active + 1 shared expert per layer
- All 122,880 per-expert NVFP4 keys (every expert calibrated)

vLLM serves it via the canonical multimodal class — fewer code paths in the inference hot loop, much better stability under load. Travis ran multiple live chat sessions (Celina) without a single crash where v1 was crashing on virtually every interaction.

---

## OpenClaw integration

The compose serves **3 model aliases** for the same backend:
- `qwen36-35b-heretic` — canonical name
- `qwen36-fast` — intended for greedy/agentic workloads (T=0 → 78% DFlash acceptance, 117 tok/s single-stream)
- `qwen36-deep` — intended for sampled/creative workloads (T=0.7 → low DFlash acceptance, ~50 tok/s; tradeoff for diversity)

OpenClaw config (validated against actual zod schemas) is in [`docs/openclaw.md`](docs/openclaw.md). The pattern: register two model entries pointing to the same backend with different default `params.temperature`. Route per agent or per channel binding.

---

## Documentation map

| Doc | Audience |
|---|---|
| [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md) | **Primary deployment guide** — start here for full Spark setup |
| [`docs/openclaw.md`](docs/openclaw.md) | OpenClaw gateway integration (validated against real zod schemas) |
| [`docs/dflash.md`](docs/dflash.md) | DFlash speculative decoding tuning + monitoring |
| [`docs/dtree.md`](docs/dtree.md) | Future-work — slot DTree in when z-lab releases |
| [`docs/quantization.md`](docs/quantization.md) | Recreating the NVFP4 quantization end-to-end (including v2 recipe) |
| [`docs/build.md`](docs/build.md) | Building the image yourself instead of pulling from GHCR |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Symptoms → root causes → fixes |
| [`docs/patches.md`](docs/patches.md) | Each patch explained, with the upstream issues they address |

---

## Credits

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **DFlash** — [z-lab/dflash](https://github.com/z-lab/dflash) (Soroush Mohri et al.)
- **Qwen3.6-35B-A3B-heretic base** — [tvall43/Qwen3.6-35B-A3B-heretic](https://huggingface.co/tvall43/Qwen3.6-35B-A3B-heretic) (`heretic v1.2.0` abliteration of unsloth/Qwen3.6-35B-A3B)
- **Qwen3.6** — [Qwen team](https://github.com/QwenLM)
- **llmcompressor** — [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
- **FlashInfer** — [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **rmagur1203/vllm-dgx-spark** — independent 4-day SM121 investigation that surfaced the Marlin requirement
- **OpenClaw** — [openclaw/openclaw](https://github.com/openclaw/openclaw) (Peter Steinberger / @steipete)

## License

Apache 2.0 (matching upstream vLLM, FlashInfer, llmcompressor).
The base model carries its own license — see [`tvall43/Qwen3.6-35B-A3B-heretic`](https://huggingface.co/tvall43/Qwen3.6-35B-A3B-heretic).
