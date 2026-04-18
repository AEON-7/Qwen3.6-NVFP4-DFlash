# Qwen3.6-35B-A3B-heretic NVFP4 + DFlash on DGX Spark

[![Image](https://img.shields.io/badge/ghcr.io-aeon--7%2Fvllm--spark--omni--q36-blue)](https://ghcr.io/aeon-7/vllm-spark-omni-q36)
[![Model](https://img.shields.io/badge/HuggingFace-AEON--7%2FQwen3.6--35B--A3B--heretic--NVFP4-yellow)](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)
[![Drafter](https://img.shields.io/badge/Drafter-z--lab%2FQwen3.6--35B--A3B--DFlash-orange)](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

A turn-key deployment of **[`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)** with **[DFlash](https://github.com/z-lab/dflash)** speculative decoding on **NVIDIA DGX Spark** (GB10 / sm_121a).

> ⚠️ **READ FIRST:** This image and the patches it carries are specifically built for the **DGX Spark / GB10 / sm_120-121** hardware running **PyTorch nightly cu130**. It will NOT work on Hopper (H100, sm_90), Ampere (A100, sm_80), or B200 (sm_100). It is also NOT compatible with the official vLLM nightly wheels (which are cu128). See [Requirements](#requirements) before pulling.

| | |
|---|---|
| **Target model** | `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (~21 GB) |
| **Drafter** | `z-lab/Qwen3.6-35B-A3B-DFlash` (905 MB, gated) |
| **Hardware** | DGX Spark (NVIDIA GB10, 128 GB unified memory, sm_121a) |
| **Stack** | vLLM HEAD source-built for cu130/sm_120 + FlashInfer 0.6.8 + 7 patches |
| **Image** | `ghcr.io/aeon-7/vllm-spark-omni-q36:v1` (~9 GB compressed) |

---

## Benchmark headline numbers (measured)

DGX Spark, Qwen3.6-35B-A3B-heretic NVFP4 + DFlash, **stable production config** (Marlin + 16-aligned CUDA graphs + DFlash), 256K context, greedy decoding:

| Test | Throughput |
|---|---|
| Single 4096 max_tokens | **53.2 tok/s** |
| Single 8192 max_tokens (full content emitted) | **77.8 tok/s** |
| 4-concurrent × 4096 | **47-49 tok/s/req** (~190 tok/s aggregate) |

**DFlash speculative decoding** (greedy):
- Position-0 acceptance: **78.5%**
- Mean accepted length: **4.21 tokens**
- Stress-tested 12+ min with no crashes

> **Earlier benchmark numbers (91 tok/s single, 729 tok/s @64) were from an earlier
> config with default CUDA graph capture sizes — that config crashes intermittently
> with `cudaErrorIllegalAddress` mid-decode after 5-15 minutes of serving on SM121.
> Use the production config in `examples/docker-compose.yml` instead. See
> [docs/troubleshooting.md](docs/troubleshooting.md) for the root cause.**

---

## Requirements

### Hardware
- **GPU**: NVIDIA GB10 (DGX Spark) — sm_120 / sm_121a Blackwell. **No other GPU is supported by this image.** Other Blackwell variants (B200=sm_100, GB200=sm_100a) require rebuilding from source.
- **Unified memory**: 128 GB (the Spark default)
- **Disk**: 35 GB free (image + model + drafter + working space)
- **Network**: HF requires ~22 GB of model downloads on first run

### Software
- **NVIDIA driver**: ≥ 580.x (CUDA 13.2 toolkit baked into image)
- **Docker**: ≥ 25.x with `nvidia-container-toolkit` (must support GB10)
- **OS**: Ubuntu 24.04 LTS confirmed; other Linux distros likely fine
- **Python on host**: 3.10+ if running OpenClaw or test scripts directly; not needed if you only use Docker

### Hugging Face access
- HF account with **access granted** to the gated drafter `z-lab/Qwen3.6-35B-A3B-DFlash`
  - Request access at: https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
  - Usually granted within hours
- `HF_TOKEN` env var or `huggingface-cli login` set up

### Optional — for OpenClaw integration
- OpenClaw installed: see https://github.com/openclaw/openclaw

---

## Quick start (5 commands)

> **Pre-flight:** confirm `docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1` succeeds anonymously before continuing. If it returns "unauthorized", the package visibility is set to private — see [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md#pre-flight-check-do-this-first) for what to do. Same applies to the gated DFlash drafter on HF.

```bash
# 1. Pull the image (~9 GB compressed)
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1

# 2. Pull both models into the canonical layout (/opt/qwen36/...)
sudo mkdir -p /opt/qwen36 && sudo chown $USER:$USER /opt/qwen36
cd /opt/qwen36
export HF_HUB_ENABLE_HF_TRANSFER=1   # 5x faster downloads
hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir ./qwen36-nvfp4 &
hf download z-lab/Qwen3.6-35B-A3B-DFlash         --local-dir ./qwen36-dflash &
wait

# 3. Get the compose file
curl -fsSL https://raw.githubusercontent.com/AEON-7/Qwen3.6-NVFP4-DFlash/main/examples/docker-compose.yml \
  -o docker-compose.yml

# 4. Start the server (compose expects /opt/qwen36/qwen36-{nvfp4,dflash}; override
#    with QWEN36_MODELS_DIR=/your/path docker compose up -d if your layout differs)
docker compose up -d
docker compose logs -f   # ~3 min to first "Application startup complete"

# 5. Smoke test
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-fast","messages":[{"role":"user","content":"What is 17 × 23?"}],"temperature":0,"max_tokens":256}'
```

For the full step-by-step guide with pre-flight checks and troubleshooting, see [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md).

---

## What's in the image

This image had to be source-built specifically because **no off-the-shelf vLLM image works on DGX Spark with this model**:

| Layer | Why |
|---|---|
| **vLLM @ HEAD source-built (cu130/sm_120)** | Official nightlies are cu128 → `libcudart.so.12 not found`. Source build is the only path for sm_121a. |
| **FlashInfer 0.6.8** | sm_120 NVFP4 KV-cache decode kernels (PRs [#2520](https://github.com/flashinfer-ai/flashinfer/pull/2520), [#2702](https://github.com/flashinfer-ai/flashinfer/pull/2702)) |
| **DFlash proposer** | Baked into vLLM `v1/spec_decode/dflash.py` — DFlash is upstream now, no plugin install. |
| **7 patches** | See [`patches/`](patches/) for full list. Required for: text-only `Qwen3_5MoeForCausalLM` registry, MXFP4 SM100 symbol bypass (RTLD_LAZY), Mamba block_size None handling (4 sites), M-RoPE text-only fallback (T=H=W=arange). |

**Skipped (not yet available):**
- **DTree** — z-lab has not published the DTree extension yet. See [`docs/dtree.md`](docs/dtree.md) for the theoretical integration recipe.
- **End-to-end NVFP4 KV cache** — vLLM PR [#40177](https://github.com/vllm-project/vllm/pull/40177) and FlashInfer PR [#3097](https://github.com/flashinfer-ai/flashinfer/pull/3097) both still open. The image *includes* the FlashInfer 0.6.8 decode kernels; flip on `--kv-cache-dtype nvfp4` once both PRs land.

---

## Documentation map

| Doc | Audience |
|---|---|
| [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md) | **Primary deployment guide** — start here for full Spark setup |
| [`docs/openclaw.md`](docs/openclaw.md) | OpenClaw gateway integration (validated against zod schema) |
| [`docs/dflash.md`](docs/dflash.md) | DFlash speculative decoding tuning + monitoring |
| [`docs/dtree.md`](docs/dtree.md) | Future-work — slot DTree in when z-lab releases |
| [`docs/quantization.md`](docs/quantization.md) | Recreating the NVFP4 quantization end-to-end |
| [`docs/build.md`](docs/build.md) | Building the image yourself instead of pulling from GHCR |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Symptoms → root causes → fixes |
| [`docs/patches.md`](docs/patches.md) | Each patch explained, with the upstream issues they address |

---

## OpenClaw integration (fast/deep modes)

The compose exposes **3 model names** that all point to the same backend:
- `qwen36-35b-heretic` — canonical name (use whichever sampling you set per request)
- `qwen36-fast` — intended for greedy/agentic workloads (T=0 → 80% DFlash acceptance)
- `qwen36-deep` — intended for sampled/creative workloads (T=0.7 → low DFlash acceptance)

OpenClaw config (validated against the zod schema) is in [`docs/openclaw.md`](docs/openclaw.md). The pattern: register two model entries with different default `params.temperature`, route per agent/binding.

---

## Credits

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **DFlash** — [z-lab/dflash](https://github.com/z-lab/dflash) (Soroush Mohri et al.)
- **Qwen3.6-35B-A3B-heretic base** — [tvall43/Qwen3.6-35B-A3B-heretic](https://huggingface.co/tvall43/Qwen3.6-35B-A3B-heretic)
- **Qwen3.6** — [Qwen team](https://github.com/QwenLM)
- **llmcompressor** — [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
- **FlashInfer** — [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **OpenClaw** — [openclaw/openclaw](https://github.com/openclaw/openclaw) (Peter Steinberger / @steipete)

## License

Apache 2.0 (matching upstream vLLM, FlashInfer, llmcompressor).
The base model carries its own license — see [`tvall43/Qwen3.6-35B-A3B-heretic`](https://huggingface.co/tvall43/Qwen3.6-35B-A3B-heretic).
