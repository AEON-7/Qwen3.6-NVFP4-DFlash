# DGX Spark deployment guide — Qwen3.6 NVFP4 + DFlash

End-to-end production setup. Follow each step in order. Total time from clean Spark to first inference: **~30 min** (mostly model downloads).

---

## ⚠️ Requirements (read FIRST)

This image is **hardware-specific** and **NOT a general vLLM build**. If you skip this section you will likely waste hours debugging.

### Hardware (mandatory)
| Component | Required |
|---|---|
| GPU | **NVIDIA GB10** (DGX Spark only). sm_120/sm_121a Blackwell. |
| Unified memory | 128 GB (Spark default) |
| Disk | 35 GB free |

### NVFP4 backend selection on SM121

The backend guidance is version-specific:

- `v1.2` and older: `VLLM_TEST_FORCE_FP8_MARLIN=1` was baked in as a defensive guard for older NVFP4 backend selection on GB10. Leave it alone unless you are intentionally auditing kernels.
- `v2` and newer: the recommended production default is `VLLM_TEST_FORCE_FP8_MARLIN=0` plus `VLLM_USE_FLASHINFER_MOE_FP4=0`. This keeps the validated FlashInfer CUTLASS NVFP4 linear GEMM path active and avoids the unsupported MoE FP4 auto-probe path.

Healthy v2 boot logs should include:

```text
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
```

Older community guidance to force Marlin came from real SM121 failures on some
FlashInfer/vLLM shapes, but current GB10 builds with FlashInfer 0.6.8+ and the
v2 patch set can run CUTLASS cleanly. If your image has run long soaks with
`FlashInferCutlassNvFp4LinearKernel` and no CUDA errors, do not force Marlin
just because the v1.2 docs did.

**This image will NOT work on:**
- H100 / H200 (sm_90 — Hopper)
- A100 / A40 (sm_80 — Ampere)
- B200 / GB200 (sm_100 — different Blackwell variant; rebuild from source)
- L40S / L40 / L4 (sm_89 — Ada)
- RTX 4090 / 5090 (sm_89/sm_120, but driver/RAM not validated)

If you need it on different hardware, see [`build.md`](build.md) — the Dockerfile is reusable with a different `TORCH_CUDA_ARCH_LIST`.

### Software (mandatory)
| Component | Version | Notes |
|---|---|---|
| NVIDIA driver | ≥ **580.x** | `nvidia-smi` should print "NVIDIA GB10" |
| CUDA toolkit | not needed on host | baked into image |
| Docker | ≥ 25.x | with `nvidia-container-toolkit` |
| OS | Ubuntu 24.04 LTS | other distros likely OK |

### DFlash drafter (no auth required)
The DFlash drafter `z-lab/Qwen3.6-35B-A3B-DFlash` is now a **public** HF repo —
no token, no access request, just pull it directly.

> ⚠️ **CRITICAL — re-pull required if cloned before 2026-04-19.** The earlier
> z-lab DFlash drafter had a long-context bug that triggered
> `cudaErrorIllegalAddress` after ~16K tokens. The fixed version is on HF as of
> 2026-04-19. If your `qwen36-dflash/` directory pre-dates that, delete it and
> re-pull. Without the fix you must add `--enforce-eager` to the vLLM command,
> losing ~30% throughput.

### Optional — for OpenClaw integration
- OpenClaw installed: https://github.com/openclaw/openclaw

---

## Pre-flight check (do this FIRST)

Run these 4 commands to confirm everything you need is reachable. If any fails, stop and fix it before proceeding:

```bash
# 1. GPU + driver
nvidia-smi | grep -E 'GB10|Driver'   # expect "NVIDIA GB10" + driver 580+

# 2. Docker GPU passthrough
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi | head -3

# 3. GHCR image is pullable (anonymous)
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
# If this returns "unauthorized" or "not found":
#   - The image visibility is set to private. Either:
#     a) Wait for it to be flipped to public (file an issue at github.com/AEON-7/Qwen3.6-NVFP4-DFlash/issues), OR
#     b) If you're AEON-7: flip at github.com/users/AEON-7/packages/container/vllm-spark-omni-q36/settings, OR
#     c) Build from source per docs/build.md (note: also requires a public base image)

# 4. DFlash drafter is publicly pullable — no auth needed
hf download z-lab/Qwen3.6-35B-A3B-DFlash README.md --local-dir /tmp/dflash-test
# Should succeed without HF_TOKEN. Repo was un-gated 2026-04-21.
```

All 4 pass → proceed. Any fails → fix that one first.

---

## Step 1 — Verify host hardware + software

```bash
# Driver + GPU detection
nvidia-smi
# Expected: "NVIDIA GB10", driver 580.x or newer

# Unified memory
free -g
# Expected: total ~121 GB (Spark's 128 GB minus reserved)

# Disk space
df -h /
# Need at least 35 GB free

# Docker + nvidia-container-toolkit
docker info | grep -i nvidia
# Should show "nvidia" in default-runtime or runtimes list
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
# Should print the same nvidia-smi output as on the host
```

If any of the above fails, **do not proceed**. Fix it first. Common issues:
- Driver too old → upgrade via NVIDIA's apt repo
- `nvidia-container-toolkit` not installed → `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker`
- Driver doesn't recognize GB10 → driver < 580.x

---

## Step 2 — Set up working directory

```bash
sudo mkdir -p /opt/qwen36 && sudo chown $USER:$USER /opt/qwen36
cd /opt/qwen36
```

You'll end up with:
```
/opt/qwen36/
├── docker-compose.yml      # the compose
├── qwen36-nvfp4/           # 21 GB — NVFP4 model
└── qwen36-dflash/          # 905 MB — DFlash drafter
```

---

## Step 3 — Pull the Docker image

```bash
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
```

~9 GB compressed, ~22 GB uncompressed. First pull is 3-5 min on a typical home connection.

> **Why `v1.2`?** v1.0/v1.1 shipped with weights stripped of the `language_model.` key prefix
> for a text-only model class. That layout was unstable in production (intermittent NaN/crash
> in the prefix-strip codepath). v1.2 image is paired with the v2 multimodal weights
> (`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`, re-quantized 2026-04-19) which load via vLLM's
> canonical `Qwen3_5MoeForConditionalGeneration` class and run rock-solid under load.

Verify:
```bash
docker images vllm-spark-omni-q36 --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'
```

---

## Step 4 — Install the HF CLI (no token required)

```bash
# Install hf CLI if missing
pip install --upgrade --user "huggingface_hub[hf_transfer]"
```

Both weight repos (`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` and `z-lab/Qwen3.6-35B-A3B-DFlash`) are **public anonymous-pull** — no `HF_TOKEN`, no access request. If you already have `HF_TOKEN` set for other work it won't hurt, but it's not required.

---

## Step 5 — Pull the models

```bash
cd /opt/qwen36

export HF_HUB_ENABLE_HF_TRANSFER=1   # 5x faster downloads on Spark

# Pull both in parallel — ~10-15 min on typical home connection
hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir ./qwen36-nvfp4 &
hf download z-lab/Qwen3.6-35B-A3B-DFlash         --local-dir ./qwen36-dflash &
wait
```

Verify:
```bash
# Main model — ~22 GB across 9 sharded safetensors files
du -sh qwen36-nvfp4
ls qwen36-nvfp4/model-*.safetensors | wc -l   # expect 9

# Drafter — ~905 MB, single safetensors file
ls -lh qwen36-dflash/model.safetensors

# Sanity check — confirm v2 multimodal layout (key prefix should be `model.language_model.`)
python3 -c "
from safetensors import safe_open
with safe_open('qwen36-nvfp4/model-00001-of-00009.safetensors', framework='pt') as f:
    keys = [k for k in f.keys() if 'experts.0.down_proj' in k]
    print('Sample expert key:', keys[0] if keys else 'NONE FOUND')
    assert any('language_model' in k for k in f.keys()), 'WRONG LAYOUT — re-pull AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4'
    print('OK — v2 multimodal layout confirmed')
"
```

---

## Step 6 — Get the compose file

```bash
curl -fsSL \
  https://raw.githubusercontent.com/AEON-7/Qwen3.6-NVFP4-DFlash/main/examples/docker-compose.yml \
  -o /opt/qwen36/docker-compose.yml
```

Or copy [`examples/docker-compose.yml`](../examples/docker-compose.yml) from this repo. The default is tuned for DGX Spark:

| Flag | Value | Why |
|---|---|---|
| `--quantization compressed-tensors` | required | NVFP4 packed weights are compressed-tensors `nvfp4-pack-quantized` format |
| `--max-model-len` | 262144 | Full 256K context (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` env required) |
| `--max-num-seqs` | 128 | Concurrent request cap (validated at 128-concurrency without OOM) |
| `--max-num-batched-tokens` | 65536 | Chunked prefill chunk size; 65536 gives best aggregate throughput |
| `--gpu-memory-utilization` | 0.85 | Conservative for 128 GB unified memory (leaves ~18 GB for OS + dflash + KV) |
| `--attention-backend flash_attn` | required | DFlash needs FlashAttention backend |
| `--speculative-config` | `{"method":"dflash","model":"/models/qwen36-dflash","num_speculative_tokens":15}` | DFlash with k=15 (sweet spot for Qwen3.6 acceptance curve) |
| `--reasoning-parser qwen3` | | Surfaces `<think>...</think>` content as `reasoning_content` |
| `--tool-call-parser qwen3_coder` | | OpenAI-format tool calls for agentic workloads |
| `--served-model-name` | `qwen36-35b-heretic qwen36-fast qwen36-deep` | 3 aliases for one backend (mode routing) |
| `--enable-chunked-prefill --enable-prefix-caching` | | Standard production flags |
| `--load-format safetensors` | | Skip auto-detect (saves ~2 s startup) |
| `--trust-remote-code` | | Required: Qwen3.6 ships custom modeling code |
| `--enable-auto-tool-choice` | | Honor `"tool_choice":"auto"` from OpenAI clients |

> **Note: `--enforce-eager` is NOT required** with the v1.2 image + the post-2026-04-19
> DFlash drafter. Earlier writeups recommended it as a workaround for two separate bugs
> (drafter long-context crash + cudagraph capture-size misalignment). Both are now fixed:
> the drafter on HF, and the alignment via the v1.2 image's [`patch_cudagraph_align.py`](../patches/patch_cudagraph_align.py).
> Running with cudagraphs enabled gives ~30% throughput over eager mode.

If you want to tune for higher concurrency at lower context:
```yaml
--max-model-len 32768       # 32K context
--max-num-seqs 256          # 256 concurrent
```

---

## Step 7 — Start the server

```bash
cd /opt/qwen36
docker compose up -d
docker compose logs -f
```

Expected boot timeline:
| Stage | Time | Log marker |
|---|---|---|
| Args parsed | ~5 s | `non-default args:` |
| Model loaded | ~125 s | `Model loading took 20.65 GiB memory and ... seconds` |
| KV cache profiled | ~85 s | `GPU KV cache size: ... tokens` |
| CUDA graphs captured | ~70 s | `init engine ... took ... s (compilation: ...)` |
| Server ready | total ~3-5 min | `Application startup complete` ← **wait for this** |

Subsequent restarts (with image + weights cached) are ~3 min.

---

## Step 8 — Smoke test

```bash
# 1. Health
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}

# 2. Confirm 3 model names registered
curl -s http://localhost:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
# Expected:
# qwen36-35b-heretic
# qwen36-fast
# qwen36-deep

# 3. Chat completion (greedy / fast mode)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-fast",
    "messages": [{"role":"user","content":"Compute 17 × 23. Show your work."}],
    "max_tokens": 256,
    "temperature": 0
  }' | python3 -m json.tool
```

The response should contain a `reasoning` field (chain-of-thought) and a `content` field (final answer). Reasoning may also be in `reasoning_content` depending on vLLM build.

---

## Step 9 — Verify DFlash is firing

```bash
# Generate a few requests
for i in 1 2 3; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen36-fast","messages":[{"role":"user","content":"What is 47 × 89? Show work."}],"max_tokens":256,"temperature":0}' > /dev/null
done

# Check spec_decode metrics
docker logs vllm-qwen36-heretic 2>&1 | grep "Per-position" | tail -2
```

Healthy DFlash with greedy decoding should show:
```
Mean acceptance length: 4-5 (out of 16)
Per-position acceptance rate: 0.78, 0.52, 0.43, 0.41, 0.37, ...
Avg Draft acceptance rate: 20-30%
```

If acceptance is < 5%, you're sending requests with `temperature` ≥ 0.5 (which is the wrong workload for DFlash). Use `temperature=0` for agentic workloads.

---

## Step 10 — Run the benchmark

```bash
# Install openai python client if missing
pip install --user openai

# Pull the bench script
curl -fsSL \
  https://raw.githubusercontent.com/AEON-7/Qwen3.6-NVFP4-DFlash/main/scripts/bench_concurrency.py \
  -o /tmp/bench.py

# Run
python3 /tmp/bench.py --max-tokens 256 --runs 1 --levels "1,4,16,64,128"
```

Expected (v2 weights + v1.2 image + cudagraphs, greedy T=0, 512-token outputs):

| Concurrency | Aggregate tok/s | Per-req tok/s | TTFT |
|---:|---:|---:|---:|
| 1   | 116.8  | 116.8 | 72 ms |
| 4   | 218.3  | 54.6  | 146 ms |
| 16  | 410.1  | 25.6  | 217 ms |
| 64  | 578.4  | 9.0   | 589 ms |
| 128 | **785.3** | 6.1 | 801 ms |

If you see substantially lower numbers (< 90 tok/s single-stream or < 600 tok/s at 128
concurrency), jump to [`troubleshooting.md`](troubleshooting.md). The most common
culprit by a wide margin is `--enforce-eager` left in the compose from a v1 deployment.

---

## Step 11 — Production hardening

### Run as a systemd service (recommended)

```bash
sudo tee /etc/systemd/system/vllm-qwen36.service <<'EOF'
[Unit]
Description=vLLM Qwen3.6 NVFP4 + DFlash
Requires=docker.service
After=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/qwen36
ExecStart=/usr/bin/docker compose -f /opt/qwen36/docker-compose.yml up
ExecStop=/usr/bin/docker compose -f /opt/qwen36/docker-compose.yml down
Restart=on-failure
RestartSec=10
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen36
sudo systemctl status vllm-qwen36
```

### Reverse proxy (optional)

If exposing externally, put behind nginx/caddy/traefik with:
- TLS termination
- Rate limiting per source IP
- API key validation (vLLM's `--api-key` flag if you want auth at backend)
- **`proxy_buffering off`** for nginx (else SSE streaming chunks pile up)

### Monitor

```bash
# Prometheus metrics endpoint
curl -s http://localhost:8000/metrics | grep -E '^vllm_'

# Container resource use
docker stats vllm-qwen36-heretic
```

Key metrics to watch:
- `vllm_spec_decode_acceptance_rate` — should be ≥ 0.6 for greedy workloads
- `vllm_request_decode_time_seconds_bucket` — p99 latency
- `vllm_gpu_cache_usage_perc` — should not be pinned at 1.0 sustained

---

## Step 12 — Updating

```bash
# Pin to a specific tag — DON'T use :latest in production (image bumps may
# require weight re-pull, e.g., the v1.x → v2 transition)
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
cd /opt/qwen36
docker compose up -d --force-recreate
```

When upgrading across major image versions, **always re-read the release notes** for
weight format changes. The v1.0/v1.1 → v1.2 cutover required re-pulling
`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` weights — same repo URL, but the v2 commit
preserved multimodal architecture instead of stripping the `language_model.` key prefix.

---

## Step 13 — Tear down

```bash
cd /opt/qwen36
docker compose down
docker rmi ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2   # optional
sudo rm -rf /opt/qwen36                                # also removes weights
sudo systemctl disable --now vllm-qwen36 2>/dev/null
sudo rm /etc/systemd/system/vllm-qwen36.service 2>/dev/null
```

---

## OpenClaw integration

See [`openclaw.md`](openclaw.md) for the full validated config that exposes `qwen36-fast` (greedy/agentic) and `qwen36-deep` (sampled/creative) with proper sampling defaults per mode.

---

## Where to go next

- **Tuning DFlash speedup**: [`docs/dflash.md`](dflash.md)
- **Adding DTree** (when z-lab releases): [`docs/dtree.md`](dtree.md)
- **Re-running quantization** with different calibration: [`docs/quantization.md`](quantization.md)
- **Building the image from source**: [`docs/build.md`](build.md)
- **When something breaks**: [`docs/troubleshooting.md`](troubleshooting.md)
