# Troubleshooting

Symptoms ↔ root causes ↔ fixes for the running deployment.

> 🕰️ **v1 → v2 history (read once, then forget).** v1.0/v1.1 of this image
> shipped with weights that had the `language_model.` prefix stripped from
> safetensors keys to match a text-only `Qwen3_5MoeForCausalLM` registry hack.
> That layout was unstable in production (intermittent `cudaErrorIllegalAddress`
> mid-decode, NaNs from the prefix-strip codepath). v1.2 image + the v2 weights
> at `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (re-quantized 2026-04-19) preserve
> the canonical multimodal layout (`Qwen3_5MoeForConditionalGeneration`) and run
> rock-solid under sustained chat load. **If you're hitting weird key errors or
> registry-class confusion, your weights are probably stale — re-pull.**

## Boot-time failures

### `ImportError: libcudart.so.12: cannot open shared object file`

You've pulled the **wrong image** (a cu128 official nightly). DGX Spark ships cu130 PyTorch.

```bash
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2   # this one is cu130
```

### `KeyError: 'experts.w2_input_global_scale'` (or similar prefix-strip key error)

You're running v2 weights through a v1.x image, OR v1 weights (no `language_model.` prefix)
through the v1.2 image. The two layouts are not interchangeable.

Verify weights:
```bash
python3 -c "
from safetensors import safe_open
with safe_open('/opt/qwen36/qwen36-nvfp4/model-00001-of-00009.safetensors','pt') as f:
    has_lm_prefix = any('language_model' in k for k in f.keys())
    print('multimodal (v2):' if has_lm_prefix else 'text-only (v1):', 'OK' if has_lm_prefix else 'STALE')
"
```

If `STALE`, re-pull from HF — the v2 commit on the same repo URL replaces v1:
```bash
rm -rf /opt/qwen36/qwen36-nvfp4
hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir /opt/qwen36/qwen36-nvfp4
```

### `Resolved architecture: Qwen3_5MoeForConditionalGeneration`

**Under v1.2 image + v2 weights this is the EXPECTED message — not an error.** The image
no longer needs to register text-only `Qwen3_5MoeForCausalLM`. The multimodal class loads
the v2 weights natively and runs in text-only mode (no image inputs in chat) with no extra
patches in the hot path.

If you're running v1.0 or v1.1 image with v1 (prefix-stripped) weights and seeing this,
your registry patch didn't apply. Either upgrade to v1.2 + v2 weights (recommended), or
inside the v1.x container run:
```bash
docker exec -it vllm-qwen36-heretic python3 /opt/patches/register_qwen3_5_text.py
docker restart vllm-qwen36-heretic
```

### `ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported']`

You set `--kv-cache-dtype fp8` but DFlash requires `--attention-backend flash_attn`, and that backend in this vLLM build doesn't accept fp8 KV.

Pick one:

```yaml
# Option A: drop KV quant (uses bf16 KV — costs 2× memory but works with DFlash)
--attention-backend flash_attn
# (no --kv-cache-dtype line)

# Option B: use FP8 KV but disable DFlash
--attention-backend flashinfer
--kv-cache-dtype fp8
# (no --speculative-config line)
```

If you need both fp8 KV *and* DFlash, watch vLLM PR [#40177](https://github.com/vllm-project/vllm/pull/40177) — when it merges, this image will be rebuilt with NVFP4 KV cache support that *does* work with `flash_attn` + DFlash.

### `RuntimeError: CUDA error: out of memory` during weight load

Three causes:

1. **Another container is using the GPU.** Spark has 1 GPU; only one large model can be loaded:
   ```bash
   docker ps --format 'table {{.Names}}\t{{.Status}}'
   docker stop <other-container>
   ```
2. **`gpu_memory_utilization` too high for KV math.** Try 0.80 instead of 0.85:
   ```yaml
   --gpu-memory-utilization 0.80
   ```
3. **`max-num-seqs × max-model-len` exceeds KV budget.** See the memory math in [`dgx-spark-setup.md`](dgx-spark-setup.md#5-compose-file). Each Qwen3.6 sequence at 256K bf16 needs ~5 GB KV. With 86 GB available, that's 16 sequences max.

## Runtime failures

### `RuntimeError: CUDA error: an illegal instruction was encountered`

Tracked by vLLM issue [#39761](https://github.com/vllm-project/vllm/issues/39761) — sm_120 NVFP4 decode kernel has a CUDA-graph-capture bug.

The v1.2 image bakes the `VLLM_TEST_FORCE_FP8_MARLIN=1` env var as default, which avoids
the broken CUTLASS NVFP4 GEMM and resolves this on most workloads. Verify it's set:

```bash
docker exec vllm-qwen36-heretic env | grep MARLIN
# expect: VLLM_TEST_FORCE_FP8_MARLIN=1
```

If you still see this error with Marlin forced, **last-resort workaround**: add `--enforce-eager`
to the compose:

```yaml
command:
  - bash
  - -c
  - |
    exec vllm serve /models/qwen36 \
      ... \
      --enforce-eager
```

Cost: ~30% throughput vs. cudagraph-enabled. File a bug — with v1.2 + v2 weights + recent
DFlash drafter, this should not occur.

### `cudaErrorIllegalAddress` / "illegal memory access" mid-decode

**Symptom:** Server boots fine, serves requests for 5-15 min, then crashes mid-decode with:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress'
```
Container auto-restarts (if you have `restart: unless-stopped`). Crashes recur every few minutes.

**Root cause clarification (updated 2026-04-21 after empirical re-test):**

Two separate things were historically conflated under "CUTLASS broken on SM121":

1. **CUTLASS NVFP4 *linear* GEMM** (q/k/v/o, mlp projections) — **works fine on SM121.** vLLM picks `FlashInferCutlassNvFp4LinearKernel` automatically; autotunes 17 fp4_gemm profiles at boot; native FP4 tensor cores fire. **No env var needed.** (Confirm: `docker logs vllm-qwen36-heretic | grep "FlashInferCutlassNvFp4LinearKernel"`.)

2. **CUTLASS NVFP4 *grouped* GEMM** (MoE experts) — broken-ish for our shape. We tested all 5 non-Marlin candidates on 2026-04-21 with `VLLM_USE_FLASHINFER_MOE_FP4=1` then again without it: **every backend** (`FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`, `FLASHINFER_CUTEDSL_BATCHED`, `FLASHINFER_CUTLASS`, `VLLM_CUTLASS`) rejected our 256-expert × 512-intermediate × NVFP4 config in their `is_supported_config()` checks. This is a **kernel shape-alignment limitation**, not an SM121-specific hardware bug — same wall the supergemma4 deploy hit with a different (704) intermediate dim. Marlin (weight-only decompress to BF16) is the only supported backend for this MoE shape until kernels are widened. The original SMEM-overflow finding from [rmagur1203/vllm-dgx-spark TLDR](https://github.com/rmagur1203/vllm-dgx-spark/blob/main/TLDR.md) is a separate, real issue affecting some shapes on SM121 — not all.

**Fix:** TWO independent fixes both required:

**1. Use Marlin for the MoE GEMM** (the only supported NVFP4 MoE backend for our shape):
```yaml
# in compose, environment (belt-and-suspenders):
- VLLM_TEST_FORCE_FP8_MARLIN=1
```
The omni image bakes this in. Verify:
```bash
docker exec vllm-qwen36-heretic env | grep MARLIN
# expect: VLLM_TEST_FORCE_FP8_MARLIN=1
```

> **Note (2026-04-21):** This env var is **redundant in the current vLLM build** — auto-selection already arrives at MARLIN since all 5 other backends reject our MoE shape. Kept as defensive in case future vLLM versions add a half-broken backend that auto-selector picks. The linear path is unaffected and uses CUTLASS NVFP4 natively — see the "Root cause clarification" above.

**2. Re-pull the latest DFlash drafter from z-lab**:

The crash that had us chasing CUDA-graph and capture-size theories was actually a **drafter bug at long context (>16K)** that z-lab fixed in an update on 2026-04-19. If your local drafter copy predates that:

```bash
hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir ./qwen36-dflash --force-download
```

With the fresh drafter, `--enforce-eager` is **no longer needed** and CUDA graphs work fine, giving back ~42% throughput.

**Fallback if you can't re-pull**: add `--enforce-eager` to disable CUDA graphs. ~25-30% throughput cost but bypasses the long-context drafter bug. Performance with eager fallback: ~54-67 tok/s single-stream long-form, ~165 tok/s aggregate at 4-concurrent, ~115 tok/s short single-stream.

> **History:** the v1.2 image bakes a separate fix to `vllm/config/compilation.py:1378` that aligns CUDA graph capture sizes to multiples of `(1+num_speculative_tokens)` — also real, also necessary, but not the actual cause of this specific crash. Both fixes are needed for the production config.

### "Agent couldn't generate a response" / `content: null` with `finish_reason: "length"`

**By far the most common gotcha.** Qwen3.6 with thinking enabled (which is the default
for this image — `--reasoning-parser qwen3` + chat template's `enable_thinking=true`)
spends **most of its `max_tokens` budget on reasoning** before emitting the final answer.
With small `max_tokens`:

```
max_tokens=256:  256 reasoning tokens, 0 content tokens, finish=length, content=null  ← fails
max_tokens=512:  ~500 reasoning tokens, 0 content, finish=length                       ← still fails
max_tokens=2048: ~1100 reasoning tokens, ~20 content tokens, finish=stop               ← OK
```

OpenClaw / pi-ai sees `content: null` and shows "Agent couldn't generate a response."

**Fix — three options:**

1. **Bump `max_tokens` to ≥ 2048 for thinking workloads** (most chat clients default to 1024 or lower):
   ```bash
   curl ... -d '{"max_tokens": 4096, ...}'
   ```
   For OpenClaw, the validated config in `docs/openclaw.md` sets `maxTokens: 32768`, which is fine. But if a client overrides `max_tokens` lower in the request, they hit this.

2. **Disable thinking for chat workloads** — add a per-request override:
   ```bash
   curl ... -d '{"chat_template_kwargs": {"enable_thinking": false}, ...}'
   ```
   Lower-quality reasoning but immediate `content`. Good for casual chat where you don't need
   visible chain-of-thought.

3. **Set the server-side default to thinking-off** if your primary use is chat:
   ```yaml
   # in docker-compose.yml command:
   --default-chat-template-kwargs '{"enable_thinking": false}'
   ```
   Then thinking only fires when the client explicitly requests it.

### Empty `reasoning_content` in OpenAI responses

Two possibilities:

1. **You requested a chat that doesn't trigger reasoning.** Qwen3.6 reasoning-content channeling is enabled per request via the chat template — verify you're using the latest `tokenizer_config.json` from the model repo.
2. **vLLM bug #38855** — Gemma4-style channel tokens stripped before parser sees them. Affects Gemma4, not Qwen3.6. Should not apply here, but if you see it: confirm `--reasoning-parser qwen3` (not `gemma4`).

### Tool-call output leaks raw markup

Confirm you're using the right parser for the model:

```yaml
--tool-call-parser qwen3_coder    # Qwen3 tool format
--enable-auto-tool-choice
```

If you mounted a host parser file over the image's, restore by removing the mount.

### Acceptance rate looks low (< 60%)

See [`dflash.md`](dflash.md#acceptance-rate-suspiciously-low--50). Most common cause: temperature mismatch between target and drafter.

### `IndexError: index out of bounds` during decode

Tokenizer mismatch between target and drafter. Verify:

```bash
docker exec vllm-qwen36-heretic python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/models/qwen36')
d = AutoTokenizer.from_pretrained('/models/qwen36-dflash')
print('target vocab:', t.vocab_size)
print('drafter vocab:', d.vocab_size)
"
```

If they differ, re-pull both from HF:

```bash
hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir /opt/qwen36/qwen36-nvfp4
hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir /opt/qwen36/qwen36-dflash
```

### Drafter HTTP 401 / 403 on first boot

**As of 2026-04-21 the drafter repo is public** — anonymous pull works. If you're seeing
401/403 anyway, the most likely causes are:

1. **HF outage / temporary block** — try `curl -I https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/resolve/main/config.json` directly; if that 5xx-s, wait and retry.
2. **Local stale auth** — your `HF_TOKEN` env var is set to an invalid/expired token that's overriding anonymous access. Try `unset HF_TOKEN && hf download ...`.
3. **Corporate proxy / DNS** — `huggingface.co` resolves but the response is being mangled. Test from a different network.

vLLM doesn't need `HF_TOKEN` at serve time regardless — it only reads from local `/models/qwen36-dflash`.

## Performance issues

### Throughput much lower than expected (< 50 tok/s)

Checklist:

```bash
# 1. Confirm CUDA graphs are enabled (unless workaround for #39761 is active)
docker logs vllm-qwen36-heretic 2>&1 | grep -E "Capturing.*graph|enforce_eager"
# Expected: "Capturing CUDA graph for ..."
# Bad: "enforce_eager=True" (you have the workaround on — disable it if #39761 is fixed)

# 2. Confirm DFlash is firing
docker logs vllm-qwen36-heretic 2>&1 | grep -i "dflash\|spec_decode\|accepted"
# Expected: "DFlashProposer" + "accepted_tokens" counters

# 3. Confirm flash_attn is the backend
docker logs vllm-qwen36-heretic 2>&1 | grep -i "attention.*backend"
# Expected: "FLASH_ATTN"

# 4. Confirm warmup completed
docker logs vllm-qwen36-heretic 2>&1 | grep -i "Warmup\|graph capture"
```

### High latency on first request after idle

CUDA graph re-capture or kv cache pre-allocation. Send a warmup request after server boot:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-heretic","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
```

### Memory growing over time / OOM after many requests

Likely a KV cache fragmentation issue. The image sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` which usually mitigates this. If you still see growth, file an issue with:

```bash
docker exec vllm-qwen36-heretic nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

over time.

## Networking / API issues

### `connection refused` on port 8000

```bash
# Container running?
docker ps --filter name=vllm-qwen36

# Listening on host port?
ss -tlnp | grep 8000

# Host firewall?
sudo ufw status
```

The compose uses `network_mode: host` so the server binds directly to the host's port 8000.

### Slow streaming (chunks arrive in batches)

You're behind a buffering proxy (nginx, traefik). Disable buffering for SSE:

```nginx
location /v1/ {
  proxy_pass http://localhost:8000;
  proxy_buffering off;
  proxy_cache off;
  proxy_set_header Connection '';
  chunked_transfer_encoding off;
  proxy_http_version 1.1;
}
```

## Filing an issue

If none of the above applies, file at https://github.com/aeon-7/Qwen3.6-NVFP4-DFlash/issues with:

```bash
# Run this and paste the output
{
  echo '=== image ==='
  docker inspect vllm-qwen36-heretic --format '{{.Config.Image}}'
  echo
  echo '=== gpu ==='
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo
  echo '=== versions ==='
  docker exec vllm-qwen36-heretic python3 -c "import vllm, flashinfer, torch; print('vllm', vllm.__version__); print('flashinfer', flashinfer.__version__); print('torch', torch.__version__, torch.version.cuda)"
  echo
  echo '=== last 50 log lines ==='
  docker logs vllm-qwen36-heretic --tail 50
}
```
