# Troubleshooting

Symptoms ↔ root causes ↔ fixes for the running deployment.

## Boot-time failures

### `ImportError: libcudart.so.12: cannot open shared object file`

You've pulled the **wrong image** (a cu128 official nightly). DGX Spark ships cu130 PyTorch.

```bash
docker pull ghcr.io/aeon-7/vllm-spark-omni-q36:v1   # this one is cu130
```

### `KeyError: 'language_model.layers.X.mlp.experts.w2_input_global_scale'`

Your vLLM is too old / missing the MoE-loader fix for text-only Qwen3_5MoeForCausalLM.

This image bakes the fix in. If you see this, you're either:
- Running an old image — pull `ghcr.io/aeon-7/vllm-spark-omni-q36:v1` again
- Running a stock vLLM image — switch to ours

### `Resolved architecture: Qwen3_5MoeForConditionalGeneration`

The registry patch didn't apply. vLLM is falling back to the multimodal class. Re-run the patch manually inside the container:

```bash
docker exec -it vllm-qwen36-heretic python3 /opt/patches/register_qwen3_5_text.py
docker restart vllm-qwen36-heretic
```

If the patch script reports "already applied" but logs still show the wrong architecture, the registry was somehow restored — check if you mounted a host directory over `/usr/local/lib/python3.12/dist-packages/vllm/`.

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

Workaround: add `--enforce-eager` to the compose:

```yaml
command:
  - bash
  - -c
  - |
    exec vllm serve /models/qwen36 \
      ... \
      --enforce-eager
```

Cost: ~10-15% throughput. The fix is being tracked upstream.

### `cudaErrorIllegalAddress` / "illegal memory access" mid-decode

**Symptom:** Server boots fine, serves requests for 5-15 min, then crashes mid-decode with:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress'
```
Container auto-restarts (if you have `restart: unless-stopped`). Crashes recur every few minutes.

**Root cause:** FlashInfer's CUTLASS NVFP4 grouped-GEMM kernel is broken on SM121. Only 101 KB SMEM per SM (vs 228 KB on SM100) — the autotuner picks tile shapes that overflow SMEM at runtime. Documented in [rmagur1203/vllm-dgx-spark TLDR](https://github.com/rmagur1203/vllm-dgx-spark/blob/main/TLDR.md): 4 days of testing 144 configs across 6 axes confirmed CUTLASS NVFP4 is unstable on SM121 without 4 custom CUTLASS header patches.

**Fix:** TWO independent fixes both required:

**1. Force the Marlin GEMM kernel** (avoids broken NVFP4 CUTLASS):
```yaml
# in compose, environment:
- VLLM_TEST_FORCE_FP8_MARLIN=1
```
The omni image bakes this in. Verify:
```bash
docker exec vllm-qwen36-heretic env | grep MARLIN
# expect: VLLM_TEST_FORCE_FP8_MARLIN=1
```

**2. Align CUDA graph capture sizes to (1+spec_tokens=16)**:
```yaml
# in compose command, add:
--compilation-config '{"cudagraph_capture_sizes":[16,32,48,64,80,96,112,128]}'
```
Root cause: `vllm/config/compilation.py:1378` only applies the spec-decode alignment filter for `cudagraph_mode=FULL`; PIECEWISE (default) skips it. Default capture sizes `[1,2,4,8,16,24,32,40,...]` contain non-multiples of 16. On partial-acceptance decode steps, vLLM dispatches to a misaligned cached graph; the kernel reads slot_mapping/positions tensors at wrong offsets → illegal-address read.

Without BOTH fixes, you'll either crash on Marlin alone (CUDA graph bug) or on CUTLASS alone (SMEM overflow) or on neither alone (still crashes after a few minutes).

Performance: with both fixes + CUDA graphs, measured **77.8 tok/s single-stream 8K, ~190 tok/s aggregate at 4-concurrent**. Without `--enforce-eager` fallback, graphs are ~30% faster than eager mode.

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

The drafter repo is gated. Request access:

1. Visit https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
2. Click "Request access" — usually granted within hours
3. Set `HF_TOKEN` and re-pull

You don't need `HF_TOKEN` at vLLM serve time if the drafter is already on disk. It's only needed during `hf download`.

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
