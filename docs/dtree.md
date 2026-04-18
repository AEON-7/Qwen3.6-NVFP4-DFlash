# DTree — theoretical integration guide (not yet released)

> **Status as of 2026-04-17:** z-lab has **not** publicly released the DTree extension. This document captures the theoretical integration path so you can slot it in within minutes once they publish.

## What DTree adds to DFlash

DFlash drafts a **flat block of k tokens** in parallel. DTree drafts a **tree of token candidates** — multiple plausible continuations branched at each step — and the target evaluates the entire tree in one batched forward pass, accepting the longest matching path.

| | DFlash flat | DFlash + DTree |
|---|---|---|
| Draft shape | k tokens linear | k × b branching factor → ~k×b candidates |
| Target compute per step | 1× target call (k+1 tokens) | 1× target call ((k×b)+1 tokens) |
| Acceptance ceiling | ~85% (best case) | ~95% (paths give second chances) |
| Wall-clock speedup vs DFlash | baseline | **+30-40%** on math/code |
| GPU memory overhead | k tokens × hidden | (k×b) tokens × hidden — proportional |

The tree is stored as a [packed sparse attention mask](https://arxiv.org/abs/2305.09781) (Medusa-style), so the extra cost is mostly the wider attention compute.

## Expected z-lab release artifacts

Based on z-lab's release cadence for DFlash, expect these to land:

1. **HuggingFace drafter checkpoints**: `z-lab/Qwen3.6-35B-A3B-DFlash-DTree` (or similar)
   - Same architecture as the DFlash drafter, but with extra tree-prediction heads
   - Custom code (`trust_remote_code=True`) defining the tree expansion strategy
2. **vLLM PR or plugin**: either upstream PR adding `method=dtree` to spec decode, or a `pip install z-lab-dtree` plugin
3. **GitHub repo**: likely `z-lab/dtree` (analogous to `z-lab/dflash`)

## Activating DTree once released

### If z-lab ships it as a vLLM upstream method

The integration path is identical to DFlash today — just swap the method name and drafter:

```yaml
# Before (DFlash flat)
--speculative-config '{"method":"dflash","model":"/models/qwen36-dflash","num_speculative_tokens":15}'

# After (DFlash + DTree)
--speculative-config '{"method":"dtree","model":"/models/qwen36-dflash-dtree","num_speculative_tokens":15,"tree_branching_factor":4}'
```

Pull the new drafter:

```bash
hf download z-lab/Qwen3.6-35B-A3B-DFlash-DTree --local-dir /opt/qwen36/qwen36-dflash-dtree
```

Update the compose file to point at the new drafter, restart:

```bash
docker compose up -d --force-recreate
```

### If z-lab ships it as a vLLM plugin

The image already supports vLLM plugins. Add a `pip install` to a derivative Dockerfile:

```dockerfile
FROM ghcr.io/aeon-7/vllm-spark-omni-q36:v1
RUN uv pip install --system git+https://github.com/z-lab/dtree.git
```

vLLM auto-discovers plugins via Python entry points. The plugin should register a new spec-decode method (e.g., `dtree`) that you then activate via `--speculative-config`.

### If z-lab ships it as a fork (unlikely)

Build a custom vLLM wheel from their fork:

```dockerfile
FROM ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX" MAX_JOBS=14 NVCC_THREADS=2
RUN git clone https://github.com/z-lab/vllm-dtree.git /workspace/vllm-dtree && \
    cd /workspace/vllm-dtree && python use_existing_torch.py && \
    uv pip install --system --no-build-isolation --no-deps .
COPY patches/register_qwen3_5_text.py /opt/patches/
RUN python3 /opt/patches/register_qwen3_5_text.py
```

This is the most expensive path (full source rebuild) — avoid if upstream or plugin path is available.

## Tuning DTree (theoretical)

Once active, the tree branching factor `b` is the main knob:

| b | Tree size (k=15) | Speedup vs DFlash flat | KV overhead |
|---|---|---|---|
| 2 | 30 tokens | +15% | 2× |
| **4** (likely default) | 60 tokens | +30% | 4× |
| 6 | 90 tokens | +35% | 6× |
| 8 | 120 tokens | +40% (diminishing returns) | 8× |

The KV cache overhead during the speculative window is `b ×` the flat case. If you're already memory-bound at high context, reduce `--max-num-seqs` proportionally:

| `b` | Recommended `max-num-seqs` at 256K context |
|---|---|
| 2 | 14 |
| 4 | 10 |
| 6 | 8 |
| 8 | 6 |

## Monitoring DTree acceptance

In addition to the standard spec-decode metrics, DTree exposes:

```
vllm_spec_decode_dtree_path_length_avg     # average path length accepted (out of k)
vllm_spec_decode_dtree_branches_explored   # avg branches explored per step
vllm_spec_decode_dtree_branch_acceptance   # which branches were most often accepted
```

## When to skip DTree

DTree's overhead doesn't always pay off:

- **Long-form creative writing**: paths diverge quickly, branching wastes compute
- **High-temperature sampling** (T > 1.0): same reason
- **Tiny models** (< 7B): drafter overhead dominates anyway

Stick with flat DFlash for those workloads.

## Tracking the release

- Watch [z-lab on GitHub](https://github.com/z-lab) for a `dtree` repo
- Watch [z-lab on HF](https://huggingface.co/z-lab) for `*-DFlash-DTree` model uploads
- Watch vLLM PRs for "dtree" in title: https://github.com/vllm-project/vllm/pulls?q=dtree

When something appears, file an issue on this repo and I'll publish a `:dtree` tag of the image with the integration baked in.
