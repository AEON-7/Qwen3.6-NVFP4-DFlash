# DFlash speculative decoding — practical guide

DFlash is a **block-diffusion-based draft** speculative decoding method from [z-lab](https://github.com/z-lab/dflash) (Soroush Mohri et al., 2026). Unlike token-by-token EAGLE / Medusa drafters, DFlash drafts a **block of tokens in parallel via a small diffusion model** that conditions on the target's hidden states.

## Why DFlash > EAGLE on Qwen3.6

| Property | EAGLE-3 | DFlash |
|---|---|---|
| Drafter call cost | ~10% target latency, sequential | ~5% target latency, single block call |
| Acceptance rate (math/code) | 60-70% | **80-90%** |
| Acceptance rate (chat/agentic) | 50-60% | **70-80%** |
| Drafter parameters | ~500 MB | 905 MB |
| Architecture coupling | Tight (depends on target's MLP+attn weights) | Loose (only needs hidden states) |
| Training method | KD on next-token | KD via block diffusion across draft length |

The published numbers for `Qwen3.6-35B-A3B-DFlash` on GB10:

| Benchmark | No-spec tok/s | DFlash tok/s | Speedup |
|---|---|---|---|
| GSM8K (math) | 22 | 95 | 4.3× |
| HumanEval (code) | 25 | 110 | 4.4× |
| MT-Bench (chat) | 28 | 86 | 3.1× |
| ShareGPT (agentic) | 30 | 82 | 2.7× |

Your speedups will vary by `num_speculative_tokens` and acceptance rate.

## How vLLM wires it up

DFlash is **upstream in vLLM** (no plugin install). The relevant code lives at:

- `vllm/v1/spec_decode/dflash.py` — `DFlashProposer` class
- `vllm/model_executor/models/qwen3_dflash.py` — drafter model class (`DFlashDraftModel`)

You activate it via `--speculative-config`:

```yaml
--speculative-config '{"method":"dflash","model":"/path/to/drafter","num_speculative_tokens":15}'
--attention-backend flash_attn   # required — DFlash needs flash_attn
```

## Tuning `num_speculative_tokens`

The speculative depth (k) controls the trade-off between:
- **High k** → more draft work per target step, higher payoff if accepted
- **Low k** → less wasted compute on rejected drafts

| k | Best for | Acceptance ceiling |
|---|---|---|
| 5 | High-temp creative generation (low coherence) | ~95% |
| 10 | General chat | ~85% |
| **15** (default) | Math, code, agentic, structured output | ~80% |
| 20 | Long deterministic completions (e.g., JSON output) | ~70% |
| 30+ | Pure greedy decode of memorized text | ~65% |

If you're seeing acceptance rates < 60%, drop k by 5. If you're seeing > 90%, raise it.

## Monitoring acceptance rate

vLLM logs spec-decode metrics every N steps:

```
INFO ... DFlash spec_decode: drafted=15 accepted=12 acceptance=80.0% effective_throughput=4.2x
```

Or query Prometheus metrics on `:8000/metrics`:

```
vllm_spec_decode_num_drafts_total
vllm_spec_decode_num_accepted_tokens_total
vllm_spec_decode_acceptance_rate
```

The **effective throughput multiplier** ≈ 1 + (acceptance_rate × k / drafter_overhead).

## Drafter requirements

The DFlash drafter must match the target architecturally:
- Same vocab (248,320 for Qwen3.6)
- Same hidden size (2048)
- Same attention layout (the drafter conditions on layer 30 hidden states for Qwen3.6)
- DFlash version compatibility (drafter's `dflash.py` must match the proposer's expected interface)

The drafter ships with `trust_remote_code=True` modeling code — vLLM will execute that code to instantiate the drafter, so:

```yaml
--trust-remote-code   # required for DFlash drafters
```

## Failure modes

### Acceptance rate suspiciously low (< 50%)

Likely causes:
1. **Sampling temperature mismatch** — DFlash assumes target and drafter use the same temperature. If you override temperature in requests, acceptance drops. Set sampling defaults via `--override-generation-config`:
   ```yaml
   --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":64}'
   ```
2. **Drafter version drift** — pull a fresh drafter checkpoint.
3. **Tokenizer mismatch** — confirm target and drafter share `tokenizer.json` SHA.

### `IndexError: index out of bounds` during decode

Usually a vocab mismatch. Verify:

```bash
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/models/qwen36')
d = AutoTokenizer.from_pretrained('/models/qwen36-dflash')
assert t.vocab_size == d.vocab_size, f'{t.vocab_size} vs {d.vocab_size}'
print('vocab match:', t.vocab_size)
"
```

### `CUDA illegal instruction` during DFlash decode

Tracked by vLLM issue [#39761](https://github.com/vllm-project/vllm/issues/39761) for sm_120 NVFP4 decode. Workaround:

```yaml
--enforce-eager   # disables CUDA graph capture; NVFP4 decode kernel works in eager mode
```

This costs ~10-15% throughput but is stable.

### Drafter "model not found" / HTTP 401

The `z-lab/Qwen3.6-35B-A3B-DFlash` repo is **gated**. Request access first:

1. Go to https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash
2. Click "Request access" — usually granted within hours
3. Set `HF_TOKEN` env var or `huggingface-cli login`

## Disabling DFlash for A/B comparison

To benchmark with DFlash off, just remove the `--speculative-config` flag and restart. Everything else stays the same.

```yaml
# DFlash OFF
exec vllm serve /models/qwen36 \
  --served-model-name qwen36-35b-heretic \
  ... \
  # --speculative-config ... (removed)
  --attention-backend flash_attn
```

## Further reading

- [z-lab/dflash on GitHub](https://github.com/z-lab/dflash) — original implementation + paper
- [vLLM v1 spec decode docs](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- [`docs/dtree.md`](dtree.md) — DTree extension (not yet released)
