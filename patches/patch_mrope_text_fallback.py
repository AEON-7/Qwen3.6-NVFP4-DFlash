#!/usr/bin/env python3
"""Add text-only fallback for M-RoPE positions in vLLM gpu_model_runner.

Background:
  Qwen3.6 declares M-RoPE in its config (mrope_interleaved=True, mrope_section=[11,11,10]).
  vLLM HEAD's `_init_mrope_positions` requires the model class to implement the
  SupportsMRoPE protocol (with `get_mrope_input_positions` method). But neither
  Qwen3_5MoeForCausalLM nor Qwen3_5MoeForConditionalGeneration implements it
  in vLLM HEAD as of 2026-04-18.

  Canonical text-only M-RoPE positions (per transformers' Qwen2.5-VL get_rope_index
  text-only branch and vLLM's qwen2_5_vl.py:1072-1109):
    mrope_positions = broadcast(arange(n), (3, n))  # T = H = W = arange(n)
    mrope_position_delta = 0

  When T=H=W=arange and mrope_section sums to head_dim/2, M-RoPE math becomes
  bit-identical to standard 1D RoPE. Critically, the DFlash drafter is
  model_type=qwen3 with standard 1D RoPE — so for spec decode acceptance
  to be high, the target's M-RoPE outputs MUST match standard RoPE, which
  only happens when T=H=W (not T=arange, H=W=0 as a naive impl might assume).

Idempotent — safe to run multiple times.
"""
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
)

src = TARGET.read_text()

if "# mrope_text_fallback" in src:
    print(f"[{TARGET.name}] already applied")
    sys.exit(0)

OLD = (
    "    def _init_mrope_positions(self, req_state: CachedRequestState):\n"
    "        model = self.get_model()\n"
    "        assert supports_mrope(model), \"M-RoPE support is not implemented.\"\n"
    "        assert req_state.prompt_token_ids is not None, (\n"
    "            \"M-RoPE requires prompt_token_ids to be available.\"\n"
    "        )\n"
    "        mrope_model = cast(SupportsMRoPE, model)\n"
    "\n"
    "        req_state.mrope_positions, req_state.mrope_position_delta = (\n"
    "            mrope_model.get_mrope_input_positions(\n"
    "                req_state.prompt_token_ids,\n"
    "                req_state.mm_features,\n"
    "            )\n"
    "        )"
)

NEW = (
    "    def _init_mrope_positions(self, req_state: CachedRequestState):\n"
    "        # mrope_text_fallback — Qwen3.6 / qwen3_5_moe_text doesn't implement\n"
    "        # SupportsMRoPE in vLLM HEAD. For text-only (no mm_features), the\n"
    "        # canonical M-RoPE positions are T=H=W=arange(n) which makes the rotary\n"
    "        # math bit-identical to standard 1D RoPE. delta = 0.\n"
    "        # See vLLM qwen2_5_vl.py:1072-1109 + transformers Qwen2.5-VL get_rope_index.\n"
    "        model = self.get_model()\n"
    "        assert req_state.prompt_token_ids is not None, (\n"
    "            \"M-RoPE requires prompt_token_ids to be available.\"\n"
    "        )\n"
    "        if supports_mrope(model):\n"
    "            mrope_model = cast(SupportsMRoPE, model)\n"
    "            req_state.mrope_positions, req_state.mrope_position_delta = (\n"
    "                mrope_model.get_mrope_input_positions(\n"
    "                    req_state.prompt_token_ids,\n"
    "                    req_state.mm_features,\n"
    "                )\n"
    "            )\n"
    "        else:\n"
    "            # Text-only canonical: T = H = W = arange(n), delta = 0\n"
    "            # Critical: NOT T=arange,H=W=0 (which produces different RoPE math\n"
    "            # and breaks DFlash drafter agreement, since drafter uses 1D RoPE).\n"
    "            import torch as _torch\n"
    "            n_tokens = len(req_state.prompt_token_ids)\n"
    "            arange = _torch.arange(n_tokens, dtype=_torch.long)\n"
    "            positions = arange.unsqueeze(0).expand(3, -1).contiguous()\n"
    "            req_state.mrope_positions = positions\n"
    "            req_state.mrope_position_delta = 0"
)

if OLD not in src:
    raise RuntimeError(
        f"Anchor not found in {TARGET}.\n"
        "vLLM gpu_model_runner.py layout may have changed — re-inspect with:\n"
        f"  grep -n '_init_mrope_positions' {TARGET}"
    )

new_src = src.replace(OLD, NEW, 1)
TARGET.write_text(new_src)
print(f"[{TARGET.name}] applied M-RoPE text-only fallback")
