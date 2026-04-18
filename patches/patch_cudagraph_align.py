#!/usr/bin/env python3
"""Patch vLLM compilation.py to apply the spec-decode capture-size alignment
filter for ALL cudagraph modes (not just FULL).

Background:
  vLLM's CUDA graph capture builds graphs at specific batch sizes. When
  speculative decoding is on (num_speculative_tokens=K), every decode step
  processes (1+K) tokens and the captured graphs must be sized as multiples
  of (1+K) — otherwise on partial-acceptance steps, vLLM dispatches to a
  cached graph keyed on a misaligned size; the kernel writes
  slot_mapping/positions tensors at offsets [0..num_query_per_req-1] but the
  replayed kernel reads at offsets matching the wrong (smaller) graph.
  → cudaErrorIllegalAddress mid-decode.

  vllm/config/compilation.py has a filter that adjusts capture sizes to
  multiples of (1+K), BUT it's gated to cudagraph_mode=FULL only. The
  default PIECEWISE mode silently skips it.

  See: vLLM #28015, #28207, #29091, PR #29102, PR #23679.

Symptoms without this patch (or the equivalent --compilation-config CLI flag):
  - cudaErrorIllegalAddress mid-decode every 5-15 minutes of serving
  - Crashes pin to specific request states (e.g., num_output_tokens=33)
  - Both Marlin and CUTLASS GEMM backends affected
  - Only --enforce-eager fully avoids it (at ~30% throughput cost)

Fix: drop the FULL-only gate. Apply alignment for any non-NONE cudagraph mode.

Idempotent — safe to run multiple times.
"""
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py")

src = TARGET.read_text()

if "# cudagraph_align_spec_decode_all_modes" in src:
    print(f"[{TARGET.name}] already applied")
    sys.exit(0)

# Anchor on the multi-line if statement. The agent traced this to lines 1378-1385.
OLD = (
    "        if (\n"
    "            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)
NEW = (
    "        # cudagraph_align_spec_decode_all_modes\n"
    "        # Original: gated to cudagraph_mode=FULL only (vllm bug — PIECEWISE\n"
    "        # silently skips alignment, causing cudaErrorIllegalAddress on\n"
    "        # partial-acceptance decode steps when capture sizes aren't multiples\n"
    "        # of (1 + num_speculative_tokens). Apply for any non-NONE mode.\n"
    "        if (\n"
    "            cudagraph_mode != CUDAGraphMode.NONE\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)

if OLD not in src:
    raise RuntimeError(
        f"Anchor not found in {TARGET}.\n"
        "vLLM compilation.py layout may have changed — re-inspect with:\n"
        f"  grep -n 'adjust_cudagraph_sizes_for_spec_decode' {TARGET}"
    )

new_src = src.replace(OLD, NEW, 1)
TARGET.write_text(new_src)
print(f"[{TARGET.name}] applied spec-decode capture-size alignment for all cudagraph modes")
