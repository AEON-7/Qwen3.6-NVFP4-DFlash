#!/usr/bin/env python3
"""Make `import vllm._C_stable_libtorch` succeed on sm_120/sm_121a (GB10/DGX Spark).

Background: vLLM HEAD's `_C_stable_libtorch.abi3.so` references SM100-only kernels
(`mxfp4_experts_quant`, `silu_and_mul_mxfp4_experts_quant`) used by gpt-oss MXFP4
MoE. These are ungated declarations of ops in `csrc/libtorch_stable/torch_bindings.cpp`,
but their CUDA implementations are guarded behind sm_100 arch — not built for
sm_120, leaving undefined symbols.

Default `dlopen` mode is RTLD_NOW which resolves all symbols at load time → fails.
With RTLD_LAZY, undefined symbols are tolerated until first call. Since Qwen3.6
NVFP4 + DFlash never invokes MXFP4 paths, the symbols stay unresolved harmlessly,
and all the OTHER ops in `_C_stable_libtorch` (including `cutlass_scaled_mm_supports_fp8`)
register cleanly.

Verified working: `torch.ops._C.cutlass_scaled_mm_supports_fp8(120) == True` after patch.

Idempotent — safe to run multiple times.
"""
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py")

src = TARGET.read_text()

if "# stable_libtorch_lazy_dlopen" in src:
    print("[patch_cuda_optional_import] already applied")
    sys.exit(0)

OLD = "import vllm._C_stable_libtorch  # noqa"
NEW = (
    "# stable_libtorch_lazy_dlopen — sm_120 (GB10/DGX Spark) workaround for missing MXFP4 sm_100 kernels\n"
    "import sys as _sys, os as _os\n"
    "_old_dlopen_flags = _sys.getdlopenflags()\n"
    "_sys.setdlopenflags(_os.RTLD_LAZY | _os.RTLD_GLOBAL)\n"
    "try:\n"
    "    import vllm._C_stable_libtorch  # noqa\n"
    "finally:\n"
    "    _sys.setdlopenflags(_old_dlopen_flags)"
)

if OLD not in src:
    raise RuntimeError(
        f"Anchor not found in {TARGET}.\n"
        "vLLM cuda.py layout may have changed — re-inspect with:\n"
        f"  grep -n '_C_stable_libtorch' {TARGET}"
    )

new_src = src.replace(OLD, NEW, 1)
TARGET.write_text(new_src)
print(f"[patch_cuda_optional_import] wrapped import in RTLD_LAZY in {TARGET}")
