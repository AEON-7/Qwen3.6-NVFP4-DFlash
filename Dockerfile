# vllm-spark-omni-q36:v1
# Source-built vLLM HEAD targeting GB10 / sm_120 / DGX Spark
#
# Strategy:
#   FROM AWQ base — already has CUDA 13.2 toolkit, PyTorch nightly cu130,
#                   modelopt patches baked in, and a working sm_120 kernel build chain.
#   Then:
#     1. Add ccache (huge speedup on rebuilds)
#     2. Clone vLLM HEAD
#     3. python use_existing_torch.py  (tell vLLM to keep OUR torch nightly cu130)
#     4. Install build deps from requirements/build/cuda.txt (not build.txt)
#     5. Compile with TORCH_CUDA_ARCH_LIST="12.0+PTX" (sm_120 + PTX → driver JITs to sm_121a)
#     6. uv pip install --no-build-isolation .   (preserves torch + flashinfer)
#     7. Upgrade flashinfer to 0.6.8 (sm_120 NVFP4 KV decode kernels)
#     8. Apply registry patch for Qwen3_5MoeForCausalLM
#
# Build time on Spark: 45-75 min clean / much faster on rebuilds via ccache
# Build env: MAX_JOBS=14, NVCC_THREADS=2 (sweet spot for 121GB/20-core)
#
# Known runtime gotchas (NOT build issues):
#   - #39761: NVFP4 decode "illegal instruction" → workaround --enforce-eager
#   - #30163: Triton bundled ptxas may lack sm_121a → symlink fix at runtime if needed
#
# Build:  docker build -t vllm-spark-omni-q36:v1 -f Dockerfile .
# Push:   docker tag vllm-spark-omni-q36:v1 ghcr.io/aeon-7/vllm-spark-omni-q36:v1 && \
#         docker push ghcr.io/aeon-7/vllm-spark-omni-q36:v1

FROM ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest

# Build extras
RUN apt-get update && apt-get install -y --no-install-recommends \
      ccache \
 && rm -rf /var/lib/apt/lists/*

# Build env — preserves PyTorch nightly cu130, targets sm_120 (PTX → JITs to sm_121a)
ENV PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    MAX_JOBS=14 \
    NVCC_THREADS=2 \
    CMAKE_BUILD_PARALLEL_LEVEL=14 \
    CCACHE_DIR=/root/.ccache \
    USE_CCACHE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    VLLM_TEST_FORCE_FP8_MARLIN=1
    # VLLM_TEST_FORCE_FP8_MARLIN=1 baked in as default — required for SM121/GB10 stability.
    # FlashInfer's CUTLASS NVFP4 path crashes with cudaErrorIllegalAddress on SM121 due
    # to 101KB SMEM limit (vs 228KB on SM100). Marlin is the only stable backend.
    # Override with -e VLLM_TEST_FORCE_FP8_MARLIN=0 if you've patched CUTLASS yourself.

# Pre-build snapshot
RUN python3 -c "import torch; print(f'torch={torch.__version__} CUDA={torch.version.cuda}')" && \
    nvcc --version | tail -1 && \
    ccache --version | head -1

# Clone vLLM HEAD (default branch: main)
ARG VLLM_REF=main
RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git /workspace/vllm-src

WORKDIR /workspace/vllm-src

# Tell vLLM to use OUR pre-installed PyTorch (not download/replace it)
RUN python3 use_existing_torch.py

# Install build deps (CUDA-specific build requirements)
RUN uv pip install --system -r requirements/build.txt 2>/dev/null || \
    uv pip install --system -r requirements/build/cuda.txt

# THE BUILD — single-arch sm_120 build via --no-build-isolation (preserves torch + flashinfer)
# Capture build log to /tmp/vllm-build.log inside image for post-mortem if needed
RUN uv pip install --system --no-build-isolation --no-deps . 2>&1 | tee /tmp/vllm-build.log | tail -100

# FlashInfer 0.6.8 (sm_120 NVFP4 KV decode)
RUN uv pip install --system --no-deps \
      "flashinfer-python>=0.6.8,<0.7" \
      "flashinfer-cubin>=0.6.8,<0.7" \
 && uv pip uninstall --system flashinfer-jit-cache 2>/dev/null || true

# Reset WORKDIR away from source tree, then nuke source.
# Without this, `import vllm` resolves to /workspace/vllm-src/vllm/ (no compiled .so → fails)
# instead of the installed /usr/local/lib/python3.12/dist-packages/vllm/.
WORKDIR /
RUN rm -rf /workspace/vllm-src

# Registry patch — Qwen3_5MoeForCausalLM (needed until upstream lands #36289/#36607/#36850 equivalent)
COPY patches/register_qwen3_5_text.py /opt/patches/
RUN python3 /opt/patches/register_qwen3_5_text.py

# Optional-import patch for vllm._C_stable_libtorch
# HEAD vLLM's _C_stable_libtorch.abi3.so depends on SM100-only kernels (mxfp4_experts_quant,
# silu_and_mul_mxfp4_experts_quant) for gpt-oss MXFP4 MoE. These don't exist on sm_120/sm_121a
# (GB10/DGX Spark), so the .so fails to load. cuda.py does an unconditional import at init,
# which cascades into vLLM being unusable. This wraps the import in RTLD_LAZY so the
# undefined MXFP4 symbols are tolerated until first call (never happens for Qwen3.6 NVFP4).
COPY patches/patch_cuda_optional_import.py /opt/patches/
RUN python3 /opt/patches/patch_cuda_optional_import.py

# KV-cache hybrid-attention patches
# Qwen3.6 has 30 linear_attention + 10 full_attention layers. Multiple sites in vLLM HEAD
# crash on None block_size for Mamba groups. Root-cause fix: default mamba_block_size to
# cache_config.block_size (typically 16) at MambaSpec construction; plus None-safe guards
# at downstream min()/cdiv() sites.
COPY patches/patch_kv_cache_utils.py /opt/patches/
RUN python3 /opt/patches/patch_kv_cache_utils.py

# M-RoPE text-only fallback patch
# Qwen3.6 declares M-RoPE in config but no model class in vLLM HEAD implements the
# SupportsMRoPE protocol. For text-only inference, M-RoPE positions are trivial
# (T=arange, H=W=0). This patch adds an inline fallback when the model doesn't
# implement the protocol.
COPY patches/patch_mrope_text_fallback.py /opt/patches/
RUN python3 /opt/patches/patch_mrope_text_fallback.py

# Verification — must pass; image is unusable otherwise
# Note: cannot test `import vllm._C` here — libcuda.so.1 is driver lib, only present
# at runtime when nvidia-container-runtime mounts it via --gpus all
RUN python3 -c "import vllm, flashinfer, torch; print(f'POST vLLM={vllm.__version__}'); print(f'POST flashinfer={flashinfer.__version__}'); print(f'POST torch={torch.__version__} CUDA={torch.version.cuda}')" && \
    python3 -c "from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS as T; assert 'Qwen3_5MoeForCausalLM' in T, 'registry patch not applied to dist-packages'; print('Qwen3_5MoeForCausalLM ->', T['Qwen3_5MoeForCausalLM'])" && \
    ls /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so && \
    echo 'image OK — vllm._C will load at runtime when --gpus all mounts libcuda.so.1'

# Runtime hint: if NVFP4 decode hits 'illegal instruction', add --enforce-eager (issue #39761)
LABEL org.opencontainers.image.title="vllm-spark-omni-q36" \
      org.opencontainers.image.description="Source-built vLLM HEAD for GB10/sm_120 + DFlash + flashinfer 0.6.8 + Qwen3.6 text-only registry fix" \
      org.opencontainers.image.source="https://github.com/aeon-7/Qwen3.6-NVFP4-DFlash" \
      org.opencontainers.image.base.name="ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest" \
      vllm.compute_capability="sm_120+PTX" \
      vllm.target_hardware="DGX Spark / GB10 / sm_121a"
