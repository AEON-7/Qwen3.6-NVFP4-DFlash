# Building the image yourself

If you don't want to pull `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`, here's how to reproduce it.

> ⚠️ **Heads-up about the base image:** The Dockerfile inherits from
> `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest`, which carries the original
> sm_120 NVFP4 kernel build chain (CUDA 13.2 toolkit, PyTorch nightly cu130,
> baked-in DFlash + modelopt patches). **That base image is currently private.**
>
> If you can't pull the base, you have three options:
> 1. **Easiest:** wait for it to be flipped to public (file an issue), OR
> 2. **Authenticated pull:** `docker login ghcr.io -u <your-gh-user>` with a PAT that
>    has the `read:packages` scope IF you've been granted access to the AEON-7 org's
>    private packages, OR
> 3. **Fully self-built:** swap the `FROM` line in the Dockerfile to
>    `nvidia/cuda:13.2.0-devel-ubuntu24.04` and add the missing layers (PyTorch nightly
>    cu130, sm_120 NVFP4 kernel build for vLLM/FlashInfer, modelopt patches). This is
>    multi-hour engineering work — see [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
>    for a reasonable starting point.

## Why source-build?

The vLLM nightly wheels at https://wheels.vllm.ai/nightly are built against **CUDA 12.8**. DGX Spark ships with **CUDA 13.x** PyTorch nightly. Mixing produces `libcudart.so.12 cannot open shared object` errors at import.

The solution is to compile vLLM from source against the host's PyTorch + CUDA toolkit. Since DGX Spark is a relatively new SKU (sm_121a / GB10), no public registry has cu13/sm_120 vLLM images yet — hence this project.

## Hardware needs for build

- **A DGX Spark** (or another sm_120 / sm_121a Blackwell box)
- 121 GB unified memory (the Spark default)
- 20+ CPU cores
- 30 GB free disk
- 60-90 min wall clock

You can build on a different sm_120 host if you have one — the `+PTX` arch tag means the resulting image will JIT to whatever sm_120/121a/121 GPU it actually runs on.

## Step-by-step

```bash
git clone https://github.com/aeon-7/Qwen3.6-NVFP4-DFlash.git
cd Qwen3.6-NVFP4-DFlash

# Build (reads ./Dockerfile, takes 45-75 min)
docker build -t vllm-spark-omni-q36:v1 .
```

That's it. Watch the build log for kernel compilation progress:

```
[1/183]  CUDA Linking executable cumem_allocator
[10/183] Building CUDA object .../moe.cu.o
[100/183] ...
[183/183] Building CUDA object .../topk_softmax.cu.o
Successfully built vllm-spark-omni-q36:v1
```

## Build environment knobs

The Dockerfile defaults are tuned for a 20-core / 121 GB Spark:

| Env var | Default | Tune up if... | Tune down if... |
|---|---|---|---|
| `MAX_JOBS` | 14 | You have > 32 cores | OOM during build |
| `NVCC_THREADS` | 2 | RAM is plentiful | nvcc OOMs |
| `TORCH_CUDA_ARCH_LIST` | `"12.0+PTX"` | You want native sm_121a SASS: `"12.0+PTX;12.1"` | (don't) |
| `CCACHE_DIR` | `/root/.ccache` | (mount as volume for cross-build cache) | |

To rebuild with custom env:

```bash
docker build \
  --build-arg VLLM_REF=v0.9.2 \      # pin a specific vLLM tag
  --build-arg MAX_JOBS=20 \
  -t vllm-spark-omni-q36:v1 \
  .
```

## Build artifacts

Inside the resulting image:

```
/workspace/vllm-src/      # full vLLM source tree (kept for incremental rebuilds)
/workspace/wheels/        # built wheels
/root/.ccache/            # nvcc artifact cache
/opt/patches/             # registry patch script
/usr/local/lib/python3.12/dist-packages/vllm/   # installed package
```

Image size: ~8 GB compressed, ~22 GB uncompressed. Most of the bulk is the CUDA toolkit + PyTorch.

## Pushing to GHCR

To publish your own variant:

```bash
# Tag
docker tag vllm-spark-omni-q36:v1 ghcr.io/<your-username>/vllm-spark-omni-q36:v1

# Login (use a GitHub PAT with write:packages scope)
echo $GITHUB_PAT | docker login ghcr.io -u <your-username> --password-stdin

# Push
docker push ghcr.io/<your-username>/vllm-spark-omni-q36:v1
```

See [`scripts/push-ghcr.sh`](../scripts/push-ghcr.sh) for the full helper.

## Customizing the image

The Dockerfile is intentionally minimal — three layers do the work:

1. **Source build**: clone + `python use_existing_torch.py` + `uv pip install --no-build-isolation .`
2. **FlashInfer 0.6.8 upgrade**: `flashinfer-python` + `flashinfer-cubin`
3. **Registry patch**: `register_qwen3_5_text.py`

Common modifications:

### Pin a specific vLLM commit

```dockerfile
ARG VLLM_REF=5cdddddd4
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm-src && \
    cd /workspace/vllm-src && git checkout ${VLLM_REF}
```

### Cherry-pick open PRs (e.g., NVFP4 KV cache wiring)

```dockerfile
RUN cd /workspace/vllm-src && \
    git fetch origin pull/40177/head:pr-40177 && \
    git checkout pr-40177
```

### Bake in different patches

Drop them in `patches/` and add to the Dockerfile:

```dockerfile
COPY patches/my_patch.py /opt/patches/
RUN python3 /opt/patches/my_patch.py
```

### Build for additional GPUs

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX"   # H100 + B200 + Spark
```

This roughly triples build time but produces a multi-arch image.

## Troubleshooting the build

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`

Your CUDA toolkit is too old. The image needs CUDA ≥ 13.0. Verify:

```bash
docker run --rm ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest nvcc --version
```

Should report `release 13.2` or newer.

### `RuntimeError: CUDA out of memory` during build

Reduce `MAX_JOBS` to 8 or 6:

```bash
docker build --build-arg MAX_JOBS=8 -t vllm-spark-omni-q36:v1 .
```

### `cmake error: Could NOT find Python3`

The base image's Python should be picked up automatically. If not, set:

```dockerfile
ENV CMAKE_PREFIX_PATH=/usr/lib/python3.12
```

### Build hangs at "Generating CUDA stubs"

This is normal for the first run — nvcc is building hundreds of object files. Watch CPU + GPU utilization to confirm progress:

```bash
docker stats   # in another terminal
htop           # CPU should be pinned at MAX_JOBS × 100%
```

## Verification

After build:

```bash
docker run --rm vllm-spark-omni-q36:v1 \
  python3 -c "
import vllm, flashinfer, torch
print(f'vLLM={vllm.__version__}')
print(f'flashinfer={flashinfer.__version__}')
print(f'torch={torch.__version__} CUDA={torch.version.cuda}')

from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS as T
assert 'Qwen3_5MoeForCausalLM' in T, 'registry patch missing'
print(f'Qwen3_5MoeForCausalLM -> {T[\"Qwen3_5MoeForCausalLM\"]}')
"
```

Expected:
```
vLLM=0.19.1rc1.dev383+g5cdddddd4   (or newer)
flashinfer=0.6.8
torch=2.12.0.dev20260408+cu130 CUDA=13.0
Qwen3_5MoeForCausalLM -> ('qwen3_5', 'Qwen3_5MoeForCausalLM')
```
