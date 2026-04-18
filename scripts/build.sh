#!/usr/bin/env bash
# build.sh — Source-build vllm-spark-omni-q36 on a DGX Spark or sm_120 host.
#
# Usage: ./scripts/build.sh [TAG]
#   TAG defaults to v1 → produces vllm-spark-omni-q36:v1
#
# Build time: 45-75 min on Spark
set -euo pipefail

TAG="${1:-v1}"
IMAGE_LOCAL="vllm-spark-omni-q36:${TAG}"

cd "$(dirname "$0")/.."

echo "== Building ${IMAGE_LOCAL} =="
echo "  source: ./Dockerfile"
echo "  context: $(pwd)"
echo "  cores: $(nproc)"
echo "  ram: $(free -g | awk 'NR==2{print $2}') GB"
echo

# Sanity checks
if ! command -v docker >/dev/null; then
  echo "ERROR: docker not installed" >&2; exit 1
fi
if ! docker info | grep -qi nvidia; then
  echo "WARN: nvidia runtime not detected — build will succeed but GPU runs won't" >&2
fi

# Capture full log
LOG="/tmp/vllm-omni-build-$(date +%s).log"
echo "== Log: $LOG =="
echo

docker build \
  -t "${IMAGE_LOCAL}" \
  -f Dockerfile \
  . 2>&1 | tee "$LOG"

echo
echo "== Done =="
echo "  image: ${IMAGE_LOCAL}"
echo "  size:  $(docker images --format '{{.Size}}' "${IMAGE_LOCAL}" | head -1)"
echo "  log:   $LOG"
echo
echo "Next:"
echo "  ./scripts/push-ghcr.sh ${TAG}        # publish to GHCR"
echo "  docker compose -f examples/docker-compose.yml up -d   # smoke test"
