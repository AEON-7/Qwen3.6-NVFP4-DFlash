#!/usr/bin/env bash
# push-ghcr.sh — Tag and push vllm-spark-omni-q36 to GitHub Container Registry.
#
# Prerequisites:
#   - GitHub PAT with `write:packages` scope, exported as GITHUB_TOKEN
#   - GitHub username, exported as GITHUB_USER (defaults to aeon-7)
#
# Usage: ./scripts/push-ghcr.sh [TAG]
set -euo pipefail

TAG="${1:-v1}"
GH_USER="${GITHUB_USER:-aeon-7}"
LOCAL="vllm-spark-omni-q36:${TAG}"
REMOTE="ghcr.io/${GH_USER}/vllm-spark-omni-q36:${TAG}"
REMOTE_LATEST="ghcr.io/${GH_USER}/vllm-spark-omni-q36:latest"

# Sanity
if ! docker image inspect "${LOCAL}" >/dev/null 2>&1; then
  echo "ERROR: image ${LOCAL} not found locally. Run ./scripts/build.sh first." >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: set GITHUB_TOKEN to a PAT with write:packages scope" >&2
  echo "  export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxx" >&2
  exit 1
fi

echo "== Pushing ${LOCAL} → ${REMOTE} =="

# Login
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GH_USER}" --password-stdin

# Tag + push versioned
docker tag "${LOCAL}" "${REMOTE}"
docker push "${REMOTE}"

# Tag + push :latest
docker tag "${LOCAL}" "${REMOTE_LATEST}"
docker push "${REMOTE_LATEST}"

# Logout (be polite)
docker logout ghcr.io

echo
echo "== Done =="
echo "  ${REMOTE}"
echo "  ${REMOTE_LATEST}"
echo
echo "Visibility on GHCR is private by default. To make public:"
echo "  https://github.com/users/${GH_USER}/packages/container/vllm-spark-omni-q36/settings"
echo "  → Danger Zone → Change visibility → Public"
