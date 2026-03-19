#!/bin/bash
set -euo pipefail

REGISTRY="gitlab-master.nvidia.com/holoscan/i4h-containers"
TAG="dreamzero_unitree_g1"
IMAGE="${REGISTRY}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building Docker image ==="
echo "Image: ${IMAGE}"
echo "Context: ${REPO_ROOT}"
echo ""

docker build \
    -t "${IMAGE}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE}"
echo ""

read -rp "Push to ${REGISTRY}? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Logging in to GitLab container registry..."
    docker login gitlab-master.nvidia.com
    echo ""
    echo "Pushing ${IMAGE}..."
    docker push "${IMAGE}"
    echo ""
    echo "=== Push complete ==="
    echo "Pull with: docker pull ${IMAGE}"
else
    echo "Skipped push. To push later:"
    echo "  docker login gitlab-master.nvidia.com"
    echo "  docker push ${IMAGE}"
fi

echo ""
echo "=== Run example ==="
echo "docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\"
echo "  -v /path/to/dreamzero_unitree_g1:/workspace \\"
echo "  -v /path/to/data:/data \\"
echo "  -v /path/to/checkpoints:/workspace/checkpoints \\"
echo "  ${IMAGE}"
