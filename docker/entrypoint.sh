#!/bin/bash
set -e

echo "[entrypoint] Installing flash_attn (compiling for this GPU arch, ~5-10 min)..."
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;9.0a;10.0;10.0a}" \
MAX_JOBS="${MAX_JOBS:-8}" \
pip install --no-build-isolation flash-attn --no-cache-dir -q 2>&1 | tail -5
echo "[entrypoint] flash_attn ready."

# Install the mounted codebase in editable mode if present
if [ -f /workspace/pyproject.toml ]; then
    pip install -e /workspace --quiet --no-deps 2>/dev/null || true
fi

exec "$@"
