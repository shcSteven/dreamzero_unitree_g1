#!/bin/bash
set -e

# Verify flash_attn is compatible with this PyTorch; rebuild if not
python -c "import flash_attn" 2>/dev/null || {
    echo "[entrypoint] flash_attn ABI mismatch detected, rebuilding (~10 min)..."
    MAX_JOBS=8 pip install --no-build-isolation flash-attn --force-reinstall --no-cache-dir -q
}

# Install the mounted codebase in editable mode if present
if [ -f /workspace/pyproject.toml ]; then
    pip install -e /workspace --quiet --no-deps 2>/dev/null || true
fi

exec "$@"
