#!/bin/bash
set -e

if [ -f /workspace/pyproject.toml ]; then
    pip install -e /workspace --quiet --no-deps 2>/dev/null || true
fi

exec "$@"
