#!/bin/bash
# Stan's ML Stack - Triton (ROCm) multi-channel installer

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

TMP_DIR=${TMPDIR:-/tmp}/triton-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone https://github.com/ROCm/triton.git
cd triton

if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    git checkout main
else
    git checkout triton-mlir || git checkout main
fi

cd python

export GPU_ARCHS="$GPU_ARCH"
export TRITON_BUILD_WITH_CCACHE=true
export MAX_JOBS=$(( $(nproc) - 1 ))

python3 setup.py install

python3 <<'PY'
import triton
print("Triton version:", triton.__version__)
PY
