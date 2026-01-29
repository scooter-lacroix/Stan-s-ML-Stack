#!/bin/bash
# Stan's ML Stack - Triton (ROCm) multi-channel installer

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

# Define stable tags for reproducibility
TRITON_STABLE_TAG="v2.3.0"
TRITON_PREVIEW_BRANCH="main"
TRITON_MLIR_BRANCH="triton-mlir"

TMP_DIR=${TMPDIR:-/tmp}/triton-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone https://github.com/ROCm/triton.git
cd triton

# Pin to stable tag or use preview branch with validation
if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    echo "WARNING: Using preview branch $TRITON_PREVIEW_BRANCH (may be unstable)"
    if ! git checkout "$TRITON_PREVIEW_BRANCH" 2>/dev/null; then
        echo "ERROR: Failed to checkout preview branch $TRITON_PREVIEW_BRANCH"
        echo "Falling back to stable tag: $TRITON_STABLE_TAG"
        git checkout "$TRITON_STABLE_TAG" || {
            echo "ERROR: Failed to checkout stable tag $TRITON_STABLE_TAG"
            echo "Please check your internet connection and repository availability."
            exit 1
        }
    fi
else
    # Try triton-mlir branch first for ROCm, fall back to stable tag
    echo "Attempting to use ROCm-optimized branch: $TRITON_MLIR_BRANCH"
    if git checkout "$TRITON_MLIR_BRANCH" 2>/dev/null; then
        echo "Successfully checked out ROCm-optimized branch"
    else
        echo "ROCm-optimized branch not available, using stable tag: $TRITON_STABLE_TAG"
        if ! git checkout "$TRITON_STABLE_TAG" 2>/dev/null; then
            echo "ERROR: Failed to checkout stable tag $TRITON_STABLE_TAG"
            echo "Please check your internet connection and repository availability."
            echo "Available tags:"
            git tag | tail -10
            exit 1
        fi
    fi
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
