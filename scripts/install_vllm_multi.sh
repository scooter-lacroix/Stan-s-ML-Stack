#!/bin/bash
# Stan's ML Stack - vLLM ROCm installer (channel-aware)

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

# Define stable tags for reproducibility
VLLM_STABLE_TAG="v0.15.0"
VLLM_PREVIEW_BRANCH="main"

TMP_DIR=${TMPDIR:-/tmp}/vllm-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone https://github.com/vllm-project/vllm.git
cd vllm

# Pin to stable tag or use preview branch with validation
if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    echo "WARNING: Using preview branch $VLLM_PREVIEW_BRANCH (may be unstable)"
    if ! git checkout "$VLLM_PREVIEW_BRANCH" 2>/dev/null; then
        echo "ERROR: Failed to checkout preview branch $VLLM_PREVIEW_BRANCH"
        echo "Falling back to stable tag: $VLLM_STABLE_TAG"
        git checkout "$VLLM_STABLE_TAG" || {
            echo "ERROR: Failed to checkout stable tag $VLLM_STABLE_TAG"
            echo "Please check your internet connection and repository availability."
            exit 1
        }
    fi
else
    echo "Using stable tag: $VLLM_STABLE_TAG"
    if ! git checkout "$VLLM_STABLE_TAG" 2>/dev/null; then
        echo "ERROR: Failed to checkout stable tag $VLLM_STABLE_TAG"
        echo "Attempting to use latest available tag..."
        LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null)
        if [ -n "$LATEST_TAG" ]; then
            echo "Using latest tag: $LATEST_TAG"
            git checkout "$LATEST_TAG" || {
                echo "ERROR: Failed to checkout latest tag $LATEST_TAG"
                echo "Please check your internet connection and repository availability."
                echo "Available tags:"
                git tag | tail -10
                exit 1
            }
        else
            echo "ERROR: No tags available in repository"
            exit 1
        fi
    fi
fi

pip3 install -U -r requirements-rocm.txt

export PYTORCH_ROCM_ARCH="$GPU_ARCH"
export BUILD_FA=0
export VLLM_TARGET_DEVICE=hip
export MAX_JOBS=$(( $(nproc) - 1 ))

python3 setup.py develop

python3 <<'PY'
import vllm
print("vLLM version:", vllm.__version__)
PY
