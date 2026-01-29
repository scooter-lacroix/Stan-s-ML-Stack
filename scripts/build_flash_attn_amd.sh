#!/bin/bash
# Stan's ML Stack - FlashAttention build helper for AMD GPUs

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

# Define stable tags for reproducibility
# ROCm flash-attention uses -cktile suffix tags
FLASH_ATTENTION_STABLE_TAG="v2.8.0-cktile"
FLASH_ATTENTION_PREVIEW_BRANCH="main_perf"

TMP_DIR=${TMPDIR:-/tmp}/flash-attention-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention

# Pin to stable tag or use preview branch with validation
if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    echo "WARNING: Using preview branch main_perf (may be unstable)"
    if ! git checkout "$FLASH_ATTENTION_PREVIEW_BRANCH" 2>/dev/null; then
        echo "ERROR: Failed to checkout preview branch $FLASH_ATTENTION_PREVIEW_BRANCH"
        echo "Falling back to stable tag: $FLASH_ATTENTION_STABLE_TAG"
        git checkout "$FLASH_ATTENTION_STABLE_TAG" || {
            echo "ERROR: Failed to checkout stable tag $FLASH_ATTENTION_STABLE_TAG"
            echo "Please check your internet connection and repository availability."
            exit 1
        }
    fi
else
    echo "Using stable tag: $FLASH_ATTENTION_STABLE_TAG"
    if ! git checkout "$FLASH_ATTENTION_STABLE_TAG" 2>/dev/null; then
        echo "ERROR: Failed to checkout stable tag $FLASH_ATTENTION_STABLE_TAG"
        echo "Please check your internet connection and repository availability."
        echo "Available tags:"
        git tag | tail -10
        exit 1
    fi
fi

case "$GPU_ARCH" in
    gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
        export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
        export FLASH_ATTENTION_CK_ENABLE=FALSE
        export GPU_ARCHS=gfx1030
        ;;
    *)
        export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
        export FLASH_ATTENTION_CK_ENABLE=FALSE
        export GPU_ARCHS="$GPU_ARCH"
        ;;
esac

export MAX_JOBS=$(( $(nproc) - 1 ))
python3 setup.py install

python3 <<'PY'
import flash_attn
print("FlashAttention version:", flash_attn.__version__)
PY
