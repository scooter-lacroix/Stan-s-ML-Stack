#!/bin/bash
# Stan's ML Stack - FlashAttention build helper for AMD GPUs

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

TMP_DIR=${TMPDIR:-/tmp}/flash-attention-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention

if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    git checkout main_perf
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
