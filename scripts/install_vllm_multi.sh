#!/bin/bash
# Stan's ML Stack - vLLM ROCm installer (channel-aware)

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

TMP_DIR=${TMPDIR:-/tmp}/vllm-rocm
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

git clone https://github.com/vllm-project/vllm.git
cd vllm

if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    git checkout main
else
    git checkout $(git describe --tags --abbrev=0)
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
