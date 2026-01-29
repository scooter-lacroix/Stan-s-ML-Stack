#!/bin/bash
# Stan's ML Stack - PyTorch ROCm Installer (channel-aware)

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_VERSION=${ROCM_VERSION:-7.2}
ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

case "$GPU_ARCH" in
    gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
        export PYTORCH_ROCM_ARCH=gfx1030
        ;;
    *)
        export PYTORCH_ROCM_ARCH="$GPU_ARCH"
        ;;
esac

if [ "$ROCM_CHANNEL" = "preview" ]; then
    INDEX_URL="https://rocm.nightlies.amd.com/v2/${GPU_ARCH}/"
    EXTRA_FLAGS=(--pre --index-url "$INDEX_URL")
else
    # Pattern matching order is critical: more specific patterns must come before
    # general patterns. For example, 7.1* must come before 7* to match correctly.
    case "${ROCM_VERSION%%.*}.${ROCM_VERSION#*.}" in
        6.4*) INDEX_URL="https://download.pytorch.org/whl/rocm6.4" ; EXTRA_FLAGS=(--index-url "$INDEX_URL") ;;
        7.1*) INDEX_URL="https://download.pytorch.org/whl/nightly/rocm7.1" ; EXTRA_FLAGS=(--pre --index-url "$INDEX_URL") ;;
        7.2*) INDEX_URL="https://download.pytorch.org/whl/nightly/rocm7.2" ; EXTRA_FLAGS=(--pre --index-url "$INDEX_URL") ;;
        7.0*|7.10*) INDEX_URL="https://download.pytorch.org/whl/nightly/rocm7.0" ; EXTRA_FLAGS=(--pre --index-url "$INDEX_URL") ;;
        *) INDEX_URL="https://download.pytorch.org/whl/nightly/rocm7.2" ; EXTRA_FLAGS=(--pre --index-url "$INDEX_URL") ;;
    esac
fi

echo "Installing PyTorch for ROCm $ROCM_VERSION ($ROCM_CHANNEL)"
pip3 install "torch" "torchvision" "torchaudio" "${EXTRA_FLAGS[@]}"

echo "Verifying installation"
python3 <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.ones(128, device='cuda')
    print("Sanity sum:", x.sum().item())
PY
