#!/bin/bash
# Stan's ML Stack - PyTorch ROCm Installer (channel-aware)

set -euo pipefail

# Source utility scripts if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi
if [ -f "$SCRIPT_DIR/env_validation_utils.sh" ]; then
    source "$SCRIPT_DIR/env_validation_utils.sh"
fi
if [ -f "$SCRIPT_DIR/gpu_detection_utils.sh" ]; then
    source "$SCRIPT_DIR/gpu_detection_utils.sh"
fi

# Dry run flag check
DRY_RUN=${DRY_RUN:-false}
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

# Require and validate .mlstack_env
if type require_mlstack_env >/dev/null 2>&1; then
    require_mlstack_env "$(basename "$0")" || true
else
    # Fallback validation if utils not available
    if [ -f "$HOME/.mlstack_env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.mlstack_env"
    fi
fi

ROCM_VERSION=${ROCM_VERSION:-7.2}
ROCM_CHANNEL=${ROCM_CHANNEL:-latest}

# Detect GPU architecture with validation
if [ -z "${GPU_ARCH:-}" ]; then
    if type detect_gpu_architecture >/dev/null 2>&1; then
        GPU_ARCH=$(detect_gpu_architecture) || exit 1
    else
        # Fallback to old method if utils not available
        GPU_ARCH=$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1)
        if [ -z "$GPU_ARCH" ]; then
            print_error "Unable to detect GPU architecture"
            print_step "Please ensure ROCm is installed and GPU is properly configured."
            exit 1
        fi
    fi
fi

# Validate GPU detection
if type validate_gpu_detection >/dev/null 2>&1; then
    validate_gpu_detection "$GPU_ARCH" "$(basename "$0")" || exit 1
fi

case "$GPU_ARCH" in
    gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
        export PYTORCH_ROCM_ARCH=gfx1030
        ;;
    *)
        export PYTORCH_ROCM_ARCH="$GPU_ARCH"
        ;;
esac

if [ "${ROCM_CHANNEL:-}" = "preview" ]; then
    INDEX_URL="https://rocm.nightlies.amd.com/v2/${GPU_ARCH:-gfx1100}/"
    EXTRA_FLAGS=(--pre --index-url "$INDEX_URL")
else
    # ROCm 7.x uses AMD's radeon repo for official wheels
    # PyTorch nightly wheels often have issues with MPI dependencies
    case "${ROCM_VERSION:-7.2}" in
        6.2*) INDEX_URL="https://download.pytorch.org/whl/rocm6.2" ; EXTRA_FLAGS=(--index-url "$INDEX_URL") ;;
        6.3*) INDEX_URL="https://download.pytorch.org/whl/rocm6.3" ; EXTRA_FLAGS=(--index-url "$INDEX_URL") ;;
        6.4*) INDEX_URL="https://download.pytorch.org/whl/rocm6.4" ; EXTRA_FLAGS=(--index-url "$INDEX_URL") ;;
        7.0*|7.1*|7.2*)
            # Use AMD's official ROCm 7.2 wheels - these are properly built
            INDEX_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"
            EXTRA_FLAGS=(--index-url "$INDEX_URL")
            ;;
        *) INDEX_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/" ; EXTRA_FLAGS=(--index-url "$INDEX_URL") ;;
    esac
fi

# Check for required ROCm packages before installing PyTorch
print_section "Checking ROCm dependencies"
MISSING_PKGS=""
if ! dpkg -l rocm-libs >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS rocm-libs"
fi
if ! dpkg -l rocm-dev >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS rocm-dev"
fi

if [ -n "$MISSING_PKGS" ]; then
    echo "⚠ Missing required ROCm packages:$MISSING_PKGS"
    echo "➤ Installing missing packages..."
    sudo apt-get update && sudo apt-get install -y $MISSING_PKGS
fi

print_section "Installing PyTorch for ROCm $ROCM_VERSION ($ROCM_CHANNEL)"
execute_command "pip3 install --break-system-packages torch torchvision torchaudio ${EXTRA_FLAGS[*]}" "Installing PyTorch components"

if [ "$DRY_RUN" = "false" ]; then
    print_section "Verifying installation"
    python3 <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.ones(128, device='cuda')
    print("Sanity sum:", x.sum().item())
PY
fi
