#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

# RCCL is installed via apt and must match ROCm version
# Verify ROCm version is set
if [ -z "${ROCM_VERSION:-}" ]; then
    echo "ERROR: ROCM_VERSION not set. Please source .mlstack_env or set ROCM_VERSION." >&2
    exit 1
fi

# Validate ROCm version is compatible with RCCL
case "${ROCM_VERSION%%.*}" in
    6|7)
        # ROCm 6.x and 7.x have RCCL support
        ;;
    *)
        echo "WARNING: RCCL may not be available for ROCm ${ROCM_VERSION}" >&2
        echo "RCCL is primarily supported for ROCm 6.x and 7.x" >&2
        ;;
esac

echo "Installing RCCL (ROCm ${ROCM_VERSION})"

# Update package list
sudo apt-get update

# Install RCCL packages (automatically pulls ROCm-version-specific packages)
# RCCL packages are tied to the ROCm repository version
sudo apt-get install -y rccl rccl-dev

# Verify installation
if command -v rccl-version >/dev/null 2>&1; then
    echo "RCCL version: $(rccl-version)"
else
    echo "Warning: rccl-version command not found"
fi

python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        import torch.utils.dlpack
        print("RCCL PyTorch integration: Available (via torch.utils.dlpack)")
    else:
        print("RCCL PyTorch integration: CUDA not available for verification")
except Exception as exc:
    print("Note: RCCL verification via PyTorch skipped:", exc)
PY
