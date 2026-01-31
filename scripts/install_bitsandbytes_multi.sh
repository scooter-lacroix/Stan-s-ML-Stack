#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

# Define compatible versions for each ROCm version
case "${ROCM_VERSION%%.*}" in
    6)
        # ROCm 6.x: Use bitsandbytes 0.43.x for compatibility
        BITSANDBYTES_VERSION=">=0.43.0,<0.44.0"
        ;;
    7)
        # ROCm 7.x: Use bitsandbytes 0.45.x for compatibility
        BITSANDBYTES_VERSION=">=0.45.0,<0.46.0"
        ;;
    *)
        # Default: Latest compatible version
        BITSANDBYTES_VERSION=">=0.45.0"
        echo "WARNING: Unknown ROCm version ${ROCM_VERSION:-unknown}, using default bitsandbytes version range"
        ;;
esac

echo "Installing bitsandbytes for ROCm ${ROCM_VERSION:-unknown}"
echo "Version constraint: $BITSANDBYTES_VERSION"

# Install with version constraint
pip3 install --upgrade "bitsandbytes$BITSANDBYTES_VERSION"

python3 - <<'PY'
import bitsandbytes as bnb
print("bitsandbytes version:", bnb.__version__)
PY
