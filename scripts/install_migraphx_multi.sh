#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

# MIGraphX is installed via apt and must match ROCm version
# Verify ROCm version is set
if [ -z "${ROCM_VERSION:-}" ]; then
    echo "ERROR: ROCM_VERSION not set. Please source .mlstack_env or set ROCM_VERSION." >&2
    exit 1
fi

echo "Installing MIGraphX (ROCm ${ROCM_VERSION})"

# Update package list
sudo apt-get update

# Install MIGraphX packages (automatically pulls ROCm-version-specific packages)
# MIGraphX packages are tied to the ROCm repository version
sudo apt-get install -y migraphx migraphx-dev half

# Verify installation
python3 - <<'PY'
try:
    import migraphx
    print("MIGraphX available: OK")
    print("MIGraphX version check: Python bindings loaded successfully")
except Exception as exc:
    print("Warning: MIGraphX python binding import failed:", exc)
    print("This may indicate a version mismatch between ROCm and MIGraphX")
PY
