#!/bin/bash
# Stan's ML Stack - bitsandbytes ROCm installer (channel-aware)
# Ensures ROCm support by building from ROCm fork when PyPI version lacks HIP binaries

set -euo pipefail

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

# Source utility scripts if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi
if [ -f "$SCRIPT_DIR/env_validation_utils.sh" ]; then
    source "$SCRIPT_DIR/env_validation_utils.sh"
fi

# Dry run flag check
DRY_RUN=${DRY_RUN:-false}
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

# Source environment file or use defaults
if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_VERSION=${ROCM_VERSION:-$(cat /opt/rocm/.info/version 2>/dev/null | head -n1 || echo "7.2.0")}
GPU_ARCH=${GPU_ARCH:-gfx1100}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ Installing bitsandbytes for ROCm ${ROCM_VERSION}"
echo "└─────────────────────────────────────────────────────────┘"

# Function to check if ROCm binary is present
check_rocm_binary() {
    python3 - <<'PY'
import importlib.util
import sys
import pathlib
spec = importlib.util.find_spec("bitsandbytes")
if not spec or not spec.origin:
    sys.exit(1)
package_dir = pathlib.Path(spec.origin).parent
# Check for ROCm binary
libs = list(package_dir.glob("libbitsandbytes_rocm*.so"))
if libs:
    print("Found ROCm binary:", libs[0])
    sys.exit(0)
# Also check for HIP backend
libs_hip = list(package_dir.glob("libbitsandbytes_hip*.so"))
if libs_hip:
    print("Found HIP binary:", libs_hip[0])
    sys.exit(0)
# Check for any binary and try to verify it's not CUDA-only
any_libs = list(package_dir.glob("libbitsandbytes*.so"))
if any_libs:
    # If there's a cuda lib, it's NVIDIA version
    cuda_libs = list(package_dir.glob("libbitsandbytes_cuda*.so"))
    if cuda_libs and not libs and not libs_hip:
        print("Only CUDA binary found - need ROCm version")
        sys.exit(1)
print("No ROCm/HIP binary found in", package_dir)
sys.exit(1)
PY
}

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY-RUN] Would install bitsandbytes with ROCm support"
    exit 0
fi

# Uninstall previous bitsandbytes to avoid conflicts
echo "➤ Removing existing bitsandbytes installation..."
"$PYTHON_BIN" -m pip uninstall -y bitsandbytes 2>/dev/null || true

# First, try installing from PyPI (newer versions may have ROCm support)
echo "➤ Attempting to install bitsandbytes from PyPI..."
if "$PYTHON_BIN" -m pip install bitsandbytes --break-system-packages --no-cache-dir 2>/dev/null; then
    echo "✓ bitsandbytes installed from PyPI"

    # Check if ROCm binary is present
    echo "➤ Checking for ROCm/HIP binary..."
    if check_rocm_binary 2>/dev/null; then
        echo "✓ bitsandbytes ROCm binary detected"
        # Verify it works
        echo "➤ Verifying bitsandbytes import..."
        python3 -c "import bitsandbytes as bnb; print('bitsandbytes version:', bnb.__version__)"
        exit 0
    else
        echo "⚠ PyPI version lacks ROCm binary, building from ROCm fork..."
    fi
else
    echo "⚠ PyPI installation failed, building from ROCm fork..."
fi

# Build from ROCm/bitsandbytes fork for proper ROCm support
repo_root="$HOME/ml_stack/bitsandbytes_source"
if [ -d "$repo_root" ]; then
    rm -rf "$repo_root"
fi
mkdir -p "$(dirname "$repo_root")"

echo "➤ Cloning ROCm bitsandbytes fork..."
if ! git clone --recursive https://github.com/ROCm/bitsandbytes.git "$repo_root" 2>&1; then
    echo "✗ Failed to clone ROCm bitsandbytes fork"
    echo "  Trying alternative: TimDettmers fork with ROCm patches..."
    if ! git clone --recursive https://github.com/TimDettmers/bitsandbytes.git "$repo_root" 2>&1; then
        echo "✗ Failed to clone bitsandbytes"
        exit 1
    fi
fi

cd "$repo_root"

# Set up build environment for ROCm
export BNB_USE_ROCM=1
export BNB_ROCM_ARCH="$GPU_ARCH"
export ROCM_PATH="$ROCM_PATH"
export ROCM_HOME="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"

# Extract version shortcode (e.g., 7.2.0 -> 72)
if [[ "$ROCM_VERSION" =~ ^([0-9]+)\.([0-9]+) ]]; then
    version_short="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
else
    version_short="72"
fi
export BNB_ROCM_VERSION="$version_short"

echo "➤ Building bitsandbytes for ROCm $ROCM_VERSION (arch: $GPU_ARCH)"

# Ensure build dependencies are present
echo "➤ Installing build dependencies..."
"$PYTHON_BIN" -m pip install scikit-build-core cmake ninja --break-system-packages 2>/dev/null || true

# Clean and build
"$PYTHON_BIN" -m pip uninstall -y bitsandbytes 2>/dev/null || true

# Build with HIP backend for ROCm
echo "➤ Building with HIP backend..."
if "$PYTHON_BIN" -m pip install . --no-build-isolation --break-system-packages \
    -Ccmake.define.COMPUTE_BACKEND=hip \
    -Ccmake.define.ROCM_PATH="$ROCM_PATH" 2>&1; then
    echo "✓ bitsandbytes built successfully from source"
else
    echo "✗ bitsandbytes source build failed"
    echo "  Trying alternative build method..."

    # Alternative: use make directly
    if [ -f "Makefile" ]; then
        make hip
        "$PYTHON_BIN" -m pip install . --break-system-packages
    else
        echo "✗ No alternative build method available"
        exit 1
    fi
fi

cd - >/dev/null 2>&1 || true

# Final verification
echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ Verifying bitsandbytes installation"
echo "└─────────────────────────────────────────────────────────┘"

python3 - <<'PY'
try:
    import bitsandbytes as bnb
    print("✓ bitsandbytes version:", bnb.__version__)

    # Try to verify HIP/ROCm support
    import pathlib
    import importlib.util
    spec = importlib.util.find_spec("bitsandbytes")
    if spec and spec.origin:
        package_dir = pathlib.Path(spec.origin).parent
        hip_libs = list(package_dir.glob("libbitsandbytes_hip*.so")) + list(package_dir.glob("libbitsandbytes_rocm*.so"))
        if hip_libs:
            print("✓ ROCm/HIP backend:", hip_libs[0].name)
        else:
            print("⚠ No ROCm/HIP backend library found")
except Exception as e:
    print(f"✗ Failed to import bitsandbytes: {e}")
    exit(1)
PY

echo "✓ bitsandbytes installation complete"
