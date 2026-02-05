#!/bin/bash
# Stan's ML Stack - ONNX Runtime Build (ROCm multi-channel)

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    set +u 2>/dev/null || true
    source "$HOME/.mlstack_env"
    set -u 2>/dev/null || true
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROCM_VERSION=${ROCM_VERSION:-$(cat /opt/rocm/.info/version 2>/dev/null || rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs || echo 7.2)}
echo "Building ONNX Runtime with ROCm version: ${ROCM_VERSION}"
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

sudo apt-get update
sudo apt-get install -y cmake libprotobuf-dev protobuf-compiler
sudo apt-get install -y migraphx migraphx-dev half || true

WORKDIR=${TMPDIR:-/tmp}/onnxruntime-rocm
sudo rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Use a stable tag known to work with ROCm
git reset --hard
git checkout v1.20.1
git submodule update --init --recursive

# Patch CMakeLists.txt to handle ROCm 7.x version format in .info/version
cat > patch_onnx_local.py << 'EOF'
import os
path = 'cmake/CMakeLists.txt'
with open(path, 'r') as f:
    content = f.read()
# Replace strict regex with a more flexible one
content = content.replace('^([0-9]+)\\.([0-9]+)\\.([0-9]+)-.*$', '^([0-9]+)\\.([0-9]+)\\.([0-9]+).*')
with open(path, 'w') as f:
    f.write(content)
EOF
python3 patch_onnx_local.py

case "$GPU_ARCH" in
    gfx103*) HIP_ARCHS=gfx1030 ;;
    gfx1100|gfx1101|gfx1102|gfx1103) HIP_ARCHS="gfx1100;gfx1101;gfx1102;gfx1103" ;;
    gfx1200|gfx1201) HIP_ARCHS="gfx1200;gfx1201" ;;
    *) HIP_ARCHS=gfx1100 ;;
esac

# Ensure ROCM_VERSION is in the format expected by ONNX (e.g., 70200 for 7.2.0)
if [[ "$ROCM_VERSION" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    patch="${BASH_REMATCH[3]}"
    # Format as Mmmpp (e.g., 7.2.0 -> 70200)
    ORT_ROCM_VERSION=$(printf "%d%02d%02d" "$major" "$minor" "$patch")
else
    ORT_ROCM_VERSION=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
fi
export ROCM_HOME="${ROCM_PATH}"
export ROCM_PATH="${ROCM_PATH}"
export HIP_PATH="${ROCM_PATH}"

# Clear environment to avoid picking up system/anaconda libs that conflict
export PYTHONPATH=""
export CMAKE_PREFIX_PATH="/opt/rocm"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

# Force use of system CMake
CMAKE_BIN=$(which cmake || echo "/usr/bin/cmake")

# Run build with CMake policy version minimum set to handle old dependencies
# This fixes: "Compatibility with CMake < 3.5 has been removed from CMake"
./build.sh --config Release \
           --cmake_path "$CMAKE_BIN" \
           --build_wheel \
           --parallel $(( $(nproc) - 1 )) \
           --use_rocm --rocm_home "${ROCM_PATH}" \
           --rocm_version "${ORT_ROCM_VERSION}" \
           --use_migraphx --migraphx_home /opt/rocm \
           --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=${HIP_ARCHS} \
           --cmake_extra_defines onnxruntime_USE_EXTERNAL_ABSEIL=OFF \
           --cmake_extra_defines CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON \
           --cmake_extra_defines CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON \
           --cmake_extra_defines CMAKE_IGNORE_PATH="$HOME/anaconda3/lib/cmake;$HOME/anaconda3/lib;$HOME/anaconda3" \
           --cmake_extra_defines CMAKE_PREFIX_PATH="/opt/rocm" \
           --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
           --allow_running_as_root

python3 -m pip uninstall -y onnxruntime onnxruntime-rocm onnxruntime-gpu || true
python3 -m pip install build/Linux/Release/dist/*.whl --break-system-packages

python3 <<'PY'
import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
PY
