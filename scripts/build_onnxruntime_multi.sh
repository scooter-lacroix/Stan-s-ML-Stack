#!/bin/bash
# Stan's ML Stack - ONNX Runtime Build (ROCm multi-channel)

set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

ROCM_VERSION=${ROCM_VERSION:-7.2}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

sudo apt-get update
sudo apt-get install -y cmake libprotobuf-dev protobuf-compiler
sudo apt-get install -y migraphx migraphx-dev half || true

WORKDIR=${TMPDIR:-/tmp}/onnxruntime-rocm
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

if [[ "$GPU_ARCH" =~ ^gfx12 ]] || [[ "$ROCM_VERSION" =~ 7\.9 ]]; then
    git checkout main
else
    git checkout v1.22.0
fi

git submodule update --init --recursive

case "$GPU_ARCH" in
    gfx103*) HIP_ARCHS=gfx1030 ;;
    gfx1100|gfx1101|gfx1102|gfx1103) HIP_ARCHS="$GPU_ARCH" ;;
    gfx1200|gfx1201) HIP_ARCHS="$GPU_ARCH" ;;
    *) HIP_ARCHS=gfx1100 ;;
esac

./build.sh --config Release \
           --build_wheel \
           --parallel $(( $(nproc) - 1 )) \
           --use_rocm --rocm_home="${ROCM_PATH}" \
           --use_migraphx --migraphx_home=/opt/rocm \
           --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=${HIP_ARCHS}

pip3 install build/Linux/Release/dist/*.whl

python3 <<'PY'
import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
PY
