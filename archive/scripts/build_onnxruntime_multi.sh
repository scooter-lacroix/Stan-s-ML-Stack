#!/bin/bash
# Stan's ML Stack - ONNX Runtime Build (ROCm multi-channel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD_LIB="$SCRIPT_DIR/lib/installer_guard.sh"
if [ ! -f "$GUARD_LIB" ]; then
    printf '[mlstack][ERROR] Missing installer guard library: %s\n' "$GUARD_LIB" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$GUARD_LIB"

EXPLICIT_MLSTACK_PYTHON_BIN="${MLSTACK_PYTHON_BIN:-}"
if [ -f "$HOME/.mlstack_env" ]; then
    set +u 2>/dev/null || true
    source "$HOME/.mlstack_env"
    set -u 2>/dev/null || true
fi
if [ -n "$EXPLICIT_MLSTACK_PYTHON_BIN" ]; then
    MLSTACK_PYTHON_BIN="$EXPLICIT_MLSTACK_PYTHON_BIN"
    export MLSTACK_PYTHON_BIN
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-false}"
NON_INTERACTIVE_MODE=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            cat <<'EOF'
Usage: build_onnxruntime_multi.sh [--dry-run]

Options:
  --dry-run   Show planned ONNX Runtime ROCm build actions without modifying the system
  -h, --help  Show this help text
EOF
            exit 0
            ;;
    esac
done

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROCM_VERSION=${ROCM_VERSION:-$(cat /opt/rocm/.info/version 2>/dev/null || rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs || echo 7.2)}
echo "Building ONNX Runtime with ROCm version: ${ROCM_VERSION}"
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
GPU_ARCH=${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo gfx1100)}

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY-RUN] build_onnxruntime_multi.sh"
    echo "[DRY-RUN] ROCM_VERSION=${ROCM_VERSION} ROCM_PATH=${ROCM_PATH} GPU_ARCH=${GPU_ARCH}"
    echo "[DRY-RUN] Would first try prebuilt onnxruntime-rocm wheel install with ${PYTHON_BIN}."
    echo "[DRY-RUN] Would install distro-specific build dependencies (cmake/protobuf/migraphx/eigen)."
    echo "[DRY-RUN] Would clone microsoft/onnxruntime (v1.20.1), patch CMake ROCm version regex, and build ROCm wheel with MIGraphX."
    echo "[DRY-RUN] Would force package-registry isolation and prefer preinstalled Eigen when available."
    echo "[DRY-RUN] Would install built wheel and verify ONNX Runtime ROCm providers."
    exit 0
fi

if [ "${MLSTACK_BATCH_MODE:-0}" = "1" ] || [ -n "${RUSTY_STACK:-}" ] || [ ! -t 0 ] || [ ! -t 1 ]; then
    NON_INTERACTIVE_MODE=1
fi

mlstack_assert_python_supported "$PYTHON_BIN"

verify_onnxruntime_rocm_provider() {
    "$PYTHON_BIN" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", providers)

if "ROCMExecutionProvider" not in providers:
    raise SystemExit("ROCMExecutionProvider is missing. ONNX Runtime ROCm install is invalid.")
PY
}

try_prebuilt_onnxruntime_rocm() {
    mlstack_log_info "Trying prebuilt onnxruntime-rocm wheel install with ${PYTHON_BIN}."
    if ! mlstack_pip_install "$PYTHON_BIN" --upgrade --prefer-binary onnxruntime-rocm; then
        mlstack_log_warn "Prebuilt onnxruntime-rocm wheel install failed; falling back to source build."
        return 1
    fi

    if ! verify_onnxruntime_rocm_provider; then
        mlstack_log_warn "onnxruntime-rocm import verification failed after wheel install; falling back to source build."
        return 1
    fi

    mlstack_log_info "Prebuilt onnxruntime-rocm wheel installation succeeded."
    return 0
}

if try_prebuilt_onnxruntime_rocm; then
    exit 0
fi

can_run_privileged_pkg_ops() {
    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi
    if [ "$NON_INTERACTIVE_MODE" -eq 1 ]; then
        command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1
        return $?
    fi
    return 0
}

if can_run_privileged_pkg_ops; then
    mlstack_pm_update
    mlstack_pm_install cmake protobuf-compiler
else
    mlstack_log_warn "Non-interactive mode without passwordless sudo/root; skipping privileged package installs."
    mlstack_log_warn "Proceeding with source-build fallback using currently installed system dependencies."
fi

# MIGraphX helper packages differ across distros/repos; install what is available.
if can_run_privileged_pkg_ops; then
    if mlstack_pm_has_package migraphx; then
        mlstack_pm_install migraphx
    fi
    if mlstack_pm_has_package migraphx-dev; then
        mlstack_pm_install migraphx-dev
    fi
    if mlstack_pm_has_package half; then
        mlstack_pm_install half
    fi
fi

# Prefer system Eigen to avoid upstream FetchContent hash drift failures.
PKG_MANAGER="$(mlstack_detect_pkg_manager)"
EIGEN_PATH="/usr/include/eigen3"
USE_PREINSTALLED_EIGEN=0
if can_run_privileged_pkg_ops; then
    case "$PKG_MANAGER" in
        apt)
            mlstack_pm_install libeigen3-dev || true
            ;;
        dnf|yum|zypper)
            if mlstack_pm_has_package eigen3-devel; then
                mlstack_pm_install eigen3-devel || true
            elif mlstack_pm_has_package eigen; then
                mlstack_pm_install eigen || true
            fi
            ;;
        pacman|apk)
            if mlstack_pm_has_package eigen; then
                mlstack_pm_install eigen || true
            fi
            ;;
    esac
fi

if [ -f "/usr/include/eigen3/Eigen/Core" ]; then
    EIGEN_PATH="/usr/include/eigen3"
    USE_PREINSTALLED_EIGEN=1
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists eigen3; then
    EIGEN_PATH="$(pkg-config --variable=includedir eigen3 2>/dev/null || echo /usr/include/eigen3)"
    if [ -f "${EIGEN_PATH}/Eigen/Core" ]; then
        USE_PREINSTALLED_EIGEN=1
    fi
fi

if [ "$USE_PREINSTALLED_EIGEN" -eq 1 ]; then
    echo "Using preinstalled Eigen at: ${EIGEN_PATH}"
else
    echo "⚠ WARNING: Preinstalled Eigen headers not found; ONNX Runtime may attempt FetchContent download."
fi

WORKDIR=${TMPDIR:-/tmp}/onnxruntime-rocm
rm -rf "$WORKDIR"
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
export CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON
export CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON
export CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
export CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
export CMAKE_IGNORE_PREFIX_PATH="${CONDA_PREFIX:-$HOME/anaconda3}:${CONDA_PREFIX_1:-}:${CONDA_PREFIX_2:-}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

# Force use of system CMake
CMAKE_BIN=$(which cmake || echo "/usr/bin/cmake")

# Run build with CMake policy version minimum set to handle old dependencies
# This fixes: "Compatibility with CMake < 3.5 has been removed from CMake"
build_args=(
    --config Release
    --cmake_path "$CMAKE_BIN"
    --build_wheel
    --parallel $(( $(nproc) - 1 ))
    --use_rocm --rocm_home "${ROCM_PATH}"
    --rocm_version "${ORT_ROCM_VERSION}"
    --use_migraphx --migraphx_home /opt/rocm
)

if [ "$USE_PREINSTALLED_EIGEN" -eq 1 ]; then
    build_args+=(--use_preinstalled_eigen --eigen_path "${EIGEN_PATH}")
fi

build_args+=(
    --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=${HIP_ARCHS}
    --cmake_extra_defines onnxruntime_USE_EXTERNAL_ABSEIL=OFF
    --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_re2=ON
    --cmake_extra_defines CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON
    --cmake_extra_defines CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON
    --cmake_extra_defines CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    --cmake_extra_defines CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
    --cmake_extra_defines CMAKE_IGNORE_PATH="$HOME/anaconda3/lib/cmake;$HOME/anaconda3/lib;$HOME/anaconda3;${CONDA_PREFIX:-}"
    --cmake_extra_defines CMAKE_IGNORE_PREFIX_PATH="${CONDA_PREFIX:-}"
    --cmake_extra_defines CMAKE_PREFIX_PATH="/opt/rocm"
    --cmake_extra_defines re2_DIR=RE2_DIR-NOTFOUND
    --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5
)

if [ "$USE_PREINSTALLED_EIGEN" -eq 1 ]; then
    build_args+=(--cmake_extra_defines FETCHCONTENT_SOURCE_DIR_EIGEN="${EIGEN_PATH}")
    build_args+=(--cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER)
fi

build_args+=(--allow_running_as_root)
./build.sh "${build_args[@]}"

python3 -m pip uninstall -y onnxruntime onnxruntime-rocm onnxruntime-gpu || true
mlstack_pip_install "$PYTHON_BIN" build/Linux/Release/dist/*.whl

verify_onnxruntime_rocm_provider
