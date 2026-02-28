#!/bin/bash
# Stan's ML Stack - Triton (ROCm) multi-channel installer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/lib/installer_guard.sh" ]; then
    # shellcheck source=lib/installer_guard.sh
    source "$SCRIPT_DIR/lib/installer_guard.sh"
fi
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi
if [ -f "$SCRIPT_DIR/env_validation_utils.sh" ]; then
    source "$SCRIPT_DIR/env_validation_utils.sh"
fi

DRY_RUN="${DRY_RUN:-false}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

if type require_mlstack_env >/dev/null 2>&1; then
    require_mlstack_env "$(basename "$0")" || true
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"
ROCM_CHANNEL="${ROCM_CHANNEL:-latest}"
ROCM_VERSION="${ROCM_VERSION:-7.2}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"

if [ -x "$PYTHON_BIN" ]; then
    :
else
    PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    print_error "Python interpreter not found: ${MLSTACK_PYTHON_BIN:-python3}"
    exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
    print_section "Triton ROCm Dry Run"
    print_step "[DRY RUN] Python: $PYTHON_BIN"
    print_step "[DRY RUN] ROCm channel/version: $ROCM_CHANNEL / $ROCM_VERSION"
    print_step "[DRY RUN] GPU arch: $GPU_ARCH"
    print_step "[DRY RUN] Would verify ROCm PyTorch contract before and after install"
    print_step "[DRY RUN] Would clone ROCm/triton and install with --no-deps to avoid resolver conflicts"
    print_step "[DRY RUN] Would run Triton + ROCm runtime verification"
    exit 0
fi

assert_no_cuda_packages() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import subprocess
import sys

blocked = []
try:
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    raise SystemExit(0)

for line in out.splitlines():
    name = line.split("==", 1)[0].strip().lower()
    if (
        name.startswith("nvidia-")
        or name in {"pytorch-cuda", "torch-cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder"}
        or name.startswith("cupy-cuda")
    ):
        blocked.append(name)

if blocked:
    print("Detected disallowed CUDA/NVIDIA packages:", ", ".join(sorted(set(blocked))))
    raise SystemExit(1)
PY
}

pip_install_with_compat() {
    if "$PYTHON_BIN" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
        "$PYTHON_BIN" -m pip install --break-system-packages "$@"
    else
        "$PYTHON_BIN" -m pip install "$@"
    fi
}

if declare -f mlstack_assert_python_supported >/dev/null 2>&1; then
    mlstack_assert_python_supported "$PYTHON_BIN"
fi

if declare -f mlstack_assert_rocm_torch >/dev/null 2>&1; then
    if ! mlstack_assert_rocm_torch "$PYTHON_BIN"; then
        if declare -f mlstack_install_rocm_torch_stack >/dev/null 2>&1; then
            print_warning "ROCm torch contract failed; repairing torch stack before Triton install..."
            mlstack_install_rocm_torch_stack "$PYTHON_BIN" "$ROCM_VERSION" "$ROCM_CHANNEL" "triton"
            mlstack_assert_rocm_torch "$PYTHON_BIN"
        else
            print_error "ROCm torch validation failed and installer guard fallback is unavailable."
            exit 1
        fi
    fi
fi

assert_no_cuda_packages "$PYTHON_BIN"

TRITON_STABLE_BRANCH="release/internal/3.6.x"
TRITON_PREVIEW_BRANCH="main_perf"
TRITON_MLIR_BRANCH="release/internal/3.6.x"

TMP_DIR="${TMPDIR:-/tmp}/triton-rocm"
print_section "Preparing Triton Installation for ROCm ${ROCM_VERSION} (${ROCM_CHANNEL})"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

print_step "Cloning ROCm Triton repository..."
git clone https://github.com/ROCm/triton.git "$TMP_DIR/triton"
cd "$TMP_DIR/triton"

if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    print_warning "Using preview branch ${TRITON_PREVIEW_BRANCH}"
    git checkout "$TRITON_PREVIEW_BRANCH"
else
    print_step "Attempting ROCm-optimized branch: ${TRITON_MLIR_BRANCH}"
    if ! git checkout "$TRITON_MLIR_BRANCH" 2>/dev/null; then
        print_warning "ROCm-optimized branch unavailable, using stable branch ${TRITON_STABLE_BRANCH}"
        git checkout "$TRITON_STABLE_BRANCH"
    fi
fi

if [ -d "python" ] && [ -f "python/setup.py" ]; then
    cd python
fi

export GPU_ARCHS="$GPU_ARCH"
export TRITON_ROCM=1
export MAX_JOBS="$(( $(nproc) - 1 ))"

if command -v ccache >/dev/null 2>&1; then
    export TRITON_BUILD_WITH_CCACHE=true
else
    export TRITON_BUILD_WITH_CCACHE=false
fi

print_step "Installing Triton build prerequisites (pybind11/cmake/ninja/build tooling)..."
pip_install_with_compat --upgrade --no-cache-dir \
    pip setuptools wheel packaging pybind11 cmake ninja

print_step "Installing Triton from source with ROCm-safe flags..."
pip_install_with_compat --no-build-isolation --no-deps .

print_section "Verifying Triton ROCm Installation"
"$PYTHON_BIN" - <<'PY'
import torch
import triton

print("Triton version:", getattr(triton, "__version__", "unknown"))
print("PyTorch version:", torch.__version__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("torch.cuda.is_available():", torch.cuda.is_available())

if not getattr(torch.version, "hip", None):
    raise SystemExit("Installed torch is not ROCm-enabled")
if getattr(torch.version, "cuda", None):
    raise SystemExit("Installed torch reports CUDA, expected ROCm-only")
if not torch.cuda.is_available():
    raise SystemExit("ROCm torch has no visible GPU device")
PY

if declare -f mlstack_assert_rocm_torch >/dev/null 2>&1; then
    mlstack_assert_rocm_torch "$PYTHON_BIN"
fi
assert_no_cuda_packages "$PYTHON_BIN"

print_success "Triton ROCm installation completed successfully"
