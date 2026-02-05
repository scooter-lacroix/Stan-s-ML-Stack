#!/bin/bash
# Stan's ML Stack - Triton (ROCm) multi-channel installer

set -euo pipefail

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

# Require and validate .mlstack_env
if type require_mlstack_env >/dev/null 2>&1; then
    require_mlstack_env "$(basename "$0")" || true
fi

ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-gfx1100}

# Define stable tags/branches for reproducibility
TRITON_STABLE_BRANCH="release/internal/3.6.x" # Updated to match Pytorch requirement
TRITON_PREVIEW_BRANCH="main_perf"
TRITON_MLIR_BRANCH="release/internal/3.6.x"

print_section "Preparing Triton Installation for ROCm $ROCM_VERSION ($ROCM_CHANNEL)"

TMP_DIR=${TMPDIR:-/tmp}/triton-rocm
sudo rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

if [ "$DRY_RUN" = "false" ]; then
    cd "$TMP_DIR"
fi

execute_command "git clone https://github.com/ROCm/triton.git \"$TMP_DIR/triton\"" "Cloning Triton repository"

if [ "$DRY_RUN" = "false" ]; then
    cd "$TMP_DIR/triton"
fi

# Pin to stable tag or use preview branch with validation
if [ "$ROCM_CHANNEL" = "preview" ] || [[ "$GPU_ARCH" =~ ^gfx12 ]]; then
    print_warning "Using preview branch $TRITON_PREVIEW_BRANCH (may be unstable)"
    execute_command "git checkout $TRITON_PREVIEW_BRANCH" "Checking out preview branch"
else
    # Try triton-mlir branch first for ROCm, fall back to stable tag
    print_step "Attempting to use ROCm-optimized branch: $TRITON_MLIR_BRANCH"
    if [ "$DRY_RUN" = "false" ] && git checkout "$TRITON_MLIR_BRANCH" 2>/dev/null; then
        print_success "Successfully checked out ROCm-optimized branch"
    elif [ "$DRY_RUN" = "true" ]; then
        execute_command "git checkout $TRITON_MLIR_BRANCH" "Checking out ROCm-optimized branch"
    else
        print_warning "ROCm-optimized branch not available, using stable tag: $TRITON_STABLE_BRANCH"
        execute_command "git checkout $TRITON_STABLE_BRANCH" "Checking out stable branch"
    fi
fi

# Determine if we need to cd into python directory
if [ -d "python" ] && [ -f "python/setup.py" ]; then
    print_step "Moving to python directory..."
    cd python
fi

export GPU_ARCHS="$GPU_ARCH"
export TRITON_ROCM=1

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"
python3() {
    "$PYTHON_BIN" "$@"
}

if [ "$DRY_RUN" = "false" ]; then
    if command -v ccache >/dev/null 2>&1; then
        export TRITON_BUILD_WITH_CCACHE=true
    else
        export TRITON_BUILD_WITH_CCACHE=false
    fi
fi

export MAX_JOBS=$(( $(nproc) - 1 ))

print_step "Current directory: $(pwd)"
ls -F

execute_command "python3 -m pip install --break-system-packages ." "Building and installing Triton"

if [ "$DRY_RUN" = "false" ]; then
    print_section "Verifying Triton installation"
    python3 <<'PY'
try:
    import triton
    print("Triton version:", triton.__version__)
    print("✓ Triton imported successfully")
except Exception as e:
    print(f"✗ Failed to import triton: {e}")
PY
fi
