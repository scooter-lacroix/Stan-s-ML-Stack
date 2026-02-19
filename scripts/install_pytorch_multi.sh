#!/usr/bin/env bash
# Stan's ML Stack - PyTorch ROCm Installer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD_LIB="$SCRIPT_DIR/lib/installer_guard.sh"

if [ ! -f "$GUARD_LIB" ]; then
    printf '[mlstack][ERROR] Missing installer guard library: %s\n' "$GUARD_LIB" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$GUARD_LIB"

DRY_RUN=false
INSTALL_METHOD="${MLSTACK_INSTALL_METHOD:-auto}"
METHOD_SET_BY_FLAG=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --method)
            if [ $# -lt 2 ]; then
                mlstack_log_error "--method requires one value: global, venv, or auto"
                exit 1
            fi
            INSTALL_METHOD="$2"
            METHOD_SET_BY_FLAG=true
            shift
            ;;
        *) mlstack_log_warn "Ignoring unknown argument: $1" ;;
    esac
    shift
done

case "${INSTALL_METHOD,,}" in
    global|venv|auto) INSTALL_METHOD="${INSTALL_METHOD,,}" ;;
    *)
        mlstack_log_error "Invalid --method value: $INSTALL_METHOD (expected: global, venv, auto)"
        exit 1
        ;;
esac

if [ "$METHOD_SET_BY_FLAG" = false ] && [ -t 0 ] && [ "${MLSTACK_BATCH_MODE:-0}" != "1" ]; then
    printf '\nPyTorch installation method:\n'
    printf '  1) Global installation\n'
    printf '  2) Virtual environment installation\n'
    printf '  3) Auto (try global then fallback to venv)\n'
    read -r -p "Choose installation method (1-3) [3]: " INSTALL_CHOICE
    INSTALL_CHOICE="${INSTALL_CHOICE:-3}"
    case "$INSTALL_CHOICE" in
        1) INSTALL_METHOD="global" ;;
        2) INSTALL_METHOD="venv" ;;
        3|*) INSTALL_METHOD="auto" ;;
    esac
fi

if [ -f "$HOME/.mlstack_env" ]; then
    set +u
    # shellcheck source=/dev/null
    source "$HOME/.mlstack_env"
    set -u
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"
ROCM_VERSION="${ROCM_VERSION:-7.2}"

if [[ "$ROCM_VERSION" =~ ^([0-9]+)\.([0-9]+) ]]; then
    ROCM_MM="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
else
    ROCM_MM="7.2"
fi

case "$ROCM_VERSION" in
    6.2*) INDEX_URL="https://download.pytorch.org/whl/rocm6.2" ;;
    6.3*) INDEX_URL="https://download.pytorch.org/whl/rocm6.3" ;;
    6.4*) INDEX_URL="https://download.pytorch.org/whl/rocm6.4" ;;
    7.*) INDEX_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_MM}/" ;;
    *) INDEX_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/" ;;
esac

if [ "$DRY_RUN" = true ]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
        mlstack_log_error "Python interpreter not found: $PYTHON_BIN"
        exit 1
    fi
    if [ "$METHOD_SET_BY_FLAG" = true ]; then
        mlstack_log_info "Dry run: install method is '$INSTALL_METHOD' (from --method)"
    else
        mlstack_log_info "Dry run: install method is '$INSTALL_METHOD'"
    fi
    case "$INSTALL_METHOD" in
        global) mlstack_log_info "Dry run: would install with interpreter $PYTHON_BIN" ;;
        venv) mlstack_log_info "Dry run: would prepare venv $(mlstack_default_venv_base)/pytorch_rocm" ;;
        auto) mlstack_log_info "Dry run: would try global install with $PYTHON_BIN, then fallback to venv on failure" ;;
    esac
    mlstack_log_info "Dry run: would install torch/torchvision/torchaudio/triton from $INDEX_URL"
    mlstack_log_info "Dry run: would purge NVIDIA packages and verify ROCm torch"
    mlstack_log_info "Dry run complete. No installation actions were performed."
    exit 0
fi

mlstack_assert_python_supported "$PYTHON_BIN"
TARGET_PYTHON="$PYTHON_BIN"
TARGET_DESC="global interpreter"
VENV_NAME="pytorch_rocm"

if [ "$INSTALL_METHOD" = "venv" ]; then
    VENV_DIR="$(mlstack_prepare_venv "$VENV_NAME" "$PYTHON_BIN")"
    TARGET_PYTHON="$(mlstack_venv_python "$VENV_NAME")"
    TARGET_DESC="virtualenv $VENV_DIR"
elif [ "$INSTALL_METHOD" = "auto" ]; then
    TARGET_DESC="global interpreter (auto mode)"
fi

mlstack_log_info "Using Python: $TARGET_PYTHON ($TARGET_DESC)"
mlstack_log_info "Using ROCm package index: $INDEX_URL"

install_torch_stack() {
    local python_bin="$1"
    mlstack_pip_install "$python_bin" --upgrade pip setuptools wheel
    mlstack_pip_install "$python_bin" --index-url "$INDEX_URL" --upgrade torch torchvision torchaudio triton
}

if [ "$INSTALL_METHOD" = "auto" ]; then
    if ! install_torch_stack "$TARGET_PYTHON"; then
        mlstack_log_warn "Auto mode global install failed; falling back to venv."
        VENV_DIR="$(mlstack_prepare_venv "$VENV_NAME" "$PYTHON_BIN")"
        TARGET_PYTHON="$(mlstack_venv_python "$VENV_NAME")"
        TARGET_DESC="virtualenv $VENV_DIR (auto fallback)"
        mlstack_log_info "Using Python: $TARGET_PYTHON ($TARGET_DESC)"
        install_torch_stack "$TARGET_PYTHON"
    fi
else
    install_torch_stack "$TARGET_PYTHON"
fi

mlstack_purge_nvidia_packages "$TARGET_PYTHON"
if mlstack_has_nvidia_packages "$TARGET_PYTHON"; then
    mlstack_log_error "NVIDIA Python packages are still present after cleanup."
    "$TARGET_PYTHON" -m pip list --format=freeze | awk -F'==' 'BEGIN{IGNORECASE=1}$1 ~ /^nvidia-/ || $1 ~ /^pytorch-cuda$/ || $1 ~ /^cuda-python$/ || $1 ~ /^cupy-cuda/ {print $0}' >&2
    exit 1
fi

mlstack_assert_rocm_torch "$TARGET_PYTHON"

"$TARGET_PYTHON" - <<'PY'
import torch
x = torch.ones(128, device="cuda")
print("Device:", torch.cuda.get_device_name(0))
print("Sanity sum:", x.sum().item())
PY
