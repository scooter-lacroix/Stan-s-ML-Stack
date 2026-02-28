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
TORCH_CHANNEL="${MLSTACK_TORCH_CHANNEL:-latest}"
CHANNEL_SET_BY_FLAG=false

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
        --channel)
            if [ $# -lt 2 ]; then
                mlstack_log_error "--channel requires one value: stable, latest, or nightly"
                exit 1
            fi
            TORCH_CHANNEL="$2"
            CHANNEL_SET_BY_FLAG=true
            shift
            ;;
        *) mlstack_log_warn "Ignoring unknown argument: $1" ;;
    esac
    shift
done

INSTALL_METHOD="${INSTALL_METHOD,,}"
case "$INSTALL_METHOD" in
    global|venv|auto) ;;
    *)
        mlstack_log_error "Invalid --method value: $INSTALL_METHOD (expected: global, venv, auto)"
        exit 1
        ;;
esac

TORCH_CHANNEL="$(mlstack_torch_channel_normalize "$TORCH_CHANNEL")"

if [ -f "$HOME/.mlstack_env" ]; then
    set +u
    # shellcheck source=/dev/null
    source "$HOME/.mlstack_env"
    set -u
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"
ROCM_VERSION="${ROCM_VERSION:-7.1.0}"

if [ "$CHANNEL_SET_BY_FLAG" = false ] && [ -t 0 ] && [ "${MLSTACK_BATCH_MODE:-0}" != "1" ]; then
    printf '\nPyTorch ROCm channel:\n'
    printf '  1) Stable\n'
    printf '  2) Latest (stable fallback to nightly)\n'
    printf '  3) Nightly\n'
    read -r -p "Choose channel (1-3) [2]: " CHANNEL_CHOICE
    CHANNEL_CHOICE="${CHANNEL_CHOICE:-2}"
    case "$CHANNEL_CHOICE" in
        1) TORCH_CHANNEL="stable" ;;
        2) TORCH_CHANNEL="latest" ;;
        3) TORCH_CHANNEL="nightly" ;;
        *) TORCH_CHANNEL="latest" ;;
    esac
fi

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
    if [ "$CHANNEL_SET_BY_FLAG" = true ]; then
        mlstack_log_info "Dry run: torch channel is '$TORCH_CHANNEL' (from --channel)"
    else
        mlstack_log_info "Dry run: torch channel is '$TORCH_CHANNEL'"
    fi
    case "$INSTALL_METHOD" in
        global) mlstack_log_info "Dry run: would install with interpreter $PYTHON_BIN" ;;
        venv) mlstack_log_info "Dry run: would prepare venv $(mlstack_default_venv_base)/pytorch_rocm" ;;
        auto) mlstack_log_info "Dry run: would try global install with $PYTHON_BIN, then fallback to venv on failure" ;;
    esac
    selected="$(mlstack_select_torch_index "$PYTHON_BIN" "$ROCM_VERSION" "$TORCH_CHANNEL" || true)"
    if [ -n "$selected" ]; then
        IFS='|' read -r dry_index dry_series dry_channel <<< "$selected"
        mlstack_log_info "Dry run: resolved index=$dry_index (series=$dry_series, effective_channel=$dry_channel)"
    else
        mlstack_log_warn "Dry run: no compatible wheel index resolved for current Python/ROCm."
    fi
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
mlstack_log_info "Requested torch channel: $TORCH_CHANNEL"

install_torch_stack() {
    local python_bin="$1"
    mlstack_install_rocm_torch_stack "$python_bin" "$ROCM_VERSION" "$TORCH_CHANNEL" "pytorch"
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

mlstack_guard_python_env "pytorch" "$TARGET_PYTHON" --purge
mlstack_assert_rocm_torch "$TARGET_PYTHON"

"$TARGET_PYTHON" - <<'PY'
import torch
x = torch.ones(128, device="cuda")
print("Device:", torch.cuda.get_device_name(0))
print("Sanity sum:", x.sum().item())
PY
