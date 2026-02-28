#!/bin/bash
# Stan's ML Stack - MIGraphX ROCm installer (channel-aware)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD_LIB="$SCRIPT_DIR/lib/installer_guard.sh"
if [ ! -f "$GUARD_LIB" ]; then
    printf '[mlstack][ERROR] Missing installer guard library: %s\n' "$GUARD_LIB" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$GUARD_LIB"

# Source utility scripts if available
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi
if [ -f "$SCRIPT_DIR/env_validation_utils.sh" ]; then
    source "$SCRIPT_DIR/env_validation_utils.sh"
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

ensure_rocm_python_contract() {
    local py_cmd="$1"
    local component="${2:-migraphx}"

    if [ -x "$py_cmd" ]; then
        :
    else
        py_cmd="$(command -v "$py_cmd" 2>/dev/null || true)"
    fi
    if [ -z "$py_cmd" ] || [ ! -x "$py_cmd" ]; then
        print_error "Python interpreter not found for ${component}: ${1}"
        return 1
    fi

    if declare -f mlstack_assert_rocm_torch >/dev/null 2>&1; then
        if ! mlstack_assert_rocm_torch "$py_cmd"; then
            print_error "ROCm PyTorch contract failed for ${component}"
            return 1
        fi
    fi

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
    require_mlstack_env "$(basename "$0")"
fi

print_section "Installing MIGraphX for ROCm ${ROCM_VERSION}"

PKG_MANAGER="$(mlstack_detect_pkg_manager)"
case "$PKG_MANAGER" in
    pacman)
        REQUIRED_PKGS=(migraphx)
        OPTIONAL_PKGS=(python3-migraphx half)
        ;;
    dnf|yum|zypper|apt|apk)
        REQUIRED_PKGS=(migraphx migraphx-dev)
        OPTIONAL_PKGS=(python3-migraphx half)
        ;;
    *)
        REQUIRED_PKGS=(migraphx migraphx-dev)
        OPTIONAL_PKGS=(python3-migraphx half)
        ;;
esac

if [ "$DRY_RUN" = "true" ]; then
    print_step "[DRY RUN] Would refresh package manager metadata"
    print_step "[DRY RUN] Required MIGraphX packages ($PKG_MANAGER): ${REQUIRED_PKGS[*]}"
    print_step "[DRY RUN] Optional MIGraphX packages ($PKG_MANAGER): ${OPTIONAL_PKGS[*]}"
else
    ensure_rocm_python_contract "$PYTHON_BIN" "migraphx preflight"

    # Update package list
    mlstack_pm_update

    # Install distro-specific required packages first.
    mlstack_pm_install "${REQUIRED_PKGS[@]}"

    # Optional packages vary across repos/distributions; skip cleanly when absent.
    for pkg in "${OPTIONAL_PKGS[@]}"; do
        if mlstack_pm_has_package "$pkg"; then
            mlstack_pm_install "$pkg"
        else
            mlstack_log_warn "Skipping unavailable optional package for $PKG_MANAGER: $pkg"
        fi
    done
fi

# Ensure no CUDA PyTorch is installed during MIGraphX setup
if [ "${MLSTACK_SKIP_TORCH_INSTALL:-0}" = "1" ]; then
    print_step "Skipping PyTorch installation (MLSTACK_SKIP_TORCH_INSTALL=1)"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
fi

if [ "$DRY_RUN" = "false" ]; then
    print_section "Verifying MIGraphX installation"
    $PYTHON_BIN - <<'PY'
import sys
import os
# Add potential migraphx paths
rocm_lib = "/opt/rocm/lib"
if os.path.exists(rocm_lib) and rocm_lib not in sys.path:
    sys.path.append(rocm_lib)
    for item in os.listdir(rocm_lib):
        if item.startswith("python") and os.path.isdir(os.path.join(rocm_lib, item, "site-packages")):
            sys.path.append(os.path.join(rocm_lib, item, "site-packages"))

try:
    import migraphx
    if hasattr(migraphx, "get_target"):
        target = migraphx.get_target("gpu")
        print("✓ MIGraphX GPU target:", target)
    print("✓ MIGraphX available: OK")
except Exception as exc:
    print("✗ MIGraphX python binding import failed:", exc)
    sys.exit(1)
PY

    ensure_rocm_python_contract "$PYTHON_BIN" "migraphx post-install"
fi
