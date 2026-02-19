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
    "$PYTHON_BIN" "$@"
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

if [ "$DRY_RUN" = "true" ]; then
    print_step "[DRY RUN] Would refresh package manager metadata"
    print_step "[DRY RUN] Would install system packages: migraphx migraphx-dev half python3-migraphx"
else
    # Update package list
    mlstack_pm_update

    # Install MIGraphX packages
    mlstack_pm_install migraphx migraphx-dev half python3-migraphx || mlstack_pm_install migraphx migraphx-dev half
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
    print("✓ MIGraphX available: OK")
except Exception as exc:
    print("✗ MIGraphX python binding import failed:", exc)
    sys.exit(1)
PY
fi
