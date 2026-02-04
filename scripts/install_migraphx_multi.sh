#!/bin/bash
# Stan's ML Stack - MIGraphX ROCm installer (channel-aware)

set -euo pipefail

# Source utility scripts if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# Update package list
execute_command "sudo apt-get update" "Updating package repositories"

# Install MIGraphX packages
execute_command "sudo apt-get install -y migraphx migraphx-dev half python3-migraphx || sudo apt-get install -y migraphx migraphx-dev half" "Installing MIGraphX packages"

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
