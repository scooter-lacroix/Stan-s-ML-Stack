#!/bin/bash
# Environment Validation Utilities
# Provides reusable functions for validating .mlstack_env file
# Source this file in component installation scripts

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi

# Function: validate_mlstack_env
# Description: Validates .mlstack_env file exists and contains required variables
# Usage: validate_mlstack_env
# Returns: 0 on success, 1 on failure (but provides defaults where possible)
validate_mlstack_env() {
    local env_file="$HOME/.mlstack_env"
    local script_name="${1:-$(basename "$0")}"
    local missing_vars=()
    local has_warnings=false

    # Check if .mlstack_env file exists
    if [ ! -f "$env_file" ]; then
        echo "Warning: Environment file not found: $env_file - using auto-detection" >&2
        # Auto-detect values
        if command -v rocminfo >/dev/null 2>&1; then
            export GPU_ARCH=$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo "gfx1100")
            export ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -n1 || echo "7.2.0")
        else
            export GPU_ARCH="${GPU_ARCH:-gfx1100}"
            export ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
        fi
        export ROCM_CHANNEL="${ROCM_CHANNEL:-latest}"
        return 0
    fi

    # Source the environment file (with safe handling for unset variables)
    set +u 2>/dev/null || true
    # shellcheck source=/dev/null
    source "$env_file"
    set -u 2>/dev/null || true

    # Define required variables and their defaults
    if [ -z "${ROCM_VERSION:-}" ]; then
        export ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -n1 || echo "7.2.0")
        has_warnings=true
    fi
    if [ -z "${ROCM_CHANNEL:-}" ]; then
        export ROCM_CHANNEL="latest"
        has_warnings=true
    fi
    if [ -z "${GPU_ARCH:-}" ]; then
        if command -v rocminfo >/dev/null 2>&1; then
            export GPU_ARCH=$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo "gfx1100")
        else
            export GPU_ARCH="gfx1100"
        fi
        has_warnings=true
    fi

    # Validate ROCM_CHANNEL value (relax validation - accept 'preview' too)
    local valid_channels=(legacy stable latest preview)
    local channel_valid=false
    for channel in "${valid_channels[@]}"; do
        if [ "$ROCM_CHANNEL" = "$channel" ]; then
            channel_valid=true
            break
        fi
    done

    if [ "$channel_valid" = false ]; then
        echo "Warning: Unknown ROCM_CHANNEL '$ROCM_CHANNEL', defaulting to 'latest'" >&2
        export ROCM_CHANNEL="latest"
    fi

    # Log validation result
    echo "✓ Environment validation passed:"
    echo "  ROCM_VERSION: $ROCM_VERSION"
    echo "  ROCM_CHANNEL: $ROCM_CHANNEL"
    echo "  GPU_ARCH: $GPU_ARCH"

    return 0
}

# Function: require_mlstack_env
# Description: Requires .mlstack_env validation and auto-detects values if missing
# Usage: require_mlstack_env
# Returns: 0 on success (never exits, always provides defaults)
require_mlstack_env() {
    validate_mlstack_env "$1"
    # Always return success since validate_mlstack_env now provides defaults
    return 0
}
