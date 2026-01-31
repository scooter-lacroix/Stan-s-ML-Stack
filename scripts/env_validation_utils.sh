#!/bin/bash
# Environment Validation Utilities
# Provides reusable functions for validating .mlstack_env file
# Source this file in component installation scripts

# Function: validate_mlstack_env
# Description: Validates .mlstack_env file exists and contains required variables
# Usage: validate_mlstack_env
# Returns: 0 on success, 1 on failure
validate_mlstack_env() {
    local env_file="$HOME/.mlstack_env"
    local script_name="${1:-$(basename "$0")}"
    local missing_vars=()

    # Check if .mlstack_env file exists
    if [ ! -f "$env_file" ]; then
        echo "ERROR: Required environment file not found: $env_file" >&2
        echo "" >&2
        echo "The $script_name script requires the .mlstack_env file to be properly configured." >&2
        echo "" >&2
        echo "This file is created during ROCm installation. Please ensure ROCm is installed:" >&2
        echo "  ./scripts/install_rocm.sh" >&2
        echo "" >&2
        echo "Or create it manually with the following variables:" >&2
        echo "  ROCM_VERSION (e.g., 6.4.3, 7.1, 7.2)" >&2
        echo "  ROCM_CHANNEL (e.g., legacy, stable, latest)" >&2
        echo "  GPU_ARCH (e.g., gfx1100, gfx1030)" >&2
        echo "" >&2
        echo "Example $env_file:" >&2
        echo "  export ROCM_VERSION=7.2" >&2
        echo "  export ROCM_CHANNEL=latest" >&2
        echo "  export GPU_ARCH=gfx1100" >&2
        return 1
    fi

    # Source the environment file
    # shellcheck source=/dev/null
    source "$env_file"

    # Define required variables
    local required_vars=("ROCM_VERSION" "ROCM_CHANNEL" "GPU_ARCH")

    # Check each required variable
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done

    # If any variables are missing, provide helpful error
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "ERROR: Required environment variables not set in $env_file" >&2
        echo "" >&2
        echo "Missing variables:" >&2
        for var in "${missing_vars[@]}"; do
            echo "  - $var" >&2
        done
        echo "" >&2
        echo "Please ensure the following are set in $env_file:" >&2
        echo "  export ROCM_VERSION=<version>  # e.g., 6.4.3, 7.1, 7.2" >&2
        echo "  export ROCM_CHANNEL=<channel>  # e.g., legacy, stable, latest" >&2
        echo "  export GPU_ARCH=<architecture>  # e.g., gfx1100, gfx1030" >&2
        echo "" >&2
        echo "To detect your GPU architecture, run:" >&2
        echo "  rocminfo | grep -o 'gfx[0-9]*' | head -n1" >&2
        return 1
    fi

    # Validate ROCM_VERSION format
    if [[ ! "$ROCM_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && [[ ! "$ROCM_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: Invalid ROCM_VERSION format: '$ROCM_VERSION'" >&2
        echo "" >&2
        echo "Expected format: X.Y.Z (e.g., 6.4.3, 7.1.0, 7.2.0)" >&2
        echo "Detected: $ROCM_VERSION" >&2
        return 1
    fi

    # Validate ROCM_CHANNEL value
    local valid_channels=(legacy stable latest)
    local channel_valid=false
    for channel in "${valid_channels[@]}"; do
        if [ "$ROCM_CHANNEL" = "$channel" ]; then
            channel_valid=true
            break
        fi
    done

    if [ "$channel_valid" = false ]; then
        echo "ERROR: Invalid ROCM_CHANNEL value: '$ROCM_CHANNEL'" >&2
        echo "" >&2
        echo "Valid channels: ${valid_channels[*]}" >&2
        echo "Detected: $ROCM_CHANNEL" >&2
        return 1
    fi

    # Validate GPU_ARCH format
    if [[ ! "$GPU_ARCH" =~ ^gfx[0-9]+$ ]]; then
        echo "ERROR: Invalid GPU_ARCH format: '$GPU_ARCH'" >&2
        echo "" >&2
        echo "Expected format: gfxXXX (e.g., gfx1100, gfx1030)" >&2
        echo "Detected: $GPU_ARCH" >&2
        echo "" >&2
        echo "To detect your GPU architecture, run:" >&2
        echo "  rocminfo | grep -o 'gfx[0-9]*' | head -n1" >&2
        return 1
    fi

    # Log successful validation
    echo "✓ Environment validation passed:"
    echo "  ROCM_VERSION: $ROCM_VERSION"
    echo "  ROCM_CHANNEL: $ROCM_CHANNEL"
    echo "  GPU_ARCH: $GPU_ARCH"

    return 0
}

# Function: require_mlstack_env
# Description: Requires .mlstack_env and exits if validation fails
# Usage: require_mlstack_env
# Returns: 0 on success, exits script on failure
require_mlstack_env() {
    if ! validate_mlstack_env "$1"; then
        exit 1
    fi
}
