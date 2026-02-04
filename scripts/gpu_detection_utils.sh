#!/bin/bash
# GPU Detection Validation Utilities
# Provides reusable functions for GPU detection and validation
# Source this file in component installation scripts

# Function: validate_gpu_detection
# Description: Validates that GPU detection succeeded and provides helpful error messages
# Usage: validate_gpu_detection
# Returns: 0 on success, 1 on failure
validate_gpu_detection() {
    local gpu_arch="${1:-$GPU_ARCH}"
    local script_name="${2:-$(basename "$0")}"

    # Check if GPU architecture was detected
    if [ -z "$gpu_arch" ] || [ "$gpu_arch" = "gfx1100" ] && [ -z "${GPU_ARCH:-}" ]; then
        # gfx1100 might be a fallback, verify it's real
        if ! rocminfo >/dev/null 2>&1; then
            echo "ERROR: GPU detection failed in $script_name" >&2
            echo "" >&2
            echo "GPU detection failed for the following reasons:" >&2
            echo "  - rocminfo command not found or not executable" >&2
            echo "  - No AMD GPU detected on this system" >&2
            echo "  - AMDGPU driver not loaded or not properly installed" >&2
            echo "" >&2
            echo "Troubleshooting steps:" >&2
            echo "  1. Verify AMD GPU is installed: lspci | grep -i vga" >&2
            echo "  2. Check AMDGPU driver: lsmod | grep amdgpu" >&2
            echo "  3. Verify ROCm installation: which rocminfo" >&2
            echo "  4. Test GPU detection: rocminfo" >&2
            echo "" >&2
            echo "For manual installation, see:" >&2
            echo "  https://rocm.docs.amd.com/en/latest/install/install.html" >&2
            return 1
        fi
    fi

    # Validate GPU architecture format
    if [[ ! "$gpu_arch" =~ ^gfx[0-9]+$ ]]; then
        echo "ERROR: Invalid GPU architecture format: '$gpu_arch'" >&2
        echo "" >&2
        echo "Expected format: gfxXXX (e.g., gfx1100, gfx1030)" >&2
        echo "Detected: $gpu_arch" >&2
        echo "" >&2
        echo "This may indicate a problem with GPU detection." >&2
        echo "Please run: rocminfo" >&2
        return 1
    fi

    # Log successful detection
    echo "✓ GPU detection successful: $gpu_arch" >&2

    return 0
}

# Function: detect_gpu_architecture
# Description: Detects GPU architecture with validation
# Usage: GPU_ARCH=$(detect_gpu_architecture) || exit 1
# Returns: GPU architecture string (e.g., gfx1100) or exits on failure
detect_gpu_architecture() {
    local gpu_arch

    # Try to detect GPU architecture
    gpu_arch=$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1)

    # Check if detection succeeded
    if [ -z "$gpu_arch" ]; then
        echo "ERROR: Unable to detect GPU architecture" >&2
        echo "" >&2
        echo "Possible reasons:" >&2
        echo "  - No AMD GPU present" >&2
        echo "  - AMDGPU driver not loaded" >&2
        echo "  - ROCm not installed" >&2
        echo "  - Insufficient permissions to access GPU" >&2
        echo "" >&2
        echo "To diagnose, run:" >&2
        echo "  lspci | grep -i amd" >&2
        echo "  lsmod | grep amdgpu" >&2
        echo "  which rocminfo" >&2
        return 1
    fi

    echo "$gpu_arch"
    return 0
}

# Function: check_rocminfo_available
# Description: Verifies rocminfo command is available and executable
# Usage: check_rocminfo_available || exit 1
# Returns: 0 if available, 1 if not
check_rocminfo_available() {
    if ! command -v rocminfo >/dev/null 2>&1; then
        echo "ERROR: rocminfo command not found" >&2
        echo "" >&2
        echo "The rocminfo utility is required for GPU detection." >&2
        echo "" >&2
        echo "To install ROCm, run:" >&2
        echo "  ./scripts/install_rocm.sh" >&2
        echo "" >&2
        echo "Or see: https://rocm.docs.amd.com/en/latest/install/install.html" >&2
        return 1
    fi

    if [ ! -x "$(command -v rocminfo)" ]; then
        echo "ERROR: rocminfo found but not executable" >&2
        echo "" >&2
        echo "Path: $(command -v rocminfo)" >&2
        echo "" >&2
        echo "Try:" >&2
        echo "  chmod +x $(command -v rocminfo)" >&2
        return 1
    fi

    return 0
}
