#!/usr/bin/env bash
# =============================================================================
# ROCm Environment Detection Library for Rusty Stack
# =============================================================================
# This library provides multi-distro ROCm path detection, supporting various
# installation locations across different Linux distributions.
#
# Usage:
#   source scripts/lib/rocm_env.sh
#   detect_rocm_path
#   echo "ROCm Path: $ROCm_PATH"
#   echo "ROCm Version: $ROCm_VERSION"
#
# Environment Variables (for override):
#   ROCm_PATH          - Force specific ROCm installation path
#   MLSTACK_ROCM_DEBUG - Enable debug logging
#
# Exported Variables:
#   ROCm_PATH        - Path to ROCm installation (e.g., /opt/rocm)
#   ROCm_VERSION     - ROCm version string (e.g., 6.4.3)
#   ROCm_BIN         - Path to ROCm bin directory
#   ROCm_LIB         - Path to ROCm lib directory
#   ROCm_INCLUDE     - Path to ROCm include directory
#   ROCM_DETECTED    - Whether ROCm was found (true/false)
#
# Exported Functions:
#   detect_rocm_path     - Detect ROCm installation path
#   get_rocm_version     - Get ROCm version string
#   get_rocm_bin_path    - Get ROCm bin path
#   get_rocm_lib_path    - Get ROCm lib path
#   get_rocm_include_path- Get ROCm include path
#   rocm_tool_exists     - Check if a ROCm tool exists
#   get_rocm_tool_path   - Get full path to a ROCm tool
# =============================================================================

# Prevent multiple sourcing
if [[ -n "${_MLSTACK_ROCM_ENV_LOADED:-}" ]]; then
    return 0
fi
_MLSTACK_ROCM_ENV_LOADED=1

# =============================================================================
# SOURCE DEPENDENCIES
# =============================================================================

# Get the directory where this script is located
# Use ${BASH_SOURCE[0]:-} to handle cases where BASH_SOURCE might not be set
_MLSTACK_ROCM_ENV_DIR="${_MLSTACK_ROCM_ENV_DIR:-}"
if [[ -z "$_MLSTACK_ROCM_ENV_DIR" ]]; then
    if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
        _MLSTACK_ROCM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    else
        # Fallback: try to find the library directory relative to common paths
        for candidate in "scripts/lib" "/usr/share/mlstack/lib" "/opt/mlstack/lib"; do
            if [[ -f "$candidate/distro_detection.sh" ]]; then
                _MLSTACK_ROCM_ENV_DIR="$candidate"
                break
            fi
        done
    fi
fi

# Source distribution detection library
if [[ -n "$_MLSTACK_ROCM_ENV_DIR" && -f "${_MLSTACK_ROCM_ENV_DIR}/distro_detection.sh" ]]; then
    # shellcheck source=distro_detection.sh
    source "${_MLSTACK_ROCM_ENV_DIR}/distro_detection.sh"
fi

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

# Colors (only if terminal supports them)
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    _ROCM_RED='\033[0;31m'
    _ROCM_GREEN='\033[0;32m'
    _ROCM_YELLOW='\033[1;33m'
    _ROCM_BLUE='\033[0;34m'
    _ROCM_CYAN='\033[0;36m'
    _ROCM_RESET='\033[0m'
else
    _ROCM_RED=''
    _ROCM_GREEN=''
    _ROCM_YELLOW=''
    _ROCM_BLUE=''
    _ROCM_CYAN=''
    _ROCM_RESET=''
fi

_rocm_log_info() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_ROCM_BLUE}[ROCm INFO]${_ROCM_RESET} $1" >&2 || true
}

_rocm_log_success() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_ROCM_GREEN}[ROCm OK]${_ROCM_RESET} $1" >&2 || true
}

_rocm_log_warning() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_ROCM_YELLOW}[ROCm WARN]${_ROCM_RESET} $1" >&2 || true
}

_rocm_log_error() {
    echo -e "${_ROCM_RED}[ROCm ERROR]${_ROCM_RESET} $1" >&2
}

_rocm_log_debug() {
    [[ -n "${MLSTACK_ROCM_DEBUG:-}" ]] && echo -e "${_ROCM_CYAN}[ROCm DEBUG]${_ROCM_RESET} $1" >&2 || true
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Check if a command exists
_rocm_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a directory exists and is readable
_rocm_dir_exists() {
    [[ -d "$1" && -r "$1" ]]
}

# Check if a file exists and is readable
_rocm_file_exists() {
    [[ -f "$1" && -r "$1" ]]
}

# Get directory from a command's path
_rocm_get_cmd_dir() {
    local cmd_path
    cmd_path=$(command -v "$1" 2>/dev/null)
    if [[ -n "$cmd_path" ]]; then
        # Resolve symlinks and get directory
        local resolved_path
        resolved_path=$(readlink -f "$cmd_path" 2>/dev/null || readlink "$cmd_path" 2>/dev/null || echo "$cmd_path")
        dirname "$resolved_path"
    fi
}

# =============================================================================
# ROCm SEARCH PATHS
# =============================================================================

# Get the list of potential ROCm installation paths to search
# Ordered by priority
_get_rocm_search_paths() {
    local search_paths=()

    # 1. Environment variable override (highest priority)
    if [[ -n "${ROCm_PATH:-}" ]]; then
        search_paths+=("$ROCm_PATH")
        _rocm_log_debug "Adding ROCm_PATH override: $ROCm_PATH"
    fi

    # 2. Standard location
    search_paths+=("/opt/rocm")

    # 3. Versioned directories (sorted by version, newest first)
    local versioned_dirs=()
    if _rocm_dir_exists "/opt"; then
        while IFS= read -r dir; do
            if [[ -n "$dir" ]]; then
                versioned_dirs+=("$dir")
            fi
        done < <(find /opt -maxdepth 1 -type d -name "rocm-*" 2>/dev/null | sort -t'-' -k2 -V -r)
    fi
    search_paths+=("${versioned_dirs[@]}")

    # 4. Distribution-specific paths
    # Arch-based systems (AUR packages often install here)
    if is_arch_based 2>/dev/null; then
        search_paths+=("/usr/lib/rocm")
        search_paths+=("/usr/local/rocm")
        # Some AUR packages install to versioned paths
        if _rocm_dir_exists "/usr/lib"; then
            while IFS= read -r dir; do
                if [[ -n "$dir" ]]; then
                    search_paths+=("$dir")
                fi
            done < <(find /usr/lib -maxdepth 1 -type d -name "rocm-*" 2>/dev/null | sort -t'-' -k2 -V -r)
        fi
    fi

    # Fedora/RHEL-based systems
    if is_fedora_based 2>/dev/null; then
        search_paths+=("/usr/lib64/rocm")
    fi

    # 5. Derive from rocminfo in PATH
    if _rocm_command_exists rocminfo; then
        local rocm_bin_dir
        rocm_bin_dir=$(_rocm_get_cmd_dir rocminfo)
        if [[ -n "$rocm_bin_dir" && "$rocm_bin_dir" != "" ]]; then
            # Go up one level from bin directory
            local potential_root
            potential_root=$(dirname "$rocm_bin_dir")
            if [[ -n "$potential_root" && "$potential_root" != "." && "$potential_root" != "/" ]]; then
                search_paths+=("$potential_root")
            fi
        fi
    fi

    # 6. Derive from rocm-smi in PATH (alternative)
    if _rocm_command_exists rocm-smi; then
        local rocm_smi_dir
        rocm_smi_dir=$(_rocm_get_cmd_dir rocm-smi)
        if [[ -n "$rocm_smi_dir" && "$rocm_smi_dir" != "" ]]; then
            local potential_root
            potential_root=$(dirname "$rocm_smi_dir")
            if [[ -n "$potential_root" && "$potential_root" != "." && "$potential_root" != "/" ]]; then
                search_paths+=("$potential_root")
            fi
        fi
    fi

    # 7. Additional common locations
    search_paths+=("/usr/share/rocm")
    search_paths+=("/usr/local/share/rocm")

    # Print paths (deduplicated)
    local seen_paths=""
    for path in "${search_paths[@]}"; do
        if [[ -n "$path" && ! "$seen_paths" =~ "$path" ]]; then
            echo "$path"
            seen_paths="$seen_paths $path"
        fi
    done
}

# Validate a ROCm installation path
_validate_rocm_path() {
    local path="$1"

    # Must exist as directory
    if ! _rocm_dir_exists "$path"; then
        return 1
    fi

    # Should have at least bin or lib directory
    if _rocm_dir_exists "$path/bin" || _rocm_dir_exists "$path/lib"; then
        return 0
    fi

    # Or contain rocminfo/rocm-smi directly
    if _rocm_file_exists "$path/rocminfo" || _rocm_file_exists "$path/rocm-smi"; then
        return 0
    fi

    return 1
}

# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================

# Detect ROCm installation path
# Sets ROCm_PATH, ROCM_DETECTED
# Returns: 0 if found, 1 if not found
detect_rocm_path() {
    _rocm_log_debug "Starting ROCm path detection..."

    # Check for pre-existing valid path
    if [[ -n "${ROCm_PATH:-}" ]] && _validate_rocm_path "$ROCm_PATH"; then
        _rocm_log_debug "Using existing ROCm_PATH: $ROCm_PATH"
        ROCM_DETECTED="true"
        return 0
    fi

    # Search for ROCm installation
    local found_path=""
    while IFS= read -r search_path; do
        if [[ -n "$search_path" ]] && _validate_rocm_path "$search_path"; then
            found_path="$search_path"
            _rocm_log_debug "Found valid ROCm at: $found_path"
            break
        fi
    done < <(_get_rocm_search_paths)

    if [[ -n "$found_path" ]]; then
        ROCm_PATH="$found_path"
        ROCM_DETECTED="true"
        _rocm_log_success "ROCm detected at: $ROCm_PATH"
        return 0
    else
        ROCm_PATH=""
        ROCM_DETECTED="false"
        _rocm_log_warning "ROCm installation not found"
        return 1
    fi
}

# Get ROCm version string
# Returns version string (e.g., "6.4.3") or empty if not found
get_rocm_version() {
    local rocm_path="${ROCm_PATH:-}"
    local version=""

    # Method 1: Read from version file (standard location)
    if [[ -n "$rocm_path" ]]; then
        local version_file="$rocm_path/.info/version"
        if _rocm_file_exists "$version_file"; then
            version=$(cat "$version_file" 2>/dev/null | tr -d '[:space:]')
            if [[ -n "$version" ]]; then
                echo "$version"
                return 0
            fi
        fi

        # Method 2: Try version file in root
        if _rocm_file_exists "$rocm_path/version"; then
            version=$(cat "$rocm_path/version" 2>/dev/null | tr -d '[:space:]')
            if [[ -n "$version" ]]; then
                echo "$version"
                return 0
            fi
        fi
    fi

    # Method 3: Use rocminfo --version
    if _rocm_command_exists rocminfo || [[ -n "$rocm_path" && -x "$rocm_path/bin/rocminfo" ]]; then
        local rocminfo_cmd="rocminfo"
        [[ -n "$rocm_path" && -x "$rocm_path/bin/rocminfo" ]] && rocminfo_cmd="$rocm_path/bin/rocminfo"

        version=$($rocminfo_cmd --version 2>/dev/null | grep -oP 'ROCm version: \K[0-9.]+' | head -1)
        if [[ -n "$version" ]]; then
            echo "$version"
            return 0
        fi
    fi

    # Method 4: Use rocm-smi --version
    if _rocm_command_exists rocm-smi || [[ -n "$rocm_path" && -x "$rocm_path/bin/rocm-smi" ]]; then
        local rocm_smi_cmd="rocm-smi"
        [[ -n "$rocm_path" && -x "$rocm_path/bin/rocm-smi" ]] && rocm_smi_cmd="$rocm_path/bin/rocm-smi"

        version=$($rocm_smi_cmd --version 2>/dev/null | grep -oP 'ROCm version: \K[0-9.]+' | head -1)
        if [[ -n "$version" ]]; then
            echo "$version"
            return 0
        fi
    fi

    # Method 5: Extract from path (e.g., /opt/rocm-6.4.3)
    if [[ -n "$rocm_path" ]]; then
        version=$(echo "$rocm_path" | grep -oP 'rocm-\K[0-9.]+' | head -1)
        if [[ -n "$version" ]]; then
            echo "$version"
            return 0
        fi
    fi

    # No version found
    echo ""
    return 1
}

# Get ROCm bin path
# Returns: bin path or empty if not found
get_rocm_bin_path() {
    local rocm_path="${ROCm_PATH:-}"

    if [[ -z "$rocm_path" ]]; then
        return 1
    fi

    # Standard location
    if _rocm_dir_exists "$rocm_path/bin"; then
        echo "$rocm_path/bin"
        return 0
    fi

    # Some distributions install directly to /usr/bin
    if _rocm_command_exists rocminfo; then
        local bin_dir
        bin_dir=$(_rocm_get_cmd_dir rocminfo)
        if [[ -n "$bin_dir" ]]; then
            echo "$bin_dir"
            return 0
        fi
    fi

    return 1
}

# Get ROCm lib path
# Returns: lib path or empty if not found
get_rocm_lib_path() {
    local rocm_path="${ROCm_PATH:-}"

    if [[ -z "$rocm_path" ]]; then
        return 1
    fi

    # Try standard lib locations
    local lib_dirs=("lib" "lib64")
    for lib_dir in "${lib_dirs[@]}"; do
        if _rocm_dir_exists "$rocm_path/$lib_dir"; then
            echo "$rocm_path/$lib_dir"
            return 0
        fi
    done

    return 1
}

# Get ROCm include path
# Returns: include path or empty if not found
get_rocm_include_path() {
    local rocm_path="${ROCm_PATH:-}"

    if [[ -z "$rocm_path" ]]; then
        return 1
    fi

    if _rocm_dir_exists "$rocm_path/include"; then
        echo "$rocm_path/include"
        return 0
    fi

    return 1
}

# Check if a ROCm tool exists
# Usage: rocm_tool_exists <tool_name>
# Returns: 0 if exists, 1 if not
rocm_tool_exists() {
    local tool="$1"
    local rocm_bin

    rocm_bin=$(get_rocm_bin_path)
    if [[ -n "$rocm_bin" && -x "$rocm_bin/$tool" ]]; then
        return 0
    fi

    # Also check in PATH
    if _rocm_command_exists "$tool"; then
        return 0
    fi

    return 1
}

# Get full path to a ROCm tool
# Usage: get_rocm_tool_path <tool_name>
# Returns: full path or empty if not found
get_rocm_tool_path() {
    local tool="$1"
    local rocm_bin

    rocm_bin=$(get_rocm_bin_path)
    if [[ -n "$rocm_bin" && -x "$rocm_bin/$tool" ]]; then
        echo "$rocm_bin/$tool"
        return 0
    fi

    # Also check in PATH
    if _rocm_command_exists "$tool"; then
        command -v "$tool"
        return 0
    fi

    return 1
}

# =============================================================================
# INITIALIZATION AND EXPORT
# =============================================================================

# Initialize ROCm environment
_init_rocm_env() {
    # Detect ROCm path
    detect_rocm_path >/dev/null 2>&1 || true

    # Set version (capture return value to handle errexit)
    ROCm_VERSION=$(get_rocm_version) || ROCm_VERSION=""

    # Set subdirectories
    ROCm_BIN=$(get_rocm_bin_path) || ROCm_BIN=""
    ROCm_LIB=$(get_rocm_lib_path) || ROCm_LIB=""
    ROCm_INCLUDE=$(get_rocm_include_path) || ROCm_INCLUDE=""

    # Debug output
    if [[ -n "${MLSTACK_ROCM_DEBUG:-}" ]]; then
        _rocm_log_debug "ROCm Environment Initialized:"
        _rocm_log_debug "  ROCm_PATH:    ${ROCm_PATH:-<not set>}"
        _rocm_log_debug "  ROCm_VERSION: ${ROCm_VERSION:-<not set>}"
        _rocm_log_debug "  ROCm_BIN:     ${ROCm_BIN:-<not set>}"
        _rocm_log_debug "  ROCm_LIB:     ${ROCm_LIB:-<not set>}"
        _rocm_log_debug "  ROCm_INCLUDE: ${ROCm_INCLUDE:-<not set>}"
        _rocm_log_debug "  ROCM_DETECTED: ${ROCM_DETECTED:-false}"
    fi
}

# Run initialization unless disabled
if [[ "${MLSTACK_ROCM_AUTO_INIT:-true}" == "true" ]]; then
    _init_rocm_env
fi

# Export functions
export -f detect_rocm_path get_rocm_version get_rocm_bin_path get_rocm_lib_path
export -f get_rocm_include_path rocm_tool_exists get_rocm_tool_path
export -f _get_rocm_search_paths _validate_rocm_path
export -f _rocm_command_exists _rocm_dir_exists _rocm_file_exists _rocm_get_cmd_dir
export -f _rocm_log_info _rocm_log_success _rocm_log_warning _rocm_log_error _rocm_log_debug

# Export variables
export ROCm_PATH ROCm_VERSION ROCm_BIN ROCm_LIB ROCm_INCLUDE ROCM_DETECTED
