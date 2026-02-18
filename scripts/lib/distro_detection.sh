#!/usr/bin/env bash
# =============================================================================
# Distribution Detection Library for Rusty Stack
# =============================================================================
# This library provides robust distribution detection for multi-distro
# compatibility. It supports all major Linux distributions and their derivatives.
#
# Usage:
#   source scripts/lib/distro_detection.sh
#   detect_distribution
#   echo "Distribution: $DISTRO_ID"
#   echo "Package Manager: $PKG_MANAGER"
#
# Environment Variables (for override):
#   MLSTACK_DISTRO_ID      - Force specific distribution ID
#   MLSTACK_PKG_MANAGER    - Force specific package manager
#   MLSTACK_SKIP_DETECTION - Skip detection (for testing)
#
# Exported Variables:
#   DISTRO_ID        - Distribution identifier (ubuntu, debian, arch, fedora, etc.)
#   DISTRO_NAME      - Full distribution name
#   DISTRO_VERSION   - Distribution version (22.04, 12, rolling, etc.)
#   DISTRO_CODENAME  - Distribution codename (jammy, bookworm, etc.)
#   DISTRO_FAMILY    - Distribution family (debian, arch, rhel, suse)
#   PKG_MANAGER      - Package manager (apt, pacman, dnf, yum, zypper)
#   IS_CONTAINER     - Whether running in a container (true/false)
#   IS_WSL           - Whether running in WSL (true/false)
#
# Exported Functions:
#   detect_distribution     - Main detection function
#   detect_package_manager  - Detect package manager
#   is_debian_based         - Check if Debian-based
#   is_arch_based           - Check if Arch-based
#   is_fedora_based         - Check if Fedora/RHEL-based
#   get_distro_codename     - Get distribution codename
# =============================================================================

# Prevent multiple sourcing
if [[ -n "${_MLSTACK_DISTRO_DETECTION_LOADED:-}" ]]; then
    return 0
fi
_MLSTACK_DISTRO_DETECTION_LOADED=1

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

# Colors (only if terminal supports them)
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    _DISTRO_RED='\033[0;31m'
    _DISTRO_GREEN='\033[0;32m'
    _DISTRO_YELLOW='\033[1;33m'
    _DISTRO_BLUE='\033[0;34m'
    _DISTRO_CYAN='\033[0;36m'
    _DISTRO_BOLD='\033[1m'
    _DISTRO_RESET='\033[0m'
else
    _DISTRO_RED=''
    _DISTRO_GREEN=''
    _DISTRO_YELLOW=''
    _DISTRO_BLUE=''
    _DISTRO_CYAN=''
    _DISTRO_BOLD=''
    _DISTRO_RESET=''
fi

# Logging functions (can be silenced by setting MLSTACK_QUIET=1)
_distro_log_info() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_DISTRO_BLUE}[INFO]${_DISTRO_RESET} $1" >&2
}

_distro_log_success() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_DISTRO_GREEN}[SUCCESS]${_DISTRO_RESET} $1" >&2
}

_distro_log_warning() {
    [[ -z "${MLSTACK_QUIET:-}" ]] && echo -e "${_DISTRO_YELLOW}[WARNING]${_DISTRO_RESET} $1" >&2
}

_distro_log_error() {
    echo -e "${_DISTRO_RED}[ERROR]${_DISTRO_RESET} $1" >&2
}

_distro_log_debug() {
    [[ -n "${MLSTACK_DEBUG:-}" ]] && echo -e "${_DISTRO_CYAN}[DEBUG]${_DISTRO_RESET} $1" >&2
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Check if a command exists
_distro_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a file exists and is readable
_distro_file_exists() {
    [[ -f "$1" && -r "$1" ]]
}

# Source a file if it exists
_distro_source_if_exists() {
    if _distro_file_exists "$1"; then
        # shellcheck source=/dev/null
        . "$1" 2>/dev/null
        return 0
    fi
    return 1
}

# Normalize string to lowercase
_distro_to_lower() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Check if a string contains a substring (case-insensitive)
_distro_contains() {
    local haystack="$1"
    local needle="$2"
    [[ "$(_distro_to_lower "$haystack")" == *"$(_distro_to_lower "$needle")"* ]]
}

# =============================================================================
# CONTAINER/ENVIRONMENT DETECTION
# =============================================================================

# Detect if running in a container (Docker, Podman, etc.)
detect_container() {
    # Check for Docker
    if [[ -f "/.dockerenv" ]]; then
        _distro_log_debug "Detected Docker container (.dockerenv exists)"
        echo "true"
        return 0
    fi

    # Check for Podman
    if [[ -f "/run/.containerenv" ]]; then
        _distro_log_debug "Detected Podman container (.containerenv exists)"
        echo "true"
        return 0
    fi

    # Check cgroup for container indicators
    if [[ -f "/proc/1/cgroup" ]]; then
        if grep -qE "docker|containerd|lxc|podman" /proc/1/cgroup 2>/dev/null; then
            _distro_log_debug "Detected container (cgroup indicator)"
            echo "true"
            return 0
        fi
    fi

    echo "false"
}

# Detect if running in Windows Subsystem for Linux
detect_wsl() {
    if [[ -f "/proc/version" ]]; then
        if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
            _distro_log_debug "Detected WSL environment"
            echo "true"
            return 0
        fi
    fi
    echo "false"
}

# =============================================================================
# PACKAGE MANAGER DETECTION
# =============================================================================

# Detect the system's package manager
# Returns: apt, pacman, dnf, yum, zypper, or unknown
detect_package_manager() {
    # Allow environment variable override
    if [[ -n "${MLSTACK_PKG_MANAGER:-}" ]]; then
        _distro_log_debug "Using override package manager: $MLSTACK_PKG_MANAGER"
        echo "$MLSTACK_PKG_MANAGER"
        return 0
    fi

    # Check for package managers in order of preference
    # Priority is given to modern package managers over legacy ones

    # Arch Linux family
    if _distro_command_exists pacman; then
        _distro_log_debug "Detected pacman package manager"
        echo "pacman"
        return 0
    fi

    # Debian/Ubuntu family (apt preferred over apt-get)
    if _distro_command_exists apt; then
        _distro_log_debug "Detected apt package manager"
        echo "apt"
        return 0
    fi

    if _distro_command_exists apt-get; then
        _distro_log_debug "Detected apt-get package manager"
        echo "apt"
        return 0
    fi

    # Fedora/RHEL family (dnf preferred over yum)
    if _distro_command_exists dnf; then
        _distro_log_debug "Detected dnf package manager"
        echo "dnf"
        return 0
    fi

    if _distro_command_exists yum; then
        _distro_log_debug "Detected yum package manager"
        echo "yum"
        return 0
    fi

    # SUSE family
    if _distro_command_exists zypper; then
        _distro_log_debug "Detected zypper package manager"
        echo "zypper"
        return 0
    fi

    # No recognized package manager
    _distro_log_warning "No recognized package manager found"
    echo "unknown"
    return 1
}

# =============================================================================
# DISTRIBUTION FAMILY CHECKS
# =============================================================================

# Check if the distribution is Debian-based (Debian, Ubuntu, etc.)
# Returns 0 (true) if Debian-based, 1 (false) otherwise
is_debian_based() {
    local distro_id="${DISTRO_ID:-}"
    local id_like="${DISTRO_ID_LIKE:-}"

    # Check direct ID
    case "$distro_id" in
        debian|ubuntu|linuxmint|pop|elementary|kali|mx|devuan|raspberry*)
            return 0
            ;;
    esac

    # Check ID_LIKE field
    if _distro_contains "$id_like" "debian" || _distro_contains "$id_like" "ubuntu"; then
        return 0
    fi

    # Check package manager as fallback
    [[ "${PKG_MANAGER:-}" == "apt" ]]
}

# Check if the distribution is Arch-based (Arch, Manjaro, CachyOS, etc.)
# Returns 0 (true) if Arch-based, 1 (false) otherwise
is_arch_based() {
    local distro_id="${DISTRO_ID:-}"
    local id_like="${DISTRO_ID_LIKE:-}"

    # Check direct ID
    case "$distro_id" in
        arch|manjaro|cachyos|endeavouros|garuda|arco|artix|rebornos)
            return 0
            ;;
    esac

    # Check ID_LIKE field
    if _distro_contains "$id_like" "arch"; then
        return 0
    fi

    # Check for /etc/arch-release as fallback
    if [[ -f "/etc/arch-release" ]]; then
        return 0
    fi

    # Check package manager as fallback
    [[ "${PKG_MANAGER:-}" == "pacman" ]]
}

# Check if the distribution is Fedora/RHEL-based (Fedora, RHEL, CentOS, Rocky, Alma, etc.)
# Returns 0 (true) if Fedora/RHEL-based, 1 (false) otherwise
is_fedora_based() {
    local distro_id="${DISTRO_ID:-}"
    local id_like="${DISTRO_ID_LIKE:-}"

    # Check direct ID
    case "$distro_id" in
        fedora|rhel|rhel|centos|rocky|almalinux|ol|oracle|scientific|centos-stream)
            return 0
            ;;
    esac

    # Check ID_LIKE field
    if _distro_contains "$id_like" "fedora" || \
       _distro_contains "$id_like" "rhel" || \
       _distro_contains "$id_like" "centos"; then
        return 0
    fi

    # Check package manager as fallback
    [[ "${PKG_MANAGER:-}" == "dnf" ]] || [[ "${PKG_MANAGER:-}" == "yum" ]]
}

# Check if the distribution is SUSE-based (openSUSE, SLES, etc.)
# Returns 0 (true) if SUSE-based, 1 (false) otherwise
is_suse_based() {
    local distro_id="${DISTRO_ID:-}"
    local id_like="${DISTRO_ID_LIKE:-}"

    # Check direct ID
    case "$distro_id" in
        opensuse*|sles|suse|sle*)
            return 0
            ;;
    esac

    # Check ID_LIKE field
    if _distro_contains "$id_like" "suse"; then
        return 0
    fi

    # Check package manager as fallback
    [[ "${PKG_MANAGER:-}" == "zypper" ]]
}

# =============================================================================
# CODENAME DETECTION
# =============================================================================

# Get the distribution codename (e.g., jammy, bookworm, noble)
# Returns: codename string or empty if not available
get_distro_codename() {
    # Return cached value if available
    if [[ -n "${DISTRO_CODENAME:-}" ]]; then
        echo "$DISTRO_CODENAME"
        return 0
    fi

    # Try VERSION_CODENAME from os-release
    if [[ -n "${_DISTRO_VERSION_CODENAME:-}" ]]; then
        echo "$_DISTRO_VERSION_CODENAME"
        return 0
    fi

    # Try lsb-release for Ubuntu
    if _distro_command_exists lsb_release; then
        local codename
        codename=$(lsb_release -c 2>/dev/null | awk '{print $2}')
        if [[ -n "$codename" ]]; then
            echo "$codename"
            return 0
        fi
    fi

    # Distribution-specific fallbacks
    case "${DISTRO_ID:-}" in
        ubuntu)
            # Check /etc/lsb-release
            if [[ -f "/etc/lsb-release" ]]; then
                # shellcheck source=/dev/null
                . /etc/lsb-release 2>/dev/null
                echo "${DISTRIB_CODENAME:-}"
                return 0
            fi
            ;;
        debian)
            # Parse /etc/debian_version for sid/testing
            if [[ -f "/etc/debian_version" ]]; then
                local version
                version=$(cat /etc/debian_version 2>/dev/null)
                case "$version" in
                    sid) echo "sid"; return 0 ;;
                    */*) echo "${version%%/*}"; return 0 ;;
                esac
            fi
            ;;
    esac

    # No codename available
    echo ""
    return 1
}

# =============================================================================
# MAIN DISTRIBUTION DETECTION
# =============================================================================

# Detect the Linux distribution
# Sets global variables: DISTRO_ID, DISTRO_NAME, DISTRO_VERSION, DISTRO_CODENAME,
#                       DISTRO_FAMILY, PKG_MANAGER, IS_CONTAINER, IS_WSL
detect_distribution() {
    # Allow skipping detection (for testing)
    if [[ "${MLSTACK_SKIP_DETECTION:-}" == "true" ]]; then
        _distro_log_debug "Detection skipped (MLSTACK_SKIP_DETECTION=true)"
        return 0
    fi

    # Allow environment variable overrides
    if [[ -n "${MLSTACK_DISTRO_ID:-}" ]]; then
        DISTRO_ID="$MLSTACK_DISTRO_ID"
        _distro_log_debug "Using override DISTRO_ID: $DISTRO_ID"
    fi

    # Initialize variables
    local os_id=""
    local os_name=""
    local os_version=""
    local os_codename=""
    local os_id_like=""

    # Method 1: Parse /etc/os-release (preferred)
    if _distro_file_exists "/etc/os-release"; then
        _distro_log_debug "Parsing /etc/os-release"

        # Source the file to get variables
        # We use a subshell to avoid polluting the namespace
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue

            # Parse key=value pairs
            if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                local key="${BASH_REMATCH[1]}"
                local value="${BASH_REMATCH[2]}"

                # Remove quotes from value
                value="${value#\"}"
                value="${value%\"}"
                value="${value#\'}"
                value="${value%\'}"

                case "$key" in
                    ID) os_id="$value" ;;
                    NAME) os_name="$value" ;;
                    VERSION_ID) os_version="$value" ;;
                    VERSION_CODENAME) os_codename="$value" ;;
                    ID_LIKE) os_id_like="$value" ;;
                esac
            fi
        done < /etc/os-release

        _distro_log_debug "os-release: ID=$os_id, VERSION=$os_version, ID_LIKE=$os_id_like"
    fi

    # Method 2: Fallback to /etc/lsb-release (older Ubuntu/Debian)
    if [[ -z "$os_id" ]] && _distro_file_exists "/etc/lsb-release"; then
        _distro_log_debug "Trying /etc/lsb-release fallback"
        # shellcheck source=/dev/null
        . /etc/lsb-release 2>/dev/null
        os_id="${DISTRIB_ID:-}"
        os_name="${DISTRIB_ID:-}"
        os_version="${DISTRIB_RELEASE:-}"
        os_codename="${DISTRIB_CODENAME:-}"
        os_id=$(echo "$os_id" | tr '[:upper:]' '[:lower:]')
    fi

    # Method 3: Fallback to /etc/arch-release (Arch Linux)
    if [[ -z "$os_id" ]] && [[ -f "/etc/arch-release" ]]; then
        _distro_log_debug "Detected Arch Linux via /etc/arch-release"
        os_id="arch"
        os_name="Arch Linux"
        os_version="rolling"
    fi

    # Method 4: Fallback to /etc/debian_version (Debian)
    if [[ -z "$os_id" ]] && [[ -f "/etc/debian_version" ]]; then
        _distro_log_debug "Detected Debian via /etc/debian_version"
        os_id="debian"
        os_name="Debian GNU/Linux"
        os_version=$(cat /etc/debian_version 2>/dev/null || echo "unknown")
    fi

    # Method 5: Fallback to uname (last resort)
    if [[ -z "$os_id" ]]; then
        _distro_log_warning "Could not detect distribution, using fallback"
        os_id="unknown"
        os_name="Unknown Linux"
        os_version=$(uname -r)
    fi

    # Normalize ID to lowercase
    os_id=$(_distro_to_lower "$os_id")

    # Set global variables (unless overridden)
    DISTRO_ID="${DISTRO_ID:-$os_id}"
    DISTRO_NAME="${DISTRO_NAME:-$os_name}"
    DISTRO_VERSION="${DISTRO_VERSION:-$os_version}"
    DISTRO_CODENAME="${DISTRO_CODENAME:-$os_codename}"
    DISTRO_ID_LIKE="$os_id_like"

    # Store for internal use
    _DISTRO_VERSION_CODENAME="$os_codename"

    # Detect package manager
    PKG_MANAGER=$(detect_package_manager)

    # Detect container/WSL environment
    IS_CONTAINER=$(detect_container)
    IS_WSL=$(detect_wsl)

    # Determine distribution family
    if is_debian_based; then
        DISTRO_FAMILY="debian"
    elif is_arch_based; then
        DISTRO_FAMILY="arch"
    elif is_fedora_based; then
        DISTRO_FAMILY="rhel"
    elif is_suse_based; then
        DISTRO_FAMILY="suse"
    else
        DISTRO_FAMILY="unknown"
    fi

    # Handle special cases
    _handle_special_cases

    # Log detection results
    _distro_log_info "Detected distribution: $DISTRO_NAME ($DISTRO_ID) $DISTRO_VERSION"
    _distro_log_debug "Distribution family: $DISTRO_FAMILY"
    _distro_log_debug "Package manager: $PKG_MANAGER"
    _distro_log_debug "Codename: $DISTRO_CODENAME"
    _distro_log_debug "Container: $IS_CONTAINER, WSL: $IS_WSL"

    return 0
}

# =============================================================================
# SPECIAL CASE HANDLING
# =============================================================================

# Handle distribution-specific special cases
_handle_special_cases() {
    case "$DISTRO_ID" in
        cachyos|manjaro|endeavouros|garuda|arco)
            # Arch derivatives - ensure rolling version
            DISTRO_VERSION="${DISTRO_VERSION:-rolling}"
            ;;

        debian)
            # Handle Debian testing/unstable
            if [[ "$DISTRO_VERSION" == "sid" ]] || \
               [[ "$DISTRO_VERSION" == "testing" ]] || \
               [[ "$DISTRO_VERSION" == "unstable" ]]; then
                DISTRO_CODENAME="${DISTRO_CODENAME:-sid}"
            fi
            ;;

        fedora)
            # Fedora - extract major version
            DISTRO_VERSION="${DISTRO_VERSION%%.*}"
            ;;

        rocky|almalinux|centos)
            # RHEL clones - extract major version for compatibility
            if [[ -n "$DISTRO_VERSION" ]]; then
                DISTRO_VERSION_MAJOR="${DISTRO_VERSION%%.*}"
            fi
            ;;

        opensuse-leap|opensuse-tumbleweed|opensuse)
            # openSUSE - normalize ID
            DISTRO_FAMILY="suse"
            ;;
    esac

    # Handle rolling release distributions
    case "$DISTRO_ID" in
        arch|cachyos|manjaro|endeavouros|garuda|arco|opensuse-tumbleweed)
            DISTRO_VERSION="${DISTRO_VERSION:-rolling}"
            ;;
    esac
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Print distribution information (for debugging)
print_distro_info() {
    detect_distribution

    echo "Distribution Information:"
    echo "  ID:          ${DISTRO_ID:-unknown}"
    echo "  Name:        ${DISTRO_NAME:-unknown}"
    echo "  Version:     ${DISTRO_VERSION:-unknown}"
    echo "  Codename:    ${DISTRO_CODENAME:-none}"
    echo "  Family:      ${DISTRO_FAMILY:-unknown}"
    echo "  ID_LIKE:     ${DISTRO_ID_LIKE:-none}"
    echo "  Pkg Manager: ${PKG_MANAGER:-unknown}"
    echo "  Container:   ${IS_CONTAINER:-false}"
    echo "  WSL:         ${IS_WSL:-false}"
}

# Get the appropriate install command for a package
# Usage: get_install_cmd <package>
get_install_cmd() {
    local package="$1"

    case "$PKG_MANAGER" in
        pacman)
            echo "sudo pacman -S --noconfirm $package"
            ;;
        apt)
            echo "sudo apt install -y $package"
            ;;
        dnf)
            echo "sudo dnf install -y $package"
            ;;
        yum)
            echo "sudo yum install -y $package"
            ;;
        zypper)
            echo "sudo zypper install -y $package"
            ;;
        *)
            echo "# Unknown package manager - please install $package manually"
            return 1
            ;;
    esac
}

# Get the appropriate update command
get_update_cmd() {
    case "$PKG_MANAGER" in
        pacman)
            echo "sudo pacman -Sy"
            ;;
        apt)
            echo "sudo apt update"
            ;;
        dnf)
            echo "sudo dnf refresh"
            ;;
        yum)
            echo "sudo yum makecache"
            ;;
        zypper)
            echo "sudo zypper refresh"
            ;;
        *)
            echo "# Unknown package manager"
            return 1
            ;;
    esac
}

# Get the appropriate upgrade command
get_upgrade_cmd() {
    case "$PKG_MANAGER" in
        pacman)
            echo "sudo pacman -Su --noconfirm"
            ;;
        apt)
            echo "sudo apt upgrade -y"
            ;;
        dnf)
            echo "sudo dnf upgrade -y"
            ;;
        yum)
            echo "sudo yum upgrade -y"
            ;;
        zypper)
            echo "sudo zypper update -y"
            ;;
        *)
            echo "# Unknown package manager"
            return 1
            ;;
    esac
}

# Install a package using the detected package manager
# Usage: distro_install <package> [description]
distro_install() {
    local package="$1"
    local description="${2:-$package}"

    _distro_log_info "Installing $description..."

    case "$PKG_MANAGER" in
        pacman)
            if sudo pacman -S --noconfirm "$package"; then
                _distro_log_success "$description installed"
                return 0
            fi
            ;;
        apt)
            if sudo apt install -y "$package"; then
                _distro_log_success "$description installed"
                return 0
            fi
            ;;
        dnf)
            if sudo dnf install -y "$package"; then
                _distro_log_success "$description installed"
                return 0
            fi
            ;;
        yum)
            if sudo yum install -y "$package"; then
                _distro_log_success "$description installed"
                return 0
            fi
            ;;
        zypper)
            if sudo zypper install -y "$package"; then
                _distro_log_success "$description installed"
                return 0
            fi
            ;;
        *)
            _distro_log_error "Unknown package manager: $PKG_MANAGER"
            _distro_log_info "Please install $package manually"
            return 1
            ;;
    esac

    _distro_log_error "Failed to install $description"
    return 1
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Auto-detect on source (unless disabled)
if [[ "${MLSTACK_AUTO_DETECT:-true}" == "true" ]]; then
    detect_distribution
fi

# Export functions and variables for use in subshells
export -f detect_distribution detect_package_manager detect_container detect_wsl
export -f is_debian_based is_arch_based is_fedora_based is_suse_based
export -f get_distro_codename print_distro_info
export -f get_install_cmd get_update_cmd get_upgrade_cmd distro_install
export -f _distro_command_exists _distro_file_exists _distro_to_lower _distro_contains
export -f _distro_log_info _distro_log_success _distro_log_warning _distro_log_error _distro_log_debug
export -f _handle_special_cases

# Export variables (set after detection)
export DISTRO_ID DISTRO_NAME DISTRO_VERSION DISTRO_CODENAME DISTRO_FAMILY DISTRO_ID_LIKE
export PKG_MANAGER IS_CONTAINER IS_WSL
