#!/usr/bin/env bash
# =============================================================================
# Package Manager Abstraction Layer for Multi-Distro Linux Compatibility
# =============================================================================
# This library provides a unified interface for package management operations
# that works across apt, pacman, dnf, yum, and zypper.
#
# Part of: Stan's ML Stack - Multi-Distro Compatibility Track (Phase 2)
#
# Usage:
#   source scripts/lib/package_manager.sh
#   pm_init
#   pm_install build-essential cmake git
#   pm_is_installed curl && echo "curl is installed"
#
# Features:
#   - Auto-translation of package names across distributions
#   - Dry-run mode (PM_DRY_RUN=1)
#   - Quiet mode (PM_QUIET=1)
#   - Batch operations for efficiency
#   - Comprehensive error handling
#   - Informative logging
#
# Dependencies:
#   - scripts/lib/distro_detection.sh
#   - scripts/lib/package_mappings.sh
#
# Environment Variables:
#   PM_DRY_RUN         - Set to 1 to preview commands without executing
#   PM_QUIET           - Set to 1 to suppress non-error output
#   PM_DEBUG           - Set to 1 for verbose debugging
#   PM_NO_TRANSLATE    - Set to 1 to disable package name translation
#   PM_SUDO            - Custom sudo command (default: sudo)
#   MLSTACK_QUIET      - Inherited from distro_detection.sh
#
# Author: Stanley Chisango (Scooter Lacroix)
# =============================================================================

# Prevent multiple sourcing
if [[ -n "${_MLSTACK_PACKAGE_MANAGER_LOADED:-}" ]]; then
    return 0
fi
_MLSTACK_PACKAGE_MANAGER_LOADED=1

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration (only set if not already defined)
[[ -z "${PM_SUDO:-}" ]] && PM_SUDO="sudo"
[[ -z "${PM_DRY_RUN:-}" ]] && PM_DRY_RUN="0"
[[ -z "${PM_QUIET:-}" ]] && PM_QUIET="0"
[[ -z "${PM_DEBUG:-}" ]] && PM_DEBUG="0"
[[ -z "${PM_NO_TRANSLATE:-}" ]] && PM_NO_TRANSLATE="0"

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

# Colors (only if terminal supports them)
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    _PM_RED='\033[0;31m'
    _PM_GREEN='\033[0;32m'
    _PM_YELLOW='\033[1;33m'
    _PM_BLUE='\033[0;34m'
    _PM_CYAN='\033[0;36m'
    _PM_MAGENTA='\033[0;35m'
    _PM_BOLD='\033[1m'
    _PM_RESET='\033[0m'
else
    _PM_RED=''
    _PM_GREEN=''
    _PM_YELLOW=''
    _PM_BLUE=''
    _PM_CYAN=''
    _PM_MAGENTA=''
    _PM_BOLD=''
    _PM_RESET=''
fi

_pm_log() {
    local level="$1"
    shift
    local message="$*"

    case "$level" in
        error)
            echo -e "${_PM_RED}[PM ERROR]${_PM_RESET} $message" >&2
            ;;
        warning)
            [[ "$PM_QUIET" != "1" ]] && echo -e "${_PM_YELLOW}[PM WARNING]${_PM_RESET} $message" >&2
            ;;
        info)
            [[ "$PM_QUIET" != "1" ]] && echo -e "${_PM_BLUE}[PM INFO]${_PM_RESET} $message" >&2
            ;;
        success)
            [[ "$PM_QUIET" != "1" ]] && echo -e "${_PM_GREEN}[PM SUCCESS]${_PM_RESET} $message" >&2
            ;;
        debug)
            [[ "$PM_DEBUG" == "1" ]] && echo -e "${_PM_CYAN}[PM DEBUG]${_PM_RESET} $message" >&2
            ;;
        dryrun)
            echo -e "${_PM_MAGENTA}[PM DRY-RUN]${_PM_RESET} $message" >&2
            ;;
    esac
}

_pm_log_error()   { _pm_log error "$*"; }
_pm_log_warning() { _pm_log warning "$*"; }
_pm_log_info()    { _pm_log info "$*"; }
_pm_log_success() { _pm_log success "$*"; }
_pm_log_debug()   { _pm_log debug "$*"; }
_pm_log_dryrun()  { _pm_log dryrun "$*"; }

# =============================================================================
# DEPENDENCY LOADING
# =============================================================================

# Get the directory where this script is located
_PM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load distro detection library
if [[ -z "${_MLSTACK_DISTRO_DETECTION_LOADED:-}" ]]; then
    if [[ -f "$_PM_LIB_DIR/distro_detection.sh" ]]; then
        # shellcheck source=distro_detection.sh
        . "$_PM_LIB_DIR/distro_detection.sh"
    else
        _pm_log_error "Required library not found: $_PM_LIB_DIR/distro_detection.sh"
        return 1
    fi
fi

# Load package mappings library
if [[ -z "${_PACKAGE_MAPPINGS_LOADED:-}" ]]; then
    if [[ -f "$_PM_LIB_DIR/package_mappings.sh" ]]; then
        # shellcheck source=package_mappings.sh
        . "$_PM_LIB_DIR/package_mappings.sh"
    else
        _pm_log_error "Required library not found: $_PM_LIB_DIR/package_mappings.sh"
        return 1
    fi
fi

# =============================================================================
# INITIALIZATION
# =============================================================================

# Tracks whether pm_init has been called
_PM_INITIALIZED=0

# Initialize the package manager abstraction layer
#
# This function:
#   - Detects the distribution and package manager
#   - Sets up internal state
#   - Validates the environment
#
# Returns:
#   0 on success, 1 on failure
#
# Example:
#   pm_init || { echo "Failed to initialize package manager"; exit 1; }
#
pm_init() {
    _pm_log_debug "Initializing package manager abstraction layer..."

    # Ensure distribution is detected
    if [[ -z "${DISTRO_ID:-}" ]] || [[ -z "${PKG_MANAGER:-}" ]]; then
        detect_distribution
    fi

    # Validate package manager
    case "$PKG_MANAGER" in
        apt|pacman|dnf|yum|zypper)
            _pm_log_debug "Package manager: $PKG_MANAGER"
            ;;
        *)
            _pm_log_error "Unsupported package manager: ${PKG_MANAGER:-unknown}"
            _pm_log_error "Supported: apt, pacman, dnf, yum, zypper"
            return 1
            ;;
    esac

    # Check sudo availability (unless in dry-run mode)
    if [[ "$PM_DRY_RUN" != "1" ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            _pm_log_warning "sudo not found - package operations may fail"
        fi
    fi

    _PM_INITIALIZED=1
    _pm_log_debug "Package manager initialized successfully"
    _pm_log_info "Distribution: $DISTRO_NAME ($DISTRO_ID)"
    _pm_log_info "Package manager: $PKG_MANAGER"

    return 0
}

# Ensure initialization before operations
_pm_ensure_init() {
    if [[ "$_PM_INITIALIZED" != "1" ]]; then
        pm_init || return 1
    fi
}

# =============================================================================
# PACKAGE NAME TRANSLATION
# =============================================================================

# Translate a package name to the native format for the current package manager
#
# Arguments:
#   $1 - package_name: The package name (usually Debian/Ubuntu style)
#
# Returns:
#   Translated package name via stdout
#
# Example:
#   native_pkg=$(_pm_translate "build-essential")  # Returns "base-devel" on Arch
#
_pm_translate() {
    local package="$1"

    # Skip translation if disabled
    if [[ "$PM_NO_TRANSLATE" == "1" ]]; then
        echo "$package"
        return 0
    fi

    # Use the mapping function from package_mappings.sh
    get_package_name "$package" "$PKG_MANAGER"
}

# Translate multiple package names
#
# Arguments:
#   $@ - package names
#
# Returns:
#   Space-separated list of translated package names via stdout
#
_pm_translate_all() {
    local translated=()

    for pkg in "$@"; do
        translated+=("$(_pm_translate "$pkg")")
    done

    echo "${translated[*]}"
}

# =============================================================================
# SUDO HELPER
# =============================================================================

# Execute a command with sudo (or just execute in dry-run mode)
#
# Arguments:
#   $@ - Command and arguments to execute
#
# Returns:
#   Exit code from the command (or 0 in dry-run mode)
#
_pm_exec() {
    local cmd=()

    # Add sudo for non-root users (unless in dry-run or if command doesn't need it)
    if [[ "$PM_DRY_RUN" == "1" ]]; then
        _pm_log_dryrun "$*"
        return 0
    fi

    # Execute the command
    _pm_log_debug "Executing: $*"
    "$@"
}

# Execute a command with sudo
#
# Arguments:
#   $@ - Command and arguments to execute with sudo
#
# Returns:
#   Exit code from the command (or 0 in dry-run mode)
#
_pm_sudo() {
    if [[ "$PM_DRY_RUN" == "1" ]]; then
        _pm_log_dryrun "$PM_SUDO $*"
        return 0
    fi

    _pm_log_debug "Executing with sudo: $PM_SUDO $*"
    $PM_SUDO "$@"
}

# =============================================================================
# CORE PACKAGE OPERATIONS
# =============================================================================

# Update package lists/cache
#
# This function refreshes the local package database from repositories.
#
# Returns:
#   0 on success, 1 on failure
#
# Example:
#   pm_update || { echo "Failed to update package lists"; exit 1; }
#
pm_update() {
    _pm_ensure_init || return 1

    _pm_log_info "Updating package lists..."

    local result=0

    case "$PKG_MANAGER" in
        apt)
            _pm_sudo apt update -qq
            result=$?
            ;;
        pacman)
            _pm_sudo pacman -Sy --noconfirm
            result=$?
            ;;
        dnf)
            _pm_sudo dnf clean all && _pm_sudo dnf makecache
            result=$?
            ;;
        yum)
            _pm_sudo yum clean all && _pm_sudo yum makecache
            result=$?
            ;;
        zypper)
            _pm_sudo zypper refresh
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Package lists updated"
    else
        _pm_log_error "Failed to update package lists"
    fi

    return $result
}

# Upgrade all installed packages
#
# This function upgrades all packages to their latest versions.
#
# Returns:
#   0 on success, 1 on failure
#
# Example:
#   pm_upgrade || { echo "Failed to upgrade packages"; exit 1; }
#
pm_upgrade() {
    _pm_ensure_init || return 1

    _pm_log_info "Upgrading all packages..."

    local result=0

    case "$PKG_MANAGER" in
        apt)
            DEBIAN_FRONTEND=noninteractive _pm_sudo apt upgrade -y -qq
            result=$?
            ;;
        pacman)
            _pm_sudo pacman -Su --noconfirm
            result=$?
            ;;
        dnf)
            _pm_sudo dnf upgrade -y --refresh
            result=$?
            ;;
        yum)
            _pm_sudo yum upgrade -y
            result=$?
            ;;
        zypper)
            _pm_sudo zypper update -y
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Packages upgraded successfully"
    else
        _pm_log_error "Failed to upgrade packages"
    fi

    return $result
}

# Install one or more packages
#
# This function installs packages, automatically translating names to the
# native format for the current package manager.
#
# Arguments:
#   $@ - Package names to install (Debian/Ubuntu style names work on all systems)
#
# Returns:
#   0 on success, 1 on failure
#
# Examples:
#   pm_install build-essential cmake git
#   pm_install python3-dev python3-pip
#
pm_install() {
    _pm_ensure_init || return 1

    if [[ $# -eq 0 ]]; then
        _pm_log_error "pm_install: No packages specified"
        return 1
    fi

    # Translate package names
    local native_packages
    native_packages=$(_pm_translate_all "$@")

    _pm_log_info "Installing packages: $*"
    _pm_log_debug "Native packages: $native_packages"

    local result=0

    case "$PKG_MANAGER" in
        apt)
            DEBIAN_FRONTEND=noninteractive _pm_sudo apt install -y -qq $native_packages
            result=$?
            ;;
        pacman)
            _pm_sudo pacman -S --noconfirm --needed $native_packages
            result=$?
            ;;
        dnf)
            _pm_sudo dnf install -y $native_packages
            result=$?
            ;;
        yum)
            _pm_sudo yum install -y $native_packages
            result=$?
            ;;
        zypper)
            _pm_sudo zypper install -y $native_packages
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Packages installed: $*"
    else
        _pm_log_error "Failed to install packages: $*"
    fi

    return $result
}

# Remove one or more packages
#
# This function removes packages but keeps configuration files.
#
# Arguments:
#   $@ - Package names to remove
#
# Returns:
#   0 on success, 1 on failure
#
# Examples:
#   pm_remove nano vim-tiny
#
pm_remove() {
    _pm_ensure_init || return 1

    if [[ $# -eq 0 ]]; then
        _pm_log_error "pm_remove: No packages specified"
        return 1
    fi

    # Translate package names
    local native_packages
    native_packages=$(_pm_translate_all "$@")

    _pm_log_info "Removing packages: $*"

    local result=0

    case "$PKG_MANAGER" in
        apt)
            _pm_sudo apt remove -y -qq $native_packages
            result=$?
            ;;
        pacman)
            _pm_sudo pacman -R --noconfirm $native_packages
            result=$?
            ;;
        dnf)
            _pm_sudo dnf remove -y $native_packages
            result=$?
            ;;
        yum)
            _pm_sudo yum remove -y $native_packages
            result=$?
            ;;
        zypper)
            _pm_sudo zypper remove -y $native_packages
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Packages removed: $*"
    else
        _pm_log_error "Failed to remove packages: $*"
    fi

    return $result
}

# Purge packages (remove with configuration files)
#
# This function removes packages along with their configuration files.
#
# Arguments:
#   $@ - Package names to purge
#
# Returns:
#   0 on success, 1 on failure
#
# Examples:
#   pm_purge nginx apache2
#
pm_purge() {
    _pm_ensure_init || return 1

    if [[ $# -eq 0 ]]; then
        _pm_log_error "pm_purge: No packages specified"
        return 1
    fi

    # Translate package names
    local native_packages
    native_packages=$(_pm_translate_all "$@")

    _pm_log_info "Purging packages (removing with config): $*"

    local result=0

    case "$PKG_MANAGER" in
        apt)
            _pm_sudo apt purge -y -qq $native_packages
            result=$?
            ;;
        pacman)
            # pacman -Rn removes config files
            _pm_sudo pacman -Rn --noconfirm $native_packages
            result=$?
            ;;
        dnf)
            _pm_sudo dnf remove -y $native_packages
            result=$?
            ;;
        yum)
            _pm_sudo yum remove -y $native_packages
            result=$?
            ;;
        zypper)
            _pm_sudo zypper remove -y $native_packages
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Packages purged: $*"
    else
        _pm_log_error "Failed to purge packages: $*"
    fi

    return $result
}

# Search for a package
#
# This function searches for packages matching a query.
#
# Arguments:
#   $1 - Search query
#
# Returns:
#   0 on success (results found or not), 1 on error
#   Outputs search results to stdout
#
# Examples:
#   pm_search python
#   pm_search "machine learning"
#
pm_search() {
    _pm_ensure_init || return 1

    local query="$1"

    if [[ -z "$query" ]]; then
        _pm_log_error "pm_search: No search query specified"
        return 1
    fi

    _pm_log_info "Searching for packages matching: $query"

    case "$PKG_MANAGER" in
        apt)
            apt search "$query" 2>/dev/null | grep -v "^Sorting\|^Full\|^  "
            ;;
        pacman)
            pacman -Ss "$query"
            ;;
        dnf)
            dnf search "$query" 2>/dev/null | tail -n +2
            ;;
        yum)
            yum search "$query" 2>/dev/null | tail -n +2
            ;;
        zypper)
            zypper search "$query" 2>/dev/null | tail -n +4
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac
}

# Show package information
#
# This function displays detailed information about a package.
#
# Arguments:
#   $1 - Package name
#
# Returns:
#   0 on success, 1 on failure
#   Outputs package info to stdout
#
# Examples:
#   pm_info python3
#   pm_info build-essential
#
pm_info() {
    _pm_ensure_init || return 1

    local package="$1"

    if [[ -z "$package" ]]; then
        _pm_log_error "pm_info: No package specified"
        return 1
    fi

    # Translate package name
    local native_package
    native_package=$(_pm_translate "$package")

    _pm_log_info "Showing info for package: $package (native: $native_package)"

    case "$PKG_MANAGER" in
        apt)
            apt show "$native_package" 2>/dev/null
            ;;
        pacman)
            pacman -Si "$native_package" 2>/dev/null
            ;;
        dnf)
            dnf info "$native_package" 2>/dev/null
            ;;
        yum)
            yum info "$native_package" 2>/dev/null
            ;;
        zypper)
            zypper info "$native_package" 2>/dev/null
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac
}

# Check if a package is installed
#
# This function checks whether a package is currently installed.
#
# Arguments:
#   $1 - Package name
#
# Returns:
#   0 if installed, 1 if not installed (or on error)
#
# Examples:
#   if pm_is_installed git; then
#       echo "Git is installed"
#   fi
#
#   pm_is_installed curl || pm_install curl
#
pm_is_installed() {
    _pm_ensure_init || return 1

    local package="$1"

    if [[ -z "$package" ]]; then
        _pm_log_error "pm_is_installed: No package specified"
        return 1
    fi

    # Translate package name
    local native_package
    native_package=$(_pm_translate "$package")

    _pm_log_debug "Checking if package is installed: $package (native: $native_package)"

    case "$PKG_MANAGER" in
        apt)
            dpkg -l "$native_package" 2>/dev/null | grep -q "^ii"
            return $?
            ;;
        pacman)
            pacman -Qi "$native_package" >/dev/null 2>&1
            return $?
            ;;
        dnf)
            rpm -q "$native_package" >/dev/null 2>&1
            return $?
            ;;
        yum)
            rpm -q "$native_package" >/dev/null 2>&1
            return $?
            ;;
        zypper)
            rpm -q "$native_package" >/dev/null 2>&1
            return $?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac
}

# Get list of installed packages matching a pattern
#
# This function lists installed packages, optionally filtering by pattern.
#
# Arguments:
#   $1 - Pattern to match (optional, shows all if not specified)
#
# Returns:
#   0 on success, 1 on failure
#   Outputs package list to stdout (one per line)
#
# Examples:
#   pm_list_installed           # List all installed packages
#   pm_list_installed python    # List installed packages matching "python"
#   pm_list_installed | wc -l   # Count installed packages
#
pm_list_installed() {
    _pm_ensure_init || return 1

    local pattern="${1:-}"

    _pm_log_debug "Listing installed packages (pattern: ${pattern:-all})"

    case "$PKG_MANAGER" in
        apt)
            if [[ -n "$pattern" ]]; then
                dpkg -l | grep "^ii" | awk '{print $2}' | grep -E "$pattern"
            else
                dpkg -l | grep "^ii" | awk '{print $2}'
            fi
            ;;
        pacman)
            if [[ -n "$pattern" ]]; then
                pacman -Qq | grep -E "$pattern"
            else
                pacman -Qq
            fi
            ;;
        dnf|yum)
            if [[ -n "$pattern" ]]; then
                rpm -qa | grep -E "$pattern"
            else
                rpm -qa
            fi
            ;;
        zypper)
            if [[ -n "$pattern" ]]; then
                zypper se -i | tail -n +5 | awk '{print $3}' | grep -E "$pattern"
            else
                zypper se -i | tail -n +5 | awk '{print $3}'
            fi
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac
}

# =============================================================================
# REPOSITORY MANAGEMENT
# =============================================================================

# Add a repository
#
# This function adds a new package repository.
#
# Arguments:
#   $1 - Repository specification (format varies by package manager)
#
# Returns:
#   0 on success, 1 on failure
#
# Repository Specification Formats:
#   apt:    "ppa:user/repo" or "deb [options] url distribution component"
#   pacman: Repository name (must be in pacman.conf format)
#   dnf:    Repository URL or ID
#   zypper: Repository URL with alias
#
# Examples:
#   pm_add_repo "ppa:deadsnakes/ppa"
#   pm_add_repo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 ubuntu main"
#
pm_add_repo() {
    _pm_ensure_init || return 1

    local repo_spec="$1"

    if [[ -z "$repo_spec" ]]; then
        _pm_log_error "pm_add_repo: No repository specification provided"
        return 1
    fi

    _pm_log_info "Adding repository: $repo_spec"

    local result=0

    case "$PKG_MANAGER" in
        apt)
            # Check if it's a PPA or a deb line
            if [[ "$repo_spec" == ppa:* ]]; then
                _pm_sudo apt-add-repository -y "$repo_spec"
                result=$?
            else
                # Add to sources.list.d
                echo "$repo_spec" | _pm_sudo tee /etc/apt/sources.list.d/mlstack-custom.list
                result=$?
            fi
            ;;
        pacman)
            # pacman requires manual pacman.conf editing
            _pm_log_warning "pacman: Please add repository to /etc/pacman.conf manually"
            _pm_log_warning "Repository: $repo_spec"
            return 1
            ;;
        dnf)
            _pm_sudo dnf config-manager --add-repo "$repo_spec"
            result=$?
            ;;
        yum)
            _pm_sudo yum-config-manager --add-repo "$repo_spec"
            result=$?
            ;;
        zypper)
            local alias="${2:-mlstack-repo}"
            _pm_sudo zypper addrepo -G "$repo_spec" "$alias"
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Repository added: $repo_spec"
    else
        _pm_log_error "Failed to add repository: $repo_spec"
    fi

    return $result
}

# Remove a repository
#
# This function removes a package repository.
#
# Arguments:
#   $1 - Repository identifier
#
# Returns:
#   0 on success, 1 on failure
#
# Examples:
#   pm_remove_repo "ppa:deadsnakes/ppa"
#   pm_remove_repo "mlstack-custom"
#
pm_remove_repo() {
    _pm_ensure_init || return 1

    local repo_id="$1"

    if [[ -z "$repo_id" ]]; then
        _pm_log_error "pm_remove_repo: No repository identifier provided"
        return 1
    fi

    _pm_log_info "Removing repository: $repo_id"

    local result=0

    case "$PKG_MANAGER" in
        apt)
            if [[ "$repo_id" == ppa:* ]]; then
                _pm_sudo apt-add-repository -y --remove "$repo_id"
                result=$?
            else
                local repo_file="/etc/apt/sources.list.d/${repo_id}.list"
                if [[ -f "$repo_file" ]]; then
                    _pm_sudo rm -f "$repo_file"
                    result=$?
                else
                    _pm_log_warning "Repository file not found: $repo_file"
                    return 1
                fi
            fi
            ;;
        pacman)
            _pm_log_warning "pacman: Please remove repository from /etc/pacman.conf manually"
            return 1
            ;;
        dnf)
            _pm_sudo dnf config-manager --disable "$repo_id"
            result=$?
            ;;
        yum)
            _pm_sudo yum-config-manager --disable "$repo_id"
            result=$?
            ;;
        zypper)
            _pm_sudo zypper removerepo "$repo_id"
            result=$?
            ;;
        *)
            _pm_log_error "Unknown package manager: $PKG_MANAGER"
            return 1
            ;;
    esac

    if [[ $result -eq 0 ]]; then
        _pm_log_success "Repository removed: $repo_id"
    else
        _pm_log_error "Failed to remove repository: $repo_id"
    fi

    return $result
}

# Update repository cache (same as pm_update but more explicit name)
#
# Returns:
#   0 on success, 1 on failure
#
pm_update_repos() {
    pm_update
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Install packages if not already installed
#
# This is a convenience function that checks before installing.
#
# Arguments:
#   $@ - Package names to install if missing
#
# Returns:
#   0 on success, 1 on failure
#
# Example:
#   pm_install_if_missing git curl wget
#
pm_install_if_missing() {
    _pm_ensure_init || return 1

    local to_install=()

    for pkg in "$@"; do
        if ! pm_is_installed "$pkg"; then
            to_install+=("$pkg")
        else
            _pm_log_debug "Package already installed: $pkg"
        fi
    done

    if [[ ${#to_install[@]} -gt 0 ]]; then
        pm_install "${to_install[@]}"
    else
        _pm_log_info "All packages already installed"
    fi
}

# Get the native package name for the current system
#
# Arguments:
#   $1 - Generic package name
#
# Returns:
#   Native package name via stdout
#
# Example:
#   native=$(pm_get_native_name "python3-dev")  # Returns "python3-devel" on Fedora
#
pm_get_native_name() {
    _pm_ensure_init || return 1
    _pm_translate "$1"
}

# Print package manager status
#
# Outputs current configuration and detected settings.
#
pm_status() {
    _pm_ensure_init || return 1

    echo "Package Manager Status"
    echo "======================"
    echo
    echo "Configuration:"
    echo "  PM_DRY_RUN:      ${PM_DRY_RUN}"
    echo "  PM_QUIET:        ${PM_QUIET}"
    echo "  PM_DEBUG:        ${PM_DEBUG}"
    echo "  PM_NO_TRANSLATE: ${PM_NO_TRANSLATE}"
    echo "  PM_SUDO:         ${PM_SUDO}"
    echo
    echo "Detection:"
    echo "  Distribution:    ${DISTRO_NAME:-unknown} (${DISTRO_ID:-unknown})"
    echo "  Version:         ${DISTRO_VERSION:-unknown}"
    echo "  Family:          ${DISTRO_FAMILY:-unknown}"
    echo "  Package Manager: ${PKG_MANAGER:-unknown}"
    echo
    echo "Initialization:"
    echo "  Status:          $([ "$_PM_INITIALIZED" == "1" ] && echo "Initialized" || echo "Not initialized")"
}

# Get the update command for the current package manager
#
# Returns:
#   Update command string via stdout
#
pm_get_update_cmd() {
    _pm_ensure_init || return 1

    case "$PKG_MANAGER" in
        apt)
            echo "$PM_SUDO apt update"
            ;;
        pacman)
            echo "$PM_SUDO pacman -Sy"
            ;;
        dnf)
            echo "$PM_SUDO dnf makecache"
            ;;
        yum)
            echo "$PM_SUDO yum makecache"
            ;;
        zypper)
            echo "$PM_SUDO zypper refresh"
            ;;
        *)
            echo "# Unknown package manager"
            return 1
            ;;
    esac
}

# Get the install command for a package
#
# Arguments:
#   $1 - Package name
#
# Returns:
#   Install command string via stdout
#
pm_get_install_cmd() {
    _pm_ensure_init || return 1

    local package="$1"
    local native_package
    native_package=$(_pm_translate "$package")

    case "$PKG_MANAGER" in
        apt)
            echo "$PM_SUDO apt install -y $native_package"
            ;;
        pacman)
            echo "$PM_SUDO pacman -S --noconfirm $native_package"
            ;;
        dnf)
            echo "$PM_SUDO dnf install -y $native_package"
            ;;
        yum)
            echo "$PM_SUDO yum install -y $native_package"
            ;;
        zypper)
            echo "$PM_SUDO zypper install -y $native_package"
            ;;
        *)
            echo "# Unknown package manager"
            return 1
            ;;
    esac
}

# =============================================================================
# BATCH OPERATIONS
# =============================================================================

# Install packages from a file (one per line)
#
# Arguments:
#   $1 - Path to file containing package names
#   $2 - (optional) "translate" to translate package names, "native" to use as-is
#
# Returns:
#   0 on success, 1 on failure
#
# Example:
#   pm_install_from_file packages.txt
#
pm_install_from_file() {
    _pm_ensure_init || return 1

    local file="$1"
    local mode="${2:-translate}"

    if [[ ! -f "$file" ]]; then
        _pm_log_error "File not found: $file"
        return 1
    fi

    _pm_log_info "Installing packages from file: $file"

    local packages=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue

        packages+=("$line")
    done < "$file"

    if [[ ${#packages[@]} -eq 0 ]]; then
        _pm_log_warning "No packages found in file: $file"
        return 0
    fi

    if [[ "$mode" == "translate" ]]; then
        pm_install "${packages[@]}"
    else
        # Install without translation
        PM_NO_TRANSLATE=1 pm_install "${packages[@]}"
    fi
}

# =============================================================================
# EXPORT
# =============================================================================

# Export all public functions
export -f pm_init
export -f pm_update
export -f pm_upgrade
export -f pm_install
export -f pm_remove
export -f pm_purge
export -f pm_search
export -f pm_info
export -f pm_is_installed
export -f pm_list_installed
export -f pm_add_repo
export -f pm_remove_repo
export -f pm_update_repos
export -f pm_install_if_missing
export -f pm_get_native_name
export -f pm_status
export -f pm_get_update_cmd
export -f pm_get_install_cmd
export -f pm_install_from_file

# Export internal functions for testing
export -f _pm_translate
export -f _pm_translate_all
export -f _pm_exec
export -f _pm_sudo
export -f _pm_ensure_init
export -f _pm_log
export -f _pm_log_error
export -f _pm_log_warning
export -f _pm_log_info
export -f _pm_log_success
export -f _pm_log_debug
export -f _pm_log_dryrun

# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================

# Auto-initialize unless disabled
if [[ "${PM_AUTO_INIT:-true}" == "true" ]]; then
    pm_init 2>/dev/null || true
fi
