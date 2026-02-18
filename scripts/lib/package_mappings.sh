#!/bin/bash
# =============================================================================
# Package Name Mappings Database for Multi-Distro Linux Compatibility
# =============================================================================
# This library provides bash associative arrays for package name mappings
# across different Linux distributions.
#
# Supported Package Managers:
#   - apt (Debian/Ubuntu)
#   - pacman (Arch Linux)
#   - dnf/yum (Fedora/RHEL/CentOS)
#   - zypper (openSUSE/SLES)
#
# Usage:
#   source /path/to/package_mappings.sh
#   mapped_name=$(get_package_name "build-essential" "pacman")
#
# Author: Stanley Chisango (Scooter Lacroix)
# Part of: Stan's ML Stack - Multi-Distro Compatibility Track
# =============================================================================

# Guard against multiple sourcing
if [ -n "${_PACKAGE_MAPPINGS_LOADED:-}" ]; then
    return 0
fi
_PACKAGE_MAPPINGS_LOADED=1

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

# Check if colors are available
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    _PM_RED='\033[0;31m'
    _PM_YELLOW='\033[0;33m'
    _PM_RESET='\033[0m'
else
    _PM_RED=''
    _PM_YELLOW=''
    _PM_RESET=''
fi

_pm_log_warning() {
    echo -e "${_PM_YELLOW}[package_mappings] Warning: $1${_PM_RESET}" >&2
}

_pm_log_error() {
    echo -e "${_PM_RED}[package_mappings] Error: $1${_PM_RESET}" >&2
}

# =============================================================================
# PACKAGE MANAGER DETECTION
# =============================================================================

# Detect the current system's package manager
# Returns: apt, pacman, dnf, yum, zypper, or unknown
detect_package_manager() {
    if command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v pacman >/dev/null 2>&1; then
        echo "pacman"
    elif command -v zypper >/dev/null 2>&1; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# =============================================================================
# BUILD ESSENTIALS MAPPINGS
# =============================================================================

# Declare associative array for build essentials
declare -gA PKG_MAP_BUILD_ESSENTIALS=(
    # Debian/Ubuntu -> Arch -> Fedora/RHEL -> openSUSE
    ["build-essential:apt"]="build-essential"
    ["build-essential:pacman"]="base-devel"
    ["build-essential:dnf"]="@development-tools"
    ["build-essential:yum"]="@development-tools"
    ["build-essential:zypper"]="patterns-devel-base-devel_basis"

    ["gcc:apt"]="gcc"
    ["gcc:pacman"]="gcc"
    ["gcc:dnf"]="gcc"
    ["gcc:yum"]="gcc"
    ["gcc:zypper"]="gcc"

    ["g++:apt"]="g++"
    ["g++:pacman"]="gcc"  # Arch includes C++ in gcc package
    ["g++:dnf"]="gcc-c++"
    ["g++:yum"]="gcc-c++"
    ["g++:zypper"]="gcc-c++"

    ["make:apt"]="make"
    ["make:pacman"]="make"
    ["make:dnf"]="make"
    ["make:yum"]="make"
    ["make:zypper"]="make"

    ["cmake:apt"]="cmake"
    ["cmake:pacman"]="cmake"
    ["cmake:dnf"]="cmake"
    ["cmake:yum"]="cmake"
    ["cmake:zypper"]="cmake"

    ["ninja-build:apt"]="ninja-build"
    ["ninja-build:pacman"]="ninja"
    ["ninja-build:dnf"]="ninja-build"
    ["ninja-build:yum"]="ninja-build"
    ["ninja-build:zypper"]="ninja"

    ["pkg-config:apt"]="pkg-config"
    ["pkg-config:pacman"]="pkgconf"
    ["pkg-config:dnf"]="pkgconf"
    ["pkg-config:yum"]="pkgconf"
    ["pkg-config:zypper"]="pkgconf"

    ["autoconf:apt"]="autoconf"
    ["autoconf:pacman"]="autoconf"
    ["autoconf:dnf"]="autoconf"
    ["autoconf:yum"]="autoconf"
    ["autoconf:zypper"]="autoconf"

    ["automake:apt"]="automake"
    ["automake:pacman"]="automake"
    ["automake:dnf"]="automake"
    ["automake:yum"]="automake"
    ["automake:zypper"]="automake"

    ["libtool:apt"]="libtool"
    ["libtool:pacman"]="libtool"
    ["libtool:dnf"]="libtool"
    ["libtool:yum"]="libtool"
    ["libtool:zypper"]="libtool"
)

# =============================================================================
# PYTHON DEVELOPMENT MAPPINGS
# =============================================================================

declare -gA PKG_MAP_PYTHON_DEV=(
    ["python3:apt"]="python3"
    ["python3:pacman"]="python"
    ["python3:dnf"]="python3"
    ["python3:yum"]="python3"
    ["python3:zypper"]="python3"

    ["python3-dev:apt"]="python3-dev"
    ["python3-dev:pacman"]="python"  # Headers included in main package
    ["python3-dev:dnf"]="python3-devel"
    ["python3-dev:yum"]="python3-devel"
    ["python3-dev:zypper"]="python3-devel"

    ["python3-pip:apt"]="python3-pip"
    ["python3-pip:pacman"]="python-pip"
    ["python3-pip:dnf"]="python3-pip"
    ["python3-pip:yum"]="python3-pip"
    ["python3-pip:zypper"]="python3-pip"

    ["python3-venv:apt"]="python3-venv"
    ["python3-venv:pacman"]="python"  # venv included in main package
    ["python3-venv:dnf"]="python3"    # venv included in main package
    ["python3-venv:yum"]="python3"
    ["python3-venv:zypper"]="python3"

    ["python3-setuptools:apt"]="python3-setuptools"
    ["python3-setuptools:pacman"]="python-setuptools"
    ["python3-setuptools:dnf"]="python3-setuptools"
    ["python3-setuptools:yum"]="python3-setuptools"
    ["python3-setuptools:zypper"]="python3-setuptools"

    ["python3-wheel:apt"]="python3-wheel"
    ["python3-wheel:pacman"]="python-wheel"
    ["python3-wheel:dnf"]="python3-wheel"
    ["python3-wheel:yum"]="python3-wheel"
    ["python3-wheel:zypper"]="python3-wheel"

    ["python3-numpy:apt"]="python3-numpy"
    ["python3-numpy:pacman"]="python-numpy"
    ["python3-numpy:dnf"]="python3-numpy"
    ["python3-numpy:yum"]="python3-numpy"
    ["python3-numpy:zypper"]="python3-numpy"
)

# =============================================================================
# ROCm TOOLS MAPPINGS
# =============================================================================

declare -gA PKG_MAP_ROCM_TOOLS=(
    # Note: ROCm packages may require AMD repositories on non-Debian systems
    ["rocminfo:apt"]="rocminfo"
    ["rocminfo:pacman"]="rocminfo"  # AUR: rocminfo-git
    ["rocminfo:dnf"]="rocminfo"
    ["rocminfo:yum"]="rocminfo"
    ["rocminfo:zypper"]="rocminfo"

    ["rocprofiler:apt"]="rocprofiler"
    ["rocprofiler:pacman"]="rocprofiler"  # AUR
    ["rocprofiler:dnf"]="rocprofiler"
    ["rocprofiler:yum"]="rocprofiler"
    ["rocprofiler:zypper"]="rocprofiler"

    ["rocm-smi-lib:apt"]="rocm-smi-lib"
    ["rocm-smi-lib:pacman"]="rocm-smi-lib"  # AUR: rocm-smi-lib
    ["rocm-smi-lib:dnf"]="rocm-smi"
    ["rocm-smi-lib:yum"]="rocm-smi"
    ["rocm-smi-lib:zypper"]="rocm-smi"

    ["rocm-dev:apt"]="rocm-dev"
    ["rocm-dev:pacman"]="rocm-dev"  # AUR: rocm-meta
    ["rocm-dev:dnf"]="rocm-dev"
    ["rocm-dev:yum"]="rocm-dev"
    ["rocm-dev:zypper"]="rocm-dev"

    ["rocm-libs:apt"]="rocm-libs"
    ["rocm-libs:pacman"]="rocm-libs"  # AUR
    ["rocm-libs:dnf"]="rocm-libs"
    ["rocm-libs:yum"]="rocm-libs"
    ["rocm-libs:zypper"]="rocm-libs"

    ["hip-runtime:apt"]="hip-runtime-amd"
    ["hip-runtime:pacman"]="hip-runtime-amd"  # AUR: hip-runtime-amd
    ["hip-runtime:dnf"]="hip-runtime"
    ["hip-runtime:yum"]="hip-runtime"
    ["hip-runtime:zypper"]="hip-runtime"
)

# =============================================================================
# ROCm LIBRARIES MAPPINGS
# =============================================================================

declare -gA PKG_MAP_ROCM_LIBS=(
    ["librccl-dev:apt"]="librccl-dev"
    ["librccl-dev:pacman"]="rccl"  # AUR: rccl-git
    ["librccl-dev:dnf"]="rccl-devel"
    ["librccl-dev:yum"]="rccl-devel"
    ["librccl-dev:zypper"]="rccl-devel"

    ["librccl1:apt"]="librccl1"
    ["librccl1:pacman"]="rccl"
    ["librccl1:dnf"]="rccl"
    ["librccl1:yum"]="rccl"
    ["librccl1:zypper"]="rccl"

    ["rccl:apt"]="rccl"
    ["rccl:pacman"]="rccl"
    ["rccl:dnf"]="rccl"
    ["rccl:yum"]="rccl"
    ["rccl:zypper"]="rccl"

    ["rccl-dev:apt"]="rccl-dev"
    ["rccl-dev:pacman"]="rccl"
    ["rccl-dev:dnf"]="rccl-devel"
    ["rccl-dev:yum"]="rccl-devel"
    ["rccl-dev:zypper"]="rccl-devel"

    ["migraphx:apt"]="migraphx"
    ["migraphx:pacman"]="migraphx"  # AUR
    ["migraphx:dnf"]="migraphx"
    ["migraphx:yum"]="migraphx"
    ["migraphx:zypper"]="migraphx"

    ["migraphx-dev:apt"]="migraphx-dev"
    ["migraphx-dev:pacman"]="migraphx"
    ["migraphx-dev:dnf"]="migraphx-devel"
    ["migraphx-dev:yum"]="migraphx-devel"
    ["migraphx-dev:zypper"]="migraphx-devel"

    ["half:apt"]="half"
    ["half:pacman"]="half"  # May need AUR: half-dev
    ["half:dnf"]="half"
    ["half:yum"]="half"
    ["half:zypper"]="half"

    ["miopen-hip:apt"]="miopen-hip"
    ["miopen-hip:pacman"]="miopen-hip"  # AUR
    ["miopen-hip:dnf"]="miopen-hip"
    ["miopen-hip:yum"]="miopen-hip"
    ["miopen-hip:zypper"]="miopen-hip"

    ["rocblas:apt"]="rocblas"
    ["rocblas:pacman"]="rocblas"  # AUR
    ["rocblas:dnf"]="rocblas"
    ["rocblas:yum"]="rocblas"
    ["rocblas:zypper"]="rocblas"
)

# =============================================================================
# MPI MAPPINGS
# =============================================================================

declare -gA PKG_MAP_MPI=(
    ["libopenmpi-dev:apt"]="libopenmpi-dev"
    ["libopenmpi-dev:pacman"]="openmpi"
    ["libopenmpi-dev:dnf"]="openmpi-devel"
    ["libopenmpi-dev:yum"]="openmpi-devel"
    ["libopenmpi-dev:zypper"]="openmpi-devel"

    ["openmpi-bin:apt"]="openmpi-bin"
    ["openmpi-bin:pacman"]="openmpi"
    ["openmpi-bin:dnf"]="openmpi"
    ["openmpi-bin:yum"]="openmpi"
    ["openmpi-bin:zypper"]="openmpi"

    ["openmpi:apt"]="openmpi-bin libopenmpi-dev"
    ["openmpi:pacman"]="openmpi"
    ["openmpi:dnf"]="openmpi openmpi-devel"
    ["openmpi:yum"]="openmpi openmpi-devel"
    ["openmpi:zypper"]="openmpi openmpi-devel"

    ["openmpi-devel:apt"]="libopenmpi-dev"
    ["openmpi-devel:pacman"]="openmpi"
    ["openmpi-devel:dnf"]="openmpi-devel"
    ["openmpi-devel:yum"]="openmpi-devel"
    ["openmpi-devel:zypper"]="openmpi-devel"

    # Environment modules (needed for module load mpi)
    ["environment-modules:apt"]="environment-modules"
    ["environment-modules:pacman"]="environment-modules"
    ["environment-modules:dnf"]="environment-modules"
    ["environment-modules:yum"]="environment-modules"
    ["environment-modules:zypper"]="environment-modules"
)

# =============================================================================
# BUILD TOOLS MAPPINGS
# =============================================================================

declare -gA PKG_MAP_BUILD_TOOLS=(
    ["llvm-dev:apt"]="llvm-dev"
    ["llvm-dev:pacman"]="llvm"
    ["llvm-dev:dnf"]="llvm-devel"
    ["llvm-dev:yum"]="llvm-devel"
    ["llvm-dev:zypper"]="llvm-devel"

    ["clang:apt"]="clang"
    ["clang:pacman"]="clang"
    ["clang:dnf"]="clang"
    ["clang:yum"]="clang"
    ["clang:zypper"]="clang"

    ["lld:apt"]="lld"
    ["lld:pacman"]="lld"
    ["lld:dnf"]="lld"
    ["lld:yum"]="lld"
    ["lld:zypper"]="lld"

    ["lldb:apt"]="lldb"
    ["lldb:pacman"]="lldb"
    ["lldb:dnf"]="lldb"
    ["lldb:yum"]="lldb"
    ["lldb:zypper"]="lldb"

    ["libssl-dev:apt"]="libssl-dev"
    ["libssl-dev:pacman"]="openssl"
    ["libssl-dev:dnf"]="openssl-devel"
    ["libssl-dev:yum"]="openssl-devel"
    ["libssl-dev:zypper"]="libopenssl-devel"

    ["libffi-dev:apt"]="libffi-dev"
    ["libffi-dev:pacman"]="libffi"
    ["libffi-dev:dnf"]="libffi-devel"
    ["libffi-dev:yum"]="libffi-devel"
    ["libffi-dev:zypper"]="libffi-devel"

    ["zlib1g-dev:apt"]="zlib1g-dev"
    ["zlib1g-dev:pacman"]="zlib"
    ["zlib1g-dev:dnf"]="zlib-devel"
    ["zlib1g-dev:yum"]="zlib-devel"
    ["zlib1g-dev:zypper"]="zlib-devel"
)

# =============================================================================
# SYSTEM UTILITIES MAPPINGS
# =============================================================================

declare -gA PKG_MAP_SYSTEM_UTILS=(
    ["pciutils:apt"]="pciutils"
    ["pciutils:pacman"]="pciutils"
    ["pciutils:dnf"]="pciutils"
    ["pciutils:yum"]="pciutils"
    ["pciutils:zypper"]="pciutils"

    ["mesa-utils:apt"]="mesa-utils"
    ["mesa-utils:pacman"]="mesa-demos"
    ["mesa-utils:dnf"]="mesa-demos"
    ["mesa-utils:yum"]="mesa-demos"
    ["mesa-utils:zypper"]="Mesa-demo"

    ["clinfo:apt"]="clinfo"
    ["clinfo:pacman"]="clinfo"
    ["clinfo:dnf"]="clinfo"
    ["clinfo:yum"]="clinfo"
    ["clinfo:zypper"]="clinfo"

    ["libnuma-dev:apt"]="libnuma-dev"
    ["libnuma-dev:pacman"]="numactl"
    ["libnuma-dev:dnf"]="numactl-devel"
    ["libnuma-dev:yum"]="numactl-devel"
    ["libnuma-dev:zypper"]="libnuma-devel"

    ["numactl:apt"]="numactl"
    ["numactl:pacman"]="numactl"
    ["numactl:dnf"]="numactl"
    ["numactl:yum"]="numactl"
    ["numactl:zypper"]="numactl"

    ["hwloc:apt"]="hwloc"
    ["hwloc:pacman"]="hwloc"
    ["hwloc:dnf"]="hwloc"
    ["hwloc:yum"]="hwloc"
    ["hwloc:zypper"]="hwloc"

    ["lshw:apt"]="lshw"
    ["lshw:pacman"]="lshw"
    ["lshw:dnf"]="lshw"
    ["lshw:yum"]="lshw"
    ["lshw:zypper"]="lshw"

    ["dmidecode:apt"]="dmidecode"
    ["dmidecode:pacman"]="dmidecode"
    ["dmidecode:dnf"]="dmidecode"
    ["dmidecode:yum"]="dmidecode"
    ["dmidecode:zypper"]="dmidecode"
)

# =============================================================================
# VERSION CONTROL MAPPINGS
# =============================================================================

declare -gA PKG_MAP_VERSION_CONTROL=(
    ["git:apt"]="git"
    ["git:pacman"]="git"
    ["git:dnf"]="git"
    ["git:yum"]="git"
    ["git:zypper"]="git"

    ["git-lfs:apt"]="git-lfs"
    ["git-lfs:pacman"]="git-lfs"
    ["git-lfs:dnf"]="git-lfs"
    ["git-lfs:yum"]="git-lfs"
    ["git-lfs:zypper"]="git-lfs"

    ["gh:apt"]="gh"
    ["gh:pacman"]="github-cli"
    ["gh:dnf"]="gh"
    ["gh:yum"]="gh"
    ["gh:zypper"]="gh"
)

# =============================================================================
# NETWORK MAPPINGS
# =============================================================================

declare -gA PKG_MAP_NETWORK=(
    ["wget:apt"]="wget"
    ["wget:pacman"]="wget"
    ["wget:dnf"]="wget"
    ["wget:yum"]="wget"
    ["wget:zypper"]="wget"

    ["curl:apt"]="curl"
    ["curl:pacman"]="curl"
    ["curl:dnf"]="curl"
    ["curl:yum"]="curl"
    ["curl:zypper"]="curl"

    ["gnupg:apt"]="gnupg"
    ["gnupg:pacman"]="gnupg"
    ["gnupg:dnf"]="gnupg2"
    ["gnupg:yum"]="gnupg2"
    ["gnupg:zypper"]="gpg2"

    ["ca-certificates:apt"]="ca-certificates"
    ["ca-certificates:pacman"]="ca-certificates"
    ["ca-certificates:dnf"]="ca-certificates"
    ["ca-certificates:yum"]="ca-certificates"
    ["ca-certificates:zypper"]="ca-certificates"

    ["rsync:apt"]="rsync"
    ["rsync:pacman"]="rsync"
    ["rsync:dnf"]="rsync"
    ["rsync:yum"]="rsync"
    ["rsync:zypper"]="rsync"

    ["ssh:apt"]="openssh-client"
    ["ssh:pacman"]="openssh"
    ["ssh:dnf"]="openssh-clients"
    ["ssh:yum"]="openssh-clients"
    ["ssh:zypper"]="openssh"
)

# =============================================================================
# ADDITIONAL LIBRARIES MAPPINGS
# =============================================================================

declare -gA PKG_MAP_LIBS=(
    # Compression libraries
    ["libbz2-dev:apt"]="libbz2-dev"
    ["libbz2-dev:pacman"]="bzip2"
    ["libbz2-dev:dnf"]="bzip2-devel"
    ["libbz2-dev:yum"]="bzip2-devel"
    ["libbz2-dev:zypper"]="libbz2-devel"

    ["liblzma-dev:apt"]="liblzma-dev"
    ["liblzma-dev:pacman"]="xz"
    ["liblzma-dev:dnf"]="xz-devel"
    ["liblzma-dev:yum"]="xz-devel"
    ["liblzma-dev:zypper"]="xz-devel"

    ["libsqlite3-dev:apt"]="libsqlite3-dev"
    ["libsqlite3-dev:pacman"]="sqlite"
    ["libsqlite3-dev:dnf"]="sqlite-devel"
    ["libsqlite3-dev:yum"]="sqlite-devel"
    ["libsqlite3-dev:zypper"]="sqlite3-devel"

    # NCurses/readline
    ["libncurses-dev:apt"]="libncurses-dev"
    ["libncurses-dev:pacman"]="ncurses"
    ["libncurses-dev:dnf"]="ncurses-devel"
    ["libncurses-dev:yum"]="ncurses-devel"
    ["libncurses-dev:zypper"]="ncurses-devel"

    ["libreadline-dev:apt"]="libreadline-dev"
    ["libreadline-dev:pacman"]="readline"
    ["libreadline-dev:dnf"]="readline-devel"
    ["libreadline-dev:yum"]="readline-devel"
    ["libreadline-dev:zypper"]="readline-devel"

    # UUID
    ["uuid-dev:apt"]="uuid-dev"
    ["uuid-dev:pacman"]="util-linux"
    ["uuid-dev:dnf"]="libuuid-devel"
    ["uuid-dev:yum"]="libuuid-devel"
    ["uuid-dev:zypper"]="libuuid-devel"

    # GDBM
    ["libgdbm-dev:apt"]="libgdbm-dev"
    ["libgdbm-dev:pacman"]="gdbm"
    ["libgdbm-dev:dnf"]="gdbm-devel"
    ["libgdbm-dev:yum"]="gdbm-devel"
    ["libgdbm-dev:zypper"]="gdbm-devel"
)

# =============================================================================
# COMBINED MAPPING ARRAY
# =============================================================================

# Build a combined mapping array from all category arrays
# This is used by get_package_name() for lookups
declare -gA PKG_MAP_COMBINED=()

_pm_build_combined_map() {
    # Combine all mapping arrays
    for arr_name in PKG_MAP_BUILD_ESSENTIALS PKG_MAP_PYTHON_DEV \
                    PKG_MAP_ROCM_TOOLS PKG_MAP_ROCM_LIBS PKG_MAP_MPI \
                    PKG_MAP_BUILD_TOOLS PKG_MAP_SYSTEM_UTILS \
                    PKG_MAP_VERSION_CONTROL PKG_MAP_NETWORK PKG_MAP_LIBS; do
        # Get reference to the array
        local -n arr_ref="$arr_name"
        for key in "${!arr_ref[@]}"; do
            PKG_MAP_COMBINED["$key"]="${arr_ref[$key]}"
        done
    done
}

# Build the combined map on first load
_pm_build_combined_map

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

# Get the mapped package name for a given package and package manager
#
# Arguments:
#   $1 - package_name: The Debian/Ubuntu-style package name
#   $2 - package_manager: The target package manager (apt, pacman, dnf, yum, zypper)
#
# Returns:
#   The mapped package name via stdout
#   Returns original name if no mapping exists (with warning)
#
# Example:
#   pkg=$(get_package_name "build-essential" "pacman")
#   # Returns: base-devel
#
get_package_name() {
    local package_name="${1:-}"
    local package_manager="${2:-}"

    # Validate arguments
    if [ -z "$package_name" ]; then
        _pm_log_error "get_package_name: package_name argument is required"
        return 1
    fi

    if [ -z "$package_manager" ]; then
        package_manager=$(detect_package_manager)
        if [ "$package_manager" = "unknown" ]; then
            _pm_log_error "get_package_name: Could not detect package manager"
            echo "$package_name"
            return 1
        fi
    fi

    # Normalize package manager name
    case "$package_manager" in
        apt|apt-get) package_manager="apt" ;;
        dnf|yum) ;;  # Keep as-is for RHEL compatibility
        pacman) ;;
        zypper) ;;
        *)
            _pm_log_warning "Unknown package manager: $package_manager, returning original package name"
            echo "$package_name"
            return 0
            ;;
    esac

    # Look up the mapping
    local lookup_key="${package_name}:${package_manager}"
    local mapped_name="${PKG_MAP_COMBINED[$lookup_key]:-}"

    if [ -n "$mapped_name" ]; then
        echo "$mapped_name"
        return 0
    else
        # No mapping found, return original name with warning
        _pm_log_warning "No mapping found for '$package_name' on '$package_manager', using original name"
        echo "$package_name"
        return 0
    fi
}

# Get mapped package names for multiple packages at once
#
# Arguments:
#   $1 - package_manager: The target package manager
#   $@ - package names (remaining arguments)
#
# Returns:
#   Space-separated list of mapped package names via stdout
#
# Example:
#   pkgs=$(get_package_names "pacman" "build-essential" "python3-dev" "cmake")
#   # Returns: base-devel python cmake
#
get_package_names() {
    local package_manager="${1:-}"
    shift

    local mapped_packages=()
    for pkg in "$@"; do
        mapped_packages+=("$(get_package_name "$pkg" "$package_manager")")
    done

    echo "${mapped_packages[*]}"
}

# Check if a package mapping exists
#
# Arguments:
#   $1 - package_name: The Debian/Ubuntu-style package name
#   $2 - package_manager: The target package manager
#
# Returns:
#   0 if mapping exists, 1 otherwise
#
has_package_mapping() {
    local package_name="${1:-}"
    local package_manager="${2:-}"

    if [ -z "$package_name" ] || [ -z "$package_manager" ]; then
        return 1
    fi

    local lookup_key="${package_name}:${package_manager}"
    [ -n "${PKG_MAP_COMBINED[$lookup_key]:-}" ]
}

# List all known package categories
#
# Returns:
#   Newline-separated list of category names via stdout
#
list_package_categories() {
    cat << 'EOF'
build_essentials
python_dev
rocm_tools
rocm_libs
mpi
build_tools
system_utils
version_control
network
libs
EOF
}

# Get all package mappings for a specific category
#
# Arguments:
#   $1 - category: The category name (build_essentials, python_dev, etc.)
#
# Returns:
#   List of package mappings in format: "package_name:apt=mapped" per line
#
get_category_mappings() {
    local category="${1:-}"
    local arr_name="PKG_MAP_$(echo "$category" | tr '[:lower:]' '[:upper:]')"

    if ! declare -p "$arr_name" >/dev/null 2>&1; then
        _pm_log_error "Unknown category: $category"
        return 1
    fi

    local -n arr_ref="$arr_name"
    for key in "${!arr_ref[@]}"; do
        echo "$key=${arr_ref[$key]}"
    done
}

# Validate that all packages have mappings for all supported package managers
#
# Returns:
#   0 if all packages are fully mapped, 1 if any are missing
#   Outputs report to stdout
#
validate_mappings() {
    local managers=("apt" "pacman" "dnf" "zypper")
    local missing_count=0
    local total_count=0

    echo "Package Mapping Validation Report"
    echo "=================================="
    echo

    # Get all unique package names
    declare -A all_packages=()
    for key in "${!PKG_MAP_COMBINED[@]}"; do
        local pkg="${key%%:*}"
        all_packages["$pkg"]=1
    done

    for pkg in "${!all_packages[@]}"; do
        total_count=$((total_count + 1))
        local missing_managers=()

        for mgr in "${managers[@]}"; do
            if ! has_package_mapping "$pkg" "$mgr"; then
                missing_managers+=("$mgr")
            fi
        done

        if [ ${#missing_managers[@]} -gt 0 ]; then
            missing_count=$((missing_count + 1))
            echo "MISSING: $pkg -> ${missing_managers[*]}"
        fi
    done

    echo
    echo "=================================="
    echo "Total packages: $total_count"
    echo "Missing mappings: $missing_count"

    if [ $missing_count -eq 0 ]; then
        echo "Status: ALL MAPPINGS COMPLETE"
        return 0
    else
        echo "Status: SOME MAPPINGS MISSING"
        return 1
    fi
}

# Export functions for use in subshells
export -f detect_package_manager
export -f get_package_name
export -f get_package_names
export -f has_package_mapping
export -f list_package_categories
export -f get_category_mappings
export -f validate_mappings
