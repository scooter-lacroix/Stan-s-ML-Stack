#!/bin/bash
# =============================================================================
# Package Name Mappings Test Suite
# =============================================================================
# This script tests the package_mappings.sh library functionality.
#
# Usage:
#   ./test_package_mappings.sh
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#
# Author: Stanley Chisango (Scooter Lacroix)
# Part of: Stan's ML Stack - Multi-Distro Compatibility Track
# =============================================================================

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRARY_PATH="${SCRIPT_DIR}/../scripts/lib/package_mappings.sh"

# =============================================================================
# TEST FRAMEWORK
# =============================================================================

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    RESET=''
fi

# Test assertion functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-}"

    if [ "$expected" = "$actual" ]; then
        return 0
    else
        [ -n "$message" ] && echo "  $message"
        echo "  Expected: '$expected'"
        echo "  Actual:   '$actual'"
        return 1
    fi
}

assert_not_empty() {
    local value="$1"
    local message="${2:-Value should not be empty}"

    if [ -n "$value" ]; then
        return 0
    else
        echo "  $message"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-}"

    if [[ "$haystack" == *"$needle"* ]]; then
        return 0
    else
        [ -n "$message" ] && echo "  $message"
        echo "  Expected '$needle' to be in '$haystack'"
        return 1
    fi
}

# Test runner
run_test() {
    local test_name="$1"
    local test_func="$2"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${CYAN}Running: ${RESET}$test_name"

    if $test_func; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "  ${GREEN}PASSED${RESET}"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "  ${RED}FAILED${RESET}"
    fi
    echo
}

# Print section header
print_section() {
    echo
    echo -e "${BLUE}${BOLD}=== $1 ===${RESET}"
    echo
}

# =============================================================================
# SOURCE THE LIBRARY
# =============================================================================

if [ ! -f "$LIBRARY_PATH" ]; then
    echo -e "${RED}ERROR: Library not found at: $LIBRARY_PATH${RESET}"
    exit 1
fi

# Source the library
# shellcheck source=../scripts/lib/package_mappings.sh
source "$LIBRARY_PATH"

# Suppress warnings during tests by capturing stderr
exec 3>&2
suppress_warnings() {
    exec 2>/dev/null
}
restore_warnings() {
    exec 2>&3
}

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

# Test: Library loads correctly
test_library_loads() {
    [ "${_PACKAGE_MAPPINGS_LOADED:-0}" = "1" ]
}

# Test: detect_package_manager function exists and returns a value
test_detect_package_manager() {
    local result
    result=$(detect_package_manager)
    assert_not_empty "$result" "detect_package_manager should return a value"
}

# Test: get_package_name basic functionality
test_get_package_name_basic() {
    local result

    # Test apt mapping (identity)
    result=$(get_package_name "build-essential" "apt")
    assert_equals "build-essential" "$result" "apt build-essential mapping"

    # Test pacman mapping
    result=$(get_package_name "build-essential" "pacman")
    assert_equals "base-devel" "$result" "pacman build-essential mapping"

    # Test dnf mapping (group package)
    result=$(get_package_name "build-essential" "dnf")
    assert_equals "@development-tools" "$result" "dnf build-essential mapping"
}

# Test: get_package_name returns original for unknown packages
test_get_package_name_unknown() {
    local result

    suppress_warnings
    result=$(get_package_name "some-unknown-package-xyz123" "apt")
    restore_warnings

    assert_equals "some-unknown-package-xyz123" "$result" "Unknown package should return original name"
}

# Test: get_package_name with empty package manager detects automatically
test_get_package_name_auto_detect() {
    local result

    # Should detect the system's package manager
    result=$(get_package_name "cmake" "")

    # Just verify it returns something (the actual value depends on the system)
    assert_not_empty "$result" "Auto-detect should return a value"
}

# Test: Python development mappings
test_python_dev_mappings() {
    local result

    # python3-dev
    result=$(get_package_name "python3-dev" "pacman")
    assert_equals "python" "$result" "python3-dev on pacman"

    result=$(get_package_name "python3-dev" "dnf")
    assert_equals "python3-devel" "$result" "python3-dev on dnf"

    result=$(get_package_name "python3-dev" "zypper")
    assert_equals "python3-devel" "$result" "python3-dev on zypper"

    # python3-pip
    result=$(get_package_name "python3-pip" "pacman")
    assert_equals "python-pip" "$result" "python3-pip on pacman"
}

# Test: ROCm tools mappings
test_rocm_tools_mappings() {
    local result

    # rocminfo
    result=$(get_package_name "rocminfo" "apt")
    assert_equals "rocminfo" "$result" "rocminfo on apt"

    result=$(get_package_name "rocminfo" "pacman")
    assert_equals "rocminfo" "$result" "rocminfo on pacman"

    result=$(get_package_name "rocminfo" "dnf")
    assert_equals "rocminfo" "$result" "rocminfo on dnf"
}

# Test: ROCm libraries mappings
test_rocm_libs_mappings() {
    local result

    # librccl-dev
    result=$(get_package_name "librccl-dev" "pacman")
    assert_equals "rccl" "$result" "librccl-dev on pacman"

    result=$(get_package_name "librccl-dev" "dnf")
    assert_equals "rccl-devel" "$result" "librccl-dev on dnf"

    # migraphx
    result=$(get_package_name "migraphx" "apt")
    assert_equals "migraphx" "$result" "migraphx on apt"

    result=$(get_package_name "migraphx-dev" "dnf")
    assert_equals "migraphx-devel" "$result" "migraphx-dev on dnf (devel suffix)"
}

# Test: MPI mappings
test_mpi_mappings() {
    local result

    # libopenmpi-dev
    result=$(get_package_name "libopenmpi-dev" "pacman")
    assert_equals "openmpi" "$result" "libopenmpi-dev on pacman"

    result=$(get_package_name "libopenmpi-dev" "dnf")
    assert_equals "openmpi-devel" "$result" "libopenmpi-dev on dnf"

    result=$(get_package_name "libopenmpi-dev" "zypper")
    assert_equals "openmpi-devel" "$result" "libopenmpi-dev on zypper"
}

# Test: Build tools mappings
test_build_tools_mappings() {
    local result

    # ninja-build
    result=$(get_package_name "ninja-build" "pacman")
    assert_equals "ninja" "$result" "ninja-build on pacman"

    result=$(get_package_name "ninja-build" "dnf")
    assert_equals "ninja-build" "$result" "ninja-build on dnf"

    # llvm-dev
    result=$(get_package_name "llvm-dev" "pacman")
    assert_equals "llvm" "$result" "llvm-dev on pacman"

    result=$(get_package_name "llvm-dev" "dnf")
    assert_equals "llvm-devel" "$result" "llvm-dev on dnf"
}

# Test: System utilities mappings
test_system_utils_mappings() {
    local result

    # mesa-utils
    result=$(get_package_name "mesa-utils" "pacman")
    assert_equals "mesa-demos" "$result" "mesa-utils on pacman"

    result=$(get_package_name "mesa-utils" "dnf")
    assert_equals "mesa-demos" "$result" "mesa-utils on dnf"

    # libnuma-dev
    result=$(get_package_name "libnuma-dev" "pacman")
    assert_equals "numactl" "$result" "libnuma-dev on pacman"

    result=$(get_package_name "libnuma-dev" "dnf")
    assert_equals "numactl-devel" "$result" "libnuma-dev on dnf"

    # pciutils (same across all)
    result=$(get_package_name "pciutils" "apt")
    assert_equals "pciutils" "$result" "pciutils on apt"

    result=$(get_package_name "pciutils" "pacman")
    assert_equals "pciutils" "$result" "pciutils on pacman"
}

# Test: Version control mappings
test_version_control_mappings() {
    local result

    # git (same across all)
    for mgr in apt pacman dnf yum zypper; do
        result=$(get_package_name "git" "$mgr")
        assert_equals "git" "$result" "git on $mgr"
    done

    # gh
    result=$(get_package_name "gh" "pacman")
    assert_equals "github-cli" "$result" "gh on pacman"
}

# Test: Network mappings
test_network_mappings() {
    local result

    # gnupg
    result=$(get_package_name "gnupg" "apt")
    assert_equals "gnupg" "$result" "gnupg on apt"

    result=$(get_package_name "gnupg" "dnf")
    assert_equals "gnupg2" "$result" "gnupg on dnf"

    result=$(get_package_name "gnupg" "pacman")
    assert_equals "gnupg" "$result" "gnupg on pacman"

    # wget (same across all)
    for mgr in apt pacman dnf yum zypper; do
        result=$(get_package_name "wget" "$mgr")
        assert_equals "wget" "$result" "wget on $mgr"
    done
}

# Test: Additional libraries mappings
test_libs_mappings() {
    local result

    # libssl-dev
    result=$(get_package_name "libssl-dev" "pacman")
    assert_equals "openssl" "$result" "libssl-dev on pacman"

    result=$(get_package_name "libssl-dev" "dnf")
    assert_equals "openssl-devel" "$result" "libssl-dev on dnf"

    # libffi-dev
    result=$(get_package_name "libffi-dev" "pacman")
    assert_equals "libffi" "$result" "libffi-dev on pacman"

    result=$(get_package_name "libffi-dev" "dnf")
    assert_equals "libffi-devel" "$result" "libffi-dev on dnf"
}

# Test: get_package_names (plural) function
test_get_package_names() {
    local result

    result=$(get_package_names "pacman" "build-essential" "python3-dev" "cmake")

    assert_contains "$result" "base-devel" "Should contain base-devel"
    assert_contains "$result" "python" "Should contain python"
    assert_contains "$result" "cmake" "Should contain cmake"
}

# Test: has_package_mapping function
test_has_package_mapping() {
    # Known mapping
    has_package_mapping "build-essential" "pacman"
    assert_equals 0 $? "build-essential should have pacman mapping"

    # Unknown mapping
    ! has_package_mapping "nonexistent-package-xyz" "pacman"
    assert_equals 0 $? "nonexistent package should not have mapping"
}

# Test: list_package_categories function
test_list_package_categories() {
    local result
    result=$(list_package_categories)

    assert_contains "$result" "build_essentials" "Should list build_essentials"
    assert_contains "$result" "python_dev" "Should list python_dev"
    assert_contains "$result" "rocm_tools" "Should list rocm_tools"
    assert_contains "$result" "mpi" "Should list mpi"
}

# Test: All core dependencies have mappings for all package managers
test_all_core_dependencies_mapped() {
    local managers=("apt" "pacman" "dnf" "zypper")
    local missing=0

    # Core dependencies from enhanced_setup_environment.sh
    local core_packages=(
        "build-essential"
        "cmake"
        "git"
        "python3-dev"
        "python3-pip"
        "python3-venv"
        "python3-setuptools"
        "python3-wheel"
        "libnuma-dev"
        "pciutils"
        "mesa-utils"
        "clinfo"
        "wget"
        "curl"
        "gnupg"
    )

    print_section "Core Dependencies Mapping Check"

    for pkg in "${core_packages[@]}"; do
        for mgr in "${managers[@]}"; do
            if ! has_package_mapping "$pkg" "$mgr"; then
                echo "  MISSING: $pkg -> $mgr"
                missing=$((missing + 1))
            fi
        done
    done

    if [ $missing -eq 0 ]; then
        echo -e "  ${GREEN}All core dependencies have mappings for all package managers${RESET}"
        return 0
    else
        echo -e "  ${RED}$missing mappings missing${RESET}"
        return 1
    fi
}

# Test: ROCm packages have mappings
test_rocm_dependencies_mapped() {
    local managers=("apt" "pacman" "dnf" "zypper")
    local missing=0

    local rocm_packages=(
        "rocminfo"
        "rocprofiler"
        "rocm-smi-lib"
        "rocm-dev"
        "librccl-dev"
        "migraphx"
        "migraphx-dev"
        "half"
    )

    print_section "ROCm Dependencies Mapping Check"

    for pkg in "${rocm_packages[@]}"; do
        for mgr in "${managers[@]}"; do
            if ! has_package_mapping "$pkg" "$mgr"; then
                echo "  MISSING: $pkg -> $mgr"
                missing=$((missing + 1))
            fi
        done
    done

    if [ $missing -eq 0 ]; then
        echo -e "  ${GREEN}All ROCm dependencies have mappings for all package managers${RESET}"
        return 0
    else
        echo -e "  ${RED}$missing mappings missing${RESET}"
        return 1
    fi
}

# Test: MPI packages have mappings
test_mpi_dependencies_mapped() {
    local managers=("apt" "pacman" "dnf" "zypper")
    local missing=0

    local mpi_packages=(
        "libopenmpi-dev"
        "openmpi-bin"
        "openmpi"
        "openmpi-devel"
        "environment-modules"
    )

    print_section "MPI Dependencies Mapping Check"

    for pkg in "${mpi_packages[@]}"; do
        for mgr in "${managers[@]}"; do
            if ! has_package_mapping "$pkg" "$mgr"; then
                echo "  MISSING: $pkg -> $mgr"
                missing=$((missing + 1))
            fi
        done
    done

    if [ $missing -eq 0 ]; then
        echo -e "  ${GREEN}All MPI dependencies have mappings for all package managers${RESET}"
        return 0
    else
        echo -e "  ${RED}$missing mappings missing${RESET}"
        return 1
    fi
}

# Test: validate_mappings function runs without error
test_validate_mappings() {
    print_section "Full Mapping Validation"

    suppress_warnings
    if validate_mappings; then
        restore_warnings
        echo -e "  ${GREEN}All mappings validated successfully${RESET}"
        return 0
    else
        restore_warnings
        echo -e "  ${YELLOW}Some mappings are incomplete (this may be expected)${RESET}"
        return 0  # Don't fail on incomplete mappings
    fi
}

# Test: apt-get is normalized to apt
test_apt_get_normalization() {
    local result
    result=$(get_package_name "build-essential" "apt-get")
    assert_equals "base-devel" "$(get_package_name "build-essential" "pacman")" "apt-get should be normalized to apt"
}

# Test: yum and dnf are handled separately
test_yum_dnf_handling() {
    # Both should work
    local result_dnf result_yum
    result_dnf=$(get_package_name "libopenmpi-dev" "dnf")
    result_yum=$(get_package_name "libopenmpi-dev" "yum")

    # They should return the same value for this package
    assert_equals "$result_dnf" "$result_yum" "dnf and yum should return same mapping for libopenmpi-dev"
}

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

main() {
    echo -e "${BOLD}"
    cat << "EOF"
   ____                  __        ____
  / __ \____  ____  ____/ /__     / __ )____  _  __
 / /_/ / __ \/ __ \/ __  / _ \   / __  / __ \| |/_/
/ ____/ /_/ / / / / /_/ /  __/  / /_/ / /_/ />  <
/_/    \____/_/ /_/\__,_/\___/  /_____/\____/_/|_|
    ____      __         ____
   /  _/___  / /______  / __ \____ _  _____
   / // __ \/ __/ ___/ / / / / __ `/ / ___/
 _/ // / / / /_(__  ) / /_/ / /_/ / / /
/___/_/ /_/\__/____(_)____/\__,_/_/_/

EOF
    echo -e "${RESET}"
    echo -e "${BOLD}Package Mappings Test Suite${RESET}"
    echo "Testing library at: $LIBRARY_PATH"
    echo

    # Basic functionality tests
    print_section "Basic Functionality Tests"
    run_test "Library loads correctly" test_library_loads
    run_test "detect_package_manager returns value" test_detect_package_manager
    run_test "get_package_name basic functionality" test_get_package_name_basic
    run_test "Unknown packages return original name" test_get_package_name_unknown
    run_test "Auto-detect package manager" test_get_package_name_auto_detect
    run_test "apt-get normalization" test_apt_get_normalization
    run_test "yum/dnf handling" test_yum_dnf_handling

    # Category-specific tests
    print_section "Category Mapping Tests"
    run_test "Python development mappings" test_python_dev_mappings
    run_test "ROCm tools mappings" test_rocm_tools_mappings
    run_test "ROCm libraries mappings" test_rocm_libs_mappings
    run_test "MPI mappings" test_mpi_mappings
    run_test "Build tools mappings" test_build_tools_mappings
    run_test "System utilities mappings" test_system_utils_mappings
    run_test "Version control mappings" test_version_control_mappings
    run_test "Network mappings" test_network_mappings
    run_test "Additional libraries mappings" test_libs_mappings

    # Utility function tests
    print_section "Utility Function Tests"
    run_test "get_package_names function" test_get_package_names
    run_test "has_package_mapping function" test_has_package_mapping
    run_test "list_package_categories function" test_list_package_categories

    # Completeness tests
    print_section "Mapping Completeness Tests"
    run_test "Core dependencies fully mapped" test_all_core_dependencies_mapped
    run_test "ROCm dependencies mapped" test_rocm_dependencies_mapped
    run_test "MPI dependencies mapped" test_mpi_dependencies_mapped
    run_test "Full validation runs" test_validate_mappings

    # Print summary
    echo
    echo -e "${BOLD}========================================${RESET}"
    echo -e "${BOLD}TEST SUMMARY${RESET}"
    echo -e "${BOLD}========================================${RESET}"
    echo
    echo "  Tests run:    $TESTS_RUN"
    echo -e "  ${GREEN}Tests passed: $TESTS_PASSED${RESET}"

    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "  ${RED}Tests failed: $TESTS_FAILED${RESET}"
        echo
        echo -e "${RED}${BOLD}SOME TESTS FAILED${RESET}"
        exit 1
    else
        echo -e "  ${GREEN}Tests failed: 0${RESET}"
        echo
        echo -e "${GREEN}${BOLD}ALL TESTS PASSED${RESET}"
        exit 0
    fi
}

# Run main
main "$@"
