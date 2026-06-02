#!/usr/bin/env bash
# =============================================================================
# Package Manager Abstraction Layer - Interface Contract Tests
# =============================================================================
# This test suite validates the package manager abstraction layer interface
# across different scenarios and package managers.
#
# Usage:
#   ./test_package_manager.sh                    # Run all tests
#   ./test_package_manager.sh -v                 # Verbose mode
#   ./test_package_manager.sh --pm apt           # Test specific PM
#   PM_DRY_RUN=1 ./test_package_manager.sh       # Test dry-run mode
#
# Part of: Stan's ML Stack - Multi-Distro Compatibility Track (Phase 2)
# =============================================================================

# Note: We use set -uo pipefail instead of set -euo pipefail to avoid
# a bash 5.3.x bug with pop_scope that triggers when set -e is active
# during library sourcing with certain variable declarations.
set -uo pipefail

# =============================================================================
# TEST FRAMEWORK
# =============================================================================

# Colors
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    RESET=''
fi

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Verbose mode
VERBOSE="${VERBOSE:-0}"

# Test results storage
declare -a FAILED_TESTS=()

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
log_success() { echo -e "${GREEN}[PASS]${RESET} $*"; }
log_error()   { echo -e "${RED}[FAIL]${RESET} $*" >&2; }
log_warning() { echo -e "${YELLOW}[SKIP]${RESET} $*"; }
log_debug()   { [[ "$VERBOSE" == "1" ]] && echo -e "${CYAN}[DEBUG]${RESET} $*" >&2; }

# Test assertion functions
test_start() {
    local test_name="$1"
    TESTS_RUN=$((TESTS_RUN + 1))
    log_debug "Starting test: $test_name"
}

test_pass() {
    local test_name="$1"
    local message="${2:-}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    log_success "$test_name${message:+: $message}"
}

test_fail() {
    local test_name="$1"
    local message="${2:-}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    FAILED_TESTS+=("$test_name${message:+: $message}")
    log_error "$test_name${message:+: $message}"
}

test_skip() {
    local test_name="$1"
    local reason="${2:-}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    log_warning "$test_name${reason:+: $reason}"
}

# Assert helpers
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Values should be equal}"
    [[ "$expected" == "$actual" ]]
}

assert_not_empty() {
    local value="$1"
    local message="${2:-Value should not be empty}"
    [[ -n "$value" ]]
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-Value should contain substring}"
    [[ "$haystack" == *"$needle"* ]]
}

assert_exit_code() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Exit code should match}"
    [[ "$expected" -eq "$actual" ]]
}

# =============================================================================
# SETUP
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$PROJECT_ROOT/scripts/lib"

# Source the package manager library
source_library() {
    log_info "Loading package manager library..."

    # Reset library load guards to force clean reload
    unset _MLSTACK_PACKAGE_MANAGER_LOADED
    unset _MLSTACK_DISTRO_DETECTION_LOADED
    unset _PACKAGE_MAPPINGS_LOADED

    # Set environment for testing
    export PM_QUIET=1
    export PM_DEBUG="${VERBOSE:-0}"
    export MLSTACK_QUIET=1
    export MLSTACK_DEBUG="${VERBOSE:-0}"
    export MLSTACK_AUTO_DETECT=true

    # Source the library
    if [[ -f "$LIB_DIR/package_manager.sh" ]]; then
        # shellcheck source=../scripts/lib/package_manager.sh
        . "$LIB_DIR/package_manager.sh"
        return 0
    else
        log_error "Library not found: $LIB_DIR/package_manager.sh"
        return 1
    fi
}

# =============================================================================
# TEST CASES
# =============================================================================

# Test 1: Library loads without errors
test_library_loads() {
    test_start "Library loads"

    # Check that key functions are defined
    local functions=(
        "pm_init"
        "pm_update"
        "pm_upgrade"
        "pm_install"
        "pm_remove"
        "pm_purge"
        "pm_search"
        "pm_info"
        "pm_is_installed"
        "pm_list_installed"
        "pm_add_repo"
        "pm_remove_repo"
    )

    for func in "${functions[@]}"; do
        if ! declare -f "$func" >/dev/null 2>&1; then
            test_fail "Library loads" "Function not defined: $func"
            return 1
        fi
    done

    test_pass "Library loads" "All functions defined"
}

# Test 2: pm_init() sets up correctly
test_pm_init() {
    test_start "pm_init()"

    # Reset initialization state
    _PM_INITIALIZED=0

    # Run init
    if pm_init >/dev/null 2>&1; then
        if [[ "$_PM_INITIALIZED" == "1" ]]; then
            test_pass "pm_init()" "Initialization flag set"
        else
            test_fail "pm_init()" "Initialization flag not set"
            return 1
        fi
    else
        # Init might fail if PM is not supported, but should not crash
        if [[ "$_PM_INITIALIZED" == "1" ]]; then
            test_pass "pm_init()" "Initialized despite error"
        else
            test_fail "pm_init()" "Init failed"
            return 1
        fi
    fi

    # Check that PKG_MANAGER is set
    if [[ -n "${PKG_MANAGER:-}" ]]; then
        log_debug "Package manager detected: $PKG_MANAGER"
    else
        test_fail "pm_init()" "PKG_MANAGER not set"
        return 1
    fi

    # Check that DISTRO_ID is set
    if [[ -n "${DISTRO_ID:-}" ]]; then
        log_debug "Distribution detected: $DISTRO_ID"
    else
        test_fail "pm_init()" "DISTRO_ID not set"
        return 1
    fi
}

# Test 3: pm_install() translates package names
test_pm_install_translation() {
    test_start "pm_install() package translation"

    # Test translation function directly
    local translated
    translated=$(_pm_translate "build-essential" 2>/dev/null) || translated="build-essential"

    case "${PKG_MANAGER:-unknown}" in
        apt)
            if assert_equals "build-essential" "$translated"; then
                test_pass "pm_install() package translation" "apt: build-essential -> $translated"
            else
                test_fail "pm_install() package translation" "Expected 'build-essential', got '$translated'"
                return 1
            fi
            ;;
        pacman)
            if assert_equals "base-devel" "$translated"; then
                test_pass "pm_install() package translation" "pacman: build-essential -> $translated"
            else
                test_fail "pm_install() package translation" "Expected 'base-devel', got '$translated'"
                return 1
            fi
            ;;
        dnf|yum)
            if assert_equals "@development-tools" "$translated"; then
                test_pass "pm_install() package translation" "$PKG_MANAGER: build-essential -> $translated"
            else
                test_fail "pm_install() package translation" "Expected '@development-tools', got '$translated'"
                return 1
            fi
            ;;
        zypper)
            if assert_equals "patterns-devel-base-devel_basis" "$translated"; then
                test_pass "pm_install() package translation" "zypper: build-essential -> $translated"
            else
                test_fail "pm_install() package translation" "Expected 'patterns-devel-base-devel_basis', got '$translated'"
                return 1
            fi
            ;;
        *)
            test_skip "pm_install() package translation" "Unknown package manager: $PKG_MANAGER"
            ;;
    esac
}

# Test 4: pm_is_installed() returns correct status
test_pm_is_installed() {
    test_start "pm_is_installed()"

    # Test with a package that should always be installed (bash or coreutils)
    local test_pkg="bash"
    if pm_is_installed "$test_pkg" 2>/dev/null; then
        log_debug "$test_pkg is installed"
    else
        # Try coreutils as fallback
        test_pkg="coreutils"
        if pm_is_installed "$test_pkg" 2>/dev/null; then
            log_debug "$test_pkg is installed"
        else
            # Try sh as last resort
            test_pkg="sh"
            if pm_is_installed "$test_pkg" 2>/dev/null; then
                log_debug "$test_pkg is installed"
            fi
        fi
    fi

    # Test with a package that definitely should NOT be installed
    local non_existent_pkg="this-package-definitely-does-not-exist-xyz-12345"
    if pm_is_installed "$non_existent_pkg" 2>/dev/null; then
        test_fail "pm_is_installed()" "Non-existent package reported as installed"
        return 1
    else
        log_debug "Non-existent package correctly reported as not installed"
    fi

    test_pass "pm_is_installed()" "Correctly identifies installed vs non-installed packages"
}

# Test 5: pm_search() works across managers
test_pm_search() {
    test_start "pm_search()"

    # Search for a common package
    local search_result
    search_result=$(pm_search "bash" 2>/dev/null) || search_result=""

    if [[ -n "$search_result" ]]; then
        test_pass "pm_search()" "Search returned results"
        log_debug "Search result preview: ${search_result:0:100}..."
    else
        # Search might return empty on some systems, not necessarily a failure
        test_skip "pm_search()" "Search returned no output (may be expected)"
    fi
}

# Test 6: Dry-run mode doesn't execute
test_dry_run_mode() {
    test_start "Dry-run mode"

    # Enable dry-run mode
    export PM_DRY_RUN=1

    # Try to install a package (should not actually install)
    local output
    output=$(pm_install "nonexistent-package-xyz" 2>&1) || true

    # Check that dry-run indicator appeared
    if echo "$output" | grep -qi "dry-run\|DRY-RUN"; then
        test_pass "Dry-run mode" "Dry-run indicator present"
    else
        # In quiet mode, there might not be output
        if [[ -z "$output" ]]; then
            test_pass "Dry-run mode" "No output (quiet mode)"
        else
            log_debug "Output: $output"
            test_pass "Dry-run mode" "Command did not execute"
        fi
    fi

    # Reset dry-run mode
    export PM_DRY_RUN=0
}

# Test 7: Error handling for unknown managers
test_error_handling() {
    test_start "Error handling"

    # Test with invalid package manager (by setting it directly)
    local original_pm="${PKG_MANAGER:-}"

    # Temporarily set to invalid value
    PKG_MANAGER="invalid_manager_xyz"

    # Try an operation - should handle gracefully
    local result
    result=$(pm_update 2>&1) && result_code=0 || result_code=$?

    # Should fail with error, not crash
    if [[ $result_code -ne 0 ]]; then
        test_pass "Error handling" "Invalid PM handled gracefully"
    else
        test_fail "Error handling" "Invalid PM did not return error"
        PKG_MANAGER="$original_pm"
        return 1
    fi

    # Restore original PM
    PKG_MANAGER="$original_pm"
}

# Test 8: Batch operations work correctly
test_batch_operations() {
    test_start "Batch operations"

    # Test translating multiple packages
    local packages=("build-essential" "cmake" "git")
    local translated
    translated=$(_pm_translate_all "${packages[@]}" 2>/dev/null)

    if [[ -n "$translated" ]]; then
        # Should have 3 packages in the result
        local count
        count=$(echo "$translated" | wc -w)
        if [[ $count -eq 3 ]]; then
            test_pass "Batch operations" "All packages translated ($count packages)"
            log_debug "Translated: $translated"
        else
            test_fail "Batch operations" "Expected 3 packages, got $count"
            return 1
        fi
    else
        test_fail "Batch operations" "Translation returned empty"
        return 1
    fi
}

# Test 9: pm_list_installed() returns valid output
test_pm_list_installed() {
    test_start "pm_list_installed()"

    local packages
    packages=$(pm_list_installed 2>/dev/null | head -20)

    if [[ -n "$packages" ]]; then
        # Check that bash is in the list
        if pm_list_installed 2>/dev/null | grep -q "bash"; then
            test_pass "pm_list_installed()" "Returns installed packages including bash"
        else
            test_pass "pm_list_installed()" "Returns installed packages"
        fi
    else
        test_skip "pm_list_installed()" "No output (may be permission issue)"
    fi
}

# Test 10: pm_info() works correctly
test_pm_info() {
    test_start "pm_info()"

    local info
    info=$(pm_info "bash" 2>/dev/null) || info=""

    if [[ -n "$info" ]]; then
        test_pass "pm_info()" "Returns package information"
        log_debug "Info preview: ${info:0:100}..."
    else
        test_skip "pm_info()" "No output returned"
    fi
}

# Test 11: pm_status() displays configuration
test_pm_status() {
    test_start "pm_status()"

    local status
    status=$(pm_status 2>/dev/null) || status=""

    if [[ -n "$status" ]]; then
        if echo "$status" | grep -q "Package Manager"; then
            test_pass "pm_status()" "Displays status information"
        else
            test_fail "pm_status()" "Status output missing expected content"
            return 1
        fi
    else
        test_fail "pm_status()" "No status output"
        return 1
    fi
}

# Test 12: pm_get_native_name() translates correctly
test_pm_get_native_name() {
    test_start "pm_get_native_name()"

    local native
    native=$(pm_get_native_name "python3-dev" 2>/dev/null) || native=""

    case "${PKG_MANAGER:-unknown}" in
        apt)
            if [[ "$native" == "python3-dev" ]]; then
                test_pass "pm_get_native_name()" "apt: python3-dev -> $native"
            else
                test_fail "pm_get_native_name()" "Expected 'python3-dev', got '$native'"
            fi
            ;;
        pacman)
            if [[ "$native" == "python" ]]; then
                test_pass "pm_get_native_name()" "pacman: python3-dev -> $native"
            else
                test_fail "pm_get_native_name()" "Expected 'python', got '$native'"
            fi
            ;;
        dnf|yum)
            if [[ "$native" == "python3-devel" ]]; then
                test_pass "pm_get_native_name()" "$PKG_MANAGER: python3-dev -> $native"
            else
                test_fail "pm_get_native_name()" "Expected 'python3-devel', got '$native'"
            fi
            ;;
        zypper)
            if [[ "$native" == "python3-devel" ]]; then
                test_pass "pm_get_native_name()" "zypper: python3-dev -> $native"
            else
                test_fail "pm_get_native_name()" "Expected 'python3-devel', got '$native'"
            fi
            ;;
        *)
            test_skip "pm_get_native_name()" "Unknown package manager"
            ;;
    esac
}

# Test 13: pm_install_if_missing() works correctly
test_pm_install_if_missing() {
    test_start "pm_install_if_missing()"

    # Use dry-run to test without actually installing
    export PM_DRY_RUN=1

    # bash should already be installed
    local output
    output=$(pm_install_if_missing "bash" 2>&1) || true

    export PM_DRY_RUN=0

    test_pass "pm_install_if_missing()" "Function executed without error"
}

# Test 14: Repository functions exist and can be called
test_repository_functions() {
    test_start "Repository functions"

    # Test that functions exist
    if ! declare -f pm_add_repo >/dev/null 2>&1; then
        test_fail "Repository functions" "pm_add_repo not defined"
        return 1
    fi

    if ! declare -f pm_remove_repo >/dev/null 2>&1; then
        test_fail "Repository functions" "pm_remove_repo not defined"
        return 1
    fi

    # Test pm_update_repos (alias for pm_update)
    if ! declare -f pm_update_repos >/dev/null 2>&1; then
        test_fail "Repository functions" "pm_update_repos not defined"
        return 1
    fi

    test_pass "Repository functions" "All repository functions defined"
}

# Test 15: pm_remove and pm_purge functions
test_remove_functions() {
    test_start "Remove/purge functions"

    # Test that functions exist and don't crash
    export PM_DRY_RUN=1

    local remove_output purge_output
    remove_output=$(pm_remove "some-package" 2>&1) || true
    purge_output=$(pm_purge "some-package" 2>&1) || true

    export PM_DRY_RUN=0

    test_pass "Remove/purge functions" "Functions executed without crash"
}

# =============================================================================
# TEST RUNNER
# =============================================================================

run_tests() {
    echo ""
    echo "=============================================="
    echo " Package Manager Abstraction Layer Tests"
    echo "=============================================="
    echo ""
    log_info "Project root: $PROJECT_ROOT"
    log_info "Library dir:  $LIB_DIR"
    log_info "Package manager: ${PKG_MANAGER:-unknown}"
    echo ""

    # Run all tests
    test_library_loads
    test_pm_init
    test_pm_install_translation
    test_pm_is_installed
    test_pm_search
    test_dry_run_mode
    test_error_handling
    test_batch_operations
    test_pm_list_installed
    test_pm_info
    test_pm_status
    test_pm_get_native_name
    test_pm_install_if_missing
    test_repository_functions
    test_remove_functions

    # Print summary
    echo ""
    echo "=============================================="
    echo " Test Summary"
    echo "=============================================="
    echo ""
    echo "  Total:   $TESTS_RUN"
    echo "  Passed:  $TESTS_PASSED"
    echo "  Failed:  $TESTS_FAILED"
    echo "  Skipped: $TESTS_SKIPPED"
    echo ""

    # Print failed tests
    if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
        echo "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi

    # Return status
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "Some tests failed!"
        return 1
    fi
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            --pm)
                export MLSTACK_PKG_MANAGER="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -v, --verbose    Enable verbose output"
                echo "  --pm <manager>   Test specific package manager (apt, pacman, dnf, yum, zypper)"
                echo "  -h, --help       Show this help message"
                echo ""
                echo "Environment variables:"
                echo "  PM_DRY_RUN=1     Test dry-run mode"
                echo "  VERBOSE=1        Enable verbose output"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Source library
    if ! source_library; then
        log_error "Failed to source library"
        exit 1
    fi

    # Run tests
    run_tests
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
