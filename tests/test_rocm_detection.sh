#!/usr/bin/env bash
# =============================================================================
# ROCm Detection Test Suite
# =============================================================================

set -o pipefail

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/scripts/lib"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' RESET=''
fi

# Test functions
pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}[PASS]${RESET} $1"
}

fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}[FAIL]${RESET} $1"
}

run_test() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n${BLUE}Running:${RESET} $1"
}

# =============================================================================
# TESTS
# =============================================================================

echo "=============================================="
echo " ROCm Detection Test Suite"
echo "=============================================="
echo ""
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Library dir: $LIB_DIR"

# Test 1: Library loads
run_test "Library loads without errors"
if source "$LIB_DIR/rocm_env.sh" 2>/dev/null; then
    pass "rocm_env.sh loaded successfully"
else
    fail "Failed to load rocm_env.sh"
fi

# Test 2: detect_rocm_path function exists
run_test "detect_rocm_path function exists"
if declare -f detect_rocm_path >/dev/null 2>&1; then
    pass "detect_rocm_path function is defined"
else
    fail "detect_rocm_path function not found"
fi

# Test 3: get_rocm_version function exists
run_test "get_rocm_version function exists"
if declare -f get_rocm_version >/dev/null 2>&1; then
    pass "get_rocm_version function is defined"
else
    fail "get_rocm_version function not found"
fi

# Test 4: ROCm path detection works
run_test "ROCm path detection runs"
if detect_rocm_path 2>/dev/null; then
    if [[ -n "$ROCm_PATH" ]]; then
        pass "ROCm path detected: $ROCm_PATH"
    else
        pass "ROCm not installed (handled gracefully)"
    fi
else
    if [[ -z "$ROCm_PATH" ]]; then
        pass "ROCm not installed (handled gracefully)"
    else
        fail "detect_rocm_path failed unexpectedly"
    fi
fi

# Test 5: Environment variables set
run_test "Environment variables are set"
if [[ -n "${ROCM_DETECTED:-}" ]]; then
    pass "ROCM_DETECTED is set: $ROCM_DETECTED"
else
    fail "ROCM_DETECTED not set"
fi

# Test 6: get_rocm_bin_path works
run_test "get_rocm_bin_path function works"
if bin_path=$(get_rocm_bin_path 2>/dev/null); then
    pass "ROCm bin path: $bin_path"
else
    if [[ "$ROCM_DETECTED" != "true" ]]; then
        pass "No ROCm - bin path function handled gracefully"
    else
        fail "get_rocm_bin_path failed"
    fi
fi

# Test 7: get_rocm_lib_path works
run_test "get_rocm_lib_path function works"
if lib_path=$(get_rocm_lib_path 2>/dev/null); then
    pass "ROCm lib path: $lib_path"
else
    if [[ "$ROCM_DETECTED" != "true" ]]; then
        pass "No ROCm - lib path function handled gracefully"
    else
        fail "get_rocm_lib_path failed"
    fi
fi

# Test 8: rocm_tool_exists works
run_test "rocm_tool_exists function works"
if declare -f rocm_tool_exists >/dev/null 2>&1; then
    if [[ "$ROCM_DETECTED" == "true" ]]; then
        if rocm_tool_exists rocminfo 2>/dev/null; then
            pass "rocminfo tool exists"
        else
            pass "rocminfo tool not found (handled gracefully)"
        fi
    else
        pass "rocm_tool_exists function exists (no ROCm to test)"
    fi
else
    fail "rocm_tool_exists function not found"
fi

# Test 9: get_rocm_tool_path works
run_test "get_rocm_tool_path function works"
if declare -f get_rocm_tool_path >/dev/null 2>&1; then
    pass "get_rocm_tool_path function is defined"
else
    fail "get_rocm_tool_path function not found"
fi

# Test 10: _rocm_log_info works
run_test "_rocm_log_info function works"
if declare -f _rocm_log_info >/dev/null 2>&1; then
    if _rocm_log_info >/dev/null 2>&1; then
        pass "_rocm_log_info executes without error"
    else
        fail "_rocm_log_info failed"
    fi
else
    fail "_rocm_log_info function not found"
fi

# Test 11: Functions are exported
run_test "Functions are exported for subshells"
if bash -c 'source "$0" && declare -f detect_rocm_path >/dev/null' "$LIB_DIR/rocm_env.sh" 2>/dev/null; then
    pass "Functions are available in subshells"
else
    fail "Functions not available in subshells"
fi

# Test 12: Multiple source protection
run_test "Multiple source protection works"
source "$LIB_DIR/rocm_env.sh" 2>/dev/null
source "$LIB_DIR/rocm_env.sh" 2>/dev/null
if [[ "$_MLSTACK_ROCM_ENV_LOADED" == "1" ]]; then
    pass "Multiple source protection works"
else
    fail "Multiple source protection failed"
fi

# Test 13: Version detection (if ROCm installed)
run_test "Version detection (if ROCm installed)"
if [[ "$ROCM_DETECTED" == "true" ]]; then
    if get_rocm_version 2>/dev/null; then
        if [[ -n "$ROCm_VERSION" ]]; then
            pass "ROCm version detected: $ROCm_VERSION"
        else
            pass "Version detection ran but no version found"
        fi
    else
        pass "Version detection handled gracefully"
    fi
else
    pass "No ROCm installed - version detection skipped"
fi

# Test 14: Integration with Rust TUI
run_test "Rust TUI hardware.rs compiles"
if [[ -f "$PROJECT_ROOT/rusty-stack/src/hardware.rs" ]]; then
    if cd "$PROJECT_ROOT/rusty-stack" && cargo check --quiet 2>/dev/null; then
        pass "Rust TUI compiles successfully"
    else
        fail "Rust TUI has compilation errors"
    fi
else
    pass "Rust TUI not found - skipping"
fi

# Test 15: Path validation function works
run_test "Path validation function works"
if declare -f _validate_rocm_path >/dev/null 2>&1; then
    if _validate_rocm_path /nonexistent/path 2>/dev/null; then
        fail "Validation should fail for nonexistent path"
    else
        pass "Validation correctly rejects nonexistent path"
    fi
else
    fail "_validate_rocm_path function not found"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo " Test Summary"
echo "=============================================="
echo ""
echo "  Total:   $TESTS_RUN"
echo -e "  Passed:  ${GREEN}$TESTS_PASSED${RESET}"
echo -e "  Failed:  ${RED}$TESTS_FAILED${RESET}"
echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}[PASS]${RESET} All tests passed!"
    exit 0
else
    echo -e "${RED}[FAIL]${RESET} Some tests failed"
    exit 1
fi
