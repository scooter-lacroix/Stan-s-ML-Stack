#!/bin/bash
# Master Test Runner for PR #5 Critical Review
# Runs all validation tests and provides comprehensive results

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32M'
YELLOW='\033[1;33M'
CYAN='\033[0;36M'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_PASSED=0
TOTAL_FAILED=0
TEST_RESULTS=()

# Function to run a test suite
run_test() {
    local test_name="$1"
    local test_script="$2"

    echo ""
    echo -e "${CYAN}${BOLD}======================================${NC}"
    echo -e "${CYAN}${BOLD}Running: $test_name${NC}"
    echo -e "${CYAN}${BOLD}======================================${NC}"

    if "$test_script"; then
        local result="✓ PASSED"
        local exit_code=0
        ((TOTAL_PASSED++))
    else
        local result="✗ FAILED"
        local exit_code=1
        ((TOTAL_FAILED++))
    fi

    TEST_RESULTS+=("$test_name: $result")
    return $exit_code
}

echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║       PR #5 Critical Review - Test Suite            ║"
echo "║       Comprehensive Validation Testing              ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "This test suite performs GENUINE validation of:"
echo "  • URL accessibility (HTTP requests to verify URLs exist)"
echo "  • Git tag existence (actual clones to verify tags exist)"
echo "  • Script syntax (bash -n validation)"
echo "  • Version consistency (grep validation across files)"
echo "  • Environment validation (tests actual error detection)"
echo ""
echo -e "${YELLOW}NOTE: Some tests require internet access to verify URLs${NC}"
echo ""

# Run all test suites
run_test "URL Validation" "$SCRIPT_DIR/test_urls.sh"
run_test "Git Tag Validation" "$SCRIPT_DIR/test_git_tags.sh"
run_test "Script Syntax Validation" "$SCRIPT_DIR/test_script_syntax.sh"
run_test "Version Consistency Check" "$SCRIPT_DIR/test_version_consistency.sh"
run_test "Environment Validation Test" "$SCRIPT_DIR/test_env_validation.sh"

# Print final summary
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                  FINAL RESULTS                          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
for result in "${TEST_RESULTS[@]}"; do
    if [[ $result == *"PASSED"* ]]; then
        echo -e "${GREEN}$result${NC}"
    else
        echo -e "${RED}$result${NC}"
    fi
done
echo ""
echo -e "${BOLD}Total Passed: $TOTAL_PASSED${NC}"
echo -e "${BOLD}Total Failed: $TOTAL_FAILED${NC}"
echo ""

if [ $TOTAL_FAILED -gt 0 ]; then
    echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                       ║${NC}"
    echo -e "${RED}║         ❌ TEST SUITE FAILED ❌                      ║${NC}"
    echo -e "${RED}║                                                       ║${NC}"
    echo -e "${RED}║   Some validation tests failed. Please review the    ║${NC}"
    echo -e "${RED}║   output above for details.                          ║${NC}"
    echo -e "${RED}║                                                       ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
    exit 1
else
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                       ║${NC}"
    echo -e "${GREEN}║         ✅ ALL TESTS PASSED ✅                       ║${NC}"
    echo -e "${GREEN}║                                                       ║${NC}"
    echo -e "${GREEN}║   All validation tests passed successfully!          ║${NC}"
    echo -e "${GREEN}║                                                       ║${NC}"
    echo -e "${GREEN}║   The codebase is ready for:                       ║${NC}"
    echo -e "${GREEN}║   • Further testing                                  ║${NC}"
    echo -e "${GREEN}║   • Code review                                      ║${NC}"
    echo -e "${GREEN}║   • Merge to main                                     ║${NC}"
    echo -e "${GREEN}║                                                       ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
    exit 0
fi
