#!/bin/bash
# Simple Environment Validation Test
set -euo pipefail

PASSED=0
FAILED=0

# Note: We don't clean up temp directories here - system will handle it

echo "Environment Validation Test"
echo "=============================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../scripts/env_validation_utils.sh" 2>/dev/null || true

# Test 1: Missing file should fail
echo "Test 1: Missing .mlstack_env"
TEST_HOME=$(mktemp -d)
if ! HOME="$TEST_HOME" validate_mlstack_env "test" >/dev/null 2>&1; then
    echo "  PASS: Correctly detects missing file"
    ((PASSED++)) || true
else
    echo "  FAIL: Does not detect missing file"
    ((FAILED++)) || true
fi

# Test 2: Invalid version should fail
echo "Test 2: Invalid ROCM_VERSION"
export ROCM_VERSION="bad"
export ROCM_CHANNEL="stable"
export GPU_ARCH="gfx1100"
if ! validate_mlstack_env "test" >/dev/null 2>&1; then
    echo "  PASS: Correctly detects invalid version"
    ((PASSED++)) || true
else
    echo "  FAIL: Does not detect invalid version"
    ((FAILED++)) || true
fi

# Test 3: Invalid channel should fail
echo "Test 3: Invalid ROCM_CHANNEL"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="bad"
export GPU_ARCH="gfx1100"
if ! validate_mlstack_env "test" >/dev/null 2>&1; then
    echo "  PASS: Correctly detects invalid channel"
    ((PASSED++)) || true
else
    echo "  FAIL: Does not detect invalid channel"
    ((FAILED++)) || true
fi

# Test 4: Invalid GPU arch should fail
echo "Test 4: Invalid GPU_ARCH"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="latest"
export GPU_ARCH="bad"
if ! validate_mlstack_env "test" >/dev/null 2>&1; then
    echo "  PASS: Correctly detects invalid GPU arch"
    ((PASSED++)) || true
else
    echo "  FAIL: Does not detect invalid GPU arch"
    ((FAILED++)) || true
fi

# Test 5: Valid config should pass
echo "Test 5: Valid environment"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="latest"
export GPU_ARCH="gfx1100"
if validate_mlstack_env "test" >/dev/null 2>&1; then
    echo "  PASS: Correctly accepts valid environment"
    ((PASSED++)) || true
else
    echo "  FAIL: Does not accept valid environment"
    ((FAILED++)) || true
fi

echo ""
echo "Results: $PASSED passed, $FAILED failed"

if [ $FAILED -eq 0 ]; then
    echo "SUCCESS: All environment validation tests passed!"
    exit 0
else
    echo "FAILURE: Some tests failed"
    exit 1
fi
