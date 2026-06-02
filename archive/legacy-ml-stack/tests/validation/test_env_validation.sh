#!/bin/bash
# Environment Validation Test
# Tests that environment validation actually catches errors

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

# Create a temporary test environment
TEST_DIR=$(mktemp -d)
ORIGINAL_HOME="$HOME"
trap "rm -rf $TEST_DIR" EXIT

echo "=========================================="
echo "Environment Validation Test Suite"
echo "Testing ACTUAL environment validation behavior"
echo "=========================================="
echo ""

# Source the validation utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../scripts/env_validation_utils.sh" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/../../scripts/env_validation_utils.sh"
else
    echo -e "${RED}ERROR: env_validation_utils.sh not found${NC}"
    exit 1
fi

# Test 1: Missing .mlstack_env file
echo "[1/5] Test: Missing .mlstack_env file"
# Create a temporary home directory
export HOME="$TEST_DIR/test_home"
mkdir -p "$HOME"

# The validation should fail (return non-zero) when file is missing
if ! validate_mlstack_env "test_script" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Correctly detects missing .mlstack_env)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Does not detect missing .mlstack_env)"
    ((FAILED++))
fi

# Restore HOME
export HOME="$ORIGINAL_HOME"

# Test 2: Invalid ROCM_VERSION format
echo "[2/5] Test: Invalid ROCM_VERSION format"
export ROCM_VERSION="invalid"
export ROCM_CHANNEL="stable"
export GPU_ARCH="gfx1100"

# The validation should fail (return non-zero) for invalid version
if ! validate_mlstack_env "test_script" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Correctly detects invalid ROCM_VERSION)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Does not detect invalid ROCM_VERSION)"
    ((FAILED++))
fi

# Test 3: Invalid ROCM_CHANNEL value
echo "[3/5] Test: Invalid ROCM_CHANNEL value"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="invalid"
export GPU_ARCH="gfx1100"

# The validation should fail (return non-zero) for invalid channel
if ! validate_mlstack_env "test_script" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Correctly detects invalid ROCM_CHANNEL)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Does not detect invalid ROCM_CHANNEL)"
    ((FAILED++))
fi

# Test 4: Invalid GPU_ARCH format
echo "[4/5] Test: Invalid GPU_ARCH format"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="latest"
export GPU_ARCH="invalid"

# The validation should fail (return non-zero) for invalid architecture
if ! validate_mlstack_env "test_script" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Correctly detects invalid GPU_ARCH)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Does not detect invalid GPU_ARCH)"
    ((FAILED++))
fi

# Test 5: Valid environment should pass
echo "[5/5] Test: Valid environment variables"
export ROCM_VERSION="7.2"
export ROCM_CHANNEL="latest"
export GPU_ARCH="gfx1100"

# The validation should succeed (return zero) for valid environment
if validate_mlstack_env "test_script" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Correctly validates good environment)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Does not accept valid environment)"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ ENVIRONMENT VALIDATION TEST FAILED${NC}"
    echo "Environment validation is not working correctly."
    exit 1
else
    echo -e "${GREEN}✅ ALL ENVIRONMENT VALIDATION TESTS PASSED${NC}"
    echo "Environment validation correctly detects errors and accepts valid input."
    exit 0
fi
