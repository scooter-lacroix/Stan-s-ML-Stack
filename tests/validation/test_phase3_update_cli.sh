#!/bin/bash
# Phase 3 Validation Tests - Version Sweep + Update CLI
# Tests for: update_helper.sh, update_stack.sh, rusty-stack-update binary

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Phase 3 Validation Tests"
echo "Version Sweep + Update CLI"
echo "=========================================="
echo ""

# Test 3.1: update_helper.sh exists
echo "[1/6] Testing update_helper.sh exists"
HELPER="$PROJECT_ROOT/scripts/lib/update_helper.sh"
if [[ -f "$HELPER" ]]; then
    echo -e "${GREEN}✓ PASS${NC} (update_helper.sh exists)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (update_helper.sh not found)"
    ((FAILED++)) || true
fi

# Test 3.2: update_helper.sh has required functions
echo "[2/6] Testing update_helper.sh has required functions"
REQUIRED_FUNCS=("up_detect_installed" "up_get_version" "up_display_name" "up_update_component" "up_is_python_module" "up_command_exists" "up_path_exists" "up_user_home")
MISSING=0
if [[ -f "$HELPER" ]]; then
    source "$HELPER"
    for func in "${REQUIRED_FUNCS[@]}"; do
        if ! declare -f "$func" &>/dev/null; then
            echo -e "  ${RED}MISSING${NC}: $func"
            ((MISSING++)) || true
        fi
    done
    if [ $MISSING -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} (All ${#REQUIRED_FUNCS[@]} functions present)"
        ((PASSED++)) || true
    else
        echo -e "${RED}✗ FAIL${NC} ($MISSING functions missing)"
        ((FAILED++)) || true
    fi
else
    echo -e "${RED}✗ FAIL${NC} (Cannot source - file not found)"
    ((FAILED++)) || true
fi

# Test 3.3: update_stack.sh exists and is executable
echo "[3/6] Testing update_stack.sh exists and is executable"
UPDATE_SCRIPT="$PROJECT_ROOT/scripts/update_stack.sh"
if [ -x "$UPDATE_SCRIPT" ]; then
    echo -e "${GREEN}✓ PASS${NC} (update_stack.sh is executable)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (update_stack.sh missing or not executable)"
    ((FAILED++)) || true
fi

# Test 3.4: update_stack.sh has no 'local' outside functions
echo "[4/6] Testing update_stack.sh has no 'local' outside functions"
# Check for 'local' keyword in main body (not inside function definitions)
# This is a bash compliance check - 'local' is only valid inside functions
OUTSIDE_LOCAL=0
if [[ -f "$UPDATE_SCRIPT" ]]; then
    # Use awk to find 'local' outside of function bodies
    OUTSIDE_LOCAL=$(awk '
        /^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*[[:space:]]*\(\)/ { in_func=1 }
        /^}/ { in_func=0 }
        !in_func && /\blocal\b/ { count++ }
        END { print count+0 }
    ' "$UPDATE_SCRIPT")
    if [ "$OUTSIDE_LOCAL" -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} (No 'local' outside functions)"
        ((PASSED++)) || true
    else
        echo -e "${RED}✗ FAIL${NC} (Found $OUTSIDE_LOCAL 'local' outside functions)"
        ((FAILED++)) || true
    fi
else
    echo -e "${RED}✗ FAIL${NC} (Cannot check - file not found)"
    ((FAILED++)) || true
fi

# Test 3.5: Cargo.toml has rusty-stack-update binary target
echo "[5/6] Testing Cargo.toml has rusty-stack-update binary target"
CARGO_TOML="$PROJECT_ROOT/rusty-stack/Cargo.toml"
if grep -q "rusty-stack-update" "$CARGO_TOML" && grep -q "update.rs" "$CARGO_TOML"; then
    echo -e "${GREEN}✓ PASS${NC} (Cargo.toml has rusty-stack-update target)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Cargo.toml missing rusty-stack-update target)"
    ((FAILED++)) || true
fi

# Test 3.6: rusty-stack-update binary target is configured and source exists
echo "[6/6] Testing rusty-stack-update binary target and source"
UPDATE_RS="$PROJECT_ROOT/rusty-stack/src/bin/update.rs"
if [ -f "$UPDATE_RS" ] && grep -q "fn main" "$UPDATE_RS"; then
    echo -e "${GREEN}✓ PASS${NC} (rusty-stack-update source exists with main fn)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (rusty-stack-update source not found or missing main)"
    ((FAILED++)) || true
fi

echo ""
echo "=========================================="
echo "Phase 3 Test Results"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ PHASE 3 VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}✅ ALL PHASE 3 TESTS PASSED${NC}"
    exit 0
fi
