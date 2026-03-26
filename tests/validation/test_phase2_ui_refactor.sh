#!/bin/bash
# Phase 2 Validation Tests - UI Module Refactor + text-generation-webui
# Tests for: ui_installer_helper.sh, ComfyUI/vLLM Studio refactors, textgen component

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
HELPER="$PROJECT_ROOT/scripts/lib/ui_installer_helper.sh"

echo "=========================================="
echo "Phase 2 Validation Tests"
echo "UI Module Refactor + text-generation-webui"
echo "=========================================="
echo ""

# Test 2.1: ui_installer_helper.sh has all 7 functions
echo "[1/7] Testing ui_installer_helper.sh has all 7 required functions"
REQUIRED_FUNCS=("ui_parse_common_args" "ui_git_clone_or_update" "ui_fix_ownership" "ui_create_launcher_shim" "ui_create_systemd_service" "ui_detect_gpu_devices" "ui_print_summary")
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
        echo -e "${GREEN}✓ PASS${NC} (All 7 functions present)"
        ((PASSED++)) || true
    else
        echo -e "${RED}✗ FAIL${NC} ($MISSING functions missing)"
        ((FAILED++)) || true
    fi
else
    echo -e "${RED}✗ FAIL${NC} ($HELPER not found)"
    ((FAILED++)) || true
fi

# Test 2.2: install_comfyui.sh sources helper
echo "[2/7] Testing install_comfyui.sh sources ui_installer_helper.sh"
if grep -q "ui_installer_helper.sh" "$PROJECT_ROOT/scripts/install_comfyui.sh"; then
    echo -e "${GREEN}✓ PASS${NC} (ComfyUI sources helper)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (ComfyUI does not source helper)"
    ((FAILED++)) || true
fi

# Test 2.3: install_comfyui.sh has declare -f guards
echo "[3/7] Testing install_comfyui.sh has declare -f guard pattern"
COMFYUI_FUNCS=("ui_parse_common_args" "ui_git_clone_or_update" "ui_fix_ownership" "ui_detect_gpu_devices" "ui_create_launcher_shim" "ui_print_summary")
GUARD_COUNT=0
for func in "${COMFYUI_FUNCS[@]}"; do
    if grep -q "declare -f $func" "$PROJECT_ROOT/scripts/install_comfyui.sh"; then
        ((GUARD_COUNT++)) || true
    fi
done
if [ $GUARD_COUNT -eq ${#COMFYUI_FUNCS[@]} ]; then
    echo -e "${GREEN}✓ PASS${NC} (All ${#COMFYUI_FUNCS[@]} functions have declare -f guards)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Only $GUARD_COUNT/${#COMFYUI_FUNCS[@]} functions have guards)"
    ((FAILED++)) || true
fi

# Test 2.4: install_textgen.sh exists and is executable
echo "[4/7] Testing install_textgen.sh exists and is executable"
TEXTGEN="$PROJECT_ROOT/scripts/install_textgen.sh"
if [ -x "$TEXTGEN" ]; then
    echo -e "${GREEN}✓ PASS${NC} (install_textgen.sh is executable)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (install_textgen.sh missing or not executable)"
    ((FAILED++)) || true
fi

# Test 2.5: install_textgen.sh filters nvidia/CUDA packages
echo "[5/7] Testing install_textgen.sh filters nvidia/CUDA packages"
CUDA_PATTERNS=("nvidia-" "cuda" "tensorrt" "xformers" "flash-attn")
PATTERN_COUNT=0
for pattern in "${CUDA_PATTERNS[@]}"; do
    if grep -q "$pattern" "$TEXTGEN"; then
        ((PATTERN_COUNT++)) || true
    fi
done
if [ $PATTERN_COUNT -ge 3 ]; then
    echo -e "${GREEN}✓ PASS${NC} ($PATTERN_COUNT/5 CUDA exclusion patterns found)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Only $PATTERN_COUNT/5 CUDA exclusion patterns found)"
    ((FAILED++)) || true
fi

# Test 2.6: textgen component in state.rs
echo "[6/7] Testing textgen component registered in state.rs"
if grep -q '"textgen"' "$PROJECT_ROOT/rusty-stack/src/state.rs" && \
   grep -q 'install_textgen.sh' "$PROJECT_ROOT/rusty-stack/src/state.rs"; then
    echo -e "${GREEN}✓ PASS${NC} (textgen component in state.rs)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (textgen component not found in state.rs)"
    ((FAILED++)) || true
fi

# Test 2.7: textgen detection in component_status.rs
echo "[7/7] Testing textgen detection in component_status.rs"
if grep -q '"textgen"' "$PROJECT_ROOT/rusty-stack/src/component_status.rs" && \
   grep -q 'server.py' "$PROJECT_ROOT/rusty-stack/src/component_status.rs"; then
    echo -e "${GREEN}✓ PASS${NC} (textgen detection in component_status.rs)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (textgen detection not found in component_status.rs)"
    ((FAILED++)) || true
fi

echo ""
echo "=========================================="
echo "Phase 2 Test Results"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ PHASE 2 VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}✅ ALL PHASE 2 TESTS PASSED${NC}"
    exit 0
fi
