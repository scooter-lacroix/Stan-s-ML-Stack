#!/bin/bash
# Shell Script Syntax Validation Test
# Tests that all modified shell scripts have valid syntax

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

# List of modified scripts to validate
SCRIPTS=(
    "scripts/install_rocm.sh"
    "scripts/install_rocm_channel.sh"
    "scripts/install_pytorch_multi.sh"
    "scripts/build_flash_attn_amd.sh"
    "scripts/install_triton_multi.sh"
    "scripts/install_vllm_multi.sh"
    "scripts/install_bitsandbytes_multi.sh"
    "scripts/install_migraphx_multi.sh"
    "scripts/install_rccl_multi.sh"
    "scripts/install_pytorch_rocm.sh"
    "scripts/build_onnxruntime_multi.sh"
    "scripts/enhanced_verify_installation.sh"
    "scripts/gpu_detection_utils.sh"
    "scripts/env_validation_utils.sh"
)

echo "=========================================="
echo "Shell Script Syntax Validation"
echo "Testing ACTUAL script syntax validity"
echo "=========================================="
echo ""

for script in "${SCRIPTS[@]}"; do
    echo -n "Testing: $script... "

    if [ ! -f "$script" ]; then
        echo -e "${YELLOW}⚠ SKIP${NC} (File not found)"
        ((SKIPPED++)) || true
        continue
    fi

    # Test syntax with bash -n
    if bash -n "$script" 2>&1; then
        echo -e "${GREEN}✓ PASS${NC} (Syntax valid)"
        ((PASSED++)) || true
    else
        echo -e "${RED}✗ FAIL${NC} (Syntax error)"
        echo "  Error output:"
        bash -n "$script" 2>&1 | sed 's/^/    /'
        ((FAILED++)) || true
    fi
done

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${YELLOW}SKIPPED: $SKIPPED:-0${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ SCRIPT SYNTAX VALIDATION FAILED${NC}"
    echo "Some scripts have syntax errors and will not run."
    exit 1
else
    echo -e "${GREEN}✅ ALL SCRIPT SYNTAX CHECKS PASSED${NC}"
    echo "All scripts have valid syntax."
    exit 0
fi
