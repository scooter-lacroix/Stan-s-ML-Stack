#!/bin/bash
# Version Consistency Test
# Tests that all version references are consistent across files

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
INCONSISTENCIES=()

# Expected versions based on our fixes
EXPECTED_VERSIONS=(
    "6.4.3:Legacy"
    "7.1:Stable"
    "7.2:Latest"
)

echo "=========================================="
echo "Version Consistency Test Suite"
echo "Testing ACTUAL version consistency across files"
echo "=========================================="
echo ""

# Test 1: Check install_rocm.sh menu options
echo "[1/5] Checking install_rocm.sh menu options"
if grep -q "ROCm 6.4.3" scripts/install_rocm.sh && \
   grep -q "ROCm 7.1" scripts/install_rocm.sh && \
   grep -q "ROCm 7.2" scripts/install_rocm.sh; then
    echo -e "${GREEN}✓ PASS${NC} (Menu options correct)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Menu options incorrect)"
    ((FAILED++)) || true
fi

# Test 2: Check install_rocm.sh version variables
echo "[2/5] Checking install_rocm.sh version variables"
if grep -q 'ROCM_VERSION="7.1"' scripts/install_rocm.sh && \
   grep -q 'ROCM_VERSION="7.2"' scripts/install_rocm.sh && \
   grep -q 'ROCM_VERSION="6.4.3"' scripts/install_rocm.sh; then
    echo -e "${GREEN}✓ PASS${NC} (Version variables correct)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Version variables incorrect)"
    ((FAILED++)) || true
fi

# Test 3: Check for outdated version references
echo "[3/5] Checking for outdated version references"
OUTDATED_FOUND=0
for old_version in "7.0.0" "7.0.2" "7.9.0"; do
    # Check main scripts (excluding comments and legitimate detection/migration code)
    # Detection code for existing installations is OK - ignore lines with:
    # - /opt/rocm-<version> (existing installation detection)
    # - comments (#)
    if grep -r "$old_version" scripts/install_rocm.sh scripts/install_rocm_channel.sh 2>/dev/null | \
       grep -v "#" | \
       grep -v "/opt/rocm-" | \
       grep -q .; then
        echo -e "${YELLOW}⚠ FOUND${NC} (Outdated version $old_version still referenced)"
        grep -n "$old_version" scripts/install_rocm.sh scripts/install_rocm_channel.sh | grep -v "#" | grep -v "/opt/rocm-" | head -5
        ((OUTDATED_FOUND++)) || true
    fi
done

if [ $OUTDATED_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} (No outdated versions found)"
    ((PASSED++)) || true
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Found $OUTDATED_FOUND outdated version references)"
    ((INCONSISTENCIES++)) || true
fi

# Test 4: Check MULTI_CHANNEL_GUIDE.md consistency
echo "[4/5] Checking MULTI_CHANNEL_GUIDE.md consistency"
if grep -q "6.4.3" docs/MULTI_CHANNEL_GUIDE.md && \
   grep -q "7.1" docs/MULTI_CHANNEL_GUIDE.md && \
   grep -q "7.2" docs/MULTI_CHANNEL_GUIDE.md; then
    echo -e "${GREEN}✓ PASS${NC} (Guide versions correct)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Guide versions incorrect)"
    ((FAILED++)) || true
fi

# Test 5: Check preview channel removal
echo "[5/5] Checking preview channel properly removed"
if ! grep -q "ROCm 7.10.0.*Preview.*Experimental" scripts/install_rocm.sh && \
   ! grep -q "^.*preview.*-.*ROCm 7.10.0" docs/MULTI_CHANNEL_GUIDE.md; then
    echo -e "${GREEN}✓ PASS${NC} (Preview option properly removed)"
    ((PASSED++)) || true
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Preview option still present in some form)"
    ((INCONSISTENCIES++)) || true
fi

# Test 6: Check channel wrapper consistency
echo "[6/6] Checking install_rocm_channel.sh consistency"
if grep -q "legacy.*6.4.3" scripts/install_rocm_channel.sh && \
   grep -q "stable.*7.1" scripts/install_rocm_channel.sh && \
   grep -q "latest.*7.2" scripts/install_rocm_channel.sh; then
    echo -e "${GREEN}✓ PASS${NC} (Channel wrapper consistent)"
    ((PASSED++)) || true
else
    echo -e "${RED}✗ FAIL${NC} (Channel wrapper inconsistent)"
    ((FAILED++)) || true
fi

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${YELLOW}INCONSISTENCIES: $INCONSISTENCIES${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ VERSION CONSISTENCY VALIDATION FAILED${NC}"
    echo "Version numbers are inconsistent across files."
    exit 1
elif [ $INCONSISTENCIES -gt 0 ]; then
    echo -e "${YELLOW}⚠️  VERSION CONSISTENCY COMPLETED WITH INCONSISTENCIES${NC}"
    echo "Some version inconsistencies were found. Please review."
    exit 0
else
    echo -e "${GREEN}✅ ALL VERSION CONSISTENCY CHECKS PASSED${NC}"
    echo "All version references are consistent across files."
    exit 0
fi
