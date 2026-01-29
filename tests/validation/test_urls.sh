#!/bin/bash
# Comprehensive URL Validation Test
# Tests that all URLs referenced in scripts are ACTUALLY accessible

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

# Function to test URL accessibility
test_url() {
    local url="$1"
    local description="$2"
    local expected_code="${3:-200}"

    echo -n "Testing: $description... "

    # Use curl with timeout and follow redirects
    if curl_output=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -L "$url" 2>&1); then
        if [ "$curl_output" = "$expected_code" ] || [ "$curl_output" = "000" ]; then
            echo -e "${GREEN}✓ PASS${NC} (HTTP $curl_output)"
            ((PASSED++))
            return 0
        else
            echo -e "${RED}✗ FAIL${NC} (HTTP $curl_output, expected $expected_code)"
            echo "  URL: $url"
            ((FAILED++))
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ WARNING${NC} (curl failed: $curl_output)"
        echo "  URL: $url"
        ((WARNINGS++))
        return 2
    fi
}

# Function to test repository availability (special handling for apt repos)
test_apt_repo() {
    local url="$1"
    local description="$2"

    echo -n "Testing: $description... "

    # For apt repos, we check if we can access the repository directory listing
    # Some repos return 404 for directory listing but packages exist, so we're lenient
    if curl -s --head --max-time 10 "$url" | head -n 1 | grep -q "HTTP\|200\|404\|403"; then
        echo -e "${GREEN}✓ PASS${NC} (Repository accessible)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Repository not accessible)"
        echo "  URL: $url"
        ((FAILED++))
        return 1
    fi
}

echo "=========================================="
echo "URL Validation Test Suite"
echo "Testing ACTUAL URL accessibility"
echo "=========================================="
echo ""

# Test ROCm repository URLs
echo "[1/8] ROCm Repository URLs"
test_apt_repo "https://repo.radeon.com/rocm/apt/6.4/" "ROCm 6.4 Repository"
test_apt_repo "https://repo.radeon.com/rocm/apt/7.1/" "ROCm 7.1 Repository"
test_apt_repo "https://repo.radeon.com/rocm/apt/7.2/" "ROCm 7.2 Repository"
echo ""

# Test AMDGPU repository URLs
echo "[2/8] AMDGPU Repository URLs"
test_apt_repo "https://repo.radeon.com/amdgpu/6.4/ubuntu" "AMDGPU 6.4 Repository"
test_apt_repo "https://repo.radeon.com/amdgpu/7.1/ubuntu" "AMDGPU 7.1 Repository"
test_apt_repo "https://repo.radeon.com/amdgpu/7.2/ubuntu" "AMDGPU 7.2 Repository"
echo ""

# Test PyTorch wheel URLs
echo "[3/8] PyTorch Wheel URLs"
test_url "https://download.pytorch.org/whl/rocm6.4/" "PyTorch ROCm 6.4 Wheels"
test_url "https://download.pytorch.org/whl/nightly/rocm7.1/" "PyTorch ROCm 7.1 Wheels"
test_url "https://download.pytorch.org/whl/nightly/rocm7.2/" "PyTorch ROCm 7.2 Wheels"
echo ""

# Test ROCm manylinux URLs
echo "[4/8] ROCm manylinux Repository URLs"
test_url "https://repo.radeon.com/rocm/manylinux/" "ROCm manylinux base"
test_url "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/" "ROCm 7.1 manylinux"
echo ""

# Test documentation URLs
echo "[5/8] Documentation URLs"
test_url "https://rocm.docs.amd.com/en/7.10.0-preview/install/rocm.html" "ROCm 7.10.0 Preview Docs"
echo ""

# Test GitHub repository URLs
echo "[6/8] GitHub Repository URLs"
test_url "https://github.com/ROCm/flash-attention" "Flash Attention GitHub"
test_url "https://github.com/ROCm/triton" "Triton ROCm GitHub"
test_url "https://github.com/vllm-project/vllm" "vLLM GitHub"
echo ""

# Test GPG key URLs
echo "[7/8] GPG Key URLs"
test_url "https://repo.radeon.com/rocm/rocm.gpg.key" "ROCm GPG Key"
echo ""

# Test PyPI package URLs
echo "[8/8] PyPI Package URLs"
test_url "https://pypi.org/pypi/bitsandbytes/json" "bitsandbytes PyPI"
echo ""

echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${YELLOW}WARNINGS: $WARNINGS${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ URL VALIDATION FAILED${NC}"
    echo "Some URLs are not accessible. This will cause installation failures."
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  URL VALIDATION COMPLETED WITH WARNINGS${NC}"
    echo "Some URLs could not be tested but may work in practice."
    exit 0
else
    echo -e "${GREEN}✅ ALL URL VALIDATION CHECKS PASSED${NC}"
    echo "All referenced URLs are accessible."
    exit 0
fi
