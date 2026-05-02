#!/bin/bash
# Git Tag Validation Test
# Tests that all pinned Git tags ACTUALLY exist in their repositories

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

# Function to test if a Git tag exists
test_git_tag() {
    local repo="$1"
    local tag="$2"
    local description="$3"

    echo -n "Testing: $description ($tag)... "

    # Create temp directory for cloning
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN

    # Clone shallow (just the tag we need)
    if git clone --depth 1 --branch "$tag" --single-branch "$repo" "$tmpdir" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC} (Tag exists)"
        ((PASSED++)) || true
        return 0
    else
        # Tag might not exist, try to fetch and check
        cd "$tmpdir"
        if git fetch --tags origin "$tag" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ PASS${NC} (Tag exists via fetch)"
            ((PASSED++)) || true
            cd -
            return 0
        else
            echo -e "${RED}✗ FAIL${NC} (Tag not found)"
            echo "  Repo: $repo"
            echo "  Tag: $tag"
            ((FAILED++)) || true
            cd -
            return 1
        fi
    fi
}

# Function to test if a Git branch exists
test_git_branch() {
    local repo="$1"
    local branch="$2"
    local description="$3"

    echo -n "Testing: $description ($branch)... "

    # Create temp directory for cloning
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN

    # Clone shallow and try to checkout branch
    if git clone --depth 1 "$repo" "$tmpdir" >/dev/null 2>&1; then
        cd "$tmpdir"
        if git checkout "$branch" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ PASS${NC} (Branch exists)"
            ((PASSED++)) || true
            cd -
            return 0
        else
            echo -e "${RED}✗ FAIL${NC} (Branch not found)"
            echo "  Repo: $repo"
            echo "  Branch: $branch"
            ((FAILED++)) || true
            cd -
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ SKIP${NC} (Could not clone repo)"
        echo "  Repo: $repo"
        ((SKIPPED++)) || true
        return 2
    fi
}

# Function to check if repository is accessible
test_repo_accessible() {
    local repo="$1"
    local description="$2"

    echo -n "Testing: $description (repo accessible)... "

    if git ls-remote --heads "$repo" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Repository not accessible)"
        echo "  Repo: $repo"
        ((FAILED++)) || true
        return 1
    fi
}

echo "=========================================="
echo "Git Tag Validation Test Suite"
echo "Testing ACTUAL Git tag/branch existence"
echo "=========================================="
echo ""

# Test repository accessibility first
echo "[Preliminary] Repository Accessibility"
test_repo_accessible "https://github.com/ROCm/flash-attention.git" "Flash Attention"
test_repo_accessible "https://github.com/ROCm/triton.git" "Triton ROCm"
test_repo_accessible "https://github.com/vllm-project/vllm.git" "vLLM"
echo ""

# Test Flash Attention tags
echo "[1/3] Flash Attention Tags"
test_git_tag "https://github.com/ROCm/flash-attention.git" "v2.8.0-cktile" "Flash Attention stable tag"
test_git_branch "https://github.com/ROCm/flash-attention.git" "main_perf" "Flash Attention preview branch"
echo ""

# Test Triton tags
echo "[2/3] Triton Tags"
test_git_branch "https://github.com/ROCm/triton.git" "3.2.0" "Triton stable branch"
test_git_branch "https://github.com/ROCm/triton.git" "triton-mlir" "Triton MLIR branch"
test_git_branch "https://github.com/ROCm/triton.git" "main" "Triton main branch"
echo ""

# Test vLLM tags
echo "[3/3] vLLM Tags"
test_git_tag "https://github.com/vllm-project/vllm.git" "v0.15.0" "vLLM stable tag"
test_git_branch "https://github.com/vllm-project/vllm.git" "main" "vLLM main branch"
echo ""

echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${YELLOW}SKIPPED: $SKIPPED${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ GIT TAG VALIDATION FAILED${NC}"
    echo "Some Git tags/branches do not exist. This will cause installation failures."
    echo "Please update the scripts with valid tags."
    exit 1
elif [ $SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}⚠️  GIT TAG VALIDATION COMPLETED WITH SKIPPED TESTS${NC}"
    echo "Some repositories could not be cloned for testing."
    exit 0
else
    echo -e "${GREEN}✅ ALL GIT TAG VALIDATION CHECKS PASSED${NC}"
    echo "All pinned Git tags/branches exist and are accessible."
    exit 0
fi
