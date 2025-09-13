#!/bin/bash

# ML Stack Complete Test Suite Runner
# Runs all tests: integration, verification, and performance

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Complete ML Stack Test Suite"
echo "===================================="

# Function to run a test category
run_test_category() {
    local script="$1"
    local category="$2"

    echo
    echo "Running $category tests..."
    echo "------------------------"

    if [ -f "$script" ]; then
        if bash "$script"; then
            echo "✓ $category tests PASSED"
            return 0
        else
            echo "✗ $category tests FAILED"
            return 1
        fi
    else
        echo "⚠ $category test script not found: $script"
        return 1
    fi
}

# Track results
TOTAL_CATEGORIES=0
PASSED_CATEGORIES=0
FAILED_CATEGORIES=0

# Run integration tests
((TOTAL_CATEGORIES++))
if run_test_category "$SCRIPT_DIR/run_integration_tests.sh" "Integration"; then
    ((PASSED_CATEGORIES++))
else
    ((FAILED_CATEGORIES++))
fi

# Run verification tests (run scripts in verification directory)
((TOTAL_CATEGORIES++))
echo
echo "Running Verification tests..."
echo "----------------------------"

VERIFICATION_DIR="$SCRIPT_DIR/verification"
if [ -d "$VERIFICATION_DIR" ]; then
    VERIFICATION_PASSED=0
    VERIFICATION_FAILED=0
    VERIFICATION_TOTAL=$(find "$VERIFICATION_DIR" -name "*.sh" -type f | wc -l)

    for test_script in "$VERIFICATION_DIR"/*.sh; do
        if [ -f "$test_script" ]; then
            echo "Running $(basename "$test_script")..."
            if bash "$test_script"; then
                echo "✓ $(basename "$test_script") PASSED"
                ((VERIFICATION_PASSED++))
            else
                echo "✗ $(basename "$test_script") FAILED"
                ((VERIFICATION_FAILED++))
            fi
        fi
    done

    if [ $VERIFICATION_FAILED -gt 0 ]; then
        echo "✗ Verification tests FAILED ($VERIFICATION_PASSED/$VERIFICATION_TOTAL passed)"
        ((FAILED_CATEGORIES++))
    else
        echo "✓ Verification tests PASSED ($VERIFICATION_PASSED/$VERIFICATION_TOTAL passed)"
        ((PASSED_CATEGORIES++))
    fi
else
    echo "⚠ Verification directory not found"
    ((FAILED_CATEGORIES++))
fi

# Run performance tests
((TOTAL_CATEGORIES++))
if run_test_category "$SCRIPT_DIR/run_performance_tests.sh" "Performance"; then
    ((PASSED_CATEGORIES++))
else
    ((FAILED_CATEGORIES++))
fi

echo
echo "Complete Test Suite Summary"
echo "==========================="
echo "Categories: $TOTAL_CATEGORIES"
echo "Passed: $PASSED_CATEGORIES"
echo "Failed: $FAILED_CATEGORIES"

if [ $FAILED_CATEGORIES -gt 0 ]; then
    echo
    echo "Some test categories failed. Check output above for details."
    exit 1
else
    echo
    echo "🎉 All test categories passed! ML Stack is ready."
    exit 0
fi