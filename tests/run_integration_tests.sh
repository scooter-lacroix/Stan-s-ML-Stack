#!/bin/bash

# ML Stack Integration Tests Runner
# Runs all integration tests in the integration directory

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$SCRIPT_DIR/integration"

echo "Running ML Stack Integration Tests"
echo "=================================="

if [ ! -d "$INTEGRATION_DIR" ]; then
    echo "Error: Integration tests directory not found: $INTEGRATION_DIR"
    exit 1
fi

# Count total tests
TOTAL_TESTS=$(find "$INTEGRATION_DIR" -name "*.sh" -type f | wc -l)
echo "Found $TOTAL_TESTS integration tests to run"
echo

PASSED=0
FAILED=0

# Run each test script
for test_script in "$INTEGRATION_DIR"/*.sh; do
    if [ -f "$test_script" ]; then
        echo "Running $(basename "$test_script")..."
        if bash "$test_script"; then
            echo "✓ $(basename "$test_script") PASSED"
            ((PASSED++))
        else
            echo "✗ $(basename "$test_script") FAILED"
            ((FAILED++))
        fi
        echo
    fi
done

echo "Integration Tests Summary"
echo "========================"
echo "Total: $TOTAL_TESTS"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo "Some tests failed. Check output above for details."
    exit 1
else
    echo "All integration tests passed!"
    exit 0
fi