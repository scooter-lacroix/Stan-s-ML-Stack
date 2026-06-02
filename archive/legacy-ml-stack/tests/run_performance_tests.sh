#!/bin/bash

# ML Stack Performance Tests Runner
# Runs performance benchmarks

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFORMANCE_DIR="$SCRIPT_DIR/performance"

echo "Running ML Stack Performance Benchmarks"
echo "======================================="

if [ ! -d "$PERFORMANCE_DIR" ]; then
    echo "Error: Performance tests directory not found: $PERFORMANCE_DIR"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not available. Please install Python 3.7+"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo

# Change to performance directory
cd "$PERFORMANCE_DIR"

# Run the main benchmark runner
echo "Starting benchmark suite..."
if $PYTHON_CMD benchmark_runner.py; then
    echo
    echo "✓ Performance benchmarks completed successfully"
    echo "Results saved to benchmark_results.json"
    echo "Report saved to benchmark_report.md"
    exit 0
else
    echo
    echo "✗ Performance benchmarks failed"
    exit 1
fi