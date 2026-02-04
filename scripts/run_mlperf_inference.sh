#!/bin/bash
#
# MLPerf Inference Benchmark Wrapper
# This script runs the Rust benchmark binary and captures output
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mlperf_inference_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting MLPerf Inference benchmark..."
log "Log file: $LOG_FILE"

# Ensure cargo is available
if ! command -v cargo &> /dev/null; then
    # Try to source cargo environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    elif [ -f "$HOME/.rustup/env" ]; then
        source "$HOME/.rustup/env"
    fi
fi

if ! command -v cargo &> /dev/null; then
    log "ERROR: cargo not found in PATH"
    log "PATH is: $PATH"
    exit 1
fi

# Build the benchmark binary first
log "Building benchmark binary..."
ML_STACK_DIR="/home/stan/Documents/Stan-s-ML-Stack"
cd "$ML_STACK_DIR"
if ! cargo build --manifest-path="$ML_STACK_DIR/rusty-stack/Cargo.toml" --bin rusty-stack-bench 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Failed to build benchmark binary"
    exit 1
fi

# Run the benchmark
log "Running all benchmarks..."
if ! "$ML_STACK_DIR/target/debug/rusty-stack-bench" all --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Benchmark execution failed"
    exit 1
fi

log "MLPerf Inference benchmark completed successfully"
exit 0
