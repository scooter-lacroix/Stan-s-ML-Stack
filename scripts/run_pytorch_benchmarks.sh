#!/bin/bash
#
# PyTorch Performance Tests Wrapper
# This script runs PyTorch benchmarks via Rust benchmark binary
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pytorch_performance_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting PyTorch Performance Tests..."
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

# Build the benchmark binary
log "Building benchmark binary..."
ML_STACK_DIR="/home/stan/Documents/Stan-s-ML-Stack"
cd "$ML_STACK_DIR"
if ! cargo build --manifest-path="$ML_STACK_DIR/rusty-stack/Cargo.toml" --bin rusty-stack-bench 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Failed to build benchmark binary"
    exit 1
fi

# Run PyTorch benchmark
log "Running PyTorch benchmark..."
if ! "$ML_STACK_DIR/target/debug/rusty-stack-bench" pytorch --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Benchmark execution failed"
    exit 1
fi

# Run additional component benchmarks
log "Running GEMM benchmark..."
if ! "$ML_STACK_DIR/target/debug/rusty-stack-bench" gemm --json 2>&1 | tee -a "$LOG_FILE"; then
    log "Warning: GEMM benchmark failed"
fi

log "Running Flash Attention benchmark..."
if ! "$ML_STACK_DIR/target/debug/rusty-stack-bench" flash-attention --json 2>&1 | tee -a "$LOG_FILE"; then
    log "Warning: Flash Attention benchmark failed"
fi

log "PyTorch Performance Tests completed successfully"
exit 0
