#!/bin/bash
#
# GPU Memory Bandwidth Test Wrapper
# This script runs memory bandwidth benchmarks via Rust benchmark binary
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gpu_memory_bandwidth_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting GPU Memory Bandwidth test..."
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

# Run memory bandwidth benchmark
log "Running memory bandwidth benchmark..."
if ! "$ML_STACK_DIR/target/debug/rusty-stack-bench" memory-bandwidth --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Benchmark execution failed"
    exit 1
fi

log "GPU Memory Bandwidth test completed successfully"
exit 0
