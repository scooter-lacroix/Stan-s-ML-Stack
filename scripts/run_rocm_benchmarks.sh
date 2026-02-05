#!/bin/bash
#
# ROCm Benchmarks Wrapper
# This script runs ROCm-specific benchmarks via Rust benchmark binary
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rocm_benchmarks_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting ROCm benchmarks..."
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
PROJECT_ROOT="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
cd "$PROJECT_ROOT"
if ! cargo build --manifest-path="rusty-stack/Cargo.toml" --bin rusty-stack-bench 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Failed to build benchmark binary"
    exit 1
fi

# Run pre-installation benchmarks
log "Running ROCm pre-installation benchmarks..."
if ! "./target/debug/rusty-stack-bench" all-pre --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Benchmark execution failed"
    exit 1
fi

log "ROCm benchmarks completed successfully"
exit 0
