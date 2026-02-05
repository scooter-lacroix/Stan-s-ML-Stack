#!/bin/bash
#
# DeepSpeed Performance Benchmark Wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deepspeed_benchmarks_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting DeepSpeed benchmarks..."
log "Log file: $LOG_FILE"

# Ensure cargo is available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

# Ensure ROCm variables
export DS_ACCELERATOR=rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Build the benchmark binary if needed
log "Ensuring benchmark binary is ready..."
if ! cargo build --manifest-path="rusty-stack/Cargo.toml" --bin rusty-stack-bench &> /dev/null; then
    log "ERROR: Failed to build benchmark binary"
    exit 1
fi

# Run DeepSpeed benchmark
log "Running DeepSpeed ZeRO throughput benchmark..."
if ! "./target/debug/rusty-stack-bench" deepspeed --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: DeepSpeed Benchmark execution failed"
    exit 1
fi

log "DeepSpeed benchmarks completed successfully"
exit 0
