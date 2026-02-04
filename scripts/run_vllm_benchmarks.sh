#!/bin/bash
#
# vLLM Performance Benchmark Wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_benchmarks_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting vLLM benchmarks..."
log "Log file: $LOG_FILE"

# Ensure cargo is available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    elif [ -f "$HOME/.rustup/env" ]; then
        source "$HOME/.rustup/env"
    fi
fi

# Build the benchmark binary if needed
log "Ensuring benchmark binary is ready..."
if ! cargo build --manifest-path="rusty-stack/Cargo.toml" --bin rusty-stack-bench &> /tmp/bench_build.log; then
    log "ERROR: Failed to build benchmark binary"
    cat /tmp/bench_build.log | tee -a "$LOG_FILE"
    exit 1
fi

# Run vLLM benchmark
log "Running vLLM throughput benchmark..."
export VLLM_TARGET_DEVICE=rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
if ! "./target/debug/rusty-stack-bench" vllm --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: vLLM Benchmark execution failed"
    exit 1
fi

log "vLLM benchmarks completed successfully"
exit 0
