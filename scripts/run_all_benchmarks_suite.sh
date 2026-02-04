#!/bin/bash
#
# Full Performance Benchmark Suite
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="${HOME}/.rusty-stack/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_benchmarks_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting Full Benchmark Suite..."

# Ensure cargo is available
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

# Run all benchmarks
log "Running all benchmarks (pre and post install)..."
export VLLM_TARGET_DEVICE=rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
if ! cargo build --manifest-path="rusty-stack/Cargo.toml" --bin rusty-stack-bench &> /dev/null; then
    log "ERROR: Failed to build benchmark binary"
    exit 1
fi
if ! "./target/debug/rusty-stack-bench" all --json 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Full Benchmark execution failed"
    exit 1
fi

log "Full Benchmark Suite completed successfully"
exit 0
