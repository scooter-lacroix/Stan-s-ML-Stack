#!/usr/bin/env bash
#
# PyTorch Performance Tests Wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/benchmark_common.sh"

benchmark_enable_colors

PROJECT_ROOT="$(benchmark_resolve_project_root "$SCRIPT_DIR")"
MANIFEST_PATH="$(benchmark_discover_manifest_path "$PROJECT_ROOT")"

LOG_DIR="$(benchmark_resolve_log_dir)"
LOG_FILE="$LOG_DIR/pytorch_performance_$(date +%Y%m%d_%H%M%S).log"
benchmark_set_log_file "$LOG_FILE"

benchmark_log "Starting PyTorch Performance Tests..."
benchmark_log "Log file: $LOG_FILE"

benchmark_require_cargo
benchmark_prepare_rocm_runtime
benchmark_info "Building benchmark binary..."
benchmark_build_rusty_stack_bench "$MANIFEST_PATH" "$LOG_FILE"

benchmark_info "Running PyTorch benchmark..."
benchmark_run_named_json "$MANIFEST_PATH" "$LOG_FILE" pytorch

benchmark_info "Running GEMM benchmark..."
if ! benchmark_run_named_json "$MANIFEST_PATH" "$LOG_FILE" gemm; then
    benchmark_warn "GEMM benchmark failed"
fi

benchmark_info "Running Flash Attention benchmark..."
if ! benchmark_run_named_json "$MANIFEST_PATH" "$LOG_FILE" flash-attention; then
    benchmark_warn "Flash Attention benchmark failed"
fi

benchmark_success "PyTorch Performance Tests completed successfully"
exit 0
