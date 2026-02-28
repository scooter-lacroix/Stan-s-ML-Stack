#!/usr/bin/env bash
#
# Full Performance Benchmark Suite
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/benchmark_common.sh"

benchmark_enable_colors

PROJECT_ROOT="$(benchmark_resolve_project_root "$SCRIPT_DIR")"
MANIFEST_PATH="$(benchmark_discover_manifest_path "$PROJECT_ROOT")"

LOG_DIR="$(benchmark_resolve_log_dir)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/full_benchmarks_${RUN_TS}.log"
RESULT_JSON="$LOG_DIR/full_benchmarks_${RUN_TS}.json"
benchmark_set_log_file "$LOG_FILE"

benchmark_log "Starting Full Benchmark Suite..."
benchmark_log "Log file: $LOG_FILE"

benchmark_require_cargo
benchmark_prepare_rocm_runtime
benchmark_ensure_vllm_runtime_basics || benchmark_warn "vLLM runtime preflight could not fully repair dependencies; full suite will continue and report details"
benchmark_ensure_deepspeed_runtime_basics || benchmark_warn "DeepSpeed runtime preflight could not fully repair dependencies; full suite will continue and report details"
benchmark_info "Ensuring benchmark binary is ready..."
benchmark_build_rusty_stack_bench "$MANIFEST_PATH" "$LOG_FILE"
: "${VLLM_TARGET_DEVICE:=rocm}"
export VLLM_TARGET_DEVICE

benchmark_info "Running all benchmarks (pre and post install)..."
benchmark_run_named_json_to_file "$MANIFEST_PATH" "$LOG_FILE" "$RESULT_JSON" all

benchmark_log "Benchmark JSON results: $RESULT_JSON"
benchmark_success "Full Benchmark Suite completed successfully"
exit 0
