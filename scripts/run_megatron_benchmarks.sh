#!/usr/bin/env bash
#
# Megatron-LM Performance Benchmark Wrapper
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
LOG_FILE="$LOG_DIR/megatron_benchmarks_${RUN_TS}.log"
RESULT_JSON="$LOG_DIR/megatron_benchmarks_${RUN_TS}.json"
benchmark_set_log_file "$LOG_FILE"

benchmark_log "Starting Megatron-LM benchmarks..."
benchmark_log "Log file: $LOG_FILE"

benchmark_require_cargo
benchmark_prepare_rocm_runtime
benchmark_info "Ensuring benchmark binary is ready..."
benchmark_build_rusty_stack_bench "$MANIFEST_PATH" "$LOG_FILE"

benchmark_info "Running Megatron-LM throughput benchmark..."
benchmark_run_named_json_to_file "$MANIFEST_PATH" "$LOG_FILE" "$RESULT_JSON" megatron

benchmark_log "Benchmark JSON results: $RESULT_JSON"

benchmark_success "Megatron-LM benchmarks completed successfully"
exit 0
