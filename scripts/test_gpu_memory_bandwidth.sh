#!/usr/bin/env bash
#
# GPU Memory Bandwidth Test Wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/benchmark_common.sh"

benchmark_enable_colors

PROJECT_ROOT="$(benchmark_resolve_project_root "$SCRIPT_DIR")"
MANIFEST_PATH="$(benchmark_discover_manifest_path "$PROJECT_ROOT")"

LOG_DIR="$(benchmark_resolve_log_dir)"
LOG_FILE="$LOG_DIR/gpu_memory_bandwidth_$(date +%Y%m%d_%H%M%S).log"
benchmark_set_log_file "$LOG_FILE"

benchmark_log "Starting GPU memory bandwidth benchmark..."
benchmark_log "Log file: $LOG_FILE"

benchmark_require_cargo
benchmark_prepare_rocm_runtime
benchmark_info "Ensuring benchmark binary is ready..."
benchmark_build_rusty_stack_bench "$MANIFEST_PATH" "$LOG_FILE"

benchmark_info "Running memory-bandwidth benchmark..."
benchmark_run_named_json "$MANIFEST_PATH" "$LOG_FILE" memory-bandwidth

benchmark_success "GPU memory bandwidth benchmark completed successfully"
exit 0
