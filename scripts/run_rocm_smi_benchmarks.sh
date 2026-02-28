#!/usr/bin/env bash
#
# ROCm SMI Benchmarks Wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/benchmark_common.sh"

benchmark_enable_colors

PROJECT_ROOT="$(benchmark_resolve_project_root "$SCRIPT_DIR")"
MANIFEST_PATH="$(benchmark_discover_manifest_path "$PROJECT_ROOT")"

LOG_DIR="$(benchmark_resolve_log_dir)"
LOG_FILE="$LOG_DIR/rocm_smi_benchmarks_$(date +%Y%m%d_%H%M%S).log"
benchmark_set_log_file "$LOG_FILE"

benchmark_log "Starting ROCm SMI benchmarks..."
benchmark_log "Log file: $LOG_FILE"

benchmark_require_cargo
benchmark_prepare_rocm_runtime
benchmark_info "Building benchmark binary..."
benchmark_build_rusty_stack_bench "$MANIFEST_PATH" "$LOG_FILE"

benchmark_info "Running GPU capability benchmark..."
benchmark_run_named_json "$MANIFEST_PATH" "$LOG_FILE" gpu-capability

if command -v rocm-smi >/dev/null 2>&1; then
    if benchmark_is_dry_run; then
        benchmark_info "[DRY-RUN] Would run: rocm-smi"
    else
        benchmark_info "Running rocm-smi..."
        rocm-smi 2>&1 | tee -a "$LOG_FILE" || true
    fi
else
    benchmark_warn "rocm-smi not found in PATH"
fi

benchmark_success "ROCm SMI benchmarks completed successfully"
exit 0
