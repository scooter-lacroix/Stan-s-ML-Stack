#!/usr/bin/env bash
#
# ML Stack Benchmark Runner
# This script runs Python benchmark scripts in the benchmarks/ directory.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/benchmark_common.sh"

benchmark_enable_colors
benchmark_prepare_rocm_runtime

PROJECT_ROOT="$(benchmark_resolve_project_root "$SCRIPT_DIR")"
BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks"
RESULTS_BASE_DIR="${MLSTACK_BENCHMARK_RESULTS_DIR:-$PROJECT_ROOT/benchmark_results}"
RESULTS_DIR="$RESULTS_BASE_DIR/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
BENCHMARK_PYTHON="${MLSTACK_BENCHMARK_PYTHON:-${MLSTACK_PYTHON_BIN:-python3}}"

print_header() {
    benchmark_log "=== $1 ==="
}

print_section() {
    benchmark_log ">>> $1"
}

print_step() {
    benchmark_log "-> $1"
}

print_success() {
    benchmark_log "OK: $1"
}

print_warning() {
    benchmark_log "WARN: $1"
}

print_error() {
    benchmark_log "ERROR: $1"
}

run_python_benchmark() {
    local script_name="$1"
    local output_dir="$2"
    shift 2

    local script_path="$BENCHMARKS_DIR/$script_name"
    if [ ! -f "$script_path" ]; then
        print_error "Benchmark script not found: $script_path"
        return 1
    fi

    mkdir -p "$output_dir"

    if benchmark_is_dry_run; then
        print_step "[DRY-RUN] Would run: $BENCHMARK_PYTHON $script_path $* --output-dir $output_dir"
        return 0
    fi

    "$BENCHMARK_PYTHON" "$script_path" "$@" --output-dir "$output_dir"
}

run_matrix_multiplication_benchmark() {
    print_section "Running Matrix Multiplication Benchmark"
    if run_python_benchmark \
        matrix_multiplication_benchmark.py \
        "$RESULTS_DIR/matrix_multiplication" \
        --sizes 1024 2048 4096 8192 \
        --dtype float32 \
        --num-runs 5; then
        print_success "Matrix multiplication benchmark completed successfully"
        print_step "Results saved to $RESULTS_DIR/matrix_multiplication"
    else
        print_error "Matrix multiplication benchmark failed"
        return 1
    fi
}

run_memory_bandwidth_benchmark() {
    print_section "Running Memory Bandwidth Benchmark"
    if run_python_benchmark \
        memory_bandwidth_benchmark.py \
        "$RESULTS_DIR/memory_bandwidth" \
        --sizes 1 2 4 8 16 32 64 128 256 512 \
        --dtype float32 \
        --num-runs 5; then
        print_success "Memory bandwidth benchmark completed successfully"
        print_step "Results saved to $RESULTS_DIR/memory_bandwidth"
    else
        print_error "Memory bandwidth benchmark failed"
        return 1
    fi
}

run_transformer_benchmark() {
    print_section "Running Transformer Benchmark"
    if run_python_benchmark \
        transformer_benchmark.py \
        "$RESULTS_DIR/transformer" \
        --batch-sizes 1 2 4 8 \
        --seq-lengths 128 256 512 1024 \
        --d-model 512 \
        --nhead 8 \
        --dim-feedforward 2048 \
        --num-layers 6 \
        --dtype float32 \
        --num-runs 3; then
        print_success "Transformer benchmark completed successfully"
        print_step "Results saved to $RESULTS_DIR/transformer"
    else
        print_error "Transformer benchmark failed"
        return 1
    fi
}

run_flash_attention_benchmark() {
    print_section "Running Flash Attention Benchmark"

    if ! benchmark_is_dry_run && ! "$BENCHMARK_PYTHON" -c "import flash_attention_amd" >/dev/null 2>&1; then
        print_warning "Flash Attention is not installed. Skipping benchmark"
        return 0
    fi

    if run_python_benchmark \
        flash_attention_benchmark.py \
        "$RESULTS_DIR/flash_attention" \
        --batch-sizes 1 2 4 8 \
        --seq-lengths 128 256 512 1024 2048 \
        --num-heads 8 \
        --head-dim 64 \
        --causal \
        --dtype float32 \
        --num-runs 3; then
        print_success "Flash attention benchmark completed successfully"
        print_step "Results saved to $RESULTS_DIR/flash_attention"
    else
        print_error "Flash attention benchmark failed"
        return 1
    fi
}

run_all_benchmarks() {
    print_header "Running All Benchmarks"
    run_matrix_multiplication_benchmark
    run_memory_bandwidth_benchmark
    run_transformer_benchmark
    run_flash_attention_benchmark
    print_header "All Benchmarks Completed"
    print_step "Results saved to $RESULTS_DIR"
}

show_menu() {
    print_header "ML Stack Benchmark Menu"
    printf '%s\n' "1) Run Matrix Multiplication Benchmark"
    printf '%s\n' "2) Run Memory Bandwidth Benchmark"
    printf '%s\n' "3) Run Transformer Benchmark"
    printf '%s\n' "4) Run Flash Attention Benchmark"
    printf '%s\n' "5) Run All Benchmarks"
    printf '%s\n' "0) Exit"

    read -r -p "Enter your choice: " choice

    case "$choice" in
        1) run_matrix_multiplication_benchmark ;;
        2) run_memory_bandwidth_benchmark ;;
        3) run_transformer_benchmark ;;
        4) run_flash_attention_benchmark ;;
        5) run_all_benchmarks ;;
        0)
            print_header "Exiting ML Stack Benchmark Runner"
            exit 0
            ;;
        *) print_error "Invalid choice. Please try again." ;;
    esac

    show_menu
}

main() {
    print_header "ML Stack Benchmark Runner"
    print_step "Benchmark scripts: $BENCHMARKS_DIR"
    print_step "Results directory: $RESULTS_DIR"
    print_step "Benchmark Python: $BENCHMARK_PYTHON"

    if ! benchmark_python_exists "$BENCHMARK_PYTHON"; then
        print_error "Benchmark Python is not executable: $BENCHMARK_PYTHON"
        exit 1
    fi

    if benchmark_is_dry_run; then
        print_step "DRY_RUN=true detected; validating paths and benchmark commands"
        run_all_benchmarks
        return 0
    fi

    if ! "$BENCHMARK_PYTHON" -c "import torch" >/dev/null 2>&1; then
        print_error "PyTorch is not installed. Please install PyTorch first"
        exit 1
    fi

    if ! "$BENCHMARK_PYTHON" -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_error "CUDA/ROCm device is not available. Please check your GPU runtime installation"
        exit 1
    fi

    show_menu
}

main
