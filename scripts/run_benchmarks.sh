#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# ML Stack Benchmark Runner
# =============================================================================
# This script runs benchmarks for the ML Stack on AMD GPUs.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                                                                 
                                ML Stack Benchmark Runner
EOF
echo

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
BLINK='\033[5m'
REVERSE='\033[7m'
RESET='\033[0m'

# Function definitions
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

# Create results directory
RESULTS_DIR="$HOME/Desktop/Stans_MLStack/benchmark_results/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $RESULTS_DIR

# Function to run matrix multiplication benchmark
run_matrix_multiplication_benchmark() {
    print_section "Running Matrix Multiplication Benchmark"
    
    # Check if the benchmark script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/benchmarks/matrix_multiplication_benchmark.py" ]; then
        print_error "Matrix multiplication benchmark script not found."
        return 1
    fi
    
    # Run the benchmark
    print_step "Running benchmark..."
    python3 $HOME/Desktop/Stans_MLStack/benchmarks/matrix_multiplication_benchmark.py \
        --sizes 1024 2048 4096 8192 \
        --dtype float32 \
        --num-runs 5 \
        --output-dir $RESULTS_DIR/matrix_multiplication
    
    if [ $? -eq 0 ]; then
        print_success "Matrix multiplication benchmark completed successfully."
        print_step "Results saved to $RESULTS_DIR/matrix_multiplication"
    else
        print_error "Matrix multiplication benchmark failed."
        return 1
    fi
    
    return 0
}

# Function to run memory bandwidth benchmark
run_memory_bandwidth_benchmark() {
    print_section "Running Memory Bandwidth Benchmark"
    
    # Check if the benchmark script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/benchmarks/memory_bandwidth_benchmark.py" ]; then
        print_error "Memory bandwidth benchmark script not found."
        return 1
    fi
    
    # Run the benchmark
    print_step "Running benchmark..."
    python3 $HOME/Desktop/Stans_MLStack/benchmarks/memory_bandwidth_benchmark.py \
        --sizes 1 2 4 8 16 32 64 128 256 512 \
        --dtype float32 \
        --num-runs 5 \
        --output-dir $RESULTS_DIR/memory_bandwidth
    
    if [ $? -eq 0 ]; then
        print_success "Memory bandwidth benchmark completed successfully."
        print_step "Results saved to $RESULTS_DIR/memory_bandwidth"
    else
        print_error "Memory bandwidth benchmark failed."
        return 1
    fi
    
    return 0
}

# Function to run transformer benchmark
run_transformer_benchmark() {
    print_section "Running Transformer Benchmark"
    
    # Check if the benchmark script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/benchmarks/transformer_benchmark.py" ]; then
        print_error "Transformer benchmark script not found."
        return 1
    fi
    
    # Run the benchmark
    print_step "Running benchmark..."
    python3 $HOME/Desktop/Stans_MLStack/benchmarks/transformer_benchmark.py \
        --batch-sizes 1 2 4 8 \
        --seq-lengths 128 256 512 1024 \
        --d-model 512 \
        --nhead 8 \
        --dim-feedforward 2048 \
        --num-layers 6 \
        --dtype float32 \
        --num-runs 3 \
        --output-dir $RESULTS_DIR/transformer
    
    if [ $? -eq 0 ]; then
        print_success "Transformer benchmark completed successfully."
        print_step "Results saved to $RESULTS_DIR/transformer"
    else
        print_error "Transformer benchmark failed."
        return 1
    fi
    
    return 0
}

# Function to run flash attention benchmark
run_flash_attention_benchmark() {
    print_section "Running Flash Attention Benchmark"
    
    # Check if the benchmark script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/benchmarks/flash_attention_benchmark.py" ]; then
        print_error "Flash attention benchmark script not found."
        return 1
    fi
    
    # Check if Flash Attention is installed
    if ! python3 -c "import flash_attention_amd" &> /dev/null; then
        print_warning "Flash Attention is not installed. Skipping benchmark."
        return 0
    fi
    
    # Run the benchmark
    print_step "Running benchmark..."
    python3 $HOME/Desktop/Stans_MLStack/benchmarks/flash_attention_benchmark.py \
        --batch-sizes 1 2 4 8 \
        --seq-lengths 128 256 512 1024 2048 \
        --num-heads 8 \
        --head-dim 64 \
        --causal \
        --dtype float32 \
        --num-runs 3 \
        --output-dir $RESULTS_DIR/flash_attention
    
    if [ $? -eq 0 ]; then
        print_success "Flash attention benchmark completed successfully."
        print_step "Results saved to $RESULTS_DIR/flash_attention"
    else
        print_error "Flash attention benchmark failed."
        return 1
    fi
    
    return 0
}

# Function to run all benchmarks
run_all_benchmarks() {
    print_header "Running All Benchmarks"
    
    # Run matrix multiplication benchmark
    run_matrix_multiplication_benchmark
    
    # Run memory bandwidth benchmark
    run_memory_bandwidth_benchmark
    
    # Run transformer benchmark
    run_transformer_benchmark
    
    # Run flash attention benchmark
    run_flash_attention_benchmark
    
    print_header "All Benchmarks Completed"
    print_step "Results saved to $RESULTS_DIR"
    
    return 0
}

# Show menu
show_menu() {
    print_header "ML Stack Benchmark Menu"
    
    echo -e "1) Run Matrix Multiplication Benchmark"
    echo -e "2) Run Memory Bandwidth Benchmark"
    echo -e "3) Run Transformer Benchmark"
    echo -e "4) Run Flash Attention Benchmark"
    echo -e "5) Run All Benchmarks"
    echo -e "0) Exit"
    echo
    
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            run_matrix_multiplication_benchmark
            ;;
        2)
            run_memory_bandwidth_benchmark
            ;;
        3)
            run_transformer_benchmark
            ;;
        4)
            run_flash_attention_benchmark
            ;;
        5)
            run_all_benchmarks
            ;;
        0)
            print_header "Exiting ML Stack Benchmark Runner"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            ;;
    esac
    
    # Show menu again
    show_menu
}

# Main function
main() {
    print_header "ML Stack Benchmark Runner"
    
    # Check if PyTorch is installed
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch is not installed. Please install PyTorch first."
        exit 1
    fi
    
    # Check if CUDA is available
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_error "CUDA is not available. Please check your ROCm installation."
        exit 1
    fi
    
    # Show menu
    show_menu
    
    return 0
}

# Run main function
main
