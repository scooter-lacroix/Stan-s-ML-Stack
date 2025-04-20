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
# ML Stack Test Runner
# =============================================================================
# This script runs tests for the ML Stack on AMD GPUs.
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
                                                                                                                 
                                ML Stack Test Runner
EOF
echo

# --- Environment Setup for Custom Builds and ROCm ---
echo "Setting up environment for tests..."
# Unset problematic HSA Tools Lib if set globally
unset HSA_TOOLS_LIB

# Add custom ONNX Runtime build to PYTHONPATH
export PYTHONPATH="/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH"

# Set environment variables for ROCm (similar to test_onnx_simple.py)
export HIP_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1" # Often needed even for ROCm
export PYTORCH_ROCM_DEVICE="0,1" # If using PyTorch alongside
export HSA_ENABLE_SDMA="0"
export GPU_MAX_HEAP_SIZE="100"
export GPU_MAX_ALLOC_PERCENT="100"
# export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM="1" # Optional MIOpen tuning
# export MIOPEN_FIND_MODE="3"               # Optional MIOpen tuning
# export MIOPEN_FIND_ENFORCE="3"            # Optional MIOpen tuning
echo "PYTHONPATH set to: $PYTHONPATH"
echo "ROCm environment variables exported."
echo "-------------------------------------------------"

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
RESULTS_DIR="$HOME/Desktop/Stans_MLStack/test_results/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $RESULTS_DIR

# Function to run GPU detection test
run_gpu_detection_test() {
    print_section "Running GPU Detection Test"
    
    # Check if the test script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/tests/test_gpu_detection.py" ]; then
        print_error "GPU detection test script not found."
        return 1
    fi
    
    # Run the test
    print_step "Running test..."
    python3 $HOME/Desktop/Stans_MLStack/tests/test_gpu_detection.py | tee $RESULTS_DIR/gpu_detection_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "GPU detection test passed."
    else
        print_error "GPU detection test failed."
        return 1
    fi
    
    return 0
}

# Function to run Flash Attention test
run_flash_attention_test() {
    print_section "Running Flash Attention Test"
    
    # Check if the test script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/tests/test_flash_attention.py" ]; then
        print_error "Flash Attention test script not found."
        return 1
    fi
    
    # Check if Flash Attention is installed
    if ! python3 -c "import flash_attention_amd" &> /dev/null; then
        print_warning "Flash Attention is not installed. Skipping test."
        return 0
    fi
    
    # Run the test
    print_step "Running test..."
    python3 $HOME/Desktop/Stans_MLStack/tests/test_flash_attention.py | tee $RESULTS_DIR/flash_attention_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "Flash Attention test passed."
    else
        print_error "Flash Attention test failed."
        return 1
    fi
    
    return 0
}

# Function to run MPI test
run_mpi_test() {
    print_section "Running MPI Test"
    
    # Check if the test script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/tests/test_mpi.py" ]; then
        print_error "MPI test script not found."
        return 1
    fi
    
    # Check if MPI is installed
    if ! command -v mpirun &> /dev/null; then
        print_warning "MPI is not installed. Skipping test."
        return 0
    fi
    
    # Run the test
    print_step "Running test..."
    mpirun -np 2 python3 $HOME/Desktop/Stans_MLStack/tests/test_mpi.py | tee $RESULTS_DIR/mpi_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "MPI test passed."
    else
        print_error "MPI test failed."
        return 1
    fi
    
    return 0
}

# Function to run ONNX Runtime test
run_onnx_test() {
    print_section "Running ONNX Runtime Test"
    
    # Check if the test script exists
    if [ ! -f "$HOME/Desktop/Stans_MLStack/tests/test_onnx.py" ]; then
        print_error "ONNX Runtime test script not found."
        return 1
    fi
    
    # Check if ONNX Runtime is installed
    if ! python3 -c "import onnxruntime" &> /dev/null; then
        print_warning "ONNX Runtime is not installed. Skipping test."
        return 0
    fi
    
    # Run the test
    print_step "Running test..."
    python3 $HOME/Desktop/Stans_MLStack/tests/test_onnx.py | tee $RESULTS_DIR/onnx_test.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "ONNX Runtime test passed."
    else
        print_error "ONNX Runtime test failed."
        return 1
    fi
    
    return 0
}

# Function to run all tests
run_all_tests() {
    print_header "Running All Tests"
    
    # Run GPU detection test
    run_gpu_detection_test
    
    # Run Flash Attention test
    run_flash_attention_test
    
    # Run MPI test
    run_mpi_test
    
    # Run ONNX Runtime test
    run_onnx_test
    
    print_header "All Tests Completed"
    print_step "Results saved to $RESULTS_DIR"
    
    return 0
}

# Show menu
show_menu() {
    print_header "ML Stack Test Menu"
    
    echo -e "1) Run GPU Detection Test"
    echo -e "2) Run Flash Attention Test"
    echo -e "3) Run MPI Test"
    echo -e "4) Run ONNX Runtime Test"
    echo -e "5) Run All Tests"
    echo -e "0) Exit"
    echo
    
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            run_gpu_detection_test
            ;;
        2)
            run_flash_attention_test
            ;;
        3)
            run_mpi_test
            ;;
        4)
            run_onnx_test
            ;;
        5)
            run_all_tests
            ;;
        0)
            print_header "Exiting ML Stack Test Runner"
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
    print_header "ML Stack Test Runner"
    
    # Check if PyTorch is installed
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch is not installed. Please install PyTorch first."
        exit 1
    fi
    
    # Show menu
    show_menu
    
    return 0
}

# Run main function
main
