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
# ML Stack Environment Setup (Basic Version)
# =============================================================================
# This script sets up a basic environment for the ML Stack on AMD GPUs.
#
# WHEN TO USE THIS SCRIPT:
# - You want a simple, straightforward environment setup
# - You have a standard AMD GPU configuration
# - You don't need advanced hardware detection or filtering
# - You prefer simplicity over customization
#
# If you need more advanced features like:
# - Automatic hardware detection
# - Filtering out integrated GPUs
# - Comprehensive system dependency checks
# - More detailed environment configuration
#
# Then use the enhanced_setup_environment.sh script instead.
#
# Author: Stanley Chisango (Scooter Lacroix)
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

                                ML Stack Environment Setup
EOF
echo

# Check if terminal supports colors
if [ -t 1 ]; then
    # Check if NO_COLOR environment variable is set
    if [ -z "$NO_COLOR" ]; then
        # Terminal supports colors
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
    else
        # NO_COLOR is set, don't use colors
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        MAGENTA=''
        CYAN=''
        BOLD=''
        UNDERLINE=''
        BLINK=''
        REVERSE=''
        RESET=''
    fi
else
    # Not a terminal, don't use colors
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    UNDERLINE=''
    BLINK=''
    REVERSE=''
    RESET=''
fi

# Function definitions
print_header() {
    echo "=== $1 ==="
    echo
}

print_section() {
    echo ">>> $1"
}

print_step() {
    echo ">> $1"
}

print_success() {
    echo "✓ $1"
}

print_warning() {
    echo "⚠ $1"
}

print_error() {
    echo "✗ $1"
}

# Function to detect GPUs
detect_gpus() {
    print_section "Detecting GPUs"

    # Check if ROCm is installed
    if ! command -v rocminfo &> /dev/null; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
    fi

    # Get GPU count
    gpu_count=$(rocminfo | grep "GPU ID" | wc -l)
    print_step "Detected $gpu_count AMD GPU(s)"

    # List AMD GPUs
    print_step "AMD GPUs:"
    rocminfo | grep -A 1 "GPU ID" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
        echo "  - $gpu"
    done

    # Set environment variables
    print_step "Setting environment variables..."
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    export PYTORCH_ROCM_DEVICE=$(seq -s, 0 $((gpu_count-1)))

    print_step "Environment variables:"
    echo "  - HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  - PYTORCH_ROCM_DEVICE: $PYTORCH_ROCM_DEVICE"

    print_success "GPUs detected successfully"
    return 0
}

# Function to create environment file
create_environment_file() {
    print_section "Creating Environment File"

    # Create environment file
    print_step "Creating environment file..."

    # Create .mlstack_env file in home directory
    cat > $HOME/.mlstack_env << EOF
# ML Stack Environment File
# Created by ML Stack Environment Setup Script

# GPU Selection
export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=1

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# PyTorch Settings
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# MPI Settings
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml_ucx_opal_cuda_support=true
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib,uct

# Path Settings
export PATH=\$PATH:/opt/rocm/bin:/opt/rocm/hip/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/hip/lib
EOF

    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.mlstack_env" $HOME/.bashrc; then
        echo -e "\n# Source ML Stack environment" >> $HOME/.bashrc
        echo "source \$HOME/.mlstack_env" >> $HOME/.bashrc
    fi

    # Source the file
    source $HOME/.mlstack_env

    print_success "Environment file created successfully"
    print_step "Environment file: $HOME/.mlstack_env"
    print_step "The environment file has been added to your .bashrc file."
    print_step "To apply the changes, run: source $HOME/.bashrc"

    return 0
}

# Function to create directory structure
create_directory_structure() {
    print_section "Creating Directory Structure"

    # Create directory structure
    print_step "Creating directory structure..."

    # Create directories
    mkdir -p $HOME/Desktop/Stans_MLStack/logs
    mkdir -p $HOME/Desktop/Stans_MLStack/data
    mkdir -p $HOME/Desktop/Stans_MLStack/models
    mkdir -p $HOME/Desktop/Stans_MLStack/benchmark_results
    mkdir -p $HOME/Desktop/Stans_MLStack/test_results

    print_success "Directory structure created successfully"

    return 0
}

# Main function
main() {
    print_header "ML Stack Environment Setup"

    # Detect GPUs
    detect_gpus
    if [ $? -ne 0 ]; then
        print_error "GPU detection failed. Exiting."
        exit 1
    fi

    # Create environment file
    create_environment_file
    if [ $? -ne 0 ]; then
        print_error "Environment file creation failed. Exiting."
        exit 1
    fi

    # Create directory structure
    create_directory_structure
    if [ $? -ne 0 ]; then
        print_error "Directory structure creation failed. Exiting."
        exit 1
    fi

    print_header "ML Stack Environment Setup Complete"
    print_step "To apply the changes, run: source $HOME/.bashrc"

    return 0
}

# Run main function
main
