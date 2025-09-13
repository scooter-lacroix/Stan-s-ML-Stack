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
# Flash Attention Build Script
# =============================================================================
# This script builds Flash Attention with AMD GPU support.
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
                                                                                                                 
                           Flash Attention Build Script for AMD GPUs
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

check_prerequisites() {
    print_section "Checking prerequisites"
    
    # Check if ROCm is installed
    if ! command -v rocminfo &> /dev/null; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
    fi
    print_success "ROCm is installed"
    
    # Check if PyTorch with ROCm is installed
    if ! python3 -c "import torch; print(torch.version.hip)" &> /dev/null; then
        print_error "PyTorch with ROCm support is not installed. Please install PyTorch with ROCm support first."
        return 1
    fi
    print_success "PyTorch with ROCm support is installed"
    
    # Check if CUDA is available through ROCm
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_warning "CUDA is not available through ROCm. Check your environment variables."
        print_step "Setting environment variables..."
        export HIP_VISIBLE_DEVICES=0,1
        export CUDA_VISIBLE_DEVICES=0,1
        export PYTORCH_ROCM_DEVICE=0,1
        
        # Check again
        if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            print_error "CUDA is still not available through ROCm. Please check your ROCm installation."
            return 1
        fi
        print_success "Environment variables set successfully"
    fi
    print_success "CUDA is available through ROCm"
    
    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"
    
    return 0
}

install_dependencies() {
    print_section "Installing dependencies"
    
    print_step "Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git python3-dev python3-pip ninja-build
    
    print_step "Installing Python dependencies..."
    pip install packaging ninja wheel setuptools
    
    print_success "Dependencies installed successfully"
}
