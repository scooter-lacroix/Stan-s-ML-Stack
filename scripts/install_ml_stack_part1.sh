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
# ML Stack Installation Script
# =============================================================================
# This script installs the ML Stack for AMD GPUs.
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
                                                                                                                 
                                ML Stack Installation Script
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
    
    # Check ROCm version
    rocm_version=$(rocminfo | grep "ROCm Version" | awk '{print $3}')
    print_step "ROCm version: $rocm_version"
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        return 1
    fi
    print_success "Python 3 is installed"
    
    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3 first."
        return 1
    fi
    print_success "pip3 is installed"
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install git first."
        return 1
    fi
    print_success "Git is installed"
    
    # Check if cmake is installed
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install cmake first."
        return 1
    fi
    print_success "CMake is installed"
    
    # Check if AMD GPUs are detected
    if ! rocminfo | grep -q "GPU ID"; then
        print_error "No AMD GPUs detected. Please check your hardware and ROCm installation."
        return 1
    fi
    
    # Count AMD GPUs
    gpu_count=$(rocminfo | grep "GPU ID" | wc -l)
    print_success "Detected $gpu_count AMD GPU(s)"
    
    # List AMD GPUs
    print_step "AMD GPUs:"
    rocminfo | grep -A 1 "GPU ID" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
        echo -e "  - $gpu"
    done
    
    # Check environment variables
    print_step "Checking environment variables..."
    
    # Set environment variables if not set
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        print_warning "HIP_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi
    
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        print_warning "CUDA_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi
    
    if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        print_warning "PYTORCH_ROCM_DEVICE is not set. Setting to all GPUs..."
        export PYTORCH_ROCM_DEVICE=$(seq -s, 0 $((gpu_count-1)))
    fi
    
    print_step "Environment variables:"
    echo -e "  - HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo -e "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo -e "  - PYTORCH_ROCM_DEVICE: $PYTORCH_ROCM_DEVICE"
    
    # Check disk space
    available_space=$(df -h $HOME | awk 'NR==2 {print $4}')
    print_step "Available disk space: $available_space"
    
    # Check if there's enough disk space (at least 20GB)
    available_space_kb=$(df -k $HOME | awk 'NR==2 {print $4}')
    if [ $available_space_kb -lt 20971520 ]; then  # 20GB in KB
        print_warning "You have less than 20GB of free disk space. Some components might fail to build."
        read -p "Do you want to continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation aborted by user."
            return 1
        fi
    fi
    
    return 0
}
