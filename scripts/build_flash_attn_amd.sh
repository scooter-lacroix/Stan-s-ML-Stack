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

build_triton() {
    print_section "Building Triton"
    
    # Determine script directory to find relative paths
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    REPO_ROOT="$( dirname "$SCRIPT_DIR" )" # Assuming scripts is one level down from repo root

    TRITON_DIR="$HOME/triton_flash_attn_build" # Build triton in a dedicated directory
    mkdir -p "$TRITON_DIR"
    cd "$TRITON_DIR"

    if [ -d "triton" ]; then
        print_step "Triton directory already exists. Pulling latest changes from default branch (main_perf)."
        cd triton
        git pull # Assumes it's already on main_perf or the desired default
        cd ..
    else
        print_step "Cloning Triton repository (default branch: main_perf)..."
        # NOTE: Using the default branch (main_perf) of ROCm/triton.
        # As of investigation, no specific branch/tag was identified for ROCm 6.4.1b.
        # If compatibility issues arise, this may need to be adjusted to a specific commit/tag/branch.
        git clone https://github.com/ROCm/triton.git
    fi
    
    cd triton/python
    
    # Set AMDGPU_TARGETS or use default
    DEFAULT_AMDGPU_TARGETS="gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100,gfx1101,gfx1102"
    if [ -z "$AMDGPU_TARGETS" ]; then
        print_warning "AMDGPU_TARGETS environment variable not set. Using default: $DEFAULT_AMDGPU_TARGETS"
        export AMDGPU_TARGETS="$DEFAULT_AMDGPU_TARGETS"
    else
        print_step "Using AMDGPU_TARGETS from environment: $AMDGPU_TARGETS"
    fi

    print_step "Installing Triton with GPU_ARCHS=$AMDGPU_TARGETS ..."
    pip install matplotlib pandas # Dependencies for Triton build/tests
    GPU_ARCHS="$AMDGPU_TARGETS" python setup.py install
    
    if [ $? -ne 0 ]; then
        print_error "Triton build failed."
        return 1
    fi
    
    print_success "Triton built and installed successfully"
    cd "$REPO_ROOT" # Return to repo root or original script dir
    return 0
}

install_flash_attention() {
    print_section "Installing Core FlashAttention for AMD"
    
    # Determine script directory to find relative paths
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    REPO_ROOT="$( dirname "$SCRIPT_DIR" )" 
    CORE_FLASH_ATTENTION_DIR="$REPO_ROOT/core/flash_attention"

    if [ ! -d "$CORE_FLASH_ATTENTION_DIR" ]; then
        print_error "Core FlashAttention directory not found at $CORE_FLASH_ATTENTION_DIR"
        return 1
    fi
    
    print_step "Changing to $CORE_FLASH_ATTENTION_DIR and installing..."
    cd "$CORE_FLASH_ATTENTION_DIR"
    
    # Use pip install . which should pick up setup_flash_attn_amd.py
    # Ensure environment is clean or use --no-cache-dir if needed
    pip install .
    
    if [ $? -ne 0 ]; then
        print_error "Core FlashAttention installation failed."
        cd "$REPO_ROOT" # Go back to repo root
        return 1
    fi
    
    print_success "Core FlashAttention installed successfully"
    cd "$REPO_ROOT" # Go back to repo root
    return 0
}

verify_installation() {
    print_section "Verifying Core FlashAttention installation"
    
    # Create a temporary directory for the test script
    TEST_DIR="$HOME/tmp_flash_attn_test"
    mkdir -p "$TEST_DIR"
    
    print_step "Creating test script in $TEST_DIR ..."
    cat > "$TEST_DIR/test_core_flash_attention.py" << 'EOF'
import torch
import time
# Ensure this import path matches what core/flash_attention/setup_flash_attn_amd.py installs
from flash_attention_amd import flash_attn_func 

def test_flash_attention():
    print("Starting Flash Attention (core version) test...")
    # Create dummy data
    batch_size = 1 # Reduced for quicker test
    seq_len = 512  # Reduced for quicker test
    num_heads = 4  # Reduced
    head_dim = 32  # Reduced

    print(f"Test parameters: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")

    if not torch.cuda.is_available():
        print("CUDA (ROCm) device not available. Skipping test.")
        exit(1) # Indicate failure if no GPU

    print(f"Using device: {torch.cuda.get_device_name(0)}")

    try:
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        print("Input tensors created successfully on CUDA device.")
    except Exception as e:
        print(f"Error creating input tensors: {e}")
        exit(1)

    # Run Flash Attention
    try:
        print("Running flash_attn_func...")
        start_time = time.time()
        output = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize() # Ensure completion for timing
        end_time = time.time()
        print(f"flash_attn_func executed. Output shape: {output.shape}")
        print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
        print("Flash Attention (core version) test executed successfully!")
    except Exception as e:
        print(f"Error during Flash Attention (core version) test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_flash_attention()
EOF
    
    # Run test script
    print_step "Running test script..."
    cd "$TEST_DIR" # Change to test directory to run the script
    python test_core_flash_attention.py
    
    local exit_code=$?
    # Determine script directory to find relative paths
    SCRIPT_DIR_CLEANUP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    REPO_ROOT_CLEANUP="$( dirname "$SCRIPT_DIR_CLEANUP" )" 
    cd "$REPO_ROOT_CLEANUP" # Go back to repo root

    if [ $exit_code -eq 0 ]; then
        print_success "Core FlashAttention is working correctly"
        return 0
    else
        print_error "Core FlashAttention test failed"
        return 1
    fi
}

cleanup() {
    print_section "Cleaning up"
    
    TEST_DIR="$HOME/tmp_flash_attn_test"
    TRITON_BUILD_DIR="$HOME/triton_flash_attn_build"

    print_step "Removing temporary test directory: $TEST_DIR"
    if [ -d "$TEST_DIR" ]; then
        rm -rf "$TEST_DIR"
    fi
    
    # Optionally, you might want to ask the user if they want to remove the Triton build directory
    # For now, let's leave it, as it might be useful for debugging or subsequent builds.
    # print_step "Removing Triton build directory: $TRITON_BUILD_DIR"
    # if [ -d "$TRITON_BUILD_DIR" ]; then
    #     rm -rf "$TRITON_BUILD_DIR"
    # fi
    
    print_success "Cleanup completed successfully"
}

main() {
    print_header "Flash Attention Build Script for AMD GPUs (Core Implementation)"
    
    # Start time
    start_time=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi
    
    # Install dependencies (general build tools, Python deps for setup)
    install_dependencies
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies. Exiting."
        exit 1
    fi

    # Build Triton (as it's a dependency for some FlashAttention backends)
    build_triton
    if [ $? -ne 0 ]; then
        print_error "Failed to build Triton. Exiting."
        exit 1
    fi
    
    # Install Core FlashAttention
    install_flash_attention
    if [ $? -ne 0 ]; then
        print_error "Failed to install Core FlashAttention. Exiting."
        exit 1
    fi
    
    # Verify installation
    verify_installation
    if [ $? -ne 0 ]; then
        print_error "Installation verification failed. Exiting."
        exit 1
    fi
    
    # Cleanup
    cleanup # This now removes the test script's temporary directory
    
    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    print_header "Core FlashAttention Build Completed Successfully!"
    echo -e "${GREEN}Total build time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"
    echo
    echo -e "${CYAN}You can now use the core Flash Attention in your PyTorch code:${RESET}"
    echo
    echo -e "${YELLOW}import torch${RESET}"
    echo -e "${YELLOW}from flash_attention_amd import flash_attn_func # Or other specific imports as needed${RESET}"
    echo
    echo -e "${YELLOW}# Example usage (ensure tensors are on CUDA and float16/bfloat16 as appropriate):${RESET}"
    echo -e "${YELLOW}# q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\", dtype=torch.float16)${RESET}"
    echo -e "${YELLOW}# k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\", dtype=torch.float16)${RESET}"
    echo -e "${YELLOW}# v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\", dtype=torch.float16)${RESET}"
    echo -e "${YELLOW}# output = flash_attn_func(q, k, v, causal=True)${RESET}"
    echo
    
    return 0
}

# Main script execution
main
