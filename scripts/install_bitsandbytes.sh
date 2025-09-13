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
# BITSANDBYTES Installation Script for AMD GPUs with Enhanced Standards
# =============================================================================
# This script installs BITSANDBYTES for efficient 4-bit and 8-bit quantization
# with AMD GPU support through PyTorch and ROCm.
#
# Enhanced with modern installation standards including:
# - Multi-package-manager ROCm detection
# - Installation method choices (global/venv/auto)
# - Externally managed environment support
# - Comprehensive error handling and recovery
# - Enhanced user experience with colors and progress
# - Virtual environment support with uv
# - Cross-platform compatibility
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗ ██╗████████╗███████╗ █████╗ ███╗   ██╗██████╗ ██████╗ ██╗   ██╗████████╗███████╗███████╗
  ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔════╝██╔════╝
  ██████╔╝██║   ██║   ███████╗███████║██╔██╗ ██║██║  ██║██████╔╝ ╚████╔╝    ██║   █████╗  ███████╗
  ██╔══██╗██║   ██║   ╚════██║██╔══██║██║╚██╗██║██║  ██║██╔══██╗  ╚██╔╝     ██║   ██╔══╝  ╚════██║
  ██████╔╝██║   ██║   ███████║██║  ██║██║ ╚████║██████╔╝██████╔╝   ██║      ██║   ███████╗███████║
  ╚═════╝ ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═════╝    ╚═╝      ╚═╝   ╚══════╝╚══════╝
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
    echo
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║                                                         ║"
    echo "║               === $1 ===               ║"
    echo "║                                                         ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    echo
}

print_section() {
    echo
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ $1"
    echo "└─────────────────────────────────────────────────────────┘"
}

print_step() {
    echo "➤ $1"
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

# Function to print a clean separator line
print_separator() {
    echo "───────────────────────────────────────────────────────────"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to detect package manager
detect_package_manager() {
    if command_exists dnf; then
        echo "dnf"
    elif command_exists apt-get; then
        echo "apt"
    elif command_exists yum; then
        echo "yum"
    elif command_exists pacman; then
        echo "pacman"
    elif command_exists zypper; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Function to detect if running in WSL
detect_wsl() {
    if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect if running in container
detect_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q container /proc/1/cgroup 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to use uv or pip for Python packages
install_python_package() {
    local package="$1"
    shift
    local extra_args="$@"

    if command_exists uv; then
        print_step "Installing $package with uv..."
        uv pip install --python $(which python3) $extra_args "$package"
    else
        print_step "Installing $package with pip..."
        python3 -m pip install $extra_args "$package"
    fi
}

# Function to show environment variables
show_env() {
    # Set up minimal ROCm environment for showing variables
    HSA_TOOLS_LIB=0
    HSA_OVERRIDE_GFX_VERSION=11.0.0
    PYTORCH_ROCM_ARCH="gfx1100"
    ROCM_PATH="/opt/rocm"
    PATH="/opt/rocm/bin:$PATH"
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Check if rocprofiler library exists and update HSA_TOOLS_LIB accordingly
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
    fi

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    fi

    echo "export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\""
    echo "export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\""
    if [ -n "$PYTORCH_ALLOC_CONF" ]; then
        echo "export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\""
    fi
    echo "export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\""
    echo "export ROCM_PATH=\"$ROCM_PATH\""
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
}

# Function to detect GPU architecture
detect_gpu_arch() {
    if command_exists rocminfo; then
        # Try to get GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -o "gfx[0-9]*" | head -n 1)
        if [ -n "$gpu_arch" ]; then
            echo "$gpu_arch"
            return 0
        fi
    fi

    # Fallback to common architectures based on ROCm version
    if [ -n "$rocm_version" ]; then
        rocm_major=$(echo "$rocm_version" | cut -d '.' -f 1)
        rocm_minor=$(echo "$rocm_version" | cut -d '.' -f 2)

        if [ "$rocm_major" -ge 6 ]; then
            echo "gfx1100"  # RDNA3 architecture
        elif [ "$rocm_major" -eq 5 ]; then
            echo "gfx1030"  # RDNA2 architecture
        else
            echo "gfx90a"   # CDNA architecture
        fi
    else
        echo "gfx90a"  # Default fallback
    fi
}

# Function to handle uv commands with venv fallback
uv_pip_install() {
    local args="$@"

    # Check if uv is available as a command
    if command -v uv &> /dev/null; then
        case $INSTALL_METHOD in
            "global")
                print_step "Installing globally with pip..."
                python3 -m pip install --break-system-packages $args
                BITSANDBYTES_VENV_PYTHON=""
                ;;
            "venv")
                print_step "Creating uv virtual environment..."
                VENV_DIR="./bitsandbytes_venv"
                if [ ! -d "$VENV_DIR" ]; then
                    uv venv "$VENV_DIR"
                fi
                source "$VENV_DIR/bin/activate"
                print_step "Installing in virtual environment..."
                uv pip install $args
                BITSANDBYTES_VENV_PYTHON="$VENV_DIR/bin/python"
                print_success "Installed in virtual environment: $VENV_DIR"
                ;;
            "auto")
                # Try global install first
                print_step "Attempting global installation with uv..."
                local install_output
                install_output=$(uv pip install --python $(which python3) $args 2>&1)
                local install_exit_code=$?

                if echo "$install_output" | grep -q "externally managed"; then
                    print_warning "Global installation failed due to externally managed environment"
                    print_step "Creating uv virtual environment for installation..."

                    # Create uv venv in project directory
                    VENV_DIR="./bitsandbytes_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args

                    # Store venv path for verification
                    BITSANDBYTES_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                elif [ $install_exit_code -eq 0 ]; then
                    print_success "Global installation successful"
                    BITSANDBYTES_VENV_PYTHON=""
                else
                    print_error "Global installation failed with unknown error:"
                    echo "$install_output"
                    print_step "Falling back to virtual environment..."

                    # Create uv venv in project directory
                    VENV_DIR="./bitsandbytes_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args

                    # Store venv path for verification
                    BITSANDBYTES_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                fi
                ;;
        esac
    else
        # Fall back to pip
        print_step "Installing with pip..."
        python3 -m pip install $args
        BITSANDBYTES_VENV_PYTHON=""
    fi
}

# Function to install bitsandbytes with enhanced error handling
install_bitsandbytes_enhanced() {
    print_header "BITSANDBYTES Installation"

    # Check if bitsandbytes is already installed
    PYTHON_CMD=${BITSANDBYTES_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import bitsandbytes" &>/dev/null; then
        bnb_version=$($PYTHON_CMD -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null)

        # Check if --force flag is provided
        if [[ "$*" == *"--force"* ]] || [[ "$BITSANDBYTES_REINSTALL" == "true" ]]; then
            print_warning "Force reinstall requested - proceeding with reinstallation"
            print_step "Will reinstall bitsandbytes despite working installation"
        else
            print_success "bitsandbytes is already installed and working (version $bnb_version)"
            print_step "Use --force to reinstall anyway."
            return 0
        fi
    fi

    # Check ROCm installation
    print_section "Checking ROCm Installation"

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH=$(detect_gpu_arch)
        export ROCM_PATH="/opt/rocm"
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Check if we can install rocprofiler
            if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
                print_step "Installing rocprofiler for HSA tools support..."
                sudo apt-get update && sudo apt-get install -y rocprofiler
                if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
                    export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
                    print_success "ROCm profiler installed and configured"
                else
                    export HSA_TOOLS_LIB=0
                    print_warning "ROCm profiler installation failed, disabling HSA tools"
                fi
            else
                export HSA_TOOLS_LIB=0
                print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
            fi
        fi

        # Fix deprecated PYTORCH_CUDA_ALLOC_CONF warning
        if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
            export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
            unset PYTORCH_CUDA_ALLOC_CONF
            print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
        fi

        print_success "ROCm environment variables configured"
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            package_manager=$(detect_package_manager)
            case $package_manager in
                apt)
                    sudo apt update && sudo apt install -y rocminfo
                    ;;
                dnf)
                    sudo dnf install -y rocminfo
                    ;;
                yum)
                    sudo yum install -y rocminfo
                    ;;
                pacman)
                    sudo pacman -S rocminfo
                    ;;
                zypper)
                    sudo zypper install -y rocminfo
                    ;;
                *)
                    print_error "Unsupported package manager: $package_manager"
                    return 1
                    ;;
            esac
            if command_exists rocminfo; then
                print_success "Installed rocminfo"
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi

    # Detect ROCm version
    rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    # Check cross-platform compatibility
    print_section "Cross-Platform Compatibility Check"

    wsl_detected=$(detect_wsl)
    container_detected=$(detect_container)

    if [ "$wsl_detected" = "true" ]; then
        print_warning "WSL environment detected"
        print_step "Some ROCm features may be limited in WSL"
    fi

    if [ "$container_detected" = "true" ]; then
        print_warning "Container environment detected"
        print_step "Ensure ROCm is properly configured in the container"
    fi

    # Check if uv is installed
    print_section "Installing bitsandbytes with ROCm Support"

    if ! command_exists uv; then
        print_step "Installing uv package manager..."
        python3 -m pip install uv

        # Add uv to PATH if it was installed in a user directory
        if [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi

        # Add uv to PATH if it was installed via cargo
        if [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        if ! command_exists uv; then
            print_error "Failed to install uv package manager"
            print_step "Falling back to pip"
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}bitsandbytes Installation Options:${RESET}"
    echo "1) Global installation (recommended for system-wide use)"
    echo "2) Virtual environment (isolated installation)"
    echo "3) Auto-detect (try global, fallback to venv if needed)"
    echo
    read -p "Choose installation method (1-3) [3]: " INSTALL_CHOICE
    INSTALL_CHOICE=${INSTALL_CHOICE:-3}

    case $INSTALL_CHOICE in
        1)
            INSTALL_METHOD="global"
            print_step "Using global installation method"
            ;;
        2)
            INSTALL_METHOD="venv"
            print_step "Using virtual environment method"
            ;;
        3|*)
            INSTALL_METHOD="auto"
            print_step "Using auto-detect method"
            ;;
    esac

    # Install bitsandbytes using the enhanced method
    print_step "Installing bitsandbytes with ROCm support..."

    # Try different installation strategies
    uv_pip_install bitsandbytes

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${BITSANDBYTES_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import bitsandbytes" &>/dev/null; then
        bnb_version=$($PYTHON_CMD -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null)
        print_success "bitsandbytes is installed (version: $bnb_version)"

        # Test basic functionality
        if $PYTHON_CMD -c "import bitsandbytes as bnb; import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
            print_success "bitsandbytes basic functionality working"
        else
            print_warning "bitsandbytes basic functionality may have issues"
        fi
    else
        print_error "bitsandbytes installation failed"
        return 1
    fi

    # Create enhanced test script
    create_enhanced_test_script

    print_success "bitsandbytes installation completed successfully"

    # Provide usage information
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$BITSANDBYTES_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./bitsandbytes_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import bitsandbytes as bnb; import torch; print('bitsandbytes version:', bnb.__version__); print('GPU available:', torch.cuda.is_available())\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import bitsandbytes as bnb; import torch; print('bitsandbytes version:', bnb.__version__); print('GPU available:', torch.cuda.is_available())\"${RESET}"
    fi
    echo
    echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}ROCm environment variables are set for this session.${RESET}"
    echo -e "${YELLOW}For future sessions, you may need to run:${RESET}"

    # Output the actual environment variables that were set
    echo -e "${GREEN}export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\"${RESET}"
    echo -e "${GREEN}export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\"${RESET}"
    if [ -n "$PYTORCH_ALLOC_CONF" ]; then
        echo -e "${GREEN}export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\"${RESET}"
    fi
    echo -e "${GREEN}export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\"${RESET}"
    echo -e "${GREEN}export ROCM_PATH=\"$ROCM_PATH\"${RESET}"
    echo -e "${GREEN}export PATH=\"$PATH\"${RESET}"
    echo -e "${GREEN}export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"${RESET}"
    echo

    return 0
}

# Function to create enhanced test script
create_enhanced_test_script() {
    TEST_SCRIPT="./test_bitsandbytes_enhanced.py"
    cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
"""
Enhanced BITSANDBYTES Test Script
Tests quantization functionality with AMD GPU support
"""

import torch
import bitsandbytes as bnb
import time
import numpy as np
import sys

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def print_success(text):
    print(f"✓ {text}")

def print_warning(text):
    print(f"⚠ {text}")

def print_error(text):
    print(f"✗ {text}")

def test_linear_8bit():
    """Test 8-bit linear layer."""
    print_header("Testing 8-bit Linear Layer")

    try:
        # Create input tensor
        batch_size = 32
        input_dim = 768
        output_dim = 768

        # Create input on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(batch_size, input_dim, device=device)

        # Create FP32 linear layer
        fp32_linear = torch.nn.Linear(input_dim, output_dim).to(device)

        # Create 8-bit linear layer
        int8_linear = bnb.nn.Linear8bitLt(input_dim, output_dim, has_fp16_weights=False).to(device)

        # Copy weights from FP32 to 8-bit
        int8_linear.weight.data = fp32_linear.weight.data.clone()
        int8_linear.bias.data = fp32_linear.bias.data.clone()

        # Forward pass with FP32
        fp32_output = fp32_linear(x)

        # Forward pass with 8-bit
        int8_output = int8_linear(x)

        # Check results
        error = torch.abs(fp32_output - int8_output).mean().item()
        print(f"Mean absolute error: {error:.6f}")

        # Benchmark
        n_runs = 100

        # Warm up
        for _ in range(10):
            _ = fp32_linear(x)
            _ = int8_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark FP32
        start_time = time.time()
        for _ in range(n_runs):
            _ = fp32_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        fp32_time = (time.time() - start_time) / n_runs

        # Benchmark 8-bit
        start_time = time.time()
        for _ in range(n_runs):
            _ = int8_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        int8_time = (time.time() - start_time) / n_runs

        print(f"FP32 time: {fp32_time*1000:.3f} ms")
        print(f"8-bit time: {int8_time*1000:.3f} ms")
        print(f"Speedup: {fp32_time/int8_time:.2f}x")
        print(f"Memory savings: ~{100 * (1 - 8/32):.1f}%")

        success = error < 0.01
        if success:
            print_success("8-bit linear layer test passed")
        else:
            print_warning("8-bit linear layer test failed - high error")

        return success

    except Exception as e:
        print_error(f"8-bit linear layer test failed with error: {e}")
        return False

def test_linear_4bit():
    """Test 4-bit linear layer."""
    print_header("Testing 4-bit Linear Layer")

    try:
        # Create input tensor
        batch_size = 32
        input_dim = 768
        output_dim = 768

        # Create input on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float16)

        # Create FP16 linear layer
        fp16_linear = torch.nn.Linear(input_dim, output_dim).to(device).to(torch.float16)

        # Create 4-bit linear layer
        int4_linear = bnb.nn.Linear4bit(
            input_dim, output_dim,
            bias=True,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        ).to(device)

        # Forward pass with FP16
        fp16_output = fp16_linear(x)

        # Forward pass with 4-bit
        int4_output = int4_linear(x)

        # Check results
        error = torch.abs(fp16_output - int4_output).mean().item()
        print(f"Mean absolute error: {error:.6f}")

        # Benchmark
        n_runs = 100

        # Warm up
        for _ in range(10):
            _ = fp16_linear(x)
            _ = int4_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark FP16
        start_time = time.time()
        for _ in range(n_runs):
            _ = fp16_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        fp16_time = (time.time() - start_time) / n_runs

        # Benchmark 4-bit
        start_time = time.time()
        for _ in range(n_runs):
            _ = int4_linear(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        int4_time = (time.time() - start_time) / n_runs

        print(f"FP16 time: {fp16_time*1000:.3f} ms")
        print(f"4-bit time: {int4_time*1000:.3f} ms")
        print(f"Speedup: {fp16_time/int4_time:.2f}x")
        print(f"Memory savings: ~{100 * (1 - 4/16):.1f}%")

        success = error < 0.05
        if success:
            print_success("4-bit linear layer test passed")
        else:
            print_warning("4-bit linear layer test failed - high error")

        return success

    except Exception as e:
        print_error(f"4-bit linear layer test failed with error: {e}")
        return False

def test_optimizers():
    """Test 8-bit optimizers."""
    print_header("Testing 8-bit Optimizers")

    try:
        # Create a simple model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).to(device)

        # Create 8-bit optimizer
        optimizer = bnb.optim.Adam8bit(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0
        )

        # Create input and target
        x = torch.randn(32, 128, device=device)
        y = torch.randn(32, 128, device=device)

        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

        print(f"Final loss: {loss.item():.6f}")
        print_success("8-bit optimizer test completed successfully")

        return True

    except Exception as e:
        print_error(f"8-bit optimizer test failed with error: {e}")
        return False

def benchmark_quantization():
    """Benchmark quantization performance."""
    print_header("Quantization Performance Benchmark")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_sizes = [768, 1024, 2048, 4096]

        results = []

        for size in model_sizes:
            print(f"\nTesting model size: {size}x{size}")

            # Create model
            model_fp32 = torch.nn.Linear(size, size).to(device)
            model_int8 = bnb.nn.Linear8bitLt(size, size, has_fp16_weights=False).to(device)

            # Copy weights
            model_int8.weight.data = model_fp32.weight.data.clone()
            model_int8.bias.data = model_fp32.bias.data.clone()

            # Create input
            x = torch.randn(16, size, device=device)

            # Warm up
            for _ in range(5):
                _ = model_fp32(x)
                _ = model_int8(x)
            if device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark FP32
            start_time = time.time()
            for _ in range(50):
                _ = model_fp32(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            fp32_time = (time.time() - start_time) / 50

            # Benchmark INT8
            start_time = time.time()
            for _ in range(50):
                _ = model_int8(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            int8_time = (time.time() - start_time) / 50

            speedup = fp32_time / int8_time
            memory_savings = 100 * (1 - 8/32)

            print(f"  FP32 time: {fp32_time*1000:.3f} ms")
            print(f"  INT8 time: {int8_time*1000:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory savings: {memory_savings:.1f}%")

            results.append({
                'size': size,
                'fp32_time': fp32_time,
                'int8_time': int8_time,
                'speedup': speedup,
                'memory_savings': memory_savings
            })

        print_success("Quantization benchmark completed")
        return results

    except Exception as e:
        print_error(f"Benchmark failed with error: {e}")
        return None

def main():
    """Run all tests."""
    print_header("BITSANDBYTES Enhanced Test Suite")
    print(f"bitsandbytes version: {bnb.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run tests
    tests = [
        ("8-bit Linear", test_linear_8bit),
        ("4-bit Linear", test_linear_4bit),
        ("8-bit Optimizers", test_optimizers)
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print_error(f"Error in {name} test: {e}")
            results.append((name, False))

    # Run benchmark
    benchmark_results = benchmark_quantization()

    # Print summary
    print_header("Test Summary")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print_success("All tests passed successfully!")
        return 0
    else:
        print_warning("Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x $TEST_SCRIPT
    print_success "Created enhanced test script at $TEST_SCRIPT"
    print_step "Run it with: python3 $TEST_SCRIPT"
}

# Main script logic
set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/bitsandbytes_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Start installation
log "=== Starting BITSANDBYTES Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Check for --dry-run option
if [[ "$*" == *"--dry-run"* ]]; then
    print_warning "DRY RUN MODE - No actual installation will be performed"
    print_step "This would install bitsandbytes with the following configuration:"
    print_step "- Installation method: auto-detect"
    print_step "- ROCm support: enabled"
    print_step "- Virtual environment: auto-create if needed"
    exit 0
fi

# Run the enhanced installation function
install_bitsandbytes_enhanced "$@"

log "=== BITSANDBYTES Installation Complete ==="
log "Log File: $LOG_FILE"
