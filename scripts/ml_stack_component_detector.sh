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
# ML Stack Component Detector Library
# =============================================================================
# This script provides common functions to detect ML Stack components
# regardless of where they are installed on the system.
# =============================================================================

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
    RESET=''
fi

# Function to print colored messages
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Set Python interpreter - prefer virtual environment if available
if [ -f "$HOME/rocm_venv/bin/python" ]; then
    PYTHON_INTERPRETER="$HOME/rocm_venv/bin/python"
    print_step "Using Python interpreter from virtual environment: $PYTHON_INTERPRETER"
else
    PYTHON_INTERPRETER="python3"
    print_step "Using system Python interpreter: $PYTHON_INTERPRETER"
fi

# Set up comprehensive PYTHONPATH to find all components
ORIGINAL_PYTHONPATH=$PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$HOME/pytorch:$HOME/ml_stack/flash_attn_amd_direct:$HOME/ml_stack/flash_attn_amd:$HOME/.local/lib/python3.13/site-packages:$HOME/.local/lib/python3.12/site-packages:$HOME/megatron/Megatron-LM:$HOME/onnxruntime_build:$HOME/migraphx_build:$HOME/vllm_build:$HOME/vllm_py313:$HOME/ml_stack/bitsandbytes/bitsandbytes"
print_step "Enhanced PYTHONPATH to find all components"

# Define search paths for components
COMPONENT_SEARCH_PATHS=(
    "$HOME/pytorch"
    "$HOME/ml_stack/flash_attn_amd_direct"
    "$HOME/ml_stack/flash_attn_amd"
    "$HOME/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-313"
    "$HOME/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-312"
    "$HOME/.local/lib/python3.13/site-packages"
    "$HOME/.local/lib/python3.12/site-packages"
    "$HOME/rocm_venv/lib/python3.13/site-packages"
    "$HOME/rocm_venv/lib/python3.12/site-packages"
    "$HOME/megatron/Megatron-LM"
    "$HOME/onnxruntime_build"
    "$HOME/migraphx_build"
    "$HOME/vllm_build"
    "$HOME/vllm_py313"
    "$HOME/ml_stack/bitsandbytes/bitsandbytes"
)

# Function to check if a component is installed by looking for its directory
check_component_dir() {
    local component_name=$1
    local search_paths=("${@:2}")

    for path in "${search_paths[@]}"; do
        if [ -d "$path/$component_name" ]; then
            print_step "Found $component_name at $path"
            return 0
        fi
    done

    return 1
}

# Function to check if a Python module can be imported
check_python_module() {
    if $PYTHON_INTERPRETER -c "import $1" 2>/dev/null; then
        return 0
    else
        # Try with enhanced path
        if $PYTHON_INTERPRETER -c "import sys; sys.path.extend(['$(echo ${COMPONENT_SEARCH_PATHS[@]} | tr ' ' "','")')']); import $1" 2>/dev/null; then
            return 0
        else
            return 1
        fi
    fi
}

# Function to detect ROCm installation
detect_rocm() {
    print_section "Detecting ROCm Installation"

    # Check if ROCm is installed
    if command_exists rocminfo; then
        print_success "ROCm is installed"

        # Get ROCm version
        rocm_version=$(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            # Try alternative method to get ROCm version
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            if [ -z "$rocm_version" ]; then
                rocm_version="unknown"
            fi
        fi

        print_step "ROCm version: $rocm_version"
        export ROCM_VERSION=$rocm_version

        # Check ROCm path
        if [ -d "/opt/rocm" ]; then
            rocm_path="/opt/rocm"
        elif [ -d "/opt/rocm-$rocm_version" ]; then
            rocm_path="/opt/rocm-$rocm_version"
        else
            # Try to find ROCm path
            rocm_path=$(dirname $(which rocminfo))/..
        fi

        print_step "ROCm path: $rocm_path"
        export ROCM_PATH=$rocm_path

        # Check if user has proper permissions
        if groups | grep -q -E '(video|render|rocm)'; then
            print_success "User has proper permissions for ROCm"
        else
            print_warning "User may not have proper permissions for ROCm"
            print_step "Recommended: Add user to video and render groups:"
            print_step "  sudo usermod -a -G video,render $USER"
            print_step "  (Requires logout/login to take effect)"
        fi
    else
        print_warning "ROCm is not installed or not in PATH"

        # Check if ROCm is installed in common locations
        if [ -d "/opt/rocm" ]; then
            print_step "Found ROCm in /opt/rocm"
            rocm_path="/opt/rocm"
        else
            # Try to find any rocm installation
            rocm_dirs=$(ls -d /opt/rocm* 2>/dev/null)
            if [ -n "$rocm_dirs" ]; then
                rocm_path=$(echo "$rocm_dirs" | head -n 1)
                print_step "Found ROCm in $rocm_path"
            else
                print_error "Could not find ROCm installation"
                print_step "Please install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
                return 1
            fi
        fi

        # Try to get ROCm version
        rocm_version=$(echo "$rocm_path" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        if [ -z "$rocm_version" ]; then
            rocm_version="unknown"
        fi

        export ROCM_PATH=$rocm_path
        export ROCM_VERSION=$rocm_version

        print_step "Adding ROCm to PATH and LD_LIBRARY_PATH..."
        export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/hip/lib

        print_warning "Please install ROCm properly for full functionality"
    fi

    return 0
}

# Function to detect AMD GPUs
detect_amd_gpus() {
    print_section "Detecting AMD GPUs"

    # Check if lspci is available
    if ! command_exists lspci; then
        print_warning "lspci command not found. Installing pciutils..."
        sudo apt-get update && sudo apt-get install -y pciutils
    fi

    # Detect AMD GPUs using lspci
    print_step "Searching for AMD GPUs..."
    amd_gpus=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display')

    if [ -z "$amd_gpus" ]; then
        print_error "No AMD GPUs detected."
        return 1
    else
        print_success "AMD GPUs detected:"
        echo "$amd_gpus" | while read -r line; do
            echo "  - $line"
        done
    fi

    # Check if ROCm is installed
    if command_exists rocminfo; then
        print_success "ROCm is installed"
        print_step "ROCm version: $(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)"

        # Get GPU count from ROCm
        gpu_count=$(rocminfo | grep "Device Type:.*GPU" | wc -l)
        print_step "ROCm detected $gpu_count GPU(s)"

        # List AMD GPUs from ROCm
        print_step "ROCm detected GPUs:"
        rocminfo | grep -A 10 "Device Type:.*GPU" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
            echo "  - $gpu"
        done
    else
        print_warning "ROCm is not installed. Some features may not work correctly."
        print_step "Attempting to detect GPUs using other methods..."

        # Try to detect GPUs using other methods
        if command_exists glxinfo; then
            print_step "GPU information from glxinfo:"
            glxinfo | grep -i "OpenGL renderer" | awk -F: '{print $2}' | xargs
        elif command_exists clinfo; then
            print_step "GPU information from clinfo:"
            clinfo | grep -i "Device Name" | awk -F: '{print $2}' | xargs
        else
            print_warning "Could not detect detailed GPU information. Installing mesa-utils and clinfo..."
            sudo apt-get update && sudo apt-get install -y mesa-utils clinfo

            if command_exists glxinfo; then
                print_step "GPU information from glxinfo:"
                glxinfo | grep -i "OpenGL renderer" | awk -F: '{print $2}' | xargs
            fi

            if command_exists clinfo; then
                print_step "GPU information from clinfo:"
                clinfo | grep -i "Device Name" | awk -F: '{print $2}' | xargs
            fi
        fi

        # Estimate GPU count
        gpu_count=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display' | wc -l)
    fi

    # Set GPU count
    export GPU_COUNT=$gpu_count
    print_success "Detected $GPU_COUNT AMD GPU(s)"

    return 0
}

# Function to check if PyTorch is installed
check_pytorch() {
    print_section "Checking PyTorch"

    if check_component_dir "torch" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/pytorch" ]; then
        print_success "PyTorch is installed"

        # Try to get version if possible
        if $PYTHON_INTERPRETER -c "import torch; print(torch.__version__)" 2>/dev/null; then
            pytorch_version=$($PYTHON_INTERPRETER -c "import torch; print(torch.__version__)" 2>&1)
            print_step "PyTorch version: $pytorch_version"

            # Check CUDA availability
            if $PYTHON_INTERPRETER -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "CUDA is available through ROCm"

                # Get GPU count
                gpu_count=$($PYTHON_INTERPRETER -c "import torch; print(torch.cuda.device_count())" 2>&1)
                print_step "GPU count: $gpu_count"

                # Get GPU names
                for i in $(seq 0 $((gpu_count-1))); do
                    gpu_name=$($PYTHON_INTERPRETER -c "import torch; print(torch.cuda.get_device_name($i))" 2>&1)
                    print_step "GPU $i: $gpu_name"
                done
            else
                print_warning "CUDA is not available through ROCm"
            fi
        else
            print_warning "PyTorch is installed but cannot be imported (Python compatibility issue)"
        fi

        return 0
    else
        print_error "PyTorch is not installed."
        return 1
    fi
}

# Function to check if ONNX Runtime is installed
check_onnxruntime() {
    print_section "Checking ONNX Runtime"

    if check_component_dir "onnxruntime" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/onnxruntime_build" ]; then
        print_success "ONNX Runtime is installed"

        # Try to get version if possible
        if $PYTHON_INTERPRETER -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null; then
            onnx_version=$($PYTHON_INTERPRETER -c "import onnxruntime; print(onnxruntime.__version__)" 2>&1)
            print_step "ONNX Runtime version: $onnx_version"

            # Check available providers
            providers=$($PYTHON_INTERPRETER -c "import onnxruntime; print(onnxruntime.get_available_providers())" 2>&1)
            print_step "Available providers: $providers"

            # Check if ROCMExecutionProvider is available
            if echo "$providers" | grep -q "ROCMExecutionProvider"; then
                print_success "ROCMExecutionProvider is available"
            else
                print_warning "ROCMExecutionProvider is not available"
            fi
        else
            print_warning "ONNX Runtime is installed but cannot be imported (Python compatibility issue)"
        fi

        return 0
    else
        print_error "ONNX Runtime is not installed."
        return 1
    fi
}

# Function to check if MIGraphX is installed
check_migraphx() {
    print_section "Checking MIGraphX"

    if check_component_dir "migraphx" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/migraphx_build" ]; then
        print_success "MIGraphX is installed"
        return 0
    else
        print_error "MIGraphX is not installed."
        return 1
    fi
}

# Function to check if Flash Attention is installed
check_flash_attention() {
    print_section "Checking Flash Attention"

    if check_component_dir "flash_attention_amd" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/ml_stack/flash_attn_amd" ] || [ -d "/home/stan/ml_stack/flash_attn_amd_direct" ]; then
        print_success "Flash Attention is installed"
        return 0
    else
        print_error "Flash Attention is not installed."
        return 1
    fi
}

# Function to check if RCCL is installed
check_rccl() {
    print_section "Checking RCCL"

    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_success "RCCL is installed"
        rccl_path=$(ls -la /opt/rocm/lib/librccl.so)
        print_step "RCCL path: $rccl_path"
        return 0
    else
        print_error "RCCL is not installed."
        return 1
    fi
}

# Function to check if MPI is installed
check_mpi() {
    print_section "Checking MPI"

    if command_exists mpirun; then
        print_success "MPI is installed"
        mpi_version=$(mpirun --version | head -n 1)
        print_step "MPI version: $mpi_version"

        # Check mpi4py
        if check_component_dir "mpi4py" "${COMPONENT_SEARCH_PATHS[@]}"; then
            print_success "mpi4py is installed"
        else
            print_warning "mpi4py is not installed"
        fi

        return 0
    else
        print_error "MPI is not installed."
        return 1
    fi
}

# Function to check if Megatron-LM is installed
check_megatron() {
    print_section "Checking Megatron-LM"

    if check_component_dir "megatron" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/megatron/Megatron-LM" ]; then
        print_success "Megatron-LM is installed"
        return 0
    else
        print_error "Megatron-LM is not installed."
        return 1
    fi
}

# Function to check if Triton is installed
check_triton() {
    print_section "Checking Triton"

    if check_component_dir "triton" "${COMPONENT_SEARCH_PATHS[@]}"; then
        print_success "Triton is installed"
        return 0
    else
        print_warning "Triton is not installed."
        return 1
    fi
}

# Function to check if BITSANDBYTES is installed
check_bitsandbytes() {
    print_section "Checking BITSANDBYTES"

    if check_component_dir "bitsandbytes" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/ml_stack/bitsandbytes" ]; then
        print_success "BITSANDBYTES is installed"
        return 0
    else
        print_warning "BITSANDBYTES is not installed."
        return 1
    fi
}

# Function to check if vLLM is installed
check_vllm() {
    print_section "Checking vLLM"

    if check_component_dir "vllm" "${COMPONENT_SEARCH_PATHS[@]}" || [ -d "/home/stan/vllm_build" ] || [ -d "/home/stan/vllm_py313" ]; then
        print_success "vLLM is installed"
        return 0
    else
        print_warning "vLLM is not installed."
        return 1
    fi
}

# Function to check if ROCm SMI is installed
check_rocm_smi() {
    print_section "Checking ROCm SMI"

    if [ -f "/opt/rocm/bin/rocm-smi" ]; then
        print_success "ROCm SMI is installed"
        return 0
    else
        print_warning "ROCm SMI is not installed."
        return 1
    fi
}

# Function to check if PyTorch Profiler is installed
check_pytorch_profiler() {
    print_section "Checking PyTorch Profiler"

    if check_pytorch; then
        if $PYTHON_INTERPRETER -c "from torch.profiler import profile" 2>/dev/null; then
            print_success "PyTorch Profiler is installed"
            return 0
        else
            print_warning "PyTorch Profiler is not installed."
            return 1
        fi
    else
        print_warning "PyTorch is not installed, so PyTorch Profiler cannot be used."
        return 1
    fi
}

# Function to check if Weights & Biases is installed
check_wandb() {
    print_section "Checking Weights & Biases"

    if check_component_dir "wandb" "${COMPONENT_SEARCH_PATHS[@]}"; then
        print_success "Weights & Biases is installed"
        return 0
    else
        print_warning "Weights & Biases is not installed."
        return 1
    fi
}

# Function to check all components
check_all_components() {
    print_header "Checking All ML Stack Components"

    # Core components
    check_pytorch
    check_onnxruntime
    check_migraphx
    check_flash_attention
    check_rccl
    check_mpi
    check_megatron

    # Extension components
    check_triton
    check_bitsandbytes
    check_vllm
    check_rocm_smi
    check_pytorch_profiler
    check_wandb

    print_header "Component Check Complete"
}

# Function to generate a summary of installed components
generate_component_summary() {
    print_header "ML Stack Component Summary"

    echo -e "${BLUE}${BOLD}Core Components:${RESET}"
    echo -e "- ROCm: $(command_exists hipcc && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- PyTorch: $(check_component_dir "torch" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- ONNX Runtime: $(check_component_dir "onnxruntime" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- MIGraphX: $(check_component_dir "migraphx" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- Flash Attention: $(check_component_dir "flash_attention_amd" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- RCCL: $([ -f "/opt/rocm/lib/librccl.so" ] && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- MPI: $(command_exists mpirun && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- Megatron-LM: $(check_component_dir "megatron" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"

    echo -e "\n${BLUE}${BOLD}Extension Components:${RESET}"
    echo -e "- Triton: $(check_component_dir "triton" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
    echo -e "- BITSANDBYTES: $(check_component_dir "bitsandbytes" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
    echo -e "- vLLM: $(check_component_dir "vllm" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
    echo -e "- ROCm SMI: $([ -f "/opt/rocm/bin/rocm-smi" ] && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
    echo -e "- PyTorch Profiler: $(check_pytorch > /dev/null && $PYTHON_INTERPRETER -c "from torch.profiler import profile" 2>/dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
    echo -e "- Weights & Biases: $(check_component_dir "wandb" "${COMPONENT_SEARCH_PATHS[@]}" > /dev/null && echo -e "${GREEN}Installed${RESET}" || echo -e "${YELLOW}Not installed${RESET}")"
}

# If this script is run directly, show usage information
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_header "ML Stack Component Detector"
    echo "This script is a library of functions for detecting ML Stack components."
    echo "It should be sourced by other scripts, not run directly."
    echo
    echo "Example usage:"
    echo "  source $(basename ${BASH_SOURCE[0]})"
    echo "  check_all_components"
    echo
    echo "Available functions:"
    echo "  detect_rocm - Detect ROCm installation"
    echo "  detect_amd_gpus - Detect AMD GPUs"
    echo "  check_pytorch - Check if PyTorch is installed"
    echo "  check_onnxruntime - Check if ONNX Runtime is installed"
    echo "  check_migraphx - Check if MIGraphX is installed"
    echo "  check_flash_attention - Check if Flash Attention is installed"
    echo "  check_rccl - Check if RCCL is installed"
    echo "  check_mpi - Check if MPI is installed"
    echo "  check_megatron - Check if Megatron-LM is installed"
    echo "  check_triton - Check if Triton is installed"
    echo "  check_bitsandbytes - Check if BITSANDBYTES is installed"
    echo "  check_vllm - Check if vLLM is installed"
    echo "  check_rocm_smi - Check if ROCm SMI is installed"
    echo "  check_pytorch_profiler - Check if PyTorch Profiler is installed"
    echo "  check_wandb - Check if Weights & Biases is installed"
    echo "  check_all_components - Check all components"
    echo "  generate_component_summary - Generate a summary of installed components"

    # Run a demo if requested
    if [[ "$1" == "--demo" ]]; then
        check_all_components
        generate_component_summary
    fi
fi
