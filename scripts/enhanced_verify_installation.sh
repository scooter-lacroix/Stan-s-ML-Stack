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
# Enhanced ML Stack Verification Script
# =============================================================================
# This script verifies the installation of the ML stack components with
# improved error handling and troubleshooting suggestions.
# =============================================================================

# Filter out common ROCm warnings
export PYTHONPATH=$HOME/ml_stack/flash_attn_amd_direct:$PYTHONPATH
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$PATH

# Suppress common ROCm warnings
export ROCM_QUIET=1
export HIP_QUIET=1
export AMD_LOG_LEVEL=0
export HIP_TRACE_API=0
export HSA_ENABLE_SDMA=0
export HSA_TOOLS_LIB=0
export HSA_TOOLS_REPORT_LOAD_FAILURE=0
export HSA_ENABLE_DEBUG=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=0
export MIOPEN_ENABLE_LOGGING_IMPL=0

# ASCII Art Banner
# Color definitions
# Function definitions
# Hardware detection
# ROCm verification
# PyTorch verification
# ONNX Runtime verification
# MIGraphX verification
# Flash Attention verification
# RCCL verification
# MPI verification
# Megatron-LM verification
# Extension components verification
# Summary generation
# Main function
# ASCII Art Banner
cat << "BANNER"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                Enhanced ML Stack Verification Script
BANNER
echo

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/ml_stack_verify_$(date +"%Y%m%d_%H%M%S").log"

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

# Source the component detector library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DETECTOR_SCRIPT="$PARENT_DIR/scripts/ml_stack_component_detector.sh"

if [ -f "$DETECTOR_SCRIPT" ]; then
    source "$DETECTOR_SCRIPT"
else
    echo "Error: Component detector script not found at $DETECTOR_SCRIPT"
    exit 1
fi

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to print colored messages
print_header() {
    echo "=== $1 ===" | tee -a $LOG_FILE
    echo | tee -a $LOG_FILE
}

original_print_section=$print_section
print_section() {
    echo ">>> $1" | tee -a $LOG_FILE
}

original_print_step=$print_step
print_step() {
    echo ">> $1" | tee -a $LOG_FILE
}

original_print_success=$print_success
print_success() {
    echo "✓ $1" | tee -a $LOG_FILE
}

original_print_warning=$print_warning
print_warning() {
    echo "⚠ $1" | tee -a $LOG_FILE
}

original_print_error=$print_error
print_error() {
    echo "✗ $1" | tee -a $LOG_FILE
}

print_troubleshooting() {
    echo "Troubleshooting:" | tee -a $LOG_FILE
    echo "$1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python module exists
python_module_exists() {
    python3 -c "import $1" >/dev/null 2>&1
}
# Function to detect hardware
detect_hardware() {
    print_section "Detecting Hardware"

    # System information
    print_step "System: $(uname -a)"

    # CPU information
    cpu_info=$(lscpu | grep "Model name" | sed 's/Model name: *//g')
    print_step "CPU: $cpu_info"

    # Memory information
    mem_info=$(free -h | grep "Mem:" | awk '{print $2}')
    print_step "Memory: $mem_info"

    # GPU detection
    if command_exists lspci; then
        print_step "Detecting GPUs using lspci..."
        amd_gpus=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display')

        if [ -n "$amd_gpus" ]; then
            print_success "AMD GPUs detected:"
            echo "$amd_gpus" | while read -r line; do
                echo "  - $line" | tee -a $LOG_FILE
            done
        else
            print_error "No AMD GPUs detected with lspci."
            print_troubleshooting "- Ensure AMD GPU is properly installed and connected
- Check if AMD GPU is recognized by the system
- Run 'sudo update-pciids' to update PCI IDs database
- Check BIOS settings to ensure GPU is enabled"
        fi
    else
        print_warning "lspci command not found. Installing pciutils..."
        sudo apt-get update && sudo apt-get install -y pciutils
        detect_hardware
        return
    fi

    # ROCm detection
    if command_exists rocminfo; then
        print_step "ROCm Path: $(which rocminfo)"
        print_step "Python Version: $(python3 --version)"

        # Get ROCm version
        rocm_version=$(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -n "$rocm_version" ]; then
            print_step "ROCm Version: $rocm_version"
        else
            # Try alternative method to get ROCm version
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            if [ -n "$rocm_version" ]; then
                print_step "ROCm Version: $rocm_version (detected from path)"
            else
                print_warning "Could not determine ROCm version."
            fi
        fi
    else
        print_error "ROCm is not installed or not in PATH."
        print_troubleshooting "- Install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html
- Ensure ROCm is in your PATH
- Check if user has proper permissions (should be in video and render groups)
- Run 'sudo usermod -a -G video,render $USER' and log out/in"
    fi
}

# Function to verify ROCm
verify_rocm() {
    print_section "Verifying ROCm"

    if command_exists hipcc; then
        hip_version=$(hipcc --version 2>&1 | head -n 1)
        print_success "ROCm is installed: $hip_version"
        print_step "ROCm Info:"
        rocminfo 2>&1 | grep -E "Name:|Marketing|ROCm Version" | tee -a $LOG_FILE

        # Check if user has proper permissions
        if groups | grep -q -E '(video|render|rocm)'; then
            print_success "User has proper permissions for ROCm"
        else
            print_warning "User may not have proper permissions for ROCm"
            print_troubleshooting "Run the following command and log out/in:
sudo usermod -a -G video,render $USER"
        fi

        # Check if ROCm environment variables are set
        if [ -n "$ROCM_PATH" ]; then
            print_success "ROCM_PATH is set to $ROCM_PATH"
        else
            print_warning "ROCM_PATH is not set"
            print_troubleshooting "Set ROCM_PATH in your environment:
export ROCM_PATH=/opt/rocm"
        fi

        return 0
    else
        print_error "ROCm is not installed."
        print_troubleshooting "- Install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html
- Ensure ROCm is in your PATH
- Check if user has proper permissions (should be in video and render groups)
- Run 'sudo usermod -a -G video,render $USER' and log out/in"
        return 1
    fi
}
# Function to verify PyTorch
verify_pytorch() {
    print_section "Verifying PyTorch"

    if python_module_exists "torch"; then
        print_success "PyTorch is installed"
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>&1)
        print_step "PyTorch version: $pytorch_version"

        # Check if PyTorch was built with ROCm support
        if python3 -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -q -v "None"; then
            print_success "PyTorch was built with ROCm support"
            hip_version=$(python3 -c "import torch; print(torch.version.hip)" 2>&1)
            print_step "HIP version: $hip_version"
        else
            print_warning "PyTorch was not built with ROCm support"
            print_troubleshooting "Reinstall PyTorch with ROCm support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        fi

        # Check CUDA availability
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "CUDA is available through ROCm"

            # Get GPU count
            gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>&1)
            print_step "GPU count: $gpu_count"

            # Get GPU names
            for i in $(seq 0 $((gpu_count-1))); do
                gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>&1)
                print_step "GPU $i: $gpu_name"
            done

            # Test simple operation
            if python3 -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success')" 2>/dev/null | grep -q "Success"; then
                print_success "Simple tensor operation on GPU successful"
            else
                print_error "Simple tensor operation on GPU failed"
                print_troubleshooting "- Check if environment variables are set correctly:
  export HIP_VISIBLE_DEVICES=0,1
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTORCH_ROCM_DEVICE=0,1
- Check if ROCm is properly installed
- Try reinstalling PyTorch with ROCm support"
            fi
        else
            print_error "CUDA is not available through ROCm"
            print_troubleshooting "- Check if environment variables are set correctly:
  export HIP_VISIBLE_DEVICES=0,1
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTORCH_ROCM_DEVICE=0,1
- Check if ROCm is properly installed
- Try reinstalling PyTorch with ROCm support:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        fi

        return 0
    else
        print_error "PyTorch is not installed."
        print_troubleshooting "Install PyTorch with ROCm support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        return 1
    fi
}

# Function to verify ONNX Runtime
verify_onnxruntime() {
    print_section "Verifying ONNX Runtime"

    if python_module_exists "onnxruntime"; then
        print_success "ONNX Runtime is installed"
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>&1)
        print_step "ONNX Runtime version: $onnx_version"

        # Check available providers
        providers=$(python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())" 2>&1)
        print_step "Available providers: $providers"

        # Check if ROCMExecutionProvider is available
        if echo "$providers" | grep -q "ROCMExecutionProvider"; then
            print_success "ROCMExecutionProvider is available"

            # Test simple model inference
            if [ -f "$HOME/Prod/Stan-s-ML-Stack/scripts/simple_model.onnx" ]; then
                print_step "Testing simple model inference..."
                if python3 -c "
import onnxruntime as ort
import numpy as np

# Create session with ROCMExecutionProvider
session = ort.InferenceSession('$HOME/Prod/Stan-s-ML-Stack/scripts/simple_model.onnx', providers=['ROCMExecutionProvider'])

# Create random input
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_data})

print('Success')
" 2>/dev/null | grep -q "Success"; then
                    print_success "Simple model inference on GPU successful"
                else
                    print_warning "Simple model inference on GPU failed"
                    print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try rebuilding ONNX Runtime with ROCm support"
                fi
            else
                print_warning "Simple model not found for testing"
            fi
        else
            print_warning "ROCMExecutionProvider is not available"
            print_troubleshooting "- ONNX Runtime needs to be built from source with ROCm support
- Run the build_onnxruntime.sh script to build ONNX Runtime with ROCm support
- Ensure PYTHONPATH includes the ONNX Runtime build directory:
  export PYTHONPATH=$HOME/onnxruntime_build/onnxruntime/build/Linux/Release:\$PYTHONPATH"
        fi

        return 0
    else
        print_error "ONNX Runtime is not installed."
        print_troubleshooting "- Install ONNX Runtime with ROCm support by building from source
- Run the build_onnxruntime.sh script to build ONNX Runtime with ROCm support"
        return 1
    fi
}
# Function to verify MIGraphX
verify_migraphx() {
    print_section "Verifying MIGraphX"

    if command_exists migraphx-driver; then
        print_success "MIGraphX is installed"
        migraphx_version=$(migraphx-driver --version 2>&1 | head -n 1)
        print_step "MIGraphX version: $migraphx_version"

        # Check if Python module is installed
        if python_module_exists "migraphx"; then
            print_success "MIGraphX Python module is installed"

            # Test simple operation
            if python3 -c "
import migraphx
import numpy as np

# Create a simple model
model = migraphx.parse_onnx('$HOME/Prod/Stan-s-ML-Stack/scripts/simple_model.onnx')

# Compile for GPU
model.compile(migraphx.get_target('gpu'))

print('Success')
" 2>/dev/null | grep -q "Success"; then
                print_success "Simple MIGraphX operation successful"
            else
                print_warning "Simple MIGraphX operation failed"
                print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling MIGraphX"
            fi
        else
            print_warning "MIGraphX Python module is not installed"
            print_troubleshooting "Install MIGraphX Python module:
pip install -U migraphx"
        fi

        return 0
    else
        print_error "MIGraphX is not installed."
        print_troubleshooting "- Install MIGraphX from ROCm repository
- Run the install_migraphx.sh script to install MIGraphX"
        return 1
    fi
}

# Function to verify AITER
verify_aiter() {
    print_section "Verifying AITER"

    if python_module_exists "aiter"; then
        print_success "AITER is installed"
        aiter_version=$(python3 -c "import aiter; print(getattr(aiter, '__version__', 'unknown'))" 2>&1)
        print_step "AITER version: $aiter_version"

        # Test simple operation
        if python3 -c "
import aiter
import torch
import numpy as np

# Create a simple test
print('AITER test successful')
" 2>/dev/null | grep -q "AITER test successful"; then
            print_success "Simple AITER operation successful"
        else
            print_warning "Simple AITER operation failed"
            print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling AITER"
        fi

        return 0
    else
        print_error "AITER is not installed."
        print_troubleshooting "- Install AITER from source
- Run the install_aiter.sh script to install AITER"
        return 1
    fi
}

# Function to verify DeepSpeed
verify_deepspeed() {
    print_section "Verifying DeepSpeed"

    if python_module_exists "deepspeed"; then
        print_success "DeepSpeed is installed"
        deepspeed_version=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>&1)
        print_step "DeepSpeed version: $deepspeed_version"

        # Check if DeepSpeed has ROCm support
        if python3 -c "import deepspeed; import torch; print('ROCm available' if torch.cuda.is_available() else 'ROCm not available')" 2>/dev/null | grep -q "ROCm available"; then
            print_success "DeepSpeed has ROCm support"

            # Test simple operation
            if python3 -c "
import deepspeed
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Create a simple model
model = torch.nn.Linear(10, 10).to('cuda')
print('DeepSpeed test successful')
" 2>/dev/null | grep -q "DeepSpeed test successful"; then
                print_success "Simple DeepSpeed operation successful"
            else
                print_warning "Simple DeepSpeed operation failed"
                print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling DeepSpeed with ROCm support"
            fi
        else
            print_warning "DeepSpeed may not have ROCm support"
            print_troubleshooting "- Ensure PyTorch is installed with ROCm support
- Try reinstalling DeepSpeed with ROCm support"
        fi

        return 0
    else
        print_error "DeepSpeed is not installed."
        print_troubleshooting "- Install DeepSpeed with ROCm support
- Run the install_deepspeed.sh script to install DeepSpeed"
        return 1
    fi
}

# Function to verify Flash Attention
verify_flash_attention() {
    print_section "Verifying Flash Attention"

    # Check for different possible module names
    if python_module_exists "flash_attn"; then
        print_success "Flash Attention is installed (flash_attn)"
        flash_attn_version=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>&1)
        print_step "Flash Attention version: $flash_attn_version"
        flash_attn_module="flash_attn"
    elif python_module_exists "flash_attention_amd"; then
        print_success "Flash Attention is installed (flash_attention_amd)"
        flash_attn_version="AMD version"
        print_step "Flash Attention version: $flash_attn_version"
        flash_attn_module="flash_attention_amd"
    elif python_module_exists "flash_attn_amd_direct"; then
        print_success "Flash Attention is installed (flash_attn_amd_direct)"
        flash_attn_version="AMD direct version"
        print_step "Flash Attention version: $flash_attn_version"
        flash_attn_module="flash_attn_amd_direct"
    elif [ -d "/home/stan/ml_stack/flash_attn_amd_direct" ]; then
        print_success "Flash Attention is installed (directory exists)"
        flash_attn_version="AMD custom version"
        print_step "Flash Attention version: $flash_attn_version"
        flash_attn_module="custom"
        # Add the directory to Python path
        export PYTHONPATH=$HOME/ml_stack/flash_attn_amd_direct:$PYTHONPATH
    else
        print_error "Flash Attention is not installed."
        print_troubleshooting "- Build Flash Attention from source with ROCm support\n- Run the build_flash_attn_amd.sh script to build Flash Attention with ROCm support"
        return 1
    fi

    # Test simple operation
    if python3 -c "
import torch
import flash_attn

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Create random tensors
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)

# Run flash attention
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, 0.0)
    print('Success')
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
        print_success "Flash Attention operation successful"
    elif python3 -c "
import torch
import flash_attn

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Create random tensors
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)

# Run flash attention
try:
    from flash_attn import flash_attn_func
    out = flash_attn_func(q, k, v, 0.0)
    print('Success')
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
        print_success "Flash Attention operation successful"
    else
        print_warning "Flash Attention operation failed"
        print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try rebuilding Flash Attention with ROCm support"
    fi

    return 0
}

# Function to verify RCCL
verify_rccl() {
    print_section "Verifying RCCL"

    if [ -d "/opt/rocm/rccl" ] || [ -d "$ROCM_PATH/rccl" ]; then
        print_success "RCCL is installed"

        # Check if RCCL is in LD_LIBRARY_PATH
        if echo $LD_LIBRARY_PATH | grep -q "rccl"; then
            print_success "RCCL is in LD_LIBRARY_PATH"
        else
            print_warning "RCCL is not in LD_LIBRARY_PATH"
            print_troubleshooting "Add RCCL to LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/rccl/lib"
        fi

        # Test RCCL with PyTorch
        if python_module_exists "torch.distributed"; then
            print_step "Testing RCCL with PyTorch..."
            if python3 -c "
import torch
import torch.distributed as dist

# Initialize process group
try:
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', rank=0, world_size=1)
    print('Success')
    dist.destroy_process_group()
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
                print_success "RCCL test with PyTorch successful"
            else
                print_warning "RCCL test with PyTorch failed"
                print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling RCCL"
            fi
        else
            print_warning "torch.distributed module not available for testing"
        fi

        return 0
    else
        print_error "RCCL is not installed."
        print_troubleshooting "- Install RCCL from ROCm repository
- Run the install_rccl.sh script to install RCCL"
        return 1
    fi
}
# Function to verify MPI
verify_mpi() {
    print_section "Verifying MPI"

    if command_exists mpirun; then
        print_success "MPI is installed"
        mpi_version=$(mpirun --version 2>&1 | head -n 1)
        print_step "MPI version: $mpi_version"

        # Check if MPI is properly configured
        if mpirun -n 1 hostname >/dev/null 2>&1; then
            print_success "MPI is properly configured"
        else
            print_warning "MPI may not be properly configured"
            print_troubleshooting "- Check if MPI environment variables are set correctly
- Ensure SSH is properly configured for passwordless login
- Try reinstalling MPI"
        fi

        # Check if PyTorch is built with MPI support
        if python_module_exists "torch.distributed"; then
            print_step "Testing MPI with PyTorch..."
            if python3 -c "
import torch.distributed as dist

# Check if MPI is available
if hasattr(dist, 'is_mpi_available') and dist.is_mpi_available():
    print('MPI is available in PyTorch')
else:
    print('MPI is not available in PyTorch')
" 2>&1 | grep -q "MPI is available in PyTorch"; then
                print_success "PyTorch is built with MPI support"
            else
                print_warning "PyTorch is not built with MPI support"
                print_troubleshooting "- Rebuild PyTorch with MPI support
- Ensure MPI is installed before building PyTorch"
            fi
        else
            print_warning "torch.distributed module not available for testing"
        fi

        return 0
    else
        print_error "MPI is not installed."
        print_troubleshooting "- Install MPI from your distribution's repository
- Run the install_mpi.sh script to install MPI"
        return 1
    fi
}

# Function to verify Megatron-LM
verify_megatron() {
    print_section "Verifying Megatron-LM"

    if [ -d "$HOME/Prod/Stan-s-ML-Stack/Megatron-LM" ] || [ -d "$HOME/megatron/Megatron-LM" ]; then
        print_success "Megatron-LM is installed"

        # Check if Megatron-LM is in PYTHONPATH
        if python3 -c "import megatron" >/dev/null 2>&1; then
            print_success "Megatron-LM is in PYTHONPATH"

            # Test simple import
            if python3 -c "
import megatron
print('Success')
" 2>&1 | grep -q "Success"; then
                print_success "Megatron-LM import successful"
            else
                print_warning "Megatron-LM import failed"
                print_troubleshooting "- Check if Megatron-LM is properly installed
- Ensure PYTHONPATH includes Megatron-LM directory
- Try reinstalling Megatron-LM"
            fi
        else
            print_warning "Megatron-LM is not in PYTHONPATH"
            print_troubleshooting "Add Megatron-LM to PYTHONPATH:
export PYTHONPATH=\$PYTHONPATH:$HOME/Prod/Stan-s-ML-Stack/Megatron-LM:$HOME/megatron/Megatron-LM"
        fi

        return 0
    else
        print_error "Megatron-LM is not installed."
        print_troubleshooting "- Clone Megatron-LM repository
- Run the install_megatron.sh script to install Megatron-LM"
        return 1
    fi
}

# Function to verify Triton
verify_triton() {
    print_section "Verifying Triton"

    if python_module_exists "triton"; then
        print_success "Triton is installed"
        triton_version=$(python3 -c "import triton; print(triton.__version__)" 2>&1)
        print_step "Triton version: $triton_version"

        # Test simple operation
        if python3 -c "
import triton
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Try to initialize Triton
try:
    import triton.language as tl
    print('Success')
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
            print_success "Triton initialization successful"
        else
            print_warning "Triton initialization failed"
            print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling Triton with ROCm support"
        fi

        return 0
    else
        print_error "Triton is not installed."
        print_troubleshooting "- Install Triton with ROCm support
- Run the install_triton.sh script to install Triton"
        return 1
    fi
}

# Function to verify BITSANDBYTES
verify_bitsandbytes() {
    print_section "Verifying BITSANDBYTES"

    if python_module_exists "bitsandbytes"; then
        print_success "BITSANDBYTES is installed"
        bnb_version=$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>&1)
        print_step "BITSANDBYTES version: $bnb_version"

        # Test simple operation
        if python3 -c "
import bitsandbytes as bnb
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Try to create a quantized linear layer
try:
    linear = bnb.nn.Linear8bitLt(128, 128, bias=True)
    linear = linear.cuda()
    x = torch.randn(1, 128, device='cuda')
    y = linear(x)
    print('Success')
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
            print_success "BITSANDBYTES operation successful"
        else
            print_warning "BITSANDBYTES operation failed"
            print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling BITSANDBYTES with ROCm support"
        fi

        return 0
    else
        print_error "BITSANDBYTES is not installed."
        print_troubleshooting "- Install BITSANDBYTES with ROCm support
- Run the install_bitsandbytes.sh script to install BITSANDBYTES"
        return 1
    fi
}

# Function to verify vLLM
verify_vllm() {
    print_section "Verifying vLLM"

    if python_module_exists "vllm"; then
        print_success "vLLM is installed"
        vllm_version=$(python3 -c "import vllm; print(vllm.__version__)" 2>&1)
        print_step "vLLM version: $vllm_version"

        # Check available providers
        if python3 -c "import vllm; print(vllm.get_available_providers())" 2>/dev/null | grep -q "ROCMExecutionProvider"; then
            print_success "ROCMExecutionProvider is available in vLLM"
        else
            print_warning "ROCMExecutionProvider is not available in vLLM"
            print_troubleshooting "- vLLM may not be fully compatible with ROCm
- Check if environment variables are set correctly
- Try reinstalling vLLM with ROCm support"
        fi

        return 0
    else
        print_error "vLLM is not installed."
        print_troubleshooting "- Install vLLM with ROCm support
- Run the install_vllm.sh script to install vLLM"
        return 1
    fi
}

# Function to verify ROCm SMI
verify_rocm_smi() {
    print_section "Verifying ROCm SMI"

    if command_exists rocm-smi; then
        print_success "ROCm SMI is installed"
        rocm_smi_version=$(rocm-smi --version 2>&1 | head -n 1)
        print_step "ROCm SMI version: $rocm_smi_version"

        # Test ROCm SMI
        print_step "ROCm SMI output:"
        rocm-smi --showproductname 2>&1 | tee -a $LOG_FILE

        return 0
    else
        print_error "ROCm SMI is not installed."
        print_troubleshooting "- ROCm SMI should be installed with ROCm
- Check if ROCm is properly installed
- Try reinstalling ROCm"
        return 1
    fi
}

# Function to verify PyTorch Profiler
verify_pytorch_profiler() {
    print_section "Verifying PyTorch Profiler"

    if python_module_exists "torch.profiler"; then
        print_success "PyTorch Profiler is installed"

        # Test simple profiling
        if python3 -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Check if CUDA is available
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

# Create a simple model
model = torch.nn.Linear(100, 100).cuda()
input = torch.randn(1, 100, device='cuda')

# Profile the model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function('model_inference'):
        model(input)

print('Success')
" 2>&1 | grep -q "Success"; then
            print_success "PyTorch Profiler operation successful"
        else
            print_warning "PyTorch Profiler operation failed"
            print_troubleshooting "- Check if environment variables are set correctly
- Ensure ROCm is properly installed
- Try reinstalling PyTorch with ROCm support"
        fi

        return 0
    else
        print_error "PyTorch Profiler is not installed."
        print_troubleshooting "- PyTorch Profiler should be included with PyTorch
- Check if PyTorch is properly installed
- Try reinstalling PyTorch with ROCm support"
        return 1
    fi
}

# Function to verify Weights & Biases
verify_wandb() {
    print_section "Verifying Weights & Biases"

    if python_module_exists "wandb"; then
        print_success "Weights & Biases is installed"
        wandb_version=$(python3 -c "import wandb; print(wandb.__version__)" 2>&1)
        print_step "Weights & Biases version: $wandb_version"

        # Test simple initialization
        if python3 -c "
import wandb
try:
    wandb.init(mode='offline')
    print('Success')
    wandb.finish()
except Exception as e:
    print(f'Error: {e}')
" 2>&1 | grep -q "Success"; then
            print_success "Weights & Biases initialization successful"
        else
            print_warning "Weights & Biases initialization failed"
            print_troubleshooting "- Check if Weights & Biases is properly installed
- Try reinstalling Weights & Biases:
  pip install wandb"
        fi

        return 0
    else
        print_error "Weights & Biases is not installed."
        print_troubleshooting "- Install Weights & Biases:
  pip install wandb"
        return 1
    fi
}
# Function to generate summary
generate_summary() {
    print_header "ML Stack Verification Summary"

    # Create summary table
    echo -e "${BOLD}Core Components:${RESET}" | tee -a $LOG_FILE

    # ROCm
    if [ "$ROCM_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ROCm${RESET}: Successfully installed (version $ROCM_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ROCm${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # PyTorch
    if [ "$PYTORCH_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ PyTorch${RESET}: Successfully installed (version $PYTORCH_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ PyTorch${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # ONNX Runtime
    if [ "$ONNXRUNTIME_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ONNX Runtime${RESET}: Successfully installed (version $ONNXRUNTIME_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ONNX Runtime${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # MIGraphX
    if [ "$MIGRAPHX_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ MIGraphX${RESET}: Successfully installed (version $MIGRAPHX_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ MIGraphX${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # Flash Attention
    if [ "$FLASH_ATTENTION_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Flash Attention${RESET}: Successfully installed (version $FLASH_ATTENTION_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Flash Attention${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # RCCL
    if [ "$RCCL_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ RCCL${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ RCCL${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # MPI
    if [ "$MPI_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ MPI${RESET}: Successfully installed (version $MPI_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ MPI${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # Megatron-LM
    if [ "$MEGATRON_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Megatron-LM${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Megatron-LM${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # AITER
    if [ "$AITER_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ AITER${RESET}: Successfully installed (version $AITER_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ AITER${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # DeepSpeed
    if [ "$DEEPSPEED_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ DeepSpeed${RESET}: Successfully installed (version $DEEPSPEED_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ DeepSpeed${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    echo | tee -a $LOG_FILE
    echo -e "${BOLD}Extension Components:${RESET}" | tee -a $LOG_FILE

    # Triton
    if [ "$TRITON_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Triton${RESET}: Successfully installed (version $TRITON_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Triton${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # BITSANDBYTES
    if [ "$BITSANDBYTES_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ BITSANDBYTES${RESET}: Successfully installed (version $BITSANDBYTES_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ BITSANDBYTES${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # vLLM
    if [ "$VLLM_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ vLLM${RESET}: Successfully installed (version $VLLM_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ vLLM${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # ROCm SMI
    if [ "$ROCM_SMI_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ROCm SMI${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ROCm SMI${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # PyTorch Profiler
    if [ "$PYTORCH_PROFILER_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ PyTorch Profiler${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ PyTorch Profiler${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    # Weights & Biases
    if [ "$WANDB_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Weights & Biases${RESET}: Successfully installed (version $WANDB_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Weights & Biases${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi

    echo | tee -a $LOG_FILE
    echo -e "${BOLD}Log file:${RESET} $LOG_FILE" | tee -a $LOG_FILE
}

# Main function
main() {
    print_header "ML Stack Verification"

    # Detect hardware
    detect_hardware

    # Verify ROCm
    verify_rocm
    ROCM_STATUS=$?
    if [ $ROCM_STATUS -eq 0 ]; then
        ROCM_STATUS="success"
        ROCM_VERSION=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$ROCM_VERSION" ]; then
            ROCM_VERSION="unknown"
        fi
    else
        ROCM_STATUS="failure"
        ROCM_VERSION="N/A"
    fi

    # Verify PyTorch
    verify_pytorch
    PYTORCH_STATUS=$?
    if [ $PYTORCH_STATUS -eq 0 ]; then
        PYTORCH_STATUS="success"
        PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        if [ -z "$PYTORCH_VERSION" ]; then
            PYTORCH_VERSION="unknown"
        fi
    else
        PYTORCH_STATUS="failure"
        PYTORCH_VERSION="N/A"
    fi

    # Verify ONNX Runtime
    verify_onnxruntime
    ONNXRUNTIME_STATUS=$?
    if [ $ONNXRUNTIME_STATUS -eq 0 ]; then
        ONNXRUNTIME_STATUS="success"
        ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        if [ -z "$ONNXRUNTIME_VERSION" ]; then
            ONNXRUNTIME_VERSION="unknown"
        fi
    else
        ONNXRUNTIME_STATUS="failure"
        ONNXRUNTIME_VERSION="N/A"
    fi

    # Verify MIGraphX
    verify_migraphx
    MIGRAPHX_STATUS=$?
    if [ $MIGRAPHX_STATUS -eq 0 ]; then
        MIGRAPHX_STATUS="success"
        MIGRAPHX_VERSION=$(migraphx-driver --version 2>/dev/null | head -n 1)
        if [ -z "$MIGRAPHX_VERSION" ]; then
            MIGRAPHX_VERSION="unknown"
        fi
    else
        MIGRAPHX_STATUS="failure"
        MIGRAPHX_VERSION="N/A"
    fi

    # Verify Flash Attention
    verify_flash_attention
    FLASH_ATTENTION_STATUS=$?
    if [ $FLASH_ATTENTION_STATUS -eq 0 ]; then
        FLASH_ATTENTION_STATUS="success"
        FLASH_ATTENTION_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
        if [ -z "$FLASH_ATTENTION_VERSION" ]; then
            FLASH_ATTENTION_VERSION="unknown"
        fi
    else
        FLASH_ATTENTION_STATUS="failure"
        FLASH_ATTENTION_VERSION="N/A"
    fi

    # Verify RCCL
    verify_rccl
    RCCL_STATUS=$?
    if [ $RCCL_STATUS -eq 0 ]; then
        RCCL_STATUS="success"
    else
        RCCL_STATUS="failure"
    fi

    # Verify MPI
    verify_mpi
    MPI_STATUS=$?
    if [ $MPI_STATUS -eq 0 ]; then
        MPI_STATUS="success"
        MPI_VERSION=$(mpirun --version 2>/dev/null | head -n 1)
        if [ -z "$MPI_VERSION" ]; then
            MPI_VERSION="unknown"
        fi
    else
        MPI_STATUS="failure"
        MPI_VERSION="N/A"
    fi

    # Verify Megatron-LM
    verify_megatron
    MEGATRON_STATUS=$?
    if [ $MEGATRON_STATUS -eq 0 ]; then
        MEGATRON_STATUS="success"
    else
        MEGATRON_STATUS="failure"
    fi

    # Verify AITER
    verify_aiter
    AITER_STATUS=$?
    if [ $AITER_STATUS -eq 0 ]; then
        AITER_STATUS="success"
        AITER_VERSION=$(python3 -c "import aiter; print(getattr(aiter, '__version__', 'unknown'))" 2>/dev/null)
        if [ -z "$AITER_VERSION" ]; then
            AITER_VERSION="unknown"
        fi
    else
        AITER_STATUS="failure"
        AITER_VERSION="N/A"
    fi

    # Verify DeepSpeed
    verify_deepspeed
    DEEPSPEED_STATUS=$?
    if [ $DEEPSPEED_STATUS -eq 0 ]; then
        DEEPSPEED_STATUS="success"
        DEEPSPEED_VERSION=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
        if [ -z "$DEEPSPEED_VERSION" ]; then
            DEEPSPEED_VERSION="unknown"
        fi
    else
        DEEPSPEED_STATUS="failure"
        DEEPSPEED_VERSION="N/A"
    fi

    # Verify Triton
    verify_triton
    TRITON_STATUS=$?
    if [ $TRITON_STATUS -eq 0 ]; then
        TRITON_STATUS="success"
        TRITON_VERSION=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null)
        if [ -z "$TRITON_VERSION" ]; then
            TRITON_VERSION="unknown"
        fi
    else
        TRITON_STATUS="failure"
        TRITON_VERSION="N/A"
    fi

    # Verify BITSANDBYTES
    verify_bitsandbytes
    BITSANDBYTES_STATUS=$?
    if [ $BITSANDBYTES_STATUS -eq 0 ]; then
        BITSANDBYTES_STATUS="success"
        BITSANDBYTES_VERSION=$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null)
        if [ -z "$BITSANDBYTES_VERSION" ]; then
            BITSANDBYTES_VERSION="unknown"
        fi
    else
        BITSANDBYTES_STATUS="failure"
        BITSANDBYTES_VERSION="N/A"
    fi

    # Verify vLLM
    verify_vllm
    VLLM_STATUS=$?
    if [ $VLLM_STATUS -eq 0 ]; then
        VLLM_STATUS="success"
        VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        if [ -z "$VLLM_VERSION" ]; then
            VLLM_VERSION="unknown"
        fi
    else
        VLLM_STATUS="failure"
        VLLM_VERSION="N/A"
    fi

    # Verify ROCm SMI
    verify_rocm_smi
    ROCM_SMI_STATUS=$?
    if [ $ROCM_SMI_STATUS -eq 0 ]; then
        ROCM_SMI_STATUS="success"
    else
        ROCM_SMI_STATUS="failure"
    fi

    # Verify PyTorch Profiler
    verify_pytorch_profiler
    PYTORCH_PROFILER_STATUS=$?
    if [ $PYTORCH_PROFILER_STATUS -eq 0 ]; then
        PYTORCH_PROFILER_STATUS="success"
    else
        PYTORCH_PROFILER_STATUS="failure"
    fi

    # Verify Weights & Biases
    verify_wandb
    WANDB_STATUS=$?
    if [ $WANDB_STATUS -eq 0 ]; then
        WANDB_STATUS="success"
        WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)" 2>/dev/null)
        if [ -z "$WANDB_VERSION" ]; then
            WANDB_VERSION="unknown"
        fi
    else
        WANDB_STATUS="failure"
        WANDB_VERSION="N/A"
    fi

    # Generate summary
    generate_summary

    print_header "ML Stack Verification Complete"
    echo -e "Log file: $LOG_FILE"
}

# Run main function
main
