#!/bin/bash
#
# Script to check for ML Stack components in specific locations
#

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

# Function to check if a directory exists
check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓ Directory exists:${RESET} $1"
        return 0
    else
        echo -e "${RED}✗ Directory does not exist:${RESET} $1"
        return 1
    fi
}

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓ File exists:${RESET} $1"
        return 0
    else
        echo -e "${RED}✗ File does not exist:${RESET} $1"
        return 1
    fi
}

# Function to check if a Python module can be imported
check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓ Python module can be imported:${RESET} $1"
        return 0
    else
        echo -e "${RED}✗ Python module cannot be imported:${RESET} $1"
        return 1
    fi
}

# Function to check if a command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Command exists:${RESET} $1"
        return 0
    else
        echo -e "${RED}✗ Command does not exist:${RESET} $1"
        return 1
    fi
}

# Check Flash Attention
echo -e "${BLUE}=== Checking Flash Attention ===${RESET}"
check_directory "/home/stan/ml_stack/flash_attn_amd_direct"
check_python_module "flash_attn" || check_python_module "flash_attention_amd" || check_python_module "flash_attn_amd_direct"

# Check RCCL
echo -e "${BLUE}=== Checking RCCL ===${RESET}"
check_directory "/opt/rocm/rccl" || check_file "/opt/rocm/lib/librccl.so"
python3 -c "import torch.distributed as dist; print('NCCL available' if hasattr(dist, 'Backend') and 'nccl' in dist.Backend._plugins else 'NCCL not available')" 2>/dev/null

# Check Megatron-LM
echo -e "${BLUE}=== Checking Megatron-LM ===${RESET}"
check_directory "/home/stan/megatron/Megatron-LM" || check_directory "$HOME/Desktop/Stans_MLStack/Megatron-LM" || check_directory "$HOME/Megatron-LM"
check_python_module "megatron" || check_python_module "megatron_core"

# Check environment variables
echo -e "${BLUE}=== Checking Environment Variables ===${RESET}"
if [ -n "$HIP_VISIBLE_DEVICES" ]; then
    echo -e "${GREEN}✓ HIP_VISIBLE_DEVICES is set:${RESET} $HIP_VISIBLE_DEVICES"
else
    echo -e "${YELLOW}⚠ HIP_VISIBLE_DEVICES is not set${RESET}"
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES is set:${RESET} $CUDA_VISIBLE_DEVICES"
else
    echo -e "${YELLOW}⚠ CUDA_VISIBLE_DEVICES is not set${RESET}"
fi

if [ -n "$PYTORCH_ROCM_DEVICE" ]; then
    echo -e "${GREEN}✓ PYTORCH_ROCM_DEVICE is set:${RESET} $PYTORCH_ROCM_DEVICE"
else
    echo -e "${YELLOW}⚠ PYTORCH_ROCM_DEVICE is not set${RESET}"
fi

if [ -n "$ROCM_HOME" ]; then
    echo -e "${GREEN}✓ ROCM_HOME is set:${RESET} $ROCM_HOME"
else
    echo -e "${YELLOW}⚠ ROCM_HOME is not set${RESET}"
fi

if [ -n "$CUDA_HOME" ]; then
    echo -e "${GREEN}✓ CUDA_HOME is set:${RESET} $CUDA_HOME"
else
    echo -e "${YELLOW}⚠ CUDA_HOME is not set${RESET}"
fi

if [ -n "$HSA_OVERRIDE_GFX_VERSION" ]; then
    echo -e "${GREEN}✓ HSA_OVERRIDE_GFX_VERSION is set:${RESET} $HSA_OVERRIDE_GFX_VERSION"
else
    echo -e "${YELLOW}⚠ HSA_OVERRIDE_GFX_VERSION is not set${RESET}"
fi

if [ -n "$HSA_TOOLS_LIB" ]; then
    echo -e "${GREEN}✓ HSA_TOOLS_LIB is set:${RESET} $HSA_TOOLS_LIB"
else
    echo -e "${YELLOW}⚠ HSA_TOOLS_LIB is not set${RESET}"
fi

# Check Python path
echo -e "${BLUE}=== Checking Python Path ===${RESET}"
python3 -c "import sys; print('\n'.join(sys.path))" | grep -E "flash|rccl|megatron|onnx"

# Check if PyTorch can see the GPUs
echo -e "${BLUE}=== Checking PyTorch GPU Detection ===${RESET}"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check if ONNX Runtime can see the ROCMExecutionProvider
echo -e "${BLUE}=== Checking ONNX Runtime Providers ===${RESET}"
python3 -c "
import onnxruntime
print(f'Available providers: {onnxruntime.get_available_providers()}')
"
