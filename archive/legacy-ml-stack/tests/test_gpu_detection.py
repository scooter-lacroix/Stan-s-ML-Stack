#!/usr/bin/env python3
# =============================================================================
# GPU Detection Test
# =============================================================================
# This script tests if AMD GPUs are properly detected by PyTorch.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import torch
import sys
import os

# Add colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}INFO: {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}SUCCESS: {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}WARNING: {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}ERROR: {text}{Colors.END}")

def test_gpu_detection():
    """Test if AMD GPUs are properly detected by PyTorch."""
    print_header("GPU Detection Test")
    
    # Check PyTorch version
    print_info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print_success("CUDA (ROCm) is available")
    else:
        print_error("CUDA (ROCm) is not available")
        print_info("Check if ROCm is installed and environment variables are set correctly:")
        print_info("  - HIP_VISIBLE_DEVICES")
        print_info("  - CUDA_VISIBLE_DEVICES")
        print_info("  - PYTORCH_ROCM_DEVICE")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print_success(f"Number of GPUs: {device_count}")
    else:
        print_error("No GPUs detected")
        return False
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        if "AMD" in device_name or "Radeon" in device_name:
            print_success(f"GPU {i}: {device_name}")
        else:
            print_warning(f"GPU {i}: {device_name} (not an AMD GPU)")
    
    # Check environment variables
    hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    pytorch_rocm_device = os.environ.get("PYTORCH_ROCM_DEVICE")
    
    print_info("Environment Variables:")
    print(f"  HIP_VISIBLE_DEVICES: {hip_visible_devices}")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"  PYTORCH_ROCM_DEVICE: {pytorch_rocm_device}")
    
    # Try a simple tensor operation on GPU
    try:
        x = torch.ones(10, device="cuda")
        y = x + 1
        print_success("Simple tensor operation on GPU successful")
    except Exception as e:
        print_error(f"Simple tensor operation on GPU failed: {e}")
        return False
    
    # Try a more complex operation (matrix multiplication)
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = torch.matmul(a, b)
        print_success("Matrix multiplication on GPU successful")
    except Exception as e:
        print_error(f"Matrix multiplication on GPU failed: {e}")
        return False
    
    print_success("All GPU detection tests passed")
    return True

if __name__ == "__main__":
    success = test_gpu_detection()
    sys.exit(0 if success else 1)
