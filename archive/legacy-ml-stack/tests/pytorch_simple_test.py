#!/usr/bin/env python3
# =============================================================================
# PyTorch Simple Test
# =============================================================================
# This script performs a simple test of PyTorch functionality on AMD GPUs.
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
import time

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

def test_pytorch_simple():
    """Perform a simple test of PyTorch functionality on AMD GPUs."""
    print_header("PyTorch Simple Test")
    
    # Check PyTorch version
    print_info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print_success("CUDA is available through ROCm")
    else:
        print_error("CUDA is not available through ROCm")
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
    
    # Test basic tensor operations
    print_info("Testing basic tensor operations...")
    
    try:
        # Create a tensor on CPU
        cpu_tensor = torch.tensor([1, 2, 3, 4, 5])
        print_success("Created tensor on CPU")
        
        # Move tensor to GPU
        gpu_tensor = cpu_tensor.cuda()
        print_success("Moved tensor to GPU")
        
        # Perform a simple operation
        result = gpu_tensor * 2
        print_success("Performed simple operation on GPU")
        
        # Move result back to CPU
        result_cpu = result.cpu()
        print_success("Moved result back to CPU")
        
        # Check result
        expected = torch.tensor([2, 4, 6, 8, 10])
        if torch.all(result_cpu == expected):
            print_success("Result is correct")
        else:
            print_error("Result is incorrect")
            print_info(f"Expected: {expected}")
            print_info(f"Got: {result_cpu}")
            return False
        
    except Exception as e:
        print_error(f"Basic tensor operations failed: {e}")
        return False
    
    # Test matrix multiplication
    print_info("Testing matrix multiplication...")
    
    try:
        # Create matrices
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        
        # Move matrices to GPU
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        
        # Perform matrix multiplication on CPU
        start_time = time.time()
        c_cpu = torch.matmul(a, b)
        cpu_time = time.time() - start_time
        
        # Perform matrix multiplication on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Move result back to CPU
        c_gpu_cpu = c_gpu.cpu()
        
        # Check result
        if torch.allclose(c_cpu, c_gpu_cpu, rtol=1e-3, atol=1e-3):
            print_success("Matrix multiplication result is correct")
        else:
            print_error("Matrix multiplication result is incorrect")
            return False
        
        # Print timing information
        print_info(f"CPU time: {cpu_time * 1000:.2f} ms")
        print_info(f"GPU time: {gpu_time * 1000:.2f} ms")
        print_info(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
    except Exception as e:
        print_error(f"Matrix multiplication failed: {e}")
        return False
    
    print_success("All PyTorch simple tests passed")
    return True

if __name__ == "__main__":
    success = test_pytorch_simple()
    sys.exit(0 if success else 1)
