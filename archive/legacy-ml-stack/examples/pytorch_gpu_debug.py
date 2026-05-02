#!/usr/bin/env python3
# =============================================================================
# PyTorch GPU Debug
# =============================================================================
# This script helps debug PyTorch GPU issues on AMD GPUs.
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
import numpy as np

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

def debug_pytorch_gpu():
    """Debug PyTorch GPU issues on AMD GPUs."""
    print_header("PyTorch GPU Debug")
    
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
    
    # Check environment variables
    print_info("Environment variables:")
    hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    pytorch_rocm_device = os.environ.get("PYTORCH_ROCM_DEVICE")
    
    print(f"  HIP_VISIBLE_DEVICES: {hip_visible_devices}")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"  PYTORCH_ROCM_DEVICE: {pytorch_rocm_device}")
    
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
    
    # Check GPU memory
    print_info("GPU memory:")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024 ** 2)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 2)
        
        print(f"  GPU {i}:")
        print(f"    Total memory: {total_memory:.2f} MB")
        print(f"    Allocated memory: {allocated_memory:.2f} MB")
        print(f"    Reserved memory: {reserved_memory:.2f} MB")
    
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
            print_info(f"Max absolute difference: {torch.max(torch.abs(c_cpu - c_gpu_cpu)).item()}")
            return False
        
        # Print timing information
        print_info(f"CPU time: {cpu_time * 1000:.2f} ms")
        print_info(f"GPU time: {gpu_time * 1000:.2f} ms")
        print_info(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        # Check if GPU is faster than CPU
        if cpu_time <= gpu_time:
            print_warning("GPU is not faster than CPU for matrix multiplication")
            print_info("This could be due to:")
            print_info("  - Small matrix size")
            print_info("  - GPU initialization overhead")
            print_info("  - Suboptimal GPU configuration")
            print_info("  - Memory transfer overhead")
        
    except Exception as e:
        print_error(f"Matrix multiplication failed: {e}")
        return False
    
    # Test max_split_size_mb parameter
    print_info("Testing max_split_size_mb parameter...")
    
    try:
        # Get current max_split_size_mb
        current_max_split_size_mb = torch.cuda.max_split_size_mb
        print_info(f"Current max_split_size_mb: {current_max_split_size_mb}")
        
        # Test different max_split_size_mb values
        split_sizes = [128, 256, 512, 1024]
        times = []
        
        for split_size in split_sizes:
            # Set max_split_size_mb
            torch.cuda.empty_cache()
            torch.cuda.max_split_size_mb = split_size
            
            print_info(f"Testing with max_split_size_mb = {split_size}")
            
            # Create large matrices
            matrix_size = 8192
            a = torch.randn(matrix_size, matrix_size, device="cuda")
            b = torch.randn(matrix_size, matrix_size, device="cuda")
            
            # Warm-up
            for _ in range(3):
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            
            print_info(f"Time: {elapsed_time * 1000:.2f} ms")
            
            # Clean up
            del a, b, c
            torch.cuda.empty_cache()
        
        # Find the best split size
        best_index = times.index(min(times))
        best_split_size = split_sizes[best_index]
        
        print_success(f"Best max_split_size_mb value: {best_split_size}")
        print_info(f"Recommended setting: torch.cuda.max_split_size_mb = {best_split_size}")
        
        # Restore original value
        torch.cuda.max_split_size_mb = current_max_split_size_mb
        
    except Exception as e:
        print_error(f"max_split_size_mb test failed: {e}")
        # Restore original value
        torch.cuda.max_split_size_mb = current_max_split_size_mb
        return False
    
    print_success("All PyTorch GPU debug tests passed")
    return True

if __name__ == "__main__":
    success = debug_pytorch_gpu()
    sys.exit(0 if success else 1)
