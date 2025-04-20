#!/usr/bin/env python3
# =============================================================================
# PyTorch Memory Test
# =============================================================================
# This script tests PyTorch memory management on AMD GPUs.
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
import matplotlib.pyplot as plt

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

def test_pytorch_memory():
    """Test PyTorch memory management on AMD GPUs."""
    print_header("PyTorch Memory Test")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    print_info(f"Number of GPUs: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print_info(f"GPU {i}: {device_name}")
    
    # Test memory allocation
    print_info("Testing memory allocation...")
    
    try:
        # Get initial memory stats
        init_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        init_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Initial allocated memory: {init_allocated:.2f} MB")
        print_info(f"Initial reserved memory: {init_reserved:.2f} MB")
        
        # Allocate a 1 GB tensor
        tensor_size = 1024 * 1024 * 1024 // 4  # 1 GB in float32
        tensor = torch.ones(tensor_size, dtype=torch.float32, device="cuda")
        
        # Get memory stats after allocation
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Allocated memory after 1 GB tensor: {allocated:.2f} MB")
        print_info(f"Reserved memory after 1 GB tensor: {reserved:.2f} MB")
        
        # Free the tensor
        del tensor
        torch.cuda.empty_cache()
        
        # Get memory stats after freeing
        final_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        final_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Final allocated memory: {final_allocated:.2f} MB")
        print_info(f"Final reserved memory: {final_reserved:.2f} MB")
        
        print_success("Memory allocation test successful")
        
    except Exception as e:
        print_error(f"Memory allocation test failed: {e}")
        return False
    
    # Test memory fragmentation
    print_info("Testing memory fragmentation...")
    
    try:
        # Allocate many small tensors
        small_tensors = []
        for i in range(1000):
            tensor = torch.ones(1000, dtype=torch.float32, device="cuda")
            small_tensors.append(tensor)
        
        # Get memory stats
        fragmented_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        fragmented_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Allocated memory after small tensors: {fragmented_allocated:.2f} MB")
        print_info(f"Reserved memory after small tensors: {fragmented_reserved:.2f} MB")
        
        # Free every other tensor
        for i in range(0, len(small_tensors), 2):
            del small_tensors[i]
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get memory stats after partial freeing
        partial_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        partial_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Allocated memory after partial freeing: {partial_allocated:.2f} MB")
        print_info(f"Reserved memory after partial freeing: {partial_reserved:.2f} MB")
        
        # Try to allocate a large tensor
        try:
            large_tensor = torch.ones(1024 * 1024 * 512 // 4, dtype=torch.float32, device="cuda")
            print_success("Large tensor allocation successful")
            del large_tensor
        except RuntimeError as e:
            print_warning(f"Large tensor allocation failed: {e}")
        
        # Free all tensors
        del small_tensors
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get final memory stats
        final_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        final_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print_info(f"Final allocated memory: {final_allocated:.2f} MB")
        print_info(f"Final reserved memory: {final_reserved:.2f} MB")
        
        print_success("Memory fragmentation test successful")
        
    except Exception as e:
        print_error(f"Memory fragmentation test failed: {e}")
        return False
    
    # Test max_split_size_mb parameter
    print_info("Testing max_split_size_mb parameter...")
    
    try:
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024 ** 2)
        
        print_info(f"Total GPU memory: {total_memory:.2f} MB")
        
        # Test different max_split_size_mb values
        split_sizes = [128, 256, 512, 1024]
        times = []
        
        for split_size in split_sizes:
            # Set max_split_size_mb
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
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
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(split_sizes)), times)
        plt.xticks(range(len(split_sizes)), [str(size) for size in split_sizes])
        plt.xlabel("max_split_size_mb")
        plt.ylabel("Time (seconds)")
        plt.title("Matrix Multiplication Time vs max_split_size_mb")
        plt.savefig("max_split_size_benchmark.png")
        
        print_info("Benchmark plot saved to max_split_size_benchmark.png")
        
    except Exception as e:
        print_error(f"max_split_size_mb test failed: {e}")
        return False
    
    print_success("All PyTorch memory tests passed")
    return True

if __name__ == "__main__":
    success = test_pytorch_memory()
    sys.exit(0 if success else 1)
