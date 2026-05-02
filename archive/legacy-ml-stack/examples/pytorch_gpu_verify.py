#!/usr/bin/env python3
# =============================================================================
# PyTorch GPU Verification Script
# =============================================================================
# This script verifies that PyTorch can access AMD GPUs through ROCm.
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
import time
import argparse

def verify_gpu():
    """Verify GPU availability and run a simple test."""
    print("=== PyTorch GPU Verification ===")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("No CUDA (ROCm) support found. Please check your installation.")
        return
    
    # Get number of GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    
    # Print GPU information
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Run a simple matrix multiplication test
    print("\n=== Running Matrix Multiplication Test ===")
    
    # Test parameters
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create matrices on CPU
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        # CPU test
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU test for each available GPU
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            
            # Move matrices to GPU
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)
            
            # Warm-up
            for _ in range(3):
                c_gpu = torch.matmul(a_gpu, b_gpu)
            
            # Synchronize before timing
            torch.cuda.synchronize(device)
            
            # GPU test
            start_time = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize(device)
            gpu_time = time.time() - start_time
            
            print(f"GPU {i} time: {gpu_time:.4f} seconds")
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PyTorch GPU support")
    args = parser.parse_args()
    
    verify_gpu()
