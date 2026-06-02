#!/usr/bin/env python3
# =============================================================================
# Matrix Multiplication Benchmark
# =============================================================================
# This script benchmarks matrix multiplication performance on AMD GPUs.
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
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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

def benchmark_matmul(sizes, dtype=torch.float32, num_runs=10, device="cuda"):
    """Benchmark matrix multiplication for different sizes."""
    results = []
    
    for size in sizes:
        print_info(f"Benchmarking matrix size: {size}x{size}")
        
        # Create matrices on CPU
        a_cpu = torch.randn(size, size, dtype=dtype)
        b_cpu = torch.randn(size, size, dtype=dtype)
        
        # CPU test
        start_time = time.time()
        for _ in range(num_runs):
            c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = (time.time() - start_time) / num_runs
        print(f"  CPU time: {cpu_time:.4f} seconds")
        
        # Skip GPU test if device is not available
        if device == "cuda" and not torch.cuda.is_available():
            print_warning("CUDA not available, skipping GPU test")
            continue
        
        # Move matrices to GPU
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        # Warm-up
        for _ in range(3):
            c_gpu = torch.matmul(a_gpu, b_gpu)
        
        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        # GPU test
        start_time = time.time()
        for _ in range(num_runs):
            c_gpu = torch.matmul(a_gpu, b_gpu)
            if device == "cuda":
                torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / num_runs
        
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
        
        # Record results
        results.append({
            "size": size,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": cpu_time / gpu_time
        })
    
    return results

def plot_results(results, output_dir="./"):
    """Plot benchmark results."""
    sizes = [result["size"] for result in results]
    cpu_times = [result["cpu_time"] for result in results]
    gpu_times = [result["gpu_time"] for result in results]
    speedups = [result["speedup"] for result in results]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot times
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'o-', label='CPU')
    plt.plot(sizes, gpu_times, 'o-', label='GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Time vs Matrix Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'matmul_time.png'))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, speedups, 'o-')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup (x)')
    plt.title('GPU Speedup vs Matrix Size')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'matmul_speedup.png'))
    
    # Plot log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, cpu_times, 'o-', label='CPU')
    plt.loglog(sizes, gpu_times, 'o-', label='GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Time vs Matrix Size (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'matmul_time_loglog.png'))
    
    # Save results to CSV
    with open(os.path.join(output_dir, 'matmul_results.csv'), 'w') as f:
        f.write('size,cpu_time,gpu_time,speedup\n')
        for result in results:
            f.write(f"{result['size']},{result['cpu_time']},{result['gpu_time']},{result['speedup']}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark matrix multiplication on AMD GPUs")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192], help="Matrix sizes to benchmark")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32", help="Data type")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    print_header("Matrix Multiplication Benchmark")
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print_warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Print system information
    print_info("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Convert dtype string to torch dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    
    # Run benchmarks
    print_info(f"Running benchmarks with {args.dtype} precision")
    results = benchmark_matmul(args.sizes, dtype=dtype, num_runs=args.num_runs, device=args.device)
    
    # Plot results
    print_info("Plotting results")
    plot_results(results, output_dir=args.output_dir)
    
    print_success(f"Benchmark complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        sys.exit(1)
