#!/usr/bin/env python3
# =============================================================================
# Memory Bandwidth Benchmark
# =============================================================================
# This script benchmarks memory bandwidth on AMD GPUs.
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

def benchmark_memory_bandwidth(sizes, dtype=torch.float32, num_runs=10, device="cuda"):
    """Benchmark memory bandwidth for different sizes."""
    results = []
    
    for size in sizes:
        # Calculate memory size in bytes
        num_elements = size * 1024 * 1024  # Convert MB to elements
        bytes_per_element = 4 if dtype == torch.float32 else 2  # 4 bytes for float32, 2 bytes for float16
        memory_size_bytes = num_elements * bytes_per_element
        memory_size_mb = memory_size_bytes / (1024 * 1024)
        
        print_info(f"Benchmarking memory size: {memory_size_mb:.2f} MB")
        
        # Create tensor on device
        try:
            x = torch.ones(num_elements, dtype=dtype, device=device)
        except RuntimeError as e:
            print_error(f"Failed to allocate memory: {e}")
            break
        
        # Warm-up
        for _ in range(3):
            y = x + 1
        
        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Measure read bandwidth
        start_time = time.time()
        for _ in range(num_runs):
            y = x.clone()
            if device == "cuda":
                torch.cuda.synchronize()
        read_time = (time.time() - start_time) / num_runs
        
        # Measure write bandwidth
        start_time = time.time()
        for _ in range(num_runs):
            x.fill_(1.0)
            if device == "cuda":
                torch.cuda.synchronize()
        write_time = (time.time() - start_time) / num_runs
        
        # Measure read+write bandwidth
        start_time = time.time()
        for _ in range(num_runs):
            y = x + 1
            if device == "cuda":
                torch.cuda.synchronize()
        read_write_time = (time.time() - start_time) / num_runs
        
        # Calculate bandwidth in GB/s
        read_bandwidth = memory_size_bytes / (read_time * 1e9)
        write_bandwidth = memory_size_bytes / (write_time * 1e9)
        read_write_bandwidth = 2 * memory_size_bytes / (read_write_time * 1e9)  # 2x because read and write
        
        print(f"  Read bandwidth: {read_bandwidth:.2f} GB/s")
        print(f"  Write bandwidth: {write_bandwidth:.2f} GB/s")
        print(f"  Read+Write bandwidth: {read_write_bandwidth:.2f} GB/s")
        
        # Record results
        results.append({
            "size_mb": memory_size_mb,
            "read_bandwidth": read_bandwidth,
            "write_bandwidth": write_bandwidth,
            "read_write_bandwidth": read_write_bandwidth
        })
    
    return results

def plot_results(results, output_dir="./"):
    """Plot benchmark results."""
    sizes = [result["size_mb"] for result in results]
    read_bandwidths = [result["read_bandwidth"] for result in results]
    write_bandwidths = [result["write_bandwidth"] for result in results]
    read_write_bandwidths = [result["read_write_bandwidth"] for result in results]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot bandwidths
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, read_bandwidths, 'o-', label='Read')
    plt.plot(sizes, write_bandwidths, 'o-', label='Write')
    plt.plot(sizes, read_write_bandwidths, 'o-', label='Read+Write')
    plt.xlabel('Memory Size (MB)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Memory Bandwidth vs Memory Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_bandwidth.png'))
    
    # Save results to CSV
    with open(os.path.join(output_dir, 'memory_bandwidth_results.csv'), 'w') as f:
        f.write('size_mb,read_bandwidth,write_bandwidth,read_write_bandwidth\n')
        for result in results:
            f.write(f"{result['size_mb']},{result['read_bandwidth']},{result['write_bandwidth']},{result['read_write_bandwidth']}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark memory bandwidth on AMD GPUs")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], help="Memory sizes to benchmark in MB")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32", help="Data type")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    print_header("Memory Bandwidth Benchmark")
    
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
    results = benchmark_memory_bandwidth(args.sizes, dtype=dtype, num_runs=args.num_runs, device=args.device)
    
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
