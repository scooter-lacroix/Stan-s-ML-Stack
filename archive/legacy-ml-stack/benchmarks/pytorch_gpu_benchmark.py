#!/usr/bin/env python3
# =============================================================================
# PyTorch GPU Benchmark
# =============================================================================
# This script benchmarks PyTorch operations on AMD GPUs.
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
import argparse
import json
from pathlib import Path

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

def benchmark_matmul(sizes, dtype, num_runs):
    """Benchmark matrix multiplication."""
    print_info("Benchmarking matrix multiplication...")
    
    # Set dtype
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        print_error(f"Unsupported dtype: {dtype}")
        return None
    
    results = []
    
    for size in sizes:
        print_info(f"Matrix size: {size}x{size}")
        
        # Create matrices
        a_cpu = torch.randn(size, size, dtype=torch_dtype)
        b_cpu = torch.randn(size, size, dtype=torch_dtype)
        
        # Move to GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warm-up
        for _ in range(5):
            _ = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        # Calculate statistics
        avg_time = sum(times) / num_runs
        min_time = min(times)
        max_time = max(times)
        std_time = np.std(times)
        
        print_info(f"Average time: {avg_time * 1000:.2f} ms")
        print_info(f"Min time: {min_time * 1000:.2f} ms")
        print_info(f"Max time: {max_time * 1000:.2f} ms")
        print_info(f"Std time: {std_time * 1000:.2f} ms")
        
        # Calculate FLOPS
        # Matrix multiplication requires 2*N^3 floating-point operations
        flops = 2 * size**3
        avg_flops = flops / avg_time
        
        print_info(f"FLOPS: {avg_flops:.2e}")
        print_info(f"TFLOPS: {avg_flops / 1e12:.2f}")
        
        # Add to results
        results.append({
            "size": size,
            "avg_time": avg_time * 1000,  # Convert to ms
            "min_time": min_time * 1000,  # Convert to ms
            "max_time": max_time * 1000,  # Convert to ms
            "std_time": std_time * 1000,  # Convert to ms
            "flops": avg_flops,
            "tflops": avg_flops / 1e12
        })
    
    return results

def benchmark_conv2d(batch_sizes, channels, image_sizes, kernel_sizes, dtype, num_runs):
    """Benchmark 2D convolution."""
    print_info("Benchmarking 2D convolution...")
    
    # Set dtype
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        print_error(f"Unsupported dtype: {dtype}")
        return None
    
    results = []
    
    for batch_size in batch_sizes:
        for in_channels, out_channels in channels:
            for image_size in image_sizes:
                for kernel_size in kernel_sizes:
                    print_info(f"Batch size: {batch_size}, Channels: {in_channels}->{out_channels}, Image size: {image_size}x{image_size}, Kernel size: {kernel_size}x{kernel_size}")
                    
                    # Create input tensor
                    x = torch.randn(batch_size, in_channels, image_size, image_size, dtype=torch_dtype, device="cuda")
                    
                    # Create convolution layer
                    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dtype=torch_dtype, device="cuda")
                    
                    # Warm-up
                    for _ in range(5):
                        _ = conv(x)
                        torch.cuda.synchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(num_runs):
                        torch.cuda.synchronize()
                        start_time = time.time()
                        _ = conv(x)
                        torch.cuda.synchronize()
                        times.append(time.time() - start_time)
                    
                    # Calculate statistics
                    avg_time = sum(times) / num_runs
                    min_time = min(times)
                    max_time = max(times)
                    std_time = np.std(times)
                    
                    print_info(f"Average time: {avg_time * 1000:.2f} ms")
                    print_info(f"Min time: {min_time * 1000:.2f} ms")
                    print_info(f"Max time: {max_time * 1000:.2f} ms")
                    print_info(f"Std time: {std_time * 1000:.2f} ms")
                    
                    # Calculate FLOPS
                    # Convolution requires 2*K^2*C_in*C_out*H*W floating-point operations per batch
                    flops = 2 * kernel_size**2 * in_channels * out_channels * image_size**2 * batch_size
                    avg_flops = flops / avg_time
                    
                    print_info(f"FLOPS: {avg_flops:.2e}")
                    print_info(f"TFLOPS: {avg_flops / 1e12:.2f}")
                    
                    # Add to results
                    results.append({
                        "batch_size": batch_size,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "image_size": image_size,
                        "kernel_size": kernel_size,
                        "avg_time": avg_time * 1000,  # Convert to ms
                        "min_time": min_time * 1000,  # Convert to ms
                        "max_time": max_time * 1000,  # Convert to ms
                        "std_time": std_time * 1000,  # Convert to ms
                        "flops": avg_flops,
                        "tflops": avg_flops / 1e12
                    })
    
    return results

def benchmark_pytorch_gpu(output_dir, dtype, num_runs):
    """Benchmark PyTorch operations on AMD GPUs."""
    print_header("PyTorch GPU Benchmark")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    print_info(f"Number of GPUs: {device_count}")
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print_info(f"GPU {i}: {device_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark results
    results = {
        "dtype": dtype,
        "num_runs": num_runs,
        "gpu_info": [],
        "matmul": None,
        "conv2d": None
    }
    
    # Add GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_properties = torch.cuda.get_device_properties(i)
        
        results["gpu_info"].append({
            "id": i,
            "name": device_name,
            "total_memory": device_properties.total_memory,
            "multi_processor_count": device_properties.multi_processor_count,
            "max_threads_per_multi_processor": device_properties.max_threads_per_multi_processor,
            "clock_rate": device_properties.clock_rate
        })
    
    # Benchmark matrix multiplication
    matmul_sizes = [1024, 2048, 4096, 8192]
    results["matmul"] = benchmark_matmul(matmul_sizes, dtype, num_runs)
    
    # Benchmark 2D convolution
    conv2d_batch_sizes = [1, 8, 16, 32]
    conv2d_channels = [(3, 64), (64, 128), (128, 256)]
    conv2d_image_sizes = [32, 64, 128, 224]
    conv2d_kernel_sizes = [3, 5, 7]
    results["conv2d"] = benchmark_conv2d(conv2d_batch_sizes, conv2d_channels, conv2d_image_sizes, conv2d_kernel_sizes, dtype, num_runs)
    
    # Save results to JSON
    json_file = os.path.join(output_dir, "pytorch_gpu_benchmark.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print_success(f"Saved benchmark results to {json_file}")
    
    # Plot matrix multiplication results
    plt.figure(figsize=(10, 6))
    sizes = [result["size"] for result in results["matmul"]]
    times = [result["avg_time"] for result in results["matmul"]]
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.title(f'Matrix Multiplication Time ({dtype})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "matmul_time.png"))
    
    plt.figure(figsize=(10, 6))
    tflops = [result["tflops"] for result in results["matmul"]]
    plt.plot(sizes, tflops, 'o-')
    plt.xlabel('Matrix Size')
    plt.ylabel('TFLOPS')
    plt.title(f'Matrix Multiplication Performance ({dtype})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "matmul_tflops.png"))
    
    # Plot 2D convolution results
    plt.figure(figsize=(10, 6))
    batch_sizes = sorted(list(set([result["batch_size"] for result in results["conv2d"]])))
    for batch_size in batch_sizes:
        batch_results = [result for result in results["conv2d"] if result["batch_size"] == batch_size and result["in_channels"] == 64 and result["out_channels"] == 128 and result["kernel_size"] == 3]
        image_sizes = [result["image_size"] for result in batch_results]
        times = [result["avg_time"] for result in batch_results]
        plt.plot(image_sizes, times, 'o-', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Image Size')
    plt.ylabel('Time (ms)')
    plt.title(f'2D Convolution Time (64->128 channels, 3x3 kernel, {dtype})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "conv2d_time.png"))
    
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        batch_results = [result for result in results["conv2d"] if result["batch_size"] == batch_size and result["in_channels"] == 64 and result["out_channels"] == 128 and result["kernel_size"] == 3]
        image_sizes = [result["image_size"] for result in batch_results]
        tflops = [result["tflops"] for result in batch_results]
        plt.plot(image_sizes, tflops, 'o-', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Image Size')
    plt.ylabel('TFLOPS')
    plt.title(f'2D Convolution Performance (64->128 channels, 3x3 kernel, {dtype})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "conv2d_tflops.png"))
    
    print_success("Created plots")
    
    print_success("Benchmark completed")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="PyTorch GPU Benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type to use")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs for each benchmark")
    
    args = parser.parse_args()
    
    success = benchmark_pytorch_gpu(args.output_dir, args.dtype, args.num_runs)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
