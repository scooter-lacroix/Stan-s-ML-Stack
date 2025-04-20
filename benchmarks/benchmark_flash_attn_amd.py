#!/usr/bin/env python3
# =============================================================================
# Benchmark Flash Attention AMD
# =============================================================================
# This script benchmarks the AMD-specific implementation of Flash Attention.
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
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("benchmark_flash_attn_amd")

def standard_attention(q, k, v, causal=False):
    """Standard attention implementation."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Reshape for batched matrix multiplication
    q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute attention output
    output = torch.matmul(attn_weights, v)
    
    # Reshape back
    output = output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
    
    return output

def benchmark_flash_attn_amd(batch_sizes, seq_lengths, num_heads, head_dim, causal, dtype, num_runs, output_dir):
    """Benchmark Flash Attention AMD implementation."""
    logger.info("Starting Flash Attention AMD benchmark")
    
    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    benchmark_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(os.path.join(benchmark_dir, "benchmark.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        logger.error("CUDA (ROCm) is not available")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {device_count}")
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i}: {device_name}")
    
    # Set dtype
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        logger.error(f"Unsupported dtype: {dtype}")
        return False
    
    # Check if Flash Attention AMD is installed
    try:
        import flash_attention_amd
        logger.info("Flash Attention AMD is installed")
    except ImportError:
        logger.error("Flash Attention AMD is not installed")
        logger.info("Please install Flash Attention AMD first")
        return False
    
    # Benchmark results
    results = {
        "batch_sizes": batch_sizes,
        "seq_lengths": seq_lengths,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "causal": causal,
        "dtype": dtype,
        "num_runs": num_runs,
        "standard_times": [],
        "flash_times": [],
        "speedups": [],
        "max_diffs": [],
        "gpu_info": []
    }
    
    # Add GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        results["gpu_info"].append({
            "id": i,
            "name": device_name
        })
    
    # Create CSV file for results
    csv_file = os.path.join(benchmark_dir, "benchmark_results.csv")
    with open(csv_file, "w") as f:
        f.write("batch_size,seq_length,standard_time,flash_time,speedup,max_diff\n")
    
    # Benchmark Flash Attention for different batch sizes and sequence lengths
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            logger.info(f"Benchmarking Flash Attention for batch_size={batch_size}, seq_length={seq_length}...")
            
            # Create input tensors
            q = torch.randn(batch_size, seq_length, num_heads, head_dim, dtype=torch_dtype, device="cuda")
            k = torch.randn(batch_size, seq_length, num_heads, head_dim, dtype=torch_dtype, device="cuda")
            v = torch.randn(batch_size, seq_length, num_heads, head_dim, dtype=torch_dtype, device="cuda")
            
            # Warm-up standard attention
            for _ in range(5):
                _ = standard_attention(q, k, v, causal=causal)
                torch.cuda.synchronize()
            
            # Benchmark standard attention
            standard_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                standard_output = standard_attention(q, k, v, causal=causal)
                torch.cuda.synchronize()
                standard_times.append(time.time() - start_time)
            
            # Calculate average standard attention time
            avg_standard_time = sum(standard_times) / num_runs
            
            logger.info(f"Standard attention time: {avg_standard_time * 1000:.2f} ms")
            
            # Warm-up Flash Attention
            for _ in range(5):
                _ = flash_attention_amd.flash_attn_func(q, k, v, causal=causal)
                torch.cuda.synchronize()
            
            # Benchmark Flash Attention
            flash_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                flash_output = flash_attention_amd.flash_attn_func(q, k, v, causal=causal)
                torch.cuda.synchronize()
                flash_times.append(time.time() - start_time)
            
            # Calculate average Flash Attention time
            avg_flash_time = sum(flash_times) / num_runs
            
            logger.info(f"Flash Attention time: {avg_flash_time * 1000:.2f} ms")
            
            # Calculate speedup
            speedup = avg_standard_time / avg_flash_time
            
            logger.info(f"Speedup: {speedup:.2f}x")
            
            # Compare outputs
            max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
            
            logger.info(f"Maximum difference: {max_diff:.6f}")
            
            # Add results
            results["standard_times"].append({
                "batch_size": batch_size,
                "seq_length": seq_length,
                "time": avg_standard_time * 1000  # Convert to ms
            })
            
            results["flash_times"].append({
                "batch_size": batch_size,
                "seq_length": seq_length,
                "time": avg_flash_time * 1000  # Convert to ms
            })
            
            results["speedups"].append({
                "batch_size": batch_size,
                "seq_length": seq_length,
                "speedup": speedup
            })
            
            results["max_diffs"].append({
                "batch_size": batch_size,
                "seq_length": seq_length,
                "max_diff": max_diff
            })
            
            # Append to CSV file
            with open(csv_file, "a") as f:
                f.write(f"{batch_size},{seq_length},{avg_standard_time * 1000:.2f},{avg_flash_time * 1000:.2f},{speedup:.2f},{max_diff:.6f}\n")
    
    # Save results to JSON
    json_file = os.path.join(benchmark_dir, "benchmark_results.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved benchmark results to {json_file}")
    
    # Plot results
    # Attention times
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        standard_times = [entry["time"] for entry in results["standard_times"] if entry["batch_size"] == batch_size]
        flash_times = [entry["time"] for entry in results["flash_times"] if entry["batch_size"] == batch_size]
        
        plt.plot(seq_lengths, standard_times, 'o-', label=f'Standard (Batch {batch_size})')
        plt.plot(seq_lengths, flash_times, 's--', label=f'Flash (Batch {batch_size})')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title(f'Attention Time ({dtype}, {"Causal" if causal else "Non-causal"})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(benchmark_dir, "attention_time.png"))
    
    # Speedups
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        speedups = [entry["speedup"] for entry in results["speedups"] if entry["batch_size"] == batch_size]
        plt.plot(seq_lengths, speedups, 'o-', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup (Standard Time / Flash Time)')
    plt.title(f'Flash Attention Speedup ({dtype}, {"Causal" if causal else "Non-causal"})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(benchmark_dir, "speedup.png"))
    
    logger.info("Created plots")
    logger.info(f"Benchmark results saved to {benchmark_dir}")
    
    # Create a symlink to the latest benchmark
    latest_link = os.path.join(output_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(benchmark_dir, latest_link)
    
    logger.info(f"Created symlink to latest benchmark: {latest_link}")
    
    logger.info("Benchmark completed successfully")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention AMD")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64,
                        help="Dimension of each attention head")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal attention")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type to use")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="flash_attn_amd_benchmarks",
                        help="Output directory for benchmark results")
    
    args = parser.parse_args()
    
    success = benchmark_flash_attn_amd(
        args.batch_sizes,
        args.seq_lengths,
        args.num_heads,
        args.head_dim,
        args.causal,
        args.dtype,
        args.num_runs,
        args.output_dir
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
