#!/usr/bin/env python3
# =============================================================================
# Flash Attention Benchmark
# =============================================================================
# This script benchmarks Flash Attention performance on AMD GPUs.
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

def standard_attention(q, k, v, causal=False):
    """Standard scaled dot-product attention."""
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

def benchmark_attention(batch_sizes, seq_lengths, num_heads=8, head_dim=64, causal=True, 
                       dtype=torch.float32, num_runs=10, device="cuda", use_flash_attn=True):
    """Benchmark attention for different batch sizes and sequence lengths."""
    results = []
    
    # Try to import flash_attention_amd
    flash_attn_available = False
    if use_flash_attn:
        try:
            from flash_attention_amd import flash_attn_func
            flash_attn_available = True
            print_success("Flash Attention is available")
        except ImportError:
            print_warning("Flash Attention is not available. Using standard attention only.")
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print_info(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
            
            # Create input data
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            
            # Warm-up standard attention
            for _ in range(3):
                _ = standard_attention(q, k, v, causal=causal)
            
            # Synchronize before timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark standard attention
            start_time = time.time()
            for _ in range(num_runs):
                _ = standard_attention(q, k, v, causal=causal)
                if device == "cuda":
                    torch.cuda.synchronize()
            standard_time = (time.time() - start_time) / num_runs
            
            # Calculate memory usage for standard attention
            torch.cuda.reset_peak_memory_stats()
            _ = standard_attention(q, k, v, causal=causal)
            if device == "cuda":
                torch.cuda.synchronize()
            standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            print(f"  Standard Attention Time: {standard_time * 1000:.2f} ms")
            print(f"  Standard Attention Memory: {standard_memory:.2f} MB")
            
            # Benchmark Flash Attention if available
            flash_time = None
            flash_memory = None
            max_diff = None
            
            if flash_attn_available:
                # Warm-up Flash Attention
                for _ in range(3):
                    _ = flash_attn_func(q, k, v, causal=causal)
                
                # Synchronize before timing
                if device == "cuda":
                    torch.cuda.synchronize()
                
                # Benchmark Flash Attention
                start_time = time.time()
                for _ in range(num_runs):
                    _ = flash_attn_func(q, k, v, causal=causal)
                    if device == "cuda":
                        torch.cuda.synchronize()
                flash_time = (time.time() - start_time) / num_runs
                
                # Calculate memory usage for Flash Attention
                torch.cuda.reset_peak_memory_stats()
                _ = flash_attn_func(q, k, v, causal=causal)
                if device == "cuda":
                    torch.cuda.synchronize()
                flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                
                # Check correctness
                with torch.no_grad():
                    standard_output = standard_attention(q, k, v, causal=causal)
                    flash_output = flash_attn_func(q, k, v, causal=causal)
                    max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
                
                print(f"  Flash Attention Time: {flash_time * 1000:.2f} ms")
                print(f"  Flash Attention Memory: {flash_memory:.2f} MB")
                print(f"  Speedup: {standard_time / flash_time:.2f}x")
                print(f"  Memory Reduction: {standard_memory / flash_memory:.2f}x")
                print(f"  Max Difference: {max_diff:.6f}")
            
            # Record results
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "standard_time": standard_time,
                "flash_time": flash_time,
                "standard_memory": standard_memory,
                "flash_memory": flash_memory,
                "max_diff": max_diff
            })
    
    return results

def plot_results(results, output_dir="./"):
    """Plot benchmark results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by batch size
    batch_sizes = sorted(list(set(result["batch_size"] for result in results)))
    
    # Plot time comparison
    plt.figure(figsize=(12, 8))
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r["batch_size"] == batch_size]
        batch_results.sort(key=lambda x: x["seq_len"])
        
        seq_lengths = [r["seq_len"] for r in batch_results]
        standard_times = [r["standard_time"] * 1000 for r in batch_results]  # Convert to ms
        
        plt.plot(seq_lengths, standard_times, 'o-', label=f'Standard (BS={batch_size})')
        
        # Plot Flash Attention times if available
        if batch_results[0]["flash_time"] is not None:
            flash_times = [r["flash_time"] * 1000 for r in batch_results]  # Convert to ms
            plt.plot(seq_lengths, flash_times, 's--', label=f'Flash (BS={batch_size})')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Time vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'attention_time.png'))
    
    # Plot speedup
    if results[0]["flash_time"] is not None:
        plt.figure(figsize=(12, 8))
        for batch_size in batch_sizes:
            batch_results = [r for r in results if r["batch_size"] == batch_size]
            batch_results.sort(key=lambda x: x["seq_len"])
            
            seq_lengths = [r["seq_len"] for r in batch_results]
            speedups = [r["standard_time"] / r["flash_time"] for r in batch_results]
            
            plt.plot(seq_lengths, speedups, 'o-', label=f'BS={batch_size}')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup (x)')
        plt.title('Flash Attention Speedup vs Sequence Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'attention_speedup.png'))
    
    # Plot memory comparison
    plt.figure(figsize=(12, 8))
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r["batch_size"] == batch_size]
        batch_results.sort(key=lambda x: x["seq_len"])
        
        seq_lengths = [r["seq_len"] for r in batch_results]
        standard_memory = [r["standard_memory"] for r in batch_results]
        
        plt.plot(seq_lengths, standard_memory, 'o-', label=f'Standard (BS={batch_size})')
        
        # Plot Flash Attention memory if available
        if batch_results[0]["flash_memory"] is not None:
            flash_memory = [r["flash_memory"] for r in batch_results]
            plt.plot(seq_lengths, flash_memory, 's--', label=f'Flash (BS={batch_size})')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.title('Attention Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'attention_memory.png'))
    
    # Plot memory reduction
    if results[0]["flash_memory"] is not None:
        plt.figure(figsize=(12, 8))
        for batch_size in batch_sizes:
            batch_results = [r for r in results if r["batch_size"] == batch_size]
            batch_results.sort(key=lambda x: x["seq_len"])
            
            seq_lengths = [r["seq_len"] for r in batch_results]
            memory_reductions = [r["standard_memory"] / r["flash_memory"] for r in batch_results]
            
            plt.plot(seq_lengths, memory_reductions, 'o-', label=f'BS={batch_size}')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Reduction (x)')
        plt.title('Flash Attention Memory Reduction vs Sequence Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'attention_memory_reduction.png'))
    
    # Save results to CSV
    with open(os.path.join(output_dir, 'attention_results.csv'), 'w') as f:
        f.write('batch_size,seq_len,standard_time,flash_time,standard_memory,flash_memory,max_diff\n')
        for result in results:
            flash_time = result["flash_time"] if result["flash_time"] is not None else ""
            flash_memory = result["flash_memory"] if result["flash_memory"] is not None else ""
            max_diff = result["max_diff"] if result["max_diff"] is not None else ""
            f.write(f"{result['batch_size']},{result['seq_len']},{result['standard_time']},{flash_time},{result['standard_memory']},{flash_memory},{max_diff}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention on AMD GPUs")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096], help="Sequence lengths to benchmark")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32", help="Data type")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--no-flash", action="store_true", help="Disable Flash Attention")
    args = parser.parse_args()
    
    print_header("Flash Attention Benchmark")
    
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
    results = benchmark_attention(
        args.batch_sizes, 
        args.seq_lengths, 
        num_heads=args.num_heads, 
        head_dim=args.head_dim, 
        causal=args.causal, 
        dtype=dtype, 
        num_runs=args.num_runs, 
        device=args.device,
        use_flash_attn=not args.no_flash
    )
    
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
