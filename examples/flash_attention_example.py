#!/usr/bin/env python3
# =============================================================================
# Flash Attention Example for AMD GPUs
# =============================================================================
# This script demonstrates how to use Flash Attention with AMD GPUs.
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

try:
    from flash_attention_amd import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

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

def benchmark_attention(batch_size, seq_len, num_heads, head_dim, causal=True, num_runs=10):
    """Benchmark Flash Attention vs standard attention."""
    print(f"\nBenchmarking with parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Causal: {causal}")
    
    # Create random inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    
    # Warm-up
    for _ in range(5):
        _ = standard_attention(q, k, v, causal=causal)
        if FLASH_ATTN_AVAILABLE:
            _ = flash_attn_func(q, k, v, causal=causal)
    
    # Benchmark standard attention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = standard_attention(q, k, v, causal=causal)
        torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_runs
    
    # Benchmark Flash Attention
    flash_time = None
    if FLASH_ATTN_AVAILABLE:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = flash_attn_func(q, k, v, causal=causal)
            torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / num_runs
    
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = standard_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    flash_memory = None
    if FLASH_ATTN_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        _ = flash_attn_func(q, k, v, causal=causal)
        torch.cuda.synchronize()
        flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Print results
    print("\nResults:")
    print(f"  Standard Attention Time: {standard_time * 1000:.2f} ms")
    if flash_time is not None:
        print(f"  Flash Attention Time: {flash_time * 1000:.2f} ms")
        print(f"  Speedup: {standard_time / flash_time:.2f}x")
    else:
        print("  Flash Attention not available")
    
    print(f"  Standard Attention Memory: {standard_memory:.2f} MB")
    if flash_memory is not None:
        print(f"  Flash Attention Memory: {flash_memory:.2f} MB")
        print(f"  Memory Reduction: {standard_memory / flash_memory:.2f}x")
    
    # Check correctness
    if FLASH_ATTN_AVAILABLE:
        with torch.no_grad():
            standard_output = standard_attention(q, k, v, causal=causal)
            flash_output = flash_attn_func(q, k, v, causal=causal)
            max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
            print(f"  Max Difference: {max_diff:.6f}")
    
    return {
        "seq_len": seq_len,
        "standard_time": standard_time * 1000,  # ms
        "flash_time": flash_time * 1000 if flash_time is not None else None,  # ms
        "standard_memory": standard_memory,  # MB
        "flash_memory": flash_memory,  # MB
    }

def plot_results(results):
    """Plot benchmark results."""
    seq_lengths = [result["seq_len"] for result in results]
    standard_times = [result["standard_time"] for result in results]
    flash_times = [result["flash_time"] for result in results if result["flash_time"] is not None]
    
    # Plot time comparison
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths[:len(flash_times)], standard_times[:len(flash_times)], 'o-', label='Standard Attention')
    plt.plot(seq_lengths[:len(flash_times)], flash_times, 'o-', label='Flash Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Attention Time vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('attention_time_comparison.png')
    
    # Plot speedup
    if flash_times:
        speedups = [standard_times[i] / flash_times[i] for i in range(len(flash_times))]
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths[:len(flash_times)], speedups, 'o-')
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup (x)')
        plt.title('Flash Attention Speedup vs Sequence Length')
        plt.grid(True)
        plt.savefig('attention_speedup.png')
    
    # Plot memory comparison
    standard_memory = [result["standard_memory"] for result in results]
    flash_memory = [result["flash_memory"] for result in results if result["flash_memory"] is not None]
    
    if flash_memory:
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths[:len(flash_memory)], standard_memory[:len(flash_memory)], 'o-', label='Standard Attention')
        plt.plot(seq_lengths[:len(flash_memory)], flash_memory, 'o-', label='Flash Attention')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (MB)')
        plt.title('Attention Memory Usage vs Sequence Length')
        plt.legend()
        plt.grid(True)
        plt.savefig('attention_memory_comparison.png')
        
        # Plot memory reduction
        memory_reductions = [standard_memory[i] / flash_memory[i] for i in range(len(flash_memory))]
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths[:len(flash_memory)], memory_reductions, 'o-')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Reduction (x)')
        plt.title('Flash Attention Memory Reduction vs Sequence Length')
        plt.grid(True)
        plt.savefig('attention_memory_reduction.png')

def main():
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention vs standard attention")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for benchmarking")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()
    
    print("=== Flash Attention Benchmark ===")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
        return
    
    # Check if Flash Attention is available
    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention is not available. Please install it first.")
        print("Standard attention will still be benchmarked.")
    
    # Print GPU information
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run benchmarks for different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    results = []
    
    for seq_len in seq_lengths:
        try:
            result = benchmark_attention(
                batch_size=args.batch_size,
                seq_len=seq_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                causal=args.causal,
                num_runs=args.num_runs
            )
            results.append(result)
        except RuntimeError as e:
            print(f"Error with sequence length {seq_len}: {e}")
            break
    
    # Plot results if requested
    if args.plot and len(results) > 0:
        plot_results(results)
        print("\nPlots saved to:")
        print("  attention_time_comparison.png")
        print("  attention_speedup.png")
        print("  attention_memory_comparison.png")
        print("  attention_memory_reduction.png")
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
