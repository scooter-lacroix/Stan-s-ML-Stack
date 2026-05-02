#!/usr/bin/env python3
# =============================================================================
# Analyze Flash Attention Results
# =============================================================================
# This script analyzes the results of Flash Attention benchmarks on AMD GPUs.
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

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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

def analyze_flash_attn_results(input_file, output_dir):
    """Analyze Flash Attention benchmark results."""
    print_header("Analyze Flash Attention Results")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print_error(f"Input file not found: {input_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load benchmark results
    try:
        with open(input_file, "r") as f:
            results = json.load(f)
        
        print_success(f"Loaded benchmark results from {input_file}")
    except Exception as e:
        print_error(f"Failed to load benchmark results: {e}")
        return False
    
    # Extract data
    batch_sizes = results.get("batch_sizes", [])
    seq_lengths = results.get("seq_lengths", [])
    num_heads = results.get("num_heads", 0)
    head_dim = results.get("head_dim", 0)
    causal = results.get("causal", False)
    dtype = results.get("dtype", "unknown")
    
    standard_times = results.get("standard_times", [])
    flash_times = results.get("flash_times", [])
    speedups = results.get("speedups", [])
    max_diffs = results.get("max_diffs", [])
    
    # Create DataFrame
    data = []
    for entry in standard_times:
        batch_size = entry.get("batch_size", 0)
        seq_length = entry.get("seq_length", 0)
        standard_time = entry.get("time", 0)
        
        # Find corresponding flash time
        flash_time = next((e.get("time", 0) for e in flash_times if e.get("batch_size") == batch_size and e.get("seq_length") == seq_length), 0)
        
        # Find corresponding speedup
        speedup = next((e.get("speedup", 0) for e in speedups if e.get("batch_size") == batch_size and e.get("seq_length") == seq_length), 0)
        
        # Find corresponding max diff
        max_diff = next((e.get("max_diff", 0) for e in max_diffs if e.get("batch_size") == batch_size and e.get("seq_length") == seq_length), 0)
        
        data.append({
            "batch_size": batch_size,
            "seq_length": seq_length,
            "standard_time": standard_time,
            "flash_time": flash_time,
            "speedup": speedup,
            "max_diff": max_diff
        })
    
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    csv_file = os.path.join(output_dir, "benchmark_data.csv")
    df.to_csv(csv_file, index=False)
    print_success(f"Saved benchmark data to {csv_file}")
    
    # Create summary
    summary = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "causal": causal,
        "dtype": dtype,
        "batch_sizes": batch_sizes,
        "seq_lengths": seq_lengths,
        "avg_speedup": df["speedup"].mean(),
        "max_speedup": df["speedup"].max(),
        "min_speedup": df["speedup"].min(),
        "avg_max_diff": df["max_diff"].mean(),
        "max_max_diff": df["max_diff"].max(),
        "min_max_diff": df["max_diff"].min()
    }
    
    # Save summary to JSON
    json_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)
    print_success(f"Saved benchmark summary to {json_file}")
    
    # Create summary markdown
    md_file = os.path.join(output_dir, "benchmark_summary.md")
    with open(md_file, "w") as f:
        f.write("# Flash Attention Benchmark Summary\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Number of attention heads: {num_heads}\n")
        f.write(f"- Attention head dimension: {head_dim}\n")
        f.write(f"- Causal attention: {causal}\n")
        f.write(f"- Data type: {dtype}\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"- Average speedup: {summary['avg_speedup']:.2f}x\n")
        f.write(f"- Maximum speedup: {summary['max_speedup']:.2f}x\n")
        f.write(f"- Minimum speedup: {summary['min_speedup']:.2f}x\n")
        f.write(f"- Average maximum difference: {summary['avg_max_diff']:.6f}\n")
        f.write(f"- Maximum maximum difference: {summary['max_max_diff']:.6f}\n")
        f.write(f"- Minimum maximum difference: {summary['min_max_diff']:.6f}\n\n")
        
        f.write(f"## Detailed Results\n\n")
        f.write("| Batch Size | Sequence Length | Standard Time (ms) | Flash Time (ms) | Speedup | Max Difference |\n")
        f.write("|------------|-----------------|-------------------|----------------|---------|----------------|\n")
        
        for _, row in df.sort_values(["batch_size", "seq_length"]).iterrows():
            f.write(f"| {row['batch_size']} | {row['seq_length']} | {row['standard_time']:.2f} | {row['flash_time']:.2f} | {row['speedup']:.2f}x | {row['max_diff']:.6f} |\n")
    
    print_success(f"Saved benchmark summary markdown to {md_file}")
    
    # Create plots
    # Attention times
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        batch_df = df[df["batch_size"] == batch_size]
        plt.plot(batch_df["seq_length"], batch_df["standard_time"], 'o-', label=f'Standard (Batch {batch_size})')
        plt.plot(batch_df["seq_length"], batch_df["flash_time"], 's--', label=f'Flash (Batch {batch_size})')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title(f'Attention Time ({dtype}, {"Causal" if causal else "Non-causal"})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "attention_time.png"))
    
    # Speedups
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        batch_df = df[df["batch_size"] == batch_size]
        plt.plot(batch_df["seq_length"], batch_df["speedup"], 'o-', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup (Standard Time / Flash Time)')
    plt.title(f'Flash Attention Speedup ({dtype}, {"Causal" if causal else "Non-causal"})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "speedup.png"))
    
    # Max differences
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        batch_df = df[df["batch_size"] == batch_size]
        plt.plot(batch_df["seq_length"], batch_df["max_diff"], 'o-', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Maximum Difference')
    plt.title(f'Flash Attention Maximum Difference ({dtype}, {"Causal" if causal else "Non-causal"})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "max_diff.png"))
    
    print_success("Created plots")
    
    print_success("Analysis completed")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze Flash Attention Results")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input JSON file with benchmark results")
    parser.add_argument("--output-dir", type=str, default="flash_attn_amd_results",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    success = analyze_flash_attn_results(args.input_file, args.output_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
