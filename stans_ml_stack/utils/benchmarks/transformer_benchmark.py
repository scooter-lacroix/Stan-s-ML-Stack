#!/usr/bin/env python3
# =============================================================================
# Transformer Model Benchmark
# =============================================================================
# This script benchmarks transformer model performance on AMD GPUs.
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
import torch.nn as nn
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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        output = self.transformer_encoder(src, src_mask)
        return output

def benchmark_transformer(batch_sizes, seq_lengths, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6, 
                         dtype=torch.float32, num_runs=10, device="cuda", use_amp=False):
    """Benchmark transformer model for different batch sizes and sequence lengths."""
    results = []
    
    # Create model
    model = TransformerModel(d_model, nhead, dim_feedforward, num_layers)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print_info(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
            
            # Create input data
            src = torch.randn(seq_len, batch_size, d_model, device=device, dtype=dtype)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _ = model(src)
                    else:
                        _ = model(src)
            
            # Synchronize before timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                for _ in range(num_runs):
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _ = model(src)
                    else:
                        _ = model(src)
                    if device == "cuda":
                        torch.cuda.synchronize()
                forward_time = (time.time() - start_time) / num_runs
            
            # Backward pass
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            target = torch.randn(seq_len, batch_size, d_model, device=device, dtype=dtype)
            criterion = nn.MSELoss()
            
            start_time = time.time()
            for _ in range(num_runs):
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(src)
                        loss = criterion(output, target)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    output = model(src)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
            backward_time = (time.time() - start_time) / num_runs
            
            # Calculate tokens per second
            tokens_per_second_forward = (batch_size * seq_len) / forward_time
            tokens_per_second_backward = (batch_size * seq_len) / backward_time
            
            print(f"  Forward time: {forward_time * 1000:.2f} ms")
            print(f"  Backward time: {backward_time * 1000:.2f} ms")
            print(f"  Tokens per second (forward): {tokens_per_second_forward:.2f}")
            print(f"  Tokens per second (backward): {tokens_per_second_backward:.2f}")
            
            # Record results
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "tokens_per_second_forward": tokens_per_second_forward,
                "tokens_per_second_backward": tokens_per_second_backward
            })
    
    return results

def plot_results(results, output_dir="./"):
    """Plot benchmark results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by sequence length
    seq_lengths = sorted(list(set(result["seq_len"] for result in results)))
    batch_sizes = sorted(list(set(result["batch_size"] for result in results)))
    
    # Plot forward time
    plt.figure(figsize=(12, 8))
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        seq_results.sort(key=lambda x: x["batch_size"])
        
        batch_sizes = [r["batch_size"] for r in seq_results]
        forward_times = [r["forward_time"] * 1000 for r in seq_results]  # Convert to ms
        
        plt.plot(batch_sizes, forward_times, 'o-', label=f'Seq Len = {seq_len}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Forward Time (ms)')
    plt.title('Transformer Forward Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'transformer_forward_time.png'))
    
    # Plot backward time
    plt.figure(figsize=(12, 8))
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        seq_results.sort(key=lambda x: x["batch_size"])
        
        batch_sizes = [r["batch_size"] for r in seq_results]
        backward_times = [r["backward_time"] * 1000 for r in seq_results]  # Convert to ms
        
        plt.plot(batch_sizes, backward_times, 'o-', label=f'Seq Len = {seq_len}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Backward Time (ms)')
    plt.title('Transformer Backward Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'transformer_backward_time.png'))
    
    # Plot tokens per second (forward)
    plt.figure(figsize=(12, 8))
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        seq_results.sort(key=lambda x: x["batch_size"])
        
        batch_sizes = [r["batch_size"] for r in seq_results]
        tokens_per_second = [r["tokens_per_second_forward"] for r in seq_results]
        
        plt.plot(batch_sizes, tokens_per_second, 'o-', label=f'Seq Len = {seq_len}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens per Second')
    plt.title('Transformer Forward Tokens per Second vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'transformer_tokens_per_second_forward.png'))
    
    # Plot tokens per second (backward)
    plt.figure(figsize=(12, 8))
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        seq_results.sort(key=lambda x: x["batch_size"])
        
        batch_sizes = [r["batch_size"] for r in seq_results]
        tokens_per_second = [r["tokens_per_second_backward"] for r in seq_results]
        
        plt.plot(batch_sizes, tokens_per_second, 'o-', label=f'Seq Len = {seq_len}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens per Second')
    plt.title('Transformer Backward Tokens per Second vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'transformer_tokens_per_second_backward.png'))
    
    # Save results to CSV
    with open(os.path.join(output_dir, 'transformer_results.csv'), 'w') as f:
        f.write('batch_size,seq_len,forward_time,backward_time,tokens_per_second_forward,tokens_per_second_backward\n')
        for result in results:
            f.write(f"{result['batch_size']},{result['seq_len']},{result['forward_time']},{result['backward_time']},{result['tokens_per_second_forward']},{result['tokens_per_second_backward']}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark transformer model on AMD GPUs")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512, 1024], help="Sequence lengths to benchmark")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32", help="Data type")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    args = parser.parse_args()
    
    print_header("Transformer Model Benchmark")
    
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
    results = benchmark_transformer(
        args.batch_sizes, 
        args.seq_lengths, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        dim_feedforward=args.dim_feedforward, 
        num_layers=args.num_layers, 
        dtype=dtype, 
        num_runs=args.num_runs, 
        device=args.device,
        use_amp=args.use_amp
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
