#!/usr/bin/env python3
# =============================================================================
# Debug Flash Attention
# =============================================================================
# This script helps debug Flash Attention issues on AMD GPUs.
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

def debug_flash_attention():
    """Debug Flash Attention issues on AMD GPUs."""
    print_header("Debug Flash Attention")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Check if Flash Attention is installed
    try:
        import flash_attention_amd
        print_success("Flash Attention is installed")
    except ImportError:
        print_error("Flash Attention is not installed")
        print_info("Please install Flash Attention first")
        return False
    
    # Print Flash Attention version
    try:
        print_info(f"Flash Attention version: {flash_attention_amd.__version__}")
    except AttributeError:
        print_warning("Could not determine Flash Attention version")
    
    # Check GPU information
    device_count = torch.cuda.device_count()
    print_info(f"Number of GPUs: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print_info(f"GPU {i}: {device_name}")
    
    # Test standard attention
    print_info("Testing standard attention...")
    
    try:
        # Create input tensors
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        
        # Define standard attention function
        def standard_attention(q, k, v, causal=False):
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
        
        # Run standard attention
        torch.cuda.synchronize()
        start_time = time.time()
        standard_output = standard_attention(q, k, v)
        torch.cuda.synchronize()
        standard_time = time.time() - start_time
        
        print_success("Standard attention successful")
        print_info(f"Standard attention time: {standard_time * 1000:.2f} ms")
        
    except Exception as e:
        print_error(f"Standard attention failed: {e}")
        return False
    
    # Test Flash Attention
    print_info("Testing Flash Attention...")
    
    try:
        # Run Flash Attention
        torch.cuda.synchronize()
        start_time = time.time()
        flash_output = flash_attention_amd.flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        flash_time = time.time() - start_time
        
        print_success("Flash Attention successful")
        print_info(f"Flash Attention time: {flash_time * 1000:.2f} ms")
        print_info(f"Speedup: {standard_time / flash_time:.2f}x")
        
        # Compare outputs
        max_diff = torch.max(torch.abs(standard_output - flash_output)).item()
        print_info(f"Maximum difference between standard and Flash Attention: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print_success("Outputs match within tolerance")
        else:
            print_warning("Outputs differ significantly")
            print_info("This could be due to numerical precision differences")
        
    except Exception as e:
        print_error(f"Flash Attention failed: {e}")
        print_info("Error details:")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Flash Attention with causal mask
    print_info("Testing Flash Attention with causal mask...")
    
    try:
        # Run standard attention with causal mask
        torch.cuda.synchronize()
        start_time = time.time()
        standard_causal_output = standard_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        standard_causal_time = time.time() - start_time
        
        # Run Flash Attention with causal mask
        torch.cuda.synchronize()
        start_time = time.time()
        flash_causal_output = flash_attention_amd.flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        flash_causal_time = time.time() - start_time
        
        print_success("Flash Attention with causal mask successful")
        print_info(f"Standard attention with causal mask time: {standard_causal_time * 1000:.2f} ms")
        print_info(f"Flash Attention with causal mask time: {flash_causal_time * 1000:.2f} ms")
        print_info(f"Speedup: {standard_causal_time / flash_causal_time:.2f}x")
        
        # Compare outputs
        max_diff = torch.max(torch.abs(standard_causal_output - flash_causal_output)).item()
        print_info(f"Maximum difference with causal mask: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print_success("Causal mask outputs match within tolerance")
        else:
            print_warning("Causal mask outputs differ significantly")
            print_info("This could be due to numerical precision differences")
        
    except Exception as e:
        print_error(f"Flash Attention with causal mask failed: {e}")
        return False
    
    # Test Flash Attention with longer sequences
    print_info("Testing Flash Attention with longer sequences...")
    
    try:
        # Create input tensors with longer sequences
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64
        
        q_long = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        k_long = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        v_long = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        
        # Run Flash Attention with longer sequences
        torch.cuda.synchronize()
        start_time = time.time()
        flash_long_output = flash_attention_amd.flash_attn_func(q_long, k_long, v_long)
        torch.cuda.synchronize()
        flash_long_time = time.time() - start_time
        
        print_success("Flash Attention with longer sequences successful")
        print_info(f"Flash Attention with longer sequences time: {flash_long_time * 1000:.2f} ms")
        
    except Exception as e:
        print_error(f"Flash Attention with longer sequences failed: {e}")
        return False
    
    # Test Flash Attention with different head dimensions
    print_info("Testing Flash Attention with different head dimensions...")
    
    try:
        # Create input tensors with different head dimensions
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 128
        
        q_diff = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        k_diff = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        v_diff = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
        
        # Run Flash Attention with different head dimensions
        torch.cuda.synchronize()
        start_time = time.time()
        flash_diff_output = flash_attention_amd.flash_attn_func(q_diff, k_diff, v_diff)
        torch.cuda.synchronize()
        flash_diff_time = time.time() - start_time
        
        print_success("Flash Attention with different head dimensions successful")
        print_info(f"Flash Attention with different head dimensions time: {flash_diff_time * 1000:.2f} ms")
        
    except Exception as e:
        print_error(f"Flash Attention with different head dimensions failed: {e}")
        return False
    
    print_success("All Flash Attention debug tests passed")
    return True

if __name__ == "__main__":
    success = debug_flash_attention()
    sys.exit(0 if success else 1)
