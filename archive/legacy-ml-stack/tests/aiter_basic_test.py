#!/usr/bin/env python3
# =============================================================================
# AITER Basic Test
# =============================================================================
# This script tests basic AITER functionality with AMD GPUs.
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

def test_aiter_basic():
    """Test basic AITER functionality."""
    print_header("AITER Basic Test")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Try to import AITER
    try:
        import aiter
        print_success("AITER module imported successfully")
    except ImportError:
        print_error("Failed to import AITER module")
        print_info("Make sure AITER is installed")
        return False
    
    # Test basic tensor operations
    try:
        # Create tensors
        a = torch.randn(10, 20, device="cuda")
        b = torch.randn(20, 30, device="cuda")
        
        # Matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print_success("Basic tensor operations successful")
        print_info(f"Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
        
    except Exception as e:
        print_error(f"Basic tensor operations failed: {e}")
        return False
    
    # Test AITER basic operations
    try:
        from aiter.ops import basic_op
        
        # Create input tensor
        x = torch.randn(8, 16, device="cuda")
        
        # Apply basic operation
        y = basic_op(x)
        
        print_success("AITER basic operations successful")
        print_info(f"Input shape: {x.shape}")
        print_info(f"Output shape: {y.shape}")
        
    except Exception as e:
        print_error(f"AITER basic operations failed: {e}")
        return False
    
    # Test AITER with a basic model
    try:
        from aiter.models import BasicModel
        
        # Create a basic model
        model = BasicModel(
            input_size=16,
            hidden_size=32,
            output_size=8
        ).to("cuda")
        
        # Create input tensor
        x = torch.randn(4, 16, device="cuda")
        
        # Forward pass
        output = model(x)
        
        print_success("AITER basic model successful")
        print_info(f"Model: {model}")
        print_info(f"Input shape: {x.shape}")
        print_info(f"Output shape: {output.shape}")
        
    except Exception as e:
        print_error(f"AITER basic model failed: {e}")
        return False
    
    print_success("All AITER basic tests passed")
    return True

if __name__ == "__main__":
    success = test_aiter_basic()
    sys.exit(0 if success else 1)
