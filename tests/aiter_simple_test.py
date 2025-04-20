#!/usr/bin/env python3
# =============================================================================
# AITER Simple Test
# =============================================================================
# This script performs a simple test of AITER functionality with AMD GPUs.
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

def test_aiter_simple():
    """Perform a simple test of AITER functionality."""
    print_header("AITER Simple Test")
    
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
    
    # Create a simple tensor on GPU
    try:
        x = torch.randn(10, 10, device="cuda")
        print_success("Created tensor on GPU successfully")
    except Exception as e:
        print_error(f"Failed to create tensor on GPU: {e}")
        return False
    
    # Test AITER simple operations
    try:
        from aiter.ops import simple_op
        
        # Apply simple operation
        y = simple_op(x)
        print_success("Applied simple operation successfully")
        print_info(f"Input shape: {x.shape}")
        print_info(f"Output shape: {y.shape}")
        
    except Exception as e:
        print_error(f"AITER simple operation failed: {e}")
        return False
    
    # Test AITER with a small network
    try:
        from aiter.nn import SimpleNetwork
        
        # Create a simple network
        net = SimpleNetwork(input_dim=10, output_dim=5).to("cuda")
        print_success("Created simple network successfully")
        
        # Forward pass
        input_data = torch.randn(8, 10, device="cuda")
        output = net(input_data)
        print_success("Forward pass successful")
        print_info(f"Input shape: {input_data.shape}")
        print_info(f"Output shape: {output.shape}")
        
    except Exception as e:
        print_error(f"AITER simple network test failed: {e}")
        return False
    
    print_success("All AITER simple tests passed")
    return True

if __name__ == "__main__":
    success = test_aiter_simple()
    sys.exit(0 if success else 1)
