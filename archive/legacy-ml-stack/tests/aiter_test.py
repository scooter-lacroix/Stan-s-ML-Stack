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

def test_aiter():
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
        print_info("Make sure AITER is installed:")
        print_info("  - Check if aiter is in the Python path")
        print_info("  - Try reinstalling AITER")
        return False
    
    # Print AITER version
    try:
        print_info(f"AITER version: {aiter.__version__}")
    except AttributeError:
        print_warning("Could not determine AITER version")
    
    # Create a simple AITER model
    try:
        from aiter import AITERModel
        
        # Create a simple model
        model = AITERModel(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            device="cuda"
        )
        print_success("Created AITER model successfully")
        
        # Test forward pass
        batch_size = 4
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 10, device="cuda")
        output = model(x)
        
        print_info(f"Input shape: {x.shape}")
        print_info(f"Output shape: {output.shape}")
        print_success("Forward pass successful")
        
        # Test backward pass
        target = torch.randn(batch_size, seq_len, 5, device="cuda")
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        print_info(f"Loss: {loss.item()}")
        print_success("Backward pass successful")
        
    except Exception as e:
        print_error(f"AITER test failed: {e}")
        return False
    
    print_success("All AITER tests passed")
    return True

if __name__ == "__main__":
    success = test_aiter()
    sys.exit(0 if success else 1)
