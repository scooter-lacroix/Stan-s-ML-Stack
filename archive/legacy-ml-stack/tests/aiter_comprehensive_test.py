#!/usr/bin/env python3
# =============================================================================
# AITER Comprehensive Test
# =============================================================================
# This script performs a comprehensive test of AITER functionality with AMD GPUs.
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

def test_aiter_comprehensive():
    """Perform a comprehensive test of AITER functionality."""
    print_header("AITER Comprehensive Test")
    
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
    
    # Print system information
    print_info("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test AITER components
    components = [
        "aiter.ops",
        "aiter.nn",
        "aiter.models",
        "aiter.utils",
        "aiter.data"
    ]
    
    for component in components:
        try:
            module = __import__(component, fromlist=["*"])
            print_success(f"Imported {component} successfully")
        except ImportError:
            print_warning(f"Failed to import {component}")
    
    # Test AITER operations
    try:
        from aiter.ops import (
            attention,
            normalization,
            activation,
            pooling
        )
        
        # Create input tensor
        x = torch.randn(4, 8, 16, device="cuda")
        
        # Test attention
        attn_output = attention(x, x, x)
        print_success("Attention operation successful")
        print_info(f"Attention output shape: {attn_output.shape}")
        
        # Test normalization
        norm_output = normalization(x)
        print_success("Normalization operation successful")
        print_info(f"Normalization output shape: {norm_output.shape}")
        
        # Test activation
        act_output = activation(x)
        print_success("Activation operation successful")
        print_info(f"Activation output shape: {act_output.shape}")
        
        # Test pooling
        pool_output = pooling(x)
        print_success("Pooling operation successful")
        print_info(f"Pooling output shape: {pool_output.shape}")
        
    except Exception as e:
        print_error(f"AITER operations test failed: {e}")
        return False
    
    # Test AITER models
    try:
        from aiter.models import (
            Transformer,
            CNN,
            RNN,
            MLP
        )
        
        # Create input tensor
        batch_size = 4
        seq_len = 16
        hidden_dim = 32
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
        
        # Test Transformer
        transformer = Transformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        ).to("cuda")
        
        transformer_output = transformer(x)
        print_success("Transformer model successful")
        print_info(f"Transformer output shape: {transformer_output.shape}")
        
        # Test CNN
        cnn = CNN(
            input_channels=hidden_dim,
            output_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        ).to("cuda")
        
        cnn_output = cnn(x.transpose(1, 2))
        print_success("CNN model successful")
        print_info(f"CNN output shape: {cnn_output.shape}")
        
        # Test RNN
        rnn = RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True
        ).to("cuda")
        
        rnn_output = rnn(x)
        print_success("RNN model successful")
        print_info(f"RNN output shape: {rnn_output.shape}")
        
        # Test MLP
        mlp = MLP(
            input_size=hidden_dim,
            hidden_sizes=[64, 32],
            output_size=hidden_dim,
            dropout=0.1
        ).to("cuda")
        
        mlp_output = mlp(x)
        print_success("MLP model successful")
        print_info(f"MLP output shape: {mlp_output.shape}")
        
    except Exception as e:
        print_error(f"AITER models test failed: {e}")
        return False
    
    # Test AITER training
    try:
        from aiter.models import SimpleModel
        from aiter.utils import train_model
        
        # Create a simple model
        model = SimpleModel(
            input_size=16,
            hidden_size=32,
            output_size=16
        ).to("cuda")
        
        # Create dummy data
        x = torch.randn(100, 16, device="cuda")
        y = torch.randn(100, 16, device="cuda")
        
        # Train model
        train_loss = train_model(
            model=model,
            x=x,
            y=y,
            batch_size=16,
            num_epochs=5,
            learning_rate=0.001
        )
        
        print_success("AITER training successful")
        print_info(f"Final training loss: {train_loss[-1]:.4f}")
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig("aiter_training_loss.png")
        print_info("Training loss plot saved to aiter_training_loss.png")
        
    except Exception as e:
        print_error(f"AITER training test failed: {e}")
        return False
    
    print_success("All AITER comprehensive tests passed")
    return True

if __name__ == "__main__":
    success = test_aiter_comprehensive()
    sys.exit(0 if success else 1)
