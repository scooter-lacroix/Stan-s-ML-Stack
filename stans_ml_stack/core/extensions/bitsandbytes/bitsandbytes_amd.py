#!/usr/bin/env python3
# =============================================================================
# BITSANDBYTES for AMD GPUs
# =============================================================================
# This module provides utilities for using BITSANDBYTES with AMD GPUs.
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
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bitsandbytes_amd")

def check_bitsandbytes_installation():
    """Check if BITSANDBYTES is installed.
    
    Returns:
        bool: True if BITSANDBYTES is installed, False otherwise
    """
    try:
        import bitsandbytes as bnb
        logger.info(f"BITSANDBYTES is installed (version {bnb.__version__})")
        return True
    except ImportError:
        logger.error("BITSANDBYTES is not installed")
        logger.info("Please install BITSANDBYTES first")
        return False

def install_bitsandbytes_for_amd():
    """Install BITSANDBYTES for AMD GPUs.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        import subprocess
        
        # Clone repository
        logger.info("Cloning BITSANDBYTES repository")
        subprocess.run(
            ["git", "clone", "https://github.com/TimDettmers/bitsandbytes.git"],
            check=True
        )
        
        # Change directory
        os.chdir("bitsandbytes")
        
        # Set environment variables for AMD
        os.environ["ROCM_HOME"] = "/opt/rocm"
        
        # Install
        logger.info("Installing BITSANDBYTES for AMD GPUs")
        subprocess.run(
            ["pip", "install", "-e", "."],
            check=True
        )
        
        logger.info("BITSANDBYTES installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install BITSANDBYTES: {e}")
        return False

def create_quantized_linear_layer(input_size, output_size, bias=True, quantization="int8"):
    """Create a quantized linear layer.
    
    Args:
        input_size: Input size
        output_size: Output size
        bias: Whether to include bias
        quantization: Quantization type (int8, int4, nf4)
    
    Returns:
        torch.nn.Module: Quantized linear layer
    """
    try:
        import bitsandbytes as bnb
        
        if quantization == "int8":
            return bnb.nn.Linear8bitLt(input_size, output_size, bias=bias)
        elif quantization == "int4":
            return bnb.nn.Linear4bit(input_size, output_size, bias=bias)
        elif quantization == "nf4":
            return bnb.nn.LinearNF4(input_size, output_size, bias=bias)
        else:
            logger.error(f"Unsupported quantization: {quantization}")
            return torch.nn.Linear(input_size, output_size, bias=bias)
    except ImportError:
        logger.error("BITSANDBYTES is not installed")
        return torch.nn.Linear(input_size, output_size, bias=bias)
    except Exception as e:
        logger.error(f"Failed to create quantized linear layer: {e}")
        return torch.nn.Linear(input_size, output_size, bias=bias)

def quantize_model(model, quantization="int8"):
    """Quantize model.
    
    Args:
        model: PyTorch model
        quantization: Quantization type (int8, int4, nf4)
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import bitsandbytes as bnb
        
        # Get linear layer class based on quantization
        if quantization == "int8":
            linear_class = bnb.nn.Linear8bitLt
        elif quantization == "int4":
            linear_class = bnb.nn.Linear4bit
        elif quantization == "nf4":
            linear_class = bnb.nn.LinearNF4
        else:
            logger.error(f"Unsupported quantization: {quantization}")
            return model
        
        # Replace linear layers with quantized linear layers
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                setattr(
                    model,
                    name,
                    linear_class(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    )
                )
            else:
                quantize_model(module, quantization)
        
        return model
    except ImportError:
        logger.error("BITSANDBYTES is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        return model

def optimize_memory_usage():
    """Optimize memory usage.
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import bitsandbytes as bnb
        
        # Enable memory efficient optimizers
        bnb.optim.GlobalOptimManager.get_instance().register_parameters_in_optimizer()
        
        logger.info("Memory usage optimized")
        return True
    except ImportError:
        logger.error("BITSANDBYTES is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to optimize memory usage: {e}")
        return False

def create_8bit_optimizer(model, lr=1e-3, optimizer_type="adam"):
    """Create 8-bit optimizer.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        optimizer_type: Optimizer type (adam, adamw, lion)
    
    Returns:
        torch.optim.Optimizer: 8-bit optimizer
    """
    try:
        import bitsandbytes as bnb
        
        if optimizer_type.lower() == "adam":
            return bnb.optim.Adam8bit(model.parameters(), lr=lr)
        elif optimizer_type.lower() == "adamw":
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        elif optimizer_type.lower() == "lion":
            return bnb.optim.Lion8bit(model.parameters(), lr=lr)
        else:
            logger.error(f"Unsupported optimizer type: {optimizer_type}")
            return torch.optim.Adam(model.parameters(), lr=lr)
    except ImportError:
        logger.error("BITSANDBYTES is not installed")
        return torch.optim.Adam(model.parameters(), lr=lr)
    except Exception as e:
        logger.error(f"Failed to create 8-bit optimizer: {e}")
        return torch.optim.Adam(model.parameters(), lr=lr)

def benchmark_quantized_model(model, input_shape, batch_size=1, num_iterations=100):
    """Benchmark quantized model.
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        batch_size: Batch size
        num_iterations: Number of iterations
    
    Returns:
        dict: Benchmark results
    """
    try:
        import time
        
        # Move model to GPU
        model = model.cuda()
        model.eval()
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, *input_shape, device="cuda")
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate results
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        logger.info(f"Average inference time: {avg_time * 1000:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} samples/s")
        
        return {
            "avg_time": avg_time,
            "throughput": throughput,
            "total_time": total_time,
            "num_iterations": num_iterations
        }
    except Exception as e:
        logger.error(f"Failed to benchmark quantized model: {e}")
        return None

if __name__ == "__main__":
    # Check BITSANDBYTES installation
    check_bitsandbytes_installation()
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024)
    )
    
    # Quantize model
    quantized_model = quantize_model(model, quantization="int8")
    
    # Create 8-bit optimizer
    optimizer = create_8bit_optimizer(quantized_model, lr=1e-3, optimizer_type="adam")
    
    # Optimize memory usage
    optimize_memory_usage()
    
    # Benchmark quantized model
    if torch.cuda.is_available():
        benchmark_quantized_model(quantized_model, (1024,), batch_size=32, num_iterations=100)
