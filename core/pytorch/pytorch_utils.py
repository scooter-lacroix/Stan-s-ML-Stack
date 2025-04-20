#!/usr/bin/env python3
# =============================================================================
# PyTorch Utilities
# =============================================================================
# This module provides utilities for working with PyTorch on AMD GPUs.
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
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pytorch_utils")

def check_pytorch_rocm():
    """Check if PyTorch with ROCm support is available."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info("CUDA is available through ROCm")
            
            # Get number of GPUs
            device_count = torch.cuda.device_count()
            logger.info(f"Number of GPUs: {device_count}")
            
            # Print GPU information
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {device_name}")
                
                if "AMD" in device_name or "Radeon" in device_name:
                    logger.info(f"GPU {i} is an AMD GPU")
                else:
                    logger.warning(f"GPU {i} is not an AMD GPU")
            
            return True
        else:
            logger.error("CUDA is not available through ROCm")
            return False
    except ImportError:
        logger.error("PyTorch is not installed")
        return False

def set_gpu_environment_variables():
    """Set GPU environment variables for optimal performance."""
    # Set environment variables
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    os.environ["PYTORCH_ROCM_DEVICE"] = os.environ.get("PYTORCH_ROCM_DEVICE", "0,1,2,3")
    
    # Set performance tuning variables
    os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for better performance
    os.environ["GPU_MAX_HEAP_SIZE"] = "100"  # Increase heap size (in %)
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"  # Allow allocating 100% of available memory
    os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"  # Allow single allocations up to 100%
    
    logger.info("GPU environment variables set")
    
    return True

def optimize_pytorch_for_amd():
    """Optimize PyTorch for AMD GPUs."""
    # Set memory allocation strategy
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
        
        # Set max split size
        device_name = torch.cuda.get_device_name(0)
        if "7900 XTX" in device_name:
            torch.cuda.max_split_size_mb = 512  # Optimal for RX 7900 XTX
        elif "7900 XT" in device_name:
            torch.cuda.max_split_size_mb = 384  # Optimal for RX 7900 XT
        elif "7800 XT" in device_name:
            torch.cuda.max_split_size_mb = 256  # Optimal for RX 7800 XT
        elif "7700 XT" in device_name:
            torch.cuda.max_split_size_mb = 128  # Optimal for RX 7700 XT
        else:
            torch.cuda.max_split_size_mb = 256  # Default value
        
        logger.info(f"PyTorch memory settings optimized for {device_name}")
        
        # Enable benchmark mode for optimal performance with fixed input sizes
        torch.backends.cudnn.benchmark = True
        logger.info("PyTorch benchmark mode enabled")
    
    return True

def get_optimal_batch_size(model, input_shape, dtype=torch.float32, min_batch=1, max_batch=128):
    """Get optimal batch size for model and GPU.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        dtype: Data type of input tensor
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
    
    Returns:
        int: Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return min_batch
    
    # Move model to GPU
    model = model.cuda()
    model.eval()
    
    # Try different batch sizes
    batch_sizes = []
    for batch_size in range(min_batch, max_batch + 1, min_batch):
        try:
            # Create input tensor
            input_tensor = torch.randn(batch_size, *input_shape, dtype=dtype, device="cuda")
            
            # Run forward pass
            with torch.no_grad():
                _ = model(input_tensor)
            
            # If successful, add to list
            batch_sizes.append(batch_size)
            logger.info(f"Batch size {batch_size} works")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.info(f"Batch size {batch_size} is too large")
                break
            else:
                raise e
    
    # Return largest successful batch size
    if batch_sizes:
        optimal_batch_size = batch_sizes[-1]
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    else:
        logger.warning("No batch size works")
        return min_batch

def benchmark_model(model, input_shape, batch_size=1, dtype=torch.float32, num_iterations=100):
    """Benchmark model on GPU.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        batch_size: Batch size
        dtype: Data type of input tensor
        num_iterations: Number of iterations
    
    Returns:
        dict: Dictionary of benchmark results
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return None
    
    # Move model to GPU
    model = model.cuda()
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, *input_shape, dtype=dtype, device="cuda")
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_iterations):
        start_event.record()
        with torch.no_grad():
            _ = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    # Calculate statistics
    avg_time = sum(times) / num_iterations
    min_time = min(times)
    max_time = max(times)
    std_time = np.std(times)
    
    logger.info(f"Average inference time: {avg_time:.2f} ms")
    logger.info(f"Min inference time: {min_time:.2f} ms")
    logger.info(f"Max inference time: {max_time:.2f} ms")
    logger.info(f"Std inference time: {std_time:.2f} ms")
    
    # Calculate throughput
    throughput = batch_size * 1000 / avg_time  # samples/second
    logger.info(f"Throughput: {throughput:.2f} samples/second")
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "throughput": throughput,
        "times": times
    }

def profile_model(model, input_shape, batch_size=1, dtype=torch.float32):
    """Profile model on GPU.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        batch_size: Batch size
        dtype: Data type of input tensor
    
    Returns:
        None
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return None
    
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
    except ImportError:
        logger.error("PyTorch profiler is not available")
        return None
    
    # Move model to GPU
    model = model.cuda()
    model.eval()
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, *input_shape, dtype=dtype, device="cuda")
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Profile model
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(input_tensor)
    
    # Print results
    logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return prof

def optimize_model_for_inference(model):
    """Optimize model for inference.
    
    Args:
        model: PyTorch model
    
    Returns:
        torch.nn.Module: Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Fuse batch normalization layers
    model = torch.quantization.fuse_modules(model, [["conv", "bn", "relu"]])
    
    # Convert to TorchScript
    try:
        scripted_model = torch.jit.script(model)
        logger.info("Model converted to TorchScript")
        return scripted_model
    except Exception as e:
        logger.error(f"Failed to convert model to TorchScript: {e}")
        return model

def mixed_precision_training(model, optimizer, loss_fn, train_loader, num_epochs=1):
    """Train model with mixed precision.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        train_loader: Training data loader
        num_epochs: Number of epochs
    
    Returns:
        torch.nn.Module: Trained model
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return model
    
    # Move model to GPU
    model = model.cuda()
    
    # Create gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            data, target = data.cuda(), target.cuda()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = loss_fn(output, target)
            
            # Backward pass with scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Print progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
    
    return model

def get_gpu_memory_usage():
    """Get GPU memory usage.
    
    Returns:
        dict: Dictionary of GPU memory usage
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return None
    
    # Get number of GPUs
    device_count = torch.cuda.device_count()
    
    # Get memory usage for each GPU
    memory_usage = {}
    for i in range(device_count):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        max_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
        max_memory_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)  # GB
        
        memory_usage[i] = {
            "allocated": memory_allocated,
            "reserved": memory_reserved,
            "max_allocated": max_memory_allocated,
            "max_reserved": max_memory_reserved
        }
        
        logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        logger.info(f"  Memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"  Memory reserved: {memory_reserved:.2f} GB")
        logger.info(f"  Max memory allocated: {max_memory_allocated:.2f} GB")
        logger.info(f"  Max memory reserved: {max_memory_reserved:.2f} GB")
    
    return memory_usage

def clear_gpu_memory():
    """Clear GPU memory."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return False
    
    # Empty cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    logger.info("GPU memory cleared")
    
    return True

if __name__ == "__main__":
    # Check PyTorch with ROCm
    check_pytorch_rocm()
    
    # Set GPU environment variables
    set_gpu_environment_variables()
    
    # Optimize PyTorch for AMD
    optimize_pytorch_for_amd()
    
    # Get GPU memory usage
    get_gpu_memory_usage()
    
    # Clear GPU memory
    clear_gpu_memory()
