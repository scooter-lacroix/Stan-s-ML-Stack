#!/usr/bin/env python3
# =============================================================================
# PyTorch Profiler Utilities
# =============================================================================
# This module provides utilities for profiling PyTorch models on AMD GPUs.
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
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("profiler_utils")

def check_profiler_availability():
    """Check if PyTorch profiler is available.
    
    Returns:
        bool: True if profiler is available, False otherwise
    """
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        logger.info("PyTorch profiler is available")
        return True
    except ImportError:
        logger.error("PyTorch profiler is not available")
        logger.info("Please upgrade PyTorch to a version that includes the profiler")
        return False

def profile_model(model, input_shape, batch_size=1, num_warmup=10, num_active=10, 
                  activities=None, record_shapes=True, profile_memory=True, with_stack=True):
    """Profile PyTorch model.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        batch_size: Batch size
        num_warmup: Number of warmup iterations
        num_active: Number of active iterations
        activities: List of activities to profile
        record_shapes: Whether to record tensor shapes
        profile_memory: Whether to profile memory
        with_stack: Whether to record stack traces
    
    Returns:
        torch.profiler.profile: Profiler object
    """
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        from torch.profiler import schedule
        
        # Set default activities
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, *input_shape, device=device)
        
        # Create profiler schedule
        prof_schedule = schedule(
            wait=0,
            warmup=num_warmup,
            active=num_active,
            repeat=1
        )
        
        # Profile model
        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        ) as prof:
            for _ in range(num_warmup + num_active):
                with record_function("model_inference"):
                    with torch.no_grad():
                        output = model(input_tensor)
                prof.step()
        
        return prof
    except Exception as e:
        logger.error(f"Failed to profile model: {e}")
        return None

def print_profiler_summary(prof, sort_by="cuda_time_total", row_limit=10):
    """Print profiler summary.
    
    Args:
        prof: Profiler object
        sort_by: Column to sort by
        row_limit: Maximum number of rows to print
    
    Returns:
        str: Profiler summary
    """
    try:
        # Print summary
        summary = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        logger.info(f"Profiler summary (sorted by {sort_by}):\n{summary}")
        
        return summary
    except Exception as e:
        logger.error(f"Failed to print profiler summary: {e}")
        return None

def export_chrome_trace(prof, path="chrome_trace.json"):
    """Export Chrome trace.
    
    Args:
        prof: Profiler object
        path: Path to save trace
    
    Returns:
        bool: True if export is successful, False otherwise
    """
    try:
        # Export trace
        logger.info(f"Exporting Chrome trace to {path}")
        prof.export_chrome_trace(path)
        
        return True
    except Exception as e:
        logger.error(f"Failed to export Chrome trace: {e}")
        return False

def export_stacks(prof, path="profiler_stacks.txt"):
    """Export stack traces.
    
    Args:
        prof: Profiler object
        path: Path to save stack traces
    
    Returns:
        bool: True if export is successful, False otherwise
    """
    try:
        # Export stacks
        logger.info(f"Exporting stack traces to {path}")
        with open(path, "w") as f:
            f.write(prof.key_averages(group_by_stack=True).table(sort_by="self_cuda_time_total", row_limit=50))
        
        return True
    except Exception as e:
        logger.error(f"Failed to export stack traces: {e}")
        return False

def analyze_memory_usage(prof):
    """Analyze memory usage.
    
    Args:
        prof: Profiler object
    
    Returns:
        dict: Memory usage statistics
    """
    try:
        # Get memory stats
        memory_stats = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
        logger.info(f"Memory usage statistics:\n{memory_stats}")
        
        # Calculate total memory usage
        total_memory = 0
        for event in prof.key_averages():
            if event.self_cuda_memory_usage > 0:
                total_memory += event.self_cuda_memory_usage
        
        logger.info(f"Total memory usage: {total_memory / (1024 * 1024):.2f} MB")
        
        return {
            "total_memory": total_memory,
            "memory_stats": memory_stats
        }
    except Exception as e:
        logger.error(f"Failed to analyze memory usage: {e}")
        return None

def analyze_operator_time(prof):
    """Analyze operator time.
    
    Args:
        prof: Profiler object
    
    Returns:
        dict: Operator time statistics
    """
    try:
        # Get operator stats
        operator_stats = {}
        
        for event in prof.key_averages():
            if event.key not in operator_stats:
                operator_stats[event.key] = {
                    "count": 0,
                    "cuda_time": 0,
                    "cpu_time": 0
                }
            
            operator_stats[event.key]["count"] += 1
            operator_stats[event.key]["cuda_time"] += event.cuda_time_total
            operator_stats[event.key]["cpu_time"] += event.cpu_time_total
        
        # Sort operators by CUDA time
        sorted_operators = sorted(operator_stats.items(), key=lambda x: x[1]["cuda_time"], reverse=True)
        
        # Print top operators
        logger.info("Top operators by CUDA time:")
        for i, (key, stats) in enumerate(sorted_operators[:10]):
            logger.info(f"{i+1}. {key}: {stats['cuda_time'] / 1000:.2f} ms (count: {stats['count']})")
        
        return {
            "operator_stats": operator_stats,
            "sorted_operators": sorted_operators
        }
    except Exception as e:
        logger.error(f"Failed to analyze operator time: {e}")
        return None

def profile_model_with_different_batch_sizes(model, input_shape, batch_sizes=None, num_iterations=100):
    """Profile model with different batch sizes.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        batch_sizes: List of batch sizes
        num_iterations: Number of iterations
    
    Returns:
        dict: Profiling results
    """
    try:
        # Set default batch sizes
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Profile model with different batch sizes
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create input tensor
                input_tensor = torch.randn(batch_size, *input_shape, device=device)
                
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
                
                logger.info(f"Batch size {batch_size}:")
                logger.info(f"  Average inference time: {avg_time * 1000:.2f} ms")
                logger.info(f"  Throughput: {throughput:.2f} samples/s")
                
                results[batch_size] = {
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "total_time": total_time
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Batch size {batch_size} is too large (out of memory)")
                    break
                else:
                    raise e
        
        return results
    except Exception as e:
        logger.error(f"Failed to profile model with different batch sizes: {e}")
        return None

def profile_model_with_different_input_sizes(model, base_input_shape, scale_factors=None, batch_size=1, num_iterations=100):
    """Profile model with different input sizes.
    
    Args:
        model: PyTorch model
        base_input_shape: Base input shape (without batch dimension)
        scale_factors: List of scale factors
        batch_size: Batch size
        num_iterations: Number of iterations
    
    Returns:
        dict: Profiling results
    """
    try:
        # Set default scale factors
        if scale_factors is None:
            scale_factors = [0.25, 0.5, 1.0, 1.5, 2.0]
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Profile model with different input sizes
        results = {}
        
        for scale_factor in scale_factors:
            try:
                # Calculate scaled input shape
                scaled_input_shape = tuple(int(dim * scale_factor) for dim in base_input_shape)
                
                # Create input tensor
                input_tensor = torch.randn(batch_size, *scaled_input_shape, device=device)
                
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
                
                logger.info(f"Scale factor {scale_factor} (input shape: {scaled_input_shape}):")
                logger.info(f"  Average inference time: {avg_time * 1000:.2f} ms")
                
                results[scale_factor] = {
                    "input_shape": scaled_input_shape,
                    "avg_time": avg_time,
                    "total_time": total_time
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Scale factor {scale_factor} is too large (out of memory)")
                    break
                else:
                    raise e
        
        return results
    except Exception as e:
        logger.error(f"Failed to profile model with different input sizes: {e}")
        return None

def create_profiler_report(prof, output_dir="profiler_report"):
    """Create profiler report.
    
    Args:
        prof: Profiler object
        output_dir: Output directory
    
    Returns:
        bool: True if report is created, False otherwise
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export Chrome trace
        export_chrome_trace(prof, os.path.join(output_dir, "chrome_trace.json"))
        
        # Export stack traces
        export_stacks(prof, os.path.join(output_dir, "profiler_stacks.txt"))
        
        # Export summary
        summary = print_profiler_summary(prof)
        with open(os.path.join(output_dir, "profiler_summary.txt"), "w") as f:
            f.write(summary)
        
        # Analyze memory usage
        memory_stats = analyze_memory_usage(prof)
        with open(os.path.join(output_dir, "memory_stats.txt"), "w") as f:
            f.write(str(memory_stats))
        
        # Analyze operator time
        operator_stats = analyze_operator_time(prof)
        with open(os.path.join(output_dir, "operator_stats.txt"), "w") as f:
            f.write(str(operator_stats))
        
        logger.info(f"Profiler report created in {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create profiler report: {e}")
        return False

if __name__ == "__main__":
    # Check profiler availability
    check_profiler_availability()
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 56 * 56, 1000)
    )
    
    # Profile model
    if torch.cuda.is_available():
        prof = profile_model(model, (3, 224, 224), batch_size=1)
        
        # Print profiler summary
        print_profiler_summary(prof)
        
        # Create profiler report
        create_profiler_report(prof)
        
        # Profile model with different batch sizes
        profile_model_with_different_batch_sizes(model, (3, 224, 224))
        
        # Profile model with different input sizes
        profile_model_with_different_input_sizes(model, (3, 224, 224))
