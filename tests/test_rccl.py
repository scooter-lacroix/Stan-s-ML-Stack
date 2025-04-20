#!/usr/bin/env python3
# =============================================================================
# RCCL Test
# =============================================================================
# This script tests RCCL functionality on AMD GPUs.
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

def test_rccl():
    """Test RCCL functionality on AMD GPUs."""
    print_header("RCCL Test")
    
    # Check if CUDA (ROCm) is available
    if not torch.cuda.is_available():
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    print_info(f"Number of GPUs: {device_count}")
    
    if device_count < 2:
        print_warning("At least 2 GPUs are required for RCCL tests")
        print_info("Some tests will be skipped")
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print_info(f"GPU {i}: {device_name}")
    
    # Check if RCCL is installed
    rccl_path = "/opt/rocm/lib/librccl.so"
    if os.path.exists(rccl_path):
        print_success(f"RCCL is installed at {rccl_path}")
    else:
        print_error("RCCL is not installed")
        print_info("Please install RCCL first")
        return False
    
    # Test basic collective operations
    print_info("Testing basic collective operations...")
    
    try:
        # Initialize process group
        if device_count >= 2:
            torch.distributed.init_process_group(backend="nccl", init_method="file:///tmp/rccl_test", rank=0, world_size=1)
            print_success("Initialized process group")
        else:
            print_warning("Skipping process group initialization (not enough GPUs)")
        
        # Test all-reduce
        if device_count >= 2:
            # Create tensors on different GPUs
            tensor1 = torch.ones(10, device="cuda:0")
            tensor2 = torch.ones(10, device="cuda:1") * 2
            
            # Create tensor list
            tensors = [tensor1, tensor2]
            
            # Perform all-reduce
            torch.distributed.all_reduce_coalesced(tensors, op=torch.distributed.ReduceOp.SUM)
            
            # Check results
            expected1 = torch.ones(10, device="cuda:0") * 3
            expected2 = torch.ones(10, device="cuda:1") * 3
            
            if torch.allclose(tensor1, expected1) and torch.allclose(tensor2, expected2):
                print_success("All-reduce operation successful")
            else:
                print_error("All-reduce operation failed")
                return False
        else:
            print_warning("Skipping all-reduce test (not enough GPUs)")
        
        # Test broadcast
        if device_count >= 2:
            # Create tensor on source GPU
            src_tensor = torch.randn(10, device="cuda:0")
            
            # Create tensor on destination GPU
            dst_tensor = torch.zeros(10, device="cuda:1")
            
            # Perform broadcast
            torch.distributed.broadcast(src_tensor, src=0)
            torch.distributed.broadcast(dst_tensor, src=0)
            
            # Check results
            if torch.allclose(src_tensor.cpu(), dst_tensor.cpu()):
                print_success("Broadcast operation successful")
            else:
                print_error("Broadcast operation failed")
                return False
        else:
            print_warning("Skipping broadcast test (not enough GPUs)")
        
        # Test all-gather
        if device_count >= 2:
            # Create tensors on different GPUs
            tensor1 = torch.ones(10, device="cuda:0")
            tensor2 = torch.ones(10, device="cuda:1") * 2
            
            # Create output tensors
            output1 = [torch.zeros(10, device="cuda:0") for _ in range(2)]
            output2 = [torch.zeros(10, device="cuda:1") for _ in range(2)]
            
            # Perform all-gather
            torch.distributed.all_gather(output1, tensor1)
            torch.distributed.all_gather(output2, tensor2)
            
            # Check results
            if torch.allclose(output1[0], tensor1) and torch.allclose(output1[1], tensor2) and \
               torch.allclose(output2[0], tensor1) and torch.allclose(output2[1], tensor2):
                print_success("All-gather operation successful")
            else:
                print_error("All-gather operation failed")
                return False
        else:
            print_warning("Skipping all-gather test (not enough GPUs)")
        
        # Clean up
        if device_count >= 2:
            torch.distributed.destroy_process_group()
            print_success("Destroyed process group")
        
    except Exception as e:
        print_error(f"Collective operations failed: {e}")
        return False
    
    # Test point-to-point communication
    print_info("Testing point-to-point communication...")
    
    try:
        if device_count >= 2:
            # Initialize process group
            torch.distributed.init_process_group(backend="nccl", init_method="file:///tmp/rccl_test", rank=0, world_size=1)
            
            # Create tensor on source GPU
            src_tensor = torch.randn(10, device="cuda:0")
            
            # Create tensor on destination GPU
            dst_tensor = torch.zeros(10, device="cuda:1")
            
            # Perform send/recv
            if torch.distributed.get_rank() == 0:
                torch.distributed.send(src_tensor, dst=0)
            else:
                torch.distributed.recv(dst_tensor, src=0)
            
            # Check results
            if torch.allclose(src_tensor.cpu(), dst_tensor.cpu()):
                print_success("Point-to-point communication successful")
            else:
                print_error("Point-to-point communication failed")
                return False
            
            # Clean up
            torch.distributed.destroy_process_group()
        else:
            print_warning("Skipping point-to-point communication test (not enough GPUs)")
        
    except Exception as e:
        print_error(f"Point-to-point communication failed: {e}")
        return False
    
    # Test performance
    print_info("Testing performance...")
    
    try:
        if device_count >= 2:
            # Initialize process group
            torch.distributed.init_process_group(backend="nccl", init_method="file:///tmp/rccl_test", rank=0, world_size=1)
            
            # Create large tensors on different GPUs
            tensor_size = 1024 * 1024 * 10  # 10 MB
            tensor1 = torch.randn(tensor_size, device="cuda:0")
            tensor2 = torch.randn(tensor_size, device="cuda:1")
            
            # Warm-up
            for _ in range(5):
                torch.distributed.all_reduce(tensor1)
                torch.distributed.all_reduce(tensor2)
            
            # Benchmark all-reduce
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(10):
                torch.distributed.all_reduce(tensor1)
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate bandwidth
            bytes_transferred = tensor_size * 4 * 10  # 4 bytes per float32, 10 iterations
            duration = end_time - start_time
            bandwidth = bytes_transferred / duration / (1024 ** 3)  # GB/s
            
            print_info(f"All-reduce bandwidth: {bandwidth:.2f} GB/s")
            
            # Clean up
            torch.distributed.destroy_process_group()
        else:
            print_warning("Skipping performance test (not enough GPUs)")
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False
    
    print_success("All RCCL tests passed")
    return True

if __name__ == "__main__":
    success = test_rccl()
    sys.exit(0 if success else 1)
