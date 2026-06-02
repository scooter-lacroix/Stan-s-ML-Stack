#!/usr/bin/env python3
# =============================================================================
# MPI Test
# =============================================================================
# This script tests MPI functionality on AMD GPUs.
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

def test_mpi():
    """Test MPI functionality on AMD GPUs."""
    print_header("MPI Test")
    
    # Check if MPI is installed
    try:
        from mpi4py import MPI
        print_success("MPI is installed")
    except ImportError:
        print_error("MPI is not installed")
        print_info("Please install MPI first")
        return False
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_info(f"Process rank: {rank}")
    print_info(f"World size: {size}")
    
    # Check if PyTorch is installed
    try:
        import torch
        print_success("PyTorch is installed")
    except ImportError:
        print_error("PyTorch is not installed")
        print_info("Please install PyTorch first")
        return False
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print_success("CUDA is available through ROCm")
    else:
        print_error("CUDA is not available through ROCm")
        print_info("Check if ROCm is installed and environment variables are set correctly:")
        print_info("  - HIP_VISIBLE_DEVICES")
        print_info("  - CUDA_VISIBLE_DEVICES")
        print_info("  - PYTORCH_ROCM_DEVICE")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print_success(f"Number of GPUs: {device_count}")
    else:
        print_error("No GPUs detected")
        return False
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        if "AMD" in device_name or "Radeon" in device_name:
            print_success(f"GPU {i}: {device_name}")
        else:
            print_warning(f"GPU {i}: {device_name} (not an AMD GPU)")
    
    # Assign GPU to process
    gpu_id = rank % device_count
    torch.cuda.set_device(gpu_id)
    print_info(f"Process {rank} using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Test basic MPI operations
    print_info("Testing basic MPI operations...")
    
    try:
        # Test broadcast
        if rank == 0:
            data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        else:
            data = np.empty(5, dtype=np.float32)
        
        comm.Bcast(data, root=0)
        
        if np.array_equal(data, np.array([1, 2, 3, 4, 5], dtype=np.float32)):
            print_success("Broadcast operation successful")
        else:
            print_error("Broadcast operation failed")
            return False
        
        # Test scatter
        if rank == 0:
            send_data = np.array(range(size * 3), dtype=np.float32).reshape(size, 3)
        else:
            send_data = None
        
        recv_data = np.empty(3, dtype=np.float32)
        comm.Scatter(send_data, recv_data, root=0)
        
        if np.array_equal(recv_data, np.array(range(rank * 3, (rank + 1) * 3), dtype=np.float32)):
            print_success("Scatter operation successful")
        else:
            print_error("Scatter operation failed")
            return False
        
        # Test gather
        send_data = np.array(range(rank * 3, (rank + 1) * 3), dtype=np.float32)
        
        if rank == 0:
            recv_data = np.empty(size * 3, dtype=np.float32).reshape(size, 3)
        else:
            recv_data = None
        
        comm.Gather(send_data, recv_data, root=0)
        
        if rank == 0:
            expected = np.array(range(size * 3), dtype=np.float32).reshape(size, 3)
            if np.array_equal(recv_data, expected):
                print_success("Gather operation successful")
            else:
                print_error("Gather operation failed")
                return False
        
        # Test reduce
        send_data = np.array([rank + 1] * 3, dtype=np.float32)
        
        if rank == 0:
            recv_data = np.empty(3, dtype=np.float32)
        else:
            recv_data = None
        
        comm.Reduce(send_data, recv_data, op=MPI.SUM, root=0)
        
        if rank == 0:
            expected = np.array([sum(range(1, size + 1))] * 3, dtype=np.float32)
            if np.array_equal(recv_data, expected):
                print_success("Reduce operation successful")
            else:
                print_error("Reduce operation failed")
                return False
        
        # Test allreduce
        send_data = np.array([rank + 1] * 3, dtype=np.float32)
        recv_data = np.empty(3, dtype=np.float32)
        
        comm.Allreduce(send_data, recv_data, op=MPI.SUM)
        
        expected = np.array([sum(range(1, size + 1))] * 3, dtype=np.float32)
        if np.array_equal(recv_data, expected):
            print_success("Allreduce operation successful")
        else:
            print_error("Allreduce operation failed")
            return False
        
    except Exception as e:
        print_error(f"Basic MPI operations failed: {e}")
        return False
    
    # Test MPI with PyTorch
    print_info("Testing MPI with PyTorch...")
    
    try:
        # Create tensor on GPU
        tensor = torch.tensor([rank + 1.0] * 3, device=f"cuda:{gpu_id}")
        
        # Convert to numpy for MPI
        tensor_np = tensor.cpu().numpy()
        
        # Allreduce
        result_np = np.empty_like(tensor_np)
        comm.Allreduce(tensor_np, result_np, op=MPI.SUM)
        
        # Convert back to PyTorch tensor
        result = torch.tensor(result_np, device=f"cuda:{gpu_id}")
        
        # Check result
        expected = torch.tensor([sum(range(1, size + 1))] * 3, device=f"cuda:{gpu_id}")
        if torch.allclose(result, expected):
            print_success("MPI with PyTorch successful")
        else:
            print_error("MPI with PyTorch failed")
            return False
        
    except Exception as e:
        print_error(f"MPI with PyTorch failed: {e}")
        return False
    
    # Test performance
    print_info("Testing performance...")
    
    try:
        # Create large tensor on GPU
        tensor_size = 1024 * 1024  # 1M elements
        tensor = torch.ones(tensor_size, device=f"cuda:{gpu_id}") * (rank + 1)
        
        # Convert to numpy for MPI
        tensor_np = tensor.cpu().numpy()
        
        # Warm-up
        for _ in range(5):
            result_np = np.empty_like(tensor_np)
            comm.Allreduce(tensor_np, result_np, op=MPI.SUM)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            result_np = np.empty_like(tensor_np)
            comm.Allreduce(tensor_np, result_np, op=MPI.SUM)
        end_time = time.time()
        
        # Calculate bandwidth
        bytes_transferred = tensor_size * 4 * 10  # 4 bytes per float32, 10 iterations
        duration = end_time - start_time
        bandwidth = bytes_transferred / duration / (1024 ** 2)  # MB/s
        
        print_info(f"Allreduce bandwidth: {bandwidth:.2f} MB/s")
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False
    
    print_success("All MPI tests passed")
    return True

if __name__ == "__main__":
    success = test_mpi()
    sys.exit(0 if success else 1)
