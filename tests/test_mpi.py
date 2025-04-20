#!/usr/bin/env python3
# =============================================================================
# MPI Test
# =============================================================================
# This script tests if MPI is working correctly with AMD GPUs.
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
import torch
from mpi4py import MPI

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
    """Test if MPI is working correctly with AMD GPUs."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if rank == 0:
        print_header("MPI Test")
        print_info(f"MPI World Size: {world_size}")
    
    # Get processor name
    processor_name = MPI.Get_processor_name()
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    
    # Get GPU information
    if cuda_available:
        device_count = torch.cuda.device_count()
        if rank < device_count:
            device = torch.device(f"cuda:{rank % device_count}")
            device_name = torch.cuda.get_device_name(rank % device_count)
        else:
            device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)
    else:
        device_count = 0
        device = torch.device("cpu")
        device_name = "CPU"
    
    # Print information from each rank
    for i in range(world_size):
        if rank == i:
            print(f"Rank {rank} on {processor_name}, Device: {device_name}")
            sys.stdout.flush()
        comm.Barrier()
    
    # Test basic MPI operations
    if rank == 0:
        print_info("Testing basic MPI operations")
    
    # Test broadcast
    if rank == 0:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        data = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    
    comm.Barrier()
    comm.Bcast(data.numpy(), root=0)
    
    if torch.all(data == torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])):
        if rank == 0:
            print_success("Broadcast test passed")
    else:
        if rank == 0:
            print_error("Broadcast test failed")
            return False
    
    # Test reduce
    local_value = torch.tensor([float(rank + 1)])
    global_sum = torch.tensor([0.0])
    
    comm.Barrier()
    comm.Reduce(local_value.numpy(), global_sum.numpy(), op=MPI.SUM, root=0)
    
    expected_sum = sum(range(1, world_size + 1))
    if rank == 0:
        if global_sum.item() == expected_sum:
            print_success("Reduce test passed")
        else:
            print_error(f"Reduce test failed: got {global_sum.item()}, expected {expected_sum}")
            return False
    
    # Test allreduce
    local_value = torch.tensor([float(rank + 1)])
    global_sum = torch.tensor([0.0])
    
    comm.Barrier()
    comm.Allreduce(local_value.numpy(), global_sum.numpy(), op=MPI.SUM)
    
    expected_sum = sum(range(1, world_size + 1))
    if global_sum.item() == expected_sum:
        if rank == 0:
            print_success("Allreduce test passed")
    else:
        if rank == 0:
            print_error(f"Allreduce test failed: got {global_sum.item()}, expected {expected_sum}")
            return False
    
    # Test GPU operations if available
    if cuda_available:
        if rank == 0:
            print_info("Testing GPU operations")
        
        # Move data to GPU
        try:
            gpu_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
            gpu_result = gpu_data * 2
            cpu_result = gpu_result.cpu()
            
            if torch.all(cpu_result == torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])):
                if rank == 0:
                    print_success("GPU operation test passed")
            else:
                if rank == 0:
                    print_error("GPU operation test failed")
                    return False
        except Exception as e:
            if rank == 0:
                print_error(f"GPU operation test failed: {e}")
                return False
    
    # Test MPI with GPU data
    if cuda_available:
        if rank == 0:
            print_info("Testing MPI with GPU data")
        
        # Create GPU data
        gpu_data = torch.tensor([float(rank + 1)], device=device)
        cpu_data = gpu_data.cpu()
        
        # Allreduce
        global_sum = torch.tensor([0.0])
        
        comm.Barrier()
        comm.Allreduce(cpu_data.numpy(), global_sum.numpy(), op=MPI.SUM)
        
        expected_sum = sum(range(1, world_size + 1))
        if global_sum.item() == expected_sum:
            if rank == 0:
                print_success("MPI with GPU data test passed")
        else:
            if rank == 0:
                print_error(f"MPI with GPU data test failed: got {global_sum.item()}, expected {expected_sum}")
                return False
    
    if rank == 0:
        print_success("All MPI tests passed")
    
    return True

if __name__ == "__main__":
    success = test_mpi()
    
    # Use MPI.Finalize() instead of sys.exit() to properly clean up MPI
    MPI.Finalize()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
