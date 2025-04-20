#!/usr/bin/env python3
# =============================================================================
# MPI Helper Functions
# =============================================================================
# This module provides helper functions for MPI operations with PyTorch.
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
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mpi_helper")

def check_mpi_availability():
    """Check if MPI is available."""
    try:
        from mpi4py import MPI
        logger.info("MPI is available")
        return True
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return False

def get_mpi_comm():
    """Get MPI communicator."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        logger.info(f"MPI communicator created with size {comm.Get_size()}")
        return comm
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return None

def get_mpi_rank_and_size():
    """Get MPI rank and size."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        logger.info(f"MPI rank: {rank}, size: {size}")
        return rank, size
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return 0, 1

def assign_gpu_to_process():
    """Assign GPU to MPI process."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False
        
        # Get number of GPUs
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.error("No GPUs detected")
            return False
        
        # Assign GPU to process
        gpu_id = rank % device_count
        torch.cuda.set_device(gpu_id)
        
        logger.info(f"Process {rank} assigned to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        return True
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return False

def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def numpy_to_tensor(array, device=None):
    """Convert NumPy array to PyTorch tensor."""
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    return array

def mpi_allreduce(tensor, op="sum"):
    """Perform MPI allreduce operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        # Convert tensor to NumPy array
        tensor_np = tensor_to_numpy(tensor)
        
        # Create output array
        result_np = np.empty_like(tensor_np)
        
        # Determine operation
        if op.lower() == "sum":
            mpi_op = MPI.SUM
        elif op.lower() == "prod":
            mpi_op = MPI.PROD
        elif op.lower() == "min":
            mpi_op = MPI.MIN
        elif op.lower() == "max":
            mpi_op = MPI.MAX
        else:
            logger.error(f"Unsupported operation: {op}")
            return tensor
        
        # Perform allreduce
        comm.Allreduce(tensor_np, result_np, op=mpi_op)
        
        # Convert back to PyTorch tensor
        result = numpy_to_tensor(result_np, device=tensor.device if isinstance(tensor, torch.Tensor) else None)
        
        return result
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return tensor

def mpi_allgather(tensor):
    """Perform MPI allgather operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        
        # Convert tensor to NumPy array
        tensor_np = tensor_to_numpy(tensor)
        
        # Create output array
        result_np = [np.empty_like(tensor_np) for _ in range(size)]
        
        # Perform allgather
        comm.Allgather(tensor_np, result_np)
        
        # Convert back to PyTorch tensor
        result = [numpy_to_tensor(arr, device=tensor.device if isinstance(tensor, torch.Tensor) else None) for arr in result_np]
        
        return result
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return [tensor]

def mpi_scatter(tensor, root=0):
    """Perform MPI scatter operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Convert tensor to NumPy array
        if rank == root:
            if not isinstance(tensor, list):
                logger.error("Tensor must be a list for scatter operation")
                return None
            tensor_np = [tensor_to_numpy(t) for t in tensor]
        else:
            tensor_np = None
        
        # Create output array
        if rank == root:
            result_np = np.empty_like(tensor_np[0])
        else:
            # Get shape from root process
            shape = comm.bcast(tensor_np[0].shape if rank == root else None, root=root)
            dtype = comm.bcast(tensor_np[0].dtype if rank == root else None, root=root)
            result_np = np.empty(shape, dtype=dtype)
        
        # Perform scatter
        comm.Scatter(tensor_np, result_np, root=root)
        
        # Convert back to PyTorch tensor
        device = tensor[0].device if rank == root and isinstance(tensor[0], torch.Tensor) else None
        result = numpy_to_tensor(result_np, device=device)
        
        return result
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return None

def mpi_gather(tensor, root=0):
    """Perform MPI gather operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Convert tensor to NumPy array
        tensor_np = tensor_to_numpy(tensor)
        
        # Create output array
        if rank == root:
            result_np = [np.empty_like(tensor_np) for _ in range(size)]
        else:
            result_np = None
        
        # Perform gather
        comm.Gather(tensor_np, result_np, root=root)
        
        # Convert back to PyTorch tensor
        if rank == root:
            device = tensor.device if isinstance(tensor, torch.Tensor) else None
            result = [numpy_to_tensor(arr, device=device) for arr in result_np]
            return result
        else:
            return None
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return None

def mpi_bcast(tensor, root=0):
    """Perform MPI broadcast operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Convert tensor to NumPy array
        if rank == root:
            tensor_np = tensor_to_numpy(tensor)
            shape = tensor_np.shape
            dtype = tensor_np.dtype
        else:
            tensor_np = None
            shape = None
            dtype = None
        
        # Broadcast shape and dtype
        shape = comm.bcast(shape, root=root)
        dtype = comm.bcast(dtype, root=root)
        
        # Create output array
        if rank != root:
            tensor_np = np.empty(shape, dtype=dtype)
        
        # Perform broadcast
        comm.Bcast(tensor_np, root=root)
        
        # Convert back to PyTorch tensor
        device = tensor.device if rank == root and isinstance(tensor, torch.Tensor) else None
        result = numpy_to_tensor(tensor_np, device=device)
        
        return result
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return tensor

def mpi_reduce(tensor, op="sum", root=0):
    """Perform MPI reduce operation on PyTorch tensor."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Convert tensor to NumPy array
        tensor_np = tensor_to_numpy(tensor)
        
        # Create output array
        if rank == root:
            result_np = np.empty_like(tensor_np)
        else:
            result_np = None
        
        # Determine operation
        if op.lower() == "sum":
            mpi_op = MPI.SUM
        elif op.lower() == "prod":
            mpi_op = MPI.PROD
        elif op.lower() == "min":
            mpi_op = MPI.MIN
        elif op.lower() == "max":
            mpi_op = MPI.MAX
        else:
            logger.error(f"Unsupported operation: {op}")
            return tensor
        
        # Perform reduce
        comm.Reduce(tensor_np, result_np, op=mpi_op, root=root)
        
        # Convert back to PyTorch tensor
        if rank == root:
            device = tensor.device if isinstance(tensor, torch.Tensor) else None
            result = numpy_to_tensor(result_np, device=device)
            return result
        else:
            return None
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return tensor

def mpi_barrier():
    """Perform MPI barrier operation."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        comm.Barrier()
        return True
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        return False

if __name__ == "__main__":
    # Check MPI availability
    if check_mpi_availability():
        # Get MPI rank and size
        rank, size = get_mpi_rank_and_size()
        
        # Assign GPU to process
        assign_gpu_to_process()
        
        # Test MPI operations
        if torch.cuda.is_available():
            # Create tensor on GPU
            tensor = torch.tensor([rank + 1.0] * 3, device="cuda")
            
            # Test allreduce
            result = mpi_allreduce(tensor)
            
            # Print result
            print(f"Process {rank}: Allreduce result = {result}")
            
            # Test barrier
            mpi_barrier()
        else:
            print("CUDA is not available")
