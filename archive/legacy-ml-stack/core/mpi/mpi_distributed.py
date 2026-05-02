#!/usr/bin/env python3
# =============================================================================
# MPI Distributed Training
# =============================================================================
# This module provides utilities for distributed training with MPI and PyTorch.
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
from typing import Dict, List, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mpi_distributed")

class MPIDistributedDataParallel:
    """Distributed data parallel model using MPI."""
    
    def __init__(self, model, device_ids=None, output_device=None, broadcast_buffers=True):
        """Initialize MPIDistributedDataParallel.
        
        Args:
            model: PyTorch model
            device_ids: List of device IDs
            output_device: Output device ID
            broadcast_buffers: Whether to broadcast buffers
        """
        self.model = model
        self.device_ids = device_ids
        self.output_device = output_device
        self.broadcast_buffers = broadcast_buffers
        
        # Check MPI availability
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.mpi_available = True
            logger.info(f"MPI initialized with rank {self.rank} and size {self.size}")
        except ImportError:
            logger.error("MPI is not available")
            logger.info("Please install MPI first")
            self.mpi_available = False
            self.rank = 0
            self.size = 1
        
        # Assign device
        if device_ids is None:
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
            else:
                device_ids = []
        
        if output_device is None and len(device_ids) > 0:
            output_device = device_ids[0]
        
        self.device_ids = device_ids
        self.output_device = output_device
        
        # Assign GPU to process
        if torch.cuda.is_available() and len(device_ids) > 0:
            gpu_id = self.rank % len(device_ids)
            torch.cuda.set_device(device_ids[gpu_id])
            logger.info(f"Process {self.rank} assigned to GPU {device_ids[gpu_id]}")
        
        # Broadcast model parameters
        self._broadcast_model_parameters()
    
    def _broadcast_model_parameters(self):
        """Broadcast model parameters from rank 0 to all processes."""
        if not self.mpi_available or self.size == 1:
            return
        
        # Get model parameters
        params = [p.data for p in self.model.parameters()]
        
        # Broadcast parameters
        for param in params:
            if self.rank == 0:
                param_np = param.cpu().numpy()
                shape = param_np.shape
                dtype = param_np.dtype
            else:
                param_np = None
                shape = None
                dtype = None
            
            # Broadcast shape and dtype
            shape = self.comm.bcast(shape, root=0)
            dtype = self.comm.bcast(dtype, root=0)
            
            # Create output array
            if self.rank != 0:
                param_np = np.empty(shape, dtype=dtype)
            
            # Perform broadcast
            self.comm.Bcast(param_np, root=0)
            
            # Update parameter
            if self.rank != 0:
                param.copy_(torch.from_numpy(param_np))
        
        # Broadcast buffers
        if self.broadcast_buffers:
            buffers = [b.data for b in self.model.buffers()]
            
            for buffer in buffers:
                if self.rank == 0:
                    buffer_np = buffer.cpu().numpy()
                    shape = buffer_np.shape
                    dtype = buffer_np.dtype
                else:
                    buffer_np = None
                    shape = None
                    dtype = None
                
                # Broadcast shape and dtype
                shape = self.comm.bcast(shape, root=0)
                dtype = self.comm.bcast(dtype, root=0)
                
                # Create output array
                if self.rank != 0:
                    buffer_np = np.empty(shape, dtype=dtype)
                
                # Perform broadcast
                self.comm.Bcast(buffer_np, root=0)
                
                # Update buffer
                if self.rank != 0:
                    buffer.copy_(torch.from_numpy(buffer_np))
        
        logger.info("Model parameters broadcasted")
    
    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Call model."""
        return self.forward(*args, **kwargs)
    
    def zero_grad(self):
        """Zero gradients."""
        self.model.zero_grad()
    
    def allreduce_gradients(self):
        """Allreduce gradients."""
        if not self.mpi_available or self.size == 1:
            return
        
        # Get model parameters
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        
        # Allreduce gradients
        for param in params:
            grad_np = param.grad.data.cpu().numpy()
            result_np = np.empty_like(grad_np)
            
            # Perform allreduce
            self.comm.Allreduce(grad_np, result_np, op=self.comm.SUM)
            
            # Update gradient
            param.grad.data.copy_(torch.from_numpy(result_np) / self.size)
        
        logger.debug("Gradients allreduced")

class MPIDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler using MPI."""
    
    def __init__(self, dataset, shuffle=True, seed=0):
        """Initialize MPIDistributedSampler.
        
        Args:
            dataset: PyTorch dataset
            shuffle: Whether to shuffle the dataset
            seed: Random seed
        """
        # Check MPI availability
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.mpi_available = True
            logger.info(f"MPI initialized with rank {self.rank} and size {self.size}")
        except ImportError:
            logger.error("MPI is not available")
            logger.info("Please install MPI first")
            self.mpi_available = False
            self.rank = 0
            self.size = 1
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset) // self.size
        self.total_size = self.num_samples * self.size
        
        logger.info(f"Distributed sampler initialized with {self.num_samples} samples per process")
    
    def __iter__(self):
        """Return iterator."""
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Subsample
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        
        # Divide indices into chunks for each process
        indices = indices[self.rank:self.total_size:self.size]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        """Return length."""
        return self.num_samples
    
    def set_epoch(self, epoch):
        """Set epoch."""
        self.epoch = epoch

def mpi_distributed_wrapper(func):
    """Wrapper for distributed training with MPI."""
    def wrapper(*args, **kwargs):
        # Check MPI availability
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            mpi_available = True
        except ImportError:
            logger.error("MPI is not available")
            logger.info("Please install MPI first")
            mpi_available = False
            rank = 0
            size = 1
        
        # Call function
        result = func(*args, **kwargs)
        
        # Synchronize processes
        if mpi_available:
            comm.Barrier()
        
        return result
    
    return wrapper

def init_mpi_distributed():
    """Initialize distributed training with MPI."""
    # Check MPI availability
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        mpi_available = True
        logger.info(f"MPI initialized with rank {rank} and size {size}")
    except ImportError:
        logger.error("MPI is not available")
        logger.info("Please install MPI first")
        mpi_available = False
        rank = 0
        size = 1
    
    # Initialize PyTorch distributed
    if mpi_available and size > 1:
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(size)
        
        # Initialize process group
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=size)
        logger.info(f"PyTorch distributed initialized with rank {rank} and world size {size}")
    
    return rank, size

def cleanup_mpi_distributed():
    """Clean up distributed training with MPI."""
    # Check if PyTorch distributed is initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("PyTorch distributed cleaned up")

if __name__ == "__main__":
    # Initialize distributed training with MPI
    rank, size = init_mpi_distributed()
    
    # Create model
    model = torch.nn.Linear(10, 10)
    
    # Wrap model with MPIDistributedDataParallel
    model = MPIDistributedDataParallel(model)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    
    # Create sampler
    sampler = MPIDistributedSampler(dataset)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=sampler)
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        
        for data, target in dataloader:
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Allreduce gradients
            model.allreduce_gradients()
            
            # Update parameters
            with torch.no_grad():
                for param in model.parameters():
                    param.data.add_(param.grad.data, alpha=-0.01)
    
    # Clean up
    cleanup_mpi_distributed()
