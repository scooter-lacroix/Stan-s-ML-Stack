#!/usr/bin/env python3
# =============================================================================
# PyTorch Distributed Training
# =============================================================================
# This module provides utilities for distributed training with PyTorch on AMD GPUs.
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
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pytorch_distributed")

def setup_distributed(rank, world_size, backend="nccl"):
    """Set up distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        backend: Backend for distributed training
    
    Returns:
        bool: True if setup is successful, False otherwise
    """
    try:
        # Set environment variables
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
        # Initialize process group
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        
        logger.info(f"Distributed setup complete (rank {rank}/{world_size})")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up distributed training: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")

def run_distributed(fn, world_size, args=(), kwargs=None):
    """Run function in distributed mode.
    
    Args:
        fn: Function to run
        world_size: Number of processes
        args: Arguments for function
        kwargs: Keyword arguments for function
    
    Returns:
        Any: Result of function
    """
    if kwargs is None:
        kwargs = {}
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return None
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        logger.warning(f"Number of GPUs ({device_count}) is less than world size ({world_size})")
        world_size = device_count
    
    # Spawn processes
    try:
        mp.spawn(
            fn,
            args=(world_size,) + args,
            nprocs=world_size,
            join=True
        )
        
        logger.info("Distributed run complete")
        
        return True
    except Exception as e:
        logger.error(f"Failed to run in distributed mode: {e}")
        return False

def get_rank():
    """Get process rank.
    
    Returns:
        int: Process rank
    """
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def get_world_size():
    """Get world size.
    
    Returns:
        int: World size
    """
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def is_main_process():
    """Check if current process is the main process.
    
    Returns:
        bool: True if current process is the main process, False otherwise
    """
    return get_rank() == 0

def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

def all_reduce(tensor, op="sum"):
    """All-reduce tensor across all processes.
    
    Args:
        tensor: Tensor to all-reduce
        op: Reduction operation
    
    Returns:
        torch.Tensor: All-reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor
    result = tensor.clone()
    
    # Determine operation
    if op.lower() == "sum":
        dist_op = dist.ReduceOp.SUM
    elif op.lower() == "prod":
        dist_op = dist.ReduceOp.PRODUCT
    elif op.lower() == "min":
        dist_op = dist.ReduceOp.MIN
    elif op.lower() == "max":
        dist_op = dist.ReduceOp.MAX
    else:
        logger.error(f"Unsupported operation: {op}")
        return tensor
    
    # All-reduce
    dist.all_reduce(result, op=dist_op)
    
    return result

def all_gather(tensor):
    """All-gather tensor across all processes.
    
    Args:
        tensor: Tensor to all-gather
    
    Returns:
        list: List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    # Create output tensors
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # All-gather
    dist.all_gather(output, tensor)
    
    return output

def broadcast(tensor, src=0):
    """Broadcast tensor from source process to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source process
    
    Returns:
        torch.Tensor: Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Broadcast
    dist.broadcast(tensor, src=src)
    
    return tensor

def reduce(tensor, dst=0, op="sum"):
    """Reduce tensor from all processes to destination process.
    
    Args:
        tensor: Tensor to reduce
        dst: Destination process
        op: Reduction operation
    
    Returns:
        torch.Tensor: Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor
    result = tensor.clone()
    
    # Determine operation
    if op.lower() == "sum":
        dist_op = dist.ReduceOp.SUM
    elif op.lower() == "prod":
        dist_op = dist.ReduceOp.PRODUCT
    elif op.lower() == "min":
        dist_op = dist.ReduceOp.MIN
    elif op.lower() == "max":
        dist_op = dist.ReduceOp.MAX
    else:
        logger.error(f"Unsupported operation: {op}")
        return tensor
    
    # Reduce
    dist.reduce(result, dst=dst, op=dist_op)
    
    return result

def gather(tensor, dst=0):
    """Gather tensor from all processes to destination process.
    
    Args:
        tensor: Tensor to gather
        dst: Destination process
    
    Returns:
        list: List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    rank = get_rank()
    
    # Create output tensors
    if rank == dst:
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        output = None
    
    # Gather
    dist.gather(tensor, output, dst=dst)
    
    return output

def scatter(tensor_list, src=0):
    """Scatter tensor list from source process to all processes.
    
    Args:
        tensor_list: List of tensors to scatter
        src: Source process
    
    Returns:
        torch.Tensor: Scattered tensor
    """
    if not dist.is_initialized():
        return tensor_list[0]
    
    rank = get_rank()
    
    # Create output tensor
    output = torch.zeros_like(tensor_list[0])
    
    # Scatter
    dist.scatter(output, tensor_list if rank == src else None, src=src)
    
    return output

def distributed_sampler(dataset, shuffle=True, seed=0):
    """Create distributed sampler.
    
    Args:
        dataset: Dataset
        shuffle: Whether to shuffle dataset
        seed: Random seed
    
    Returns:
        torch.utils.data.distributed.DistributedSampler: Distributed sampler
    """
    if not dist.is_initialized():
        return None
    
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed
    )

def distributed_data_parallel(model, device_ids=None, output_device=None):
    """Create distributed data parallel model.
    
    Args:
        model: PyTorch model
        device_ids: List of device IDs
        output_device: Output device ID
    
    Returns:
        torch.nn.parallel.DistributedDataParallel: Distributed data parallel model
    """
    if not dist.is_initialized():
        return model
    
    # Set default device IDs
    if device_ids is None and torch.cuda.is_available():
        device_ids = [get_rank() % torch.cuda.device_count()]
    
    # Set default output device
    if output_device is None and device_ids:
        output_device = device_ids[0]
    
    # Create distributed data parallel model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device
    )

def fully_sharded_data_parallel(model, mixed_precision=False, cpu_offload=False):
    """Create fully sharded data parallel model.
    
    Args:
        model: PyTorch model
        mixed_precision: Whether to use mixed precision
        cpu_offload: Whether to offload parameters to CPU
    
    Returns:
        torch.distributed.fsdp.FullyShardedDataParallel: Fully sharded data parallel model
    """
    if not dist.is_initialized():
        return model
    
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            CPUOffload,
            MixedPrecision,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import (
            default_auto_wrap_policy,
            enable_wrap,
            wrap,
        )
    except ImportError:
        logger.error("FSDP is not available")
        return model
    
    # Set CPU offload
    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)
    
    # Set mixed precision
    mixed_precision_config = None
    if mixed_precision:
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # Create FSDP model
    return FSDP(
        model,
        auto_wrap_policy=default_auto_wrap_policy,
        cpu_offload=cpu_offload_config,
        mixed_precision=mixed_precision_config,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filename: Checkpoint filename
    
    Returns:
        bool: True if checkpoint is saved, False otherwise
    """
    if not is_main_process():
        return True
    
    try:
        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }
        
        # Save checkpoint
        torch.save(checkpoint, filename)
        
        logger.info(f"Checkpoint saved to {filename}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False

def load_checkpoint(model, optimizer, filename):
    """Load checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filename: Checkpoint filename
    
    Returns:
        tuple: Epoch and loss
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(filename, map_location="cpu")
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Get epoch and loss
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        
        logger.info(f"Checkpoint loaded from {filename}")
        
        return epoch, loss
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, float("inf")

def distributed_training_loop(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    loss_fn,
    epochs,
    batch_size,
    checkpoint_dir="checkpoints",
    checkpoint_interval=10
):
    """Distributed training loop.
    
    Args:
        model: PyTorch model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        epochs: Number of epochs
        batch_size: Batch size
        checkpoint_dir: Checkpoint directory
        checkpoint_interval: Checkpoint interval
    
    Returns:
        torch.nn.Module: Trained model
    """
    # Set up distributed training
    rank = get_rank()
    world_size = get_world_size()
    
    # Create checkpoint directory
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create distributed samplers
    train_sampler = distributed_sampler(train_dataset)
    val_sampler = distributed_sampler(val_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Wrap model with DistributedDataParallel
    model = distributed_data_parallel(model)
    
    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update train loss
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0 and is_main_process():
                logger.info(f"Train Epoch: {epoch} [{batch_idx * len(data) * world_size}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        # Calculate average train loss
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to GPU
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                val_loss += loss_fn(output, target).item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate validation accuracy
        val_accuracy = 100. * correct / len(val_loader.dataset)
        
        # Print validation results
        if is_main_process():
            logger.info(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)")
        
        # Save checkpoint
        if epoch % checkpoint_interval == 0 and is_main_process():
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_file)
    
    # Save final checkpoint
    if is_main_process():
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, epochs - 1, val_loss, checkpoint_file)
    
    return model

if __name__ == "__main__":
    # Example usage
    def main(rank, world_size):
        # Set up distributed training
        setup_distributed(rank, world_size)
        
        # Create model
        model = torch.nn.Linear(10, 10)
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Wrap model with DistributedDataParallel
        model = distributed_data_parallel(model)
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Create loss function
        loss_fn = torch.nn.MSELoss()
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 10)
        )
        
        # Create sampler
        sampler = distributed_sampler(dataset)
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            sampler=sampler
        )
        
        # Training loop
        for epoch in range(10):
            # Set epoch for sampler
            sampler.set_epoch(epoch)
            
            for data, target in dataloader:
                # Move data to GPU
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                # Forward pass
                output = model(data)
                loss = loss_fn(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Clean up
        cleanup_distributed()
    
    # Run distributed training
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        run_distributed(main, world_size)
