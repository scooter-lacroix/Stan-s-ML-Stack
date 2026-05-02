#!/usr/bin/env python3
# =============================================================================
# MPI with PyTorch Example for AMD GPUs
# =============================================================================
# This script demonstrates how to use MPI with PyTorch for distributed training.
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
from mpi4py import MPI

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def setup_mpi():
    """Set up MPI environment."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set PyTorch environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    return rank, world_size, comm

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def train(model, optimizer, criterion, input_data, target_data, epochs=10):
    """Train the model."""
    model.train()
    
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target_data)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Return the last loss
        if epoch == epochs - 1:
            return loss.item()

def main():
    parser = argparse.ArgumentParser(description="MPI with PyTorch Example")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Set up MPI
    rank, world_size, comm = setup_mpi()
    
    if rank == 0:
        print("=== MPI with PyTorch Example ===")
        print(f"World size: {world_size}")
    
    # Set device
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Using {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create model and move to GPU
    model = SimpleModel(hidden_size=args.hidden_size).to(device)
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Create optimizer
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dummy data
    input_data = torch.randn(args.batch_size, args.hidden_size, device=device)
    target_data = torch.randn(args.batch_size, args.hidden_size, device=device)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    comm.Barrier()
    
    # Time the training
    start_time = time.time()
    
    # Train the model
    loss = train(ddp_model, optimizer, criterion, input_data, target_data, epochs=args.epochs)
    
    # Synchronize after training
    torch.cuda.synchronize()
    comm.Barrier()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Gather losses from all processes
    losses = comm.gather(loss, root=0)
    times = comm.gather(training_time, root=0)
    
    # Print results on rank 0
    if rank == 0:
        avg_loss = sum(losses) / len(losses)
        avg_time = sum(times) / len(times)
        
        print("\nTraining Results:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Average Training Time: {avg_time:.4f} seconds")
        print(f"  Total Training Time: {max(times):.4f} seconds")
        
        print("\nPer-Process Results:")
        for i, (l, t) in enumerate(zip(losses, times)):
            print(f"  Rank {i}: Loss = {l:.6f}, Time = {t:.4f} seconds")
    
    # Clean up
    cleanup()
    
    if rank == 0:
        print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
