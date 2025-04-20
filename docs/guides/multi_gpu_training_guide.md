# Multi-GPU Training Guide for AMD GPUs

## Introduction

This guide provides comprehensive information on setting up and optimizing multi-GPU training for deep learning models on AMD GPUs. It covers various parallelism strategies, communication libraries, and best practices for achieving efficient distributed training.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Setup](#hardware-setup)
3. [Software Configuration](#software-configuration)
4. [Parallelism Strategies](#parallelism-strategies)
5. [Communication Libraries](#communication-libraries)
6. [PyTorch Distributed Training](#pytorch-distributed-training)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Case Studies](#case-studies)

## Overview

Multi-GPU training is essential for:

1. **Training larger models**: Models that don't fit in a single GPU's memory
2. **Faster training**: Reducing training time by parallelizing computation
3. **Scaling to larger datasets**: Processing more data in parallel

AMD GPUs with ROCm support provide excellent performance for distributed training, especially when properly configured and optimized.

## Hardware Setup

### Recommended Hardware

For optimal multi-GPU training:

- **GPUs**: AMD Radeon RX 7900 XTX or AMD Instinct MI series
- **CPU**: 32+ cores with high memory bandwidth
- **RAM**: 128GB+ system memory
- **Storage**: NVMe SSDs with high throughput
- **Network**: 10GbE or higher for multi-node setups (100GbE preferred)
- **PCIe**: PCIe 4.0 or higher with sufficient lanes

### GPU Topology

Understanding the GPU topology is crucial for optimal performance:

```bash
# Check GPU topology
rocm-smi --showtoponuma
```

For best performance:

- Place GPUs on the same NUMA node when possible
- Use GPUs connected to the same PCIe switch
- Ensure balanced PCIe lane allocation

## Software Configuration

### ROCm Setup

Ensure ROCm is properly installed and configured:

```bash
# Check ROCm version
rocm-smi --version

# Set environment variables
export HIP_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, 3
export PYTORCH_ROCM_DEVICE=0,1,2,3  # Use GPUs 0, 1, 2, 3
```

### RCCL Configuration

RCCL (ROCm Communication Collective Library) is crucial for efficient GPU-to-GPU communication:

```bash
# Check RCCL installation
ls -la /opt/rocm/lib/librccl*

# Set RCCL environment variables
export NCCL_DEBUG=INFO  # Enable debugging info
export NCCL_SOCKET_IFNAME=^lo  # Use all interfaces except loopback
export NCCL_P2P_DISABLE=0  # Enable peer-to-peer communication
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not using it
```

### MPI Configuration

For multi-node training with MPI:

```bash
# Install OpenMPI with ROCm support
sudo apt install openmpi-bin libopenmpi-dev

# Set MPI environment variables
export OMPI_MCA_btl=^openib  # Disable OpenIB
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
```

## Parallelism Strategies

### Data Parallelism

The simplest form of parallelism, where each GPU processes a different batch of data:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")

# Create model and move to GPU
model = MyModel().to(torch.device("cuda", dist.get_rank()))

# Wrap model with DDP
model = DDP(model, device_ids=[dist.get_rank()])

# Create distributed sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler
)

# Training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Model Parallelism

Splitting the model across multiple GPUs:

```python
import torch

# Define model parts
class ModelPart1(torch.nn.Module):
    # First part of the model
    ...

class ModelPart2(torch.nn.Module):
    # Second part of the model
    ...

# Create model parts on different GPUs
model_part1 = ModelPart1().to("cuda:0")
model_part2 = ModelPart2().to("cuda:1")

# Forward pass
def forward(x):
    x = model_part1(x.to("cuda:0"))
    x = x.to("cuda:1")
    return model_part2(x)
```

### Pipeline Parallelism

Combining model and data parallelism with pipelining:

```python
# Using PyTorch's RPC framework for pipeline parallelism
import torch.distributed.rpc as rpc
import torch.distributed.pipeline.sync as pipe

# Initialize RPC
rpc.init_rpc("worker", rank=rank, world_size=world_size)

# Create pipeline model
model = pipe.Pipe(
    module=model,
    chunks=8,  # Number of micro-batches
    checkpoint="never"  # Gradient checkpointing strategy
)

# Training loop
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Tensor Parallelism

Splitting individual tensors across GPUs:

```python
# Using Megatron-LM for tensor parallelism
from megatron import initialize_megatron
from megatron.model import get_args, get_model

# Initialize Megatron
initialize_megatron(extra_args_provider=get_args)

# Create model with tensor parallelism
model = get_model(model_provider, wrap_with_ddp=True)

# Training loop
for batch in train_data_iterator:
    output = model(batch)
    loss = criterion(output, batch)
    model.backward(loss)
    optimizer.step()
```

## Communication Libraries

### RCCL

RCCL is the ROCm equivalent of NVIDIA's NCCL, providing efficient collective communication operations:

```python
import torch.distributed as dist

# Initialize process group with RCCL backend
dist.init_process_group(backend="nccl")

# Perform all-reduce operation
tensor = torch.randn(10).cuda()
dist.all_reduce(tensor)
```

### MPI

MPI provides more flexibility for complex distributed setups:

```python
from mpi4py import MPI
import torch

# Get MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Assign GPU to process
torch.cuda.set_device(rank % torch.cuda.device_count())

# Perform all-reduce operation
tensor = torch.randn(10).cuda()
tensor_np = tensor.cpu().numpy()
result_np = np.empty_like(tensor_np)
comm.Allreduce(tensor_np, result_np, op=MPI.SUM)
result = torch.tensor(result_np).cuda()
```

## PyTorch Distributed Training

### Single-Node Multi-GPU Training

```bash
# Launch training script with torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

```python
# train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Create model and move to GPU
    model = MyModel().to(torch.device("cuda", dist.get_rank()))
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[dist.get_rank()])
    
    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
```

### Multi-Node Training

```bash
# On node 0 (master)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 train.py

# On node 1
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=29500 train.py
```

### Fully Sharded Data Parallel (FSDP)

For training very large models with limited GPU memory:

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# Initialize process group
dist.init_process_group(backend="nccl")

# Define FSDP wrapping policy
my_auto_wrap_policy = functools.partial(
    default_auto_wrap_policy,
    min_num_params=100_000
)

# Create model with FSDP
with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=my_auto_wrap_policy):
    model = wrap(MyModel().cuda())

# Training loop
for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## Performance Optimization

### Gradient Accumulation

Increase effective batch size without increasing memory usage:

```python
# Training loop with gradient accumulation
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

Use FP16 or BF16 for faster training:

```python
# Using PyTorch's AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop with mixed precision
for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    
    # Forward pass in mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Optimizing Batch Size

Find the optimal batch size for your hardware:

```python
# Start with a small batch size and increase until OOM
batch_sizes = [16, 32, 64, 128, 256]
for batch_size in batch_sizes:
    try:
        # Create dataloader with current batch size
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
        
        # Run a few iterations
        for i, (data, target) in enumerate(train_loader):
            if i >= 5:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Batch size {batch_size} works")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Batch size {batch_size} is too large")
            break
        else:
            raise e
```

### Optimizing Communication

Reduce communication overhead:

```python
# Gradient bucketing
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

# Apply gradient bucketing communication hook
model = DDP(model, device_ids=[dist.get_rank()])
model.register_comm_hook(None, default_hooks.allreduce_hook)

# Gradient compression
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

# Apply PowerSGD compression hook
state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,
    start_powerSGD_iter=10,
)
model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)
```

## Monitoring and Debugging

### GPU Monitoring

```bash
# Monitor GPU usage
watch -n 1 rocm-smi

# Detailed GPU monitoring
rocm-smi --showmeminfo vram --showuse --showclocks
```

### Distributed Training Debugging

```python
# Enable NCCL debugging
os.environ["NCCL_DEBUG"] = "INFO"

# Print rank information
print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")

# Synchronize processes for debugging
dist.barrier()
```

### Performance Profiling

```python
# Using PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= 5:
            break
        with record_function("forward"):
            output = model(data.cuda())
        with record_function("backward"):
            loss = criterion(output, target.cuda())
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        prof.step()

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Case Studies

### Training ResNet-50 on ImageNet

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
from torch.cuda.amp import autocast, GradScaler

def main():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    # Create model and move to GPU
    model = resnet50().to(torch.device("cuda", local_rank))
    model = DDP(model, device_ids=[local_rank])
    
    # Define transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    train_dataset = ImageNet(root="path/to/imagenet", split="train", transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, sampler=train_sampler, num_workers=8, pin_memory=True
    )
    
    # Define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(90):
        train_sampler.set_epoch(epoch)
        
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)
        
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            
            # Forward pass with mixed precision
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass with scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    main()
```

### Training BERT with Megatron-LM

```python
from megatron import initialize_megatron
from megatron.training import pretrain
from megatron.arguments import parse_args
from megatron.checkpointing import load_checkpoint
from megatron.model import get_args, get_model
from megatron.optimizer import get_optimizer
from megatron.learning_rates import get_learning_rate_scheduler
from megatron.training import train_step
from megatron.utils import get_ltor_masks_and_position_ids

def main():
    # Initialize Megatron
    initialize_megatron(extra_args_provider=get_args)
    args = get_args()
    
    # Build model
    model = get_model(model_provider=model_provider, wrap_with_ddp=True)
    
    # Load checkpoint if needed
    iteration = 0
    if args.load:
        iteration = load_checkpoint(model, optimizer)
    
    # Get optimizer and learning rate scheduler
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)
    
    # Training loop
    for i in range(iteration, args.train_iters):
        # Get batch
        batch = get_batch()
        
        # Forward and backward pass
        loss = train_step(batch, model, optimizer, lr_scheduler)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if i % args.save_interval == 0:
            save_checkpoint(i, model, optimizer)

if __name__ == "__main__":
    main()
```

## Conclusion

Multi-GPU training on AMD GPUs with ROCm provides excellent performance and scalability for deep learning workloads. By following the techniques and best practices in this guide, you can efficiently train large models across multiple GPUs and nodes.

For more information, refer to the [PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/beginner/dist_overview.html) and the [ROCm Documentation](https://rocmdocs.amd.com/).


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

