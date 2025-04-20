# PyTorch Profiler for AMD GPUs: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Profiling](#basic-profiling)
4. [Advanced Profiling](#advanced-profiling)
5. [Memory Profiling](#memory-profiling)
6. [TensorBoard Integration](#tensorboard-integration)
7. [Analyzing Results](#analyzing-results)
8. [Optimization Strategies](#optimization-strategies)
9. [AMD-Specific Considerations](#amd-specific-considerations)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

## Introduction

PyTorch Profiler is a powerful tool for analyzing the performance of PyTorch models. It provides detailed insights into CPU and GPU execution times, memory usage, and operator-level performance. This guide focuses on using PyTorch Profiler with AMD GPUs through ROCm.

### Key Features

- **Operator-Level Profiling**: Analyze the performance of individual PyTorch operators
- **Memory Profiling**: Track memory allocations and identify memory bottlenecks
- **Stack Traces**: Identify the source code responsible for performance issues
- **TensorBoard Integration**: Visualize profiling results in TensorBoard
- **Custom Events**: Add custom events to track specific parts of your code
- **ROCm Support**: Works with AMD GPUs through PyTorch's ROCm backend

### Benefits for ML Workflows

1. **Performance Optimization**: Identify and eliminate bottlenecks in your models
2. **Memory Optimization**: Reduce memory usage and avoid out-of-memory errors
3. **Training Speedup**: Optimize training loops for faster iteration
4. **Inference Optimization**: Improve inference latency and throughput
5. **Hardware Utilization**: Ensure efficient use of AMD GPU resources

## Installation

### Prerequisites

- ROCm 5.0+ installed
- PyTorch with ROCm support
- Python 3.6+
- AMD Radeon GPU with ROCm support

### Automated Installation

We provide an installation script that handles all dependencies and configuration:

```bash
# Make the script executable
chmod +x $HOME/Desktop/ml_stack_extensions/install_pytorch_profiler.sh

# Run the installation script
$HOME/Desktop/ml_stack_extensions/install_pytorch_profiler.sh
```

### Manual Installation

PyTorch Profiler is included with PyTorch, but you'll need to install additional dependencies:

```bash
# Install dependencies
pip install tensorboard pandas matplotlib

# Verify installation
python -c "from torch.profiler import profile, record_function, ProfilerActivity; print('PyTorch Profiler is available')"
```

## Basic Profiling

### Simple Profiling Example

Here's a basic example of profiling a PyTorch model:

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Define a simple model
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
).to('cuda')

# Create input data
x = torch.randn(32, 100, device='cuda')

# Profile the model
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("model_inference"):
        model(x)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export trace
prof.export_chrome_trace("trace.json")
```

### Profiling Activities

PyTorch Profiler can track different types of activities:

```python
# Profile CPU and CUDA activities
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(x)

# Profile only CUDA activities
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(x)

# Profile only CPU activities
with profile(activities=[ProfilerActivity.CPU]) as prof:
    model(x)
```

### Recording Functions

You can add custom events to your profiling trace:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Record the entire forward pass
    with record_function("forward_pass"):
        output = model(x)
    
    # Record the loss calculation
    with record_function("loss_calculation"):
        loss = criterion(output, target)
    
    # Record the backward pass
    with record_function("backward_pass"):
        loss.backward()
```

### Profiling Training Loops

Here's how to profile a complete training loop:

```python
# Warm-up
for _ in range(3):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Profile training
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(10):
        with record_function(f"iteration_{i}"):
            optimizer.zero_grad()
            
            with record_function("forward"):
                output = model(x)
                loss = criterion(output, y)
            
            with record_function("backward"):
                loss.backward()
            
            with record_function("optimizer_step"):
                optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## Advanced Profiling

### Profiling Schedules

PyTorch Profiler supports custom profiling schedules to reduce overhead:

```python
from torch.profiler import schedule

# Define a custom schedule
my_schedule = schedule(
    skip_first=10,   # Skip first 10 steps (warm-up)
    wait=5,          # Wait 5 steps
    warmup=1,        # Warmup for 1 step
    active=3,        # Profile for 3 steps
    repeat=2         # Repeat the cycle twice
)

# Profile with schedule
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule
) as prof:
    for i in range(30):
        model(x)
        prof.step()  # Important: step the profiler
```

### Profiling with Stack Traces

Stack traces help identify the source code responsible for performance issues:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    model(x)

# Export stack traces
prof.export_stacks("stacks.txt", "self_cuda_time_total")
```

### FLOP Counting

PyTorch Profiler can estimate the number of floating-point operations:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    model(x)

# Print results with FLOP counts
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Profiling Distributed Training

For multi-GPU training with DistributedDataParallel:

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Profile only on rank 0
    if rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # Training loop
            for i in range(10):
                ddp_model(x)
                prof.step()
        
        # Print results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        # Training loop without profiling
        for i in range(10):
            ddp_model(x)

# Launch distributed training
world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

## Memory Profiling

### Basic Memory Profiling

PyTorch Profiler can track memory allocations and usage:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    model(x)

# Print memory stats
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

### Identifying Memory Bottlenecks

To identify operations that consume the most memory:

```python
import pandas as pd
import matplotlib.pyplot as plt

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    model(x)

# Get events
events = prof.events()
memory_events = []

for evt in events:
    if evt.cuda_memory_usage != 0:
        memory_events.append({
            'name': evt.name,
            'memory_usage': evt.cuda_memory_usage / (1024 * 1024),  # Convert to MB
            'self_memory_usage': evt.self_cuda_memory_usage / (1024 * 1024),  # Convert to MB
            'device': evt.device
        })

# Convert to DataFrame
df = pd.DataFrame(memory_events)

# Plot memory usage by operator
plt.figure(figsize=(12, 6))
top_memory_ops = df.nlargest(10, 'self_memory_usage')
plt.barh(top_memory_ops['name'], top_memory_ops['self_memory_usage'])
plt.xlabel('Self Memory Usage (MB)')
plt.title('Top 10 Memory-Intensive Operations')
plt.tight_layout()
plt.savefig('memory_usage_by_op.png')
```

### Memory Optimization Techniques

Based on profiling results, you can apply these memory optimization techniques:

1. **Gradient Checkpointing**: Trade computation for memory
   ```python
   from torch.utils.checkpoint import checkpoint
   
   # Use checkpointing for memory-intensive layers
   output = checkpoint(model.expensive_layer, input)
   ```

2. **Mixed Precision Training**: Use lower precision to reduce memory usage
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(x)
       loss = criterion(output, y)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Batch Size Optimization**: Find the optimal batch size for your GPU
   ```python
   # Profile different batch sizes
   batch_sizes = [16, 32, 64, 128]
   for bs in batch_sizes:
       x = torch.randn(bs, 100, device='cuda')
       with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
           model(x)
       print(f"Batch size: {bs}")
       print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5))
   ```

4. **Model Optimization**: Identify and optimize memory-intensive layers
   ```python
   # Replace memory-intensive layers
   # model.memory_intensive_layer = EfficientLayer()
   ```

## TensorBoard Integration

### Basic TensorBoard Integration

PyTorch Profiler integrates with TensorBoard for visualization:

```python
from torch.profiler import tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./log/pytorch_profiler")
) as prof:
    model(x)

# View in TensorBoard
# tensorboard --logdir=./log/pytorch_profiler
```

### Profiling Multiple Runs

To profile multiple runs and compare them in TensorBoard:

```python
for run_idx in range(3):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(f"./log/pytorch_profiler/run_{run_idx}")
    ) as prof:
        model(x)
```

### Custom TensorBoard Logging

You can customize the TensorBoard logging:

```python
def custom_trace_handler(prof):
    dir_name = f"./log/pytorch_profiler/custom_{prof.step_num}"
    prof.export_chrome_trace(f"{dir_name}/trace.json")
    prof.export_stacks(f"{dir_name}/stacks.txt", "self_cuda_time_total")
    print(f"Exported trace to {dir_name}")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=custom_trace_handler
) as prof:
    for i in range(5):
        model(x)
        prof.step()
```

## Analyzing Results

### Underuserding Profiler Output

The profiler output includes several key columns:

- **self_cpu_time_total**: CPU time spent in the operator, excluding child operators
- **cpu_time_total**: Total CPU time spent in the operator, including child operators
- **self_cuda_time_total**: GPU time spent in the operator, excluding child operators
- **cuda_time_total**: Total GPU time spent in the operator, including child operators
- **self_cuda_memory_usage**: Memory allocated by the operator
- **cuda_memory_usage**: Total memory allocated by the operator and its children
- **self_cpu_memory_usage**: CPU memory allocated by the operator
- **cpu_memory_usage**: Total CPU memory allocated by the operator and its children

### Filtering and Sorting Results

You can filter and sort profiler results:

```python
# Sort by GPU time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Sort by memory usage
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# Filter by operator name
print(prof.key_averages(group_by_input_shape=True).filter(lambda x: "conv" in x.name).table())

# Group by input shape
print(prof.key_averages(group_by_input_shape=True).table())

# Group by operator name
print(prof.key_averages(group_by_input_shape=False).table())
```

### Analyzing Chrome Traces

To analyze Chrome traces:

1. Export the trace:
   ```python
   prof.export_chrome_trace("trace.json")
   ```

2. Open Chrome and navigate to `chrome://tracing`

3. Click "Load" and select the trace file

4. Use the Chrome tracing UI to:
   - Zoom in/out with the mouse wheel
   - Pan by clicking and dragging
   - Select events to see details
   - Use WASD keys to navigate

### Analyzing Stack Traces

To analyze stack traces:

1. Export the stacks:
   ```python
   prof.export_stacks("stacks.txt", "self_cuda_time_total")
   ```

2. Open the stacks file to see:
   - The most time-consuming operations
   - The source code that called these operations
   - The call hierarchy

## Optimization Strategies

### CPU Bottlenecks

If profiling shows CPU bottlenecks:

1. **Data Loading Optimization**:
   ```python
   # Use more workers for DataLoader
   dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
   ```

2. **CPU Preprocessing Optimization**:
   ```python
   # Move preprocessing to GPU
   def preprocess(x):
       return x.to('cuda') * 2  # Perform operations on GPU
   ```

3. **Asynchronous Operations**:
   ```python
   # Use non-blocking transfers
   x = x.to('cuda', non_blocking=True)
   ```

### GPU Bottlenecks

If profiling shows GPU bottlenecks:

1. **Operator Fusion**:
   ```python
   # Fuse operations when possible
   x = torch.nn.functional.relu(model.conv(x))  # Fused in one kernel
   ```

2. **Kernel Selection**:
   ```python
   # Use optimized implementations
   x = torch.nn.functional.conv2d(x, weight, padding=1, bias=None)
   ```

3. **Batch Size Tuning**:
   ```python
   # Find optimal batch size
   batch_size = 64  # Adjust based on profiling
   ```

### Memory Bottlenecks

If profiling shows memory bottlenecks:

1. **Gradient Accumulation**:
   ```python
   # Accumulate gradients over multiple batches
   for i, (x, y) in enumerate(dataloader):
       output = model(x)
       loss = criterion(output, y) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Model Parallelism**:
   ```python
   # Split model across multiple GPUs
   model = ModelParallel()  # Custom implementation
   ```

3. **Activation Checkpointing**:
   ```python
   # Use checkpointing for memory-intensive layers
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(model.layer, input)
   ```

## AMD-Specific Considerations

### ROCm Profiling Tools

In addition to PyTorch Profiler, AMD provides ROCm profiling tools:

1. **rocprof**: Command-line profiler for ROCm applications
   ```bash
   rocprof --stats python your_script.py
   ```

2. **roctracer**: API for tracing ROCm runtime and HIP APIs
   ```bash
   ROCTRACER_ENABLE=1 python your_script.py
   ```

3. **Radeon Compute Profiler (RCP)**: GUI-based profiler
   ```bash
   rcp -f python your_script.py
   ```

### ROCm-Specific Optimizations

1. **Environment Variables**:
   ```bash
   # Set visible devices
   export HIP_VISIBLE_DEVICES=0,1
   
   # Set memory pool size
   export HSA_TOOLS_LIB=1
   export HSA_ENABLE_SDMA=0
   ```

2. **Kernel Launch Parameters**:
   ```python
   # Adjust grid and block sizes for AMD GPUs
   # This is handled automatically by PyTorch, but custom CUDA kernels may need adjustment
   ```

3. **Memory Management**:
   ```python
   # Clear cache periodically
   torch.cuda.empty_cache()
   
   # Set memory fraction
   torch.cuda.set_per_process_memory_fraction(0.8)
   ```

### AMD vs NVIDIA Profiling Differences

When using PyTorch Profiler with AMD GPUs, be aware of these differences:

1. **Terminology**: ROCm uses HIP instead of CUDA, but PyTorch Profiler still uses CUDA terminology
2. **Kernel Names**: Kernel names may differ between AMD and NVIDIA
3. **Performance Characteristics**: Some operations may have different performance characteristics on AMD GPUs
4. **Memory Management**: Memory allocation patterns may differ
5. **Tensor Cores**: AMD's matrix cores are used differently than NVIDIA's Tensor Cores

## Troubleshooting

### Common Issues

1. **No CUDA Events**:
   ```
   No CUDA events recorded
   ```
   Solution: Ensure PyTorch is built with ROCm support and GPU is visible:
   ```python
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

2. **Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch size or use memory optimization techniques:
   ```python
   # Reduce batch size
   batch_size = batch_size // 2
   
   # Clear cache
   torch.cuda.empty_cache()
   ```

3. **Profiler Overhead**:
   ```
   Profiling is slow
   ```
   Solution: Use profiling schedules to reduce overhead:
   ```python
   with profile(
       activities=[ProfilerActivity.CUDA],
       schedule=schedule(wait=1, warmup=1, active=3, repeat=1)
   ) as prof:
       # ...
   ```

4. **Missing Stack Traces**:
   ```
   No stack traces available
   ```
   Solution: Enable stack trace collection:
   ```python
   with profile(
       activities=[ProfilerActivity.CUDA],
       with_stack=True
   ) as prof:
       # ...
   ```

### Debugging Tips

1. **Isolate Components**:
   ```python
   # Profile individual components
   with profile(activities=[ProfilerActivity.CUDA]) as prof:
       model.layer1(x)
   print(prof.key_averages().table())
   
   with profile(activities=[ProfilerActivity.CUDA]) as prof:
       model.layer2(x)
   print(prof.key_averages().table())
   ```

2. **Compare Implementations**:
   ```python
   # Profile different implementations
   with profile(activities=[ProfilerActivity.CUDA]) as prof:
       output1 = implementation1(x)
   print("Implementation 1:")
   print(prof.key_averages().table())
   
   with profile(activities=[ProfilerActivity.CUDA]) as prof:
       output2 = implementation2(x)
   print("Implementation 2:")
   print(prof.key_averages().table())
   ```

3. **Check GPU Utilization**:
   ```bash
   # Monitor GPU utilization while profiling
   watch -n 1 rocm-smi
   ```

## References

1. [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
2. [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
3. [TensorBoard Profiler Plugin](https://pytorch.org/docs/stable/tensorboard.html)
4. [ROCm Documentation](https://rocm.docs.amd.com/)
5. [PyTorch ROCm Support](https://pytorch.org/docs/stable/notes/hip.html)
6. [Chrome Tracing Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
7. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
8. [ROCm Profiling Tools](https://rocm.docs.amd.com/projects/ROCm_Tools/en/latest/)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

