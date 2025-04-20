# AMD GPU Optimization Guide

## Introduction

This guide provides detailed information on optimizing machine learning workloads for AMD GPUs using ROCm. It covers various optimization techniques, best practices, and performance tuning tips to get the most out of your AMD GPU hardware.

## Table of Contents

1. [Hardware Considerations](#hardware-considerations)
2. [Environment Setup](#environment-setup)
3. [Memory Management](#memory-management)
4. [Compute Optimization](#compute-optimization)
5. [Multi-GPU Scaling](#multi-gpu-scaling)
6. [Profiling and Debugging](#profiling-and-debugging)
7. [Framework-Specific Optimizations](#framework-specific-optimizations)
8. [Custom Kernel Development](#custom-kernel-development)

## Hardware Considerations

### GPU Selection

AMD's Radeon RX 7000 series GPUs offer excellent performance for machine learning workloads:

- **RX 7900 XTX**: Top-tier performance with 24GB VRAM
- **RX 7900 XT**: High performance with 20GB VRAM
- **RX 7800 XT**: Good balance of performance and cost with 16GB VRAM
- **RX 7700 XT**: Entry-level ML performance with 12GB VRAM

For professional workloads, consider the AMD Instinct series:

- **MI300X**: Flagship accelerator with 192GB HBM3 memory
- **MI250X**: High-performance accelerator with 128GB HBM2e memory
- **MI210**: Mid-range accelerator with 64GB HBM2e memory

### System Configuration

For optimal performance:

- Use PCIe 4.0 or higher slots for maximum bandwidth
- Ensure adequate power supply (850W+ for high-end GPUs)
- Provide sufficient cooling (liquid cooling recommended for intensive workloads)
- Use high-speed NVMe storage to avoid I/O bottlenecks
- Configure at least 32GB system RAM, 64GB+ recommended for large models

## Environment Setup

### ROCm Configuration

Set these environment variables for optimal performance:

```bash
# Specify which GPUs to use (comma-separated indices)
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_DEVICE=0,1

# Performance tuning
export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance in some workloads
export GPU_MAX_HEAP_SIZE=100  # Increase heap size (in %)
export GPU_MAX_ALLOC_PERCENT=100  # Allow allocating 100% of available memory
export GPU_SINGLE_ALLOC_PERCENT=100  # Allow single allocations up to 100%

# For multi-GPU workloads
export NCCL_DEBUG=INFO  # Enable NCCL debugging info
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not using it
export NCCL_SOCKET_IFNAME=^lo  # Use all interfaces except loopback
```

### PyTorch Configuration

For PyTorch workloads:

```python
# Set memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory

# Optimize for specific workloads
torch.cuda.max_split_size_mb = 512  # Optimal for RX 7900 XTX

# Enable benchmark mode for optimal performance with fixed input sizes
torch.backends.cudnn.benchmark = True

# Disable gradient synchronization for faster training
torch.set_grad_enabled(False)  # During inference
```

## Memory Management

### Memory Allocation Strategies

AMD GPUs perform best with these memory management practices:

1. **Pre-allocate tensors** when possible to reduce allocation overhead
2. **Reuse tensors** instead of creating new ones
3. **Use mixed precision training** (float16/bfloat16) to reduce memory usage
4. **Implement gradient checkpointing** for large models
5. **Optimize batch size** based on available memory

### Example: Memory-Efficient Training

```python
# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop with memory optimization
for data, target in dataloader:
    # Move data to GPU
    data, target = data.to('cuda'), target.to('cuda')
    
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Update weights with gradient unscaling
    scaler.step(optimizer)
    scaler.update()
    
    # Explicitly clear cache periodically
    if step % 10 == 0:
        torch.cuda.empty_cache()
```

### Optimal Memory Settings

For different AMD GPU models:

| GPU Model | max_split_size_mb | Memory Fraction | Optimal Batch Size |
|-----------|-------------------|-----------------|-------------------|
| RX 7900 XTX | 512 | 0.95 | Model-dependent |
| RX 7900 XT | 384 | 0.95 | Model-dependent |
| RX 7800 XT | 256 | 0.90 | Model-dependent |
| RX 7700 XT | 128 | 0.90 | Model-dependent |

## Compute Optimization

### Kernel Selection

ROCm provides several optimized kernels for common operations:

1. **MIOpen** for deep learning primitives
2. **rocBLAS** for linear algebra operations
3. **hipFFT** for fast Fourier transforms
4. **rocRAND** for random number generation

### Precision Optimization

Choose the appropriate precision for your workload:

- **FP32 (float)**: Use for training when accuracy is critical
- **FP16 (half)**: Use for inference and training with mixed precision
- **BF16 (bfloat16)**: Better numerical stability than FP16, good for training
- **INT8**: Use for quantized inference when accuracy requirements allow

### Example: Mixed Precision Training

```python
# Define model in default precision
model = MyModel().cuda()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize gradient scaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # Forward pass in mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        # Backward pass with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Multi-GPU Scaling

### Data Parallelism

For simple multi-GPU training:

```python
# Wrap model in DataParallel
model = torch.nn.DataParallel(model)
```

### Distributed Training

For more advanced multi-GPU and multi-node training:

```python
# Initialize distributed process group
torch.distributed.init_process_group(backend="nccl")

# Create distributed model
model = torch.nn.parallel.DistributedDataParallel(model)

# Use distributed sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler
)
```

### RCCL Optimization

For optimal communication between GPUs:

```bash
# Set RCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
```

## Profiling and Debugging

### ROCm Profiling Tools

Use these tools to identify performance bottlenecks:

1. **rocprof**: Command-line profiler for ROCm applications
2. **roctracer**: API tracing library
3. **PyTorch Profiler**: Built-in profiling for PyTorch workloads

### Example: PyTorch Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

# Profile model training
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (inputs, targets) in enumerate(dataloader):
        if step >= 10:
            break
        with record_function("training_step"):
            outputs = model(inputs.cuda())
            loss = loss_fn(outputs, targets.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Framework-Specific Optimizations

### PyTorch

```python
# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# Disable gradient synchronization during validation
with torch.no_grad():
    validation_output = model(validation_input)

# Use channels_last memory format for convolutional models
model = model.to(memory_format=torch.channels_last)
input_tensor = input_tensor.to(memory_format=torch.channels_last)
```

### TensorFlow

```python
# Use mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Enable XLA compilation
tf.config.optimizer.set_jit(True)
```

### ONNX Runtime

```python
# Create session with ROCm provider
session = onnxruntime.InferenceSession(
    model_path,
    providers=['ROCMExecutionProvider', 'CPUExecutionProvider']
)
```

## Custom Kernel Development

### HIP Programming

For custom kernels, use HIP (Heterogeneous-Computing Interface for Portability):

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(float* a, float* b, float* c, int n) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    
    vector_add<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
}
```

### PyTorch Custom CUDA Extensions

Create custom CUDA extensions for PyTorch:

```python
from torch.utils.cpp_extension import load

# Compile and load custom kernel
vector_add = load(
    name="vector_add",
    sources=["vector_add.cpp", "vector_add_kernel.cu"],
    extra_cuda_cflags=["-O3"]
)

# Use the custom kernel
output = vector_add.forward(input1, input2)
```

## Conclusion

Optimizing for AMD GPUs requires careful attention to memory management, compute optimization, and proper environment configuration. By following the guidelines in this document, you can achieve significant performance improvements for your machine learning workloads on AMD GPU hardware.

## References

- [ROCm Documentation](https://rocmdocs.amd.com/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD Developer Resources](https://developer.amd.com/resources/)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

