# Triton for AMD GPUs: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture Overview](#architecture-overview)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Integration with PyTorch](#integration-with-pytorch)
8. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
9. [Benchmarking](#benchmarking)
10. [Examples](#examples)
11. [References](#references)

## Introduction

Triton is an open-source language and compiler for parallel programming that can generate highly optimized GPU kernels. It provides a Python-based programming model that allows developers to write efficient GPU code without deep knowledge of GPU architecture.

### Key Features

- **Python-based DSL**: Write GPU kernels in a Python-like syntax
- **Automatic Optimization**: Compiler automatically handles tiling, memory coalescing, and other optimizations
- **Integration with PyTorch**: Seamlessly use Triton kernels within PyTorch models
- **Support for AMD GPUs**: Works with ROCm platform for AMD hardware
- **Dynamic Shape Support**: Efficiently handle tensors with varying dimensions

### Why Triton for AMD GPUs?

While AMD GPUs have excellent hardware capabilities, they often lack the software ecosystem that NVIDIA enjoys. Triton helps bridge this gap by providing:

1. **Performance**: Custom kernels that can outperform vendor libraries
2. **Flexibility**: Write specialized operations not available in userdard libraries
3. **Portability**: Same code can run on both NVIDIA and AMD hardware
4. **Optimization**: Automatic handling of complex GPU-specific optimizations

## Installation

### Prerequisites

- ROCm 5.0+ installed
- PyTorch with ROCm support
- Python 3.7+
- CMake 3.18+
- Ninja build system

### Automated Installation

We provide an installation script that handles all dependencies and configuration:

```bash
# Make the script executable
chmod +x $HOME/Desktop/ml_stack_extensions/install_triton.sh

# Run the installation script
$HOME/Desktop/ml_stack_extensions/install_triton.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/openai/triton.git
cd triton

# Check out a stable version
git checkout tags/v2.2.0 -b v2.2.0-stable

# Install Python dependencies
pip install --upgrade cmake ninja pytest packaging wheel

# Build and install
cd python
pip install -e .

# Verify installation
python -c "import triton; print(triton.__version__)"
```

### Verifying ROCm Support

To verify that Triton is working with ROCm:

```python
import torch
import triton

# Check if ROCm is available
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Check Triton version
print(f"Triton version: {triton.__version__}")
```

## Architecture Overview

Triton consists of several key components:

1. **Triton Language**: A Python-based domain-specific language (DSL) for expressing parallel computations
2. **Triton Compiler**: Translates Triton code to optimized GPU kernels
3. **Runtime System**: Manages kernel launches and memory transfers
4. **PyTorch Integration**: Allows seamless use of Triton kernels in PyTorch models

### Compilation Pipeline

```
Triton DSL (Python) → IR Generation → Optimization Passes → LLVM IR → GPU Assembly → Executable Kernel
```

### Memory Hierarchy

Triton abstracts the GPU memory hierarchy into:

- **Global Memory**: Main GPU memory (accessible by all threads)
- **Shared Memory**: Fast memory shared within a block
- **Registers**: Per-thread private memory

## Basic Usage

### Writing Your First Triton Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start
    block_start = pid * BLOCK_SIZE
    # Offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle case where the block size doesn't divide the number of elements
    mask = offsets < n_elements
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Add x and y
    output = x + y
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x, y):
    # Check input dimensions
    assert x.shape == y.shape, "Input shapes must match"
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    
    # Output tensor
    output = torch.empty_like(x)
    
    # Get tensor dimensions
    n_elements = output.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_kernel[grid, BLOCK_SIZE](
        x, y, output, n_elements, BLOCK_SIZE
    )
    
    return output
```

### Key Concepts

1. **@triton.jit**: Decorator that marks a function for JIT compilation
2. **tl.program_id**: Returns the ID of the current program inuserce
3. **tl.load/tl.store**: Load from and store to GPU memory
4. **tl.constexpr**: Compile-time conusert parameter
5. **Kernel Launch**: Using square brackets notation `kernel[grid, block](...)`

## Advanced Features

### Auto-Tuning

Triton provides auto-tuning capabilities to find optimal kernel configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(...):
    # Kernel implementation
    ...
```

### Tensor Core Acceleration

For operations that can benefit from Tensor Cores:

```python
@triton.jit
def matmul_kernel(...):
    # Use tensor cores for matrix multiplication
    c = tl.dot(a, b)
    ...
```

### Memory Optimizations

Triton provides several memory optimization primitives:

```python
# Software pipelining
a = tl.load(a_ptr + offsets)
b = tl.load(b_ptr + offsets)

# Prefetching
tl.prefetch(c_ptr + offsets, 2)

# Shared memory
a_shared = tl.load(a_ptr + offsets)
tl.store(shared_ptr + offsets, a_shared)
```

## Performance Optimization

### Key Optimization Strategies

1. **Block Size Selection**: Choose appropriate block sizes for your workload
2. **Memory Access Patterns**: Ensure coalesced memory access
3. **Occupancy Optimization**: Balance resource usage to maximize GPU occupancy
4. **Instruction-Level Parallelism**: Interleave independent operations
5. **Software Pipelining**: Overlap memory operations with computation

### Profiling Triton Kernels

```python
# Enable profiling
torch.cuda.cudart().cudaProfilerStart()

# Run your kernel
result = my_triton_kernel(...)

# Stop profiling
torch.cuda.cudart().cudaProfilerStop()
```

### Common Bottlenecks

1. **Memory Bandwidth**: Optimize memory access patterns
2. **Instruction Throughput**: Reduce instruction count
3. **Synchronization**: Minimize barriers and synchronization points
4. **Register Pressure**: Reduce register usage for higher occupancy

## Integration with PyTorch

### Using Triton Kernels in PyTorch Modules

```python
class TritonModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Module initialization
        
    def forward(self, x, y):
        # Use Triton kernel
        return my_triton_kernel(x, y)
```

### Custom Autograd Functions

```python
class TritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return triton_forward_kernel(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = triton_backward_kernel_x(grad_output, y)
        grad_y = triton_backward_kernel_y(grad_output, x)
        return grad_x, grad_y
```

## Debugging and Troubleshooting

### Common Issues with AMD GPUs

1. **Compiler Errors**: Ensure ROCm version compatibility
2. **Memory Errors**: Check for out-of-bounds access
3. **Performance Issues**: Verify kernel configurations
4. **Numerical Precision**: Be aware of differences in floating-point behavior

### Debugging Techniques

1. **Print Debugging**: Use `tl.debug_barrier()` and `print` statements
2. **Kernel Validation**: Compare results with CPU implementation
3. **Incremental Development**: Start with simple kernels and add complexity
4. **Error Checking**: Validate inputs and outputs

## Benchmarking

### Benchmarking Against PyTorch

```python
import time
import torch
import triton

def benchmark(fn, args, n_runs=100):
    # Warm-up
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_runs):
        fn(*args)
    torch.cuda.synchronize()
    
    return (time.time() - start_time) / n_runs

# Benchmark Triton kernel
triton_time = benchmark(triton_fn, (x, y))

# Benchmark PyTorch equivalent
torch_time = benchmark(torch_fn, (x, y))

print(f"Triton: {triton_time*1000:.3f} ms")
print(f"PyTorch: {torch_time*1000:.3f} ms")
print(f"Speedup: {torch_time/triton_time:.2f}x")
```

### Performance Metrics

1. **Execution Time**: Milliseconds per operation
2. **Throughput**: Operations per second
3. **Memory Bandwidth**: GB/s utilized
4. **Compute Utilization**: Percentage of theoretical FLOPS
5. **Occupancy**: Percentage of maximum possible warps

## Examples

### Matrix Multiplication

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication kernel implementation
    # ...
```

### Layer Normalization

```python
@triton.jit
def layer_norm_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_x_batch, stride_x_m,
    n_elements, eps,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Layer normalization kernel implementation
    # ...
```

### Attention Mechanism

```python
@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Attention mechanism kernel implementation
    # ...
```

## References

1. [Triton GitHub Repository](https://github.com/openai/triton)
2. [Triton Documentation](https://triton-lang.org/)
3. [ROCm Documentation](https://rocm.docs.amd.com/)
4. [PyTorch ROCm Support](https://pytorch.org/docs/stable/notes/hip.html)
5. [OpenAI Triton Blog Post](https://openai.com/blog/triton/)
6. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
7. [LLVM Documentation](https://llvm.org/docs/)
8. [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

