# ML Stack Master Guide

```
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**A Comprehensive Machine Learning Stack for AMD GPUs**

## Overview

### Introduction

Welcome to the ML Stack, a comprehensive machine learning ecosystem designed specifically for AMD GPUs. This guide provides detailed information on all components of the stack, their installation, configuration, and usage.

The ML Stack is optimized for AMD Radeon RX 7900 XTX and RX 7800 XT GPUs, providing high-performance tools and frameworks for machine learning research and development. It integrates core deep learning frameworks, optimization libraries, and extension components to create a seamless experience for machine learning on AMD hardware.

### Purpose

The purpose of this ML Stack is to:

1. **Provide a Complete Ecosystem**: Offer all necessary components for machine learning on AMD GPUs
2. **Optimize Performance**: Ensure maximum performance from AMD hardware
3. **Simplify Development**: Make it easy to develop and deploy machine learning models
4. **Enable Research**: Support cutting-edge research with high-performance tools
5. **Offer Flexibility**: Allow customization for different use cases

### Components

The ML Stack consists of two main categories of components:

#### Core Components

These form the foundation of the ML Stack:

- **ROCm Platform**: The foundation for GPU computing on AMD hardware
- **PyTorch**: Deep learning framework with ROCm support
- **ONNX Runtime**: Optimized inference for ONNX models
- **MIGraphX**: AMD's graph optimization library
- **Megatron-LM**: Framework for training large language models
- **Flash Attention**: Efficient attention computation
- **RCCL**: Collective communication library for multi-GPU training
- **MPI**: Message Passing Interface for distributed computing

#### Extension Components

These provide additional functionality:

- **Triton**: Compiler for parallel programming
- **BITSANDBYTES**: Efficient quantization for deep learning models
- **vLLM**: High-throughput inference engine for LLMs
- **ROCm SMI**: System monitoring and management for AMD GPUs
- **PyTorch Profiler**: Performance analysis for PyTorch models
- **Weights & Biases**: Experiment tracking and visualization

### Architecture

The ML Stack is organized in a layered architecture:

1. **Hardware Layer**: AMD GPUs (RX 7900 XTX, RX 7800 XT)
2. **System Layer**: ROCm platform and drivers
3. **Framework Layer**: PyTorch, ONNX Runtime, MIGraphX
4. **Model Layer**: Megatron-LM, Flash Attention
5. **Communication Layer**: RCCL, MPI
6. **Extension Layer**: Triton, BITSANDBYTES, vLLM, ROCm SMI, PyTorch Profiler, WandB

This layered approach allows for flexibility and customization while ensuring all components work together seamlessly.
## Core Components

The core components form the foundation of the ML stack, providing essential functionality for machine learning on AMD GPUs.

### ROCm Platform

The Radeon Open Compute (ROCm) platform is the foundation for GPU computing on AMD hardware.

- **Version**: ROCm 6.3/6.4
- **Key Features**: 
  - HIP programming model (CUDA compatibility layer)
  - Performance libraries (rocBLAS, MIOpen, etc.)
  - Tools and utilities for GPU management
- **Documentation**: [ROCm Guide](/docs/core/rocm_installation_guide.md)
- **Installation Script**: `/scripts/install_rocm.sh`
- **Use Cases**: All GPU computing tasks

### PyTorch

PyTorch is the primary deep learning framework, with ROCm support for AMD GPUs.

- **Version**: PyTorch 2.6.0+rocm6.2.4
- **Key Features**: 
  - Dynamic computation graph
  - CUDA compatibility layer
  - Optimized kernels for AMD GPUs
  - Comprehensive neural network library
- **Documentation**: [PyTorch Guide](/docs/core/pytorch_rocm_guide.md)
- **Installation Script**: `/scripts/install_pytorch.sh`
- **Use Cases**: Neural network training and inference

### ONNX Runtime

ONNX Runtime provides optimized inference for ONNX models on AMD GPUs.

- **Version**: Latest with ROCm support
- **Key Features**: 
  - Model optimization
  - Cross-platform compatibility
  - ROCm backend for AMD GPUs
  - Quantization support
- **Documentation**: [ONNX Runtime Guide](/docs/core/onnxruntime_build_guide.md)
- **Installation Script**: `/scripts/build_onnxruntime.sh`
- **Use Cases**: Model inference, model deployment

### MIGraphX

MIGraphX is AMD's graph optimization library for deep learning.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - Graph optimization
  - Operator fusion
  - Quantization support
  - Performance tuning
- **Documentation**: [MIGraphX Guide](/docs/core/migraphx_build_guide.md)
- **Installation Script**: `/scripts/install_migraphx.sh`
- **Use Cases**: Model optimization, inference acceleration

### Megatron-LM

Megatron-LM is a framework for training large language models, modified for AMD GPUs.

- **Version**: Custom fork with AMD support
- **Key Features**: 
  - Model parallelism
  - Distributed training
  - Optimized for large language models
  - Memory optimization techniques
- **Documentation**: [Megatron-LM Guide](/docs/core/megatron_adaptation_guide.md)
- **Installation Script**: `/scripts/install_megatron.sh`
- **Use Cases**: Large language model training

### Flash Attention

Flash Attention provides efficient attention computation for transformer models.

- **Version**: Flash Attention 2 (v2.6)
- **Key Features**: 
  - Memory-efficient attention
  - Optimized for AMD GPUs
  - Significant speedup for transformer models
  - Support for different attention patterns
- **Documentation**: [Flash Attention Guide](/docs/core/flash_attention_build_guide.md)
- **Installation Script**: `/scripts/build_flash_attn_amd.sh`
- **Use Cases**: Transformer model training and inference

### RCCL

ROCm Collective Communication Library (RCCL) enables multi-GPU communication.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - Collective operations for distributed training
  - Optimized for AMD GPU interconnect
  - NCCL-compatible API
  - Support for various communication patterns
- **Documentation**: [RCCL Guide](/docs/core/rccl_build_guide.md)
- **Installation Script**: `/scripts/install_rccl.sh`
- **Use Cases**: Multi-GPU training, distributed computing

### MPI

Message Passing Interface (MPI) provides distributed computing capabilities.

- **Version**: OpenMPI with ROCm support
- **Key Features**: 
  - Process communication
  - Job management
  - Integration with RCCL
  - Scalable distributed computing
- **Documentation**: [MPI Guide](/docs/core/mpi_installation_guide.md)
- **Installation Script**: `/scripts/install_mpi.sh`
- **Use Cases**: Distributed training, multi-node computing
## Extension Components

The extension components enhance the core stack with additional functionality.

### Triton

Triton is an open-source language and compiler for parallel programming that can generate highly optimized GPU kernels.

- **Version**: 2.2.0
- **Key Features**: 
  - Python-based DSL for GPU programming
  - Automatic optimization of kernels
  - Integration with PyTorch
  - Support for AMD GPUs through ROCm
- **Documentation**: [Triton Guide](/docs/extensions/triton_guide.md)
- **Installation Script**: `/scripts/install_triton.sh`
- **Use Cases**: Custom operators, performance-critical operations

### BITSANDBYTES

BITSANDBYTES provides efficient quantization for deep learning models, reducing memory usage and improving inference speed.

- **Version**: 0.41.3
- **Key Features**: 
  - 8-bit quantization for linear layers
  - 4-bit quantization for weights
  - Memory-efficient optimizers
  - ROCm support for AMD GPUs
- **Documentation**: [BITSANDBYTES Guide](/docs/extensions/bitsandbytes_guide.md)
- **Installation Script**: `/scripts/install_bitsandbytes.sh`
- **Use Cases**: Large model inference, memory-constrained environments

### vLLM

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs).

- **Version**: 0.3.0
- **Key Features**: 
  - PagedAttention for memory efficiency
  - Continuous batching for high throughput
  - Tensor parallelism for multi-GPU inference
  - ROCm support for AMD GPUs
- **Documentation**: [vLLM Guide](/docs/extensions/vllm_guide.md)
- **Installation Script**: `/scripts/install_vllm.sh`
- **Use Cases**: LLM inference, model serving

### ROCm SMI

ROCm System Management Interface (ROCm SMI) provides monitoring and management capabilities for AMD GPUs.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - GPU monitoring (utilization, temperature, memory)
  - Performance control (clock speeds, power limits)
  - Python API for programmatic access
  - Integration with ML workflows
- **Documentation**: [ROCm SMI Guide](/docs/extensions/rocm_smi_guide.md)
- **Installation Script**: `/scripts/install_rocm_smi.sh`
- **Use Cases**: Performance monitoring, thermal management

### PyTorch Profiler

PyTorch Profiler provides detailed performance analysis for PyTorch models.

- **Version**: Included with PyTorch
- **Key Features**: 
  - Operator-level profiling
  - Memory profiling
  - TensorBoard integration
  - ROCm support for AMD GPUs
- **Documentation**: [PyTorch Profiler Guide](/docs/extensions/pytorch_profiler_guide.md)
- **Installation Script**: `/scripts/install_pytorch_profiler.sh`
- **Use Cases**: Performance optimization, bottleneck identification

### Weights & Biases (WandB)

Weights & Biases (WandB) is an experiment tracking platform for machine learning.

- **Version**: Latest
- **Key Features**: 
  - Experiment tracking
  - Hyperparameter optimization
  - Artifact management
  - Visualization
  - Team collaboration
- **Documentation**: [WandB Guide](/docs/extensions/wandb_guide.md)
- **Installation Script**: `/scripts/install_wandb.sh`
- **Use Cases**: Experiment tracking, collaboration, visualization
## Installation

The ML stack can be installed using the provided installation scripts.

### Prerequisites

Before installation, ensure you have:

1. **ROCm Installed**: ROCm 6.3/6.4 should be installed and working
2. **Python Environment**: Python 3.8+ with pip
3. **System Dependencies**: Required system libraries and tools
4. **GPU Access**: Proper permissions to access AMD GPUs

### Core Components Installation

The core components should be installed first:

1. **ROCm**: Follow the [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
2. **PyTorch**: Install with ROCm support
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```
3. **ONNX Runtime**: Build from source with ROCm support
4. **MIGraphX**: Install from ROCm repositories
5. **Megatron-LM**: Clone and install the AMD-compatible fork
6. **Flash Attention**: Install the AMD-compatible version
7. **RCCL**: Install from ROCm repositories
8. **MPI**: Install OpenMPI with ROCm support

### Extension Components Installation

The extension components can be installed using the master installation script:

```bash
# Make the script executable
chmod +x $HOME/Desktop/Stans_MLStack/scripts/install_ml_stack_extensions.sh

# Run the installation script
$HOME/Desktop/Stans_MLStack/scripts/install_ml_stack_extensions.sh
```

The script will prompt you to select which components to install:

1. Triton - Compiler for parallel programming
2. BITSANDBYTES - Efficient quantization
3. vLLM - High-throughput inference engine
4. ROCm SMI - Monitoring and profiling
5. PyTorch Profiler - Performance analysis
6. WandB - Experiment tracking
7. All components

### Individual Component Installation

You can also install individual components using their respective installation scripts:

```bash
# Install Triton
$HOME/Desktop/Stans_MLStack/scripts/install_triton.sh

# Install BITSANDBYTES
$HOME/Desktop/Stans_MLStack/scripts/install_bitsandbytes.sh

# Install vLLM
$HOME/Desktop/Stans_MLStack/scripts/install_vllm.sh

# Install ROCm SMI
$HOME/Desktop/Stans_MLStack/scripts/install_rocm_smi.sh

# Install PyTorch Profiler
$HOME/Desktop/Stans_MLStack/scripts/install_pytorch_profiler.sh

# Install WandB
$HOME/Desktop/Stans_MLStack/scripts/install_wandb.sh
```

### Verification

After installation, verify that the components are working correctly:

```bash
# Verify PyTorch with ROCm
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# Verify ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.__version__); print(onnxruntime.get_device())"

# Verify MIGraphX
python -c "import migraphx; print(migraphx.__version__)"

# Verify Triton
python -c "import triton; print(triton.__version__)"

# Verify BITSANDBYTES
python -c "import bitsandbytes as bnb; print(bnb.__version__); print(bnb.CUDA_AVAILABLE)"

# Verify vLLM
python -c "import vllm; print(vllm.__version__)"

# Verify ROCm SMI
python -c "from rocm_smi_lib import rsmi; print('ROCm SMI available')"

# Verify WandB
python -c "import wandb; print(wandb.__version__)"
```
## Configuration

Proper configuration is essential for optimal performance of the ML stack on AMD GPUs.

### Environment Variables

Set these environment variables for optimal performance:

```bash
# GPU Selection
export HIP_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1  # For CUDA compatibility layer
export PYTORCH_ROCM_DEVICE=0,1  # For PyTorch

# Memory Management
export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance
export GPU_MAX_HEAP_SIZE=100  # Set maximum heap size (%)
export GPU_MAX_ALLOC_PERCENT=100  # Set maximum allocation size (%)
export HSA_TOOLS_LIB=1  # Enable HSA tools library

# Performance
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1  # Use implicit GEMM for convolutions
export MIOPEN_FIND_MODE=3  # Aggressive kernel search
export MIOPEN_FIND_ENFORCE=3  # Enforce kernel search

# Logging
export HIP_TRACE_API=0  # Disable API tracing for production
export AMD_LOG_LEVEL=4  # Set log level (0-4)
```

### PyTorch Configuration

Configure PyTorch for optimal performance:

```python
import torch

# Set memory split size for large operations
torch.cuda.max_split_size_mb = 512  # Optimal for RX 7900 XTX

# Set default tensor type
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Enable TF32 (tensor float 32) for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set benchmark mode for optimal performance
torch.backends.cudnn.benchmark = True

# Set deterministic mode for reproducibility (if needed)
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
```

### Multi-GPU Configuration

Configure multi-GPU training:

```python
import torch.distributed as dist
import os

# Initialize process group
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group("nccl", rank=0, world_size=2)

# Set device
torch.cuda.set_device(0)  # Use GPU 0 for this process

# Create DistributedDataParallel model
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[0])
```

### Memory Optimization

Configure memory optimization techniques:

```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient accumulation
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    with autocast():
        output = model(input)
        loss = criterion(output, target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Component-Specific Configuration

Each component may require specific configuration:

1. **Triton**: Set compilation options for AMD GPUs
2. **BITSANDBYTES**: Configure quantization parameters
3. **vLLM**: Set PagedAttention and batching parameters
4. **ROCm SMI**: Configure monitoring intervals
5. **PyTorch Profiler**: Set profiling activities and schedules
6. **WandB**: Configure project and experiment settings

Refer to the individual component documentation for detailed configuration options.
## Usage Examples

These examples demonstrate how to use the ML stack for common tasks.

### Basic PyTorch Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Initialize WandB
wandb.init(project="amd-gpu-example")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 16 * 16, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc(x)
        return x

# Create model and move to GPU
model = SimpleModel().to(device)

# Log model architecture
wandb.watch(model, log="all")

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
batch_size = 64
x = torch.randn(batch_size, 3, 32, 32, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / batch_size
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "accuracy": accuracy
    })
    
    # Print progress
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
wandb.save("model.pt")

# Finish WandB run
wandb.finish()
```

### Distributed Training with RCCL and MPI

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    # Setup process group
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Create model
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Create dummy data
    batch_size = 64
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = ddp_model(x)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress on rank 0
        if rank == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    # Cleanup
    cleanup()

# Run distributed training
world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### Quantization with BITSANDBYTES

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Create a standard model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024)
).to("cuda")

# Convert to 8-bit model
model_8bit = nn.Sequential(
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
    nn.ReLU(),
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
    nn.ReLU(),
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False)
).to("cuda")

# Copy weights from original model to 8-bit model
for i in range(0, len(model), 2):
    model_8bit[i].weight.data = model[i].weight.data
    model_8bit[i].bias.data = model[i].bias.data

# Create input data
x = torch.randn(32, 1024, device="cuda")

# Compare outputs
with torch.no_grad():
    output_fp32 = model(x)
    output_int8 = model_8bit(x)
    
    # Check error
    error = torch.abs(output_fp32 - output_int8).mean()
    print(f"Mean absolute error: {error.item()}")
    
    # Check memory usage
    print(f"FP32 model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.2f} MB")
    print(f"8-bit model size: {sum(p.numel() * (1 if '8bit' in p.__class__.__name__ else 4) for p in model_8bit.parameters()) / 1024 / 1024:.2f} MB")
```

### LLM Inference with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-1.3b")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The best programming language is"
]

# Generate completions
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
```

### Custom Kernels with Triton

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
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

# Test the kernel
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
output = add_vectors(x, y)
print(f"Max error: {torch.max(torch.abs(output - (x + y)))}")
```

### Profiling with PyTorch Profiler

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Create model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024)
).to("cuda")

# Create input data
x = torch.randn(32, 1024, device="cuda")

# Warm-up
for _ in range(5):
    model(x)

# Profile with PyTorch Profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        model(x)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export trace
prof.export_chrome_trace("trace.json")
```

### GPU Monitoring with ROCm SMI

```python
from rocm_smi_lib import rsmi
import time

# Initialize ROCm SMI
rsmi.rsmi_init(0)

try:
    # Get number of devices
    num_devices = rsmi.rsmi_num_monitor_devices()
    print(f"Found {num_devices} GPU device(s)")
    
    # Monitor GPUs
    for i in range(5):  # Monitor for 5 iterations
        print(f"\nIteration {i+1}/5")
        
        for device_id in range(num_devices):
            # Get device name
            name = rsmi.rsmi_dev_name_get(device_id)[1]
            print(f"\nGPU {device_id}: {name}")
            
            # Get GPU utilization
            util = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
            print(f"  Utilization: {util}%")
            
            # Get temperature
            temp = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C
            print(f"  Temperature: {temp}°C")
            
            # Get memory usage
            mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
            mem_used = mem_info[1] / (1024 * 1024)  # Convert to MB
            mem_total = mem_info[2] / (1024 * 1024)  # Convert to MB
            print(f"  Memory: {mem_used:.2f}/{mem_total:.2f} MB ({(mem_used/mem_total)*100:.2f}%)")
            
            # Get power consumption
            power = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W
            print(f"  Power: {power:.2f} W")
        
        # Wait before next iteration
        time.sleep(1)

finally:
    # Clean up
    rsmi.rsmi_shut_down()
```
## Performance Optimization

Optimizing performance is crucial for getting the most out of AMD GPUs.

### Hardware Optimization

1. **GPU Selection**: Use the most powerful GPU (RX 7900 XTX) as the primary device
2. **Cooling**: Ensure adequate cooling for sustained performance
3. **Power Supply**: Provide sufficient power for peak performance
4. **PCIe Configuration**: Use PCIe 4.0 or higher slots with x16 lanes
5. **System Memory**: Use high-speed RAM with sufficient capacity

### Memory Optimization

1. **Batch Size Optimization**: Find the optimal batch size for your GPU memory
   ```python
   # Test different batch sizes
   for batch_size in [16, 32, 64, 128, 256]:
       try:
           x = torch.randn(batch_size, 3, 224, 224, device="cuda")
           y = model(x)
           print(f"Batch size {batch_size} works")
       except RuntimeError as e:
           print(f"Batch size {batch_size} failed: {e}")
           break
   ```

2. **Mixed Precision Training**: Use FP16 or BF16 for reduced memory usage
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Gradient Checkpointing**: Trade computation for memory
   ```python
   from torch.utils.checkpoint import checkpoint
   
   # Use checkpointing for memory-intensive layers
   output = checkpoint(model.expensive_layer, input)
   ```

4. **Memory Fragmentation**: Clear cache periodically
   ```python
   # Clear cache
   torch.cuda.empty_cache()
   ```

### Computation Optimization

1. **Kernel Selection**: Use optimized kernels for AMD GPUs
   ```python
   # Set environment variables for kernel selection
   os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
   os.environ["MIOPEN_FIND_MODE"] = "3"
   ```

2. **Operator Fusion**: Fuse operations when possible
   ```python
   # Use fused operations
   nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
   ```

3. **Custom Kernels**: Use Triton for custom kernels
   ```python
   # See Triton example in Usage Examples section
   ```

4. **Quantization**: Use BITSANDBYTES for quantization
   ```python
   # See BITSANDBYTES example in Usage Examples section
   ```

### Distributed Training Optimization

1. **Data Parallelism**: Use DistributedDataParallel for multi-GPU training
   ```python
   # See distributed training example in Usage Examples section
   ```

2. **NCCL Tuning**: Optimize NCCL parameters for AMD GPUs
   ```python
   # Set NCCL parameters
   os.environ["NCCL_DEBUG"] = "INFO"
   os.environ["NCCL_IB_DISABLE"] = "1"
   os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
   ```

3. **Gradient Accumulation**: Accumulate gradients for larger effective batch sizes
   ```python
   # Accumulate gradients
   for i, (input, target) in enumerate(dataloader):
       output = model(input)
       loss = criterion(output, target) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Model Optimization

1. **Architecture Optimization**: Choose architectures that work well on AMD GPUs
   ```python
   # Use efficient attention mechanisms
   from flash_attention_amd import FlashAttention
   attention = FlashAttention()
   ```

2. **Activation Functions**: Use efficient activation functions
   ```python
   # Use efficient activation functions
   nn.SiLU()  # Swish/SiLU is efficient on AMD GPUs
   ```

3. **Model Pruning**: Reduce model size through pruning
   ```python
   # Prune model
   from torch.nn.utils import prune
   prune.l1_unstructured(module, name="weight", amount=0.2)
   ```

4. **Knowledge Distillation**: Distill large models into smaller ones
   ```python
   # Knowledge distillation
   teacher_output = teacher_model(input)
   student_output = student_model(input)
   distillation_loss = nn.KLDivLoss()(
       F.log_softmax(student_output / temperature, dim=1),
       F.softmax(teacher_output / temperature, dim=1)
   ) * (temperature * temperature)
   ```
## Troubleshooting

Common issues and their solutions.

### GPU Detection Issues

1. **GPU Not Detected**:
   ```
   No CUDA GPUs are available
   ```
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
   - Check permissions: `groups` (should include video or render)
   - Update drivers: `sudo apt update && sudo apt upgrade`

2. **Multiple GPUs Not Detected**:
   ```
   Only one GPU is visible
   ```
   
   Solutions:
   - Set environment variables: `export HIP_VISIBLE_DEVICES=0,1`
   - Check PCIe configuration: `lspci | grep -i amd`
   - Verify ROCm multi-GPU support: `rocminfo`

### Memory Issues

1. **Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce batch size
   - Use mixed precision training
   - Use gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`
   - Monitor memory usage: `torch.cuda.memory_summary()`

2. **Memory Fragmentation**:
   ```
   RuntimeError: CUDA out of memory (fragmented memory)
   ```
   
   Solutions:
   - Clear cache periodically
   - Allocate tensors in order of size (largest first)
   - Use persistent RNN for recurrent models
   - Restart the process if fragmentation is severe

### Performance Issues

1. **Slow Training**:
   ```
   Training is slower than expected
   ```
   
   Solutions:
   - Profile with PyTorch Profiler
   - Check GPU utilization with ROCm SMI
   - Optimize data loading (more workers, pin_memory)
   - Use optimized kernels and operations
   - Check for CPU bottlenecks

2. **Low GPU Utilization**:
   ```
   GPU utilization is below 50%
   ```
   
   Solutions:
   - Increase batch size
   - Use more workers for data loading
   - Check for CPU bottlenecks
   - Use non-blocking transfers: `tensor.to('cuda', non_blocking=True)`
   - Profile with PyTorch Profiler to identify bottlenecks

### Compatibility Issues

1. **PyTorch Version Mismatch**:
   ```
   ImportError: cannot import name 'X' from 'torch'
   ```
   
   Solutions:
   - Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Install compatible version: `pip install torch==X.Y.Z+rocmA.B.C`
   - Check ROCm compatibility: `rocminfo`

2. **CUDA Compatibility Layer Issues**:
   ```
   AttributeError: module 'torch.cuda' has no attribute 'X'
   ```
   
   Solutions:
   - Use ROCm-specific APIs when available
   - Check PyTorch ROCm documentation
   - Update to latest PyTorch version with ROCm support

### Extension Component Issues

1. **Triton Compilation Errors**:
   ```
   Error: Failed to compile kernel
   ```
   
   Solutions:
   - Check ROCm version compatibility
   - Use simpler kernel implementations
   - Check for AMD-specific limitations

2. **BITSANDBYTES Quantization Errors**:
   ```
   RuntimeError: CUDA error: device-side assert triggered
   ```
   
   Solutions:
   - Check model compatibility with 8-bit quantization
   - Use different quantization parameters
   - Try different quantization methods

3. **vLLM Memory Errors**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce model size or sequence length
   - Adjust PagedAttention parameters
   - Use tensor parallelism across multiple GPUs

## References

1. [ROCm Documentation](https://rocm.docs.amd.com/)
2. [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
3. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
4. [ONNX Runtime Documentation](https://onnxruntime.ai/)
5. [MIGraphX Documentation](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
6. [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
7. [Flash Attention Documentation](https://github.com/Dao-AILab/flash-attention)
8. [Triton Documentation](/docs/extensions/triton_guide.md)
9. [BITSANDBYTES Documentation](/docs/extensions/bitsandbytes_guide.md)
10. [vLLM Documentation](/docs/extensions/vllm_guide.md)
11. [ROCm SMI Documentation](/docs/extensions/rocm_smi_guide.md)
12. [PyTorch Profiler Documentation](/docs/extensions/pytorch_profiler_guide.md)
13. [WandB Documentation](/docs/extensions/wandb_guide.md)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

