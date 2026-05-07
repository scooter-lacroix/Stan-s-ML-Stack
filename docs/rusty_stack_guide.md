# Rusty Stack Guide

> **Rusty Stack** (formerly Stan's ML Stack) is a comprehensive machine learning environment optimized for AMD GPUs with ROCm support. It provides a unified Rust CLI/TUI for installing, configuring, updating, and benchmarking the full ML stack.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Extension Components](#extension-components)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

Rusty Stack provides a complete machine learning ecosystem designed specifically for AMD GPUs. It integrates core deep learning frameworks, optimization libraries, and extension components into a single, unified toolchain managed through a native Rust CLI.

### What It Does

1. **Install** -- One-command installation of 30+ ML components optimized for AMD ROCm
2. **Update** -- Scan, plan, and apply component updates with risk classification
3. **Verify** -- Validate installation health across all components
4. **Benchmark** -- Run GPU, memory, and ML performance benchmarks with before/after comparison
5. **Repair** -- Automatically detect and reinstall broken components

### Supported Hardware

- AMD RDNA 2 (RX 6800 XT, RX 6900 XT)
- AMD RDNA 3 (RX 7900 XTX, RX 7800 XT)
- AMD RDNA 4 (RX 9070 XT)

### ROCm Channels

Three ROCm channels are available, selectable via the TUI or `INSTALL_ROCM_PRESEEDED_CHOICE` environment variable:

| Channel | ROCm Version | Use Case |
|---------|-------------|----------|
| **Legacy** | 6.4.3 | Production-proven stability |
| **Stable** | 7.1 | Production-ready for RDNA 3 |
| **Latest** | 7.2.1 | Default, expanded RDNA 4 support |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/scooter-lacroix/Stan-s-ML-Stack.git
cd Stan-s-ML-Stack

# Build the Rust CLI
cd rusty-stack && cargo build --release

# Launch the interactive TUI installer
./target/release/rusty

# Or use CLI subcommands directly
./target/release/rusty update --scan-only   # Check for updates
./target/release/rusty verify --full        # Verify installation
./target/release/rusty bench --all          # Run benchmarks
./target/release/rusty deps                 # Check Rust dependency updates
```

### One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash
```

---

## Architecture

Rusty Stack uses a fully native Rust architecture -- no shell subprocesses for installation. All 34 installer components are pure Rust modules.

```
┌─────────────────────────────────────────────────────┐
│                 rusty CLI / TUI                      │
│           (clap subcommands + ratatui)               │
├─────────────────────────────────────────────────────┤
│  update    │  upgrade  │  bench  │  verify  │  deps │
├─────────────────────────────────────────────────────┤
│              Native Rust Installer Engine            │
│         34 components, zero shell scripts            │
├──────────┬──────────┬──────────┬────────────────────┤
│ Platform │ Core     │ Orchestr.│ Verification        │
│ Detect   │ Types    │ Planner │ Engine               │
│ Registry │ Manifest │ Apply   │ Status               │
│ Linux    │ Config   │ Verify  │ Repair               │
└──────────┴──────────┴──────────┴────────────────────┘
```

### Directory Layout

```
rusty-stack/
├── src/
│   ├── main.rs              # TUI entry point
│   ├── lib.rs               # Library crate root
│   ├── bin/rusty.rs         # Unified CLI (all subcommands)
│   ├── installer.rs         # Native installer dispatch
│   ├── app.rs               # TUI state machine
│   ├── installers/
│   │   ├── common/          # Shared infra (pkg manager, distro, guard, utils)
│   │   └── components/      # Per-component installers (34 modules)
│   ├── verification/        # Native verification engine
│   ├── benchmark_runners/   # Native benchmark runners
│   ├── platform/            # Hardware/distro detection, component registry
│   ├── orchestrator/        # Update planner, apply engine, verify runner
│   ├── core/                # Shared types, manifest schema, validation
│   ├── adapter/             # Rust and legacy script executors
│   ├── telemetry/           # Stability benchmark, anonymous payload
│   ├── bootstrap/           # install.sh equivalents in native Rust
│   └── widgets/             # TUI UI components
├── Cargo.toml
└── tests/                   # Integration tests (CLI, upgrade, verify, bench)
```

---

## Core Components

The core components form the foundation of the ML stack.

### ROCm Platform

The Radeon Open Compute (ROCm) platform is the foundation for GPU computing on AMD hardware.

- **Versions**: 6.4.3 (Legacy), 7.1 (Stable), 7.2.1 (Latest)
- **Key Features**:
  - HIP programming model (CUDA compatibility layer)
  - Performance libraries (rocBLAS, MIOpen, rccl, hipFFT, etc.)
  - Tools and utilities for GPU management
- **Component ID**: `rocm`
- **Use Cases**: All GPU computing tasks

### PyTorch

PyTorch is the primary deep learning framework with ROCm support for AMD GPUs.

- **Key Features**:
  - Dynamic computation graph
  - CUDA compatibility layer via HIP
  - Optimized kernels for AMD GPUs
  - Comprehensive neural network library
- **Component ID**: `pytorch`
- **Use Cases**: Neural network training and inference

### ONNX Runtime

ONNX Runtime provides optimized inference for ONNX models on AMD GPUs.

- **Key Features**:
  - Model optimization
  - Cross-platform compatibility
  - ROCm backend for AMD GPUs
  - Quantization support
- **Component ID**: `onnxruntime`
- **Use Cases**: Model inference, model deployment

### MIGraphX

MIGraphX is AMD's graph optimization library for deep learning inference.

- **Key Features**:
  - Graph optimization and operator fusion
  - Quantization support
  - Performance tuning for AMD hardware
- **Component ID**: `migraphx`
- **Use Cases**: Model optimization, inference acceleration

### Megatron-LM

Megatron-LM is a framework for training large language models, modified for AMD GPUs.

- **Key Features**:
  - Model parallelism (tensor, pipeline, expert)
  - Distributed training support
  - Optimized for large language models
  - Memory optimization techniques
- **Component ID**: `megatron`
- **Use Cases**: Large language model training

### Flash Attention

Flash Attention provides efficient attention computation for transformer models.

- **Key Features**:
  - Memory-efficient attention (IO-aware)
  - Optimized for AMD GPUs (CK backend for gfx11)
  - Significant speedup for transformer models
  - Support for different attention patterns
- **Component ID**: `flash_attention`
- **Use Cases**: Transformer model training and inference

### RCCL

ROCm Collective Communication Library enables multi-GPU communication.

- **Key Features**:
  - Collective operations for distributed training
  - Optimized for AMD GPU interconnect
  - NCCL-compatible API
- **Component ID**: `rccl`
- **Use Cases**: Multi-GPU training, distributed computing

### MPI

Message Passing Interface provides distributed computing capabilities.

- **Key Features**:
  - Process communication
  - Job management
  - Integration with RCCL
- **Component ID**: `mpi`
- **Use Cases**: Distributed training, multi-node computing

---

## Extension Components

Extension components enhance the core stack with additional functionality.

### Triton

Triton is an open-source language and compiler for parallel programming that generates highly optimized GPU kernels.

- **Key Features**:
  - Python-based DSL for GPU programming
  - Automatic kernel optimization
  - Integration with PyTorch
  - AMD GPU support through ROCm
- **Component ID**: `triton`
- **Use Cases**: Custom operators, performance-critical operations

### BITSANDBYTES

BITSANDBYTES provides efficient quantization for deep learning models.

- **Key Features**:
  - 8-bit quantization for linear layers
  - 4-bit quantization (QLoRA) for weights
  - Memory-efficient optimizers
  - ROCm support for AMD GPUs
- **Component ID**: `bitsandbytes`
- **Use Cases**: Large model inference, memory-constrained environments, QLoRA fine-tuning

### vLLM

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models.

- **Key Features**:
  - PagedAttention for memory efficiency
  - Continuous batching for high throughput
  - Tensor parallelism for multi-GPU inference
  - ROCm support for AMD GPUs
- **Component ID**: `vllm`
- **Use Cases**: LLM inference, model serving

### DeepSpeed

DeepSpeed is a distributed training framework for large-scale model training.

- **Key Features**:
  - ZeRO optimization stages 1-3
  - Pipeline parallelism
  - Mixed precision training
  - Gradient checkpointing
- **Component ID**: `deepspeed`
- **Use Cases**: Distributed training, large model training

### AITER

AITER (AMD Inferencing and Training Engine for ROCm) provides optimized kernels.

- **Key Features**:
  - Optimized attention kernels
  - Fusion kernels for AMD GPUs
  - Integration with PyTorch
- **Component ID**: `aiter`
- **Use Cases**: Kernel optimization, inference acceleration

### FastVideo

FastVideo provides accelerated video generation with ROCm support.

- **Key Features**:
  - ROCm-native build with gfx11 support
  - Builds from scooter-lacroix/FastVideo feature/rocm-gfx11-support branch
  - Native kernel compilation
- **Component ID**: `fastvideo`
- **Use Cases**: AI video generation

### ComfyUI (ROCm Edition)

ComfyUI is a node-based UI for AI image generation with full AMD GPU acceleration.

- **Key Features**:
  - Full ROCm GPU acceleration
  - Automatic torch dependency filtering
  - Model preservation during updates
  - ComfyUI Manager integration
- **Component ID**: `comfyui`
- **Run**: `comfy` (starts ComfyUI at http://localhost:8188)

### vLLM Studio

Web UI for vLLM model management and deployment.

- **Component ID**: `vllm_studio`
- **Run**: `vllm-studio`

### ROCm SMI

ROCm System Management Interface provides monitoring and management capabilities for AMD GPUs.

- **Key Features**:
  - GPU monitoring (utilization, temperature, memory)
  - Performance control (clock speeds, power limits)
  - Python API for programmatic access
- **Component ID**: `rocm_smi`
- **Use Cases**: Performance monitoring, thermal management

### PyTorch Profiler

PyTorch Profiler provides detailed performance analysis for PyTorch models.

- **Key Features**:
  - Operator-level profiling
  - Memory profiling
  - TensorBoard integration
  - ROCm support for AMD GPUs
- **Component ID**: `pytorch_profiler`
- **Use Cases**: Performance optimization, bottleneck identification

### Weights & Biases (WandB)

Weights & Biases is an experiment tracking platform for machine learning.

- **Key Features**:
  - Experiment tracking and visualization
  - Hyperparameter optimization
  - Artifact management
  - Team collaboration
- **Component ID**: `wandb`
- **Use Cases**: Experiment tracking, collaboration

### Text Generation WebUI (Oobabooga)

Text generation inference server with a web interface.

- **Key Features**:
  - Multiple model loader backends
  - ROCm/CUDA support
  - CUDA dependency filtering for AMD GPUs
  - Chat and notebook modes
- **Component ID**: `textgen`
- **Use Cases**: LLM inference with web UI

---

## Installation

### Prerequisites

Before installation, ensure you have:

1. **AMD GPU**: RDNA 2/3/4 GPU with proper drivers
2. **Linux System**: Arch, Ubuntu, or other supported distribution
3. **Rust Toolchain**: `rustc` + `cargo` for building from source
4. **Internet Access**: For downloading packages and source code

### Using the TUI Installer

The recommended way to install is via the interactive TUI:

```bash
cd rusty-stack && cargo build --release
./target/release/rusty
```

The TUI provides:
- Hardware detection with AMD GPU and ROCm awareness
- Preflight checks for disk, memory, GPU presence, and ROCm availability
- Component selection across foundation, core, extensions, UI/UX, environment, and verification categories
- Performance category for GPU benchmarking
- Configuration screen with batch mode, auto-confirm, theme, and performance profile toggles
- Live installation progress with captured logs
- Completion summary with install/failed/skipped breakdown

### Using the CLI

For non-interactive or scripted installations:

```bash
# Verify prerequisites
rusty verify --full

# Install specific components via the TUI in batch mode
# (Set MLSTACK_BATCH=1 and MLSTACK_AUTO_CONFIRM=1 for non-interactive)
MLSTACK_BATCH=1 MLSTACK_AUTO_CONFIRM=1 ./target/release/rusty
```

### Verification

After installation, verify all components:

```bash
# Full verification (core components)
rusty verify --full

# Enhanced verification (all components including Python imports, environment)
rusty verify --enhanced

# Verify and identify components needing rebuild
rusty verify --build

# JSON output for scripting
rusty verify --full --json
```

### Manual Verification

You can also verify individual components manually:

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
```

---

## Configuration

### Config Locations

| File | Purpose |
|------|---------|
| `~/.mlstack/config/config.json` | Main configuration |
| `~/.mlstack/logs/rusty-stack.log` | Runtime logs |
| `~/.mlstack_env` | Environment variables |
| `~/.mlstack/reports/` | Benchmark HTML reports |

### Environment Variables

Set these environment variables for optimal performance:

```bash
# GPU Selection
export HIP_VISIBLE_DEVICES=0,1           # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1          # For CUDA compatibility layer
export PYTORCH_ROCM_DEVICE=0,1           # For PyTorch

# Memory Management
export HSA_ENABLE_SDMA=0                 # Disable SDMA for better performance
export GPU_MAX_HEAP_SIZE=100             # Set maximum heap size (%)
export GPU_MAX_ALLOC_PERCENT=100         # Set maximum allocation size (%)

# Performance
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1 # Use implicit GEMM for convolutions
export MIOPEN_FIND_MODE=3                # Aggressive kernel search
export MIOPEN_FIND_ENFORCE=3             # Enforce kernel search

# Logging
export AMD_LOG_LEVEL=4                   # Set log level (0-4)
```

### PyTorch Configuration

```python
import torch

# Memory split for large operations
torch.cuda.max_split_size_mb = 512  # Optimal for RX 7900 XTX

# Enable TF32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Benchmark mode for optimal kernel selection
torch.backends.cudnn.benchmark = True
```

### Multi-GPU Configuration

```python
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group("nccl", rank=0, world_size=2)

from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[0])
```

### Memory Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)

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

---

## CLI Reference

### `rusty` (no arguments)

Launches the interactive TUI installer.

### `rusty update`

Scan, plan, and apply component updates.

```bash
rusty update                    # Interactive update
rusty update --scan-only        # Scan without applying changes
rusty update --all-safe         # Apply all safe updates
rusty update --yes              # Skip confirmation prompts
rusty update --json             # Machine-readable JSON output
rusty update --include-experimental  # Include experimental components
rusty update <component>        # Update specific component(s)
```

Updates are classified into risk tiers:
- **Safe**: Patch-level updates with no API changes
- **Guarded**: Minor version updates requiring compatibility checks
- **Blocked**: Major version updates or known breaking changes
- **Candidate**: Newly available components not yet installed
- **Experimental**: Cutting-edge components requiring explicit opt-in

### `rusty upgrade`

Upgrade the Rusty Stack binary itself.

```bash
rusty upgrade                   # Interactive upgrade
rusty upgrade --dry-run         # Check without applying
rusty upgrade --yes             # Skip confirmation
```

### `rusty bench`

Run performance benchmarks.

```bash
rusty bench                     # Interactive benchmark selection
rusty bench --all               # Run all benchmarks
rusty bench --rocm              # ROCm-specific benchmarks
rusty bench --json <name>       # JSON output for specific benchmark
rusty bench --list              # List available benchmarks
```

Available benchmarks:
- **Pre-installation**: `gpu-capability`, `memory-bandwidth`, `tensor-core`
- **GPU Performance**: `gemm`
- **Component**: `pytorch`, `flash-attention`, `vllm`, `deepspeed`, `megatron`
- **Combined**: `all-pre`, `all`

### `rusty verify`

Verify ML Stack installation.

```bash
rusty verify --full             # Core component verification
rusty verify --enhanced         # All-component verification
rusty verify --build            # Verify and identify rebuild targets
rusty verify --full --json      # JSON output
```

### `rusty deps`

Check for Rust dependency updates from crates.io.

```bash
rusty deps                      # Check all deps with 7-day lag
rusty deps --lag 14             # Use 14-day lag period
rusty deps --verbose            # Show full API responses
rusty deps --dir ./my-crate     # Check a specific crate directory
rusty deps --json               # JSON output
```

Exit codes: 0 = all up to date, 1 = updates available, 2 = error.

---

## Usage Examples

### Basic PyTorch Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

wandb.init(project="amd-gpu-example")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        return self.fc(x)

model = SimpleModel().to(device)
wandb.watch(model, log="all")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x = torch.randn(64, 3, 32, 32, device=device)
y = torch.randint(0, 10, (64,), device=device)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y).sum().item() / 64

    wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "model.pt")
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
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    x = torch.randn(64, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    cleanup()

world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### Quantization with BITSANDBYTES

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Standard FP32 model
model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
).to("cuda")

# 8-bit quantized model
model_8bit = nn.Sequential(
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
    nn.ReLU(),
    bnb.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False),
).to("cuda")

# Copy weights
for i in range(0, len(model), 2):
    model_8bit[i].weight.data = model[i].weight.data
    model_8bit[i].bias.data = model[i].bias.data

# Compare outputs
x = torch.randn(32, 1024, device="cuda")
with torch.no_grad():
    output_fp32 = model(x)
    output_int8 = model_8bit(x)
    error = torch.abs(output_fp32 - output_int8).mean()
    print(f"Mean absolute error: {error.item()}")
```

### LLM Inference with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-1.3b")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The best programming language is",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

### Custom Kernels with Triton

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add_vectors(x, y):
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n = output.numel()
    BLOCK_SIZE = 1024
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[grid, BLOCK_SIZE](x, y, output, n, BLOCK_SIZE)
    return output

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

model = nn.Sequential(
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Linear(1024, 1024)
).to("cuda")

x = torch.randn(32, 1024, device="cuda")
for _ in range(5):
    model(x)  # warm-up

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True, profile_memory=True, with_stack=True
) as prof:
    with record_function("model_inference"):
        model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

### GPU Monitoring with ROCm SMI

```python
from rocm_smi_lib import rsmi
import time

rsmi.rsmi_init(0)
try:
    num_devices = rsmi.rsmi_num_monitor_devices()
    print(f"Found {num_devices} GPU device(s)")

    for i in range(5):
        print(f"\nIteration {i+1}/5")
        for dev in range(num_devices):
            name = rsmi.rsmi_dev_name_get(dev)[1]
            util = rsmi.rsmi_dev_gpu_busy_percent_get(dev)[1]
            temp = rsmi.rsmi_dev_temp_metric_get(dev, 0, 0)[1] / 1000.0
            mem = rsmi.rsmi_dev_memory_usage_get(dev, 0)
            mem_used = mem[1] / (1024 * 1024)
            mem_total = mem[2] / (1024 * 1024)
            power = rsmi.rsmi_dev_power_ave_get(dev)[1] / 1000000.0

            print(f"GPU {dev}: {name}")
            print(f"  Util: {util}%  Temp: {temp}C  Mem: {mem_used:.0f}/{mem_total:.0f}MB  Power: {power:.1f}W")
        time.sleep(1)
finally:
    rsmi.rsmi_shut_down()
```

---

## Performance Optimization

### Hardware Optimization

1. **GPU Selection**: Use the most powerful GPU (RX 7900 XTX) as primary device
2. **Cooling**: Ensure adequate cooling for sustained performance
3. **PCIe Configuration**: Use PCIe 4.0+ slots with x16 lanes
4. **System Memory**: Use high-speed RAM with sufficient capacity

### Memory Optimization

1. **Batch Size Optimization**: Find the optimal batch size for your GPU memory
   ```python
   for batch_size in [16, 32, 64, 128, 256]:
       try:
           x = torch.randn(batch_size, 3, 224, 224, device="cuda")
           model(x)
           print(f"Batch size {batch_size} works")
       except RuntimeError:
           break
   ```

2. **Mixed Precision Training**: Use FP16 or BF16 for reduced memory
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       output = model(input)
   ```

3. **Gradient Checkpointing**: Trade computation for memory
   ```python
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(model.expensive_layer, input)
   ```

4. **Cache Management**: Clear cache periodically
   ```python
   torch.cuda.empty_cache()
   ```

### Computation Optimization

1. **Kernel Selection**: Use optimized kernels for AMD GPUs
   ```python
   os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
   os.environ["MIOPEN_FIND_MODE"] = "3"
   ```

2. **Operator Fusion**: Fuse operations when possible
   ```python
   nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
   ```

3. **Custom Kernels**: Use Triton for custom GPU kernels

4. **Quantization**: Use BITSANDBYTES for 8-bit/4-bit quantization

### Distributed Training Optimization

1. **Data Parallelism**: Use DistributedDataParallel for multi-GPU training

2. **NCCL Tuning**: Optimize NCCL parameters for AMD GPUs
   ```python
   os.environ["NCCL_DEBUG"] = "INFO"
   os.environ["NCCL_IB_DISABLE"] = "1"
   os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
   ```

3. **Gradient Accumulation**: Accumulate gradients for larger effective batch sizes

### Model Optimization

1. **Architecture Optimization**: Use efficient attention mechanisms (Flash Attention)
2. **Activation Functions**: Use efficient activations (SiLU/Swish)
3. **Model Pruning**: Reduce model size through pruning
4. **Knowledge Distillation**: Distill large models into smaller ones

---

## Benchmarking

Rusty Stack includes a comprehensive performance validation suite.

### Running Benchmarks

```bash
# Interactive selection
rusty bench

# Full benchmark suite
rusty bench --all

# Specific benchmark
rusty bench --json pytorch
```

### Benchmark Categories

| Category | Benchmarks | Description |
|----------|-----------|-------------|
| **GPU Summary** | `gpu-capability` | Peak TFLOPS (FP16), bandwidth, hardware telemetry |
| **Memory** | `memory-bandwidth` | HBM throughput, PCIe bandwidth |
| **PyTorch** | `pytorch`, `gemm` | GEMM and Convolution GFLOPS |
| **vLLM** | `vllm` | Tokens/second throughput |
| **Flash Attention** | `flash-attention` | Speedup and memory savings analysis |
| **DeepSpeed** | `deepspeed` | Distributed training benchmarks |
| **Megatron** | `megatron` | Large model training benchmarks |

### Comparative Analysis (Before vs. After)

The benchmark system automatically manages a performance baseline:

- **Baseline**: The first successful benchmark run, stored in `~/.rusty-stack/logs`
- **Latest**: Your most recent run
- **Deltas**: Percentage change displayed in the TUI (green = improvement, red = regression)

This helps validate the impact of ROCm updates, kernel optimizations, or driver changes.

### HTML Export

From the TUI benchmark screen, press `E` to export a full HTML report to `~/.mlstack/reports/`.

### Multi-GPU and iGPU Filtering

On mixed iGPU+dGPU systems, benchmark runtime prep filters integrated GPUs and exports only discrete devices in `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

---

## Troubleshooting

### GPU Detection Issues

**GPU Not Detected** (`No CUDA GPUs are available`):
- Check ROCm installation: `rocminfo`
- Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
- Check permissions: `groups` (should include `video` or `render`)
- Update drivers: `sudo pacman -Syu` (Arch) or `sudo apt update && sudo apt upgrade` (Ubuntu)

**Multiple GPUs Not Detected**:
- Set environment variables: `export HIP_VISIBLE_DEVICES=0,1`
- Check PCIe configuration: `lspci | grep -i amd`
- Verify ROCm multi-GPU support: `rocminfo`

### Memory Issues

**Out of Memory** (`RuntimeError: CUDA out of memory`):
- Reduce batch size
- Use mixed precision training
- Use gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`
- Monitor memory: `torch.cuda.memory_summary()`

**Memory Fragmentation**:
- Clear cache periodically
- Allocate tensors largest-first
- Restart the process if fragmentation is severe

### Performance Issues

**Slow Training**:
- Profile with PyTorch Profiler or `rusty bench`
- Check GPU utilization with ROCm SMI
- Optimize data loading (more workers, `pin_memory=True`)
- Use optimized kernels and operations

**Low GPU Utilization** (below 50%):
- Increase batch size
- Use more workers for data loading
- Use non-blocking transfers: `tensor.to('cuda', non_blocking=True)`
- Profile to identify CPU bottlenecks

### Compatibility Issues

**PyTorch Version Mismatch** (`ImportError: cannot import name 'X' from 'torch'`):
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Reinstall with correct ROCm version via `rusty` TUI
- Check ROCm compatibility: `rocminfo`

**CUDA Compatibility Layer Issues** (`AttributeError: module 'torch.cuda' has no attribute 'X'`):
- Use ROCm-specific APIs when available
- Check PyTorch ROCm documentation
- Update to latest PyTorch version with ROCm support

### Extension Component Issues

**Triton Compilation Errors**:
- Check ROCm version compatibility
- Use simpler kernel implementations
- Check for AMD-specific limitations

**BITSANDBYTES Quantization Errors**:
- Check model compatibility with 8-bit quantization
- Try different quantization parameters

**vLLM Memory Errors**:
- Reduce model size or sequence length
- Adjust PagedAttention parameters
- Use tensor parallelism across multiple GPUs

### TUI Issues

- Run with an interactive terminal (TTY) and a valid `$TERM`
- Set `MLSTACK_NO_ALT_SCREEN=1` if your terminal struggles with alternate screen buffers
- If screen artifacts appear, press `Ctrl+C` to exit safely
- Ensure `rustc` + `cargo` are installed if building manually

---

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [MIGraphX Documentation](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Flash Attention Documentation](https://github.com/Dao-AILab/flash-attention)
- [Triton Documentation](https://triton-lang.org/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [BITSANDBYTES Documentation](https://github.com/TimDettmers/bitsandbytes)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

---

## Author

**Stanley Chisango (Scooter Lacroix)**

- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)
