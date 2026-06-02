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


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

