# PyTorch with ROCm Guide

## Introduction

This guide provides detailed instructions for installing and configuring PyTorch with ROCm support for AMD GPUs. PyTorch is a popular deep learning framework, and with ROCm support, it can leverage the power of AMD GPUs for accelerated training and inference.

## Prerequisites

Before installing PyTorch with ROCm, ensure you have:

1. **ROCm Installed**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **Python Environment**: Python 3.8+ with pip
3. **System Dependencies**: Required system libraries and tools

## Installation

### 1. Install PyTorch with ROCm Support

PyTorch provides pre-built wheels with ROCm support. Install PyTorch with the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

This installs PyTorch 2.6.0 with ROCm 6.2.4 support.

### 2. Verify Installation

Verify that PyTorch is installed correctly and can access your AMD GPU:

```python
import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA (ROCm) is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check number of GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check GPU name
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

You should see output similar to:

```
PyTorch version: 2.6.0+rocm6.2.4
CUDA available: True
Number of GPUs: 2
GPU 0: AMD Radeon RX 7900 XTX
GPU 1: AMD Radeon RX 7800 XT
```

## Configuration

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
```

## Usage Examples

### Basic Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create model and move to GPU
model = SimpleModel().to(device)

# Create dummy data
x = torch.randn(100, 10, device=device)
y = torch.randint(0, 2, (100, 1), dtype=torch.float32, device=device)

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(100):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
```

### Mixed Precision Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model and move to GPU
model = SimpleModel().to(device)

# Create dummy data
x = torch.randn(100, 10, device=device)
y = torch.randint(0, 2, (100, 1), dtype=torch.float32, device=device)

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loss function
criterion = nn.BCELoss()

# Create gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop
for epoch in range(100):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with autocast():
        outputs = model(x)
        loss = criterion(outputs, y)
    
    # Backward pass and optimize with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```
   No CUDA GPUs are available
   ```
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
   - Check permissions: `groups` (should include video or render)

2. **Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce batch size
   - Use mixed precision training
   - Use gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`

3. **hipBLASLt Warning**:
   ```
   hipBLASLt warning: No suitable algorithm found for the matmul operation
   ```
   
   Solutions:
   - This is a warning, not an error, and can be ignored in most cases
   - Set `torch.cuda.max_split_size_mb = 512` for better performance

4. **Slow Performance**:
   
   Solutions:
   - Use mixed precision training
   - Optimize data loading with more workers and pin_memory
   - Set benchmark mode: `torch.backends.cudnn.benchmark = True`
   - Profile your code to identify bottlenecks

## Performance Optimization

### Memory Optimization

1. **Batch Size Optimization**: Find the optimal batch size for your GPU memory
2. **Mixed Precision Training**: Use FP16 or BF16 for reduced memory usage
3. **Gradient Checkpointing**: Trade computation for memory
4. **Memory Fragmentation**: Clear cache periodically

### Computation Optimization

1. **Kernel Selection**: Use optimized kernels for AMD GPUs
2. **Operator Fusion**: Fuse operations when possible
3. **Custom Kernels**: Use Triton for custom kernels
4. **Quantization**: Use BITSANDBYTES for quantization

## Additional Resources

- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)
- [PyTorch Forums](https://discuss.pytorch.org/)

## Next Steps

After installing PyTorch with ROCm support, you can proceed to install other components of the ML stack, such as ONNX Runtime, MIGraphX, and Flash Attention.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

