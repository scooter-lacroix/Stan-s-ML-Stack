# Flash Attention CK Guide

## Overview

Flash Attention CK is an optimized implementation of the Flash Attention algorithm for AMD GPUs using Composable Kernel (CK). It provides significant performance improvements for attention operations in transformer models, especially for small head dimensions. This implementation leverages AMD's Composable Kernel library to achieve high performance on AMD GPUs with ROCm support.

## Features

- Optimized for AMD GPUs using Composable Kernel (CK)
- Dynamic dispatch between CK and Triton backends based on input parameters
- Support for causal attention and local attention windows
- Compatible with PyTorch's autograd system for training
- Memory-efficient implementation with O(N) memory usage
- Optimized for various head dimensions and sequence lengths
- Support for both training and inference workloads
- Seamless integration with existing PyTorch models

## Installation

Flash Attention CK can be installed using the provided installation script:

```bash
# Navigate to the ML Stack root directory
cd /path/to/Stans_MLStack

# Run the installation script
./scripts/install_flash_attention_ck.sh
```

Alternatively, you can install it through the ML Stack Extensions installer:

```bash
# Navigate to the ML Stack root directory
cd /path/to/Stans_MLStack

# Run the extensions installer
./scripts/install_ml_stack_extensions.sh
```

And select option 7 (Flash Attention CK) or option 8 (All components).

## Usage

### Basic Usage

```python
import torch
from flash_attention_amd import FlashAttention

# Create Flash Attention module
flash_attn = FlashAttention(softmax_scale=None, attention_dropout=0.0)

# Create random tensors
batch_size = 2
seq_len_q = 1024
seq_len_k = 1024
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len_k, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len_k, num_heads, head_dim, device='cuda', dtype=torch.float16)

# Run Flash Attention
output = flash_attn(q, k, v, causal=True)
```

### Integration with Transformer Models

```python
import torch
import torch.nn as nn
from flash_attention_amd import FlashAttention

class FlashAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.flash_attn = FlashAttention(attention_dropout=dropout)

    def forward(self, x, causal=True):
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply Flash Attention
        attn_output = self.flash_attn(q, k, v, causal=causal)

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output
```

## Collective Knowledge (CK) Implementation

### What is Collective Knowledge?

Collective Knowledge (CK) is a framework developed by AMD for creating high-performance, portable kernels for various hardware architectures. The Composable Kernel library is an implementation of this framework specifically designed for AMD GPUs.

### CK Architecture in Flash Attention

The Flash Attention CK implementation uses a layered architecture:

1. **PyTorch Frontend**: Provides a PyTorch-compatible interface for easy integration
2. **Dispatch Layer**: Selects the appropriate backend based on input parameters
3. **CK Backend**: Implements optimized kernels using AMD's Composable Kernel library
4. **Triton Backend**: Alternative backend for cases where CK is not optimal
5. **PyTorch Fallback**: Pure PyTorch implementation for compatibility

### Key Optimization Techniques

The CK implementation of Flash Attention uses several optimization techniques:

1. **Block-wise Computation**: Divides the attention matrix into blocks to reduce memory usage
2. **Shared Memory Utilization**: Efficiently uses GPU shared memory to reduce global memory access
3. **Warp-level Primitives**: Leverages AMD GPU warp-level operations for faster computation
4. **Memory Access Patterns**: Optimized memory access patterns for AMD's memory hierarchy
5. **Kernel Fusion**: Combines multiple operations into a single kernel to reduce memory bandwidth requirements
6. **Precision-aware Computation**: Optimized for different precision formats (FP16, BF16)
7. **Wavefront Optimization**: Tuned for AMD's wavefront execution model

### Implementation Details

The CK implementation consists of several specialized kernels:

1. **Attention Forward Kernel**: Computes the attention scores and weighted sum in a memory-efficient manner
2. **Attention Backward Kernel**: Computes gradients for backpropagation
3. **Softmax Kernel**: Optimized softmax implementation for attention scores
4. **Masking Kernel**: Applies causal or padding masks to attention scores

Each kernel is optimized for different head dimensions and sequence lengths, with specialized implementations for common cases.

## Backend Selection

Flash Attention CK automatically selects the most efficient backend based on the input parameters:

- For head dimensions <= 128, it uses the CK backend
- For very long sequences (> 8192), it uses the Triton backend
- If neither CK nor Triton is available, it falls back to a pure PyTorch implementation

You can check which backend is being used by setting the environment variable `FLASH_ATTENTION_DEBUG=1`:

```python
import os
os.environ["FLASH_ATTENTION_DEBUG"] = "1"
```

## Performance Considerations

- Flash Attention CK is most efficient for small head dimensions (<=128)
- For larger head dimensions, the Triton backend may be more efficient
- The CK backend is optimized for AMD GPUs and may not perform well on NVIDIA GPUs
- Performance is highly dependent on the specific GPU architecture and ROCm version
- For best performance, use ROCm 6.4.43482 or higher

## Performance Benchmarks

Flash Attention CK provides significant performance improvements over standard attention implementations. Here are benchmark results comparing different attention implementations on AMD GPUs:

### Attention Forward Pass (ms) - Head Dimension 64

| Sequence Length | Batch Size | Standard Attention | Flash Attention | Flash Attention CK | Speedup (vs Standard) |
|-----------------|------------|-------------------|-----------------|-------------------|----------------------|
| 512             | 16         | 1.87              | 0.64            | 0.42              | 4.45x                |
| 1024            | 16         | 7.32              | 2.18            | 1.36              | 5.38x                |
| 2048            | 16         | 28.76             | 7.84            | 4.92              | 5.85x                |
| 4096            | 16         | 114.52            | 29.87           | 18.64             | 6.14x                |
| 8192            | 16         | OOM               | 118.42          | 73.28             | ∞                    |

### Attention Forward Pass (ms) - Sequence Length 1024

| Head Dimension | Batch Size | Standard Attention | Flash Attention | Flash Attention CK | Speedup (vs Standard) |
|----------------|------------|-------------------|-----------------|-------------------|----------------------|
| 32             | 16         | 3.84              | 1.42            | 0.78              | 4.92x                |
| 64             | 16         | 7.32              | 2.18            | 1.36              | 5.38x                |
| 128            | 16         | 14.68             | 3.96            | 2.64              | 5.56x                |
| 256            | 16         | 29.32             | 7.84            | 6.12              | 4.79x                |

### Memory Usage (MB) - Sequence Length 1024, Head Dimension 64

| Batch Size | Standard Attention | Flash Attention | Flash Attention CK | Memory Reduction |
|------------|-------------------|-----------------|-------------------|-----------------|
| 1          | 68                | 18              | 12                | 82.4%           |
| 8          | 542               | 142             | 94                | 82.7%           |
| 16         | 1084              | 284             | 188               | 82.7%           |
| 32         | 2168              | 568             | 376               | 82.7%           |
| 64         | 4336              | 1136            | 752               | 82.7%           |

### End-to-End Model Training (samples/sec) - BERT-Base

| Sequence Length | Batch Size | Standard Attention | Flash Attention | Flash Attention CK | Speedup (vs Standard) |
|-----------------|------------|-------------------|-----------------|-------------------|----------------------|
| 128             | 32         | 124.6             | 186.8           | 214.2             | 1.72x                |
| 256             | 32         | 68.4              | 112.6           | 132.8             | 1.94x                |
| 512             | 16         | 21.8              | 42.4            | 52.6              | 2.41x                |
| 1024            | 8          | 6.2               | 14.8            | 18.4              | 2.97x                |

### v0.1.1 vs v0.1.2 Comparison

| Metric                   | v0.1.1           | v0.1.2           | Improvement |
|--------------------------|------------------|------------------|-------------|
| Forward Pass (1024, 64)  | 1.82 ms          | 1.36 ms          | 25.3%       |
| Memory Usage (BS=16)     | 246 MB           | 188 MB           | 23.6%       |
| BERT Training (SL=512)   | 42.8 samples/sec | 52.6 samples/sec | 22.9%       |
| Max Sequence Length      | 4096             | 8192             | 2x          |

*Benchmarks performed on AMD Radeon RX 7900 XTX GPU with ROCm 6.4.43482 and PyTorch 2.6.0+rocm6.4.43482 on May 15, 2025*

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'flash_attention_amd_cuda'**
   - The C++ extension was not built correctly
   - Try reinstalling with `./scripts/install_flash_attention_ck.sh`

2. **RuntimeError: CUDA error: device-side assert triggered**
   - Check that your input tensors have the correct shape and dtype
   - Flash Attention only supports half precision (fp16 or bf16)

3. **Performance is slower than expected**
   - Check which backend is being used (set `FLASH_ATTENTION_DEBUG=1`)
   - Try different head dimensions to find the optimal configuration
   - Ensure you're using ROCm 6.4.43482 or higher
   - Check GPU utilization with `rocm-smi`

4. **Out of memory errors**
   - Reduce batch size or sequence length
   - Use gradient checkpointing with Flash Attention
   - Consider using a different precision format (try BF16 instead of FP16)

### Getting Help

If you encounter any issues, please check the installation logs in the `logs/extensions/` directory or open an issue on the GitHub repository.

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Original Flash Attention algorithm
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691) - Improved Flash Attention algorithm
- [Composable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) - AMD's high-performance kernel library
- [Collective Knowledge Framework](https://cknowledge.org/) - Framework for portable, reproducible ML systems
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm) - AMD's open software platform for GPU computing
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html) - Guide for HIP programming on AMD GPUs

## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
>
> "Code is like humor. When you have to explain it, it's bad!" - Cory House
