# Flash Attention Build Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for building and installing Flash Attention with AMD GPU support. Flash Attention is a fast and memory-efficient exact attention algorithm that speeds up transformer model training and inference.

## Prerequisites

Before building Flash Attention for AMD GPUs, ensure you have:

1. **ROCm Installed**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **PyTorch with ROCm**: Follow the [PyTorch ROCm Guide](/docs/core/pytorch_rocm_guide.md)
3. **Python Environment**: Python 3.8+ with pip
4. **System Dependencies**: Required system libraries and tools
5. **GPU Access**: Proper permissions to access AMD GPUs
6. **Disk Space**: At least 2GB of free disk space for the build

## Build Steps

### 1. Install Build Dependencies

First, install the necessary build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    ninja-build
```

Install Python dependencies:

```bash
pip install \
    packaging \
    ninja \
    wheel \
    setuptools
```

### 2. Clone Flash Attention Repository

Clone the Flash Attention repository:

```bash
cd $HOME
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
```

### 3. Create AMD-Compatible Implementation

Create a custom implementation for AMD GPUs:

```bash
mkdir -p flash_attn_amd
cd flash_attn_amd
```

Create the Python implementation file:

```bash
cat > flash_attention_amd.py << 'EOF'
import torch
import torch.nn.functional as F

class FlashAttention(torch.nn.Module):
    """
    Flash Attention implementation for AMD GPUs using PyTorch operations.
    This is a pure PyTorch implementation that works on AMD GPUs.
    """
    def __init__(self, dropout=0.0, causal=False):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)
        
        Returns: (batch_size, seq_len, num_heads, head_dim)
        """
        # Reshape q, k, v for multi-head attention
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, head_dim) @ (batch_size, num_heads, head_dim, seq_len_k)
        # -> (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if mask is not None:
            # Expand mask to match attention weights shape
            if mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            
            # Apply mask
            attn_weights.masked_fill_(~mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Compute attention output
        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_k, head_dim)
        # -> (batch_size, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back to (batch_size, seq_len_q, num_heads, head_dim)
        output = output.transpose(1, 2)
        
        return output

def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, return_attn_probs=False):
    """
    Functional interface for Flash Attention.
    
    Args:
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        dropout_p: dropout probability
        causal: whether to apply causal masking
        return_attn_probs: whether to return attention probabilities
        
    Returns:
        output: (batch_size, seq_len, num_heads, head_dim)
        attn_weights: (batch_size, num_heads, seq_len, seq_len) if return_attn_probs=True
    """
    flash_attn = FlashAttention(dropout=dropout_p, causal=causal)
    output = flash_attn(q, k, v)
    
    if return_attn_probs:
        # Compute attention weights for return
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        return output, attn_weights
    
    return output

# For compatibility with the original Flash Attention API
class FlashAttentionInterface:
    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, causal=False):
        output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        return output
EOF
```

Create the setup file:

```bash
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="flash_attention_amd",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["flash_attention_amd"],
    install_requires=[
        "torch>=2.0.0",
    ],
    author="User",
    author_email="user@example.com",
    description="Flash Attention implementation for AMD GPUs",
    keywords="flash attention, amd, gpu, pytorch",
    python_requires=">=3.8",
)
EOF
```

### 4. Install the AMD Implementation

Install the AMD implementation:

```bash
cd $HOME/flash-attention
pip install -e .
```

### 5. Verify Installation

Verify that Flash Attention is installed correctly:

```python
import torch
from flash_attention_amd import flash_attn_func

# Create dummy data
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

# Run Flash Attention
output = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {output.shape}")
```

## Performance Benchmarking

### Benchmark Script

Create a benchmark script to compare Flash Attention with standard attention:

```python
import torch
import time
from flash_attention_amd import flash_attn_func

def standard_attention(q, k, v, causal=False):
    """Standard scaled dot-product attention."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Reshape for batched matrix multiplication
    q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute attention output
    output = torch.matmul(attn_weights, v)
    
    # Reshape back
    output = output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
    
    return output

def benchmark(batch_size, seq_len, num_heads, head_dim, causal=False, num_runs=10):
    """Benchmark Flash Attention vs standard attention."""
    # Create dummy data
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    
    # Warm-up
    for _ in range(5):
        _ = flash_attn_func(q, k, v, causal=causal)
        _ = standard_attention(q, k, v, causal=causal)
    
    # Benchmark Flash Attention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = flash_attn_func(q, k, v, causal=causal)
        torch.cuda.synchronize()
    flash_time = (time.time() - start_time) / num_runs
    
    # Benchmark standard attention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = standard_attention(q, k, v, causal=causal)
        torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_runs
    
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attn_func(q, k, v, causal=causal)
    torch.cuda.synchronize()
    flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    torch.cuda.reset_peak_memory_stats()
    _ = standard_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Check correctness
    with torch.no_grad():
        flash_output = flash_attn_func(q, k, v, causal=causal)
        standard_output = standard_attention(q, k, v, causal=causal)
        max_diff = torch.max(torch.abs(flash_output - standard_output)).item()
    
    return {
        "flash_time": flash_time * 1000,  # ms
        "standard_time": standard_time * 1000,  # ms
        "speedup": standard_time / flash_time,
        "flash_memory": flash_memory,  # MB
        "standard_memory": standard_memory,  # MB
        "memory_reduction": standard_memory / flash_memory,
        "max_diff": max_diff
    }

# Run benchmarks for different sequence lengths
seq_lengths = [128, 256, 512, 1024, 2048, 4096]
results = []

for seq_len in seq_lengths:
    print(f"Benchmarking sequence length: {seq_len}")
    result = benchmark(
        batch_size=2,
        seq_len=seq_len,
        num_heads=8,
        head_dim=64,
        causal=True,
        num_runs=10
    )
    results.append((seq_len, result))
    print(f"  Flash Attention: {result['flash_time']:.2f} ms")
    print(f"  Standard Attention: {result['standard_time']:.2f} ms")
    print(f"  Speedup: {result['speedup']:.2f}x")
    print(f"  Memory Reduction: {result['memory_reduction']:.2f}x")
    print(f"  Max Difference: {result['max_diff']:.6f}")
    print()

# Print summary
print("Summary:")
print("Sequence Length | Speedup | Memory Reduction | Max Difference")
print("---------------|---------|-----------------|---------------")
for seq_len, result in results:
    print(f"{seq_len:15d} | {result['speedup']:7.2f}x | {result['memory_reduction']:15.2f}x | {result['max_diff']:13.6f}")
```

Run the benchmark script:

```bash
python benchmark_flash_attn.py
```

## Troubleshooting

### Common Issues

1. **Import Error**:
   ```
   ImportError: No module named flash_attention_amd
   ```
   
   Solutions:
   - Check if Flash Attention is installed: `pip list | grep flash-attention`
   - Verify installation path
   - Reinstall the package

2. **CUDA/ROCm Errors**:
   ```
   RuntimeError: CUDA error: device-side assert triggered
   ```
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify PyTorch with ROCm support: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU compatibility

3. **Performance Issues**:
   
   Solutions:
   - Optimize batch size and sequence length
   - Use mixed precision (FP16 or BF16)
   - Check for memory fragmentation
   - Profile with PyTorch Profiler

## Additional Resources

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention GitHub Repository](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Next Steps

After installing Flash Attention with AMD GPU support, you can proceed to install other components of the ML stack, such as RCCL and MPI for distributed training.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

