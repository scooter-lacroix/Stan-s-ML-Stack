# Flash Attention for AMD GPUs

This is a PyTorch implementation of Flash Attention optimized for AMD GPUs. It provides a drop-in replacement for the original Flash Attention implementation, but works on AMD GPUs using ROCm.

## Overview

Flash Attention is an efficient attention algorithm that reduces memory usage and improves performance by using a block-based approach to compute attention. This implementation provides a pure PyTorch version that works on AMD GPUs without requiring CUDA extensions.

## Features

- Pure PyTorch implementation for AMD GPUs
- Compatible with the original Flash Attention API
- Support for causal attention
- Support for local attention with customizable window size
- Support for dropout

## Installation

### Prerequisites

- **PyTorch with ROCm support**: Ensure you have a compatible PyTorch version installed for your ROCm environment.
- **AMD GPU with ROCm support**: Supported AMD GPUs (e.g., RX 6000/7000 series, MI series).
- **ROCm Version**: ROCm 5.6 or higher is recommended for best compatibility and RDNA3 support.
- **Build Tools**: `git`, `cmake`, `python3-dev`, `build-essential`. The main build script will attempt to install these if necessary.

### Recommended Installation using Build Script

The most straightforward way to build and install this FlashAttention implementation (including its Composable Kernel C++ extension and a compatible Triton version) is by using the main build script located in the `scripts` directory of the Stan-s-ML-Stack repository.

1.  **Navigate to the Stan-s-ML-Stack Repository Root:**
    ```bash
    cd /path/to/Stan-s-ML-Stack
    ```

2.  **(Optional but Recommended) Set Target AMD GPU Architectures:**
    The `AMDGPU_TARGETS` environment variable controls the compilation targets for both the FlashAttention C++ extension and Triton. Set this to a comma-separated list of architectures relevant to your GPU(s).
    ```bash
    # Example for RDNA3 GPUs (e.g., RX 7900 XTX, RX 7800 XT):
    export AMDGPU_TARGETS="gfx1100,gfx1101,gfx1102"
    
    # Example for a mix of RDNA2 and RDNA3:
    # export AMDGPU_TARGETS="gfx1030,gfx1100"
    
    # If not set, a default comprehensive list will be used by the build script, e.g.:
    # export AMDGPU_TARGETS="gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100,gfx1101,gfx1102"
    ```

3.  **Run the Build Script:**
    ```bash
    ./scripts/build_flash_attn_amd.sh
    ```
    This script will handle dependency checks, Triton compilation, and the FlashAttention C++ extension build and installation.

### Manual Installation (Advanced / Developers)

If you prefer to build this specific FlashAttention component manually (e.g., for development or integration into a different build system):

1.  **Ensure Prerequisites are Met:** Including a compatible Triton if required by this version of FlashAttention.
2.  **Navigate to this Directory:**
    ```bash
    cd /path/to/Stan-s-ML-Stack/core/flash_attention
    ```
3.  **Set Target Architectures:**
    ```bash
    export AMDGPU_TARGETS="gfx1100,gfx1101" # Or your desired list
    ```
4.  **Install the Package:**
    ```bash
    python setup_flash_attn_amd.py install
    # Or for development:
    # python setup_flash_attn_amd.py develop
    ```
    This will invoke the CMake build for the C++ extension, respecting `AMDGPU_TARGETS`.

## Usage

```python
import torch
from flash_attention_amd import flash_attn_func

# Create input tensors
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

# Run Flash Attention
output = flash_attn_func(q, k, v, causal=True)
```

## API Reference

### `flash_attn_func`

```python
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Flash Attention function.
    
    Args:
        q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax. If None, defaults to 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        window_size: Local attention window size (left, right). (-1, -1) means global attention
        
    Returns:
        Output tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
    """
```

### `FlashAttention`

```python
class FlashAttention(torch.nn.Module):
    """
    Flash Attention module.
    """
    def __init__(
        self,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        """
        Initialize Flash Attention module.
        
        Args:
            softmax_scale: Scaling factor for softmax. If None, defaults to 1/sqrt(head_dim)
            attention_dropout: Dropout probability
        """
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            causal: Whether to apply causal masking
            window_size: Local attention window size (left, right). (-1, -1) means global attention
            
        Returns:
            Output tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
        """
```

## Performance

This implementation, leveraging Composable Kernel C++ extensions and potentially Triton, is optimized for AMD GPUs and provides significant performance improvements over standard PyTorch attention, especially for long sequences. Performance characteristics can vary based on the specific AMD GPU architecture (RDNA2, RDNA3, CDNA) and the workload. Refer to the detailed [Flash Attention for AMD GPUs Guide](../../docs/guides/flash_attention_amd_guide.md) for more comprehensive benchmark information.

## Limitations

- While highly optimized, performance relative to NVIDIA's cuDNN-accelerated FlashAttention can vary.
- Ensure that the `AMDGPU_TARGETS` used during compilation match the GPU you are running on for optimal performance. If a pre-compiled package is used, check its supported architectures.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The original Flash Attention paper: [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- The original Flash Attention implementation: [flash-attention](https://github.com/HazyResearch/flash-attention)
