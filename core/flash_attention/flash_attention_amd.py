#!/usr/bin/env python3
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

"""
Flash Attention implementation for AMD GPUs using PyTorch
with dynamic dispatch between Triton and Composable Kernel (CK) backends
"""
import os
import torch
import math
import warnings
from typing import Optional, Tuple, List, Union

# Set environment variables for AMD GPUs
os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ROCM_DEVICE"] = "0,1"

# Add ROCm tools to PATH if they're not already there
def add_rocm_to_path():
    """Add ROCm tools to PATH if they're not already there."""
    rocm_paths = [
        "/opt/rocm/bin",
        "/opt/rocm/hip/bin",
        "/opt/rocm/opencl/bin",
        "/opt/rocm/llvm/bin"
    ]

    current_path = os.environ.get("PATH", "")
    path_dirs = current_path.split(":")

    # Check if any of the ROCm paths are already in PATH
    rocm_in_path = any(rocm_path in path_dirs for rocm_path in rocm_paths)

    if not rocm_in_path:
        # Add ROCm paths to PATH
        new_path = current_path
        for rocm_path in rocm_paths:
            if os.path.exists(rocm_path):
                new_path = f"{rocm_path}:{new_path}"

        # Update PATH
        os.environ["PATH"] = new_path
        warnings.warn(f"Added ROCm tools to PATH: {', '.join(rocm_path for rocm_path in rocm_paths if os.path.exists(rocm_path))}")

# Try to add ROCm to PATH
add_rocm_to_path()

# Try to import the Triton backend
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    warnings.warn("Triton is not available. Falling back to CK implementation only.")

# Try to import the CK backend
try:
    import flash_attention_amd_cuda
    HAS_CK = True
    # Test if the CK backend actually works
    try:
        # Check if CUDA/HIP is available
        if not torch.cuda.is_available():
            warnings.warn("CUDA/HIP is not available. Falling back to CPU implementation.")
            HAS_CK = False
        else:
            # Create small test tensors
            test_q = torch.randn(1, 8, 1, 32, device='cuda')
            test_k = torch.randn(1, 8, 1, 32, device='cuda')
            test_v = torch.randn(1, 8, 1, 32, device='cuda')
            # Try to run a forward pass
            flash_attention_amd_cuda.forward(test_q, test_k, test_v, 0.0, None, False, [-1, -1])
    except Exception as e:
        error_str = str(e)
        if "Tool lib 1 failed to load" in error_str:
            HAS_CK = False
            # Try to fix the PATH issue automatically
            rocm_bin = "/opt/rocm/bin"
            if os.path.exists(rocm_bin):
                os.environ["PATH"] = f"{rocm_bin}:{os.environ.get('PATH', '')}"
                warnings.warn(
                    f"Added {rocm_bin} to PATH. You may need to restart your application for this to take effect. "
                    "Falling back to pure Python implementation for now."
                )
            else:
                warnings.warn(
                    "CK implementation is available but AMD GPU tools could not be loaded. "
                    "This is likely because the ROCm tools are not in your PATH. "
                    "Try adding /opt/rocm/bin to your PATH. "
                    "Falling back to pure Python implementation."
                )
        elif "hipErrorNoDevice" in error_str or "no CUDA GPUs are available" in error_str:
            HAS_CK = False
            warnings.warn("No AMD GPUs detected. Falling back to CPU implementation.")
        else:
            warnings.warn(f"CK implementation test failed with error: {e}. It may not work correctly.")
except ImportError:
    HAS_CK = False
    warnings.warn("CK implementation is not available. Falling back to Triton implementation only.")

# Constants for dispatch decisions
CK_MAX_HEAD_DIM = 128  # Maximum head dimension for CK implementation
TRITON_MIN_HEAD_DIM = 32  # Minimum head dimension for Triton implementation
TRITON_MAX_SEQ_LEN = 8192  # Maximum sequence length for Triton implementation

def should_use_ck(head_dim: int, seq_len_q: int, seq_len_k: int) -> bool:
    """
    Determine whether to use the CK backend based on input parameters.

    Args:
        head_dim: Dimension of each attention head
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length

    Returns:
        True if CK backend should be used, False otherwise
    """
    # RDNA3 Note: The current dispatch logic is based on tensor dimensions (head_dim, seq_len)
    # and backend availability (HAS_CK). It does not explicitly consider the specific AMD GPU
    # architecture (e.g., RDNA2 vs. RDNA3/gfx11xx).
    # Future performance tuning could involve detecting the GPU architecture.
    # For instance, the relative performance of the Composable Kernel (CK) C++ extension
    # versus the Triton backend might differ on RDNA3 compared to older architectures
    # due to changes in instruction sets, cache hierarchy, or other hardware features.
    # An architecture-aware dispatch could potentially select a more optimal backend
    # for RDNA3 GPUs under certain conditions. For now, this is a general heuristic.

    # Use CK for small head dimensions
    if head_dim <= CK_MAX_HEAD_DIM:
        return True # CK is often efficient for smaller head dimensions.

    # Use Triton for large sequence lengths (currently this path leads to CK if not CK_MAX_HEAD_DIM)
    # This condition seems to be incorrectly leading to 'return False' which would imply *not* using CK,
    # but the final fallback is 'return HAS_CK'.
    # Assuming the intent is that Triton might be better for very large sequences if it were an option here.
    if seq_len_q > TRITON_MAX_SEQ_LEN or seq_len_k > TRITON_MAX_SEQ_LEN:
        # If Triton were the alternative, this might be 'return False' (i.e., don't use CK, use Triton).
        # However, since Triton isn't fully integrated as a selectable backend in this function's logic
        # beyond the HAS_TRITON global flag (which isn't used here for selection),
        # this condition might not have the intended effect if HAS_CK is true.
        # For now, if sequence length is very large, and CK is not preferred by head_dim,
        # it will fall through to the HAS_CK default.
        pass # Placeholder for potential future Triton-specific logic for very large sequences.

    # Default to CK if available and not ruled out by other conditions.
    return HAS_CK


class FlashAttentionFunction(torch.autograd.Function):
    """
    Flash Attention implementation using PyTorch operations.
    This is a pure Python implementation that works with AMD GPUs.
    """
    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        """
        Forward pass for Flash Attention.

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
        # Get dimensions
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, _, _ = k.shape

        # Reshape for batch matrix multiplication
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seqlen_q, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seqlen_k, head_dim)
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seqlen_k, head_dim)

        # Scale query
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        q = q * softmax_scale

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size * num_heads, seqlen_q, seqlen_k)

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask, float('-inf'))

        # Apply local attention if needed
        if window_size[0] >= 0 or window_size[1] >= 0:
            left, right = window_size
            window_mask = torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool)
            for i in range(seqlen_q):
                window_mask[i, max(0, i - left):min(seqlen_k, i + right + 1)] = False
            scores.masked_fill_(window_mask, float('-inf'))

        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        if dropout_p > 0.0 and not torch.jit.is_scripting():
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

        # Compute output
        output = torch.bmm(attn_weights, v)  # (batch_size * num_heads, seqlen_q, head_dim)

        # Reshape output
        output = output.reshape(batch_size, num_heads, seqlen_q, head_dim).transpose(1, 2)

        # Save for backward
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Flash Attention.
        """
        q, k, v, attn_weights = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal
        window_size = ctx.window_size

        # Get dimensions
        batch_size, num_heads, seqlen_q, head_dim = grad_output.shape
        _, _, seqlen_k, _ = k.shape

        # Reshape grad_output
        grad_output = grad_output.transpose(1, 2).reshape(batch_size * num_heads, seqlen_q, head_dim)

        # Compute gradient for v
        grad_v = torch.bmm(attn_weights.transpose(1, 2), grad_output)
        grad_v = grad_v.reshape(batch_size, num_heads, seqlen_k, head_dim).transpose(1, 2)

        # Compute gradient for attn_weights
        grad_attn_weights = torch.bmm(grad_output, v.transpose(1, 2))

        # Apply softmax backward
        grad_scores = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))

        # Apply dropout backward
        if dropout_p > 0.0 and not torch.jit.is_scripting():
            grad_scores = torch.nn.functional.dropout(grad_scores, p=dropout_p)

        # Apply causal mask backward
        if causal:
            causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=grad_scores.device, dtype=torch.bool), diagonal=1)
            grad_scores.masked_fill_(causal_mask, 0.0)

        # Apply local attention backward
        if window_size[0] >= 0 or window_size[1] >= 0:
            left, right = window_size
            window_mask = torch.ones(seqlen_q, seqlen_k, device=grad_scores.device, dtype=torch.bool)
            for i in range(seqlen_q):
                window_mask[i, max(0, i - left):min(seqlen_k, i + right + 1)] = False
            grad_scores.masked_fill_(window_mask, 0.0)

        # Compute gradient for q and k
        q_reshaped = q.reshape(batch_size * num_heads, seqlen_q, head_dim)
        k_reshaped = k.reshape(batch_size * num_heads, seqlen_k, head_dim)

        grad_q = torch.bmm(grad_scores, k_reshaped) * softmax_scale
        grad_k = torch.bmm(grad_scores.transpose(1, 2), q_reshaped)

        # Reshape gradients
        grad_q = grad_q.reshape(batch_size, num_heads, seqlen_q, head_dim).transpose(1, 2)
        grad_k = grad_k.reshape(batch_size, num_heads, seqlen_k, head_dim).transpose(1, 2)

        return grad_q, grad_k, grad_v, None, None, None, None


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
    return FlashAttentionFunction.apply(q, k, v, dropout_p, softmax_scale, causal, window_size)


class FlashAttention(torch.nn.Module):
    """
    Flash Attention module with dynamic dispatch between Triton and CK backends.
    """
    def __init__(
        self,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
    ) -> torch.Tensor:
        """
        Forward pass with dynamic dispatch between Triton and CK backends.

        Args:
            q: Query tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seqlen_k, num_heads, head_dim)
            causal: Whether to apply causal masking
            window_size: Local attention window size (left, right). (-1, -1) means global attention

        Returns:
            Output tensor of shape (batch_size, seqlen_q, num_heads, head_dim)
        """
        # q, k, v are (B, Nq, H, D). Transpose to (B, H, Nq, D) for some internal processing.
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)

        # Define self.scale based on self.softmax_scale or head_dim of original q
        # q is (B, Nq, H, D), so q.size(3) is head_dim
        self.scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(q.size(3))

        # Attempt to use Triton if available and conditions are suitable (no dropout, global window)
        if HAS_TRITON and self.dropout_p == 0.0 and window_size[0] == -1 and window_size[1] == -1:
            try:
                # Triton's flash_attention op expects contiguous inputs of shape (B, H, Nq, D).
                q_contiguous = q_reshaped.contiguous()
                k_contiguous = k_reshaped.contiguous()
                v_contiguous = v_reshaped.contiguous()

                out_reshaped_triton = triton.ops.flash_attention.attn_forward(
                    q_contiguous, 
                    k_contiguous, 
                    v_contiguous,
                    bias=None,      # Assuming no bias is used in this FlashAttention variant
                    causal=causal,  # Triton's op handles the causal flag
                    sm_scale=self.scale
                )
                # Output from triton is (B, H, Nq, D). Transpose back to q's original shape (B, Nq, H, D).
                out = out_reshaped_triton.transpose(1, 2)
                return out
            except Exception as e:
                warnings.warn(f"Triton flash_attention execution failed: {e}. Falling back.", UserWarning)
                pass # Fall through to other methods if Triton fails

        # Fallback to C++ extension IF Triton was not available or failed AND specific conditions are met.
        # (dropout_p == 0.0, NOT causal, and global window).
        if not HAS_TRITON and HAS_CK and self.dropout_p == 0.0 and not causal and window_size[0] == -1 and window_size[1] == -1:
            warnings.warn(
                "Using C++ Flash Attention extension (flash_attention_amd_cuda), "
                "which has a non-functional backward pass. Training will be incorrect or fail if this path is taken for training.",
                UserWarning
            )
            try:
                # flash_attention_amd_cuda.forward expects q,k,v as (B, Nq, H, D)
                window_size_list = list(window_size)
                out_cpp = flash_attention_amd_cuda.forward( # Correct function name
                    q, k, v, # Pass original q, k, v
                    self.dropout_p, # This will be 0.0 due to the condition
                    self.scale,
                    causal,        # This will be False due to the condition
                    window_size_list # This will be [-1,-1]
                )
                # Output from C++ extension is already (B, Nq, H, D)
                return out_cpp
            except Exception as e:
                warnings.warn(f"C++ (CK) flash_attention_amd_cuda.forward execution failed: {e}. Falling back to PyTorch.", UserWarning)
                pass # Fall through to PyTorch implementation if C++ fails

        # Ultimate fallback to PyTorch implementation (FlashAttentionFunction.apply)
        # This path is taken if:
        # - Dropout is enabled (self.dropout_p > 0.0)
        # - Windowed attention is used (window_size[0] != -1 or window_size[1] != -1)
        # - OR if Triton was skipped or failed, AND the C++ extension was skipped or failed.
        warnings.warn("Falling back to PyTorch FlashAttention implementation (FlashAttentionFunction).", UserWarning)
        # flash_attn_func expects q,k,v as (B, Nq, H, D)
        return flash_attn_func(
            q, k, v, # Pass original q, k, v
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale, # Pass the calculated scale
            causal=causal,
            window_size=window_size,
        )
