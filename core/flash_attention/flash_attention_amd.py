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
"""
import os
import torch
import math
from typing import Optional, Tuple

# Set environment variables for AMD GPUs
os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ROCM_DEVICE"] = "0,1"


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
    Flash Attention module.
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
        return flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            window_size=window_size,
        )
