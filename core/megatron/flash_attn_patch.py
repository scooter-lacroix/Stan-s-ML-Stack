#!/usr/bin/env python3
# =============================================================================
# Flash Attention Patch for Megatron-LM
# =============================================================================
# This script provides a patch to integrate Flash Attention AMD with Megatron-LM.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import torch
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_attn_patch")

def patch_self_attention(module):
    """Patch self-attention module with Flash Attention."""
    try:
        import flash_attention_amd
        logger.info("Flash Attention AMD is available")
    except ImportError:
        logger.warning("Flash Attention AMD is not available")
        return module
    
    # Check if module has core_attention method
    if not hasattr(module, "core_attention"):
        logger.warning("Module does not have core_attention method")
        return module
    
    # Store original core_attention method
    original_core_attention = module.core_attention
    
    # Create Flash Attention module
    flash_attn = flash_attention_amd.FlashAttention(
        attention_dropout=getattr(module, "attention_dropout", 0.0)
    )
    
    # Define new core_attention method
    def core_attention_with_flash(query_layer, key_layer, value_layer, attention_mask=None):
        # Get dimensions
        batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head = query_layer.shape
        
        # Check if we can use Flash Attention
        use_flash = True
        
        # Flash Attention has limitations
        if attention_mask is not None and attention_mask.dim() > 2:
            # Complex attention mask not supported
            use_flash = False
            logger.warning("Complex attention mask not supported by Flash Attention")
        
        if use_flash:
            try:
                # Reshape for Flash Attention
                q = query_layer
                k = key_layer
                v = value_layer
                
                # Determine if causal mask should be used
                causal = False
                if attention_mask is not None:
                    # Check if attention mask is causal
                    if attention_mask.dim() == 2:
                        causal = torch.all(attention_mask.triu(diagonal=1) == 0).item()
                
                # Apply Flash Attention
                context_layer = flash_attn(q, k, v, causal=causal)
                
                return context_layer
            except Exception as e:
                logger.warning(f"Flash Attention failed: {e}")
                logger.warning("Falling back to original attention implementation")
                return original_core_attention(query_layer, key_layer, value_layer, attention_mask)
        else:
            # Use original attention implementation
            return original_core_attention(query_layer, key_layer, value_layer, attention_mask)
    
    # Replace core_attention method
    module.core_attention = core_attention_with_flash.__get__(module, type(module))
    
    logger.info("Self-attention module patched with Flash Attention")
    
    return module

def patch_model_with_flash_attention(model):
    """Patch model with Flash Attention."""
    logger.info("Patching model with Flash Attention")
    
    # Find self-attention modules in the model
    patched_count = 0
    for name, module in model.named_modules():
        if "self_attention" in name and hasattr(module, "core_attention"):
            logger.info(f"Patching {name} with Flash Attention")
            patch_self_attention(module)
            patched_count += 1
    
    logger.info(f"Patched {patched_count} self-attention modules with Flash Attention")
    
    return model

def create_flash_attention_forward(original_forward, attention_dropout=0.0):
    """Create a new forward method using Flash Attention."""
    try:
        import flash_attention_amd
        logger.info("Flash Attention AMD is available")
    except ImportError:
        logger.warning("Flash Attention AMD is not available")
        return original_forward
    
    # Create Flash Attention module
    flash_attn = flash_attention_amd.FlashAttention(attention_dropout=attention_dropout)
    
    # Define new forward method
    def forward_with_flash(self, query_layer, key_layer, value_layer, attention_mask=None):
        # Get dimensions
        batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head = query_layer.shape
        
        # Check if we can use Flash Attention
        use_flash = True
        
        # Flash Attention has limitations
        if attention_mask is not None and attention_mask.dim() > 2:
            # Complex attention mask not supported
            use_flash = False
            logger.warning("Complex attention mask not supported by Flash Attention")
        
        if use_flash:
            try:
                # Determine if causal mask should be used
                causal = False
                if attention_mask is not None:
                    # Check if attention mask is causal
                    if attention_mask.dim() == 2:
                        causal = torch.all(attention_mask.triu(diagonal=1) == 0).item()
                
                # Apply Flash Attention
                context_layer = flash_attn(query_layer, key_layer, value_layer, causal=causal)
                
                return context_layer
            except Exception as e:
                logger.warning(f"Flash Attention failed: {e}")
                logger.warning("Falling back to original attention implementation")
                return original_forward(self, query_layer, key_layer, value_layer, attention_mask)
        else:
            # Use original attention implementation
            return original_forward(self, query_layer, key_layer, value_layer, attention_mask)
    
    return forward_with_flash

def benchmark_flash_attention(batch_size=2, seq_length=1024, num_heads=16, head_dim=64, num_runs=100):
    """Benchmark Flash Attention against standard attention."""
    try:
        import flash_attention_amd
        logger.info("Flash Attention AMD is available")
    except ImportError:
        logger.warning("Flash Attention AMD is not available")
        return
    
    # Create input tensors
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_length, num_heads, head_dim, device="cuda")
    
    # Define standard attention function
    def standard_attention(q, k, v, causal=False):
        batch_size, seq_length, num_heads, head_dim = q.shape
        
        # Reshape for batched matrix multiplication
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if causal:
            mask = torch.triu(torch.ones(seq_length, seq_length, device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2)  # (batch_size, seq_length, num_heads, head_dim)
        
        return output
    
    # Create Flash Attention module
    flash_attn = flash_attention_amd.FlashAttention()
    
    # Warm-up
    for _ in range(10):
        _ = standard_attention(q, k, v)
        _ = flash_attn(q, k, v)
    
    # Benchmark standard attention
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_runs):
        _ = standard_attention(q, k, v)
    end_time.record()
    torch.cuda.synchronize()
    standard_time = start_time.elapsed_time(end_time) / num_runs
    
    # Benchmark Flash Attention
    start_time.record()
    for _ in range(num_runs):
        _ = flash_attn(q, k, v)
    end_time.record()
    torch.cuda.synchronize()
    flash_time = start_time.elapsed_time(end_time) / num_runs
    
    # Print results
    logger.info(f"Standard attention time: {standard_time:.2f} ms")
    logger.info(f"Flash Attention time: {flash_time:.2f} ms")
    logger.info(f"Speedup: {standard_time / flash_time:.2f}x")
    
    return standard_time, flash_time

if __name__ == "__main__":
    # Benchmark Flash Attention
    benchmark_flash_attention()
