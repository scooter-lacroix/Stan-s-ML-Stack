#!/usr/bin/env python3
# =============================================================================
# Megatron-LM AMD Adaptation
# =============================================================================
# This script provides adaptations for running Megatron-LM on AMD GPUs.
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

import os
import sys
import torch
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("megatron_amd_adaptation")

def check_amd_gpu_compatibility():
    """Check if AMD GPUs are available and compatible."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available through ROCm")
        return False
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.error("No GPUs detected")
        return False
    
    logger.info(f"Number of GPUs: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i}: {device_name}")
        
        if "AMD" in device_name or "Radeon" in device_name:
            logger.info(f"GPU {i} is an AMD GPU")
        else:
            logger.warning(f"GPU {i} is not an AMD GPU")
    
    return True

def setup_amd_environment():
    """Set up environment variables for AMD GPUs."""
    # Set environment variables for AMD GPUs
    os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for better performance
    
    # Set RCCL environment variables
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "^lo"  # Use all interfaces except loopback
    
    logger.info("AMD environment variables set")
    
    return True

def replace_apex_with_native_amp(model, optimizer, fp16_config):
    """Replace NVIDIA Apex with PyTorch native AMP."""
    logger.info("Replacing Apex with PyTorch native AMP")
    
    # Create GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(
        init_scale=fp16_config.get("initial_scale_power", 2) ** 16,
        growth_factor=fp16_config.get("growth_factor", 2.0),
        backoff_factor=fp16_config.get("backoff_factor", 0.5),
        growth_interval=fp16_config.get("growth_interval", 1000),
        enabled=True
    )
    
    # Return model, optimizer, and scaler
    return model, optimizer, scaler

def adapt_fused_kernels():
    """Adapt fused kernels for AMD GPUs."""
    logger.info("Adapting fused kernels for AMD GPUs")
    
    # Check if Flash Attention AMD is available
    try:
        import flash_attention_amd
        logger.info("Flash Attention AMD is available")
        has_flash_attention = True
    except ImportError:
        logger.warning("Flash Attention AMD is not available")
        has_flash_attention = False
    
    return has_flash_attention

def adapt_model_for_amd(model):
    """Adapt model for AMD GPUs."""
    logger.info("Adapting model for AMD GPUs")
    
    # Check if Flash Attention AMD is available
    has_flash_attention = adapt_fused_kernels()
    
    if has_flash_attention:
        # Replace attention implementation with Flash Attention
        try:
            from flash_attention_amd import FlashAttention
            
            # Find attention modules in the model
            for name, module in model.named_modules():
                if "self_attention" in name and hasattr(module, "core_attention"):
                    logger.info(f"Replacing attention in {name} with Flash Attention")
                    
                    # Store original attention parameters
                    attention_dropout = getattr(module, "attention_dropout", 0.0)
                    
                    # Create Flash Attention module
                    flash_attn = FlashAttention(attention_dropout=attention_dropout)
                    
                    # Replace core attention method
                    def forward_with_flash_attention(self, query_layer, key_layer, value_layer, attention_mask):
                        # Reshape for Flash Attention
                        batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head = query_layer.shape
                        query_layer = query_layer.reshape(batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head)
                        key_layer = key_layer.reshape(batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head)
                        value_layer = value_layer.reshape(batch_size, seq_length, num_attention_heads, hidden_size_per_attention_head)
                        
                        # Apply Flash Attention
                        context_layer = flash_attn(query_layer, key_layer, value_layer, causal=True)
                        
                        # Reshape back
                        context_layer = context_layer.reshape(batch_size, seq_length, num_attention_heads * hidden_size_per_attention_head)
                        
                        return context_layer
                    
                    # Replace core attention method
                    module.core_attention = forward_with_flash_attention.__get__(module, type(module))
            
            logger.info("Flash Attention adaptation completed")
        except Exception as e:
            logger.error(f"Failed to adapt Flash Attention: {e}")
    
    return model

def adapt_optimizer_for_amd(optimizer, lr_scheduler, fp16):
    """Adapt optimizer for AMD GPUs."""
    logger.info("Adapting optimizer for AMD GPUs")
    
    # If using FP16, replace Apex with native AMP
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Created GradScaler for mixed precision training")
    else:
        scaler = None
    
    return optimizer, lr_scheduler, scaler

def adapt_training_loop(train_step_func, fp16):
    """Adapt training loop for AMD GPUs."""
    logger.info("Adapting training loop for AMD GPUs")
    
    # If using FP16, wrap train_step_func with AMP
    if fp16:
        original_train_step = train_step_func
        
        def train_step_with_amp(batch, model, optimizer, lr_scheduler, scaler):
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                loss = original_train_step(batch, model, optimizer, lr_scheduler)
            
            # Backward pass with scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            lr_scheduler.step()
            
            return loss
        
        return train_step_with_amp
    else:
        return train_step_func

def adapt_megatron_for_amd(args):
    """Adapt Megatron-LM for AMD GPUs."""
    logger.info("Adapting Megatron-LM for AMD GPUs")
    
    # Check AMD GPU compatibility
    if not check_amd_gpu_compatibility():
        logger.error("AMD GPUs are not compatible")
        return False
    
    # Set up AMD environment
    setup_amd_environment()
    
    # Modify args for AMD GPUs
    args.no_cuda = False
    args.use_cpu_initialization = False
    args.fp16 = args.fp16 if hasattr(args, "fp16") else True
    args.bf16 = args.bf16 if hasattr(args, "bf16") else False
    
    # Set tensor model parallel environment variables
    if args.tensor_model_parallel_size > 1:
        logger.info(f"Using tensor model parallelism with size {args.tensor_model_parallel_size}")
        os.environ["NCCL_ALGO"] = "Ring"
    
    # Set pipeline model parallel environment variables
    if args.pipeline_model_parallel_size > 1:
        logger.info(f"Using pipeline model parallelism with size {args.pipeline_model_parallel_size}")
        os.environ["NCCL_ALGO"] = "Ring"
    
    logger.info("Megatron-LM adaptation for AMD GPUs completed")
    
    return True

if __name__ == "__main__":
    # Test AMD GPU compatibility
    check_amd_gpu_compatibility()
    
    # Set up AMD environment
    setup_amd_environment()
    
    print("Megatron-LM AMD adaptation module loaded successfully")
