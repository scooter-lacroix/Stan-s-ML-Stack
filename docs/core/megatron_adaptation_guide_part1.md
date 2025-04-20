# Megatron-LM Adaptation Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for adapting Megatron-LM to work with AMD GPUs using ROCm. Megatron-LM is a powerful framework for training large language models, originally developed for NVIDIA GPUs. With some modifications, it can be adapted to work with AMD GPUs, enabling efficient training of large language models on AMD hardware.

### Purpose of this Guide

The purpose of this guide is to:

1. Provide step-by-step instructions for adapting Megatron-LM to AMD GPUs
2. Document the necessary code changes and patches
3. Offer configuration guidance for optimal performance
4. Share troubleshooting tips for common issues

### Overview of Megatron-LM

Megatron-LM is a framework for training large language models with model and pipeline parallelism. It was developed by NVIDIA and is optimized for their GPUs. Key features include:

- Model parallelism for training large models that don't fit in a single GPU's memory
- Pipeline parallelism for efficient multi-GPU training
- Optimized transformer implementation
- Support for pre-training and fine-tuning
- Integration with NVIDIA's libraries for optimal performance

### Challenges with AMD Adaptation

Adapting Megatron-LM to AMD GPUs presents several challenges:

1. **NVIDIA-Specific Dependencies**: Megatron-LM relies on NVIDIA-specific libraries like NCCL, cuDNN, and CUDA extensions
2. **CUDA Code**: Some parts of the codebase use CUDA directly
3. **Performance Optimization**: The code is optimized for NVIDIA's architecture
4. **Library Compatibility**: Some libraries used by Megatron-LM may not have AMD equivalents

Despite these challenges, it is possible to adapt Megatron-LM to work with AMD GPUs by replacing NVIDIA-specific components with AMD equivalents and modifying the code to use ROCm instead of CUDA.

## Prerequisites

Before adapting Megatron-LM to AMD GPUs, ensure you have the following:

### Required Hardware

- **AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
- **System Memory**: At least 32GB of RAM
- **Storage**: At least 100GB of free disk space

### Required Software

- **ROCm**: Version 6.3/6.4 or later
- **PyTorch with ROCm**: Version 2.6.0 or later
- **Python**: Version 3.8 or later
- **RCCL**: ROCm Collective Communication Library
- **MPI**: Message Passing Interface (OpenMPI with ROCm support)

### Environment Setup

Before proceeding, ensure your environment is properly set up:

1. **ROCm Installation**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **PyTorch with ROCm**: Follow the [PyTorch ROCm Guide](/docs/core/pytorch_rocm_guide.md)
3. **Environment Variables**:
   ```bash
   # GPU Selection
   export HIP_VISIBLE_DEVICES=0,1  # Adjust based on your GPU configuration
   export CUDA_VISIBLE_DEVICES=0,1  # For CUDA compatibility layer
   export PYTORCH_ROCM_DEVICE=0,1  # For PyTorch
   
   # Performance Settings
   export HSA_ENABLE_SDMA=0
   export GPU_MAX_HEAP_SIZE=100
   export GPU_MAX_ALLOC_PERCENT=100
   ```


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

