# RCCL Build Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for building and installing RCCL (ROCm Collective Communication Library) for AMD GPUs. RCCL is a library of multi-GPU collective communication primitives optimized for AMD GPUs, similar to NVIDIA's NCCL.

### Purpose of this Guide

The purpose of this guide is to:

1. Provide step-by-step instructions for building RCCL from source
2. Document the configuration options for optimal performance
3. Offer usage examples for common scenarios
4. Share troubleshooting tips for common issues

### Overview of RCCL

RCCL (ROCm Collective Communication Library) is AMD's implementation of collective communication primitives for multi-GPU training. It provides the following key features:

- **Collective Operations**: AllReduce, Broadcast, Reduce, AllGather, and ReduceScatter
- **Multi-GPU Support**: Efficient communication between multiple GPUs
- **Multi-Node Support**: Communication across multiple nodes in a cluster
- **NCCL API Compatibility**: API compatible with NVIDIA's NCCL for easy porting

### Importance for Distributed Training

RCCL is essential for distributed training of deep learning models on AMD GPUs for several reasons:

1. **Efficient Communication**: Optimized for high-bandwidth, low-latency communication between GPUs
2. **Scalability**: Enables scaling to multiple GPUs and nodes
3. **Performance**: Critical for achieving good performance in distributed training
4. **Framework Integration**: Used by PyTorch, TensorFlow, and other frameworks for distributed training

## Prerequisites

Before building RCCL, ensure you have the following:

### Required Hardware

- **AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
- **System Memory**: At least 16GB of RAM
- **Storage**: At least 2GB of free disk space

### Required Software

- **ROCm**: Version 6.3/6.4 or later
- **CMake**: Version 3.5 or later
- **GCC**: Version 7.0 or later
- **Python**: Version 3.6 or later (for testing)
- **PyTorch with ROCm**: For integration testing

### Environment Setup

Before proceeding, ensure your environment is properly set up:

1. **ROCm Installation**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **Environment Variables**:
   ```bash
   # ROCm Path
   export ROCM_PATH=/opt/rocm
   export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
   
   # GPU Selection
   export HIP_VISIBLE_DEVICES=0,1  # Adjust based on your GPU configuration
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

