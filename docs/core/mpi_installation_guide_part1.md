# MPI Installation Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for installing and configuring MPI (Message Passing Interface) for use with AMD GPUs. MPI is a standardized and portable message-passing system designed for parallel computing, essential for distributed training of machine learning models across multiple nodes.

### Purpose of this Guide

The purpose of this guide is to:

1. Provide step-by-step instructions for installing MPI with ROCm support
2. Document the configuration options for optimal performance
3. Offer usage examples for common scenarios
4. Share troubleshooting tips for common issues

### Overview of MPI

MPI (Message Passing Interface) is a standardized and portable message-passing system designed for parallel computing. Key features include:

- **Standardized API**: Well-defined interface for message passing
- **Portability**: Works across different hardware and operating systems
- **Scalability**: Scales from a few processes to thousands
- **Performance**: Optimized for high-performance computing
- **Flexibility**: Supports various communication patterns

The most common implementations of MPI are:

- **OpenMPI**: Open-source implementation with broad hardware support
- **MPICH**: Another popular open-source implementation
- **Intel MPI**: Optimized for Intel hardware
- **Microsoft MPI**: Windows-specific implementation

For AMD GPUs, OpenMPI is the recommended implementation due to its good support for ROCm.

### Importance for Distributed Training

MPI is essential for distributed training of machine learning models for several reasons:

1. **Process Communication**: Enables communication between processes across multiple nodes
2. **Data Parallelism**: Facilitates data-parallel training across multiple GPUs and nodes
3. **Model Parallelism**: Supports model-parallel training for large models
4. **Scalability**: Allows scaling to hundreds or thousands of GPUs
5. **Integration**: Works well with other distributed training libraries like Horovod

## Prerequisites

Before installing MPI, ensure you have the following:

### Required Hardware

- **AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
- **System Memory**: At least 16GB of RAM
- **Network**: High-speed network connection (preferably InfiniBand or high-speed Ethernet)

### Required Software

- **ROCm**: Version 6.3/6.4 or later
- **GCC**: Version 7.0 or later
- **Python**: Version 3.6 or later (for integration with PyTorch)
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

