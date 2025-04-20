# ML Stack Master Guide

```
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**A Comprehensive Machine Learning Stack for AMD GPUs**

## Overview

### Introduction

Welcome to the ML Stack, a comprehensive machine learning ecosystem designed specifically for AMD GPUs. This guide provides detailed information on all components of the stack, their installation, configuration, and usage.

The ML Stack is optimized for AMD Radeon RX 7900 XTX and RX 7800 XT GPUs, providing high-performance tools and frameworks for machine learning research and development. It integrates core deep learning frameworks, optimization libraries, and extension components to create a seamless experience for machine learning on AMD hardware.

### Purpose

The purpose of this ML Stack is to:

1. **Provide a Complete Ecosystem**: Offer all necessary components for machine learning on AMD GPUs
2. **Optimize Performance**: Ensure maximum performance from AMD hardware
3. **Simplify Development**: Make it easy to develop and deploy machine learning models
4. **Enable Research**: Support cutting-edge research with high-performance tools
5. **Offer Flexibility**: Allow customization for different use cases

### Components

The ML Stack consists of two main categories of components:

#### Core Components

These form the foundation of the ML Stack:

- **ROCm Platform**: The foundation for GPU computing on AMD hardware
- **PyTorch**: Deep learning framework with ROCm support
- **ONNX Runtime**: Optimized inference for ONNX models
- **MIGraphX**: AMD's graph optimization library
- **Megatron-LM**: Framework for training large language models
- **Flash Attention**: Efficient attention computation
- **RCCL**: Collective communication library for multi-GPU training
- **MPI**: Message Passing Interface for distributed computing

#### Extension Components

These provide additional functionality:

- **Triton**: Compiler for parallel programming
- **BITSANDBYTES**: Efficient quantization for deep learning models
- **vLLM**: High-throughput inference engine for LLMs
- **ROCm SMI**: System monitoring and management for AMD GPUs
- **PyTorch Profiler**: Performance analysis for PyTorch models
- **Weights & Biases**: Experiment tracking and visualization

### Architecture

The ML Stack is organized in a layered architecture:

1. **Hardware Layer**: AMD GPUs (RX 7900 XTX, RX 7800 XT)
2. **System Layer**: ROCm platform and drivers
3. **Framework Layer**: PyTorch, ONNX Runtime, MIGraphX
4. **Model Layer**: Megatron-LM, Flash Attention
5. **Communication Layer**: RCCL, MPI
6. **Extension Layer**: Triton, BITSANDBYTES, vLLM, ROCm SMI, PyTorch Profiler, WandB

This layered approach allows for flexibility and customization while ensuring all components work together seamlessly.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

