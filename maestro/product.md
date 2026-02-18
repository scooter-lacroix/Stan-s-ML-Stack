# Rusty Stack - Product Guide

## Initial Concept

Rusty Stack (formerly Stan's ML Stack) is a comprehensive machine learning environment optimized for AMD GPUs with ROCm support. The project provides installation tools, core ML components, and utilities for training and deploying ML models on AMD hardware.

## Product Vision

Rusty Stack aims to be the definitive solution for AMD GPU machine learning, lowering the barrier to entry and providing enterprise-grade tooling for the AMD ecosystem.

## Target Users

- **ML Researchers**: Researchers working on AMD GPUs who need optimized environments for experimentation and publication
- **ML Engineers**: Developers deploying ML models on AMD hardware in production environments
- **Hobbyists/Enthusiasts**: Individual users setting up local AMD GPU environments for personal projects and learning
- **Enterprise/DevOps Teams**: Organizations deploying AMD GPU infrastructure at scale

## Primary Goals

1. **Simplify AMD GPU ML Setup**: Automate and streamline the installation of ROCm, PyTorch, and associated ML tools
2. **Performance Optimization**: Provide AMD-optimized implementations of core ML algorithms and attention mechanisms
3. **Benchmarking & Metrics**: Offer comprehensive benchmarking infrastructure to measure and compare GPU performance
4. **Multi-Version ROCm Support**: Support multiple ROCm channels (Legacy, Stable, Latest) to balance stability and cutting-edge features
5. **Multi-Platform Support**: Enable cross-Linux distro compatibility with Windows support as the next target platform

## Key Features

### Interactive TUI Installer
A modern Rust-based terminal user interface (ratatui + crossterm) that provides:
- Hardware detection (GPU, ROCm version)
- Guided component selection
- Real-time installation progress
- Pre/post installation benchmarking

### Flash Attention Support
High-performance attention mechanisms with:
- Triton-based kernels
- Composable Kernel (CK) variant
- Optimized for RDNA 2/3/4 architectures

### LLM Training & Inference
Comprehensive large language model support:
- Megatron-LM integration for distributed training
- vLLM for high-throughput inference
- DeepSpeed for memory-efficient training
- bitsandbytes for quantization

### ComfyUI Integration
Node-based AI image generation UI with full ROCm GPU acceleration.

## Problems Solved

### Complex ROCm Installation
ROCm installation has traditionally been complex and error-prone. Rusty Stack automates this process with intelligent hardware detection and channel selection.

### Fragmented Tool Ecosystem
The AMD ML ecosystem lacks a unified installer. Rusty Stack provides a single entry point for installing PyTorch, ONNX Runtime, MIGraphX, Flash Attention, RCCL, MPI, and all associated extensions.

### ROCm Version Management
Managing multiple ROCm versions is difficult. Rusty Stack's multi-channel approach allows users to choose between Legacy (6.4.3), Stable (7.1), and Latest (7.2) based on their stability needs.

## Future Vision

### Windows Platform Support
Extend support to Windows operating system, enabling a broader user base to leverage AMD GPUs for ML workloads.

### Extended Library Support
Continue adding AMD-optimized ML libraries and tools as the ecosystem evolves.
