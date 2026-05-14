## Core Components

The core components form the foundation of the ML stack, providing essential functionality for machine learning on AMD GPUs.

### ROCm Platform

The Radeon Open Compute (ROCm) platform is the foundation for GPU computing on AMD hardware.

- **Version**: ROCm 6.3/6.4
- **Key Features**: 
  - HIP programming model (CUDA compatibility layer)
  - Performance libraries (rocBLAS, MIOpen, etc.)
  - Tools and utilities for GPU management
- **Documentation**: [ROCm Guide](/docs/core/rocm_installation_guide.md)
- **Installation Script**: `/scripts/install_rocm.sh`
- **Use Cases**: All GPU computing tasks

### PyTorch

PyTorch is the primary deep learning framework, with ROCm support for AMD GPUs.

- **Version**: PyTorch 2.6.0+rocm6.2.4
- **Key Features**: 
  - Dynamic computation graph
  - CUDA compatibility layer
  - Optimized kernels for AMD GPUs
  - Comprehensive neural network library
- **Documentation**: [PyTorch Guide](/docs/core/pytorch_rocm_guide.md)
- **Installation Script**: `/scripts/install_pytorch.sh`
- **Use Cases**: Neural network training and inference

### ONNX Runtime

ONNX Runtime provides optimized inference for ONNX models on AMD GPUs.

- **Version**: Latest with ROCm support
- **Key Features**: 
  - Model optimization
  - Cross-platform compatibility
  - ROCm backend for AMD GPUs
  - Quantization support
- **Documentation**: [ONNX Runtime Guide](/docs/core/onnxruntime_build_guide.md)
- **Installation Script**: `/scripts/build_onnxruntime.sh`
- **Use Cases**: Model inference, model deployment

### MIGraphX

MIGraphX is AMD's graph optimization library for deep learning.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - Graph optimization
  - Operator fusion
  - Quantization support
  - Performance tuning
- **Documentation**: [MIGraphX Guide](/docs/core/migraphx_build_guide.md)
- **Installation Script**: `/scripts/install_migraphx.sh`
- **Use Cases**: Model optimization, inference acceleration

### Megatron-LM

Megatron-LM is a framework for training large language models, modified for AMD GPUs.

- **Version**: Custom fork with AMD support
- **Key Features**: 
  - Model parallelism
  - Distributed training
  - Optimized for large language models
  - Memory optimization techniques
- **Documentation**: [Megatron-LM Guide](/docs/core/megatron_adaptation_guide.md)
- **Installation Script**: `/scripts/install_megatron.sh`
- **Use Cases**: Large language model training

### Flash Attention

Flash Attention provides efficient attention computation for transformer models.

- **Version**: Flash Attention 2 (v2.6)
- **Key Features**: 
  - Memory-efficient attention
  - Optimized for AMD GPUs
  - Significant speedup for transformer models
  - Support for different attention patterns
- **Documentation**: [Flash Attention Guide](/docs/core/flash_attention_build_guide.md)
- **Installation Script**: `/scripts/build_flash_attn_amd.sh`
- **Use Cases**: Transformer model training and inference

### RCCL

ROCm Collective Communication Library (RCCL) enables multi-GPU communication.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - Collective operations for distributed training
  - Optimized for AMD GPU interconnect
  - NCCL-compatible API
  - Support for various communication patterns
- **Documentation**: [RCCL Guide](/docs/core/rccl_build_guide.md)
- **Installation Script**: `/scripts/install_rccl.sh`
- **Use Cases**: Multi-GPU training, distributed computing

### MPI

Message Passing Interface (MPI) provides distributed computing capabilities.

- **Version**: OpenMPI with ROCm support
- **Key Features**: 
  - Process communication
  - Job management
  - Integration with RCCL
  - Scalable distributed computing
- **Documentation**: [MPI Guide](/docs/core/mpi_installation_guide.md)
- **Installation Script**: `/scripts/install_mpi.sh`
- **Use Cases**: Distributed training, multi-node computing


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

