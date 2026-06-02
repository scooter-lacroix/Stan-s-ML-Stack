## Extension Components

The extension components enhance the core stack with additional functionality.

### Triton

Triton is an open-source language and compiler for parallel programming that can generate highly optimized GPU kernels.

- **Version**: 2.2.0
- **Key Features**: 
  - Python-based DSL for GPU programming
  - Automatic optimization of kernels
  - Integration with PyTorch
  - Support for AMD GPUs through ROCm
- **Documentation**: [Triton Guide](/docs/extensions/triton_guide.md)
- **Installation Script**: `/scripts/install_triton.sh`
- **Use Cases**: Custom operators, performance-critical operations

### BITSANDBYTES

BITSANDBYTES provides efficient quantization for deep learning models, reducing memory usage and improving inference speed.

- **Version**: 0.41.3
- **Key Features**: 
  - 8-bit quantization for linear layers
  - 4-bit quantization for weights
  - Memory-efficient optimizers
  - ROCm support for AMD GPUs
- **Documentation**: [BITSANDBYTES Guide](/docs/extensions/bitsandbytes_guide.md)
- **Installation Script**: `/scripts/install_bitsandbytes.sh`
- **Use Cases**: Large model inference, memory-constrained environments

### vLLM

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs).

- **Version**: 0.3.0
- **Key Features**: 
  - PagedAttention for memory efficiency
  - Continuous batching for high throughput
  - Tensor parallelism for multi-GPU inference
  - ROCm support for AMD GPUs
- **Documentation**: [vLLM Guide](/docs/extensions/vllm_guide.md)
- **Installation Script**: `/scripts/install_vllm.sh`
- **Use Cases**: LLM inference, model serving

### ROCm SMI

ROCm System Management Interface (ROCm SMI) provides monitoring and management capabilities for AMD GPUs.

- **Version**: Latest compatible with ROCm
- **Key Features**: 
  - GPU monitoring (utilization, temperature, memory)
  - Performance control (clock speeds, power limits)
  - Python API for programmatic access
  - Integration with ML workflows
- **Documentation**: [ROCm SMI Guide](/docs/extensions/rocm_smi_guide.md)
- **Installation Script**: `/scripts/install_rocm_smi.sh`
- **Use Cases**: Performance monitoring, thermal management

### PyTorch Profiler

PyTorch Profiler provides detailed performance analysis for PyTorch models.

- **Version**: Included with PyTorch
- **Key Features**: 
  - Operator-level profiling
  - Memory profiling
  - TensorBoard integration
  - ROCm support for AMD GPUs
- **Documentation**: [PyTorch Profiler Guide](/docs/extensions/pytorch_profiler_guide.md)
- **Installation Script**: `/scripts/install_pytorch_profiler.sh`
- **Use Cases**: Performance optimization, bottleneck identification

### Weights & Biases (WandB)

Weights & Biases (WandB) is an experiment tracking platform for machine learning.

- **Version**: Latest
- **Key Features**: 
  - Experiment tracking
  - Hyperparameter optimization
  - Artifact management
  - Visualization
  - Team collaboration
- **Documentation**: [WandB Guide](/docs/extensions/wandb_guide.md)
- **Installation Script**: `/scripts/install_wandb.sh`
- **Use Cases**: Experiment tracking, collaboration, visualization


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

