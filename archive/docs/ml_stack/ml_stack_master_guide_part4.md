## Installation

The ML stack can be installed using the provided installation scripts.

### Prerequisites

Before installation, ensure you have:

1. **ROCm Installed**: ROCm 6.3/6.4 should be installed and working
2. **Python Environment**: Python 3.8+ with pip
3. **System Dependencies**: Required system libraries and tools
4. **GPU Access**: Proper permissions to access AMD GPUs

### Core Components Installation

The core components should be installed first:

1. **ROCm**: Follow the [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
2. **PyTorch**: Install with ROCm support
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```
3. **ONNX Runtime**: Build from source with ROCm support
4. **MIGraphX**: Install from ROCm repositories
5. **Megatron-LM**: Clone and install the AMD-compatible fork
6. **Flash Attention**: Install the AMD-compatible version
7. **RCCL**: Install from ROCm repositories
8. **MPI**: Install OpenMPI with ROCm support

### Extension Components Installation

The extension components can be installed using the master installation script:

```bash
# Make the script executable
chmod +x $HOME/Desktop/Stans_MLStack/scripts/install_ml_stack_extensions.sh

# Run the installation script
$HOME/Desktop/Stans_MLStack/scripts/install_ml_stack_extensions.sh
```

The script will prompt you to select which components to install:

1. Triton - Compiler for parallel programming
2. BITSANDBYTES - Efficient quantization
3. vLLM - High-throughput inference engine
4. ROCm SMI - Monitoring and profiling
5. PyTorch Profiler - Performance analysis
6. WandB - Experiment tracking
7. All components

### Individual Component Installation

You can also install individual components using their respective installation scripts:

```bash
# Install Triton
$HOME/Desktop/Stans_MLStack/scripts/install_triton.sh

# Install BITSANDBYTES
$HOME/Desktop/Stans_MLStack/scripts/install_bitsandbytes.sh

# Install vLLM
$HOME/Desktop/Stans_MLStack/scripts/install_vllm.sh

# Install ROCm SMI
$HOME/Desktop/Stans_MLStack/scripts/install_rocm_smi.sh

# Install PyTorch Profiler
$HOME/Desktop/Stans_MLStack/scripts/install_pytorch_profiler.sh

# Install WandB
$HOME/Desktop/Stans_MLStack/scripts/install_wandb.sh
```

### Verification

After installation, verify that the components are working correctly:

```bash
# Verify PyTorch with ROCm
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# Verify ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.__version__); print(onnxruntime.get_device())"

# Verify MIGraphX
python -c "import migraphx; print(migraphx.__version__)"

# Verify Triton
python -c "import triton; print(triton.__version__)"

# Verify BITSANDBYTES
python -c "import bitsandbytes as bnb; print(bnb.__version__); print(bnb.CUDA_AVAILABLE)"

# Verify vLLM
python -c "import vllm; print(vllm.__version__)"

# Verify ROCm SMI
python -c "from rocm_smi_lib import rsmi; print('ROCm SMI available')"

# Verify WandB
python -c "import wandb; print(wandb.__version__)"
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

