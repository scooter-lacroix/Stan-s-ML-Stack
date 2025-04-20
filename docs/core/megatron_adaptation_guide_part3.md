## Installation Steps

Follow these steps to install the AMD-adapted version of Megatron-LM:

### Clone the Repository

Clone the Megatron-LM repository:

```bash
cd $HOME
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

### Apply Patches

Apply the patch to remove NVIDIA-specific dependencies:

```bash
# Create the patch file
cat > remove_nvidia_deps.patch << 'EOF'
# (Patch content from previous section)
EOF

# Apply the patch
git apply remove_nvidia_deps.patch
```

### Install Dependencies

Install the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install tensorboard
pip install scipy
pip install nltk
```

### Install Megatron-LM

Install Megatron-LM in development mode:

```bash
pip install -e .
```

### Verify Installation

Verify that Megatron-LM is installed correctly:

```bash
python -c "import torch; import megatron; print('Megatron-LM imported successfully')"
```

## Configuration

Proper configuration is essential for optimal performance of Megatron-LM on AMD GPUs.

### Environment Variables

Set these environment variables for optimal performance:

```bash
# GPU Selection
export HIP_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1  # For CUDA compatibility layer
export PYTORCH_ROCM_DEVICE=0,1  # For PyTorch

# Memory Management
export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance
export GPU_MAX_HEAP_SIZE=100  # Set maximum heap size (%)
export GPU_MAX_ALLOC_PERCENT=100  # Set maximum allocation size (%)

# Distributed Training
export MASTER_ADDR=localhost
export MASTER_PORT=6000
```

### Model Configuration

Configure the model parameters for optimal performance:

```bash
# Model size parameters
HIDDEN_SIZE=1024
NUM_LAYERS=24
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048

# Training parameters
BATCH_SIZE=4
LR=1e-4
MIN_LR=1e-5
LR_DECAY_STYLE=cosine
LR_WARMUP_FRACTION=0.01
WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-8

# Parallelism parameters
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
```

### Distributed Training Setup

Configure distributed training for multi-GPU setups:

```bash
# Number of GPUs
NUM_GPUS=2

# Distributed training parameters
DISTRIBUTED_BACKEND=nccl  # Use NCCL backend (RCCL for AMD)
WORLD_SIZE=$NUM_GPUS
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

