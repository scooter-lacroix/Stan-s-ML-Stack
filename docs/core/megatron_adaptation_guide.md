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
## Adaptation Process

The process of adapting Megatron-LM to AMD GPUs involves several steps:

### Forking the Repository

First, fork the Megatron-LM repository to make your own version:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
```

### Removing NVIDIA-Specific Dependencies

The main challenge is removing or replacing NVIDIA-specific dependencies. Create a patch file to remove these dependencies:

```bash
cat > remove_nvidia_deps.patch << 'EOF'
diff --git a/megatron/model/fused_softmax.py b/megatron/model/fused_softmax.py
index 7a5b2e5..3e5c2e5 100644
--- a/megatron/model/fused_softmax.py
+++ b/megatron/model/fused_softmax.py
@@ -15,7 +15,7 @@
 """Fused softmax."""
 
 import torch
-import torch.nn.functional as F
+import torch.nn.functional as F  # Use PyTorch's native implementation
 
 
 class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
@@ -24,8 +24,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             scale (float): scaling factor
 
         Returns:
@@ -33,10 +32,10 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
-        # Create a mask for the upper triangular part (including the diagonal)
+        # Create a mask for the upper triangular part
         seq_len = inputs.size(2)
         mask = torch.triu(
             torch.ones(seq_len, seq_len, device=inputs.device, dtype=torch.bool),
@@ -59,7 +58,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
@@ -77,8 +76,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, mask, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             mask (Tensor): attention mask (b, 1, sq, sk)
             scale (float): scaling factor
 
@@ -87,7 +85,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
         # Apply the mask
@@ -110,7 +108,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
diff --git a/megatron/training.py b/megatron/training.py
index 9a5b2e5..3e5c2e5 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -30,7 +30,7 @@ import torch
 from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
 
 from megatron import get_args
-from megatron import get_timers
+from megatron import get_timers  # Timing utilities
 from megatron import get_tensorboard_writer
 from megatron import mpu
 from megatron import print_rank_0
@@ -38,7 +38,7 @@ from megatron.checkpointing import load_checkpoint
 from megatron.checkpointing import save_checkpoint
 from megatron.model import DistributedDataParallel as LocalDDP
 from megatron.model import Float16Module
-from megatron.model.realm_model import ICTBertModel
+from megatron.model.realm_model import ICTBertModel  # Import model
 from megatron.utils import check_adlr_autoresume_termination
 from megatron.utils import unwrap_model
 from megatron.data.data_samplers import build_pretraining_data_loader
@@ -46,7 +46,7 @@ from megatron.utils import report_memory
 
 
 def pretrain(train_valid_test_dataset_provider, model_provider,
-             forward_step_func, extra_args_provider=None, args_defaults={}):
+             forward_step_func, extra_args_provider=None, args_defaults=None):
     """Main training program.
 
     This function will run the followings in the order provided:
@@ -59,6 +59,9 @@ def pretrain(train_valid_test_dataset_provider, model_provider,
         5) validation
 
     """
+    if args_defaults is None:
+        args_defaults = {}
+        
     # Initalize and get arguments, timers, and Tensorboard writer.
     initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
diff --git a/requirements.txt b/requirements.txt
index 9a5b2e5..3e5c2e5 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,5 @@
 torch>=1.7
 numpy
-apex
 pybind11
 regex
 nltk
EOF
```

### Patching the Code

Apply the patch to remove NVIDIA-specific dependencies:

```bash
git apply remove_nvidia_deps.patch
```

### Additional Modifications

Some additional modifications may be needed:

1. **Replace NCCL with RCCL**: Update the distributed communication code to use RCCL instead of NCCL
2. **Remove CUDA Extensions**: Replace CUDA extensions with PyTorch native implementations
3. **Update Optimizer**: Replace NVIDIA's Apex optimizer with PyTorch native optimizers

### Testing the Adaptation

Test the adaptation to ensure it works with AMD GPUs:

```bash
# Set up environment
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ROCM_DEVICE=0

# Run a simple test
python -c "import torch; import megatron; print('Megatron-LM imported successfully')"
```
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
## Usage Examples

Here are some examples of how to use Megatron-LM with AMD GPUs:

### Single-GPU Training

To train a GPT model on a single GPU:

```bash
# Set environment variables
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ROCM_DEVICE=0

# Run training
python pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --train-iters 100000 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/dataset \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```

### Multi-GPU Training

To train a GPT model on multiple GPUs:

```bash
# Set environment variables
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_DEVICE=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# Run training with torchrun
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 100000 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/dataset \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```

### Model Parallelism

To train a large model with tensor model parallelism:

```bash
# Set environment variables
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_DEVICE=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# Run training with torchrun and tensor model parallelism
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 100000 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/dataset \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```

### Pipeline Parallelism

To train a very large model with pipeline parallelism:

```bash
# Set environment variables
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_DEVICE=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# Run training with torchrun and pipeline parallelism
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --pipeline-model-parallel-size 2 \
    --num-layers 48 \
    --hidden-size 1536 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 100000 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --data-path /path/to/dataset \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```
## Performance Optimization

Optimizing performance is crucial for efficient training of large language models on AMD GPUs.

### Memory Optimization

1. **Gradient Checkpointing**: Trade computation for memory
   ```bash
   # Enable gradient checkpointing
   --checkpoint-activations
   ```

2. **Mixed Precision Training**: Use FP16 for reduced memory usage
   ```bash
   # Enable mixed precision training
   --fp16
   ```

3. **Batch Size Optimization**: Find the optimal batch size for your GPU memory
   ```bash
   # Set micro batch size and global batch size
   --micro-batch-size 4
   --global-batch-size 32
   ```

4. **Memory Fragmentation**: Clear cache periodically
   ```python
   # Clear cache
   torch.cuda.empty_cache()
   ```

### Computation Optimization

1. **Kernel Selection**: Use optimized kernels for AMD GPUs
   ```bash
   # Set environment variables for kernel selection
   export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
   export MIOPEN_FIND_MODE=3
   ```

2. **Operator Fusion**: Fuse operations when possible
   ```bash
   # Enable operator fusion
   --fused-bias-gelu
   --fused-bias-mha
   ```

3. **Custom Kernels**: Use optimized kernels for critical operations
   ```bash
   # Enable custom kernels
   --use-flash-attn
   ```

### Communication Optimization

1. **RCCL Tuning**: Optimize RCCL parameters for AMD GPUs
   ```bash
   # Set NCCL parameters
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1
   export NCCL_SOCKET_IFNAME=eth0
   ```

2. **Gradient Accumulation**: Accumulate gradients for larger effective batch sizes
   ```bash
   # Set gradient accumulation steps
   --gradient-accumulation-steps 8
   ```

3. **Overlap Communication and Computation**: Overlap communication with computation
   ```bash
   # Enable overlapping communication and computation
   --overlap-comm
   ```

### Mixed Precision Training

Configure mixed precision training for optimal performance:

```bash
# Enable mixed precision training
--fp16

# Set loss scaling parameters
--loss-scale 0  # Use dynamic loss scaling
--initial-loss-scale 4096
--min-loss-scale 1
--loss-scale-window 1000
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```
   No CUDA GPUs are available
   ```
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
   - Check permissions: `groups` (should include video or render)
   - Update drivers: `sudo apt update && sudo apt upgrade`

2. **Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce batch size
   - Use gradient checkpointing: `--checkpoint-activations`
   - Use mixed precision training: `--fp16`
   - Reduce model size or sequence length
   - Use model parallelism: `--tensor-model-parallel-size 2`

3. **Slow Training**:
   ```
   Training is slower than expected
   ```
   
   Solutions:
   - Profile with PyTorch Profiler
   - Check GPU utilization with ROCm SMI
   - Optimize data loading (more workers, pin_memory)
   - Use optimized kernels and operations
   - Check for CPU bottlenecks

4. **Distributed Training Issues**:
   ```
   Process group initialization failed
   ```
   
   Solutions:
   - Check RCCL installation
   - Verify environment variables: `MASTER_ADDR`, `MASTER_PORT`
   - Check network connectivity between nodes
   - Use a different backend: `--distributed-backend gloo`

### Debugging Tips

1. **Enable Verbose Logging**:
   ```bash
   # Enable verbose logging
   --log-level debug
   ```

2. **Check GPU Utilization**:
   ```bash
   # Monitor GPU utilization
   watch -n 1 rocm-smi
   ```

3. **Profile with PyTorch Profiler**:
   ```python
   # Profile with PyTorch Profiler
   from torch.profiler import profile, record_function, ProfilerActivity
   
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           output = model(input)
   
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   ```

4. **Check Memory Usage**:
   ```python
   # Check memory usage
   print(torch.cuda.memory_summary())
   ```
## References

### Documentation Links

- [Megatron-LM GitHub Repository](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM/tree/main/docs)
- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [RCCL Documentation](https://github.com/ROCmSoftwarePlatform/rccl)

### Community Resources

- [PyTorch Forums](https://discuss.pytorch.org/)
- [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)
- [AMD Developer Forums](https://community.amd.com/t5/AMD-ROCm/bd-p/amd-rocm)

### Papers and Articles

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

## Conclusion

Adapting Megatron-LM to work with AMD GPUs requires some effort, but it is possible to achieve good performance with the right modifications and optimizations. By removing NVIDIA-specific dependencies, replacing them with AMD equivalents, and optimizing for AMD's architecture, you can train large language models efficiently on AMD GPUs.

The key points to remember are:

1. **Remove NVIDIA-Specific Dependencies**: Replace NCCL with RCCL, remove CUDA extensions, and use PyTorch native implementations
2. **Optimize for AMD GPUs**: Set the right environment variables, use optimized kernels, and configure for optimal performance
3. **Use Model and Pipeline Parallelism**: For large models that don't fit in a single GPU's memory
4. **Monitor and Debug**: Use ROCm SMI, PyTorch Profiler, and other tools to monitor performance and debug issues

With these adaptations, you can leverage the power of AMD GPUs for training large language models with Megatron-LM.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

