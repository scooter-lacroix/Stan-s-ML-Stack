# Megatron-LM for AMD GPUs

## Introduction

Megatron-LM is a powerful framework for training large language models (LLMs) with model parallelism. This guide covers the adaptation and optimization of Megatron-LM for AMD GPUs using ROCm, enabling efficient training of large models on AMD hardware.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Adapting Megatron-LM for AMD](#adapting-megatron-lm-for-amd)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Performance Optimization](#performance-optimization)
7. [Distributed Training](#distributed-training)
8. [Troubleshooting](#troubleshooting)

## Overview

Megatron-LM provides several key features for training large language models:

1. **Model Parallelism**: Splits model layers across multiple GPUs.
2. **Pipeline Parallelism**: Splits the model into stages that run on different GPUs.
3. **Data Parallelism**: Processes different batches of data on different GPUs.
4. **Mixed Precision Training**: Uses FP16/BF16 for faster training.
5. **Checkpointing**: Saves and loads model states for resuming training.

The AMD adaptation ensures these features work efficiently on AMD GPUs with ROCm.

## Installation

### Prerequisites

- AMD GPU with ROCm support (RX 7000 series or Instinct MI series recommended)
- ROCm 6.0 or higher
- PyTorch 2.0 or higher with ROCm support
- RCCL (ROCm Communication Collective Library)
- MPI (Message Passing Interface)

### Installation Steps

#### Option 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Apply AMD adaptation patch
git apply /home/stan/Desktop/Stans_MLStack/core/megatron/remove_nvidia_deps.patch

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

#### Option 2: Use the Installation Script

```bash
# Navigate to the scripts directory
cd /home/stan/Desktop/Stans_MLStack/scripts

# Run the installation script
./install_megatron_amd.sh
```

### Verifying Installation

To verify that Megatron-LM is installed correctly:

```python
import megatron
from megatron import initialize_megatron
from megatron.arguments import parse_args

# Initialize Megatron
args = parse_args([])
initialize_megatron(args)

print("Megatron-LM is installed correctly!")
```

## Adapting Megatron-LM for AMD

The main adaptations for AMD GPUs include:

1. **Removing NVIDIA-specific dependencies**: Replacing NVIDIA Apex with PyTorch native AMP.
2. **Using ROCm-compatible libraries**: Replacing NCCL with RCCL for communication.
3. **Optimizing CUDA kernels**: Adapting or replacing CUDA kernels for ROCm compatibility.
4. **Using Flash Attention AMD**: Integrating the AMD-specific Flash Attention implementation.

### Key Changes

#### Replacing Apex with PyTorch AMP

```python
# Original NVIDIA code
from apex import amp
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

# AMD adaptation
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(input_ids, labels=input_ids)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Using RCCL for Communication

```python
# Original NVIDIA code
torch.distributed.init_process_group(backend='nccl')

# AMD adaptation
torch.distributed.init_process_group(backend='nccl')  # RCCL is used automatically on ROCm
```

#### Adapting CUDA Kernels

```python
# Original NVIDIA code
from megatron.fused_kernels import load_fused_kernels
load_fused_kernels()

# AMD adaptation
try:
    from megatron.fused_kernels import load_fused_kernels
    load_fused_kernels()
except ImportError:
    print("Fused kernels not available, using PyTorch implementations")
    # Use PyTorch native implementations instead
```

## Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Set visible devices
export HIP_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, 3

# RCCL configuration
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo  # Use all interfaces except loopback

# MPI configuration
export OMPI_MCA_btl=^openib  # Disable OpenIB
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx

# ROCm configuration
export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance
```

### Model Configuration

Example configuration for a 1B parameter model:

```bash
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --fp16 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2
```

## Training

### Data Preparation

Prepare your dataset in the format expected by Megatron-LM:

```bash
# Convert raw text to Megatron-LM format
python tools/preprocess_data.py \
       --input my_dataset.txt \
       --output-prefix my_dataset \
       --vocab-file vocab.json \
       --merge-file merges.txt \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --workers 8
```

### Training a GPT Model

```bash
# Start training
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --fp16 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2
```

### Training a BERT Model

```bash
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 2000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --fp16 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2
```

## Performance Optimization

### Flash Attention Integration

To use Flash Attention with Megatron-LM:

```python
# In megatron/model/transformer.py
from flash_attention_amd import FlashAttention

class SelfAttention(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
        self.flash_attn = FlashAttention(attention_dropout=attention_dropout)
        
    def forward(self, query, key, value, attention_mask):
        ...
        if hasattr(self, 'flash_attn') and not self.sequence_parallel:
            # Reshape for Flash Attention
            q = q.reshape(batch_size, seq_length, self.num_attention_heads, self.hidden_size_per_attention_head)
            k = k.reshape(batch_size, seq_length, self.num_attention_heads, self.hidden_size_per_attention_head)
            v = v.reshape(batch_size, seq_length, self.num_attention_heads, self.hidden_size_per_attention_head)
            
            # Apply Flash Attention
            context_layer = self.flash_attn(q, k, v, causal=True)
            
            # Reshape back
            context_layer = context_layer.reshape(batch_size, seq_length, self.hidden_size)
        else:
            # Original attention implementation
            ...
        
        return context_layer
```

### Memory Optimization

```python
# Set memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory

# Optimize for specific workloads
torch.cuda.max_split_size_mb = 512  # Optimal for RX 7900 XTX

# Use gradient checkpointing to reduce memory usage
args.checkpoint_activations = True
args.checkpoint_num_layers = 1
```

### Mixed Precision Training

```bash
# Use FP16 mixed precision
--fp16

# Or use BF16 mixed precision (if supported by your GPU)
--bf16
```

## Distributed Training

### Multi-Node Training

For training across multiple nodes:

```bash
# On node 0 (master)
MASTER_ADDR=<master-node-ip>
MASTER_PORT=6000
NODE_RANK=0

# On node 1
MASTER_ADDR=<master-node-ip>
MASTER_PORT=6000
NODE_RANK=1

# And so on for other nodes
```

### Model Parallelism Configuration

Choose the right parallelism strategy based on your model size and hardware:

| Model Size | GPUs | Tensor Parallel | Pipeline Parallel | Data Parallel |
|------------|------|-----------------|-------------------|---------------|
| 1B         | 4    | 2               | 2                 | 1             |
| 7B         | 8    | 4               | 2                 | 1             |
| 13B        | 8    | 8               | 1                 | 1             |
| 30B        | 16   | 8               | 2                 | 1             |
| 70B        | 32   | 8               | 4                 | 1             |

## Troubleshooting

### Common Issues

#### RCCL Errors

```
NCCL error: unhandled system error
```

**Solution**: Check network configuration and ensure RCCL is properly installed.

```bash
# Check RCCL installation
ls -la /opt/rocm/lib/librccl*

# Set debug mode
export NCCL_DEBUG=INFO
```

#### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size, use gradient checkpointing, or increase model parallelism.

```bash
# Use gradient checkpointing
--checkpoint-activations

# Reduce micro-batch size
--micro-batch-size 2

# Increase model parallelism
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
```

#### Performance Issues

If training is slower than expected:

1. **Check GPU utilization**: Use `rocm-smi` to monitor GPU usage.
2. **Verify ROCm installation**: Ensure ROCm is properly installed and configured.
3. **Optimize batch size**: Find the optimal batch size for your model and hardware.
4. **Use Flash Attention**: Enable Flash Attention for faster attention computation.
5. **Tune learning rate**: Adjust learning rate based on batch size and model size.

### Debugging Tools

```bash
# Enable verbose logging
--log-level trace

# Profile training
python -m torch.utils.bottleneck pretrain_gpt.py [args...]

# Use PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Run training step
    ...

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Conclusion

Adapting Megatron-LM for AMD GPUs enables efficient training of large language models on AMD hardware. By following the techniques and best practices in this guide, you can achieve good performance and scalability for your language model training workloads.

For more information, refer to the [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM) and the AMD adaptation patches in the ML Stack.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

