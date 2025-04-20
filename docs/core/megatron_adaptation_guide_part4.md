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


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

