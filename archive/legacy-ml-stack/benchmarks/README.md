# ML Stack Benchmarks

This directory contains benchmarks for various components of the ML Stack on AMD GPUs.

## Available Benchmarks

1. **Matrix Multiplication Benchmark**: Measures the performance of matrix multiplication operations on AMD GPUs.
2. **Memory Bandwidth Benchmark**: Measures the memory bandwidth of AMD GPUs.
3. **Transformer Benchmark**: Measures the performance of transformer models on AMD GPUs.
4. **Flash Attention Benchmark**: Compares the performance of Flash Attention with standard attention on AMD GPUs.

## Running Benchmarks

### Matrix Multiplication Benchmark

```bash
python matrix_multiplication_benchmark.py --sizes 1024 2048 4096 8192 --dtype float32 --num-runs 10 --output-dir ./results
```

Options:
- `--sizes`: Matrix sizes to benchmark
- `--dtype`: Data type (float32 or float16)
- `--num-runs`: Number of runs for each benchmark
- `--output-dir`: Output directory for results
- `--device`: Device to use (cuda or cpu)

### Memory Bandwidth Benchmark

```bash
python memory_bandwidth_benchmark.py --sizes 1 2 4 8 16 32 64 128 256 512 --dtype float32 --num-runs 10 --output-dir ./results
```

Options:
- `--sizes`: Memory sizes to benchmark in MB
- `--dtype`: Data type (float32 or float16)
- `--num-runs`: Number of runs for each benchmark
- `--output-dir`: Output directory for results
- `--device`: Device to use (cuda or cpu)

### Transformer Benchmark

```bash
python transformer_benchmark.py --batch-sizes 1 2 4 8 16 32 --seq-lengths 128 256 512 1024 --d-model 512 --nhead 8 --dim-feedforward 2048 --num-layers 6 --dtype float32 --num-runs 10 --output-dir ./results
```

Options:
- `--batch-sizes`: Batch sizes to benchmark
- `--seq-lengths`: Sequence lengths to benchmark
- `--d-model`: Model dimension
- `--nhead`: Number of attention heads
- `--dim-feedforward`: Feedforward dimension
- `--num-layers`: Number of transformer layers
- `--dtype`: Data type (float32 or float16)
- `--num-runs`: Number of runs for each benchmark
- `--output-dir`: Output directory for results
- `--device`: Device to use (cuda or cpu)
- `--use-amp`: Use automatic mixed precision

### Flash Attention Benchmark

```bash
python flash_attention_benchmark.py --batch-sizes 1 2 4 8 16 --seq-lengths 128 256 512 1024 2048 4096 --num-heads 8 --head-dim 64 --causal --dtype float32 --num-runs 10 --output-dir ./results
```

Options:
- `--batch-sizes`: Batch sizes to benchmark
- `--seq-lengths`: Sequence lengths to benchmark
- `--num-heads`: Number of attention heads
- `--head-dim`: Attention head dimension
- `--causal`: Use causal attention
- `--dtype`: Data type (float32 or float16)
- `--num-runs`: Number of runs for each benchmark
- `--output-dir`: Output directory for results
- `--device`: Device to use (cuda or cpu)
- `--no-flash`: Disable Flash Attention

## Interpreting Results

Each benchmark generates CSV files and plots in the specified output directory. The plots show:

1. **Matrix Multiplication Benchmark**:
   - Matrix multiplication time vs matrix size
   - GPU speedup vs matrix size
   - Log-log scale of time vs matrix size

2. **Memory Bandwidth Benchmark**:
   - Memory bandwidth vs memory size for read, write, and read+write operations

3. **Transformer Benchmark**:
   - Forward and backward time vs batch size for different sequence lengths
   - Tokens per second vs batch size for different sequence lengths

4. **Flash Attention Benchmark**:
   - Attention time vs sequence length for standard and Flash Attention
   - Speedup of Flash Attention vs sequence length
   - Memory usage vs sequence length for standard and Flash Attention
   - Memory reduction of Flash Attention vs sequence length

## Example Results

### Matrix Multiplication

| Matrix Size | CPU Time (s) | GPU Time (s) | Speedup |
|-------------|--------------|--------------|---------|
| 1024        | 0.0123       | 0.0005       | 24.6x   |
| 2048        | 0.0982       | 0.0021       | 46.8x   |
| 4096        | 0.7856       | 0.0112       | 70.1x   |
| 8192        | 6.2848       | 0.0723       | 86.9x   |

### Flash Attention

| Sequence Length | Standard Time (ms) | Flash Time (ms) | Speedup | Memory Reduction |
|-----------------|-------------------|-----------------|---------|------------------|
| 128             | 0.42              | 0.21            | 2.0x    | 1.5x             |
| 256             | 1.23              | 0.45            | 2.7x    | 2.1x             |
| 512             | 4.56              | 1.12            | 4.1x    | 3.2x             |
| 1024            | 17.89             | 3.45            | 5.2x    | 4.8x             |
| 2048            | 71.23             | 10.78           | 6.6x    | 7.2x             |
| 4096            | 285.67            | 32.45           | 8.8x    | 10.5x            |

## Notes

- The benchmarks are designed to run on AMD GPUs with ROCm support.
- For best results, ensure that no other GPU-intensive tasks are running during the benchmarks.
- The benchmarks use the CUDA compatibility layer provided by ROCm, so they use CUDA terminology (e.g., `torch.cuda.is_available()`).
- The Flash Attention benchmark requires the Flash Attention implementation for AMD GPUs to be installed.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

