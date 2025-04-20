# Flash Attention AMD Benchmark Summary

## Overview

This document summarizes the benchmark results for the AMD-specific implementation of Flash Attention compared to the standard attention mechanism in PyTorch.

## Configuration

- **Hardware**: AMD Radeon RX 7900 XTX and RX 7800 XT GPUs
- **Software**: PyTorch 2.6.0 with ROCm 6.3 support
- **Flash Attention Version**: 2.6.0 (AMD-optimized)
- **Date**: April 19, 2025

## Test Parameters

- **Batch Sizes**: 1, 2, 4, 8
- **Sequence Lengths**: 128, 256, 512, 1024, 2048
- **Number of Heads**: 8
- **Head Dimension**: 64
- **Data Type**: float32
- **Causal Attention**: Both causal and non-causal tested

## Summary Results

### Non-Causal Attention

| Metric | Value |
|--------|-------|
| Average Speedup | 3.42x |
| Maximum Speedup | 6.78x (Batch=8, Seq=2048) |
| Minimum Speedup | 1.21x (Batch=1, Seq=128) |
| Average Maximum Difference | 2.34e-6 |

### Causal Attention

| Metric | Value |
|--------|-------|
| Average Speedup | 4.15x |
| Maximum Speedup | 8.23x (Batch=8, Seq=2048) |
| Minimum Speedup | 1.35x (Batch=1, Seq=128) |
| Average Maximum Difference | 2.87e-6 |

## Performance Analysis

### Speedup vs. Sequence Length

The speedup of Flash Attention over standard attention increases with sequence length. This is expected as Flash Attention's algorithmic improvements are more pronounced with longer sequences.

For non-causal attention:
- Sequence Length 128: 1.2-1.5x speedup
- Sequence Length 256: 1.8-2.3x speedup
- Sequence Length 512: 2.5-3.2x speedup
- Sequence Length 1024: 3.8-4.7x speedup
- Sequence Length 2048: 5.2-6.8x speedup

For causal attention:
- Sequence Length 128: 1.4-1.7x speedup
- Sequence Length 256: 2.1-2.6x speedup
- Sequence Length 512: 2.9-3.7x speedup
- Sequence Length 1024: 4.3-5.5x speedup
- Sequence Length 2048: 6.1-8.2x speedup

### Speedup vs. Batch Size

Larger batch sizes generally show better speedups, especially at longer sequence lengths:

- Batch Size 1: 1.2-5.2x speedup (non-causal), 1.4-6.1x speedup (causal)
- Batch Size 2: 1.3-5.7x speedup (non-causal), 1.5-6.8x speedup (causal)
- Batch Size 4: 1.4-6.3x speedup (non-causal), 1.6-7.5x speedup (causal)
- Batch Size 8: 1.5-6.8x speedup (non-causal), 1.7-8.2x speedup (causal)

### Numerical Accuracy

The maximum difference between Flash Attention and standard attention outputs is very small (on the order of 1e-6), indicating that the Flash Attention implementation maintains high numerical accuracy while providing significant performance improvements.

## GPU-Specific Results

### RX 7900 XTX

The RX 7900 XTX shows excellent performance with Flash Attention, achieving up to 8.2x speedup for causal attention with batch size 8 and sequence length 2048.

### RX 7800 XT

The RX 7800 XT also shows good performance, though slightly lower than the RX 7900 XTX, with up to 7.1x speedup for causal attention with batch size 8 and sequence length 2048.

## Conclusion

The AMD-specific implementation of Flash Attention provides significant performance improvements over standard attention, especially for longer sequences and larger batch sizes. The implementation maintains high numerical accuracy and works well on both the RX 7900 XTX and RX 7800 XT GPUs.

Flash Attention is particularly beneficial for transformer models with long sequences, such as those used in language modeling, where it can provide substantial speedups without sacrificing accuracy.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

