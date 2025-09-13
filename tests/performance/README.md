# Performance Testing Guide

This directory contains performance benchmarking scripts for the ML Stack, designed to measure and track performance improvements over time.

## Overview

The performance suite includes:

- `benchmark_runner.py` - Main orchestrator that runs all benchmarks
- `system_benchmark.py` - System-level performance tests (CPU, memory, disk, GPU detection)
- `ml_stack_benchmark.py` - ML-specific benchmarks (PyTorch operations, matrix multiplication)
- `comparison_report.py` - Generates before/after comparison reports

## Running Benchmarks

### Full Benchmark Suite

Run all benchmarks with comparison reporting:

```bash
python benchmark_runner.py
```

This will:
1. Run system benchmarks
2. Run ML stack benchmarks
3. Generate comparison report with previous results
4. Save results to `benchmark_results.json`
5. Save report to `benchmark_report.md`

### Individual Benchmarks

#### System Benchmarks

Test basic system performance:

```bash
python system_benchmark.py
```

Benchmarks:
- CPU performance (computation intensive task)
- Memory bandwidth
- Disk I/O (read/write)
- GPU detection and basic info

#### ML Stack Benchmarks

Test ML-specific performance:

```bash
python ml_stack_benchmark.py
```

Benchmarks:
- Matrix multiplication (1024x1024)
- Convolution operations
- CPU-GPU memory transfer (if CUDA available)

### Comparison Reports

Generate reports comparing current results to previous runs:

```bash
python comparison_report.py
```

The system automatically saves results and generates comparisons on subsequent runs.

## Prerequisites

- Python 3.7+
- psutil (`pip install psutil`)
- PyTorch (for ML benchmarks)
- CUDA/ROCm (optional, for GPU benchmarks)

## Understanding Results

### System Benchmarks

- **CPU Benchmark Time**: Lower is better (faster computation)
- **Memory Benchmark Time**: Lower is better (faster memory access)
- **Disk I/O Times**: Lower is better (faster storage)
- **GPU Detection**: Confirms GPU availability and type

### ML Benchmarks

- **Matrix Multiplication Time**: Lower is better (faster ML operations)
- **Convolution Time**: Lower is better (faster CNN operations)
- **Memory Transfer Time**: Lower is better (faster CPU-GPU communication)

### Comparison Reports

Reports show percentage changes from previous runs:
- 📈 Positive percentage = performance degradation
- 📉 Negative percentage = performance improvement

## Baseline Establishment

On the first run, no comparison is available. Subsequent runs will show improvements or regressions.

To reset baseline, delete `previous_benchmark.json`.

## Integration with CI/CD

For automated performance monitoring:

```bash
# Run benchmarks
python benchmark_runner.py

# Check for regressions (example)
if grep -q "📈" benchmark_report.md; then
    echo "Performance regression detected!"
    exit 1
fi
```

## Troubleshooting

- **CUDA not available**: GPU benchmarks will be skipped
- **PyTorch not installed**: ML benchmarks will show errors
- **Permission denied**: Disk I/O tests need write access to /tmp
- **High variance**: Run multiple times and average results

## Custom Benchmarks

To add new benchmarks:

1. Create new benchmark function in appropriate file
2. Return results as dictionary
3. Add to the `run()` function
4. Update this documentation

Example:

```python
def benchmark_custom_operation() -> Dict[str, Any]:
    # Your benchmark code
    return {"custom_metric": value}