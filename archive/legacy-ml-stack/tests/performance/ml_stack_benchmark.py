#!/usr/bin/env python3
"""
ML Stack specific performance benchmarking.
Benchmarks PyTorch operations, matrix multiplication, and GPU performance.
"""

import time
import json
from typing import Dict, Any

def benchmark_pytorch_matrix_mult() -> Dict[str, Any]:
    """Benchmark matrix multiplication using PyTorch."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = 1024
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm up
        for _ in range(5):
            c = torch.mm(a, b)

        # Benchmark
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        end_time = time.time()

        return {
            "matrix_mult_time": end_time - start_time,
            "device": str(device),
            "matrix_size": size
        }
    except ImportError:
        return {"error": "PyTorch not available"}

def benchmark_pytorch_conv() -> Dict[str, Any]:
    """Benchmark convolution operation."""
    try:
        import torch
        import torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).to(device)
        input_tensor = torch.randn(1, 3, 224, 224, device=device)

        # Warm up
        for _ in range(5):
            output = model(input_tensor)

        # Benchmark
        start_time = time.time()
        for _ in range(10):
            output = model(input_tensor)
        end_time = time.time()

        return {
            "conv_time": end_time - start_time,
            "device": str(device),
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape)
        }
    except ImportError:
        return {"error": "PyTorch not available"}

def benchmark_memory_transfer() -> Dict[str, Any]:
    """Benchmark CPU to GPU memory transfer."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        size = 1024 * 1024 * 100  # 100M elements
        data = torch.randn(size)

        start_time = time.time()
        for _ in range(10):
            gpu_data = data.cuda()
            cpu_data = gpu_data.cpu()
        end_time = time.time()

        return {
            "memory_transfer_time": end_time - start_time,
            "data_size_mb": size * 4 / (1024 * 1024)  # float32
        }
    except ImportError:
        return {"error": "PyTorch not available"}

def run() -> Dict[str, Any]:
    """Run all ML stack benchmarks."""
    results = {
        "timestamp": time.time(),
        "benchmarks": {
            "matrix_multiplication": benchmark_pytorch_matrix_mult(),
            "convolution": benchmark_pytorch_conv(),
            "memory_transfer": benchmark_memory_transfer()
        }
    }
    return results

if __name__ == "__main__":
    results = run()
    print(json.dumps(results, indent=2))