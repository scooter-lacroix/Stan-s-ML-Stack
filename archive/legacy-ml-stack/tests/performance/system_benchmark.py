#!/usr/bin/env python3
"""
System-level performance benchmarking for ML Stack.
Benchmarks CPU, memory, disk I/O, and basic GPU detection.
"""

import time
import psutil
import os
import json
from typing import Dict, Any

def benchmark_cpu() -> Dict[str, Any]:
    """Benchmark CPU performance using simple computation."""
    start_time = time.time()
    # Simple CPU intensive task
    result = 0
    for i in range(1000000):
        result += i ** 2
    end_time = time.time()
    return {
        "cpu_benchmark_time": end_time - start_time,
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }

def benchmark_memory() -> Dict[str, Any]:
    """Benchmark memory bandwidth."""
    # Simple memory allocation and access
    size = 1000000
    data = [i for i in range(size)]
    start_time = time.time()
    for _ in range(10):
        sum(data)
    end_time = time.time()
    return {
        "memory_benchmark_time": end_time - start_time,
        "total_memory": psutil.virtual_memory().total,
        "available_memory": psutil.virtual_memory().available
    }

def benchmark_disk() -> Dict[str, Any]:
    """Benchmark disk I/O performance."""
    test_file = "/tmp/ml_stack_disk_benchmark.tmp"
    data = b"0" * 1024 * 1024  # 1MB
    # Write
    start_time = time.time()
    with open(test_file, "wb") as f:
        for _ in range(10):
            f.write(data)
    write_time = time.time() - start_time
    # Read
    start_time = time.time()
    with open(test_file, "rb") as f:
        for _ in range(10):
            f.read(1024 * 1024)
    read_time = time.time() - start_time
    os.remove(test_file)
    return {
        "disk_write_time": write_time,
        "disk_read_time": read_time
    }

def detect_gpu() -> Dict[str, Any]:
    """Detect GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            }
        else:
            return {"gpu_available": False}
    except ImportError:
        return {"gpu_available": False, "error": "torch not available"}

def run() -> Dict[str, Any]:
    """Run all system benchmarks."""
    results = {
        "timestamp": time.time(),
        "benchmarks": {
            "cpu": benchmark_cpu(),
            "memory": benchmark_memory(),
            "disk": benchmark_disk(),
            "gpu": detect_gpu()
        }
    }
    return results

if __name__ == "__main__":
    results = run()
    print(json.dumps(results, indent=2))