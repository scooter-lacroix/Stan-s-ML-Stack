# MIGraphX Optimization Guide

## Introduction

MIGraphX is AMD's graph optimization library for deep learning models. It provides a high-performance execution engine for deep learning models on AMD GPUs. This guide covers optimization techniques and best practices for using MIGraphX to accelerate inference on AMD GPUs.

## Table of Contents

1. [Overview](#overview)
2. [Model Optimization](#model-optimization)
3. [Performance Tuning](#performance-tuning)
4. [Quantization](#quantization)
5. [Memory Optimization](#memory-optimization)
6. [Operator Fusion](#operator-fusion)
7. [Benchmarking](#benchmarking)
8. [Troubleshooting](#troubleshooting)

## Overview

MIGraphX provides several key features for optimizing deep learning models:

1. **Graph Optimization**: Automatically optimizes the computational graph of the model.
2. **Operator Fusion**: Combines multiple operations into a single kernel for better performance.
3. **Memory Planning**: Efficiently allocates and reuses memory to minimize memory usage.
4. **Quantization**: Supports reduced precision (FP16, INT8) for faster inference.
5. **Target-Specific Compilation**: Compiles models for specific AMD GPU architectures.

## Model Optimization

### Converting Models to MIGraphX

MIGraphX supports importing models from various frameworks:

#### ONNX Models

```python
import migraphx

# Parse ONNX model
model = migraphx.parse_onnx("model.onnx")

# Optimize the model
model = migraphx.optimize(model)

# Compile for GPU
context = migraphx.get_gpu_context()
model.compile(context)

# Save the compiled model
model.save("model.migraphx")
```

#### TensorFlow Models

```python
import migraphx

# Parse TensorFlow model
model = migraphx.parse_tf("model.pb")

# Optimize and compile
model = migraphx.optimize(model)
context = migraphx.get_gpu_context()
model.compile(context)
```

### Optimization Levels

MIGraphX provides different optimization levels:

```python
# Default optimization
model = migraphx.optimize(model)

# Aggressive optimization (may change numerical precision)
model = migraphx.optimize(model, {"aggressiveness": 2})
```

## Performance Tuning

### Target-Specific Compilation

Compile models for specific AMD GPU architectures for best performance:

```python
# Get GPU context
context = migraphx.get_gpu_context()

# Compile with target-specific optimizations
model.compile(context, {"target": "gfx1100"})  # For RDNA 3 GPUs (RX 7000 series)
```

Available targets:
- `gfx908`: AMD Instinct MI100
- `gfx90a`: AMD Instinct MI200 series
- `gfx1030`: AMD RDNA 2 (RX 6000 series)
- `gfx1100`: AMD RDNA 3 (RX 7000 series)

### Tuning Parameters

```python
# Set tuning parameters
tuning_params = {
    "exhaustive_tune": True,  # Perform exhaustive tuning (slower compilation, faster runtime)
    "fast_math": True,        # Use fast math operations (may reduce precision)
    "use_fp16": True,         # Use FP16 where possible
    "use_nhwc": True          # Use NHWC layout for convolutional operations
}

# Compile with tuning parameters
model.compile(context, tuning_params)
```

### Batch Size Optimization

Find the optimal batch size for your model and hardware:

```python
import migraphx
import numpy as np
import time

# Load model
model = migraphx.load("model.migraphx")

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
results = {}

for batch_size in batch_sizes:
    # Create input data
    input_shape = list(model.get_parameter_shapes()[0])
    input_shape[0] = batch_size  # Set batch dimension
    input_data = np.random.random(input_shape).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        model.run({"input": migraphx.argument(input_data)})
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        model.run({"input": migraphx.argument(input_data)})
        times.append(time.time() - start)
    
    # Calculate throughput (samples/second)
    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    results[batch_size] = throughput
    
    print(f"Batch size: {batch_size}, Throughput: {throughput:.2f} samples/sec")

# Find optimal batch size
optimal_batch_size = max(results, key=results.get)
print(f"Optimal batch size: {optimal_batch_size}")
```

## Quantization

### FP16 Quantization

```python
import migraphx

# Load model
model = migraphx.parse_onnx("model.onnx")

# Quantize to FP16
model = migraphx.quantize_fp16(model)

# Compile for GPU
context = migraphx.get_gpu_context()
model.compile(context)

# Save quantized model
model.save("model_fp16.migraphx")
```

### INT8 Quantization

```python
import migraphx
import numpy as np

# Load model
model = migraphx.parse_onnx("model.onnx")

# Prepare calibration data
calibration_data = []
for i in range(100):
    # Generate representative input data
    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    calibration_data.append({"input": migraphx.argument(input_data)})

# Quantize to INT8
model = migraphx.quantize_int8(model, calibration_data)

# Compile for GPU
context = migraphx.get_gpu_context()
model.compile(context)

# Save quantized model
model.save("model_int8.migraphx")
```

## Memory Optimization

### Memory Planning

MIGraphX automatically optimizes memory usage, but you can provide hints:

```python
# Set memory planning parameters
memory_params = {
    "memory_reuse": True,      # Enable memory reuse
    "aggressive_memory": True  # Aggressively reuse memory (may affect performance)
}

# Compile with memory parameters
model.compile(context, memory_params)
```

### Reducing Memory Footprint

To reduce memory usage:

1. **Use smaller batch sizes**: Smaller batches require less memory.
2. **Quantize the model**: FP16 or INT8 quantization reduces memory usage.
3. **Optimize input/output handling**: Avoid unnecessary copies between CPU and GPU.

```python
# Efficient input/output handling
input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
input_arg = migraphx.argument(input_data)

# Reuse the same input argument for multiple runs
for _ in range(100):
    result = model.run({"input": input_arg})
    # Process result without copying back to CPU if possible
    output = np.array(result[0])
```

## Operator Fusion

MIGraphX automatically fuses compatible operators to reduce kernel launches and memory transfers.

### Viewing Fused Operators

```python
# Print the optimized program to see fused operators
print(model)
```

### Custom Fusion Rules

For advanced users, custom fusion rules can be defined:

```python
# Define custom fusion rules
fusion_rules = {
    "conv_bias_relu": True,  # Fuse convolution + bias + ReLU
    "conv_bias": True,       # Fuse convolution + bias
    "conv_relu": True        # Fuse convolution + ReLU
}

# Apply custom fusion rules during optimization
model = migraphx.optimize(model, {"fusion_rules": fusion_rules})
```

## Benchmarking

### Basic Benchmarking

```python
import migraphx
import numpy as np
import time

# Load model
model = migraphx.load("model.migraphx")

# Get input shape
input_shape = model.get_parameter_shapes()[0]
input_name = model.get_parameter_names()[0]

# Create input data
input_data = np.random.random(input_shape).astype(np.float32)
input_arg = migraphx.argument(input_data)

# Warm-up
for _ in range(10):
    model.run({input_name: input_arg})

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    model.run({input_name: input_arg})
    times.append(time.time() - start)

# Calculate statistics
avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)
std_time = np.std(times)

print(f"Average inference time: {avg_time * 1000:.2f} ms")
print(f"Min inference time: {min_time * 1000:.2f} ms")
print(f"Max inference time: {max_time * 1000:.2f} ms")
print(f"Std inference time: {std_time * 1000:.2f} ms")
```

### Comparing with PyTorch

```python
import torch
import migraphx
import numpy as np
import time
import onnx

# Load PyTorch model
pytorch_model = torch.load("model.pt")
pytorch_model.eval().cuda()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    pytorch_model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

# Load and compile MIGraphX model
migraphx_model = migraphx.parse_onnx("model.onnx")
migraphx_model = migraphx.optimize(migraphx_model)
context = migraphx.get_gpu_context()
migraphx_model.compile(context)

# Create input data
input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
torch_input = torch.tensor(input_data).cuda()
migraphx_input = migraphx.argument(input_data)

# Benchmark PyTorch
torch.cuda.synchronize()
torch_times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = pytorch_model(torch_input)
    torch.cuda.synchronize()
    torch_times.append(time.time() - start)

# Benchmark MIGraphX
migraphx_times = []
for _ in range(100):
    start = time.time()
    _ = migraphx_model.run({"input": migraphx_input})
    migraphx_times.append(time.time() - start)

# Calculate statistics
torch_avg = sum(torch_times) / len(torch_times)
migraphx_avg = sum(migraphx_times) / len(migraphx_times)

print(f"PyTorch average time: {torch_avg * 1000:.2f} ms")
print(f"MIGraphX average time: {migraphx_avg * 1000:.2f} ms")
print(f"Speedup: {torch_avg / migraphx_avg:.2f}x")
```

## Troubleshooting

### Common Issues

#### Unsupported Operators

If you encounter unsupported operators:

```
Error: Unsupported operator: Operator
```

**Solution**: Check if the operator is supported in the current MIGraphX version. If not, consider using a different operator or implementing a custom operator.

#### Memory Allocation Failures

```
Error: Failed to allocate memory
```

**Solution**: Reduce batch size, use quantization, or try a smaller model.

#### Performance Issues

If performance is lower than expected:

1. **Check GPU utilization**: Use `rocm-smi` to monitor GPU usage.
2. **Verify ROCm installation**: Ensure ROCm is properly installed and configured.
3. **Check model compilation**: Ensure the model is compiled for the correct GPU architecture.
4. **Try different batch sizes**: Find the optimal batch size for your model and hardware.
5. **Use quantization**: Try FP16 or INT8 quantization for better performance.

### Debugging Tools

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Print model information
print(model)

# Print parameter shapes
for i, name in enumerate(model.get_parameter_names()):
    shape = model.get_parameter_shapes()[i]
    print(f"Parameter {name}: {shape}")

# Print output shapes
for i, shape in enumerate(model.get_output_shapes()):
    print(f"Output {i}: {shape}")
```

## Conclusion

MIGraphX provides powerful optimization capabilities for deep learning models on AMD GPUs. By following the techniques and best practices in this guide, you can achieve significant performance improvements for your models.

For more information, refer to the [MIGraphX Documentation](https://rocm.docs.amd.com/projects/MIGraphX/en/latest/).


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

