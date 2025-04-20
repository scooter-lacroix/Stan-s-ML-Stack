#!/usr/bin/env python3
# =============================================================================
# MIGraphX ONNX Example
# =============================================================================
# This script demonstrates how to use MIGraphX with ONNX models on AMD GPUs.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}INFO: {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}SUCCESS: {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}WARNING: {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}ERROR: {text}{Colors.END}")

# Define a more complex PyTorch model
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def main():
    """Main function."""
    print_header("MIGraphX ONNX Example")
    
    # Check if MIGraphX is installed
    try:
        import migraphx
        print_success("MIGraphX is installed")
    except ImportError:
        print_error("MIGraphX is not installed")
        print_info("Please install MIGraphX first")
        return False
    
    # Create a PyTorch model
    model = SimpleResNet(num_classes=10)
    model.eval()
    print_success("Created PyTorch model")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export model to ONNX
    onnx_file = "resnet_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print_success("Exported PyTorch model to ONNX")
    
    # Parse ONNX file with MIGraphX
    print_info("Parsing ONNX file with MIGraphX...")
    program = migraphx.parse_onnx(onnx_file)
    print_success("Parsed ONNX file with MIGraphX")
    
    # Compile program for GPU
    print_info("Compiling program for GPU...")
    context = migraphx.get_gpu_context()
    program.compile(context)
    print_success("Compiled program for GPU")
    
    # Get parameter shapes
    param_shapes = program.get_parameter_shapes()
    print_info("Parameter shapes:")
    for i, name in enumerate(program.get_parameter_names()):
        print(f"  {name}: {param_shapes[i]}")
    
    # Get output shapes
    output_shapes = program.get_output_shapes()
    print_info("Output shapes:")
    for i, shape in enumerate(output_shapes):
        print(f"  Output {i}: {shape}")
    
    # Create input data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference with PyTorch
    print_info("Running inference with PyTorch...")
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    
    # Run inference with MIGraphX
    print_info("Running inference with MIGraphX...")
    migraphx_input = {"input": migraphx.argument(input_data)}
    migraphx_result = program.run(migraphx_input)
    migraphx_output = np.array(migraphx_result[0])
    
    # Compare results
    print_info("Comparing results...")
    if np.allclose(pytorch_output, migraphx_output, rtol=1e-3, atol=1e-3):
        print_success("PyTorch and MIGraphX outputs match")
    else:
        print_warning("PyTorch and MIGraphX outputs differ")
        print_info(f"Max absolute difference: {np.max(np.abs(pytorch_output - migraphx_output))}")
    
    # Benchmark PyTorch CPU
    print_info("Benchmarking PyTorch CPU...")
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        pytorch_cpu_time = (time.time() - start_time) / 100
    
    # Benchmark PyTorch GPU
    print_info("Benchmarking PyTorch GPU...")
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        dummy_input_gpu = dummy_input.cuda()
        
        with torch.no_grad():
            # Warm-up
            for _ in range(10):
                _ = model_gpu(dummy_input_gpu)
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(100):
                _ = model_gpu(dummy_input_gpu)
                torch.cuda.synchronize()
            pytorch_gpu_time = (time.time() - start_time) / 100
    else:
        print_warning("CUDA is not available, skipping PyTorch GPU benchmark")
        pytorch_gpu_time = float('inf')
    
    # Benchmark MIGraphX
    print_info("Benchmarking MIGraphX...")
    # Warm-up
    for _ in range(10):
        _ = program.run(migraphx_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = program.run(migraphx_input)
    migraphx_time = (time.time() - start_time) / 100
    
    # Print benchmark results
    print_info("Benchmark results:")
    print(f"  PyTorch CPU inference time: {pytorch_cpu_time * 1000:.2f} ms")
    if torch.cuda.is_available():
        print(f"  PyTorch GPU inference time: {pytorch_gpu_time * 1000:.2f} ms")
    print(f"  MIGraphX inference time: {migraphx_time * 1000:.2f} ms")
    print(f"  Speedup over PyTorch CPU: {pytorch_cpu_time / migraphx_time:.2f}x")
    if torch.cuda.is_available():
        print(f"  Speedup over PyTorch GPU: {pytorch_gpu_time / migraphx_time:.2f}x")
    
    # Clean up
    os.remove(onnx_file)
    
    print_success("MIGraphX ONNX example completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
