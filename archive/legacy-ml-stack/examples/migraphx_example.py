#!/usr/bin/env python3
# =============================================================================
# MIGraphX Example
# =============================================================================
# This script demonstrates how to use MIGraphX with AMD GPUs.
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

def main():
    """Main function."""
    print_header("MIGraphX Example")
    
    # Check if MIGraphX is installed
    try:
        import migraphx
        print_success("MIGraphX is installed")
    except ImportError:
        print_error("MIGraphX is not installed")
        print_info("Please install MIGraphX first")
        return False
    
    # Create a simple PyTorch model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = torch.nn.ReLU()
            self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = torch.nn.ReLU()
            self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc = torch.nn.Linear(32 * 56 * 56, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create model and set to evaluation mode
    model = SimpleModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export model to ONNX
    onnx_file = "simple_model.onnx"
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
    program = migraphx.parse_onnx(onnx_file)
    print_success("Parsed ONNX file with MIGraphX")
    
    # Print program
    print_info("Program:")
    print(program)
    
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
    
    # Benchmark PyTorch
    print_info("Benchmarking PyTorch...")
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        pytorch_time = (time.time() - start_time) / 100
    
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
    print(f"  PyTorch inference time: {pytorch_time * 1000:.2f} ms")
    print(f"  MIGraphX inference time: {migraphx_time * 1000:.2f} ms")
    print(f"  Speedup: {pytorch_time / migraphx_time:.2f}x")
    
    # Clean up
    os.remove(onnx_file)
    
    print_success("MIGraphX example completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
