#!/usr/bin/env python3
# =============================================================================
# MIGraphX Test
# =============================================================================
# This script tests MIGraphX functionality on AMD GPUs.
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

import torch
import sys
import os
import time
import numpy as np

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

def test_migraphx():
    """Test MIGraphX functionality on AMD GPUs."""
    print_header("MIGraphX Test")
    
    # Check if MIGraphX is installed
    try:
        import migraphx
        print_success("MIGraphX is installed")
    except ImportError:
        print_error("MIGraphX is not installed")
        print_info("Please install MIGraphX first")
        return False
    
    # Check MIGraphX version
    try:
        migraphx_version = migraphx.__version__
        print_info(f"MIGraphX version: {migraphx_version}")
    except AttributeError:
        print_warning("Could not determine MIGraphX version")
    
    # Test MIGraphX basic functionality
    print_info("Testing MIGraphX basic functionality...")
    
    try:
        # Create a MIGraphX program
        program = migraphx.program()
        print_success("Created MIGraphX program")
        
        # Create a MIGraphX shape
        shape = migraphx.shape(migraphx.shape.float_type, [1, 3, 224, 224])
        print_success("Created MIGraphX shape")
        
        # Add a parameter to the program
        param = program.add_parameter("input", shape)
        print_success("Added parameter to program")
        
        # Add a simple operation (relu)
        relu = program.add_instruction(migraphx.op("relu"), [param])
        print_success("Added relu operation to program")
        
        # Compile the program for GPU
        context = migraphx.get_gpu_context()
        program.compile(context)
        print_success("Compiled program for GPU")
        
        # Create input data
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run the program
        result = program.run({"input": migraphx.argument(input_data)})
        print_success("Ran program on GPU")
        
        # Check result
        output = np.array(result[0])
        expected = np.maximum(input_data, 0)
        
        if np.allclose(output, expected):
            print_success("Result is correct")
        else:
            print_error("Result is incorrect")
            return False
        
    except Exception as e:
        print_error(f"MIGraphX basic functionality test failed: {e}")
        return False
    
    # Test MIGraphX with ONNX
    print_info("Testing MIGraphX with ONNX...")
    
    try:
        # Create a simple PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.pool(x)
                return x
        
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
        
        # Compile program for GPU
        context = migraphx.get_gpu_context()
        program.compile(context)
        print_success("Compiled ONNX model for GPU")
        
        # Run inference with PyTorch
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()
        
        # Run inference with MIGraphX
        migraphx_input = {"input": migraphx.argument(dummy_input.numpy())}
        migraphx_result = program.run(migraphx_input)
        migraphx_output = np.array(migraphx_result[0])
        
        # Compare results
        if np.allclose(pytorch_output, migraphx_output, rtol=1e-3, atol=1e-3):
            print_success("PyTorch and MIGraphX outputs match")
        else:
            print_warning("PyTorch and MIGraphX outputs differ")
            print_info(f"Max absolute difference: {np.max(np.abs(pytorch_output - migraphx_output))}")
        
        # Benchmark PyTorch
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(100):
                _ = model(dummy_input.cuda())
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / 100
        
        # Benchmark MIGraphX
        start_time = time.time()
        for _ in range(100):
            _ = program.run(migraphx_input)
        migraphx_time = (time.time() - start_time) / 100
        
        print_info(f"PyTorch inference time: {pytorch_time * 1000:.2f} ms")
        print_info(f"MIGraphX inference time: {migraphx_time * 1000:.2f} ms")
        print_info(f"Speedup: {pytorch_time / migraphx_time:.2f}x")
        
        # Clean up
        os.remove(onnx_file)
        
    except Exception as e:
        print_error(f"MIGraphX with ONNX test failed: {e}")
        return False
    
    print_success("All MIGraphX tests passed")
    return True

if __name__ == "__main__":
    success = test_migraphx()
    sys.exit(0 if success else 1)
