#!/usr/bin/env python3
# =============================================================================
# MIGraphX Simple Example
# =============================================================================
# This script demonstrates a simple example of using MIGraphX with AMD GPUs.
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
    print_header("MIGraphX Simple Example")
    
    # Check if MIGraphX is installed
    try:
        import migraphx
        print_success("MIGraphX is installed")
    except ImportError:
        print_error("MIGraphX is not installed")
        print_info("Please install MIGraphX first")
        return False
    
    # Create a MIGraphX program
    print_info("Creating MIGraphX program...")
    program = migraphx.program()
    
    # Create input shape
    input_shape = migraphx.shape(migraphx.shape.float_type, [1, 3, 224, 224])
    
    # Add input parameter
    input_param = program.add_parameter("input", input_shape)
    print_success("Added input parameter")
    
    # Add convolution operation
    conv_op = migraphx.op("convolution", {"padding": [1, 1], "stride": [1, 1], "dilation": [1, 1], "group": 1})
    
    # Create weights for convolution
    weights_shape = migraphx.shape(migraphx.shape.float_type, [16, 3, 3, 3])
    weights = np.random.randn(16, 3, 3, 3).astype(np.float32)
    weights_param = program.add_literal(weights)
    
    # Add convolution
    conv_output = program.add_instruction(conv_op, [input_param, weights_param])
    print_success("Added convolution operation")
    
    # Add ReLU activation
    relu_op = migraphx.op("relu")
    relu_output = program.add_instruction(relu_op, [conv_output])
    print_success("Added ReLU activation")
    
    # Add pooling operation
    pooling_op = migraphx.op("pooling", {"mode": "max", "padding": [0, 0], "stride": [2, 2], "lengths": [2, 2]})
    pooling_output = program.add_instruction(pooling_op, [relu_output])
    print_success("Added pooling operation")
    
    # Print program
    print_info("Program:")
    print(program)
    
    # Compile program for GPU
    print_info("Compiling program for GPU...")
    context = migraphx.get_gpu_context()
    program.compile(context)
    print_success("Compiled program for GPU")
    
    # Create input data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    print_info("Running inference...")
    result = program.run({"input": migraphx.argument(input_data)})
    output = np.array(result[0])
    
    # Print output shape
    print_info(f"Output shape: {output.shape}")
    
    # Benchmark inference
    print_info("Benchmarking inference...")
    
    # Warm-up
    for _ in range(10):
        _ = program.run({"input": migraphx.argument(input_data)})
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        _ = program.run({"input": migraphx.argument(input_data)})
    end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) / num_runs
    print_info(f"Average inference time: {avg_time * 1000:.2f} ms")
    
    print_success("MIGraphX simple example completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
