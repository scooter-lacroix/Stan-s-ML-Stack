#!/usr/bin/env python3
# =============================================================================
# MIGraphX Simple Wrapper
# =============================================================================
# This script provides a simple wrapper for MIGraphX functionality.
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
import numpy as np
import torch

try:
    import migraphx
except ImportError:
    raise ImportError("MIGraphX is not installed. Please install MIGraphX first.")

class MIGraphXSimpleWrapper:
    """Simple wrapper for MIGraphX functionality."""
    
    def __init__(self, device="gpu"):
        """Initialize MIGraphX wrapper.
        
        Args:
            device (str): Device to use. Either "gpu" or "cpu".
        """
        self.device = device
        self.model = None
        
        # Create MIGraphX context
        if device == "gpu":
            self.context = migraphx.get_gpu_context()
        else:
            self.context = migraphx.get_cpu_context()
    
    def load_onnx(self, onnx_file):
        """Load ONNX model.
        
        Args:
            onnx_file (str): Path to ONNX model file.
        """
        # Check if file exists
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
        
        # Parse ONNX file
        self.model = migraphx.parse_onnx(onnx_file)
        
        # Compile model for the target device
        self.model.compile(self.context)
        
        return self
    
    def run(self, inputs):
        """Run inference on the loaded model.
        
        Args:
            inputs (dict): Dictionary of input tensors.
        
        Returns:
            list: List of output tensors.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        # Convert inputs to MIGraphX arguments
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Run inference
        results = self.model.run(migraphx_inputs)
        
        # Convert results to numpy arrays
        outputs = [np.array(result) for result in results]
        
        return outputs
    
    def benchmark(self, inputs, num_iterations=100):
        """Benchmark inference performance.
        
        Args:
            inputs (dict): Dictionary of input tensors.
            num_iterations (int): Number of iterations to run.
        
        Returns:
            float: Average inference time in milliseconds.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        # Convert inputs to MIGraphX arguments
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Warm-up
        for _ in range(10):
            self.model.run(migraphx_inputs)
        
        # Benchmark
        start_time = migraphx.now()
        for _ in range(num_iterations):
            self.model.run(migraphx_inputs)
        end_time = migraphx.now()
        
        # Calculate average time
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        
        return avg_time * 1000  # Convert to milliseconds


# Example usage
if __name__ == "__main__":
    # Create wrapper
    wrapper = MIGraphXSimpleWrapper(device="gpu")
    
    # Load ONNX model
    onnx_file = "model.onnx"
    if os.path.exists(onnx_file):
        wrapper.load_onnx(onnx_file)
        
        # Get input names and shapes
        input_names = wrapper.model.get_parameter_names()
        input_shapes = wrapper.model.get_parameter_shapes()
        
        print("Input names:", input_names)
        print("Input shapes:", input_shapes)
        
        # Create dummy inputs
        inputs = {}
        for name, shape in zip(input_names, input_shapes):
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        
        # Run inference
        outputs = wrapper.run(inputs)
        
        # Print output shapes
        print("Output shapes:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")
        
        # Benchmark
        avg_time = wrapper.benchmark(inputs, num_iterations=100)
        print(f"Average inference time: {avg_time:.2f} ms")
    else:
        print(f"ONNX file not found: {onnx_file}")
        print("Please provide a valid ONNX model file.")
