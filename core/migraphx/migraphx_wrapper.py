#!/usr/bin/env python3
# =============================================================================
# MIGraphX Wrapper
# =============================================================================
# This script provides a wrapper for MIGraphX functionality.
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

class MIGraphXWrapper:
    """Wrapper for MIGraphX functionality."""
    
    def __init__(self, device="gpu"):
        """Initialize MIGraphX wrapper.
        
        Args:
            device (str): Device to use. Either "gpu" or "cpu".
        """
        self.device = device
        self.program = None
        self.params = {}
        
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
        self.program = migraphx.parse_onnx(onnx_file)
        
        # Compile program for the target device
        self.program.compile(self.context)
        
        # Get parameter shapes
        self.params = {}
        for i, param_name in enumerate(self.program.get_parameter_names()):
            self.params[param_name] = self.program.get_parameter_shapes()[i]
        
        return self
    
    def run(self, inputs):
        """Run inference on the loaded model.
        
        Args:
            inputs (dict): Dictionary of input tensors.
        
        Returns:
            dict: Dictionary of output tensors.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        # Convert inputs to MIGraphX tensors
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Run inference
        results = self.program.run(migraphx_inputs)
        
        # Convert results to numpy arrays
        outputs = {}
        for i, name in enumerate(self.program.get_output_names()):
            outputs[name] = np.array(results[i])
        
        return outputs
    
    def benchmark(self, inputs, num_iterations=100):
        """Benchmark inference performance.
        
        Args:
            inputs (dict): Dictionary of input tensors.
            num_iterations (int): Number of iterations to run.
        
        Returns:
            float: Average inference time in milliseconds.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        # Convert inputs to MIGraphX tensors
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Warm-up
        for _ in range(10):
            self.program.run(migraphx_inputs)
        
        # Benchmark
        start_time = migraphx.now()
        for _ in range(num_iterations):
            self.program.run(migraphx_inputs)
        end_time = migraphx.now()
        
        # Calculate average time
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        
        return avg_time * 1000  # Convert to milliseconds
    
    def get_parameter_shapes(self):
        """Get parameter shapes of the loaded model.
        
        Returns:
            dict: Dictionary of parameter shapes.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        return self.params
    
    def get_output_shapes(self):
        """Get output shapes of the loaded model.
        
        Returns:
            dict: Dictionary of output shapes.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        output_shapes = {}
        for i, name in enumerate(self.program.get_output_names()):
            output_shapes[name] = self.program.get_output_shapes()[i]
        
        return output_shapes
    
    def print_model_info(self):
        """Print model information."""
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() first.")
        
        print("=== Model Information ===")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {len(self.params)}")
        print(f"Number of outputs: {len(self.program.get_output_names())}")
        
        print("\nParameter shapes:")
        for name, shape in self.params.items():
            print(f"  {name}: {shape}")
        
        print("\nOutput shapes:")
        for name, shape in self.get_output_shapes().items():
            print(f"  {name}: {shape}")
    
    def __str__(self):
        """String representation of the wrapper."""
        if self.program is None:
            return "MIGraphXWrapper(No model loaded)"
        
        return f"MIGraphXWrapper(device={self.device}, params={len(self.params)}, outputs={len(self.program.get_output_names())})"


# Example usage
if __name__ == "__main__":
    # Create wrapper
    wrapper = MIGraphXWrapper(device="gpu")
    
    # Load ONNX model
    onnx_file = "model.onnx"
    if os.path.exists(onnx_file):
        wrapper.load_onnx(onnx_file)
        
        # Print model information
        wrapper.print_model_info()
        
        # Create dummy input
        input_name = list(wrapper.get_parameter_shapes().keys())[0]
        input_shape = wrapper.get_parameter_shapes()[input_name]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = wrapper.run({input_name: dummy_input})
        
        # Print output shapes
        print("\nOutput values:")
        for name, tensor in outputs.items():
            print(f"  {name}: shape={tensor.shape}, min={tensor.min()}, max={tensor.max()}")
        
        # Benchmark
        avg_time = wrapper.benchmark({input_name: dummy_input}, num_iterations=100)
        print(f"\nAverage inference time: {avg_time:.2f} ms")
    else:
        print(f"ONNX file not found: {onnx_file}")
        print("Please provide a valid ONNX model file.")
