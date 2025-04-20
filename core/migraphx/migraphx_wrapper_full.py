#!/usr/bin/env python3
# =============================================================================
# MIGraphX Full Wrapper
# =============================================================================
# This script provides a comprehensive wrapper for MIGraphX functionality.
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
import logging
import time
from typing import Dict, List, Tuple, Union, Optional

try:
    import migraphx
except ImportError:
    raise ImportError("MIGraphX is not installed. Please install MIGraphX first.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migraphx_wrapper_full")

class MIGraphXFullWrapper:
    """Comprehensive wrapper for MIGraphX functionality."""
    
    def __init__(self, device="gpu", log_level=logging.INFO):
        """Initialize MIGraphX wrapper.
        
        Args:
            device (str): Device to use. Either "gpu" or "cpu".
            log_level (int): Logging level.
        """
        self.device = device
        self.program = None
        self.params = {}
        self.input_names = []
        self.output_names = []
        self.model_path = None
        
        # Set logging level
        logger.setLevel(log_level)
        
        # Create MIGraphX context
        if device == "gpu":
            try:
                self.context = migraphx.get_gpu_context()
                logger.info("Using GPU context")
            except Exception as e:
                logger.warning(f"Failed to get GPU context: {e}")
                logger.info("Falling back to CPU context")
                self.device = "cpu"
                self.context = migraphx.get_cpu_context()
        else:
            self.context = migraphx.get_cpu_context()
            logger.info("Using CPU context")
    
    def load_onnx(self, onnx_file, input_names=None, output_names=None):
        """Load ONNX model.
        
        Args:
            onnx_file (str): Path to ONNX model file.
            input_names (List[str], optional): Names of input tensors.
            output_names (List[str], optional): Names of output tensors.
        """
        # Check if file exists
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
        
        logger.info(f"Loading ONNX model from {onnx_file}")
        
        # Parse ONNX file
        self.program = migraphx.parse_onnx(onnx_file)
        self.model_path = onnx_file
        
        # Set input and output names
        if input_names is not None:
            self.input_names = input_names
        else:
            self.input_names = self.program.get_parameter_names()
        
        if output_names is not None:
            self.output_names = output_names
        else:
            self.output_names = self.program.get_output_names()
        
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")
        
        # Compile program for the target device
        logger.info(f"Compiling program for {self.device}")
        self.program.compile(self.context)
        
        # Get parameter shapes
        self.params = {}
        for i, param_name in enumerate(self.program.get_parameter_names()):
            self.params[param_name] = self.program.get_parameter_shapes()[i]
        
        logger.info(f"Model loaded successfully with {len(self.params)} parameters")
        
        return self
    
    def load_tf(self, tf_file, input_names=None, output_names=None):
        """Load TensorFlow model.
        
        Args:
            tf_file (str): Path to TensorFlow model file.
            input_names (List[str], optional): Names of input tensors.
            output_names (List[str], optional): Names of output tensors.
        """
        # Check if file exists
        if not os.path.exists(tf_file):
            raise FileNotFoundError(f"TensorFlow file not found: {tf_file}")
        
        logger.info(f"Loading TensorFlow model from {tf_file}")
        
        # Parse TensorFlow file
        self.program = migraphx.parse_tf(tf_file)
        self.model_path = tf_file
        
        # Set input and output names
        if input_names is not None:
            self.input_names = input_names
        else:
            self.input_names = self.program.get_parameter_names()
        
        if output_names is not None:
            self.output_names = output_names
        else:
            self.output_names = self.program.get_output_names()
        
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")
        
        # Compile program for the target device
        logger.info(f"Compiling program for {self.device}")
        self.program.compile(self.context)
        
        # Get parameter shapes
        self.params = {}
        for i, param_name in enumerate(self.program.get_parameter_names()):
            self.params[param_name] = self.program.get_parameter_shapes()[i]
        
        logger.info(f"Model loaded successfully with {len(self.params)} parameters")
        
        return self
    
    def run(self, inputs):
        """Run inference on the loaded model.
        
        Args:
            inputs (dict): Dictionary of input tensors.
        
        Returns:
            dict: Dictionary of output tensors.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        # Convert inputs to MIGraphX tensors
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Run inference
        logger.debug("Running inference")
        results = self.program.run(migraphx_inputs)
        
        # Convert results to numpy arrays
        outputs = {}
        for i, name in enumerate(self.output_names):
            outputs[name] = np.array(results[i])
        
        return outputs
    
    def benchmark(self, inputs, num_iterations=100, warmup_iterations=10):
        """Benchmark inference performance.
        
        Args:
            inputs (dict): Dictionary of input tensors.
            num_iterations (int): Number of iterations to run.
            warmup_iterations (int): Number of warmup iterations.
        
        Returns:
            dict: Dictionary of benchmark results.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        # Convert inputs to MIGraphX tensors
        migraphx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input tensor must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_inputs[name] = migraphx.argument(tensor)
        
        # Warm-up
        logger.info(f"Warming up for {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            self.program.run(migraphx_inputs)
        
        # Benchmark
        logger.info(f"Benchmarking for {num_iterations} iterations")
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.program.run(migraphx_inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = sum(times) / num_iterations
        min_time = min(times)
        max_time = max(times)
        std_time = np.std(times)
        
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        logger.info(f"Min inference time: {min_time:.2f} ms")
        logger.info(f"Max inference time: {max_time:.2f} ms")
        logger.info(f"Std inference time: {std_time:.2f} ms")
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "times": times
        }
    
    def get_parameter_shapes(self):
        """Get parameter shapes of the loaded model.
        
        Returns:
            dict: Dictionary of parameter shapes.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        return self.params
    
    def get_output_shapes(self):
        """Get output shapes of the loaded model.
        
        Returns:
            dict: Dictionary of output shapes.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        output_shapes = {}
        for i, name in enumerate(self.output_names):
            output_shapes[name] = self.program.get_output_shapes()[i]
        
        return output_shapes
    
    def print_model_info(self):
        """Print model information."""
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        print("=== Model Information ===")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {len(self.params)}")
        print(f"Number of outputs: {len(self.output_names)}")
        
        print("\nParameter shapes:")
        for name, shape in self.params.items():
            print(f"  {name}: {shape}")
        
        print("\nOutput shapes:")
        for name, shape in self.get_output_shapes().items():
            print(f"  {name}: {shape}")
    
    def save(self, file_path):
        """Save the compiled model.
        
        Args:
            file_path (str): Path to save the model.
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        logger.info(f"Saving model to {file_path}")
        self.program.save(file_path)
        
        return self
    
    def load(self, file_path):
        """Load a compiled model.
        
        Args:
            file_path (str): Path to the compiled model.
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        logger.info(f"Loading model from {file_path}")
        self.program = migraphx.load(file_path)
        self.model_path = file_path
        
        # Get parameter shapes
        self.params = {}
        for i, param_name in enumerate(self.program.get_parameter_names()):
            self.params[param_name] = self.program.get_parameter_shapes()[i]
        
        # Set input and output names
        self.input_names = self.program.get_parameter_names()
        self.output_names = self.program.get_output_names()
        
        logger.info(f"Model loaded successfully with {len(self.params)} parameters")
        
        return self
    
    def quantize(self, calibration_data, quantization_mode="fp16"):
        """Quantize the model.
        
        Args:
            calibration_data (dict): Dictionary of calibration data.
            quantization_mode (str): Quantization mode. Either "fp16" or "int8".
        """
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        logger.info(f"Quantizing model with mode: {quantization_mode}")
        
        # Convert calibration data to MIGraphX tensors
        migraphx_calibration_data = {}
        for name, tensor in calibration_data.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Calibration data must be a numpy array or torch tensor, got {type(tensor)}")
            
            migraphx_calibration_data[name] = migraphx.argument(tensor)
        
        # Quantize the model
        if quantization_mode == "fp16":
            self.program = migraphx.quantize_fp16(self.program)
        elif quantization_mode == "int8":
            self.program = migraphx.quantize_int8(self.program, migraphx_calibration_data)
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization_mode}")
        
        # Recompile the program
        logger.info(f"Recompiling quantized program for {self.device}")
        self.program.compile(self.context)
        
        logger.info("Model quantized successfully")
        
        return self
    
    def optimize(self):
        """Optimize the model."""
        if self.program is None:
            raise ValueError("No model loaded. Call load_onnx() or load_tf() first.")
        
        logger.info("Optimizing model")
        
        # Optimize the model
        self.program = migraphx.optimize(self.program)
        
        # Recompile the program
        logger.info(f"Recompiling optimized program for {self.device}")
        self.program.compile(self.context)
        
        logger.info("Model optimized successfully")
        
        return self
    
    def __str__(self):
        """String representation of the wrapper."""
        if self.program is None:
            return "MIGraphXFullWrapper(No model loaded)"
        
        return f"MIGraphXFullWrapper(device={self.device}, params={len(self.params)}, outputs={len(self.output_names)})"


# Example usage
if __name__ == "__main__":
    # Create wrapper
    wrapper = MIGraphXFullWrapper(device="gpu")
    
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
        benchmark_results = wrapper.benchmark({input_name: dummy_input}, num_iterations=100)
        print(f"\nAverage inference time: {benchmark_results['avg_time']:.2f} ms")
        
        # Save model
        wrapper.save("model.migraphx")
    else:
        print(f"ONNX file not found: {onnx_file}")
        print("Please provide a valid ONNX model file.")
