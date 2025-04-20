#!/usr/bin/env python3
# =============================================================================
# ONNX Utilities
# =============================================================================
# This module provides utilities for working with ONNX models on AMD GPUs.
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
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("onnx_utils")

def check_onnx_availability():
    """Check if ONNX is available."""
    try:
        import onnx
        logger.info(f"ONNX is available (version {onnx.__version__})")
        return True
    except ImportError:
        logger.error("ONNX is not available")
        logger.info("Please install ONNX first")
        return False

def check_onnxruntime_availability():
    """Check if ONNX Runtime is available."""
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime is available (version {onnxruntime.__version__})")
        
        # Check available providers
        providers = onnxruntime.get_available_providers()
        logger.info(f"Available providers: {providers}")
        
        # Check if ROCMExecutionProvider is available
        if 'ROCMExecutionProvider' in providers:
            logger.info("ROCMExecutionProvider is available")
        else:
            logger.warning("ROCMExecutionProvider is not available")
            logger.info("ONNX Runtime will use CPUExecutionProvider")
        
        return True
    except ImportError:
        logger.error("ONNX Runtime is not available")
        logger.info("Please install ONNX Runtime first")
        return False

def export_pytorch_model_to_onnx(model, dummy_input, onnx_path, input_names=None, output_names=None, dynamic_axes=None, opset_version=12):
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        dummy_input: Dummy input for tracing
        onnx_path: Path to save ONNX model
        input_names: Names of input tensors
        output_names: Names of output tensors
        dynamic_axes: Dynamic axes for inputs/outputs
        opset_version: ONNX opset version
    
    Returns:
        bool: True if export is successful, False otherwise
    """
    try:
        # Set default input and output names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        
        # Set model to evaluation mode
        model.eval()
        
        # Export model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Model exported to ONNX format: {onnx_path}")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info("ONNX model verified")
        
        return True
    except Exception as e:
        logger.error(f"Failed to export model to ONNX format: {e}")
        return False

def optimize_onnx_model(input_path, output_path=None):
    """Optimize ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized ONNX model
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_optimized.onnx")
        
        # Load ONNX model
        model = onnx.load(input_path)
        
        # Optimize model
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type="bert",
            num_heads=12,
            hidden_size=768
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        logger.info(f"Model optimized and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to optimize ONNX model: {e}")
        return False

def create_onnx_session(onnx_path, use_rocm=True):
    """Create ONNX Runtime session.
    
    Args:
        onnx_path: Path to ONNX model
        use_rocm: Whether to use ROCMExecutionProvider
    
    Returns:
        onnxruntime.InferenceSession: ONNX Runtime session
    """
    try:
        import onnxruntime
        
        # Get available providers
        providers = onnxruntime.get_available_providers()
        
        # Set session options
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        if use_rocm and 'ROCMExecutionProvider' in providers:
            session = onnxruntime.InferenceSession(
                onnx_path,
                session_options,
                providers=['ROCMExecutionProvider', 'CPUExecutionProvider']
            )
            logger.info("Created ONNX Runtime session with ROCMExecutionProvider")
        else:
            session = onnxruntime.InferenceSession(
                onnx_path,
                session_options,
                providers=['CPUExecutionProvider']
            )
            logger.info("Created ONNX Runtime session with CPUExecutionProvider")
        
        return session
    except Exception as e:
        logger.error(f"Failed to create ONNX Runtime session: {e}")
        return None

def run_onnx_inference(session, inputs):
    """Run inference with ONNX Runtime.
    
    Args:
        session: ONNX Runtime session
        inputs: Dictionary of input tensors
    
    Returns:
        list: List of output tensors
    """
    try:
        # Get input names
        input_names = [input.name for input in session.get_inputs()]
        
        # Get output names
        output_names = [output.name for output in session.get_outputs()]
        
        # Convert inputs to NumPy arrays
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                inputs[name] = tensor.detach().cpu().numpy()
        
        # Run inference
        outputs = session.run(output_names, inputs)
        
        return outputs
    except Exception as e:
        logger.error(f"Failed to run ONNX inference: {e}")
        return None

def benchmark_onnx_model(session, inputs, num_iterations=100):
    """Benchmark ONNX model.
    
    Args:
        session: ONNX Runtime session
        inputs: Dictionary of input tensors
        num_iterations: Number of iterations
    
    Returns:
        dict: Dictionary of benchmark results
    """
    try:
        import time
        
        # Get input names
        input_names = [input.name for input in session.get_inputs()]
        
        # Get output names
        output_names = [output.name for output in session.get_outputs()]
        
        # Convert inputs to NumPy arrays
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                inputs[name] = tensor.detach().cpu().numpy()
        
        # Warm-up
        for _ in range(10):
            _ = session.run(output_names, inputs)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = session.run(output_names, inputs)
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
    except Exception as e:
        logger.error(f"Failed to benchmark ONNX model: {e}")
        return None

def compare_pytorch_and_onnx(model, onnx_session, inputs):
    """Compare PyTorch and ONNX model outputs.
    
    Args:
        model: PyTorch model
        onnx_session: ONNX Runtime session
        inputs: Dictionary of input tensors
    
    Returns:
        dict: Dictionary of comparison results
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Get input names
        input_names = [input.name for input in onnx_session.get_inputs()]
        
        # Get output names
        output_names = [output.name for output in onnx_session.get_outputs()]
        
        # Convert inputs to PyTorch tensors
        pytorch_inputs = {}
        onnx_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, np.ndarray):
                pytorch_inputs[name] = torch.from_numpy(tensor)
            else:
                pytorch_inputs[name] = tensor
            
            if isinstance(tensor, torch.Tensor):
                onnx_inputs[name] = tensor.detach().cpu().numpy()
            else:
                onnx_inputs[name] = tensor
        
        # Run PyTorch inference
        with torch.no_grad():
            pytorch_outputs = model(**pytorch_inputs)
        
        # Run ONNX inference
        onnx_outputs = onnx_session.run(output_names, onnx_inputs)
        
        # Compare outputs
        if isinstance(pytorch_outputs, torch.Tensor):
            pytorch_outputs = [pytorch_outputs]
        
        if isinstance(pytorch_outputs, dict):
            pytorch_outputs = list(pytorch_outputs.values())
        
        # Calculate differences
        diffs = []
        for i, (pytorch_output, onnx_output) in enumerate(zip(pytorch_outputs, onnx_outputs)):
            if isinstance(pytorch_output, torch.Tensor):
                pytorch_output = pytorch_output.detach().cpu().numpy()
            
            diff = np.abs(pytorch_output - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            logger.info(f"Output {i}:")
            logger.info(f"  Max difference: {max_diff}")
            logger.info(f"  Mean difference: {mean_diff}")
            
            diffs.append({
                "max_diff": max_diff,
                "mean_diff": mean_diff
            })
        
        return {
            "diffs": diffs,
            "pytorch_outputs": pytorch_outputs,
            "onnx_outputs": onnx_outputs
        }
    except Exception as e:
        logger.error(f"Failed to compare PyTorch and ONNX model outputs: {e}")
        return None

def quantize_onnx_model(input_path, output_path=None, quantization_mode="dynamic"):
    """Quantize ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized ONNX model
        quantization_mode: Quantization mode (dynamic or static)
    
    Returns:
        bool: True if quantization is successful, False otherwise
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", f"_quantized_{quantization_mode}.onnx")
        
        # Quantize model
        if quantization_mode == "dynamic":
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QInt8
            )
        elif quantization_mode == "static":
            # Static quantization requires calibration data
            # This is a simplified example
            quantize_static(
                input_path,
                output_path,
                calibration_data_reader=None  # Replace with actual calibration data reader
            )
        else:
            logger.error(f"Unsupported quantization mode: {quantization_mode}")
            return False
        
        logger.info(f"Model quantized and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to quantize ONNX model: {e}")
        return False

if __name__ == "__main__":
    # Check ONNX availability
    check_onnx_availability()
    
    # Check ONNX Runtime availability
    check_onnxruntime_availability()
    
    # Example: Export PyTorch model to ONNX
    if torch.cuda.is_available():
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)
        ).cuda()
        
        # Create dummy input
        dummy_input = torch.randn(1, 10).cuda()
        
        # Export model to ONNX
        export_pytorch_model_to_onnx(
            model,
            dummy_input,
            "simple_model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Create ONNX session
        session = create_onnx_session("simple_model.onnx", use_rocm=True)
        
        # Run inference
        inputs = {"input": dummy_input.cpu().numpy()}
        outputs = run_onnx_inference(session, inputs)
        
        # Benchmark model
        benchmark_results = benchmark_onnx_model(session, inputs)
        
        # Compare PyTorch and ONNX
        compare_results = compare_pytorch_and_onnx(model, session, {"input": dummy_input})
