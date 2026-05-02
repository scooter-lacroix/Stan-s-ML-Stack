#!/usr/bin/env python3
# =============================================================================
# ROCm Quantization Utilities
# =============================================================================
# This module provides utilities for quantizing models with ROCm.
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
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rocm_quantization_utils")

def check_rocm_quantization_installation():
    """Check if ROCm Quantization is installed.
    
    Returns:
        bool: True if ROCm Quantization is installed, False otherwise
    """
    try:
        import torch_migraphx
        logger.info(f"ROCm Quantization is installed (torch_migraphx version {torch_migraphx.__version__})")
        return True
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        logger.info("Please install ROCm Quantization first")
        return False

def install_rocm_quantization():
    """Install ROCm Quantization.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        import subprocess
        
        # Install ROCm Quantization
        logger.info("Installing ROCm Quantization")
        subprocess.run(
            ["pip", "install", "torch-migraphx"],
            check=True
        )
        
        logger.info("ROCm Quantization installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install ROCm Quantization: {e}")
        return False

def quantize_model_to_int8(model, example_input, calibration_data=None):
    """Quantize model to INT8.
    
    Args:
        model: PyTorch model
        example_input: Example input
        calibration_data: Calibration data
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import torch_migraphx
        
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        logger.info("Creating quantization configuration")
        
        quant_config = torch_migraphx.QuantizationConfig(
            quantization_type="int8",
            calibration_method="histogram"
        )
        
        # Create quantizer
        logger.info("Creating quantizer")
        
        quantizer = torch_migraphx.Quantizer(quant_config)
        
        # Prepare model for quantization
        logger.info("Preparing model for quantization")
        
        prepared_model = quantizer.prepare(model)
        
        # Calibrate model if calibration data is provided
        if calibration_data is not None:
            logger.info("Calibrating model")
            
            for data in calibration_data:
                with torch.no_grad():
                    _ = prepared_model(data)
        else:
            # Use example input for calibration
            logger.info("Calibrating model with example input")
            
            with torch.no_grad():
                _ = prepared_model(example_input)
        
        # Convert model to quantized model
        logger.info("Converting model to quantized model")
        
        quantized_model = quantizer.convert(prepared_model)
        
        logger.info("Model quantized successfully")
        return quantized_model
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model to INT8: {e}")
        return model

def quantize_model_to_int4(model, example_input, calibration_data=None):
    """Quantize model to INT4.
    
    Args:
        model: PyTorch model
        example_input: Example input
        calibration_data: Calibration data
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import torch_migraphx
        
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        logger.info("Creating quantization configuration")
        
        quant_config = torch_migraphx.QuantizationConfig(
            quantization_type="int4",
            calibration_method="histogram"
        )
        
        # Create quantizer
        logger.info("Creating quantizer")
        
        quantizer = torch_migraphx.Quantizer(quant_config)
        
        # Prepare model for quantization
        logger.info("Preparing model for quantization")
        
        prepared_model = quantizer.prepare(model)
        
        # Calibrate model if calibration data is provided
        if calibration_data is not None:
            logger.info("Calibrating model")
            
            for data in calibration_data:
                with torch.no_grad():
                    _ = prepared_model(data)
        else:
            # Use example input for calibration
            logger.info("Calibrating model with example input")
            
            with torch.no_grad():
                _ = prepared_model(example_input)
        
        # Convert model to quantized model
        logger.info("Converting model to quantized model")
        
        quantized_model = quantizer.convert(prepared_model)
        
        logger.info("Model quantized successfully")
        return quantized_model
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model to INT4: {e}")
        return model

def quantize_model_to_fp16(model):
    """Quantize model to FP16.
    
    Args:
        model: PyTorch model
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Convert model to FP16
        logger.info("Converting model to FP16")
        
        fp16_model = model.half()
        
        logger.info("Model converted to FP16 successfully")
        return fp16_model
    except Exception as e:
        logger.error(f"Failed to convert model to FP16: {e}")
        return model

def quantize_model_to_bf16(model):
    """Quantize model to BF16.
    
    Args:
        model: PyTorch model
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Convert model to BF16
        logger.info("Converting model to BF16")
        
        bf16_model = model.to(torch.bfloat16)
        
        logger.info("Model converted to BF16 successfully")
        return bf16_model
    except Exception as e:
        logger.error(f"Failed to convert model to BF16: {e}")
        return model

def quantize_model_weights_only(model, quantization_type="int8"):
    """Quantize model weights only.
    
    Args:
        model: PyTorch model
        quantization_type: Quantization type (int8, int4, fp16, bf16)
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import torch_migraphx
        
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        logger.info(f"Creating quantization configuration for {quantization_type} weights-only")
        
        quant_config = torch_migraphx.QuantizationConfig(
            quantization_type=quantization_type,
            weights_only=True
        )
        
        # Create quantizer
        logger.info("Creating quantizer")
        
        quantizer = torch_migraphx.Quantizer(quant_config)
        
        # Quantize model
        logger.info("Quantizing model weights")
        
        quantized_model = quantizer.quantize(model)
        
        logger.info("Model weights quantized successfully")
        return quantized_model
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model weights: {e}")
        return model

def quantize_model_with_dynamic_range(model, example_input):
    """Quantize model with dynamic range.
    
    Args:
        model: PyTorch model
        example_input: Example input
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import torch_migraphx
        
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        logger.info("Creating quantization configuration for dynamic range")
        
        quant_config = torch_migraphx.QuantizationConfig(
            quantization_type="int8",
            calibration_method="dynamic_range"
        )
        
        # Create quantizer
        logger.info("Creating quantizer")
        
        quantizer = torch_migraphx.Quantizer(quant_config)
        
        # Prepare model for quantization
        logger.info("Preparing model for quantization")
        
        prepared_model = quantizer.prepare(model)
        
        # Calibrate model with example input
        logger.info("Calibrating model with example input")
        
        with torch.no_grad():
            _ = prepared_model(example_input)
        
        # Convert model to quantized model
        logger.info("Converting model to quantized model")
        
        quantized_model = quantizer.convert(prepared_model)
        
        logger.info("Model quantized successfully")
        return quantized_model
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model with dynamic range: {e}")
        return model

def quantize_model_with_kl_divergence(model, example_input, calibration_data=None):
    """Quantize model with KL divergence.
    
    Args:
        model: PyTorch model
        example_input: Example input
        calibration_data: Calibration data
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        import torch_migraphx
        
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        logger.info("Creating quantization configuration for KL divergence")
        
        quant_config = torch_migraphx.QuantizationConfig(
            quantization_type="int8",
            calibration_method="kl_divergence"
        )
        
        # Create quantizer
        logger.info("Creating quantizer")
        
        quantizer = torch_migraphx.Quantizer(quant_config)
        
        # Prepare model for quantization
        logger.info("Preparing model for quantization")
        
        prepared_model = quantizer.prepare(model)
        
        # Calibrate model if calibration data is provided
        if calibration_data is not None:
            logger.info("Calibrating model")
            
            for data in calibration_data:
                with torch.no_grad():
                    _ = prepared_model(data)
        else:
            # Use example input for calibration
            logger.info("Calibrating model with example input")
            
            with torch.no_grad():
                _ = prepared_model(example_input)
        
        # Convert model to quantized model
        logger.info("Converting model to quantized model")
        
        quantized_model = quantizer.convert(prepared_model)
        
        logger.info("Model quantized successfully")
        return quantized_model
    except ImportError:
        logger.error("ROCm Quantization (torch_migraphx) is not installed")
        return model
    except Exception as e:
        logger.error(f"Failed to quantize model with KL divergence: {e}")
        return model

def save_quantized_model(model, model_path):
    """Save quantized model.
    
    Args:
        model: Quantized model
        model_path: Model path
    
    Returns:
        bool: True if model is saved, False otherwise
    """
    try:
        # Save model
        logger.info(f"Saving quantized model to {model_path}")
        
        torch.save(model.state_dict(), model_path)
        
        logger.info("Quantized model saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save quantized model: {e}")
        return False

def load_quantized_model(model_class, model_path, quantization_type="int8"):
    """Load quantized model.
    
    Args:
        model_class: Model class
        model_path: Model path
        quantization_type: Quantization type (int8, int4, fp16, bf16)
    
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        # Create model instance
        model = model_class()
        
        # Load model state dict
        logger.info(f"Loading quantized model from {model_path}")
        
        model.load_state_dict(torch.load(model_path))
        
        # Convert model to specified quantization type
        if quantization_type == "fp16":
            model = model.half()
        elif quantization_type == "bf16":
            model = model.to(torch.bfloat16)
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("Quantized model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load quantized model: {e}")
        return None

def benchmark_quantized_model(model, input_shape, batch_size=1, num_iterations=100):
    """Benchmark quantized model.
    
    Args:
        model: Quantized model
        input_shape: Input shape
        batch_size: Batch size
        num_iterations: Number of iterations
    
    Returns:
        dict: Benchmark results
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Move model to GPU
        model = model.cuda()
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, *input_shape, device="cuda")
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate results
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        logger.info(f"Benchmark completed successfully")
        logger.info(f"Average inference time: {avg_time * 1000:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} samples/s")
        
        return {
            "total_time": total_time,
            "avg_time": avg_time,
            "throughput": throughput,
            "num_iterations": num_iterations
        }
    except Exception as e:
        logger.error(f"Failed to benchmark quantized model: {e}")
        return None

def compare_model_accuracy(original_model, quantized_model, test_data, test_labels):
    """Compare model accuracy.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_data: Test data
        test_labels: Test labels
    
    Returns:
        dict: Accuracy comparison results
    """
    try:
        # Set models to evaluation mode
        original_model.eval()
        quantized_model.eval()
        
        # Move models to GPU
        original_model = original_model.cuda()
        quantized_model = quantized_model.cuda()
        
        # Move test data and labels to GPU
        test_data = test_data.cuda()
        test_labels = test_labels.cuda()
        
        # Evaluate original model
        with torch.no_grad():
            original_outputs = original_model(test_data)
            original_predictions = torch.argmax(original_outputs, dim=1)
            original_accuracy = torch.sum(original_predictions == test_labels).item() / len(test_labels)
        
        # Evaluate quantized model
        with torch.no_grad():
            quantized_outputs = quantized_model(test_data)
            quantized_predictions = torch.argmax(quantized_outputs, dim=1)
            quantized_accuracy = torch.sum(quantized_predictions == test_labels).item() / len(test_labels)
        
        # Calculate accuracy difference
        accuracy_difference = quantized_accuracy - original_accuracy
        
        logger.info(f"Accuracy comparison completed successfully")
        logger.info(f"Original model accuracy: {original_accuracy:.4f}")
        logger.info(f"Quantized model accuracy: {quantized_accuracy:.4f}")
        logger.info(f"Accuracy difference: {accuracy_difference:.4f}")
        
        return {
            "original_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "accuracy_difference": accuracy_difference
        }
    except Exception as e:
        logger.error(f"Failed to compare model accuracy: {e}")
        return None

def compare_model_size(original_model, quantized_model):
    """Compare model size.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
    
    Returns:
        dict: Size comparison results
    """
    try:
        import tempfile
        
        # Create temporary files
        with tempfile.NamedTemporaryFile() as original_file, tempfile.NamedTemporaryFile() as quantized_file:
            # Save models
            torch.save(original_model.state_dict(), original_file.name)
            torch.save(quantized_model.state_dict(), quantized_file.name)
            
            # Get file sizes
            original_size = os.path.getsize(original_file.name)
            quantized_size = os.path.getsize(quantized_file.name)
            
            # Calculate size reduction
            size_reduction = original_size - quantized_size
            size_reduction_percentage = (size_reduction / original_size) * 100
            
            logger.info(f"Size comparison completed successfully")
            logger.info(f"Original model size: {original_size / (1024 * 1024):.2f} MB")
            logger.info(f"Quantized model size: {quantized_size / (1024 * 1024):.2f} MB")
            logger.info(f"Size reduction: {size_reduction / (1024 * 1024):.2f} MB ({size_reduction_percentage:.2f}%)")
            
            return {
                "original_size": original_size,
                "quantized_size": quantized_size,
                "size_reduction": size_reduction,
                "size_reduction_percentage": size_reduction_percentage
            }
    except Exception as e:
        logger.error(f"Failed to compare model size: {e}")
        return None

if __name__ == "__main__":
    # Check ROCm Quantization installation
    check_rocm_quantization_installation()
    
    # Example usage
    if check_rocm_quantization_installation():
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 56 * 56, 1000)
        )
        
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Quantize model to INT8
        quantized_model = quantize_model_to_int8(model, example_input)
        
        # Benchmark quantized model
        benchmark_results = benchmark_quantized_model(
            quantized_model,
            (3, 224, 224),
            batch_size=1,
            num_iterations=100
        )
        
        # Print benchmark results
        logger.info(f"Benchmark results: {benchmark_results}")
