#!/usr/bin/env python3
# =============================================================================
# ONNX Optimizer
# =============================================================================
# This module provides utilities for optimizing ONNX models for AMD GPUs.
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
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("onnx_optimizer")

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import onnx
        logger.info(f"ONNX is available (version {onnx.__version__})")
    except ImportError:
        logger.error("ONNX is not available")
        logger.info("Please install ONNX first")
        return False
    
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime is available (version {onnxruntime.__version__})")
    except ImportError:
        logger.error("ONNX Runtime is not available")
        logger.info("Please install ONNX Runtime first")
        return False
    
    return True

def optimize_model_for_rocm(input_path, output_path=None, optimization_level=99):
    """Optimize ONNX model for ROCm.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized ONNX model
        optimization_level: Optimization level (0-99)
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import onnx
        import onnxruntime
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_optimized.onnx")
        
        # Load ONNX model
        model = onnx.load(input_path)
        
        # Create session options
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.optimized_model_filepath = output_path
        
        # Get available providers
        providers = onnxruntime.get_available_providers()
        
        # Create session with ROCMExecutionProvider if available
        if 'ROCMExecutionProvider' in providers:
            session = onnxruntime.InferenceSession(
                input_path,
                session_options,
                providers=['ROCMExecutionProvider']
            )
            logger.info("Created ONNX Runtime session with ROCMExecutionProvider")
        else:
            session = onnxruntime.InferenceSession(
                input_path,
                session_options,
                providers=['CPUExecutionProvider']
            )
            logger.info("Created ONNX Runtime session with CPUExecutionProvider")
            logger.warning("ROCMExecutionProvider is not available")
        
        # Run a dummy inference to trigger optimization
        input_names = [input.name for input in session.get_inputs()]
        input_shapes = {input.name: input.shape for input in session.get_inputs()}
        
        # Create dummy inputs
        inputs = {}
        for name, shape in input_shapes.items():
            # Replace dynamic dimensions with fixed values
            fixed_shape = [1 if dim == 0 or dim is None else dim for dim in shape]
            inputs[name] = np.random.randn(*fixed_shape).astype(np.float32)
        
        # Run inference
        _ = session.run(None, inputs)
        
        logger.info(f"Model optimized and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to optimize ONNX model for ROCm: {e}")
        return False

def fuse_nodes(input_path, output_path=None):
    """Fuse nodes in ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save fused ONNX model
    
    Returns:
        bool: True if fusion is successful, False otherwise
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_fused.onnx")
        
        # Create optimizer
        opt = optimizer.optimize_model(
            input_path,
            model_type="bert",  # Change based on your model type
            num_heads=12,  # Change based on your model
            hidden_size=768  # Change based on your model
        )
        
        # Save optimized model
        opt.save_model_to_file(output_path)
        
        logger.info(f"Nodes fused and model saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to fuse nodes in ONNX model: {e}")
        return False

def convert_float16(input_path, output_path=None):
    """Convert ONNX model to float16.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save float16 ONNX model
    
    Returns:
        bool: True if conversion is successful, False otherwise
    """
    try:
        import onnx
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_float16.onnx")
        
        # Load ONNX model
        model = onnx.load(input_path)
        
        # Convert to float16
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=True
        )
        
        # Save float16 model
        onnx.save(model_fp16, output_path)
        
        logger.info(f"Model converted to float16 and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to convert ONNX model to float16: {e}")
        return False

def optimize_for_inference(input_path, output_path=None):
    """Optimize ONNX model for inference.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized ONNX model
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import onnx
        from onnx import optimizer
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_inference.onnx")
        
        # Load ONNX model
        model = onnx.load(input_path)
        
        # Optimize model
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_bn_into_conv',
            'fuse_add_bias_into_conv',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm'
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        logger.info(f"Model optimized for inference and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to optimize ONNX model for inference: {e}")
        return False

def constant_folding(input_path, output_path=None):
    """Apply constant folding to ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized ONNX model
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import onnx
        from onnx import optimizer
        
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_folded.onnx")
        
        # Load ONNX model
        model = onnx.load(input_path)
        
        # Apply constant folding
        passes = ['extract_constant_to_initializer', 'constant_folding']
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        logger.info(f"Constant folding applied and model saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to apply constant folding to ONNX model: {e}")
        return False

def optimize_onnx_model(input_path, output_path=None, optimization_options=None):
    """Optimize ONNX model with multiple optimizations.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized ONNX model
        optimization_options: Dictionary of optimization options
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        # Set default output path
        if output_path is None:
            output_path = input_path.replace(".onnx", "_optimized.onnx")
        
        # Set default optimization options
        if optimization_options is None:
            optimization_options = {
                "constant_folding": True,
                "optimize_for_inference": True,
                "fuse_nodes": True,
                "convert_float16": False,
                "optimize_for_rocm": True
            }
        
        # Create temporary paths
        temp_path = input_path
        temp_paths = []
        
        # Apply optimizations
        if optimization_options.get("constant_folding", True):
            temp_path_folded = input_path.replace(".onnx", "_temp_folded.onnx")
            if constant_folding(temp_path, temp_path_folded):
                temp_path = temp_path_folded
                temp_paths.append(temp_path_folded)
        
        if optimization_options.get("optimize_for_inference", True):
            temp_path_inference = input_path.replace(".onnx", "_temp_inference.onnx")
            if optimize_for_inference(temp_path, temp_path_inference):
                temp_path = temp_path_inference
                temp_paths.append(temp_path_inference)
        
        if optimization_options.get("fuse_nodes", True):
            temp_path_fused = input_path.replace(".onnx", "_temp_fused.onnx")
            if fuse_nodes(temp_path, temp_path_fused):
                temp_path = temp_path_fused
                temp_paths.append(temp_path_fused)
        
        if optimization_options.get("convert_float16", False):
            temp_path_float16 = input_path.replace(".onnx", "_temp_float16.onnx")
            if convert_float16(temp_path, temp_path_float16):
                temp_path = temp_path_float16
                temp_paths.append(temp_path_float16)
        
        if optimization_options.get("optimize_for_rocm", True):
            optimize_model_for_rocm(temp_path, output_path)
        else:
            # Copy the last temporary file to the output path
            import shutil
            shutil.copy(temp_path, output_path)
        
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
        
        logger.info(f"Model optimized and saved to: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to optimize ONNX model: {e}")
        return False

if __name__ == "__main__":
    # Check dependencies
    if check_dependencies():
        # Example usage
        if len(sys.argv) > 1:
            input_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            
            # Optimize model
            optimize_onnx_model(input_path, output_path)
        else:
            logger.error("Please provide input ONNX model path")
            logger.info("Usage: python onnx_optimizer.py input_model.onnx [output_model.onnx]")
