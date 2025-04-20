#!/usr/bin/env python3
# =============================================================================
# ONNX Runtime Test
# =============================================================================
# This script tests if ONNX Runtime is working correctly with AMD GPUs.
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

def test_onnx_runtime():
    """Test if ONNX Runtime is working correctly with AMD GPUs."""
    print_header("ONNX Runtime Test")
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print_error("CUDA (ROCm) is not available")
        return False
    
    # Try to import onnxruntime
    try:
        import onnxruntime as ort
        print_success("ONNX Runtime imported successfully")
    except ImportError:
        print_error("Failed to import ONNX Runtime")
        print_info("Make sure ONNX Runtime is installed:")
        print_info("  - Check if onnxruntime is in the Python path")
        print_info("  - Try reinstalling ONNX Runtime")
        return False
    
    # Print ONNX Runtime version
    print_info(f"ONNX Runtime version: {ort.__version__}")
    
    # Check available providers
    providers = ort.get_available_providers()
    print_info(f"Available providers: {providers}")
    
    # Check if ROCMExecutionProvider is available
    if 'ROCMExecutionProvider' in providers:
        print_success("ROCMExecutionProvider is available")
    else:
        print_warning("ROCMExecutionProvider is not available")
        print_info("ONNX Runtime will fall back to CPUExecutionProvider")
        print_info("Make sure ONNX Runtime is built with ROCm support")
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(5, 2)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Create input data
    x = torch.randn(1, 10)
    
    # Export model to ONNX
    try:
        onnx_file = "simple_model.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_file,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print_success("Model exported to ONNX successfully")
    except Exception as e:
        print_error(f"Failed to export model to ONNX: {e}")
        return False
    
    # Run inference with PyTorch
    with torch.no_grad():
        torch_output = model(x).numpy()
    
    # Create ONNX Runtime session
    try:
        if 'ROCMExecutionProvider' in providers:
            session = ort.InferenceSession(onnx_file, providers=['ROCMExecutionProvider'])
            print_success("Created ONNX Runtime session with ROCMExecutionProvider")
        else:
            session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
            print_warning("Created ONNX Runtime session with CPUExecutionProvider")
    except Exception as e:
        print_error(f"Failed to create ONNX Runtime session: {e}")
        return False
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference with ONNX Runtime
    try:
        ort_inputs = {input_name: x.numpy()}
        ort_output = session.run([output_name], ort_inputs)[0]
        print_success("ONNX Runtime inference successful")
    except Exception as e:
        print_error(f"ONNX Runtime inference failed: {e}")
        return False
    
    # Compare PyTorch and ONNX Runtime outputs
    try:
        np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
        print_success("PyTorch and ONNX Runtime outputs match")
    except Exception as e:
        print_error(f"PyTorch and ONNX Runtime outputs do not match: {e}")
        return False
    
    # Test with different batch sizes
    for batch_size in [2, 4, 8]:
        print_info(f"Testing with batch size {batch_size}")
        
        # Create input data
        x_batch = torch.randn(batch_size, 10)
        
        # Run inference with PyTorch
        with torch.no_grad():
            torch_output_batch = model(x_batch).numpy()
        
        # Run inference with ONNX Runtime
        try:
            ort_inputs_batch = {input_name: x_batch.numpy()}
            ort_output_batch = session.run([output_name], ort_inputs_batch)[0]
            print_success(f"ONNX Runtime inference successful for batch size {batch_size}")
        except Exception as e:
            print_error(f"ONNX Runtime inference failed for batch size {batch_size}: {e}")
            return False
        
        # Compare PyTorch and ONNX Runtime outputs
        try:
            np.testing.assert_allclose(torch_output_batch, ort_output_batch, rtol=1e-3, atol=1e-5)
            print_success(f"PyTorch and ONNX Runtime outputs match for batch size {batch_size}")
        except Exception as e:
            print_error(f"PyTorch and ONNX Runtime outputs do not match for batch size {batch_size}: {e}")
            return False
    
    # Clean up
    try:
        os.remove(onnx_file)
        print_info(f"Removed temporary file: {onnx_file}")
    except Exception as e:
        print_warning(f"Failed to remove temporary file: {e}")
    
    print_success("All ONNX Runtime tests passed")
    return True

if __name__ == "__main__":
    success = test_onnx_runtime()
    sys.exit(0 if success else 1)
