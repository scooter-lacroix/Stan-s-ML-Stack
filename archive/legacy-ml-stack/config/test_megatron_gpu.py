#!/usr/bin/env python3
"""
Test script to verify Megatron-LM is working properly with ROCm support.
"""

import os
import sys
import torch
import time

# Set environment variables for ROCm
os.environ["AMD_LOG_LEVEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2"
os.environ["ROCR_VISIBLE_DEVICES"] = "0,1,2"

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def test_pytorch_gpu():
    """Test PyTorch GPU support."""
    print_separator("Testing PyTorch GPU Support")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Test tensor operations on GPU
        print("\nTesting tensor operations on GPU...")

        # Create a tensor on GPU
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()

        # Perform matrix multiplication
        start_time = time.time()
        z = torch.matmul(x, y)
        end_time = time.time()

        print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
        print(f"Result tensor shape: {z.shape}")
        print(f"Result tensor device: {z.device}")

        print_success("PyTorch GPU test completed successfully")
    else:
        print_error("CUDA is not available")

def test_megatron_import():
    """Test Megatron-LM import."""
    print_separator("Testing Megatron-LM Import")

    try:
        import megatron
        print_success("Megatron-LM imported successfully")

        # Try to import core modules
        try:
            from megatron.core import tensor_parallel
            print_success("Megatron-LM tensor_parallel module imported successfully")
        except ImportError as e:
            print_error(f"Failed to import tensor_parallel module: {e}")

        try:
            from megatron.core import distributed
            print_success("Megatron-LM distributed module imported successfully")
        except ImportError as e:
            print_error(f"Failed to import distributed module: {e}")

        try:
            from megatron.core import models
            print_success("Megatron-LM models module imported successfully")
        except ImportError as e:
            print_error(f"Failed to import models module: {e}")

    except ImportError as e:
        print_error(f"Failed to import Megatron-LM: {e}")
        return False

    return True

def test_megatron_basic_functionality():
    """Test basic Megatron-LM functionality."""
    print_separator("Testing Megatron-LM Basic Functionality")

    try:
        # Import necessary modules
        from megatron.core import tensor_parallel
        from megatron.core.transformer.transformer_config import TransformerConfig

        # Get the parameters accepted by TransformerConfig
        import inspect
        params = inspect.signature(TransformerConfig.__init__).parameters
        print(f"TransformerConfig accepts these parameters: {list(params.keys())}")

        # Create a basic transformer config with correct parameters
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=512,
            kv_channels=32,
            num_query_groups=4
        )

        print(f"Created transformer config: {config}")
        print_success("Megatron-LM basic functionality test passed")
        return True
    except Exception as e:
        print_error(f"Megatron-LM basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print_separator("Megatron-LM ROCm Compatibility Test")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Test PyTorch GPU support
    test_pytorch_gpu()

    # Test Megatron-LM import
    if test_megatron_import():
        # Test Megatron-LM basic functionality
        test_megatron_basic_functionality()

    print_separator("Test Complete")

if __name__ == "__main__":
    main()
