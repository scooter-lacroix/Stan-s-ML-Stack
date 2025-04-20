#!/usr/bin/env python3
# =============================================================================
# PyTorch GPU Test
# =============================================================================
# This script tests PyTorch functionality on AMD GPUs.
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
import matplotlib.pyplot as plt

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

def test_pytorch_gpu():
    """Test PyTorch functionality on AMD GPUs."""
    print_header("PyTorch GPU Test")
    
    # Check PyTorch version
    print_info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA (ROCm) is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print_success("CUDA is available through ROCm")
    else:
        print_error("CUDA is not available through ROCm")
        print_info("Check if ROCm is installed and environment variables are set correctly:")
        print_info("  - HIP_VISIBLE_DEVICES")
        print_info("  - CUDA_VISIBLE_DEVICES")
        print_info("  - PYTORCH_ROCM_DEVICE")
        return False
    
    # Check number of GPUs
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print_success(f"Number of GPUs: {device_count}")
    else:
        print_error("No GPUs detected")
        return False
    
    # Print GPU information
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        if "AMD" in device_name or "Radeon" in device_name:
            print_success(f"GPU {i}: {device_name}")
        else:
            print_warning(f"GPU {i}: {device_name} (not an AMD GPU)")
    
    # Test basic tensor operations
    print_info("Testing basic tensor operations...")
    
    try:
        # Create tensors on CPU and GPU
        cpu_tensor = torch.ones(10)
        gpu_tensor = torch.ones(10, device="cuda")
        
        # Test addition
        cpu_result = cpu_tensor + 1
        gpu_result = gpu_tensor + 1
        
        # Check results
        if torch.allclose(cpu_result.cpu(), gpu_result.cpu()):
            print_success("Basic tensor addition successful")
        else:
            print_error("Basic tensor addition failed")
            return False
        
        # Test matrix multiplication
        a_cpu = torch.randn(100, 100)
        b_cpu = torch.randn(100, 100)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Compute on CPU
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # Compute on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Check results
        if torch.allclose(c_cpu, c_gpu.cpu(), rtol=1e-3, atol=1e-3):
            print_success("Matrix multiplication successful")
            print_info(f"CPU time: {cpu_time * 1000:.2f} ms")
            print_info(f"GPU time: {gpu_time * 1000:.2f} ms")
            print_info(f"Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            print_error("Matrix multiplication failed")
            return False
        
    except Exception as e:
        print_error(f"Basic tensor operations failed: {e}")
        return False
    
    # Test neural network operations
    print_info("Testing neural network operations...")
    
    try:
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = torch.nn.Linear(100, 50)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(50, 10)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # Create model and move to GPU
        model = SimpleModel().to("cuda")
        
        # Create input data
        x = torch.randn(32, 100, device="cuda")
        
        # Forward pass
        output = model(x)
        
        print_success("Neural network forward pass successful")
        print_info(f"Input shape: {x.shape}")
        print_info(f"Output shape: {output.shape}")
        
        # Test backward pass
        target = torch.randn(32, 10, device="cuda")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Compute loss
        loss = loss_fn(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print_success("Neural network backward pass successful")
        print_info(f"Loss: {loss.item()}")
        
    except Exception as e:
        print_error(f"Neural network operations failed: {e}")
        return False
    
    # Test GPU memory
    print_info("Testing GPU memory...")
    
    try:
        # Get total memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print_info(f"Total GPU memory: {total_memory / 1024**2:.2f} MB")
        
        # Allocate memory
        tensors = []
        allocated_memory = 0
        
        # Try to allocate up to 80% of total memory
        target_memory = int(total_memory * 0.8)
        
        while allocated_memory < target_memory:
            # Allocate 100 MB tensor
            tensor_size = 100 * 1024 * 1024 // 4  # 100 MB in float32
            try:
                tensor = torch.ones(tensor_size, dtype=torch.float32, device="cuda")
                tensors.append(tensor)
                allocated_memory += tensor_size * 4
                print_info(f"Allocated {allocated_memory / 1024**2:.2f} MB")
            except RuntimeError as e:
                print_warning(f"Could not allocate more memory: {e}")
                break
        
        # Free memory
        tensors = []
        torch.cuda.empty_cache()
        
        print_success("GPU memory test successful")
        
    except Exception as e:
        print_error(f"GPU memory test failed: {e}")
        return False
    
    print_success("All PyTorch GPU tests passed")
    return True

if __name__ == "__main__":
    success = test_pytorch_gpu()
    sys.exit(0 if success else 1)
