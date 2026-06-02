#!/usr/bin/env python3
# =============================================================================
# ML Stack Test
# =============================================================================
# This script tests the entire ML Stack on AMD GPUs.
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

def test_ml_stack():
    """Test the entire ML Stack on AMD GPUs."""
    print_header("ML Stack Test")
    
    # Check if PyTorch is installed
    try:
        import torch
        print_success("PyTorch is installed")
        print_info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print_error("PyTorch is not installed")
        print_info("Please install PyTorch first")
        return False
    
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
    
    # Test PyTorch
    print_info("Testing PyTorch...")
    
    try:
        # Create tensors on CPU and GPU
        cpu_tensor = torch.ones(10)
        gpu_tensor = torch.ones(10, device="cuda")
        
        # Test addition
        cpu_result = cpu_tensor + 1
        gpu_result = gpu_tensor + 1
        
        # Check results
        if torch.allclose(cpu_result.cpu(), gpu_result.cpu()):
            print_success("PyTorch basic operations successful")
        else:
            print_error("PyTorch basic operations failed")
            return False
        
    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        return False
    
    # Check if ONNX Runtime is installed
    try:
        import onnxruntime
        print_success("ONNX Runtime is installed")
        print_info(f"ONNX Runtime version: {onnxruntime.__version__}")
        
        # Check available providers
        providers = onnxruntime.get_available_providers()
        print_info(f"Available providers: {providers}")
        
        # Check if ROCMExecutionProvider is available
        if 'ROCMExecutionProvider' in providers:
            print_success("ROCMExecutionProvider is available")
        else:
            print_warning("ROCMExecutionProvider is not available")
            print_info("ONNX Runtime will use CPUExecutionProvider")
        
    except ImportError:
        print_warning("ONNX Runtime is not installed")
        print_info("Some tests will be skipped")
    
    # Check if MIGraphX is installed
    try:
        import migraphx
        print_success("MIGraphX is installed")
    except ImportError:
        print_warning("MIGraphX is not installed")
        print_info("Some tests will be skipped")
    
    # Check if Flash Attention is installed
    try:
        import flash_attention_amd
        print_success("Flash Attention is installed")
    except ImportError:
        print_warning("Flash Attention is not installed")
        print_info("Some tests will be skipped")
    
    # Check if MPI is installed
    try:
        from mpi4py import MPI
        print_success("MPI is installed")
        print_info(f"MPI version: {MPI.Get_version()}")
    except ImportError:
        print_warning("MPI is not installed")
        print_info("Some tests will be skipped")
    
    # Check if Megatron-LM is installed
    try:
        import megatron
        print_success("Megatron-LM is installed")
    except ImportError:
        print_warning("Megatron-LM is not installed")
        print_info("Some tests will be skipped")
    
    # Test ONNX Runtime
    if 'onnxruntime' in sys.modules:
        print_info("Testing ONNX Runtime...")
        
        try:
            # Create a simple model
            import onnx
            from onnx import helper
            from onnx import TensorProto
            
            # Create a simple model
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
            
            node_def = helper.make_node(
                'Relu',
                inputs=['X'],
                outputs=['Y']
            )
            
            graph_def = helper.make_graph(
                [node_def],
                'test-model',
                [X],
                [Y]
            )
            
            model_def = helper.make_model(graph_def, producer_name='onnx-example')
            
            # Save model
            onnx_file = "test_model.onnx"
            onnx.save(model_def, onnx_file)
            
            # Create input data
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Create session
            session = onnxruntime.InferenceSession(onnx_file)
            
            # Run inference
            result = session.run(None, {'X': input_data})
            
            # Check result
            expected = np.maximum(input_data, 0)
            if np.allclose(result[0], expected):
                print_success("ONNX Runtime inference successful")
            else:
                print_error("ONNX Runtime inference failed")
                return False
            
            # Clean up
            os.remove(onnx_file)
            
        except Exception as e:
            print_error(f"ONNX Runtime test failed: {e}")
            return False
    
    # Test MIGraphX
    if 'migraphx' in sys.modules:
        print_info("Testing MIGraphX...")
        
        try:
            # Create a MIGraphX program
            program = migraphx.program()
            
            # Create input shape
            input_shape = migraphx.shape(migraphx.shape.float_type, [1, 3, 224, 224])
            
            # Add input parameter
            input_param = program.add_parameter("input", input_shape)
            
            # Add relu operation
            relu_op = migraphx.op("relu")
            relu_output = program.add_instruction(relu_op, [input_param])
            
            # Compile program for GPU
            context = migraphx.get_gpu_context()
            program.compile(context)
            
            # Create input data
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Run inference
            result = program.run({"input": migraphx.argument(input_data)})
            
            # Check result
            expected = np.maximum(input_data, 0)
            if np.allclose(np.array(result[0]), expected):
                print_success("MIGraphX inference successful")
            else:
                print_error("MIGraphX inference failed")
                return False
            
        except Exception as e:
            print_error(f"MIGraphX test failed: {e}")
            return False
    
    # Test Flash Attention
    if 'flash_attention_amd' in sys.modules:
        print_info("Testing Flash Attention...")
        
        try:
            # Create input tensors
            batch_size = 2
            seq_len = 128
            num_heads = 8
            head_dim = 64
            
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
            
            # Run Flash Attention
            output = flash_attention_amd.flash_attn_func(q, k, v)
            
            print_success("Flash Attention inference successful")
            
        except Exception as e:
            print_error(f"Flash Attention test failed: {e}")
            return False
    
    # Test MPI
    if 'mpi4py' in sys.modules:
        print_info("Testing MPI...")
        
        try:
            # Initialize MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            print_info(f"Process rank: {rank}")
            print_info(f"World size: {size}")
            
            # Test allreduce
            send_data = np.array([rank + 1] * 3, dtype=np.float32)
            recv_data = np.empty(3, dtype=np.float32)
            
            comm.Allreduce(send_data, recv_data, op=MPI.SUM)
            
            expected = np.array([sum(range(1, size + 1))] * 3, dtype=np.float32)
            if np.array_equal(recv_data, expected):
                print_success("MPI allreduce successful")
            else:
                print_error("MPI allreduce failed")
                return False
            
        except Exception as e:
            print_error(f"MPI test failed: {e}")
            return False
    
    # Test Megatron-LM
    if 'megatron' in sys.modules:
        print_info("Testing Megatron-LM...")
        
        try:
            # Import Megatron modules
            from megatron import initialize_megatron
            from megatron.arguments import parse_args
            
            # Initialize Megatron
            args = parse_args([])
            initialize_megatron(args)
            
            print_success("Megatron-LM initialization successful")
            
        except Exception as e:
            print_error(f"Megatron-LM test failed: {e}")
            return False
    
    print_success("All ML Stack tests passed")
    return True

if __name__ == "__main__":
    success = test_ml_stack()
    sys.exit(0 if success else 1)
