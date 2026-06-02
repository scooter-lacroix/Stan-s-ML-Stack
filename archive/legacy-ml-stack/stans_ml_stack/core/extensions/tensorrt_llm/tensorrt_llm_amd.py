#!/usr/bin/env python3
# =============================================================================
# TensorRT-LLM for AMD GPUs
# =============================================================================
# This module provides utilities for using TensorRT-LLM with AMD GPUs.
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
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tensorrt_llm_amd")

def check_tensorrt_llm_installation():
    """Check if TensorRT-LLM is installed.
    
    Returns:
        bool: True if TensorRT-LLM is installed, False otherwise
    """
    try:
        import tensorrt_llm
        logger.info(f"TensorRT-LLM is installed (version {tensorrt_llm.__version__})")
        return True
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        logger.info("Please install TensorRT-LLM first")
        return False

def install_tensorrt_llm_for_amd():
    """Install TensorRT-LLM for AMD GPUs.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        # Clone repository
        logger.info("Cloning TensorRT-LLM repository")
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/TensorRT-LLM.git"],
            check=True
        )
        
        # Change directory
        os.chdir("TensorRT-LLM")
        
        # Set environment variables for AMD
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"  # For RDNA 3 GPUs (RX 7000 series)
        
        # Install dependencies
        logger.info("Installing dependencies")
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            check=True
        )
        
        # Build TensorRT-LLM
        logger.info("Building TensorRT-LLM for AMD GPUs")
        subprocess.run(
            ["python", "scripts/build_wheel.py", "--rocm"],
            check=True
        )
        
        # Install TensorRT-LLM
        logger.info("Installing TensorRT-LLM")
        subprocess.run(
            ["pip", "install", "build/tensorrt_llm*.whl"],
            check=True
        )
        
        logger.info("TensorRT-LLM installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install TensorRT-LLM: {e}")
        return False

def convert_hf_model_to_tensorrt_llm(model_name, output_dir, precision="fp16", max_batch_size=1, max_input_len=1024, max_output_len=1024):
    """Convert Hugging Face model to TensorRT-LLM.
    
    Args:
        model_name: Hugging Face model name
        output_dir: Output directory
        precision: Precision (fp32, fp16, int8, int4)
        max_batch_size: Maximum batch size
        max_input_len: Maximum input length
        max_output_len: Maximum output length
    
    Returns:
        bool: True if conversion is successful, False otherwise
    """
    try:
        import tensorrt_llm
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set precision flag
        if precision == "fp32":
            precision_flag = "--use_fp32"
        elif precision == "fp16":
            precision_flag = "--use_fp16"
        elif precision == "int8":
            precision_flag = "--use_int8"
        elif precision == "int4":
            precision_flag = "--use_int4"
        else:
            logger.error(f"Unsupported precision: {precision}")
            return False
        
        # Convert model
        logger.info(f"Converting {model_name} to TensorRT-LLM")
        
        subprocess.run(
            [
                "python", "examples/hf_to_tensorrt_llm.py",
                "--model_name", model_name,
                "--output_dir", output_dir,
                precision_flag,
                "--max_batch_size", str(max_batch_size),
                "--max_input_len", str(max_input_len),
                "--max_output_len", str(max_output_len)
            ],
            check=True
        )
        
        logger.info(f"Model converted successfully to {output_dir}")
        return True
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        return False

def build_tensorrt_llm_engine(model_dir, engine_dir, precision="fp16", max_batch_size=1, max_input_len=1024, max_output_len=1024):
    """Build TensorRT-LLM engine.
    
    Args:
        model_dir: Model directory
        engine_dir: Engine directory
        precision: Precision (fp32, fp16, int8, int4)
        max_batch_size: Maximum batch size
        max_input_len: Maximum input length
        max_output_len: Maximum output length
    
    Returns:
        bool: True if build is successful, False otherwise
    """
    try:
        import tensorrt_llm
        
        # Create engine directory
        os.makedirs(engine_dir, exist_ok=True)
        
        # Set precision flag
        if precision == "fp32":
            precision_flag = "--use_fp32"
        elif precision == "fp16":
            precision_flag = "--use_fp16"
        elif precision == "int8":
            precision_flag = "--use_int8"
        elif precision == "int4":
            precision_flag = "--use_int4"
        else:
            logger.error(f"Unsupported precision: {precision}")
            return False
        
        # Build engine
        logger.info(f"Building TensorRT-LLM engine")
        
        subprocess.run(
            [
                "python", "examples/build_engine.py",
                "--model_dir", model_dir,
                "--output_dir", engine_dir,
                precision_flag,
                "--max_batch_size", str(max_batch_size),
                "--max_input_len", str(max_input_len),
                "--max_output_len", str(max_output_len)
            ],
            check=True
        )
        
        logger.info(f"Engine built successfully to {engine_dir}")
        return True
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to build engine: {e}")
        return False

def run_tensorrt_llm_inference(engine_dir, input_text, max_output_len=1024, temperature=0.7, top_p=0.9, top_k=50):
    """Run TensorRT-LLM inference.
    
    Args:
        engine_dir: Engine directory
        input_text: Input text
        max_output_len: Maximum output length
        temperature: Temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        str: Generated text
    """
    try:
        import tensorrt_llm
        
        # Run inference
        logger.info(f"Running TensorRT-LLM inference")
        
        result = subprocess.run(
            [
                "python", "examples/run_inference.py",
                "--engine_dir", engine_dir,
                "--input", input_text,
                "--max_output_len", str(max_output_len),
                "--temperature", str(temperature),
                "--top_p", str(top_p),
                "--top_k", str(top_k)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        
        # Parse output
        output = result.stdout.strip()
        
        logger.info(f"Inference completed successfully")
        logger.info(f"Output: {output}")
        
        return output
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        return None

def benchmark_tensorrt_llm(engine_dir, input_text, num_iterations=10, max_output_len=1024):
    """Benchmark TensorRT-LLM.
    
    Args:
        engine_dir: Engine directory
        input_text: Input text
        num_iterations: Number of iterations
        max_output_len: Maximum output length
    
    Returns:
        dict: Benchmark results
    """
    try:
        import tensorrt_llm
        import time
        
        # Run benchmark
        logger.info(f"Benchmarking TensorRT-LLM")
        
        result = subprocess.run(
            [
                "python", "examples/benchmark.py",
                "--engine_dir", engine_dir,
                "--input", input_text,
                "--num_iterations", str(num_iterations),
                "--max_output_len", str(max_output_len)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        
        # Parse output
        output = result.stdout.strip()
        
        # Extract benchmark results
        lines = output.split("\n")
        results = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                results[key.strip()] = value.strip()
        
        logger.info(f"Benchmark completed successfully")
        logger.info(f"Results: {results}")
        
        return results
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        return None

def optimize_tensorrt_llm_for_amd(engine_dir):
    """Optimize TensorRT-LLM for AMD GPUs.
    
    Args:
        engine_dir: Engine directory
    
    Returns:
        bool: True if optimization is successful, False otherwise
    """
    try:
        import tensorrt_llm
        
        # Optimize engine
        logger.info(f"Optimizing TensorRT-LLM for AMD GPUs")
        
        subprocess.run(
            [
                "python", "examples/optimize_engine.py",
                "--engine_dir", engine_dir,
                "--rocm"
            ],
            check=True
        )
        
        logger.info(f"Engine optimized successfully")
        return True
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to optimize engine: {e}")
        return False

def create_tensorrt_llm_server(engine_dir, port=8000):
    """Create TensorRT-LLM server.
    
    Args:
        engine_dir: Engine directory
        port: Port number
    
    Returns:
        subprocess.Popen: Server process
    """
    try:
        import tensorrt_llm
        
        # Start server
        logger.info(f"Starting TensorRT-LLM server on port {port}")
        
        server_process = subprocess.Popen(
            [
                "python", "examples/server.py",
                "--engine_dir", engine_dir,
                "--port", str(port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for server to start
        time.sleep(5)
        
        logger.info(f"Server started successfully")
        return server_process
    except ImportError:
        logger.error("TensorRT-LLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None

def query_tensorrt_llm_server(input_text, port=8000, max_output_len=1024, temperature=0.7, top_p=0.9, top_k=50):
    """Query TensorRT-LLM server.
    
    Args:
        input_text: Input text
        port: Port number
        max_output_len: Maximum output length
        temperature: Temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        str: Generated text
    """
    try:
        import requests
        
        # Query server
        logger.info(f"Querying TensorRT-LLM server")
        
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "input": input_text,
                "max_output_len": max_output_len,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        
        # Parse response
        result = response.json()
        
        logger.info(f"Query completed successfully")
        logger.info(f"Output: {result['output']}")
        
        return result["output"]
    except ImportError:
        logger.error("Requests is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to query server: {e}")
        return None

if __name__ == "__main__":
    # Check TensorRT-LLM installation
    check_tensorrt_llm_installation()
    
    # Example usage
    if check_tensorrt_llm_installation():
        # Convert model
        convert_hf_model_to_tensorrt_llm(
            model_name="gpt2",
            output_dir="tensorrt_llm_models/gpt2",
            precision="fp16",
            max_batch_size=1,
            max_input_len=1024,
            max_output_len=1024
        )
        
        # Build engine
        build_tensorrt_llm_engine(
            model_dir="tensorrt_llm_models/gpt2",
            engine_dir="tensorrt_llm_engines/gpt2",
            precision="fp16",
            max_batch_size=1,
            max_input_len=1024,
            max_output_len=1024
        )
        
        # Optimize engine
        optimize_tensorrt_llm_for_amd(
            engine_dir="tensorrt_llm_engines/gpt2"
        )
        
        # Run inference
        run_tensorrt_llm_inference(
            engine_dir="tensorrt_llm_engines/gpt2",
            input_text="Once upon a time",
            max_output_len=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Benchmark
        benchmark_tensorrt_llm(
            engine_dir="tensorrt_llm_engines/gpt2",
            input_text="Once upon a time",
            num_iterations=10,
            max_output_len=100
        )
