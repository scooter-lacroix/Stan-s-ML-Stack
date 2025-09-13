#!/usr/bin/env python3
# =============================================================================
# vLLM for AMD GPUs
# =============================================================================
# This module provides utilities for using vLLM with AMD GPUs.
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
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm_amd")

def check_vllm_installation():
    """Check if vLLM is installed.
    
    Returns:
        bool: True if vLLM is installed, False otherwise
    """
    try:
        import vllm
        logger.info(f"vLLM is installed (version {vllm.__version__})")
        return True
    except ImportError:
        logger.error("vLLM is not installed")
        logger.info("Please install vLLM first")
        return False

def install_vllm_for_amd():
    """Install vLLM for AMD GPUs.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        # Clone repository
        logger.info("Cloning vLLM repository")
        subprocess.run(
            ["git", "clone", "https://github.com/vllm-project/vllm.git"],
            check=True
        )
        
        # Change directory
        os.chdir("vllm")
        
        # Set environment variables for AMD
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"  # For RDNA 3 GPUs (RX 7000 series)
        
        # Install dependencies
        logger.info("Installing dependencies")
        subprocess.run(
            ["pip", "install", "-e", ".[rocm]"],
            check=True
        )
        
        logger.info("vLLM installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install vLLM: {e}")
        return False

def load_model(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048):
    """Load model with vLLM.
    
    Args:
        model_name: Model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model length
    
    Returns:
        vllm.LLM: vLLM model
    """
    try:
        import vllm
        
        # Load model
        logger.info(f"Loading model {model_name}")
        
        model = vllm.LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        
        logger.info(f"Model loaded successfully")
        return model
    except ImportError:
        logger.error("vLLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def generate_text(model, prompts, max_tokens=100, temperature=0.7, top_p=0.9, top_k=50):
    """Generate text with vLLM.
    
    Args:
        model: vLLM model
        prompts: List of prompts
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        list: List of generated texts
    """
    try:
        import vllm
        
        # Create sampling params
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # Generate text
        logger.info(f"Generating text for {len(prompts)} prompts")
        
        outputs = model.generate(prompts, sampling_params)
        
        # Extract generated texts
        generated_texts = [output.outputs[0].text for output in outputs]
        
        logger.info(f"Text generation completed successfully")
        
        return generated_texts
    except ImportError:
        logger.error("vLLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        return None

def benchmark_vllm(model_name, prompt, num_iterations=10, max_tokens=100, tensor_parallel_size=1):
    """Benchmark vLLM.
    
    Args:
        model_name: Model name
        prompt: Prompt
        num_iterations: Number of iterations
        max_tokens: Maximum number of tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        dict: Benchmark results
    """
    try:
        import vllm
        
        # Load model
        model = load_model(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size
        )
        
        # Create sampling params
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Warm-up
        logger.info("Warming up")
        _ = model.generate([prompt], sampling_params)
        
        # Benchmark
        logger.info(f"Benchmarking vLLM with {num_iterations} iterations")
        
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = model.generate([prompt], sampling_params)
        
        end_time = time.time()
        
        # Calculate results
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        tokens_per_second = max_tokens / avg_time
        
        logger.info(f"Benchmark completed successfully")
        logger.info(f"Average time: {avg_time:.2f} seconds")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        
        return {
            "total_time": total_time,
            "avg_time": avg_time,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "max_tokens": max_tokens
        }
    except ImportError:
        logger.error("vLLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to benchmark vLLM: {e}")
        return None

def create_vllm_engine(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048):
    """Create vLLM engine.
    
    Args:
        model_name: Model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model length
    
    Returns:
        vllm.LLMEngine: vLLM engine
    """
    try:
        import vllm
        
        # Create engine
        logger.info(f"Creating vLLM engine for {model_name}")
        
        engine = vllm.LLMEngine.from_engine_args(
            vllm.EngineArgs(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len
            )
        )
        
        logger.info(f"Engine created successfully")
        return engine
    except ImportError:
        logger.error("vLLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create engine: {e}")
        return None

def create_vllm_server(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048, port=8000):
    """Create vLLM server.
    
    Args:
        model_name: Model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model length
        port: Port number
    
    Returns:
        subprocess.Popen: Server process
    """
    try:
        # Start server
        logger.info(f"Starting vLLM server on port {port}")
        
        server_process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.api_server",
                "--model", model_name,
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
                "--port", str(port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for server to start
        time.sleep(10)
        
        logger.info(f"Server started successfully")
        return server_process
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None

def query_vllm_server(prompt, port=8000, max_tokens=100, temperature=0.7, top_p=0.9, top_k=50):
    """Query vLLM server.
    
    Args:
        prompt: Prompt
        port: Port number
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        str: Generated text
    """
    try:
        import requests
        
        # Query server
        logger.info(f"Querying vLLM server")
        
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        
        # Parse response
        result = response.json()
        
        logger.info(f"Query completed successfully")
        logger.info(f"Output: {result['text']}")
        
        return result["text"]
    except ImportError:
        logger.error("Requests is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to query server: {e}")
        return None

def optimize_vllm_for_amd(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048):
    """Optimize vLLM for AMD GPUs.
    
    Args:
        model_name: Model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model length
    
    Returns:
        vllm.LLM: Optimized vLLM model
    """
    try:
        import vllm
        
        # Set environment variables for AMD
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for better performance
        
        # Load model with optimized settings
        logger.info(f"Loading model {model_name} with optimized settings")
        
        model = vllm.LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,  # Use CUDA graphs for better performance
            max_context_len_to_capture=max_model_len  # Capture the entire context
        )
        
        logger.info(f"Model loaded successfully with optimized settings")
        return model
    except ImportError:
        logger.error("vLLM is not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to optimize vLLM: {e}")
        return None

def run_vllm_with_kv_cache_visualization(model_name, prompt, max_tokens=100, tensor_parallel_size=1):
    """Run vLLM with KV cache visualization.
    
    Args:
        model_name: Model name
        prompt: Prompt
        max_tokens: Maximum number of tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        tuple: Generated text and KV cache visualization
    """
    try:
        import vllm
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Load model
        model = load_model(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size
        )
        
        # Create sampling params
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Generate text
        logger.info(f"Generating text with KV cache visualization")
        
        outputs = model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Get KV cache stats
        kv_cache_stats = model.get_kv_cache_stats()
        
        # Visualize KV cache
        plt.figure(figsize=(10, 6))
        
        # Plot KV cache usage
        plt.subplot(2, 1, 1)
        plt.bar(["Used", "Free"], [kv_cache_stats["used_slots"], kv_cache_stats["free_slots"]])
        plt.title("KV Cache Usage")
        plt.ylabel("Number of Slots")
        
        # Plot KV cache efficiency
        plt.subplot(2, 1, 2)
        efficiency = kv_cache_stats["used_slots"] / (kv_cache_stats["used_slots"] + kv_cache_stats["free_slots"]) * 100
        plt.bar(["Efficiency"], [efficiency])
        plt.title("KV Cache Efficiency")
        plt.ylabel("Efficiency (%)")
        plt.ylim(0, 100)
        
        plt.tight_layout()
        
        # Save visualization
        plt.savefig("kv_cache_visualization.png")
        
        logger.info(f"Text generation and KV cache visualization completed successfully")
        
        return generated_text, kv_cache_stats
    except ImportError:
        logger.error("vLLM or matplotlib is not installed")
        return None, None
    except Exception as e:
        logger.error(f"Failed to run vLLM with KV cache visualization: {e}")
        return None, None

if __name__ == "__main__":
    # Check vLLM installation
    check_vllm_installation()
    
    # Example usage
    if check_vllm_installation():
        # Load model
        model = load_model(
            model_name="gpt2",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )
        
        # Generate text
        generated_texts = generate_text(
            model=model,
            prompts=["Once upon a time"],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Print generated texts
        for i, text in enumerate(generated_texts):
            logger.info(f"Generated text {i+1}: {text}")
        
        # Benchmark vLLM
        benchmark_results = benchmark_vllm(
            model_name="gpt2",
            prompt="Once upon a time",
            num_iterations=10,
            max_tokens=100,
            tensor_parallel_size=1
        )
        
        # Print benchmark results
        logger.info(f"Benchmark results: {benchmark_results}")
