#!/usr/bin/env python3
# =============================================================================
# RCCL Utilities
# =============================================================================
# This module provides utilities for working with RCCL (ROCm Communication Collective Library).
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
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rccl_utils")

def check_rccl_installation():
    """Check if RCCL is installed.
    
    Returns:
        bool: True if RCCL is installed, False otherwise
    """
    try:
        # Check if RCCL library exists
        result = subprocess.run(
            ["ls", "-la", "/opt/rocm/lib/librccl*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info("RCCL is installed")
            logger.info(result.stdout)
            return True
        else:
            logger.error("RCCL is not installed")
            return False
    except Exception as e:
        logger.error(f"Failed to check RCCL installation: {e}")
        return False

def set_rccl_environment_variables():
    """Set RCCL environment variables for optimal performance.
    
    Returns:
        bool: True if environment variables are set, False otherwise
    """
    try:
        # Set environment variables
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "^lo")
        os.environ["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "0")
        os.environ["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
        
        logger.info("RCCL environment variables set")
        logger.info(f"NCCL_DEBUG: {os.environ['NCCL_DEBUG']}")
        logger.info(f"NCCL_SOCKET_IFNAME: {os.environ['NCCL_SOCKET_IFNAME']}")
        logger.info(f"NCCL_P2P_DISABLE: {os.environ['NCCL_P2P_DISABLE']}")
        logger.info(f"NCCL_IB_DISABLE: {os.environ['NCCL_IB_DISABLE']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set RCCL environment variables: {e}")
        return False

def run_rccl_tests(test_type="all", num_gpus=None):
    """Run RCCL tests.
    
    Args:
        test_type: Type of test to run (all, allreduce, broadcast, reduce, allgather, reducescatter)
        num_gpus: Number of GPUs to use (default: all available GPUs)
    
    Returns:
        bool: True if tests pass, False otherwise
    """
    try:
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available through ROCm")
            return False
        
        # Get number of GPUs
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        # Check if RCCL tests are installed
        result = subprocess.run(
            ["which", "rccl-tests"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("RCCL tests are not installed")
            logger.info("Please install RCCL tests first")
            return False
        
        # Set test command
        if test_type == "all":
            test_commands = [
                ["all_reduce_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)],
                ["broadcast_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)],
                ["reduce_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)],
                ["all_gather_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)],
                ["reduce_scatter_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]
            ]
        elif test_type == "allreduce":
            test_commands = [["all_reduce_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]]
        elif test_type == "broadcast":
            test_commands = [["broadcast_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]]
        elif test_type == "reduce":
            test_commands = [["reduce_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]]
        elif test_type == "allgather":
            test_commands = [["all_gather_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]]
        elif test_type == "reducescatter":
            test_commands = [["reduce_scatter_perf", "-b", "8", "-e", "128M", "-f", "2", "-g", str(num_gpus)]]
        else:
            logger.error(f"Unsupported test type: {test_type}")
            return False
        
        # Run tests
        for test_command in test_commands:
            logger.info(f"Running RCCL test: {' '.join(test_command)}")
            
            result = subprocess.run(
                test_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode == 0:
                logger.info(f"RCCL test {test_command[0]} passed")
                logger.info(result.stdout)
            else:
                logger.error(f"RCCL test {test_command[0]} failed")
                logger.error(result.stderr)
                return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to run RCCL tests: {e}")
        return False

def check_rccl_with_pytorch():
    """Check RCCL with PyTorch.
    
    Returns:
        bool: True if RCCL works with PyTorch, False otherwise
    """
    try:
        import torch
        import torch.distributed as dist
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available through ROCm")
            return False
        
        # Initialize process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        
        # Create tensor on GPU
        tensor = torch.randn(10, device="cuda")
        
        # Perform all-reduce
        dist.all_reduce(tensor)
        
        # Clean up
        dist.destroy_process_group()
        
        logger.info("RCCL works with PyTorch")
        
        return True
    except Exception as e:
        logger.error(f"Failed to check RCCL with PyTorch: {e}")
        return False

def install_rccl():
    """Install RCCL.
    
    Returns:
        bool: True if installation is successful, False otherwise
    """
    try:
        # Check if RCCL is already installed
        if check_rccl_installation():
            logger.info("RCCL is already installed")
            return True
        
        # Install RCCL
        logger.info("Installing RCCL")
        
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "rccl"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info("RCCL installed successfully")
            return True
        else:
            logger.error("Failed to install RCCL")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Failed to install RCCL: {e}")
        return False

def build_rccl_from_source():
    """Build RCCL from source.
    
    Returns:
        bool: True if build is successful, False otherwise
    """
    try:
        # Clone RCCL repository
        logger.info("Cloning RCCL repository")
        
        result = subprocess.run(
            ["git", "clone", "https://github.com/ROCmSoftwarePlatform/rccl.git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to clone RCCL repository")
            logger.error(result.stderr)
            return False
        
        # Build RCCL
        logger.info("Building RCCL")
        
        os.chdir("rccl")
        
        # Create build directory
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        
        # Configure
        result = subprocess.run(
            ["cmake", ".."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to configure RCCL")
            logger.error(result.stderr)
            return False
        
        # Build
        result = subprocess.run(
            ["make", "-j4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to build RCCL")
            logger.error(result.stderr)
            return False
        
        # Install
        result = subprocess.run(
            ["sudo", "make", "install"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to install RCCL")
            logger.error(result.stderr)
            return False
        
        logger.info("RCCL built and installed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to build RCCL from source: {e}")
        return False

def benchmark_rccl(num_gpus=None, data_size="1G"):
    """Benchmark RCCL performance.
    
    Args:
        num_gpus: Number of GPUs to use (default: all available GPUs)
        data_size: Data size for benchmark (e.g., 1G, 10G)
    
    Returns:
        dict: Dictionary of benchmark results
    """
    try:
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available through ROCm")
            return None
        
        # Get number of GPUs
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        # Check if RCCL tests are installed
        result = subprocess.run(
            ["which", "all_reduce_perf"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("RCCL tests are not installed")
            logger.info("Please install RCCL tests first")
            return None
        
        # Run benchmark
        logger.info(f"Running RCCL benchmark with {num_gpus} GPUs and data size {data_size}")
        
        result = subprocess.run(
            ["all_reduce_perf", "-b", data_size, "-e", data_size, "-f", "2", "-g", str(num_gpus), "-c", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to run RCCL benchmark")
            logger.error(result.stderr)
            return None
        
        # Parse benchmark results
        output = result.stdout
        lines = output.strip().split("\n")
        
        # Find result line
        result_line = None
        for line in lines:
            if line.startswith("#"):
                continue
            result_line = line
            break
        
        if result_line is None:
            logger.error("Failed to parse benchmark results")
            return None
        
        # Parse result line
        fields = result_line.split()
        
        if len(fields) < 7:
            logger.error("Failed to parse benchmark results")
            return None
        
        # Extract results
        size = fields[0]
        count = fields[1]
        time = float(fields[2])
        algbw = float(fields[3])
        busbw = float(fields[4])
        
        # Create result dictionary
        results = {
            "size": size,
            "count": count,
            "time": time,
            "algbw": algbw,
            "busbw": busbw
        }
        
        logger.info(f"RCCL benchmark results: {results}")
        
        return results
    except Exception as e:
        logger.error(f"Failed to benchmark RCCL: {e}")
        return None

if __name__ == "__main__":
    # Check RCCL installation
    check_rccl_installation()
    
    # Set RCCL environment variables
    set_rccl_environment_variables()
    
    # Check RCCL with PyTorch
    check_rccl_with_pytorch()
    
    # Run RCCL tests
    run_rccl_tests()
    
    # Benchmark RCCL
    benchmark_rccl()
