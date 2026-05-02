#!/usr/bin/env python3
# =============================================================================
# ROCm Utilities
# =============================================================================
# This module provides utilities for working with ROCm on AMD GPUs.
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
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rocm_utils")

def check_rocm_installation():
    """Check if ROCm is installed.
    
    Returns:
        bool: True if ROCm is installed, False otherwise
    """
    try:
        # Check if rocm-smi is available
        result = subprocess.run(
            ["which", "rocm-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info(f"ROCm is installed: {result.stdout.strip()}")
            
            # Get ROCm version
            version_result = subprocess.run(
                ["rocm-smi", "--showversion"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if version_result.returncode == 0:
                logger.info(f"ROCm version: {version_result.stdout.strip()}")
            
            return True
        else:
            logger.error("ROCm is not installed")
            return False
    except Exception as e:
        logger.error(f"Failed to check ROCm installation: {e}")
        return False

def get_gpu_info():
    """Get information about AMD GPUs.
    
    Returns:
        list: List of dictionaries with GPU information
    """
    try:
        # Run rocm-smi with JSON output
        result = subprocess.run(
            ["rocm-smi", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to get GPU information")
            logger.error(result.stderr)
            return None
        
        # Parse JSON output
        gpu_info = json.loads(result.stdout)
        
        # Print GPU information
        for i, gpu in enumerate(gpu_info):
            logger.info(f"GPU {i}:")
            logger.info(f"  Name: {gpu.get('Card name', 'Unknown')}")
            logger.info(f"  Vendor: {gpu.get('Card vendor', 'Unknown')}")
            logger.info(f"  VRAM: {gpu.get('GPU memory', {}).get('total', 'Unknown')}")
            logger.info(f"  Temperature: {gpu.get('Temperature', {}).get('edge', 'Unknown')}")
            logger.info(f"  Clock: {gpu.get('Clocks', {}).get('sclk', 'Unknown')}")
            logger.info(f"  Fan: {gpu.get('Fan speed', 'Unknown')}")
            logger.info(f"  Power: {gpu.get('Power', {}).get('average', 'Unknown')}")
        
        return gpu_info
    except Exception as e:
        logger.error(f"Failed to get GPU information: {e}")
        return None

def set_gpu_environment_variables():
    """Set GPU environment variables for optimal performance.
    
    Returns:
        bool: True if environment variables are set, False otherwise
    """
    try:
        # Set environment variables
        os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        os.environ["PYTORCH_ROCM_DEVICE"] = os.environ.get("PYTORCH_ROCM_DEVICE", "0,1,2,3")
        
        # Set performance tuning variables
        os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for better performance
        os.environ["GPU_MAX_HEAP_SIZE"] = "100"  # Increase heap size (in %)
        os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"  # Allow allocating 100% of available memory
        os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"  # Allow single allocations up to 100%
        
        logger.info("GPU environment variables set")
        logger.info(f"HIP_VISIBLE_DEVICES: {os.environ['HIP_VISIBLE_DEVICES']}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logger.info(f"PYTORCH_ROCM_DEVICE: {os.environ['PYTORCH_ROCM_DEVICE']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set GPU environment variables: {e}")
        return False

def monitor_gpu_usage(interval=1, count=10):
    """Monitor GPU usage.
    
    Args:
        interval: Interval between measurements in seconds
        count: Number of measurements
    
    Returns:
        list: List of dictionaries with GPU usage information
    """
    try:
        import time
        
        logger.info(f"Monitoring GPU usage every {interval} seconds for {count} measurements")
        
        # Initialize results
        results = []
        
        # Monitor GPU usage
        for i in range(count):
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showclocks", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to monitor GPU usage")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_usage = json.loads(result.stdout)
            
            # Add timestamp
            for gpu in gpu_usage:
                gpu["timestamp"] = time.time()
            
            # Add to results
            results.append(gpu_usage)
            
            # Print GPU usage
            for j, gpu in enumerate(gpu_usage):
                logger.info(f"GPU {j}:")
                logger.info(f"  GPU use: {gpu.get('GPU use (%)', 'Unknown')}")
                logger.info(f"  Memory use: {gpu.get('GPU memory use (%)', 'Unknown')}")
                logger.info(f"  VRAM used: {gpu.get('GPU memory', {}).get('used', 'Unknown')}")
                logger.info(f"  Clock: {gpu.get('Clocks', {}).get('sclk', 'Unknown')}")
            
            # Wait for next measurement
            if i < count - 1:
                time.sleep(interval)
        
        return results
    except Exception as e:
        logger.error(f"Failed to monitor GPU usage: {e}")
        return None

def set_gpu_clock(level="high"):
    """Set GPU clock level.
    
    Args:
        level: Clock level (low, medium, high, auto)
    
    Returns:
        bool: True if clock level is set, False otherwise
    """
    try:
        # Map level to clock level
        if level == "low":
            clock_level = 0
        elif level == "medium":
            clock_level = 1
        elif level == "high":
            clock_level = 2
        elif level == "auto":
            clock_level = "auto"
        else:
            logger.error(f"Unsupported clock level: {level}")
            return False
        
        # Set clock level
        logger.info(f"Setting GPU clock level to {level}")
        
        result = subprocess.run(
            ["sudo", "rocm-smi", "--setsclk", str(clock_level)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info("GPU clock level set successfully")
            return True
        else:
            logger.error("Failed to set GPU clock level")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Failed to set GPU clock level: {e}")
        return False

def reset_gpu_clock():
    """Reset GPU clock to default.
    
    Returns:
        bool: True if clock is reset, False otherwise
    """
    try:
        # Reset clock
        logger.info("Resetting GPU clock to default")
        
        result = subprocess.run(
            ["sudo", "rocm-smi", "--resetclocks"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info("GPU clock reset successfully")
            return True
        else:
            logger.error("Failed to reset GPU clock")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Failed to reset GPU clock: {e}")
        return False

def set_gpu_fan(speed=None):
    """Set GPU fan speed.
    
    Args:
        speed: Fan speed in percentage (0-100), None for auto
    
    Returns:
        bool: True if fan speed is set, False otherwise
    """
    try:
        # Set fan speed
        if speed is None:
            logger.info("Setting GPU fan speed to auto")
            
            result = subprocess.run(
                ["sudo", "rocm-smi", "--resetfans"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            logger.info(f"Setting GPU fan speed to {speed}%")
            
            result = subprocess.run(
                ["sudo", "rocm-smi", "--setfan", str(speed)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        
        if result.returncode == 0:
            logger.info("GPU fan speed set successfully")
            return True
        else:
            logger.error("Failed to set GPU fan speed")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Failed to set GPU fan speed: {e}")
        return False

def set_gpu_power_limit(limit=None):
    """Set GPU power limit.
    
    Args:
        limit: Power limit in watts, None for default
    
    Returns:
        bool: True if power limit is set, False otherwise
    """
    try:
        # Set power limit
        if limit is None:
            logger.info("Resetting GPU power limit to default")
            
            result = subprocess.run(
                ["sudo", "rocm-smi", "--resetpoweroverdrive"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            logger.info(f"Setting GPU power limit to {limit}W")
            
            result = subprocess.run(
                ["sudo", "rocm-smi", "--setpoweroverdrive", str(limit)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        
        if result.returncode == 0:
            logger.info("GPU power limit set successfully")
            return True
        else:
            logger.error("Failed to set GPU power limit")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Failed to set GPU power limit: {e}")
        return False

def check_gpu_topology():
    """Check GPU topology.
    
    Returns:
        dict: Dictionary with GPU topology information
    """
    try:
        # Check GPU topology
        logger.info("Checking GPU topology")
        
        result = subprocess.run(
            ["rocm-smi", "--showtoponuma"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to check GPU topology")
            logger.error(result.stderr)
            return None
        
        # Print topology information
        logger.info(result.stdout)
        
        # Parse topology information
        lines = result.stdout.strip().split("\n")
        
        topology = {}
        current_gpu = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("GPU"):
                current_gpu = line.split()[1]
                topology[current_gpu] = {}
            elif ":" in line and current_gpu is not None:
                key, value = line.split(":", 1)
                topology[current_gpu][key.strip()] = value.strip()
        
        return topology
    except Exception as e:
        logger.error(f"Failed to check GPU topology: {e}")
        return None

def load_gpu_modules():
    """Load GPU kernel modules.
    
    Returns:
        bool: True if modules are loaded, False otherwise
    """
    try:
        # Load GPU modules
        logger.info("Loading GPU kernel modules")
        
        modules = ["amdgpu"]
        
        for module in modules:
            logger.info(f"Loading module: {module}")
            
            result = subprocess.run(
                ["sudo", "modprobe", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to load module: {module}")
                logger.error(result.stderr)
                return False
        
        logger.info("GPU kernel modules loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load GPU kernel modules: {e}")
        return False

def check_gpu_drivers():
    """Check GPU drivers.
    
    Returns:
        dict: Dictionary with GPU driver information
    """
    try:
        # Check GPU drivers
        logger.info("Checking GPU drivers")
        
        result = subprocess.run(
            ["lsmod", "|", "grep", "amdgpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to check GPU drivers")
            logger.error(result.stderr)
            return None
        
        # Print driver information
        logger.info(result.stdout)
        
        # Get driver version
        version_result = subprocess.run(
            ["modinfo", "amdgpu", "|", "grep", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        if version_result.returncode == 0:
            logger.info(version_result.stdout)
        
        # Parse driver information
        lines = result.stdout.strip().split("\n")
        
        drivers = {}
        
        for line in lines:
            fields = line.split()
            
            if len(fields) >= 3:
                drivers[fields[0]] = {
                    "size": fields[1],
                    "used_by": fields[2:]
                }
        
        return drivers
    except Exception as e:
        logger.error(f"Failed to check GPU drivers: {e}")
        return None

def benchmark_gpu():
    """Benchmark GPU performance.
    
    Returns:
        dict: Dictionary with benchmark results
    """
    try:
        # Check if rocm-bandwidth-test is available
        result = subprocess.run(
            ["which", "rocm-bandwidth-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("rocm-bandwidth-test is not available")
            logger.info("Please install rocm-bandwidth-test first")
            return None
        
        # Run benchmark
        logger.info("Running GPU benchmark")
        
        result = subprocess.run(
            ["rocm-bandwidth-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to run GPU benchmark")
            logger.error(result.stderr)
            return None
        
        # Print benchmark results
        logger.info(result.stdout)
        
        # Parse benchmark results
        lines = result.stdout.strip().split("\n")
        
        benchmark = {
            "devices": [],
            "bandwidths": {}
        }
        
        device_section = False
        bandwidth_section = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Devices:"):
                device_section = True
                bandwidth_section = False
                continue
            elif line.startswith("Unidirectional copy bandwidth"):
                device_section = False
                bandwidth_section = True
                continue
            
            if device_section and line:
                if line.startswith("Device"):
                    continue
                
                fields = line.split(":", 1)
                
                if len(fields) >= 2:
                    device_id = fields[0].strip()
                    device_info = fields[1].strip()
                    benchmark["devices"].append({
                        "id": device_id,
                        "info": device_info
                    })
            
            if bandwidth_section and line:
                if "to" in line and ":" in line:
                    source, dest = line.split("to", 1)
                    dest, bandwidth = dest.split(":", 1)
                    
                    source = source.strip()
                    dest = dest.strip()
                    bandwidth = bandwidth.strip()
                    
                    if source not in benchmark["bandwidths"]:
                        benchmark["bandwidths"][source] = {}
                    
                    benchmark["bandwidths"][source][dest] = bandwidth
        
        return benchmark
    except Exception as e:
        logger.error(f"Failed to benchmark GPU: {e}")
        return None

if __name__ == "__main__":
    # Check ROCm installation
    check_rocm_installation()
    
    # Get GPU information
    get_gpu_info()
    
    # Set GPU environment variables
    set_gpu_environment_variables()
    
    # Check GPU topology
    check_gpu_topology()
    
    # Check GPU drivers
    check_gpu_drivers()
    
    # Monitor GPU usage
    monitor_gpu_usage(interval=1, count=3)
