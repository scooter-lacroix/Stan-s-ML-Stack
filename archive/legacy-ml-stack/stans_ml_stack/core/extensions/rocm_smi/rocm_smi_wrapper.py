#!/usr/bin/env python3
# =============================================================================
# ROCm SMI Wrapper
# =============================================================================
# This module provides a Python wrapper for ROCm System Management Interface.
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
import json
import subprocess
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rocm_smi_wrapper")

class ROCmSMI:
    """ROCm System Management Interface wrapper."""
    
    def __init__(self):
        """Initialize ROCmSMI wrapper."""
        self.check_rocm_smi()
    
    def check_rocm_smi(self):
        """Check if ROCm SMI is available.
        
        Returns:
            bool: True if ROCm SMI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "rocm-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode == 0:
                logger.info(f"ROCm SMI is available: {result.stdout.strip()}")
                return True
            else:
                logger.error("ROCm SMI is not available")
                return False
        except Exception as e:
            logger.error(f"Failed to check ROCm SMI: {e}")
            return False
    
    def get_gpu_info(self):
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
    
    def get_gpu_usage(self):
        """Get GPU usage.
        
        Returns:
            list: List of dictionaries with GPU usage information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showclocks", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU usage")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_usage = json.loads(result.stdout)
            
            # Print GPU usage
            for i, gpu in enumerate(gpu_usage):
                logger.info(f"GPU {i}:")
                logger.info(f"  GPU use: {gpu.get('GPU use (%)', 'Unknown')}")
                logger.info(f"  Memory use: {gpu.get('GPU memory use (%)', 'Unknown')}")
                logger.info(f"  VRAM used: {gpu.get('GPU memory', {}).get('used', 'Unknown')}")
                logger.info(f"  Clock: {gpu.get('Clocks', {}).get('sclk', 'Unknown')}")
            
            return gpu_usage
        except Exception as e:
            logger.error(f"Failed to get GPU usage: {e}")
            return None
    
    def get_gpu_temperature(self):
        """Get GPU temperature.
        
        Returns:
            list: List of dictionaries with GPU temperature information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU temperature")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_temp = json.loads(result.stdout)
            
            # Print GPU temperature
            for i, gpu in enumerate(gpu_temp):
                logger.info(f"GPU {i}:")
                logger.info(f"  Edge temperature: {gpu.get('Temperature', {}).get('edge', 'Unknown')}")
                logger.info(f"  Junction temperature: {gpu.get('Temperature', {}).get('junction', 'Unknown')}")
                logger.info(f"  Memory temperature: {gpu.get('Temperature', {}).get('memory', 'Unknown')}")
            
            return gpu_temp
        except Exception as e:
            logger.error(f"Failed to get GPU temperature: {e}")
            return None
    
    def get_gpu_clock(self):
        """Get GPU clock.
        
        Returns:
            list: List of dictionaries with GPU clock information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showclocks", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU clock")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_clock = json.loads(result.stdout)
            
            # Print GPU clock
            for i, gpu in enumerate(gpu_clock):
                logger.info(f"GPU {i}:")
                logger.info(f"  SCLK: {gpu.get('Clocks', {}).get('sclk', 'Unknown')}")
                logger.info(f"  MCLK: {gpu.get('Clocks', {}).get('mclk', 'Unknown')}")
            
            return gpu_clock
        except Exception as e:
            logger.error(f"Failed to get GPU clock: {e}")
            return None
    
    def get_gpu_fan(self):
        """Get GPU fan speed.
        
        Returns:
            list: List of dictionaries with GPU fan information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showfan", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU fan speed")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_fan = json.loads(result.stdout)
            
            # Print GPU fan speed
            for i, gpu in enumerate(gpu_fan):
                logger.info(f"GPU {i}:")
                logger.info(f"  Fan speed: {gpu.get('Fan speed', 'Unknown')}")
            
            return gpu_fan
        except Exception as e:
            logger.error(f"Failed to get GPU fan speed: {e}")
            return None
    
    def get_gpu_power(self):
        """Get GPU power.
        
        Returns:
            list: List of dictionaries with GPU power information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU power")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_power = json.loads(result.stdout)
            
            # Print GPU power
            for i, gpu in enumerate(gpu_power):
                logger.info(f"GPU {i}:")
                logger.info(f"  Average power: {gpu.get('Power', {}).get('average', 'Unknown')}")
                logger.info(f"  Cap: {gpu.get('Power', {}).get('cap', 'Unknown')}")
            
            return gpu_power
        except Exception as e:
            logger.error(f"Failed to get GPU power: {e}")
            return None
    
    def get_gpu_memory(self):
        """Get GPU memory.
        
        Returns:
            list: List of dictionaries with GPU memory information
        """
        try:
            # Run rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if result.returncode != 0:
                logger.error("Failed to get GPU memory")
                logger.error(result.stderr)
                return None
            
            # Parse JSON output
            gpu_memory = json.loads(result.stdout)
            
            # Print GPU memory
            for i, gpu in enumerate(gpu_memory):
                logger.info(f"GPU {i}:")
                logger.info(f"  Total memory: {gpu.get('GPU memory', {}).get('total', 'Unknown')}")
                logger.info(f"  Used memory: {gpu.get('GPU memory', {}).get('used', 'Unknown')}")
                logger.info(f"  Free memory: {gpu.get('GPU memory', {}).get('free', 'Unknown')}")
            
            return gpu_memory
        except Exception as e:
            logger.error(f"Failed to get GPU memory: {e}")
            return None
    
    def set_gpu_clock(self, level="high", gpu_id=None):
        """Set GPU clock level.
        
        Args:
            level: Clock level (low, medium, high, auto)
            gpu_id: GPU ID (None for all GPUs)
        
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
            
            # Set GPU ID
            gpu_id_arg = f"--id {gpu_id}" if gpu_id is not None else ""
            
            # Set clock level
            logger.info(f"Setting GPU clock level to {level}")
            
            result = subprocess.run(
                f"sudo rocm-smi --setsclk {clock_level} {gpu_id_arg}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True
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
    
    def reset_gpu_clock(self, gpu_id=None):
        """Reset GPU clock to default.
        
        Args:
            gpu_id: GPU ID (None for all GPUs)
        
        Returns:
            bool: True if clock is reset, False otherwise
        """
        try:
            # Set GPU ID
            gpu_id_arg = f"--id {gpu_id}" if gpu_id is not None else ""
            
            # Reset clock
            logger.info("Resetting GPU clock to default")
            
            result = subprocess.run(
                f"sudo rocm-smi --resetclocks {gpu_id_arg}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True
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
    
    def set_gpu_fan(self, speed=None, gpu_id=None):
        """Set GPU fan speed.
        
        Args:
            speed: Fan speed in percentage (0-100), None for auto
            gpu_id: GPU ID (None for all GPUs)
        
        Returns:
            bool: True if fan speed is set, False otherwise
        """
        try:
            # Set GPU ID
            gpu_id_arg = f"--id {gpu_id}" if gpu_id is not None else ""
            
            # Set fan speed
            if speed is None:
                logger.info("Setting GPU fan speed to auto")
                
                result = subprocess.run(
                    f"sudo rocm-smi --resetfans {gpu_id_arg}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
                )
            else:
                logger.info(f"Setting GPU fan speed to {speed}%")
                
                result = subprocess.run(
                    f"sudo rocm-smi --setfan {speed} {gpu_id_arg}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
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
    
    def set_gpu_power_limit(self, limit=None, gpu_id=None):
        """Set GPU power limit.
        
        Args:
            limit: Power limit in watts, None for default
            gpu_id: GPU ID (None for all GPUs)
        
        Returns:
            bool: True if power limit is set, False otherwise
        """
        try:
            # Set GPU ID
            gpu_id_arg = f"--id {gpu_id}" if gpu_id is not None else ""
            
            # Set power limit
            if limit is None:
                logger.info("Resetting GPU power limit to default")
                
                result = subprocess.run(
                    f"sudo rocm-smi --resetpoweroverdrive {gpu_id_arg}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
                )
            else:
                logger.info(f"Setting GPU power limit to {limit}W")
                
                result = subprocess.run(
                    f"sudo rocm-smi --setpoweroverdrive {limit} {gpu_id_arg}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
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
    
    def monitor_gpu(self, interval=1, count=None, output_file=None):
        """Monitor GPU.
        
        Args:
            interval: Interval between measurements in seconds
            count: Number of measurements (None for infinite)
            output_file: Output file path
        
        Returns:
            list: List of dictionaries with GPU information
        """
        try:
            logger.info(f"Monitoring GPU every {interval} seconds")
            
            # Initialize results
            results = []
            
            # Open output file
            if output_file:
                f = open(output_file, "w")
                f.write("timestamp,gpu_id,gpu_use,memory_use,temperature,power\n")
            
            # Monitor GPU
            i = 0
            while count is None or i < count:
                # Run rocm-smi
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                if result.returncode != 0:
                    logger.error("Failed to monitor GPU")
                    logger.error(result.stderr)
                    break
                
                # Parse JSON output
                gpu_info = json.loads(result.stdout)
                
                # Add timestamp
                timestamp = time.time()
                for gpu in gpu_info:
                    gpu["timestamp"] = timestamp
                
                # Add to results
                results.append(gpu_info)
                
                # Print GPU information
                for j, gpu in enumerate(gpu_info):
                    gpu_use = gpu.get("GPU use (%)", "Unknown")
                    memory_use = gpu.get("GPU memory use (%)", "Unknown")
                    temperature = gpu.get("Temperature", {}).get("edge", "Unknown")
                    power = gpu.get("Power", {}).get("average", "Unknown")
                    
                    logger.info(f"GPU {j}:")
                    logger.info(f"  GPU use: {gpu_use}")
                    logger.info(f"  Memory use: {memory_use}")
                    logger.info(f"  Temperature: {temperature}")
                    logger.info(f"  Power: {power}")
                    
                    # Write to output file
                    if output_file:
                        f.write(f"{timestamp},{j},{gpu_use},{memory_use},{temperature},{power}\n")
                        f.flush()
                
                # Increment counter
                i += 1
                
                # Wait for next measurement
                if count is None or i < count:
                    time.sleep(interval)
            
            # Close output file
            if output_file:
                f.close()
            
            return results
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            
            # Close output file
            if output_file:
                f.close()
            
            return results
        except Exception as e:
            logger.error(f"Failed to monitor GPU: {e}")
            
            # Close output file
            if output_file:
                f.close()
            
            return results
    
    def start_monitoring_thread(self, interval=1, output_file=None):
        """Start monitoring thread.
        
        Args:
            interval: Interval between measurements in seconds
            output_file: Output file path
        
        Returns:
            threading.Thread: Monitoring thread
        """
        try:
            # Create monitoring thread
            self.monitoring_thread_stop = threading.Event()
            
            def monitoring_thread_func():
                try:
                    # Open output file
                    if output_file:
                        f = open(output_file, "w")
                        f.write("timestamp,gpu_id,gpu_use,memory_use,temperature,power\n")
                    
                    # Monitor GPU
                    while not self.monitoring_thread_stop.is_set():
                        # Run rocm-smi
                        result = subprocess.run(
                            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True
                        )
                        
                        if result.returncode != 0:
                            logger.error("Failed to monitor GPU")
                            logger.error(result.stderr)
                            break
                        
                        # Parse JSON output
                        gpu_info = json.loads(result.stdout)
                        
                        # Add timestamp
                        timestamp = time.time()
                        
                        # Print GPU information
                        for j, gpu in enumerate(gpu_info):
                            gpu_use = gpu.get("GPU use (%)", "Unknown")
                            memory_use = gpu.get("GPU memory use (%)", "Unknown")
                            temperature = gpu.get("Temperature", {}).get("edge", "Unknown")
                            power = gpu.get("Power", {}).get("average", "Unknown")
                            
                            # Write to output file
                            if output_file:
                                f.write(f"{timestamp},{j},{gpu_use},{memory_use},{temperature},{power}\n")
                                f.flush()
                        
                        # Wait for next measurement
                        self.monitoring_thread_stop.wait(interval)
                    
                    # Close output file
                    if output_file:
                        f.close()
                except Exception as e:
                    logger.error(f"Monitoring thread error: {e}")
                    
                    # Close output file
                    if output_file:
                        f.close()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=monitoring_thread_func)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Monitoring thread started")
            
            return self.monitoring_thread
        except Exception as e:
            logger.error(f"Failed to start monitoring thread: {e}")
            return None
    
    def stop_monitoring_thread(self):
        """Stop monitoring thread.
        
        Returns:
            bool: True if thread is stopped, False otherwise
        """
        try:
            # Stop monitoring thread
            if hasattr(self, "monitoring_thread") and self.monitoring_thread.is_alive():
                self.monitoring_thread_stop.set()
                self.monitoring_thread.join()
                logger.info("Monitoring thread stopped")
                return True
            else:
                logger.warning("Monitoring thread is not running")
                return False
        except Exception as e:
            logger.error(f"Failed to stop monitoring thread: {e}")
            return False
    
    def get_gpu_topology(self):
        """Get GPU topology.
        
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

if __name__ == "__main__":
    # Create ROCmSMI wrapper
    rocm_smi = ROCmSMI()
    
    # Get GPU information
    rocm_smi.get_gpu_info()
    
    # Get GPU usage
    rocm_smi.get_gpu_usage()
    
    # Get GPU temperature
    rocm_smi.get_gpu_temperature()
    
    # Get GPU clock
    rocm_smi.get_gpu_clock()
    
    # Get GPU fan speed
    rocm_smi.get_gpu_fan()
    
    # Get GPU power
    rocm_smi.get_gpu_power()
    
    # Get GPU memory
    rocm_smi.get_gpu_memory()
    
    # Get GPU topology
    rocm_smi.get_gpu_topology()
    
    # Monitor GPU
    rocm_smi.monitor_gpu(interval=1, count=3)
