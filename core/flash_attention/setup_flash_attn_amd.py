#!/usr/bin/env python3
# =============================================================================
# Setup Script for Flash Attention AMD
# =============================================================================
# This script sets up the Flash Attention package for AMD GPUs.
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
from setuptools import setup, find_packages

# Check if PyTorch is installed
try:
    import torch
except ImportError:
    print("PyTorch is not installed. Please install PyTorch first.")
    sys.exit(1)

# Check if CUDA (ROCm) is available
if not torch.cuda.is_available():
    print("CUDA (ROCm) is not available. Please check your installation.")
    sys.exit(1)

# Get ROCm version
rocm_version = torch.version.hip if hasattr(torch.version, 'hip') else "unknown"
print(f"ROCm version: {rocm_version}")

# Get PyTorch version
pytorch_version = torch.__version__
print(f"PyTorch version: {pytorch_version}")

# Build the C++ extension using CMake
def build_cpp_extension():
    print("Building C++ extension using CMake...")

    # Create build directory
    os.makedirs("build", exist_ok=True)
    os.chdir("build")

    # Configure CMake
    cmake_cmd = [
        "cmake", "..",
        f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        "-DCMAKE_BUILD_TYPE=Release"
    ]

    # Set ROCm path if available
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    cmake_cmd.append(f"-DROCM_PATH={rocm_path}")

    # Run CMake
    print(f"Running: {' '.join(cmake_cmd)}")
    subprocess.run(cmake_cmd, check=True)

    # Build
    build_cmd = ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())]
    print(f"Running: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, check=True)

    # Install
    install_cmd = ["cmake", "--install", "."]
    print(f"Running: {' '.join(install_cmd)}")
    subprocess.run(install_cmd, check=True)

    # Go back to the original directory
    os.chdir("..")

    # Copy the built extension to the package directory
    if os.path.exists('flash_attention_amd_cuda.so'):
        import shutil
        shutil.copy('flash_attention_amd_cuda.so', 'flash_attention_amd/')
        print("Copied extension to package directory")

    print("C++ extension built successfully")

# Build the C++ extension
try:
    build_cpp_extension()
except Exception as e:
    print(f"Failed to build C++ extension: {e}")
    print("Falling back to pure Python implementation")

# Create package directory structure
os.makedirs('flash_attention_amd', exist_ok=True)

# Create __init__.py
with open('flash_attention_amd/__init__.py', 'w') as f:
    f.write("""#!/usr/bin/env python3
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

from .flash_attention_amd import FlashAttention, flash_attn_func

__all__ = ['FlashAttention', 'flash_attn_func']
""")

# Copy the main module file
import shutil
if os.path.exists('flash_attention_amd.py'):
    shutil.copy('flash_attention_amd.py', 'flash_attention_amd/flash_attention_amd.py')

# Define package metadata
setup(
    name="flash_attention_amd",
    version="0.1.0",
    description="Flash Attention implementation for AMD GPUs",
    author="Stanley Chisango",
    author_email="scooterlacroix@gmail.com",
    url="https://github.com/scooter-lacroix/flash-attention-amd",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    # No ext_modules needed as we're building with CMake
    # Include pre-built extension
    package_data={
        "flash_attention_amd": ["*.so"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
