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
from setuptools import setup, find_packages

# Check if PyTorch is installed
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
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

# Define package metadata
setup(
    name="flash_attention_amd",
    version="0.1.0",
    description="Flash Attention implementation for AMD GPUs",
    author="User",
    author_email="user@example.com",
    url="https://github.com/user/flash-attention-amd",
    packages=find_packages(),
    py_modules=["flash_attention_amd"],
    install_requires=[
        "torch>=2.0.0",
    ],
    ext_modules=[
        CUDAExtension(
            name="flash_attention_amd_cuda",
            sources=[
                "flash_attention_amd_cuda.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
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
