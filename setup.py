#!/usr/bin/env python3
"""
Setup script for Stan's ML Stack.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from VERSION file
version = "0.1.2"  # Default version
version_file = os.path.join(os.path.dirname(__file__), "VERSION")
if os.path.exists(version_file):
    with open(version_file, "r", encoding="utf-8") as f:
        version = f.read().strip()

setup(
    name="stans-ml-stack",
    version=version,
    author="Stanley Chisango (Scooter Lacroix)",
    author_email="scooterlacroix@gmail.com",
    description="A comprehensive machine learning environment optimized for AMD GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scooter-lacroix/Stans_MLStack",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "torch>=2.6.0",
        "onnxruntime",
        "psutil",
        "tqdm",
        "requests",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "ml-stack-install=scripts.install_ml_stack_curses:main",
            "ml-stack-verify=scripts.enhanced_verify_installation_wrapper:main",
            "ml-stack-repair=scripts.repair_ml_stack_wrapper:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.sh", "*.py", "assets/*", "docs/*"],
    },
    scripts=[
        "scripts/install_ml_stack_curses.py",
        "scripts/enhanced_verify_installation.sh",
        "scripts/repair_ml_stack.sh",
        "scripts/install_rocm.sh",
        "scripts/install_pytorch_rocm.sh",
        "scripts/install_amdgpu_drivers.sh",
        "scripts/install_aiter.sh",
        "scripts/install_deepspeed.sh",
        "scripts/install_ml_stack.sh",
        "scripts/install_flash_attention_ck.sh",
        "scripts/install_megatron.sh",
        "scripts/install_triton.sh",
        "scripts/install_bitsandbytes.sh",
        "scripts/install_vllm.sh",
        "scripts/install_rocm_smi.sh",
        "scripts/install_pytorch_profiler.sh",
        "scripts/install_wandb.sh",
        "scripts/install_mpi4py.sh",
        "scripts/setup_environment.sh",
        "scripts/enhanced_setup_environment.sh",
        "scripts/verify_installation.sh",
        "scripts/enhanced_verify_installation.sh",
        "scripts/custom_verify_installation.sh",
        "scripts/verify_and_build.sh",
        "scripts/create_persistent_env.sh",
    ],
)
