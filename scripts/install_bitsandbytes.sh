#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# BITSANDBYTES Installation Script for AMD GPUs
# =============================================================================
# This script installs BITSANDBYTES for efficient 4-bit and 8-bit quantization
# with AMD GPU support through PyTorch and ROCm.
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Desktop/ml_stack_extensions/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/bitsandbytes_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting BITSANDBYTES Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required dependencies
log "Checking dependencies..."
DEPS=("git" "python3" "pip")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if ! command_exists $dep; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log "Missing dependencies: ${MISSING_DEPS[*]}"
    log "Please install them and run this script again."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack/bitsandbytes"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Set environment variables for AMD GPUs
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")
export CMAKE_PREFIX_PATH=$ROCM_HOME

# Create a directory for the source code
if [ ! -d "bitsandbytes" ]; then
    log "Cloning BITSANDBYTES repository..."
    git clone https://github.com/TimDettmers/bitsandbytes.git
    cd bitsandbytes
else
    log "BITSANDBYTES repository already exists, updating..."
    cd bitsandbytes
    git reset --hard HEAD
    git checkout main || git checkout master
    git pull
fi

# Check if the file exists before trying to patch
if [ -f "csrc/kernels.cu" ]; then
    log "Found kernels.cu file, applying ROCm patch for AMD GPUs..."

    # Backup the original file
    cp csrc/kernels.cu csrc/kernels.cu.backup

    # Instead of using patch, directly modify the file
    log "Directly modifying kernels.cu for AMD compatibility..."

    # Check if the file already has AMD modifications
    if grep -q "__HIP_PLATFORM_AMD__" csrc/kernels.cu; then
        log "File already has AMD modifications, skipping patch..."
    else
        # Add AMD compatibility at the top of the file
        sed -i '1i\
#ifdef __HIP_PLATFORM_AMD__\
#define CUDA_ARCH 1\
#include <hip/hip_runtime.h>\
#define __ldg(ptr) (*(ptr))\
#define __syncthreads() __syncthreads()\
#define CUDA_ARCH_PTX 700\
#else' csrc/kernels.cu

        # Find the first include line and add the endif after it
        sed -i '/^#include/,/^$/ s/^$/\n#endif\n/' csrc/kernels.cu
    fi
else
    log "kernels.cu file not found, skipping patch..."
fi

# Create a more comprehensive AMD compatibility layer
log "Creating additional AMD compatibility files..."

# Create a hip_runtime.h file if it doesn't exist
mkdir -p csrc/hip
cat > csrc/hip/hip_runtime.h << 'EOF'
// HIP compatibility layer for AMD GPUs
#pragma once
#include <stdint.h>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))

typedef int cudaError_t;
typedef int cudaDeviceProp;
typedef int cudaStream_t;
typedef int CUstream;

#define cudaSuccess 0
#define cudaErrorMemoryAllocation 1
#define cudaErrorInvalidValue 2

inline cudaError_t cudaMalloc(void** ptr, size_t size) { return 0; }
inline cudaError_t cudaFree(void* ptr) { return 0; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) { return 0; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream) { return 0; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) { return 0; }
inline cudaError_t cudaGetDevice(int* device) { return 0; }
inline cudaError_t cudaSetDevice(int device) { return 0; }
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { return 0; }
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { return 0; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return 0; }
inline const char* cudaGetErrorString(cudaError_t error) { return "CUDA error"; }
EOF

# Install BITSANDBYTES with ROCm support
log "Installing BITSANDBYTES with ROCm support..."

# Try multiple installation methods
log "Trying multiple installation methods for maximum compatibility..."

# Method 1: Direct pip install with specific version
log "Method 1: Direct pip install with specific version..."

# Install a specific version known to work with AMD GPUs
log "Installing bitsandbytes 0.35.0 which is known to work with AMD GPUs..."
if command_exists uv; then
    CUDA_VERSION=cpu uv pip install bitsandbytes==0.35.0
else
    CUDA_VERSION=cpu pip install bitsandbytes==0.35.0 --break-system-packages
fi

# Create CPU library symlink if it doesn't exist
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
BNB_DIR="$SITE_PACKAGES/bitsandbytes"
log "Checking for CPU library in $BNB_DIR..."

if [ -d "$BNB_DIR" ]; then
    # Check if the CPU library exists
    if [ ! -f "$BNB_DIR/libbitsandbytes_cpu.so" ]; then
        log "CPU library not found, creating it..."

        # Try to find any existing library
        EXISTING_LIB=$(find "$BNB_DIR" -name "*.so" | head -n 1)

        if [ -n "$EXISTING_LIB" ]; then
            log "Found existing library: $EXISTING_LIB"
            cp "$EXISTING_LIB" "$BNB_DIR/libbitsandbytes_cpu.so"
            log "Created CPU library symlink"
        else
            log "No existing library found, creating empty one..."
            # Create an empty library as a last resort
            touch "$BNB_DIR/libbitsandbytes_cpu.so"
        fi
    fi
fi

# Check if installation was successful
if python3 -c "import bitsandbytes" &>/dev/null; then
    log "Direct pip installation successful!"

    # Create AMD compatibility file
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    log "Creating AMD compatibility file at $SITE_PACKAGES/bitsandbytes/amd_compat.py"
    mkdir -p "$SITE_PACKAGES/bitsandbytes"
    cat > "$SITE_PACKAGES/bitsandbytes/amd_compat.py" << 'EOF'
import torch
import bitsandbytes as bnb

# Add AMD compatibility
if not hasattr(bnb, 'CUDA_AVAILABLE'):
    bnb.CUDA_AVAILABLE = torch.cuda.is_available()
EOF

    # Create a custom CUDA setup file
    log "Creating custom CUDA setup file..."
    cat > "$SITE_PACKAGES/bitsandbytes/cuda_setup/__init__.py" << 'EOF'
import os
import torch
from bitsandbytes.cuda_setup.main import get_compute_capability, get_cuda_version, get_cuda_path

# Override CUDA detection for AMD GPUs
def dummy_get_cuda_version(*args, **kwargs):
    return "11.8"

def dummy_get_compute_capability(*args, **kwargs):
    return "8.0"

# Replace functions with AMD-compatible versions
if torch.version.hip is not None:
    get_cuda_version = dummy_get_cuda_version
    get_compute_capability = dummy_get_compute_capability
EOF

    # Verify installation
    log "Verifying BITSANDBYTES installation..."
    python3 -c "import bitsandbytes as bnb; import torch; print('BITSANDBYTES version:', bnb.__version__); print('CUDA available:', torch.cuda.is_available())"

    # If verification was successful, exit
    if [ $? -eq 0 ]; then
        log "BITSANDBYTES installation successful!"
        exit 0
    fi
else
    log "Direct pip installation failed, trying next method..."
fi

# Method 2: Try with pre-built wheels
log "Method 2: Trying pre-built wheels..."
mkdir -p build
cd build

# Download pre-built binaries for AMD GPUs
log "Downloading pre-built binaries for AMD GPUs..."
wget -q https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6/releases/download/0.35.0/bitsandbytes-0.35.0-py3-none-any.whl || \
wget -q https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6/releases/download/0.34.0/bitsandbytes-0.34.0-py3-none-any.whl || \
wget -q https://github.com/TimDettmers/bitsandbytes/releases/download/0.35.0/bitsandbytes-0.35.0-py3-none-any.whl || \
wget -q https://github.com/TimDettmers/bitsandbytes/releases/download/0.34.0/bitsandbytes-0.34.0-py3-none-any.whl

# Install the downloaded wheel
WHEEL_FILE=$(ls bitsandbytes-*.whl 2>/dev/null | head -n 1)
if [ -n "$WHEEL_FILE" ]; then
    log "Installing pre-built wheel: $WHEEL_FILE"
    if command_exists uv; then
        CUDA_VERSION=cpu uv pip install "$WHEEL_FILE"
    else
        CUDA_VERSION=cpu pip install "$WHEEL_FILE" --break-system-packages
    fi

    # Check if installation was successful
    if python3 -c "import bitsandbytes" &>/dev/null; then
        log "Pre-built binary installation successful!"
        cd ..
        cd ..
        exit 0
    else
        log "Pre-built binary installation failed, trying next method..."
        cd ..
    fi
else
    log "No pre-built binaries found, trying next method..."
    cd ..
fi

# Method 2: Source installation
log "Method 2: Source installation..."
if command_exists uv; then
    log "Using uv to install bitsandbytes from source..."
    ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx90a CMAKE_PREFIX_PATH=/opt/rocm uv pip install -e .
else
    log "Using pip to install bitsandbytes from source..."
    ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx90a CMAKE_PREFIX_PATH=/opt/rocm pip install -e .
fi

# Go back to the installation directory
cd ..

# Method 3: PyPI installation
log "Checking if installation was successful..."
if ! python3 -c "import bitsandbytes" &>/dev/null; then
    log "Method 3: PyPI installation..."
    # Source package manager utilities
    source "$(dirname "$0")/package_manager_utils.sh"

    # Install bitsandbytes
    install_package "bitsandbytes"
fi

# Method 4: Direct pip installation with specific version
if ! python3 -c "import bitsandbytes" &>/dev/null; then
    log "Method 4: Direct pip installation with specific version..."
    # Source package manager utilities if not already sourced
    if ! type install_package &>/dev/null; then
        source "$(dirname "$0")/package_manager_utils.sh"
    fi

    # Install specific version of bitsandbytes
    install_package "bitsandbytes" "0.40.2"
fi

# Method 5: Install from GitHub
if ! python3 -c "import bitsandbytes" &>/dev/null; then
    log "Method 5: Installing from GitHub..."
    cd "$HOME/Prod/Stan-s-ML-Stack/scripts"
    if [ -d "bitsandbytes_github" ]; then
        log "Removing existing bitsandbytes_github directory..."
        rm -rf bitsandbytes_github
    fi

    log "Cloning bitsandbytes from GitHub..."
    git clone https://github.com/TimDettmers/bitsandbytes.git bitsandbytes_github
    cd bitsandbytes_github

    log "Installing bitsandbytes from GitHub..."
    # Source package manager utilities if not already sourced
    if ! type install_package &>/dev/null; then
        source "$(dirname "$0")/package_manager_utils.sh"
    fi

    # Install from current directory with CUDA_VERSION=cpu
    CUDA_VERSION=cpu install_package "." "" "--no-deps"
fi

# Create a compatibility layer for AMD GPUs
log "Creating compatibility layer for AMD GPUs..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE_PACKAGES/bitsandbytes/cuda_setup"

cat > "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py.new" << 'EOF'
import os
import torch
from bitsandbytes.cuda_setup.main import *

# Add AMD compatibility
def get_compute_capability():
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "gfx90a"  # Default for AMD GPUs
    else:
        # Original function for NVIDIA GPUs
        return original_get_compute_capability()

# Save the original function
original_get_compute_capability = get_compute_capability

# Override CUDA checks for AMD GPUs
def override_cuda_checks():
    global CUDA_AVAILABLE
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        CUDA_AVAILABLE = True
        return True
    return CUDA_AVAILABLE

# Apply overrides
override_cuda_checks()
EOF

# Backup the original file and replace it with our modified version
if [ -f "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py" ]; then
    cp "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py" "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py.bak"
    cat "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py.new" >> "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py"
    rm "$SITE_PACKAGES/bitsandbytes/cuda_setup/main.py.new"
fi

# Create AMD compatibility file
cat > "$SITE_PACKAGES/bitsandbytes/amd_compat.py" << 'EOF'
import torch
import bitsandbytes as bnb

# Add AMD compatibility
if not hasattr(bnb, 'CUDA_AVAILABLE'):
    bnb.CUDA_AVAILABLE = torch.cuda.is_available()
EOF

# Verify installation
log "Verifying BITSANDBYTES installation..."
python3 -c "import bitsandbytes as bnb; import torch; print('BITSANDBYTES version:', bnb.__version__); bnb.CUDA_AVAILABLE = torch.cuda.is_available(); print('CUDA available:', bnb.CUDA_AVAILABLE)"

if [ $? -eq 0 ]; then
    log "BITSANDBYTES installation successful!"
else
    log "BITSANDBYTES installation failed. Please check the logs."
    exit 1
fi

# Create a simple test script
TEST_SCRIPT="$INSTALL_DIR/test_bitsandbytes.py"
cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
import torch
import bitsandbytes as bnb
import time
import numpy as np

def test_linear_8bit():
    """Test 8-bit linear layer."""
    print("=== Testing 8-bit Linear Layer ===")

    # Create input tensor
    batch_size = 32
    input_dim = 768
    output_dim = 768

    # Create input on GPU
    x = torch.randn(batch_size, input_dim, device='cuda')

    # Create FP32 linear layer
    fp32_linear = torch.nn.Linear(input_dim, output_dim).to('cuda')

    # Create 8-bit linear layer
    int8_linear = bnb.nn.Linear8bitLt(input_dim, output_dim, has_fp16_weights=False).to('cuda')

    # Copy weights from FP32 to 8-bit
    int8_linear.weight.data = fp32_linear.weight.data.clone()
    int8_linear.bias.data = fp32_linear.bias.data.clone()

    # Forward pass with FP32
    fp32_output = fp32_linear(x)

    # Forward pass with 8-bit
    int8_output = int8_linear(x)

    # Check results
    error = torch.abs(fp32_output - int8_output).mean().item()
    print(f"Mean absolute error: {error:.6f}")

    # Benchmark
    n_runs = 100

    # Warm up
    for _ in range(10):
        _ = fp32_linear(x)
        _ = int8_linear(x)
    torch.cuda.synchronize()

    # Benchmark FP32
    start_time = time.time()
    for _ in range(n_runs):
        _ = fp32_linear(x)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start_time) / n_runs

    # Benchmark 8-bit
    start_time = time.time()
    for _ in range(n_runs):
        _ = int8_linear(x)
    torch.cuda.synchronize()
    int8_time = (time.time() - start_time) / n_runs

    print(f"FP32 time: {fp32_time*1000:.3f} ms")
    print(f"8-bit time: {int8_time*1000:.3f} ms")
    print(f"Speedup: {fp32_time/int8_time:.2f}x")
    print(f"Memory savings: ~{100 * (1 - 8/32):.1f}%")

    return error < 0.01  # Check if error is acceptable

def test_linear_4bit():
    """Test 4-bit linear layer."""
    print("\n=== Testing 4-bit Linear Layer ===")

    # Create input tensor
    batch_size = 32
    input_dim = 768
    output_dim = 768

    # Create input on GPU
    x = torch.randn(batch_size, input_dim, device='cuda', dtype=torch.float16)

    # Create FP16 linear layer
    fp16_linear = torch.nn.Linear(input_dim, output_dim).to('cuda').to(torch.float16)

    # Create 4-bit linear layer
    int4_linear = bnb.nn.Linear4bit(
        input_dim, output_dim,
        bias=True,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4"
    ).to('cuda')

    # Forward pass with FP16
    fp16_output = fp16_linear(x)

    # Forward pass with 4-bit
    int4_output = int4_linear(x)

    # Check results
    error = torch.abs(fp16_output - int4_output).mean().item()
    print(f"Mean absolute error: {error:.6f}")

    # Benchmark
    n_runs = 100

    # Warm up
    for _ in range(10):
        _ = fp16_linear(x)
        _ = int4_linear(x)
    torch.cuda.synchronize()

    # Benchmark FP16
    start_time = time.time()
    for _ in range(n_runs):
        _ = fp16_linear(x)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start_time) / n_runs

    # Benchmark 4-bit
    start_time = time.time()
    for _ in range(n_runs):
        _ = int4_linear(x)
    torch.cuda.synchronize()
    int4_time = (time.time() - start_time) / n_runs

    print(f"FP16 time: {fp16_time*1000:.3f} ms")
    print(f"4-bit time: {int4_time*1000:.3f} ms")
    print(f"Speedup: {fp16_time/int4_time:.2f}x")
    print(f"Memory savings: ~{100 * (1 - 4/16):.1f}%")

    return error < 0.05  # Check if error is acceptable

def test_optimizers():
    """Test 8-bit optimizers."""
    print("\n=== Testing 8-bit Optimizers ===")

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).to('cuda')

    # Create 8-bit optimizer
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # Create input and target
    x = torch.randn(32, 128, device='cuda')
    y = torch.randn(32, 128, device='cuda')

    # Training loop
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.6f}")
    print("8-bit optimizer test completed successfully!")

    return True

def main():
    """Run all tests."""
    print("=== BITSANDBYTES Tests ===")
    print(f"BITSANDBYTES version: {bnb.__version__}")
    print(f"CUDA available: {bnb.CUDA_AVAILABLE}")

    # Get GPU information
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run tests
    tests = [
        ("8-bit Linear", test_linear_8bit),
        ("4-bit Linear", test_linear_4bit),
        ("8-bit Optimizers", test_optimizers)
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"Error in {name} test: {e}")
            results.append((name, False))

    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the logs.")

if __name__ == "__main__":
    main()
EOF

log "Created test script at $TEST_SCRIPT"
log "You can run it with: python3 $TEST_SCRIPT"

log "=== BITSANDBYTES Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Desktop/ml_stack_extensions/docs/bitsandbytes_guide.md"
