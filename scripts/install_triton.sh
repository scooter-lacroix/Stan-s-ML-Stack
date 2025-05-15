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
# Triton Installation Script for AMD GPUs
# =============================================================================
# This script installs OpenAI's Triton compiler for AMD GPUs with ROCm support.
# Triton is an open-source language and compiler for parallel programming that
# can generate highly optimized GPU kernels.
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/triton_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start installation
log "=== Starting Triton Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required dependencies
log "Checking dependencies..."
DEPS=("git" "cmake" "python3" "pip")
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
INSTALL_DIR="$HOME/ml_stack/triton"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Install Python dependencies
log "Installing Python dependencies..."
if command_exists uv; then
    log "Using uv to install dependencies..."
    uv pip install --upgrade cmake ninja pytest packaging wheel
else
    log "Using pip to install dependencies..."
    pip install --upgrade cmake ninja pytest packaging wheel
fi

# Install Triton directly from PyPI first (simpler approach)
log "Installing Triton from PyPI..."
if command_exists uv; then
    uv pip install triton
else
    pip install triton
fi

# Check if the PyPI installation worked
if python3 -c "import triton; print('Triton version:', triton.__version__)" &>/dev/null; then
    log "Triton installed successfully from PyPI!"
else
    log "PyPI installation failed, trying from source..."

    # Clone Triton repository
    if [ ! -d "triton" ]; then
        log "Cloning Triton repository..."
        git clone https://github.com/openai/triton.git --recursive
        cd triton
    else
        log "Triton repository already exists, updating..."
        cd triton
        git fetch --all
        git reset --hard origin/main
        git submodule update --init --recursive
    fi

    # Try different versions if needed
    for version in "main" "master" "dev" "rocm-5.6"; do
        log "Trying branch/tag: $version"
        git checkout $version || continue

        # Apply ROCm patch if needed
        log "Applying ROCm compatibility patches..."
        sed -i 's/cuda/hip/g' python/triton/backends/hip.py || true

        # Build and install Triton
        log "Building and installing Triton from source..."
        cd python
        if command_exists uv; then
            TRITON_BUILD_WITH_ROCM=1 uv pip install -e .
        else
            TRITON_BUILD_WITH_ROCM=1 pip install -e .
        fi

        # Check if installation worked
        if python3 -c "import triton; print('Triton version:', triton.__version__)" &>/dev/null; then
            log "Triton installed successfully from source!"
            break
        else
            log "Installation of branch $version failed, trying next version..."
            cd ..
        fi
    done
fi

# Verify installation
log "Verifying Triton installation..."
python3 -c "import triton; print('Triton version:', triton.__version__)"

if [ $? -eq 0 ]; then
    log "Triton installation successful!"
else
    log "Triton installation failed. Please check the logs."
    exit 1
fi

# Create a simple test script
TEST_SCRIPT="$INSTALL_DIR/test_triton.py"
cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start
    block_start = pid * BLOCK_SIZE
    # Offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle case where the block size doesn't divide the number of elements
    mask = offsets < n_elements
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Add x and y
    output = x + y
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x, y):
    # Check input dimensions
    assert x.shape == y.shape, "Input shapes must match"
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"

    # Output tensor
    output = torch.empty_like(x)

    # Get tensor dimensions
    n_elements = output.numel()

    # Define block size
    BLOCK_SIZE = 1024

    # Calculate grid dimensions
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    add_kernel[grid, BLOCK_SIZE](
        x, y, output, n_elements, BLOCK_SIZE
    )

    return output

# Test the kernel
def test_add_vectors():
    # Create input tensors on GPU
    x = torch.rand(1024, 1024, device='cuda')
    y = torch.rand(1024, 1024, device='cuda')

    # Compute with Triton
    output_triton = add_vectors(x, y)

    # Compute with PyTorch
    output_torch = x + y

    # Check results
    assert torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)
    print("Test passed!")

    # Benchmark
    import time

    # Warm up
    for _ in range(10):
        _ = add_vectors(x, y)
    torch.cuda.synchronize()

    # Benchmark Triton
    n_runs = 100
    start_time = time.time()
    for _ in range(n_runs):
        _ = add_vectors(x, y)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / n_runs

    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(n_runs):
        _ = x + y
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / n_runs

    print(f"Triton time: {triton_time*1000:.3f} ms")
    print(f"PyTorch time: {torch_time*1000:.3f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

if __name__ == "__main__":
    test_add_vectors()
EOF

log "Created test script at $TEST_SCRIPT"
log "You can run it with: python3 $TEST_SCRIPT"

log "=== Triton Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Prod/Stan-s-ML-Stack/docs/extensions/triton_guide.md"
