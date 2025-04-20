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

# Install BITSANDBYTES directly from PyPI
log "Installing BITSANDBYTES from PyPI..."
pip install bitsandbytes --break-system-packages

# Create a directory for the source code (for reference)
if [ ! -d "bitsandbytes" ]; then
    log "Cloning BITSANDBYTES repository for reference..."
    git clone https://github.com/TimDettmers/bitsandbytes.git
    cd bitsandbytes
else
    log "BITSANDBYTES repository already exists, updating..."
    cd bitsandbytes
    git reset --hard HEAD
    git checkout main || git checkout master
    git pull
fi

# Set environment variables for AMD GPUs
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")
export CMAKE_PREFIX_PATH=$ROCM_HOME

# Skip local installation since we already installed from PyPI
log "BITSANDBYTES already installed from PyPI, skipping local installation..."

# Verify installation
log "Verifying BITSANDBYTES installation..."
python3 -c "import bitsandbytes as bnb; print('BITSANDBYTES version:', bnb.__version__); print('CUDA available:', bnb.CUDA_AVAILABLE)"

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
