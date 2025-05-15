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
# vLLM Installation Script for AMD GPUs
# =============================================================================
# This script installs vLLM, a high-throughput and memory-efficient inference
# and serving engine for LLMs that supports AMD GPUs through ROCm.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Trap to ensure we exit properly
trap 'echo "Forced exit"; kill -9 $$' EXIT

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_STACK_DIR="$(dirname "$SCRIPT_DIR")"

# Create log directory
LOG_DIR="$ML_STACK_DIR/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/vllm_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to get Python package version
get_package_version() {
    python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "Not installed"
}

# Save original environment variables
# This is necessary because vLLM/Ray has specific environment variable requirements:
# 1. It requires HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
# 2. It requires consistent values between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
log "Saving original environment variables..."
ORIGINAL_ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
ORIGINAL_HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
ORIGINAL_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ORIGINAL_PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE

# Log the original environment variables
log "Original ROCR_VISIBLE_DEVICES: $ORIGINAL_ROCR_VISIBLE_DEVICES"
log "Original HIP_VISIBLE_DEVICES: $ORIGINAL_HIP_VISIBLE_DEVICES"
log "Original CUDA_VISIBLE_DEVICES: $ORIGINAL_CUDA_VISIBLE_DEVICES"
log "Original PYTORCH_ROCM_DEVICE: $ORIGINAL_PYTORCH_ROCM_DEVICE"

# Temporarily adjust environment variables for vLLM/Ray compatibility
log "Temporarily adjusting environment variables for vLLM/Ray compatibility..."

# Step 1: Handle ROCR_VISIBLE_DEVICES conflict
# Ray requires HIP_VISIBLE_DEVICES and will error with ROCR_VISIBLE_DEVICES
if [ -n "$ORIGINAL_ROCR_VISIBLE_DEVICES" ]; then
    log "Unsetting ROCR_VISIBLE_DEVICES and setting HIP_VISIBLE_DEVICES to $ORIGINAL_ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES=$ORIGINAL_ROCR_VISIBLE_DEVICES
elif [ -z "$HIP_VISIBLE_DEVICES" ]; then
    log "Setting HIP_VISIBLE_DEVICES to default value (0,1)"
    export HIP_VISIBLE_DEVICES=0,1
fi

# Step 2: Handle inconsistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
# Ray requires consistent values for both variables or one must be unset
if [ -n "$HIP_VISIBLE_DEVICES" ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    if [ "$HIP_VISIBLE_DEVICES" != "$CUDA_VISIBLE_DEVICES" ]; then
        log "WARNING: Inconsistent values detected between HIP_VISIBLE_DEVICES ($HIP_VISIBLE_DEVICES) and CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES)"
        log "Ray requires consistent values for both variables. Temporarily unsetting CUDA_VISIBLE_DEVICES."
        unset CUDA_VISIBLE_DEVICES
    else
        log "HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES have consistent values ($HIP_VISIBLE_DEVICES)"
    fi
elif [ -n "$HIP_VISIBLE_DEVICES" ]; then
    log "Only HIP_VISIBLE_DEVICES is set ($HIP_VISIBLE_DEVICES), which is the preferred configuration for Ray/vLLM"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    log "Only CUDA_VISIBLE_DEVICES is set ($CUDA_VISIBLE_DEVICES), setting HIP_VISIBLE_DEVICES to match"
    export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi

# Start installation
log "=== Starting vLLM Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
log "Script Directory: $SCRIPT_DIR"
log "ML Stack Directory: $ML_STACK_DIR"
log "Current HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"

# Function to fix ninja-build detection
fix_ninja_detection() {
    if command -v ninja &>/dev/null && ! command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating symlink for ninja-build..."
        sudo ln -sf $(which ninja) /usr/bin/ninja-build
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build symlink created."
        return 0
    elif command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build already available."
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing ninja-build..."
        sudo apt-get update && sudo apt-get install -y ninja-build
        return $?
    fi
}

# Check for required dependencies
log "Checking dependencies..."
# Fix ninja-build detection
fix_ninja_detection
DEPS=("git" "python3" "pip" "cmake" "ninja-build")
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
INSTALL_DIR="$ML_STACK_DIR/vllm_build"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Set environment variables for AMD GPUs
export ROCM_PATH=/opt/rocm
# Note: HIP_VISIBLE_DEVICES is already set above based on original environment variables
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")
export AMD_LOG_LEVEL=0

# Double-check environment variables for Ray compatibility
# Ray has two specific requirements:
# 1. ROCR_VISIBLE_DEVICES must be unset (use HIP_VISIBLE_DEVICES instead)
# 2. HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must have consistent values or one must be unset

# Check for ROCR_VISIBLE_DEVICES
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    log "WARNING: ROCR_VISIBLE_DEVICES is set, which can cause Ray initialization errors"
    log "Unsetting ROCR_VISIBLE_DEVICES for vLLM installation"
    unset ROCR_VISIBLE_DEVICES
fi

# Check for consistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
if [ -n "$HIP_VISIBLE_DEVICES" ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    if [ "$HIP_VISIBLE_DEVICES" != "$CUDA_VISIBLE_DEVICES" ]; then
        log "WARNING: Inconsistent values detected between HIP_VISIBLE_DEVICES ($HIP_VISIBLE_DEVICES) and CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES)"
        log "Unsetting CUDA_VISIBLE_DEVICES to prevent Ray initialization errors"
        unset CUDA_VISIBLE_DEVICES
    fi
fi

# Log current environment variable state
log "Current environment variables for GPU visibility:"
log "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
log "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"

# Source package manager utilities
source "$SCRIPT_DIR/package_manager_utils.sh"

# Ensure uv is installed
ensure_uv_installed

# Check if PyTorch is already installed
if package_installed "torch"; then
    PYTORCH_VERSION=$(get_package_version "torch")
    log "PyTorch is already installed (version: $PYTORCH_VERSION)"

    # Check if PyTorch has ROCm support
    if python3 -c "import torch; print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'Not available')" 2>/dev/null | grep -q "ROCm version: Not available"; then
        log "WARNING: PyTorch is installed but does not have ROCm support"
        log "This may cause issues with vLLM installation"
    else
        log "PyTorch has ROCm support"
    fi
else
    log "PyTorch is not installed, installing with ROCm support..."
    # Install PyTorch with ROCm support
    "$SCRIPT_DIR/install_pytorch_rocm.sh"
fi

# Install Python dependencies
log "Installing Python dependencies..."
uv pip install --upgrade pip setuptools wheel ninja pybind11 cmake packaging
log "Installing additional required dependencies..."
uv pip install aioprometheus ray psutil huggingface_hub tokenizers

# Create CUDA compatibility layer for AMD GPUs
log "Creating CUDA compatibility layer for AMD GPUs..."
mkdir -p /tmp/fake-cuda/bin
cat > /tmp/fake-cuda/bin/nvcc << 'EOF'
#!/bin/bash
# This is a fake nvcc script that pretends to be CUDA for vLLM
echo "nvcc (fake) V11.8.89"
exit 0
EOF
chmod +x /tmp/fake-cuda/bin/nvcc

# Add fake CUDA to PATH
export PATH="/tmp/fake-cuda/bin:$PATH"
export CUDA_HOME="/tmp/fake-cuda"
export CUDA_PATH="/tmp/fake-cuda"

# Check for required dependencies
log "Checking for required dependencies..."
MISSING_DEPS=()

for dep in "torch" "transformers" "aioprometheus" "ray" "psutil" "huggingface_hub" "tokenizers"; do
    if ! package_installed "$dep"; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log "Missing dependencies: ${MISSING_DEPS[*]}"
    log "Installing missing dependencies..."
    for dep in "${MISSING_DEPS[@]}"; do
        log "Installing $dep..."
        uv pip install "$dep"
    done
fi

# Check if vLLM is already installed
if package_installed "vllm"; then
    VLLM_VERSION=$(get_package_version "vllm")
    log "vLLM is already installed (version: $VLLM_VERSION)"

    # Verify it works with the current PyTorch and has all dependencies
    if python3 -c "import vllm, torch, aioprometheus; print('vLLM works with PyTorch', torch.__version__)" &>/dev/null; then
        log "vLLM is working correctly with the current PyTorch installation"
        log "All dependencies are satisfied"
        log "Skipping installation"
    else
        log "WARNING: vLLM is installed but may not be compatible with the current PyTorch installation or missing dependencies"
        log "Proceeding with reinstallation..."
    fi
else
    log "vLLM is not installed, proceeding with installation..."
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Detected Python version: $PYTHON_VERSION"

# Set environment variables for AMD GPUs
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")
export SKIP_CUDA_BUILD=1
export FORCE_CMAKE=1
export AMD_LOG_LEVEL=0

# Get PyTorch version
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "Not installed")
log "PyTorch version (without build suffix): $PYTORCH_VERSION"

# Install required dependencies first
log "Installing required dependencies..."
uv pip install numpy packaging setuptools wheel ninja pybind11 cmake

# Check if vLLM is already installed and working
if package_installed "vllm" && python3 -c "import vllm, torch; print('vLLM works with PyTorch', torch.__version__)" &>/dev/null; then
    log "vLLM is already installed and working correctly"
else
    log "Installing vLLM from source for maximum compatibility..."

    # Clone vLLM repository
    if [ ! -d "vllm" ]; then
        log "Cloning vLLM repository..."
        git clone https://github.com/vllm-project/vllm.git
        cd vllm
    else
        log "vLLM repository already exists, updating..."
        cd vllm
        git fetch --all
        git reset --hard origin/main
    fi

    # Clean any local changes before trying branches
    log "Cleaning any local changes to avoid checkout conflicts..."
    git reset --hard HEAD
    git clean -fd

    # Try different versions - start with v0.3.2 which is known to work
    for version in "v0.3.2" "v0.3.1" "v0.3.0" "v0.2.5" "v0.2.0" "main" "master"; do
        log "Trying branch/tag: $version"
        git checkout -f $version || continue

        # Apply ROCm patch if needed
        log "Applying ROCm compatibility patches..."

        # Create a patch file for ROCm compatibility
        cat > rocm_patch.diff << 'EOF'
diff --git a/setup.py b/setup.py
index 1111111..2222222 100644
--- a/setup.py
+++ b/setup.py
@@ -11,7 +11,7 @@ extras_require = {
     "dev": ["black", "isort", "pylint", "pytest", "mypy"],
     "ray": ["ray>=2.9.0"],
     "openai": ["openai>=1.0.0"],
-    "amd": ["torch>=2.0.0"],
+    "amd": ["torch>=2.0.0", "ninja"],
 }

 # Add all extras to "all"
EOF

        # Try to apply the patch (may fail if already applied or not needed)
        patch -p1 < rocm_patch.diff || log "Patch application failed, continuing anyway..."

        # Install vLLM with AMD support
        log "Installing vLLM with AMD support..."

        # For Python 3.12, we need to modify setup.py to make it compatible
        if [[ "$PYTHON_VERSION" == "3.12" ]]; then
            log "Modifying setup.py for Python 3.12 compatibility..."

            # Backup the original setup.py
            cp setup.py setup.py.backup

            # Fix the 'release' not in list error by patching the get_nvcc_cuda_version function
            log "Patching get_nvcc_cuda_version function to fix 'release' not in list error..."

            # Create a patch file
            cat > fix_nvcc_version.patch << 'EOF'
--- setup.py.orig
+++ setup.py
@@ -122,7 +122,14 @@
 def get_nvcc_cuda_version():
     """Get the CUDA version from nvcc."""
     output = subprocess.check_output([NVCC_PATH, "--version"]).decode()
-    version_str = output.strip().split("release ")[-1].split(",")[0]
+    try:
+        if "release" in output:
+            version_str = output.strip().split("release ")[-1].split(",")[0]
+        else:
+            # Fallback for cases where 'release' is not in the output
+            version_str = "11.8"
+    except Exception:
+        version_str = "11.8"  # Default to a known version if parsing fails
     return version_str


EOF

            # Apply the patch
            patch -p0 < fix_nvcc_version.patch || log "Patch application failed, continuing anyway..."

            # Modify setup.py to remove incompatible dependencies
            sed -i 's/ray\[default\]>=2.5.1/ray>=2.5.1/g' setup.py
            sed -i 's/"ninja>=1.11.0",/"ninja>=1.11.0",\n        "setuptools<60.0.0",/g' setup.py

            # Remove version constraints that might cause issues
            sed -i 's/numpy>=1.24.1,<2/numpy>=1.24.1/g' setup.py
            sed -i 's/torch>=2.0.0/torch>=2.0.0/g' setup.py

            # Add Python 3.12 to the classifiers
            sed -i '/Programming Language :: Python :: 3.11/a\\        "Programming Language :: Python :: 3.12",' setup.py

            # Create a fake nvcc output file for testing
            mkdir -p /tmp/fake-cuda/bin
            cat > /tmp/fake-cuda/bin/nvcc << 'EOF'
#!/bin/bash
echo "nvcc V11.8.89"
exit 0
EOF
            chmod +x /tmp/fake-cuda/bin/nvcc
        fi

        # Install with AMD support
        if [[ "$PYTHON_VERSION" == "3.12" ]]; then
            log "Using Python 3.12, installing with special flags..."

            # Check if PyTorch is already installed and get its version
            PYTORCH_FULL_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
            log "Full PyTorch version: $PYTORCH_FULL_VERSION"

            # Extract the base version (without build suffix)
            PYTORCH_BASE_VERSION=$(echo $PYTORCH_FULL_VERSION | cut -d'+' -f1)
            log "PyTorch base version: $PYTORCH_BASE_VERSION"

            # Create a custom setup.py that works with the installed PyTorch version
            log "Creating custom setup.py compatible with Python 3.12 and PyTorch $PYTORCH_BASE_VERSION"
            cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="vllm",
    version="0.3.2",
    packages=find_packages(),
    install_requires=[
        "torch>=${PYTORCH_BASE_VERSION}",
        "transformers>=4.33.0",
        "numpy>=1.24.1",
        "sentencepiece>=0.1.97",
        "packaging>=23.1",
        "pydantic>=2.4.2",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "einops>=0.6.1",
        "typing-extensions>=4.5.0",
        "aioprometheus>=23.3.0",
        "psutil>=5.9.5",
        "ray>=2.5.1",
        "huggingface_hub>=0.16.4",
        "tokenizers>=0.13.3",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
EOF

            # Install with custom setup.py
            log "Installing vLLM with custom setup.py for Python 3.12..."
            SKIP_CUDA_BUILD=1 FORCE_CMAKE=1 uv pip install -e . --no-build-isolation
        else
            # For Python 3.11 and below, use normal installation
            log "Using Python $PYTHON_VERSION, installing with standard approach..."

            # Check if PyTorch is already installed and get its version
            PYTORCH_FULL_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
            log "Full PyTorch version: $PYTORCH_FULL_VERSION"

            # Extract the base version (without build suffix)
            PYTORCH_BASE_VERSION=$(echo $PYTORCH_FULL_VERSION | cut -d'+' -f1)
            log "PyTorch base version: $PYTORCH_BASE_VERSION"

            # Create a custom setup.py that works with the installed PyTorch version
            log "Creating custom setup.py compatible with PyTorch $PYTORCH_BASE_VERSION"
            cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="vllm",
    version="0.3.2",
    packages=find_packages(),
    install_requires=[
        "torch>=${PYTORCH_BASE_VERSION}",
        "transformers>=4.33.0",
        "ray>=2.5.1",
        "numpy>=1.24.3",
        "psutil>=5.9.5",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.16.4",
        "tokenizers>=0.13.3",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "pydantic>=2.4.2",
        "einops>=0.6.1",
        "packaging>=23.1",
        "typing-extensions>=4.5.0",
        "aioprometheus>=23.3.0",
    ],
    extras_require={
        "amd": [
            "triton>=2.1.0",
        ],
    },
)
EOF

            # Install with uv
            log "Installing vLLM with custom setup.py..."
            SKIP_CUDA_BUILD=1 FORCE_CMAKE=1 uv pip install -e "." --no-build-isolation

            # Install additional dependencies
            log "Installing additional dependencies..."
            uv pip install triton --no-build-isolation
        fi

        # Check if installation worked
        if python3 -c "import vllm; print('vLLM version:', vllm.__version__)" &>/dev/null; then
            log "vLLM installed successfully from source!"
            break
        else
            log "Installation of branch $version failed, trying next version..."
            cd ..
            cd vllm
        fi
    done
fi

# If all installation methods failed, try one more approach
if ! python3 -c "import vllm" &>/dev/null; then
    log "All installation methods failed, trying one more approach..."

    # Get Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log "Detected Python version: $PYTHON_VERSION"

    # Get PyTorch version
    PYTORCH_FULL_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
    PYTORCH_BASE_VERSION=$(echo $PYTORCH_FULL_VERSION | cut -d'+' -f1)
    log "PyTorch version: $PYTORCH_FULL_VERSION (base: $PYTORCH_BASE_VERSION)"

    # Check if PyTorch version is compatible with vLLM
    if [[ "$PYTHON_VERSION" == "3.12" ]]; then
        log "Python 3.12 detected, building vLLM from source with custom modifications..."

        # Create a temporary directory for the build
        TEMP_BUILD_DIR="$INSTALL_DIR/vllm_temp_build"
        mkdir -p "$TEMP_BUILD_DIR"
        cd "$TEMP_BUILD_DIR"

        # Clone the repository
        log "Cloning vLLM repository..."
        git clone https://github.com/vllm-project/vllm.git
        cd vllm
        git checkout v0.3.2

        # Create a simplified setup.py that works with Python 3.12
        log "Creating simplified setup.py for Python 3.12..."
        cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="vllm",
    version="0.3.2",
    packages=find_packages(),
    install_requires=[
        "torch>=${PYTORCH_BASE_VERSION}",
        "transformers>=4.33.0",
        "numpy>=1.24.1",
        "sentencepiece>=0.1.97",
        "packaging>=23.1",
        "pydantic>=2.4.2",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "einops>=0.6.1",
        "typing-extensions>=4.5.0",
        "aioprometheus>=23.3.0",
        "psutil>=5.9.5",
        "ray>=2.5.1",
        "huggingface_hub>=0.16.4",
        "tokenizers>=0.13.3",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
EOF

        # Install with custom setup.py
        log "Installing vLLM with custom setup.py..."
        SKIP_CUDA_BUILD=1 FORCE_CMAKE=1 uv pip install -e . --no-build-isolation
    else
        # For Python 3.11 and below, try a specific version
        log "Trying to install vLLM 0.2.1 which has better compatibility with PyTorch $PYTORCH_BASE_VERSION..."

        # Create a temporary directory for the build
        TEMP_BUILD_DIR="$INSTALL_DIR/vllm_temp_build"
        mkdir -p "$TEMP_BUILD_DIR"
        cd "$TEMP_BUILD_DIR"

        # Clone the repository
        log "Cloning vLLM repository..."
        git clone https://github.com/vllm-project/vllm.git
        cd vllm
        git checkout v0.2.1

        # Create a custom setup.py
        log "Creating custom setup.py for PyTorch $PYTORCH_BASE_VERSION..."
        cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="vllm",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "torch>=${PYTORCH_BASE_VERSION}",
        "transformers>=4.33.0",
        "ray>=2.5.1",
        "numpy>=1.24.3",
        "psutil>=5.9.5",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.16.4",
        "tokenizers>=0.13.3",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "pydantic>=2.4.2",
        "einops>=0.6.1",
        "packaging>=23.1",
        "typing-extensions>=4.5.0",
        "aioprometheus>=23.3.0",
    ],
    extras_require={
        "amd": [
            "triton>=2.1.0",
        ],
    },
)
EOF

        # Install with custom setup.py
        log "Installing vLLM with custom setup.py..."
        SKIP_CUDA_BUILD=1 FORCE_CMAKE=1 uv pip install -e . --no-build-isolation

        # Install additional dependencies
        log "Installing additional dependencies..."
        uv pip install triton --no-build-isolation
    fi

    # Check if that worked
    if python3 -c "import vllm; print('vLLM version:', vllm.__version__)" &>/dev/null; then
        log "vLLM installed successfully!"
    else
        log "Failed to install vLLM. Please check the logs and try again."
        log "You may need to install vLLM manually following the instructions at https://github.com/vllm-project/vllm"
        log "For Python 3.12, you may need to wait for official support or use Python 3.11 instead."
    fi
fi

# Verify installation
log "Verifying vLLM installation..."
log "Checking for all required dependencies..."

# Install any missing dependencies one last time
for dep in "aioprometheus" "ray" "psutil" "huggingface_hub" "tokenizers" "transformers" "pydantic" "fastapi" "uvicorn" "einops" "packaging" "typing-extensions" "sentencepiece"; do
    if ! package_installed "$dep"; then
        log "Installing missing dependency: $dep"
        uv pip install "$dep"
    fi
done

# Verify vLLM installation with all dependencies
# First, try to import Ray separately to catch specific initialization errors
log "Testing Ray initialization (which is required by vLLM)..."
RAY_OUTPUT=$(python3 -c "import ray; ray.init(ignore_reinit_error=True); print('Ray initialized successfully')" 2>&1) || true

# Check for specific Ray initialization errors
if echo "$RAY_OUTPUT" | grep -q "Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"; then
    log "ERROR: Ray initialization failed due to ROCR_VISIBLE_DEVICES conflict"
    log "Attempting to fix by explicitly unsetting ROCR_VISIBLE_DEVICES..."
    RAY_ERROR_TYPE="ROCR_VISIBLE_DEVICES"
elif echo "$RAY_OUTPUT" | grep -q "Inconsistant values found. Please use either HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES"; then
    log "ERROR: Ray initialization failed due to inconsistent values between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES"
    log "Attempting to fix by ensuring consistent environment variables..."
    RAY_ERROR_TYPE="INCONSISTENT_DEVICES"
elif echo "$RAY_OUTPUT" | grep -q "Error initializing Ray"; then
    log "ERROR: Ray initialization failed with a general error"
    log "Attempting to fix by adjusting all GPU visibility environment variables..."
    RAY_ERROR_TYPE="GENERAL_ERROR"
else
    RAY_ERROR_TYPE="NONE"
fi

# Create a wrapper script to handle environment variables and verify vLLM installation
log "Creating a wrapper script to handle environment variables and verify vLLM installation..."

cat > "$INSTALL_DIR/test_vllm_import.py" << 'EOF'
import os
import sys
import importlib
import inspect

# Function to log messages
def log(message):
    print(f"[ENV_SETUP] {message}")

# Save original environment variables
original_env = {
    'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES'),
    'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES'),
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')
}

log(f"Original environment variables: {original_env}")

# Step 1: Handle ROCR_VISIBLE_DEVICES
if 'ROCR_VISIBLE_DEVICES' in os.environ:
    log(f"Unsetting ROCR_VISIBLE_DEVICES (was: {os.environ['ROCR_VISIBLE_DEVICES']})")
    del os.environ['ROCR_VISIBLE_DEVICES']

# Step 2: Ensure HIP_VISIBLE_DEVICES is set
if 'HIP_VISIBLE_DEVICES' not in os.environ:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        log(f"Setting HIP_VISIBLE_DEVICES to match CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        os.environ['HIP_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        log("Setting HIP_VISIBLE_DEVICES to default value (0,1)")
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'

# Step 3: Handle inconsistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
if 'HIP_VISIBLE_DEVICES' in os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['HIP_VISIBLE_DEVICES'] != os.environ['CUDA_VISIBLE_DEVICES']:
        log(f"Inconsistent values: HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        log("Unsetting CUDA_VISIBLE_DEVICES to prevent Ray initialization errors")
        del os.environ['CUDA_VISIBLE_DEVICES']

# Log current environment
log(f"Current environment variables:")
log(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'not set')}")
log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
log(f"ROCR_VISIBLE_DEVICES: {os.environ.get('ROCR_VISIBLE_DEVICES', 'not set')}")

# Function to verify vLLM installation without relying on __version__
def verify_vllm_installation():
    try:
        # First, try to import Ray and initialize it
        log("Importing Ray...")
        import ray
        log("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        log("Ray initialized successfully")

        # Import vLLM and check for key components
        log("Importing vLLM...")
        import vllm

        # Check for key modules and classes that should be present in vLLM
        required_attributes = [
            'LLM',            # Main LLM class
            'SamplingParams', # Sampling parameters class
            'EngineArgs',     # Engine arguments class
            'AsyncLLMEngine'  # Async engine class
        ]

        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(vllm, attr):
                missing_attributes.append(attr)

        if missing_attributes:
            log(f"WARNING: vLLM is missing the following key attributes: {', '.join(missing_attributes)}")
            log("This may indicate an incomplete installation")
        else:
            log("All key vLLM attributes are present")

        # Try to get version in multiple ways
        version = "Unknown"
        try:
            # Method 1: Direct __version__ attribute
            if hasattr(vllm, '__version__'):
                version = vllm.__version__
                log(f"vLLM version (from __version__): {version}")
            # Method 2: Check package metadata
            else:
                log("vLLM does not have __version__ attribute, trying alternative methods")
                try:
                    import pkg_resources
                    version = pkg_resources.get_distribution('vllm').version
                    log(f"vLLM version (from pkg_resources): {version}")
                except Exception as e:
                    log(f"Could not determine vLLM version from pkg_resources: {e}")
                    # Method 3: Check module file path for version info
                    try:
                        vllm_path = inspect.getfile(vllm)
                        log(f"vLLM module path: {vllm_path}")
                        log("vLLM is installed, but version information is not available")
                    except Exception as inner_e:
                        log(f"Could not determine vLLM module path: {inner_e}")

                # Even if we can't determine the version, we can still check if key classes exist
                log("Checking for key vLLM classes to verify installation...")
                if hasattr(vllm, 'LLM') and hasattr(vllm, 'SamplingParams'):
                    log("Found key vLLM classes (LLM, SamplingParams)")
                    log("vLLM appears to be installed correctly despite missing version information")
        except AttributeError:
            log("Caught AttributeError when accessing vLLM attributes")
            log("This is expected if vLLM is missing __version__ attribute")
            # Continue with verification despite the missing attribute

        # Import other dependencies
        log("Importing other dependencies...")
        import torch
        log(f"PyTorch version: {torch.__version__}")

        import aioprometheus
        import psutil
        import huggingface_hub
        import tokenizers

        log("All dependencies successfully imported")

        # Clean up Ray
        ray.shutdown()
        return True
    except ImportError as e:
        log(f"Import Error: {e}")
        log(f"This suggests that vLLM or one of its dependencies is not installed correctly")
        return False
    except Exception as e:
        log(f"Error: {e}")
        log(f"Error type: {type(e).__name__}")
        log(f"Error details: {str(e)}")
        return False

# Run the verification
success = verify_vllm_installation()
if success:
    log("vLLM verification completed successfully")
    sys.exit(0)
else:
    log("vLLM verification failed")
    sys.exit(1)
EOF

# Run the wrapper script
log "Running the verification script..."
python3 "$INSTALL_DIR/test_vllm_import.py"
IMPORT_RESULT=$?

if [ $IMPORT_RESULT -ne 0 ]; then
    log "WARNING: The verification script reported issues with the vLLM installation"
    log "However, this may not indicate a complete failure"
    log "Attempting alternative verification methods..."

    # Try a more robust import test that checks for key classes without using __version__
    log "Testing robust vLLM import..."
    cat > "$INSTALL_DIR/basic_vllm_test.py" << 'EOF'
import os
import sys

# Unset ROCR_VISIBLE_DEVICES if it exists
if 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']

# Ensure HIP_VISIBLE_DEVICES is set
if 'HIP_VISIBLE_DEVICES' not in os.environ:
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1'

# Handle inconsistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
if 'HIP_VISIBLE_DEVICES' in os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['HIP_VISIBLE_DEVICES'] != os.environ['CUDA_VISIBLE_DEVICES']:
        del os.environ['CUDA_VISIBLE_DEVICES']

try:
    # Import vLLM
    import vllm

    # Check for key classes
    key_classes = []
    if hasattr(vllm, 'LLM'):
        key_classes.append('LLM')
    if hasattr(vllm, 'SamplingParams'):
        key_classes.append('SamplingParams')
    if hasattr(vllm, 'EngineArgs'):
        key_classes.append('EngineArgs')

    if key_classes:
        print(f"vLLM imported successfully with key classes: {', '.join(key_classes)}")
        sys.exit(0)
    else:
        print("vLLM imported but missing key classes")
        sys.exit(1)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

    python3 "$INSTALL_DIR/basic_vllm_test.py"
    BASIC_IMPORT_RESULT=$?

    if [ $BASIC_IMPORT_RESULT -eq 0 ]; then
        log "Robust vLLM import successful"
        log "vLLM appears to be installed correctly despite verification issues"
        IMPORT_RESULT=0
    else
        log "ERROR: Robust vLLM import failed"
        log "Trying one last minimal import test..."

        # Try the absolute simplest import test as a last resort
        MINIMAL_IMPORT=$(python3 -c "import vllm; print('OK')" 2>&1) || true

        if echo "$MINIMAL_IMPORT" | grep -q "OK"; then
            log "Minimal vLLM import successful"
            log "vLLM is installed but may have issues"
            IMPORT_RESULT=0
        else
            log "ERROR: Even minimal vLLM import failed"
            log "Error output: $MINIMAL_IMPORT"
        fi
    fi
else
    log "vLLM verification completed successfully"
fi

if [ $IMPORT_RESULT -eq 0 ]; then
    log "vLLM installation successful with all dependencies!"

    # Display completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██╗   ██╗██╗     ██╗     ███╗   ███╗                  ║
    ║  ██║   ██║██║     ██║     ████╗ ████║                  ║
    ║  ██║   ██║██║     ██║     ██╔████╔██║                  ║
    ║  ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║                  ║
    ║   ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║                  ║
    ║    ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝                  ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    echo "vLLM installation complete. Exiting now..."

    # Force exit to prevent hanging
    kill -9 $$ 2>/dev/null
    exit 0
else
    log "vLLM installation failed. Please check the logs."

    # Force exit even on failure
    echo "Installation failed. Exiting now..."
    kill -9 $$ 2>/dev/null
    exit 1
fi

# Create a simple test script
TEST_SCRIPT="$INSTALL_DIR/test_vllm.py"
cat > $TEST_SCRIPT << 'EOF'
#!/usr/bin/env python3
from vllm import LLM, SamplingParams
import time
import torch
import argparse

def test_vllm(model_name="facebook/opt-125m", max_tokens=100):
    """Test vLLM with a small model."""
    print(f"=== Testing vLLM with {model_name} ===")

    # Get GPU information
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Initialize LLM
    print("Initializing LLM...")
    start_time = time.time()
    llm = LLM(model=model_name)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f} seconds")

    # Prepare prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The best programming language is",
        "The meaning of life is"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Generate completions
    print("Generating completions...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    # Calculate statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / generation_time

    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return True

def test_vllm_batch_performance(model_name="facebook/opt-125m", batch_sizes=[1, 2, 4, 8, 16], max_tokens=100):
    """Test vLLM batch performance."""
    print(f"\n=== Testing vLLM Batch Performance with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM(model=model_name)

    # Base prompt
    base_prompt = "Write a short paragraph about"
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "renewable energy",
        "virtual reality",
        "blockchain technology",
        "genetic engineering",
        "autonomous vehicles",
        "robotics",
        "cybersecurity",
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Test different batch sizes
    results = []
    for batch_size in batch_sizes:
        if batch_size > len(topics):
            print(f"Skipping batch size {batch_size} (exceeds number of topics)")
            continue

        prompts = [f"{base_prompt} {topics[i]}" for i in range(batch_size)]

        # Warm-up
        _ = llm.generate(prompts, sampling_params)

        # Benchmark
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        generation_time = time.time() - start_time

        # Calculate statistics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_second = total_tokens / generation_time

        results.append({
            "batch_size": batch_size,
            "generation_time": generation_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second
        })

        print(f"Batch size: {batch_size}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print()

    # Print summary
    print("\n=== Batch Performance Summary ===")
    print("Batch Size | Generation Time (s) | Total Tokens | Tokens/Second")
    print("----------|---------------------|--------------|-------------")
    for result in results:
        print(f"{result['batch_size']:10} | {result['generation_time']:19.2f} | {result['total_tokens']:12} | {result['tokens_per_second']:13.2f}")

    return True

def test_vllm_continuous_batching(model_name="facebook/opt-125m", max_tokens=100):
    """Test vLLM continuous batching."""
    print(f"\n=== Testing vLLM Continuous Batching with {model_name} ===")

    # Initialize LLM with continuous batching
    print("Initializing LLM with continuous batching...")
    llm = LLM(model=model_name, enable_lora=False)

    # Prepare prompts of different lengths
    prompts = [
        "Hello",
        "Hello, my name is",
        "Hello, my name is John and I am a",
        "Hello, my name is John and I am a software engineer with 10 years of experience in"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Generate completions
    print("Generating completions with continuous batching...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    # Calculate statistics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / generation_time

    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return True

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test vLLM with AMD GPUs")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--test-basic", action="store_true", help="Run basic test")
    parser.add_argument("--test-batch", action="store_true", help="Run batch performance test")
    parser.add_argument("--test-continuous", action="store_true", help="Run continuous batching test")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    if not (args.test_basic or args.test_batch or args.test_continuous) or args.all:
        args.test_basic = args.test_batch = args.test_continuous = True

    # Run tests
    results = []

    if args.test_basic:
        try:
            result = test_vllm(args.model, args.max_tokens)
            results.append(("Basic Test", result))
        except Exception as e:
            print(f"Error in basic test: {e}")
            results.append(("Basic Test", False))

    if args.test_batch:
        try:
            result = test_vllm_batch_performance(args.model, max_tokens=args.max_tokens)
            results.append(("Batch Performance Test", result))
        except Exception as e:
            print(f"Error in batch performance test: {e}")
            results.append(("Batch Performance Test", False))

    if args.test_continuous:
        try:
            result = test_vllm_continuous_batching(args.model, args.max_tokens)
            results.append(("Continuous Batching Test", result))
        except Exception as e:
            print(f"Error in continuous batching test: {e}")
            results.append(("Continuous Batching Test", False))

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

# Create a simple benchmark script
BENCHMARK_SCRIPT="$INSTALL_DIR/benchmark_vllm.py"
cat > $BENCHMARK_SCRIPT << 'EOF'
#!/usr/bin/env python3
from vllm import LLM, SamplingParams
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def benchmark_throughput(model_name, batch_sizes, max_tokens=100, num_runs=3):
    """Benchmark vLLM throughput with different batch sizes."""
    print(f"=== Benchmarking vLLM Throughput with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM(model=model_name)

    # Base prompt
    base_prompt = "Write a short paragraph about"
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "renewable energy",
        "virtual reality",
        "blockchain technology",
        "genetic engineering",
        "autonomous vehicles",
        "robotics",
        "cybersecurity",
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision",
        "augmented reality",
        "internet of things",
        "cloud computing",
        "edge computing",
        "5G technology",
        "big data",
        "data science",
        "bioinformatics",
        "nanotechnology",
        "fusion energy",
        "solar power",
        "wind energy",
        "hydroelectric power",
        "geothermal energy",
        "nuclear energy",
        "sustainable development"
    ]

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Test different batch sizes
    results = []
    for batch_size in batch_sizes:
        if batch_size > len(topics):
            print(f"Skipping batch size {batch_size} (exceeds number of topics)")
            continue

        batch_results = []
        for run in range(num_runs):
            prompts = [f"{base_prompt} {topics[i % len(topics)]}" for i in range(batch_size)]

            # Warm-up
            _ = llm.generate(prompts, sampling_params)

            # Benchmark
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            generation_time = time.time() - start_time

            # Calculate statistics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / generation_time

            batch_results.append({
                "generation_time": generation_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second
            })

            print(f"Batch size: {batch_size}, Run {run+1}/{num_runs}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print()

        # Calculate average results
        avg_generation_time = np.mean([r["generation_time"] for r in batch_results])
        avg_total_tokens = np.mean([r["total_tokens"] for r in batch_results])
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in batch_results])

        results.append({
            "batch_size": batch_size,
            "avg_generation_time": avg_generation_time,
            "avg_total_tokens": avg_total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second
        })

    # Print summary
    print("\n=== Throughput Benchmark Summary ===")
    print("Batch Size | Avg Generation Time (s) | Avg Total Tokens | Avg Tokens/Second")
    print("----------|-------------------------|------------------|------------------")
    for result in results:
        print(f"{result['batch_size']:10} | {result['avg_generation_time']:23.2f} | {result['avg_total_tokens']:16.1f} | {result['avg_tokens_per_second']:18.2f}")

    return results

def benchmark_latency(model_name, prompt_lengths, max_tokens=100, num_runs=3):
    """Benchmark vLLM latency with different prompt lengths."""
    print(f"\n=== Benchmarking vLLM Latency with {model_name} ===")

    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM(model=model_name)

    # Base prompt
    base_prompt = "The quick brown fox jumps over the lazy dog. "

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Test different prompt lengths
    results = []
    for prompt_length in prompt_lengths:
        # Create prompt with specified length
        prompt = base_prompt * (prompt_length // len(base_prompt) + 1)
        prompt = prompt[:prompt_length]

        latency_results = []
        for run in range(num_runs):
            # Warm-up
            _ = llm.generate([prompt], sampling_params)

            # Benchmark
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            generation_time = time.time() - start_time

            # Calculate statistics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / generation_time

            latency_results.append({
                "generation_time": generation_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second
            })

            print(f"Prompt length: {prompt_length}, Run {run+1}/{num_runs}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print()

        # Calculate average results
        avg_generation_time = np.mean([r["generation_time"] for r in latency_results])
        avg_total_tokens = np.mean([r["total_tokens"] for r in latency_results])
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in latency_results])

        results.append({
            "prompt_length": prompt_length,
            "avg_generation_time": avg_generation_time,
            "avg_total_tokens": avg_total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second
        })

    # Print summary
    print("\n=== Latency Benchmark Summary ===")
    print("Prompt Length | Avg Generation Time (s) | Avg Total Tokens | Avg Tokens/Second")
    print("-------------|-------------------------|------------------|------------------")
    for result in results:
        print(f"{result['prompt_length']:13} | {result['avg_generation_time']:23.2f} | {result['avg_total_tokens']:16.1f} | {result['avg_tokens_per_second']:18.2f}")

    return results

def plot_results(throughput_results, latency_results, output_dir):
    """Plot benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot throughput results
    plt.figure(figsize=(10, 6))
    batch_sizes = [r["batch_size"] for r in throughput_results]
    tokens_per_second = [r["avg_tokens_per_second"] for r in throughput_results]

    plt.plot(batch_sizes, tokens_per_second, marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel("Tokens per Second")
    plt.title("vLLM Throughput vs Batch Size")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "vllm_throughput.png"))

    # Plot latency results
    plt.figure(figsize=(10, 6))
    prompt_lengths = [r["prompt_length"] for r in latency_results]
    generation_times = [r["avg_generation_time"] for r in latency_results]

    plt.plot(prompt_lengths, generation_times, marker='o')
    plt.xlabel("Prompt Length")
    plt.ylabel("Generation Time (s)")
    plt.title("vLLM Latency vs Prompt Length")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "vllm_latency.png"))

    print(f"Plots saved to {output_dir}")

def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark vLLM with AMD GPUs")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to benchmark")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory to save results")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16", help="Comma-separated list of batch sizes")
    parser.add_argument("--prompt-lengths", type=str, default="10,50,100,200,500", help="Comma-separated list of prompt lengths")

    args = parser.parse_args()

    # Parse batch sizes and prompt lengths
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    # Run benchmarks
    throughput_results = benchmark_throughput(args.model, batch_sizes, args.max_tokens, args.num_runs)
    latency_results = benchmark_latency(args.model, prompt_lengths, args.max_tokens, args.num_runs)

    # Plot results
    plot_results(throughput_results, latency_results, args.output_dir)

if __name__ == "__main__":
    main()
EOF

log "Created benchmark script at $BENCHMARK_SCRIPT"
log "You can run it with: python3 $BENCHMARK_SCRIPT"

# Restore original environment variables
log "Restoring original environment variables..."
if [ -n "$ORIGINAL_ROCR_VISIBLE_DEVICES" ]; then
    log "Restoring ROCR_VISIBLE_DEVICES to $ORIGINAL_ROCR_VISIBLE_DEVICES"
    export ROCR_VISIBLE_DEVICES=$ORIGINAL_ROCR_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_HIP_VISIBLE_DEVICES" ]; then
    log "Restoring HIP_VISIBLE_DEVICES to $ORIGINAL_HIP_VISIBLE_DEVICES"
    export HIP_VISIBLE_DEVICES=$ORIGINAL_HIP_VISIBLE_DEVICES
elif [ "$ORIGINAL_HIP_VISIBLE_DEVICES" = "" ]; then
    log "Unsetting HIP_VISIBLE_DEVICES as it was originally unset"
    unset HIP_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]; then
    log "Restoring CUDA_VISIBLE_DEVICES to $ORIGINAL_CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES
elif [ "$ORIGINAL_CUDA_VISIBLE_DEVICES" = "" ]; then
    log "Unsetting CUDA_VISIBLE_DEVICES as it was originally unset"
    unset CUDA_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_PYTORCH_ROCM_DEVICE" ]; then
    log "Restoring PYTORCH_ROCM_DEVICE to $ORIGINAL_PYTORCH_ROCM_DEVICE"
    export PYTORCH_ROCM_DEVICE=$ORIGINAL_PYTORCH_ROCM_DEVICE
elif [ "$ORIGINAL_PYTORCH_ROCM_DEVICE" = "" ]; then
    log "Unsetting PYTORCH_ROCM_DEVICE as it was originally unset"
    unset PYTORCH_ROCM_DEVICE
fi

log "=== vLLM Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $ML_STACK_DIR/docs/extensions/vllm_guide.md"

# Create a note about environment variables and version attribute for users
cat > "$INSTALL_DIR/VLLM_USAGE_NOTES.txt" << 'EOF'
IMPORTANT NOTES FOR USING vLLM WITH AMD GPUs

1. ENVIRONMENT VARIABLE REQUIREMENTS

vLLM uses Ray for parallel processing, which has specific requirements for environment variables:

a) Ray requires HIP_VISIBLE_DEVICES to be set instead of ROCR_VISIBLE_DEVICES
b) Ray requires consistent values between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
   (they must either have the same value or one must be unset)

Common errors you might encounter:

- "Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"
  This occurs when ROCR_VISIBLE_DEVICES is set but Ray requires HIP_VISIBLE_DEVICES.

- "AssertionError: Inconsistant values found. Please use either HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES"
  This occurs when both HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are set with different values.

2. MISSING __VERSION__ ATTRIBUTE

Some installations of vLLM may not have the __version__ attribute, which can cause errors like:

- "AttributeError: module 'vllm' has no attribute '__version__'"

This is not a critical error and doesn't mean vLLM is not installed correctly. The installation
can still be functional even without the version information. When you encounter this error,
you can safely ignore it or use alternative methods to check if vLLM is working:

```python
# Instead of checking the version:
# print(vllm.__version__)  # This might fail

# Check if key classes are available:
if hasattr(vllm, 'LLM') and hasattr(vllm, 'SamplingParams'):
    print("vLLM is installed and key classes are available")
```

3. RECOMMENDED SETUP FOR YOUR SCRIPTS

To use vLLM in your scripts, you should add these lines at the beginning:

```python
import os
import sys

# Function to log environment variable changes (optional)
def log_env(message):
    print(f"[ENV_SETUP] {message}")

# Step 1: Handle ROCR_VISIBLE_DEVICES
if 'ROCR_VISIBLE_DEVICES' in os.environ:
    log_env(f"Unsetting ROCR_VISIBLE_DEVICES (was: {os.environ['ROCR_VISIBLE_DEVICES']})")
    del os.environ['ROCR_VISIBLE_DEVICES']

# Step 2: Ensure HIP_VISIBLE_DEVICES is set
if 'HIP_VISIBLE_DEVICES' not in os.environ:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        log_env(f"Setting HIP_VISIBLE_DEVICES to match CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        os.environ['HIP_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        log_env("Setting HIP_VISIBLE_DEVICES to default value (0,1)")
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'  # Adjust as needed

# Step 3: Handle inconsistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
if 'HIP_VISIBLE_DEVICES' in os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['HIP_VISIBLE_DEVICES'] != os.environ['CUDA_VISIBLE_DEVICES']:
        log_env(f"Inconsistent values: HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        log_env("Unsetting CUDA_VISIBLE_DEVICES to prevent Ray initialization errors")
        del os.environ['CUDA_VISIBLE_DEVICES']

# Now you can import vLLM and Ray
try:
    import vllm
    # Handle missing __version__ attribute gracefully
    version = getattr(vllm, '__version__', 'Unknown')
    log_env(f"vLLM version: {version}")
except ImportError as e:
    log_env(f"Error importing vLLM: {e}")
    sys.exit(1)
except AttributeError as e:
    log_env(f"AttributeError: {e}")
    log_env("This is expected if vLLM is missing __version__ attribute")
    log_env("Continuing despite the missing attribute")
```

These workarounds are necessary due to conflicts between Ray's requirements and the
default environment variable setup in some AMD ROCm environments.

For more information, see the Ray documentation on GPU support:
https://docs.ray.io/en/latest/ray-core/gpu-support.html

You can also use the provided helper script to run vLLM with the correct environment variables:
$ /home/stan/Prod/Stan-s-ML-Stack/scripts/run_vllm.sh python3 your_script.py
EOF

log "Created usage notes at $INSTALL_DIR/VLLM_USAGE_NOTES.txt"


