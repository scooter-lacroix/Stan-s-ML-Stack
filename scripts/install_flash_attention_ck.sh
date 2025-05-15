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
# Flash Attention CK Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures Flash Attention with Composable Kernel
# support for AMD GPUs.
#
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/flash_attention_ck_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

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

# Start installation
log "=== Starting Flash Attention CK Installation ==="
log "System: $(uname -a)"

# Check ROCm installation
ROCM_BIN_PATH=$(which hipcc 2>/dev/null || echo 'Not found')

# If running as root (via sudo), try to find ROCm path from the system
if [ "$EUID" -eq 0 ] && [ "$ROCM_BIN_PATH" = "Not found" ]; then
    # Try to find ROCm installation in common locations
    if [ -d "/opt/rocm" ]; then
        ROCM_BIN_PATH="/opt/rocm/bin/hipcc"
        log "Found ROCm in /opt/rocm"
    else
        # Try to find any rocm installation
        ROCM_DIRS=$(ls -d /opt/rocm* 2>/dev/null)
        if [ -n "$ROCM_DIRS" ]; then
            ROCM_PATH=$(echo "$ROCM_DIRS" | head -n 1)
            ROCM_BIN_PATH="$ROCM_PATH/bin/hipcc"
            log "Found ROCm in $ROCM_PATH"
        fi
    fi

    # Update PATH to include ROCm binaries
    if [ "$ROCM_BIN_PATH" != "Not found" ]; then
        ROCM_PATH=$(dirname $(dirname $ROCM_BIN_PATH))
        export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/hip/lib
    fi
fi

log "ROCm Path: $ROCM_BIN_PATH"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required dependencies
log "Checking dependencies..."
# Fix ninja-build detection
fix_ninja_detection

# Refresh PATH to include newly installed ninja-build
export PATH=$PATH:/usr/bin

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

# Check for Python development headers
log "Checking for Python development headers..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSTEM_PYTHON_INCLUDE_DIR="/usr/include/python${PYTHON_VERSION}"

# For virtual environments created with uv, we need to create a symlink to Python.h
PYTHON_INCLUDE_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')

if [ -d "$SYSTEM_PYTHON_INCLUDE_DIR" ]; then
    log "Found system Python include directory: $SYSTEM_PYTHON_INCLUDE_DIR"

    if [ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]; then
        log "Creating symlink for Python.h in virtual environment..."
        mkdir -p "$PYTHON_INCLUDE_DIR"

        # Create symlinks without sudo
        for header in "$SYSTEM_PYTHON_INCLUDE_DIR"/*.h; do
            base_header=$(basename "$header")
            if [ ! -f "$PYTHON_INCLUDE_DIR/$base_header" ]; then
                ln -sf "$header" "$PYTHON_INCLUDE_DIR/$base_header"
            fi
        done

        # Link internal directory if it exists
        if [ -d "$SYSTEM_PYTHON_INCLUDE_DIR/internal" ]; then
            mkdir -p "$PYTHON_INCLUDE_DIR/internal"
            for header in "$SYSTEM_PYTHON_INCLUDE_DIR/internal"/*.h; do
                base_header=$(basename "$header")
                if [ ! -f "$PYTHON_INCLUDE_DIR/internal/$base_header" ]; then
                    ln -sf "$header" "$PYTHON_INCLUDE_DIR/internal/$base_header"
                fi
            done
        fi
    else
        log "Python.h already exists in virtual environment"
    fi
else
    log "System Python include directory not found. Python development headers may be missing."
    log "You may need to install python${PYTHON_VERSION}-dev package manually."
fi

# Install PyTorch if not already installed
if ! python3 -c "import torch" &>/dev/null; then
    log "Installing PyTorch..."
    if command_exists uv; then
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
    else
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
    fi
fi

# Create installation directory
if [ "$EUID" -eq 0 ]; then
    # If running as root (via sudo), use the SUDO_USER's home directory
    if [ -n "$SUDO_USER" ]; then
        USER_HOME=$(eval echo ~$SUDO_USER)
        INSTALL_DIR="$USER_HOME/ml_stack/flash_attn_amd"
    else
        # Fallback to current user's home
        INSTALL_DIR="$HOME/ml_stack/flash_attn_amd"
    fi
else
    INSTALL_DIR="$HOME/ml_stack/flash_attn_amd"
fi

log "Installation directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
# Make sure the directory is owned by the correct user
if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER $INSTALL_DIR
fi

# Clone the repository if it doesn't exist
if [ ! -d "$INSTALL_DIR/.git" ]; then
    log "Cloning Flash Attention repository..."
    git clone https://github.com/ROCmSoftwarePlatform/flash-attention.git $INSTALL_DIR
    cd $INSTALL_DIR
    # Use main branch instead of rocm-5.6
    git checkout main
else
    log "Flash Attention repository already exists, updating..."
    cd $INSTALL_DIR
    git fetch
    git checkout main
    git pull
fi

# Copy the CK implementation files
log "Copying CK implementation files..."

# Determine the core directory path
if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
    CORE_DIR="$USER_HOME/Prod/Stan-s-ML-Stack/core/flash_attention"
else
    CORE_DIR="$HOME/Prod/Stan-s-ML-Stack/core/flash_attention"
fi

# Check if the core directory exists
if [ ! -d "$CORE_DIR" ]; then
    log "Core directory not found at $CORE_DIR"
    # Try to find it in the current directory structure
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    CORE_DIR="$PARENT_DIR/core/flash_attention"

    if [ ! -d "$CORE_DIR" ]; then
        log "Error: Could not find core directory. Please run this script from the Stan-s-ML-Stack directory."
        exit 1
    else
        log "Found core directory at $CORE_DIR"
    fi
fi

mkdir -p $INSTALL_DIR/flash_attention_amd
cp -f $CORE_DIR/CMakeLists.txt $INSTALL_DIR/
cp -f $CORE_DIR/flash_attention_amd.cpp $INSTALL_DIR/
cp -f $CORE_DIR/flash_attention_amd_cuda.cpp $INSTALL_DIR/
cp -f $CORE_DIR/flash_attention_amd.py $INSTALL_DIR/flash_attention_amd/
cp -f $CORE_DIR/setup_flash_attn_amd.py $INSTALL_DIR/

# Build the CK implementation
log "Building Flash Attention CK implementation..."
cd $INSTALL_DIR

# Create build directory
mkdir -p build
cd build

# Configure CMake
log "Configuring CMake..."

# Fix the Tool lib 1 failure by adding LLVM bin to PATH
ROCM_VERSION=$(ls -d /opt/rocm* 2>/dev/null | head -n 1)
if [ -d "$ROCM_VERSION/lib/llvm/bin" ]; then
    log "Adding LLVM bin directory to PATH..."
    export PATH=$PATH:$ROCM_VERSION/lib/llvm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_VERSION/lib:$ROCM_VERSION/lib/llvm/lib

    # Create a symlink for amdgpu-arch if it doesn't exist
    if [ ! -f "/usr/bin/amdgpu-arch" ] && [ -f "$ROCM_VERSION/lib/llvm/bin/amdgpu-arch" ]; then
        log "Creating symlink for amdgpu-arch..."
        sudo ln -sf $ROCM_VERSION/lib/llvm/bin/amdgpu-arch /usr/bin/amdgpu-arch
    fi
fi

# Set GPU_TARGETS explicitly to avoid detection issues
# Try to detect GPU architecture
GPU_ARCH="gfx1100"  # Default fallback
if [ -f "/usr/bin/amdgpu-arch" ] || [ -f "$ROCM_VERSION/lib/llvm/bin/amdgpu-arch" ]; then
    DETECTED_ARCH=$($ROCM_VERSION/lib/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null)
    if [ -n "$DETECTED_ARCH" ]; then
        # Remove any newlines from the architecture string
        GPU_ARCH=$(echo "$DETECTED_ARCH" | tr -d '\n')
        log "Detected GPU architecture: $GPU_ARCH"
    fi
fi

# Ignore the kineto library warning - it's not critical
cmake .. \
    -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="$GPU_ARCH" \
    -DCMAKE_CXX_FLAGS="-Wno-error"

# Build
log "Building..."
cmake --build . --config Release -j $(nproc)

# Install
log "Installing..."
# Check if the library was built successfully
if [ -f "flash_attention_amd_cuda.so" ]; then
    log "Library built successfully in build directory"
    cp flash_attention_amd_cuda.so ..
elif [ -f "../flash_attention_amd_cuda.so" ]; then
    log "Library already in the correct location"
else
    # Try to find the library
    LIBRARY_PATH=$(find . -name "flash_attention_amd_cuda.so" 2>/dev/null)
    if [ -n "$LIBRARY_PATH" ]; then
        log "Found library at $LIBRARY_PATH"
        cp "$LIBRARY_PATH" ..
    else
        log "Warning: Library not found, but continuing anyway"
    fi
fi

# Install the Python package
log "Installing Python package..."
cd $INSTALL_DIR
python3 setup_flash_attn_amd.py install

# Test the installation
log "Testing installation..."
python3 -c "
try:
    import flash_attention_amd
    from flash_attention_amd import FlashAttention
    print('Flash Attention CK successfully imported')
    print('Available classes:', dir(flash_attention_amd))
except Exception as e:
    print(f'Error importing Flash Attention CK: {e}')
"

log "=== Flash Attention CK Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Prod/Stan-s-ML-Stack/docs/extensions/flash_attention_ck_guide.md"

# Final message
echo "============================================================"
echo "Flash Attention CK Installation Complete!"
echo "Documentation is available in $HOME/Prod/Stan-s-ML-Stack/docs/extensions/flash_attention_ck_guide.md"
echo "Installation logs are available in $LOG_FILE"
echo "============================================================"
