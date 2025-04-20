#!/bin/bash
# Comprehensive ML Stack Rebuild and Verification Script
# Author: Stanley Chisango

set -e  # Exit immediately if a command exits with a non-zero status

# Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration Paths
ROCM_PATH="/opt/rocm"
ONNXRUNTIME_BUILD_DIR="$HOME/onnxruntime_build"
OPENMPI_BUILD_DIR="$HOME/mpi_build"
INSTALL_PREFIX="$HOME/ml_stack_install"

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Dependency Checks
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # ROCm dependencies
    if [ ! -d "$ROCM_PATH" ]; then
        log_error "ROCm not found at $ROCM_PATH"
        return 1
    fi
    log_success "ROCm found at $ROCM_PATH"
    
    # Python and pip
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        return 1
    fi
    log_success "Python3 found"
    
    # CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found"
        return 1
    fi
    log_success "CMake found"
    
    return 0
}

# Verify PyTorch Installation
verify_pytorch() {
    log_info "Verifying PyTorch Installation..."
    
    if python3 -c "import torch; print(torch.cuda.is_available(), torch.version.hip)" 2>/dev/null; then
        log_success "PyTorch with ROCm support is installed"
        return 0
    else
        log_error "PyTorch ROCm support not functioning"
        return 1
    fi
}

# Rebuild ONNX Runtime
rebuild_onnxruntime() {
    log_info "Rebuilding ONNX Runtime..."
    
    # Cleanup previous build
    rm -rf "$ONNXRUNTIME_BUILD_DIR/onnxruntime/build"
    
    # Clone or update repository
    if [ ! -d "$ONNXRUNTIME_BUILD_DIR/onnxruntime" ]; then
        git clone --recursive https://github.com/microsoft/onnxruntime.git "$ONNXRUNTIME_BUILD_DIR/onnxruntime"
    else
        cd "$ONNXRUNTIME_BUILD_DIR/onnxruntime"
        git pull
    fi
    
    cd "$ONNXRUNTIME_BUILD_DIR/onnxruntime"
    
    # Create build configuration
    mkdir -p build/Linux/Release
    cd build/Linux/Release
    
    cmake ../../../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -Donnxruntime_RUN_ONNX_TESTS=OFF \
        -Donnxruntime_GENERATE_TEST_REPORTS=OFF \
        -Donnxruntime_USE_ROCM=ON \
        -Donnxruntime_ROCM_HOME="$ROCM_PATH" \
        -Donnxruntime_ROCM_ARCH="gfx1100" \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_ENABLE_PYTHON=ON \
        -DPYTHON_EXECUTABLE=$(which python3)
    
    cmake --build . --config Release -- -j$(nproc)
    cmake --install .
    
    # Build Python wheel
    cd ../../../
    python3 setup.py bdist_wheel
    pip install dist/onnxruntime_rocm*.whl
    
    log_success "ONNX Runtime rebuilt successfully"
}

# Rebuild OpenMPI with ROCm Support
rebuild_openmpi() {
    log_info "Rebuilding OpenMPI with ROCm Support..."
    
    # Download OpenMPI source
    cd "$OPENMPI_BUILD_DIR"
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.bz2
    tar -xjf openmpi-5.0.2.tar.bz2
    cd openmpi-5.0.2
    
    # Configure with ROCm support, skipping C++ bindings
    ./configure \
        --prefix="$INSTALL_PREFIX" \
        --with-rocm="$ROCM_PATH" \
        --disable-mpi-cxx \
        --enable-mpi-fortran=no \
        CC=hipcc CXX=hipcc
    
    make -j$(nproc)
    make install
    
    log_success "OpenMPI rebuilt successfully"
}

# Main Verification Function
verify_ml_stack() {
    log_header "ML Stack Verification"
    
    # Check dependencies
    check_dependencies || return 1
    
    # Verify PyTorch
    verify_pytorch || rebuild_pytorch
    
    # Verify ONNX Runtime
    if ! python3 -c "import onnxruntime; print('ROCMExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null; then
        log_warning "ONNX Runtime needs rebuilding"
        rebuild_onnxruntime
    else
        log_success "ONNX Runtime verified"
    fi
    
    # Verify OpenMPI
    if ! command -v mpirun &> /dev/null; then
        log_warning "OpenMPI needs installation"
        rebuild_openmpi
    else
        log_success "OpenMPI verified"
    fi
    
    log_success "ML Stack Verification Complete!"
}

# Header for logs
log_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n"
}

# Main Execution
main() {
    log_header "Stan's ML Stack Comprehensive Rebuild"
    
    # Create installation directory
    mkdir -p "$INSTALL_PREFIX"
    
    # Verify and potentially rebuild ML Stack components
    verify_ml_stack
    
    log_success "ML Stack Configuration Complete!"
}

# Execute main function
main