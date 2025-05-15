#!/bin/bash

# verify_and_build.sh
# Comprehensive script to verify, build, and fix ML stack components

# Environment Configuration
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_DEVICE=0,1
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1

# Logging and Color Configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# --- Verification Functions ---

verify_pytorch() {
    log_warning "Verifying PyTorch installation..."
    python3 -c "import torch; print(torch.cuda.is_available())"
    if [ $? -eq 0 ]; then
        log_success "PyTorch is installed and detects CUDA/ROCm."
        return 0
    else
        log_error "PyTorch verification failed. CUDA/ROCm not detected or PyTorch not installed."
        return 1
    fi
}

verify_flash_attention() {
    log_warning "Verifying Flash Attention installation..."
    python3 -c "import flash_attention_amd"
    if [ $? -eq 0 ]; then
        log_success "Flash Attention is installed."
        return 0
    else
        log_error "Flash Attention verification failed. Module not found."
        return 1
    fi
}

# --- Build/Fix Functions ---

build_mpi_with_rocm() {
    log_warning "Attempting to build OpenMPI with ROCm/HIP support..."
    # Estimated time: 15-30 minutes

    mkdir -p $HOME/mpi_build
    cd $HOME/mpi_build

    # Check if MPI source already exists and is complete
    if [ ! -d "openmpi-5.0.2" ]; then
        log_warning "OpenMPI source not found. Downloading..."
        wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.bz2 || { log_error "Failed to download OpenMPI source."; return 1; }
        tar -xjf openmpi-5.0.2.tar.bz2 || { log_error "Failed to extract OpenMPI source."; return 1; }
    else
        log_success "OpenMPI source already present. Skipping download."
    fi

    cd openmpi-5.0.2

    # Check if already built and installed
    if [ ! -d "$HOME/openmpi" ] || [ -z "$(ls -A $HOME/openmpi)" ]; then
        log_warning "OpenMPI not found in installation directory. Configuring and building..."
        # Configure with ROCm support and disable C++ bindings
        ./configure \
            --prefix=$HOME/openmpi \
            --with-rocm=$ROCM_PATH \
            --with-hcc=/opt/rocm \
            --enable-shared \
            --disable-static \
            --enable-heterogeneous \
            --disable-mpi-cxx || { log_error "OpenMPI configure failed."; return 1; }

        make clean || { log_warning "make clean failed, proceeding anyway."; }
        make -j$(nproc) || { log_error "OpenMPI build failed."; return 1; }
        make install || { log_error "OpenMPI install failed."; return 1; }
    else
        log_success "OpenMPI already installed. Skipping build."
    fi

    # Update PATH and LD_LIBRARY_PATH
    export PATH=$HOME/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/openmpi/lib:$LD_LIBRARY_PATH

    log_success "OpenMPI build/fix attempt complete."
    return 0
}

verify_mpi() {
    log_warning "Verifying MPI installation with ROCm/HIP support..."
    # Check MPI version
    mpirun --version > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_error "MPI not found or mpirun command failed."
        return 1
    fi

    # Attempt a simple MPI program that uses GPU pointers (requires mpi4py and hip4py/cupy)
    # This is a more robust check for GPU support
    python3 -c "
import sys
try:
    from mpi4py import MPI
    import torch
    # Check if MPI can handle GPU pointers (simplified check)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.COMM_WORLD.Get_size()
    if size > 1:
        if rank == 0:
            gpu_tensor = torch.randn(10).to('cuda')
            comm.Send(gpu_tensor, dest=1, tag=11)
            print('MPI sent GPU tensor.')
        elif rank == 1:
            gpu_tensor = torch.empty(10).to('cuda')
            comm.Recv(gpu_tensor, source=0, tag=11)
            print('MPI received GPU tensor.')
    else:
         print('MPI size is 1, skipping GPU pointer test.')

    print('MPI verification passed.')
    sys.exit(0)
except ImportError as e:
    print(f'Import error during MPI verification: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error during MPI verification: {e}', file=sys.stderr)
    sys.exit(1)
"
    if [ $? -eq 0 ]; then
        log_success "MPI is installed and appears to have ROCm/HIP support."
        return 0
    else
        log_error "MPI verification failed or ROCm/HIP support not detected."
        return 1
    fi
}

verify_onnxruntime() {
    log_warning "Verifying ONNX Runtime installation with ROCm support..."
    python3 -c "
import sys
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('ONNX Runtime Version:', ort.__version__)
    print('Providers:', providers)
    if 'ROCMExecutionProvider' in providers:
        log_success('ONNX Runtime is installed and detects ROCMExecutionProvider.')
        sys.exit(0)
    else:
        log_error('ONNX Runtime verification failed. ROCMExecutionProvider not found.')
        sys.exit(1)
except ImportError as e:
    print(f'Import error during ONNX Runtime verification: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error during ONNX Runtime verification: {e}', file=sys.stderr)
    sys.exit(1)
"
    if [ $? -eq 0 ]; then
        log_success "ONNX Runtime verification passed."
        return 0
    else
        log_error "ONNX Runtime verification failed."
        return 1
    fi
}

build_onnxruntime() {
    log_warning "Attempting to build ONNX Runtime with ROCm support for gfx1100;gfx1101..."
    # Estimated time: 1-2 hours+

    # Ensure prerequisites
    pip install --break-system-packages cmake numpy pybind11 || { log_error "Failed to install ONNX Runtime prerequisites."; return 1; }

    mkdir -p $HOME/onnxruntime_build
    cd $HOME/onnxruntime_build

    # Check if repository exists and is complete
    if [ ! -d "onnxruntime" ]; then
        log_warning "ONNX Runtime source not found. Cloning..."
        git clone --recursive https://github.com/microsoft/onnxruntime.git || { log_error "Failed to clone ONNX Runtime repository."; return 1; }
    else
        cd onnxruntime
        # Check if build is incomplete or corrupted
        if [ ! -d "build/Linux/Release" ] || [ -z "$(ls -A build/Linux/Release)" ]; then
            log_warning "Existing ONNX Runtime build appears incomplete or corrupted. Cleaning and rebuilding."
            git clean -fdx || { log_warning "git clean failed, proceeding anyway."; }
            git reset --hard HEAD || { log_warning "git reset failed, proceeding anyway."; }
        else
            log_success "Existing ONNX Runtime source found. Skipping full clone."
        fi
        cd ..
    fi

    cd onnxruntime

    # Specific build configuration for ROCm and gfx1100;gfx1101
    log_warning "Configuring and building ONNX Runtime..."
    ./build.sh --parallel --config Release \
        --use_rocm \
        --rocm_home=$ROCM_PATH \
        --rocm_version=60400 \
        --cmake_extra_defines=\
"CMAKE_INSTALL_PREFIX=$HOME/onnxruntime_install,\
Donnxruntime_ROCM_ARCH=gfx1100;gfx1101,\
Donnxruntime_USE_COMPOSABLE_KERNEL=ON,\
Donnxruntime_ENABLE_PYTHON=ON" || { log_error "ONNX Runtime build configuration failed."; return 1; }

    # Build Python wheel
    cd build/Linux/Release
    cmake --build . --config Release -- -j$(nproc) || { log_error "ONNX Runtime build failed."; return 1; }
    cmake --install . || { log_error "ONNX Runtime install failed."; return 1; }

    # Install wheel
    find . -name "onnxruntime_rocm*.whl" | xargs pip install -U --break-system-packages || { log_error "Failed to install ONNX Runtime wheel."; return 1; }

    log_success "ONNX Runtime build/fix attempt complete."
    return 0
}


# --- Main Execution ---

main() {
    # Fix ninja and ninja-build symlinks
    fix_ninja_symlinks
    log_warning "Starting ML Stack Rebuild and Verification"

    # Check GPU Architecture
    GPU_ARCH=$(rocminfo | grep "Name:" | head -n 1 | awk '{print $2}')
    log_success "Detected GPU Architecture: $GPU_ARCH"

    # Verify and Build Components
    verify_pytorch || log_error "PyTorch verification failed."
    verify_flash_attention || log_error "Flash Attention verification failed."

    verify_mpi
    if [ $? -ne 0 ]; then
        build_mpi_with_rocm
        verify_mpi || log_error "MPI verification failed after rebuild attempt."
    fi

    verify_onnxruntime
    if [ $? -ne 0 ]; then
        build_onnxruntime
        verify_onnxruntime || log_error "ONNX Runtime verification failed after rebuild attempt."
    fi

    # Final Verification
    log_warning "Running final verification tests..."
    verify_pytorch
    verify_flash_attention
    verify_mpi
    verify_onnxruntime

    log_success "ML Stack Rebuild and Verification Complete!"
}

# Run the script
main
# Fix ninja and ninja-build symlinks
fix_ninja_symlinks() {
    if ! command -v ninja-build &>/dev/null && command -v ninja &>/dev/null; then
        log_warning "Creating symlink for ninja-build..."
        sudo_with_pass ln -sf $(which ninja) /usr/bin/ninja-build
        log_success "Ninja-build symlink created."
    fi
}
