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
# ONNX Runtime Build Script
# =============================================================================
# This script builds ONNX Runtime with ROCm support for AMD GPUs.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                           ONNX Runtime Build Script for AMD GPUs
EOF
echo

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
BLINK='\033[5m'
REVERSE='\033[7m'
RESET='\033[0m'

# Function definitions
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

show_progress() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    local progress=0
    local total=100
    local width=50

    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"

        # Increment progress (simulate progress)
        progress=$((progress + 1))
        if [ $progress -gt $total ]; then
            progress=$total
        fi

        # Calculate the number of filled and empty slots
        local filled=$(( progress * width / total ))
        local empty=$(( width - filled ))

        # Build the progress bar
        local bar="["
        for ((i=0; i<filled; i++)); do
            bar+="#"
        done
        for ((i=0; i<empty; i++)); do
            bar+="."
        done
        bar+="] $progress%"

        # Print the progress bar
        printf "\r${bar}"
    done
    printf "\r%${width}s\r" ""
}

check_prerequisites() {
    print_section "Checking prerequisites"

    # Check if ROCm is installed
    if ! command -v rocminfo &> /dev/null; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
    fi
    print_success "ROCm is installed"

    # Check if PyTorch with ROCm is installed
    if ! python3 -c "import torch; print(torch.version.hip)" &> /dev/null; then
        print_error "PyTorch with ROCm support is not installed. Please install PyTorch with ROCm support first."
        return 1
    fi
    print_success "PyTorch with ROCm support is installed"

    # Check if CUDA is available through ROCm
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_warning "CUDA is not available through ROCm. Check your environment variables."
        print_step "Setting environment variables..."
        export HIP_VISIBLE_DEVICES=0,1
        export CUDA_VISIBLE_DEVICES=0,1
        export PYTORCH_ROCM_DEVICE=0,1

        # Check again
        if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            print_error "CUDA is still not available through ROCm. Please check your ROCm installation."
            return 1
        fi
        print_success "Environment variables set successfully"
    fi
    print_success "CUDA is available through ROCm"

    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"

    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install git first."
        return 1
    fi
    print_success "Git is installed"

    # Check if cmake is installed
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install cmake first."
        return 1
    fi
    print_success "CMake is installed"

    # Check disk space
    available_space=$(df -h $HOME | awk 'NR==2 {print $4}')
    print_step "Available disk space: $available_space"

    # Check if there's enough disk space (at least 10GB)
    available_space_kb=$(df -k $HOME | awk 'NR==2 {print $4}')
    if [ $available_space_kb -lt 10485760 ]; then  # 10GB in KB
        print_warning "You have less than 10GB of free disk space. The build might fail."
        read -p "Do you want to continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Build aborted by user."
            return 1
        fi
    fi

    return 0
}

install_dependencies() {
    print_section "Installing dependencies"

    print_step "Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-setuptools \
        libpython3-dev \
        libnuma-dev \
        numactl

    print_step "Installing Python dependencies..."
    pip install numpy pybind11

    print_success "Dependencies installed successfully"
}
clone_repository() {
    print_section "Cloning ONNX Runtime repository"

    # Create build directory
    print_step "Creating build directory..."
    mkdir -p $HOME/onnxruntime-build
    cd $HOME/onnxruntime-build

    # Clone repository
    print_step "Cloning repository..."
    if [ -d "onnxruntime" ]; then
        print_warning "ONNX Runtime repository already exists. Updating..."
        cd onnxruntime
        git pull
        cd ..
    else
        git clone --recursive https://github.com/microsoft/onnxruntime.git
    fi

    print_success "Repository cloned successfully"
}

configure_build() {
    print_section "Configuring build"

    # Set ROCm path
    print_step "Setting ROCm path..."
    export ROCM_PATH=/opt/rocm

    # Set Python path
    print_step "Setting Python path..."
    export PYTHON_BIN_PATH=$(which python3)

    # Set build type
    print_step "Setting build type..."
    export BUILD_TYPE=Release

    # Source hipify utilities
    print_step "Sourcing hipify utilities..."
    source "$(dirname "$0")/hipify_utils.sh"

    # Apply hipify to ONNX Runtime source
    print_step "Applying hipify to ONNX Runtime source..."
    cd $HOME/onnxruntime-build/onnxruntime

    # Hipify CUDA kernels in onnxruntime
    print_step "Hipifying CUDA kernels..."

    # Find all CUDA directories
    CUDA_DIRS=$(find . -type d -name "cuda")
    for dir in $CUDA_DIRS; do
        print_step "Hipifying directory: $dir"
        # Get the parent directory
        parent_dir=$(dirname "$dir")
        # Create hip directory if it doesn't exist
        hip_dir="$parent_dir/hip"
        mkdir -p "$hip_dir"
        # Hipify the directory
        hipify_directory "$dir" "$hip_dir"
        # Apply post-hipify fixes
        apply_post_hipify_fixes "$hip_dir"
    done

    # Hipify ONNX components that might be missed
    print_step "Hipifying additional ONNX components..."

    # Find all CUDA files in the entire repository
    print_step "Finding all CUDA files in the repository..."
    CUDA_FILES=$(find . -type f -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" -o -name "*.cc" | grep -i cuda)

    # Process each file
    for file in $CUDA_FILES; do
        # Skip files in already processed cuda directories
        if echo "$file" | grep -q "/cuda/"; then
            continue
        fi

        print_step "Hipifying file: $file"

        # Determine output file path
        dir_name=$(dirname "$file")
        base_name=$(basename "$file")

        # Create hip directory if needed
        hip_dir="$dir_name/hip"
        mkdir -p "$hip_dir"

        # Determine output file name
        if [[ "$base_name" == *.cu ]]; then
            output_file="$hip_dir/${base_name%.cu}.hip.cpp"
        elif [[ "$base_name" == *.cuh ]]; then
            output_file="$hip_dir/${base_name%.cuh}.hip.hpp"
        else
            output_file="$hip_dir/$base_name"
        fi

        # Hipify the file
        hipify_file "$file" "$output_file"

        # Apply post-hipify fixes to individual file
        apply_post_hipify_fixes "$hip_dir"
    done

    # Hipify specific ONNX components that are known to need it
    print_step "Hipifying specific ONNX components..."

    # List of specific directories to hipify
    SPECIFIC_DIRS=(
        "./onnxruntime/core/providers/cuda"
        "./onnxruntime/core/providers/rocm"
        "./onnxruntime/core/providers/dml"
        "./onnxruntime/core/providers/tensorrt"
        "./onnxruntime/core/optimizer/cuda"
        "./onnxruntime/core/mlas/lib/cuda"
    )

    for dir in "${SPECIFIC_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            print_step "Hipifying specific directory: $dir"
            # Create hip directory
            hip_dir="${dir%/*}/hip"
            mkdir -p "$hip_dir"
            # Hipify the directory
            hipify_directory "$dir" "$hip_dir"
            # Apply post-hipify fixes
            apply_post_hipify_fixes "$hip_dir"
        fi
    done

    # Create build configuration script
    print_step "Creating build configuration script..."
    cd $HOME/onnxruntime-build

    cat > build_rocm.sh << 'EOF'
#!/bin/bash

# Set build directory
BUILD_DIR="$HOME/onnxruntime-build/onnxruntime/build/Linux/Release"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure build with ROCm support
cmake ../../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/onnxruntime-build/onnxruntime/install \
    -Donnxruntime_RUN_ONNX_TESTS=OFF \
    -Donnxruntime_GENERATE_TEST_REPORTS=OFF \
    -Donnxruntime_USE_ROCM=ON \
    -Donnxruntime_ROCM_HOME=/opt/rocm \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_ENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$PYTHON_BIN_PATH \
    -Donnxruntime_USE_COMPOSABLE_KERNEL=ON \
    -Donnxruntime_USE_MIMALLOC=OFF \
    -Donnxruntime_ENABLE_ROCM_PROFILING=OFF

# Build ONNX Runtime
cmake --build . --config Release -- -j$(nproc)

# Install ONNX Runtime
cmake --install .

# Build Python wheel
cd ../../../
$PYTHON_BIN_PATH setup.py bdist_wheel
EOF

    chmod +x build_rocm.sh

    print_success "Build configured successfully"
}

build_onnxruntime() {
    print_section "Building ONNX Runtime"

    # Run build script
    print_step "Running build script..."
    cd $HOME/onnxruntime-build

    # Start the build in the background and show progress
    ./build_rocm.sh > build.log 2>&1 &
    build_pid=$!

    print_step "Building ONNX Runtime (this may take a while)..."
    show_progress $build_pid

    # Check if build was successful
    if wait $build_pid; then
        print_success "ONNX Runtime built successfully"
    else
        print_error "ONNX Runtime build failed. Check build.log for details."
        return 1
    fi

    return 0
}

install_onnxruntime() {
    print_section "Installing ONNX Runtime"

    # Install the Python wheel
    print_step "Installing Python wheel..."
    cd $HOME/onnxruntime-build/onnxruntime

    # Find the wheel file
    wheel_file=$(find dist -name "onnxruntime_rocm-*.whl")

    if [ -z "$wheel_file" ]; then
        print_error "Could not find ONNX Runtime wheel file."
        return 1
    fi

    # Install the wheel
    pip install $wheel_file

    print_success "ONNX Runtime installed successfully"
}

verify_installation() {
    print_section "Verifying installation"

    # Create test script
    print_step "Creating test script..."
    cd $HOME/onnxruntime-build

    cat > test_onnxruntime.py << 'EOF'
import onnxruntime as ort
import torch
import numpy as np

def test_onnxruntime():
    # Print ONNX Runtime version
    print(f"ONNX Runtime version: {ort.__version__}")

    # Check available providers
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleModel()
    model.eval()

    # Create input data
    x = torch.randn(1, 10)

    # Export model to ONNX
    onnx_file = "simple_model.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_file,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # Run inference with PyTorch
    with torch.no_grad():
        torch_output = model(x).numpy()

    # Create ONNX Runtime session
    if 'ROCMExecutionProvider' in providers:
        session = ort.InferenceSession(onnx_file, providers=['ROCMExecutionProvider'])
        print("Using ROCMExecutionProvider")
    else:
        session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        print("Using CPUExecutionProvider")

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference with ONNX Runtime
    ort_inputs = {input_name: x.numpy()}
    ort_output = session.run([output_name], ort_inputs)[0]

    # Compare PyTorch and ONNX Runtime outputs
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
    print("PyTorch and ONNX Runtime outputs match")

    print("ONNX Runtime test passed!")

if __name__ == "__main__":
    test_onnxruntime()
EOF

    # Run test script
    print_step "Running test script..."
    python3 test_onnxruntime.py

    if [ $? -eq 0 ]; then
        print_success "ONNX Runtime is working correctly"
        return 0
    else
        print_error "ONNX Runtime test failed"
        return 1
    fi
}
cleanup() {
    print_section "Cleaning up"

    # Remove temporary files
    print_step "Removing temporary files..."
    cd $HOME/onnxruntime-build
    rm -f test_onnxruntime.py simple_model.onnx

    print_success "Cleanup completed successfully"
}

main() {
    print_header "ONNX Runtime Build Script for AMD GPUs"

    # Start time
    start_time=$(date +%s)

    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi

    # Install dependencies
    install_dependencies
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies. Exiting."
        exit 1
    fi

    # Clone repository
    clone_repository
    if [ $? -ne 0 ]; then
        print_error "Failed to clone repository. Exiting."
        exit 1
    fi

    # Configure build
    configure_build
    if [ $? -ne 0 ]; then
        print_error "Failed to configure build. Exiting."
        exit 1
    fi

    # Build ONNX Runtime
    build_onnxruntime
    if [ $? -ne 0 ]; then
        print_error "Failed to build ONNX Runtime. Exiting."
        exit 1
    fi

    # Install ONNX Runtime
    install_onnxruntime
    if [ $? -ne 0 ]; then
        print_error "Failed to install ONNX Runtime. Exiting."
        exit 1
    fi

    # Verify installation
    verify_installation
    if [ $? -ne 0 ]; then
        print_error "Installation verification failed. Exiting."
        exit 1
    fi

    # Cleanup
    cleanup

    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))

    print_header "ONNX Runtime Build Completed Successfully!"
    echo -e "${GREEN}Total build time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"
    echo
    echo -e "${CYAN}You can now use ONNX Runtime in your Python code:${RESET}"
    echo
    echo -e "${YELLOW}import onnxruntime as ort${RESET}"
    echo -e "${YELLOW}import numpy as np${RESET}"
    echo
    echo -e "${YELLOW}# Create ONNX Runtime session${RESET}"
    echo -e "${YELLOW}session = ort.InferenceSession(\"model.onnx\", providers=['ROCMExecutionProvider'])${RESET}"
    echo
    echo -e "${YELLOW}# Run inference${RESET}"
    echo -e "${YELLOW}input_name = session.get_inputs()[0].name${RESET}"
    echo -e "${YELLOW}output_name = session.get_outputs()[0].name${RESET}"
    echo -e "${YELLOW}ort_inputs = {input_name: np.random.randn(1, 10).astype(np.float32)}${RESET}"
    echo -e "${YELLOW}ort_output = session.run([output_name], ort_inputs)[0]${RESET}"
    echo

    return 0
}

# Main script execution
main
