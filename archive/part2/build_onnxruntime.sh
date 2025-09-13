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

# Check if terminal supports colors
if [ -t 1 ]; then
    # Check if NO_COLOR environment variable is set
    if [ -z "$NO_COLOR" ]; then
        # Terminal supports colors
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
    else
        # NO_COLOR is set, don't use colors
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        MAGENTA=''
        CYAN=''
        BOLD=''
        UNDERLINE=''
        BLINK=''
        REVERSE=''
        RESET=''
    fi
else
    # Not a terminal, don't use colors
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    UNDERLINE=''
    BLINK=''
    REVERSE=''
    RESET=''
fi

# Function definitions
print_header() {
    echo
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║                                                         ║"
    echo "║               === $1 ===               ║"
    echo "║                                                         ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    echo
}

print_section() {
    echo
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ $1"
    echo "└─────────────────────────────────────────────────────────┘"
}

print_step() {
    echo "➤ $1"
}

print_success() {
    echo "✓ $1"
}

print_warning() {
    echo "⚠ $1"
}

print_error() {
    echo "✗ $1"
}

# Function to print a clean separator line
print_separator() {
    echo "───────────────────────────────────────────────────────────"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to detect package manager
detect_package_manager() {
    if command_exists dnf; then
        echo "dnf"
    elif command_exists apt-get; then
        echo "apt"
    elif command_exists yum; then
        echo "yum"
    elif command_exists pacman; then
        echo "pacman"
    elif command_exists zypper; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Function to use uv or pip for Python packages
install_python_package() {
    local package="$1"
    shift
    local extra_args="$@"

    if command_exists uv; then
        print_step "Installing $package with uv..."
        uv pip install --python $(which python3) $extra_args "$package"
    else
        print_step "Installing $package with pip..."
        python3 -m pip install $extra_args "$package"
    fi
}

# Function to show environment variables
show_env() {
    # Set up minimal ROCm environment for showing variables
    HSA_TOOLS_LIB=0
    HSA_OVERRIDE_GFX_VERSION=11.0.0
    PYTORCH_ROCM_ARCH="gfx1100"
    ROCM_PATH="/opt/rocm"
    PATH="/opt/rocm/bin:$PATH"
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Check if rocprofiler library exists and update HSA_TOOLS_LIB accordingly
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
    fi

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    fi

    echo "export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\""
    echo "export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\""
    if [ -n "$PYTORCH_ALLOC_CONF" ]; then
        echo "export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\""
    fi
    echo "export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\""
    echo "export ROCM_PATH=\"$ROCM_PATH\""
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
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
    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH="gfx1100"
        export ROCM_PATH="/opt/rocm"
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Check if we can install rocprofiler
            if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
                print_step "Installing rocprofiler for HSA tools support..."
                sudo apt-get update && sudo apt-get install -y rocprofiler
                if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
                    export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
                    print_success "ROCm profiler installed and configured"
                else
                    export HSA_TOOLS_LIB=0
                    print_warning "ROCm profiler installation failed, disabling HSA tools"
                fi
            else
                export HSA_TOOLS_LIB=0
                print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
            fi
        fi

        # Fix deprecated PYTORCH_CUDA_ALLOC_CONF warning
        if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
            export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
            unset PYTORCH_CUDA_ALLOC_CONF
            print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
        fi

        print_success "ROCm environment variables configured"
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            package_manager=$(detect_package_manager)
            case $package_manager in
                apt)
                    sudo apt update && sudo apt install -y rocminfo
                    ;;
                dnf)
                    sudo dnf install -y rocminfo
                    ;;
                yum)
                    sudo yum install -y rocminfo
                    ;;
                pacman)
                    sudo pacman -S rocminfo
                    ;;
                zypper)
                    sudo zypper install -y rocminfo
                    ;;
                *)
                    print_error "Unsupported package manager: $package_manager"
                    return 1
                    ;;
            esac
            if command_exists rocminfo; then
                print_success "Installed rocminfo"
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi

    # Detect ROCm version
    rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

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
    if ! command_exists git; then
        print_error "Git is not installed. Please install git first."
        return 1
    fi
    print_success "Git is installed"

    # Check if cmake is installed
    if ! command_exists cmake; then
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

    # Check if uv is installed
    if ! command_exists uv; then
        print_step "Installing uv package manager..."
        python3 -m pip install uv

        # Add uv to PATH if it was installed in a user directory
        if [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi

        # Add uv to PATH if it was installed via cargo
        if [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        if ! command_exists uv; then
            print_error "Failed to install uv package manager"
            print_step "Falling back to pip"
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}ONNX Runtime Installation Options:${RESET}"
    echo "1) Global installation (recommended for system-wide use)"
    echo "2) Virtual environment (isolated installation)"
    echo "3) Auto-detect (try global, fallback to venv if needed)"
    echo
    read -p "Choose installation method (1-3) [3]: " INSTALL_CHOICE
    INSTALL_CHOICE=${INSTALL_CHOICE:-3}

    case $INSTALL_CHOICE in
        1)
            INSTALL_METHOD="global"
            print_step "Using global installation method"
            ;;
        2)
            INSTALL_METHOD="venv"
            print_step "Using virtual environment method"
            ;;
        3|*)
            INSTALL_METHOD="auto"
            print_step "Using auto-detect method"
            ;;
    esac

    # Create a function to handle uv commands properly with venv fallback
    uv_pip_install() {
        local args="$@"

        # Check if uv is available as a command
        if command_exists uv; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    python3 -m pip install --break-system-packages $args
                    ONNXRUNTIME_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    VENV_DIR="./onnxruntime_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args
                    ONNXRUNTIME_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                    ;;
                "auto")
                    # Try global install first
                    print_step "Attempting global installation with uv..."
                    local install_output
                    install_output=$(uv pip install --python $(which python3) $args 2>&1)
                    local install_exit_code=$?

                    if echo "$install_output" | grep -q "externally managed"; then
                        print_warning "Global installation failed due to externally managed environment"
                        print_step "Creating uv virtual environment for installation..."

                        # Create uv venv in project directory
                        VENV_DIR="./onnxruntime_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        ONNXRUNTIME_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global installation successful"
                        ONNXRUNTIME_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        VENV_DIR="./onnxruntime_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        ONNXRUNTIME_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing with pip..."
            python3 -m pip install $args
            ONNXRUNTIME_VENV_PYTHON=""
        fi
    }

    # Install build dependencies using detected package manager
    package_manager=$(detect_package_manager)
    print_step "Installing build dependencies using $package_manager..."

    case $package_manager in
        apt)
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
            ;;
        dnf)
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                python3-devel \
                python3-pip \
                python3-numpy \
                python3-setuptools \
                libnuma-devel \
                numactl
            ;;
        yum)
            sudo yum install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                python3-devel \
                python3-pip \
                python3-numpy \
                python3-setuptools \
                libnuma-devel \
                numactl
            ;;
        pacman)
            sudo pacman -S --noconfirm \
                gcc \
                cmake \
                git \
                python \
                python-pip \
                python-numpy \
                python-setuptools \
                numactl
            ;;
        zypper)
            sudo zypper install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                python3-devel \
                python3-pip \
                python3-numpy \
                python3-setuptools \
                libnuma-devel \
                numactl
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            print_step "Attempting to install with apt as fallback..."
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
            ;;
    esac

    print_step "Installing Python dependencies..."
    uv_pip_install numpy pybind11

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

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${ONNXRUNTIME_VENV_PYTHON:-python3}

    # Set ROCm path
    print_step "Setting ROCm path..."
    export ROCM_PATH=/opt/rocm

    # Set Python path
    print_step "Setting Python path..."
    export PYTHON_BIN_PATH=$(which $PYTHON_CMD)

    # Set build type
    print_step "Setting build type..."
    export BUILD_TYPE=Release

    # Check for hipify utilities
    HIPIFY_UTILS_PATH="$(dirname "$0")/hipify_utils.sh"
    if [ -f "$HIPIFY_UTILS_PATH" ]; then
        # Source hipify utilities
        print_step "Sourcing hipify utilities..."
        source "$HIPIFY_UTILS_PATH"

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
    else
        print_warning "Hipify utilities not found at $HIPIFY_UTILS_PATH"
        print_step "Skipping hipify transformations - using ROCm provider directly"
        print_step "Note: This may result in limited GPU acceleration for some operations"
        cd $HOME/onnxruntime-build/onnxruntime
    fi

    # Create build configuration script
    print_step "Creating build configuration script..."
    cd $HOME/onnxruntime-build

    cat > build_rocm.sh << EOF
#!/bin/bash

# Set build directory
BUILD_DIR="\$HOME/onnxruntime-build/onnxruntime/build/Linux/Release"
mkdir -p \$BUILD_DIR
cd \$BUILD_DIR

# Configure build with ROCm support
cmake ../../../ \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_INSTALL_PREFIX=\$HOME/onnxruntime-build/onnxruntime/install \\
    -Donnxruntime_RUN_ONNX_TESTS=OFF \\
    -Donnxruntime_GENERATE_TEST_REPORTS=OFF \\
    -Donnxruntime_USE_ROCM=ON \\
    -Donnxruntime_ROCM_HOME=/opt/rocm \\
    -Donnxruntime_BUILD_SHARED_LIB=ON \\
    -Donnxruntime_ENABLE_PYTHON=ON \\
    -DPYTHON_EXECUTABLE=$PYTHON_BIN_PATH \\
    -Donnxruntime_USE_COMPOSABLE_KERNEL=ON \\
    -Donnxruntime_USE_MIMALLOC=OFF \\
    -Donnxruntime_ENABLE_ROCM_PROFILING=OFF

# Build ONNX Runtime
cmake --build . --config Release -- -j\$(nproc)

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

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${ONNXRUNTIME_VENV_PYTHON:-python3}

    # Install the Python wheel
    print_step "Installing Python wheel..."
    cd $HOME/onnxruntime-build/onnxruntime

    # Find the wheel file
    wheel_file=$(find dist -name "onnxruntime_rocm-*.whl")

    if [ -z "$wheel_file" ]; then
        print_error "Could not find ONNX Runtime wheel file."
        return 1
    fi

    # Install the wheel using the appropriate method
    if command_exists uv; then
        if [ -n "$ONNXRUNTIME_VENV_PYTHON" ]; then
            # Install in virtual environment
            source "$(dirname "$ONNXRUNTIME_VENV_PYTHON")/../bin/activate"
            uv pip install "$wheel_file"
        else
            # Global installation
            uv pip install --python $(which python3) "$wheel_file"
        fi
    else
        # Fallback to pip
        if [ -n "$ONNXRUNTIME_VENV_PYTHON" ]; then
            $PYTHON_CMD -m pip install "$wheel_file"
        else
            python3 -m pip install "$wheel_file"
        fi
    fi

    print_success "ONNX Runtime installed successfully"
}

verify_installation() {
    print_section "Verifying installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${ONNXRUNTIME_VENV_PYTHON:-python3}

    # Create test script
    print_step "Creating test script..."
    cd $HOME/onnxruntime-build

    cat > test_onnxruntime.py << EOF
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
    $PYTHON_CMD test_onnxruntime.py

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

    # Show a visually appealing completion message
    clear
    cat << "EOF"

    ██████╗ ███╗   ██╗███╗   ██╗██╗  ██╗    ██████╗ ██╗   ██╗███╗   ██╗████████╗██╗███╗   ███╗███████╗
    ██╔═══██╗████╗  ██║████╗  ██║╚██╗██╔╝    ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██║████╗ ████║██╔════╝
    ██║   ██║██╔██╗ ██║██╔██╗ ██║ ╚███╔╝     ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║██╔████╔██║█████╗
    ██║   ██║██║╚██╗██║██║╚██╗██║ ██╔██╗     ██╔══██╗██║   ██║██║╚██╗██║   ██║   ██║██║╚██╔╝██║██╔══╝
    ╚██████╔╝██║ ╚████║██║ ╚████║██╔╝ ██╗    ██║  ██║╚██████╔╝██║ ╚████║   ██║   ██║██║ ╚═╝ ██║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝

                            Build Completed Successfully!
EOF

    print_success "ONNX Runtime with ROCm support has been built and installed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$ONNXRUNTIME_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./onnxruntime_rocm_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())\"${RESET}"
    fi
    echo
    echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}ROCm environment variables are set for this session.${RESET}"
    echo -e "${YELLOW}For future sessions, you may need to run:${RESET}"

    # Output the actual environment variables that were set
    echo -e "${GREEN}export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\"${RESET}"
    echo -e "${GREEN}export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\"${RESET}"
    if [ -n "$PYTORCH_ALLOC_CONF" ]; then
        echo -e "${GREEN}export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\"${RESET}"
    fi
    echo -e "${GREEN}export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\"${RESET}"
    echo -e "${GREEN}export ROCM_PATH=\"$ROCM_PATH\"${RESET}"
    echo -e "${GREEN}export PATH=\"$PATH\"${RESET}"
    echo -e "${GREEN}export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"${RESET}"
    echo
    echo -e "${CYAN}${BOLD}To apply these settings to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$(./build_onnxruntime.sh --show-env)\"${RESET}"
    echo
    echo -e "${GREEN}${BOLD}Build completed in: ${hours}h ${minutes}m ${seconds}s${RESET}"

    return 0
}

# Parse command line arguments
DRY_RUN=false
FORCE=false
SHOW_ENV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --show-env)
            SHOW_ENV=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --dry-run     Show what would be done without actually doing it"
            echo "  --force       Force reinstallation even if already installed"
            echo "  --show-env    Show required environment variables"
            echo "  --help, -h    Show this help message"
            echo
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for --show-env option
if [[ "$SHOW_ENV" == "true" ]]; then
    show_env
    exit 0
fi

# Function to detect if running in WSL
detect_wsl() {
    if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect if running in container
detect_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q container /proc/1/cgroup 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Enhanced error handling function
handle_error() {
    local exit_code=$1
    local error_message=$2
    local recovery_suggestion=$3

    print_error "$error_message"
    if [ -n "$recovery_suggestion" ]; then
        echo -e "${YELLOW}Suggestion: $recovery_suggestion${RESET}"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "This was a dry run - no actual changes were made"
        return 0
    fi

    return $exit_code
}

# Check if ONNX Runtime is already installed and handle --force flag
check_existing_installation() {
    if package_installed "onnxruntime" || package_installed "onnxruntime-gpu"; then
        onnxruntime_version=$(python3 -c "import onnxruntime as ort; print(ort.__version__)" 2>/dev/null)

        if [[ "$FORCE" == "true" ]]; then
            print_warning "ONNX Runtime $onnxruntime_version is already installed. Force reinstall requested - proceeding with reinstallation"
            return 0
        else
            print_success "ONNX Runtime $onnxruntime_version is already installed"
            print_step "Use --force to reinstall anyway"
            return 1
        fi
    fi
    return 0
}

# Detect environment
print_step "Detecting environment..."
IS_WSL=$(detect_wsl)
IS_CONTAINER=$(detect_container)

if [[ "$IS_WSL" == "true" ]]; then
    print_step "Running in WSL environment"
fi

if [[ "$IS_CONTAINER" == "true" ]]; then
    print_step "Running in container environment"
fi

# Check existing installation
if ! check_existing_installation; then
    exit 0
fi

# Dry run mode
if [[ "$DRY_RUN" == "true" ]]; then
    print_warning "DRY RUN MODE - No actual changes will be made"
    echo
fi

# Main script execution
main
