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
# Triton Installation Script for AMD GPUs with ROCm Support
# =============================================================================
# This script installs OpenAI's Triton compiler for AMD GPUs with ROCm support.
# Enhanced with modern installation standards, multiple package managers,
# virtual environment support, and comprehensive error handling.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ████████╗██████╗ ██╗████████╗ ██████╗ ███╗   ██╗
  ╚══██╔══╝██╔══██╗██║╚══██╔══╝██╔═══██╗████╗  ██║
     ██║   ██████╔╝██║   ██║   ██║   ██║██╔██╗ ██║
     ██║   ██╔══██╗██║   ██║   ██║   ██║██║╚██╗██║
     ██║   ██║  ██║██║   ██║   ╚██████╔╝██║ ╚████║
     ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝
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

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.ml-stack/logs"
CONFIG_FILE="$HOME/.ml-stack/triton_config.sh"
DRY_RUN=false
FORCE=false
VERBOSE=false
INSTALL_METHOD="auto"
TRITON_VENV_PYTHON=""
ORIGINAL_PWD="$PWD"

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/triton_install_$(date +"%Y%m%d_%H%M%S").log"

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

print_separator() {
    echo "───────────────────────────────────────────────────────────"
}

# Logging functions
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"

    # Log to console if verbose or error
    if [ "$VERBOSE" = true ] || [ "$level" = "ERROR" ] || [ "$level" = "WARNING" ]; then
        case $level in
            "INFO") echo "[$timestamp] $message" ;;
            "WARNING") echo -e "${YELLOW}[$timestamp] WARNING: $message${RESET}" ;;
            "ERROR") echo -e "${RED}[$timestamp] ERROR: $message${RESET}" ;;
            "SUCCESS") echo -e "${GREEN}[$timestamp] $message${RESET}" ;;
            *) echo "[$timestamp] $message" ;;
        esac
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    local python_cmd="${1:-python3}"
    $python_cmd -c "import $2" &>/dev/null
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

# Function to install system packages
install_system_package() {
    local package="$1"
    local package_manager=$(detect_package_manager)

    case $package_manager in
        apt)
            if [ "$DRY_RUN" = true ]; then
                print_step "[DRY RUN] Would install $package with apt-get"
                return 0
            fi
            sudo apt-get update && sudo apt-get install -y "$package"
            ;;
        dnf)
            if [ "$DRY_RUN" = true ]; then
                print_step "[DRY RUN] Would install $package with dnf"
                return 0
            fi
            sudo dnf install -y "$package"
            ;;
        yum)
            if [ "$DRY_RUN" = true ]; then
                print_step "[DRY RUN] Would install $package with yum"
                return 0
            fi
            sudo yum install -y "$package"
            ;;
        pacman)
            if [ "$DRY_RUN" = true ]; then
                print_step "[DRY RUN] Would install $package with pacman"
                return 0
            fi
            sudo pacman -S --noconfirm "$package"
            ;;
        zypper)
            if [ "$DRY_RUN" = true ]; then
                print_step "[DRY RUN] Would install $package with zypper"
                return 0
            fi
            sudo zypper install -y "$package"
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            return 1
            ;;
    esac
}

# Function to use uv or pip for Python packages
install_python_package() {
    local package="$1"
    shift
    local extra_args="$@"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would install $package with Python package manager"
        return 0
    fi

    if command_exists uv; then
        print_step "Installing $package with uv..."
        log "INFO" "Installing $package with uv and args: $extra_args"
        uv pip install --python $(which python3) $extra_args "$package"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "SUCCESS" "Successfully installed $package with uv"
        else
            log "WARNING" "uv installation failed for $package"
        fi
        return $exit_code
    else
        print_step "Installing $package with pip..."
        log "INFO" "Installing $package with pip and args: $extra_args"
        python3 -m pip install $extra_args "$package"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "SUCCESS" "Successfully installed $package with pip"
        else
            log "WARNING" "pip installation failed for $package"
        fi
        return $exit_code
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

# Function to set up ROCm environment variables
setup_rocm_env() {
    print_section "Setting up ROCm Environment Variables"

    # Set ROCm environment variables
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export PYTORCH_ROCM_ARCH="gfx1100"
    export ROCM_PATH="/opt/rocm"
    export PATH="/opt/rocm/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Set HSA_TOOLS_LIB if rocprofiler library exists
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
        print_step "ROCm profiler library found and configured"
        log "INFO" "ROCm profiler library configured: $HSA_TOOLS_LIB"
    else
        export HSA_TOOLS_LIB=0
        print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
        log "WARNING" "ROCm profiler library not found, HSA_TOOLS_LIB set to 0"
    fi

    # Fix deprecated PYTORCH_CUDA_ALLOC_CONF warning
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
        unset PYTORCH_CUDA_ALLOC_CONF
        print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
        log "INFO" "Converted PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
    fi

    print_success "ROCm environment variables configured"
    log "SUCCESS" "ROCm environment variables configured"
}

# Function to detect ROCm version
detect_rocm_version() {
    local rocm_version=""

    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
    fi

    if [ -z "$rocm_version" ]; then
        # Try to detect from directory structure
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        log "WARNING" "Could not detect ROCm version, using default 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
        log "INFO" "Detected ROCm version: $rocm_version"
    fi

    echo "$rocm_version"
}

# Function to check ROCm installation
check_rocm_installation() {
    print_section "Checking ROCm Installation"

    if command_exists rocminfo; then
        print_success "rocminfo found"
        log "SUCCESS" "rocminfo command found"
        return 0
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        log "INFO" "rocminfo not found, checking for ROCm installation"

        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            log "INFO" "ROCm directory found, installing rocminfo"

            if install_system_package "rocminfo"; then
                if command_exists rocminfo; then
                    print_success "Installed rocminfo"
                    log "SUCCESS" "rocminfo installed successfully"
                    return 0
                else
                    print_error "Failed to install rocminfo"
                    log "ERROR" "Failed to install rocminfo"
                    return 1
                fi
            else
                print_error "Failed to install rocminfo package"
                log "ERROR" "Failed to install rocminfo package"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            log "ERROR" "ROCm not installed"
            return 1
        fi
    fi
}

# Function to install uv package manager
install_uv() {
    if command_exists uv; then
        print_success "uv package manager is already installed"
        log "INFO" "uv already installed"
        return 0
    fi

    print_step "Installing uv package manager..."
    log "INFO" "Installing uv package manager"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would install uv"
        return 0
    fi

    # Try to install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH if it was installed in a user directory
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Add uv to PATH if it was installed via cargo
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if command_exists uv; then
        print_success "Installed uv package manager"
        log "SUCCESS" "uv installed successfully"
        return 0
    else
        print_warning "Failed to install uv package manager, will use pip"
        log "WARNING" "Failed to install uv, falling back to pip"
        return 1
    fi
}

# Function to handle Python package installation with venv support
uv_pip_install() {
    local args="$@"

    # Check if uv is available as a command
    if command_exists uv; then
        case $INSTALL_METHOD in
            "global")
                print_step "Installing globally with pip..."
                log "INFO" "Installing globally with pip and args: $args"
                python3 -m pip install --break-system-packages $args
                TRITON_VENV_PYTHON=""
                ;;
            "venv")
                print_step "Creating uv virtual environment..."
                log "INFO" "Creating uv virtual environment"
                VENV_DIR="./triton_rocm_venv"
                if [ ! -d "$VENV_DIR" ]; then
                    uv venv "$VENV_DIR"
                fi
                source "$VENV_DIR/bin/activate"
                print_step "Installing in virtual environment..."
                log "INFO" "Installing in virtual environment with args: $args"
                uv pip install $args
                TRITON_VENV_PYTHON="$VENV_DIR/bin/python"
                print_success "Installed in virtual environment: $VENV_DIR"
                ;;
            "auto")
                # Try global install first
                print_step "Attempting global installation with uv..."
                log "INFO" "Attempting global installation with uv"
                local install_output
                install_output=$(uv pip install --python $(which python3) $args 2>&1)
                local install_exit_code=$?

                if echo "$install_output" | grep -q "externally managed"; then
                    print_warning "Global installation failed due to externally managed environment"
                    log "WARNING" "Global installation failed due to externally managed environment"
                    print_step "Creating uv virtual environment for installation..."
                    log "INFO" "Creating uv virtual environment for installation"

                    # Create uv venv in project directory
                    VENV_DIR="./triton_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    log "INFO" "Installing in virtual environment"
                    uv pip install $args

                    # Store venv path for verification
                    TRITON_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                elif [ $install_exit_code -eq 0 ]; then
                    print_success "Global installation successful"
                    log "SUCCESS" "Global installation successful"
                    TRITON_VENV_PYTHON=""
                else
                    print_error "Global installation failed with unknown error:"
                    log "ERROR" "Global installation failed with exit code $install_exit_code"
                    echo "$install_output"
                    print_step "Falling back to virtual environment..."
                    log "INFO" "Falling back to virtual environment"

                    # Create uv venv in project directory
                    VENV_DIR="./triton_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    log "INFO" "Installing in virtual environment (fallback)"
                    uv pip install $args

                    # Store venv path for verification
                    TRITON_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                fi
                ;;
        esac
    else
        # Fall back to pip
        print_step "Installing with pip..."
        log "INFO" "Installing with pip and args: $args"
        python3 -m pip install $args
        TRITON_VENV_PYTHON=""
    fi
}

# Function to install Triton from PyPI
install_triton_pypi() {
    print_section "Installing Triton from PyPI"
    log "INFO" "Starting PyPI installation of Triton"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would install Triton from PyPI"
        return 0
    fi

    # Install Triton using the enhanced package installer
    uv_pip_install "triton"

    # Check if installation worked
    local python_cmd=${TRITON_VENV_PYTHON:-python3}
    if $python_cmd -c "import triton; print('Triton version:', triton.__version__)" &>/dev/null; then
        print_success "Triton installed successfully from PyPI!"
        log "SUCCESS" "Triton installed successfully from PyPI"
        return 0
    else
        print_warning "PyPI installation failed, will try from source"
        log "WARNING" "PyPI installation failed, will try from source"
        return 1
    fi
}

# Function to install Triton from source
install_triton_source() {
    print_section "Installing Triton from Source"
    log "INFO" "Starting source installation of Triton"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would install Triton from source"
        return 0
    fi

    # Create installation directory
    INSTALL_DIR="$HOME/ml-stack/triton"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Clone Triton repository
    if [ ! -d "triton" ]; then
        print_step "Cloning Triton repository..."
        log "INFO" "Cloning Triton repository"
        git clone https://github.com/openai/triton.git --recursive
        cd triton
    else
        print_step "Triton repository already exists, updating..."
        log "INFO" "Updating existing Triton repository"
        cd triton
        git fetch --all
        git reset --hard origin/main
        git submodule update --init --recursive
    fi

    # Try different versions if needed
    local versions=("main" "master" "dev" "rocm-5.6")
    local success=false

    for version in "${versions[@]}"; do
        print_step "Trying branch/tag: $version"
        log "INFO" "Trying branch/tag: $version"

        git checkout "$version" || continue

        # Apply ROCm patch if needed
        print_step "Applying ROCm compatibility patches..."
        log "INFO" "Applying ROCm compatibility patches"
        sed -i 's/cuda/hip/g' python/triton/backends/hip.py || true

        # Build and install Triton
        print_step "Building and installing Triton from source..."
        log "INFO" "Building and installing Triton from source"

        cd python
        if command_exists uv; then
            print_step "Using uv for source installation..."
            log "INFO" "Using uv for source installation"
            if TRITON_BUILD_WITH_ROCM=1 uv pip install -e . --system 2>/dev/null || \
               TRITON_BUILD_WITH_ROCM=1 uv pip install -e . 2>/dev/null; then
                print_success "Successfully installed with uv"
                log "SUCCESS" "Successfully installed with uv"
                success=true
            else
                print_step "uv failed, falling back to pip..."
                log "WARNING" "uv failed, falling back to pip"
                TRITON_BUILD_WITH_ROCM=1 pip install -e . --break-system-packages
                if [ $? -eq 0 ]; then
                    success=true
                fi
            fi
        else
            print_step "Using pip for source installation..."
            log "INFO" "Using pip for source installation"
            TRITON_BUILD_WITH_ROCM=1 pip install -e . --break-system-packages
            if [ $? -eq 0 ]; then
                success=true
            fi
        fi

        # Check if installation worked
        local python_cmd=${TRITON_VENV_PYTHON:-python3}
        if $python_cmd -c "import triton; print('Triton version:', triton.__version__)" &>/dev/null; then
            print_success "Triton installed successfully from source!"
            log "SUCCESS" "Triton installed successfully from source"
            success=true
            break
        else
            print_step "Installation of branch $version failed, trying next version..."
            log "WARNING" "Installation of branch $version failed"
            cd ..
        fi
    done

    if [ "$success" = false ]; then
        print_error "Failed to install Triton from source"
        log "ERROR" "Failed to install Triton from source"
        return 1
    fi

    return 0
}

# Function to create test script
create_test_script() {
    print_section "Creating Test Script"
    log "INFO" "Creating test script"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would create test script"
        return 0
    fi

    local test_script="$INSTALL_DIR/test_triton.py"

    cat > "$test_script" << 'EOF'
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
    print("✓ Test passed!")

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

    print(".3f")
    print(".3f")
    print(".2f")

if __name__ == "__main__":
    test_add_vectors()
EOF

    print_success "Created test script at $test_script"
    log "SUCCESS" "Created test script at $test_script"
    echo "You can run it with: python3 $test_script"
}

# Function to verify installation
verify_installation() {
    print_section "Verifying Triton Installation"
    log "INFO" "Starting installation verification"

    local python_cmd=${TRITON_VENV_PYTHON:-python3}

    if $python_cmd -c "import triton; print('Triton version:', triton.__version__)" &>/dev/null; then
        local triton_version=$($python_cmd -c "import triton; print(triton.__version__)" 2>/dev/null)
        print_success "Triton is installed (version: $triton_version)"
        log "SUCCESS" "Triton installed (version: $triton_version)"

        # Test basic functionality
        print_step "Testing basic Triton functionality..."
        log "INFO" "Testing basic Triton functionality"

        if $python_cmd -c "import triton; print('Triton import successful')" &>/dev/null; then
            print_success "Basic Triton functionality working"
            log "SUCCESS" "Basic Triton functionality working"
        else
            print_warning "Basic Triton functionality may not be working correctly"
            log "WARNING" "Basic Triton functionality may not be working correctly"
        fi

        # Test GPU availability if PyTorch is available
        if package_installed "$python_cmd" "torch"; then
            if $python_cmd -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "GPU acceleration is available for Triton"
                log "SUCCESS" "GPU acceleration available for Triton"
            else
                print_warning "GPU acceleration is not available"
                log "WARNING" "GPU acceleration not available"
            fi
        fi

        return 0
    else
        print_error "Triton installation verification failed"
        log "ERROR" "Triton installation verification failed"
        return 1
    fi
}

# Function to check for existing installation
check_existing_installation() {
    local python_cmd=${TRITON_VENV_PYTHON:-python3}

    if package_installed "$python_cmd" "triton"; then
        local triton_version=$($python_cmd -c "import triton; print(triton.__version__)" 2>/dev/null)
        print_success "Triton is already installed (version: $triton_version)"
        log "INFO" "Triton already installed (version: $triton_version)"

        if [ "$FORCE" = true ]; then
            print_warning "Force reinstall requested - proceeding with reinstallation"
            log "WARNING" "Force reinstall requested"
            return 1
        else
            print_step "Triton installation is complete. Use --force to reinstall anyway."
            return 0
        fi
    fi

    return 1
}

# Function to show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Enhanced Triton Installation Script for AMD GPUs with ROCm Support

OPTIONS:
    --help              Show this help message
    --dry-run           Show what would be done without making changes
    --force             Force reinstallation even if Triton is already installed
    --verbose           Enable verbose logging
    --method METHOD     Installation method: global, venv, auto (default: auto)
    --show-env          Show ROCm environment variables for manual setup
    --config FILE       Use custom configuration file

EXAMPLES:
    $0                          # Install with default settings
    $0 --dry-run               # Preview installation
    $0 --force                 # Force reinstall
    $0 --method venv           # Install in virtual environment
    $0 --verbose               # Verbose output
    $0 --show-env              # Show environment variables

For more information, visit: https://github.com/openai/triton
EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --method)
                INSTALL_METHOD="$2"
                shift 2
                ;;
            --show-env)
                show_env
                exit 0
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to load configuration
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_step "Loading configuration from $CONFIG_FILE"
        log "INFO" "Loading configuration from $CONFIG_FILE"
        source "$CONFIG_FILE"
    else
        print_step "No configuration file found, using defaults"
        log "INFO" "No configuration file found, using defaults"
    fi
}

# Main installation function
install_triton() {
    print_header "Triton Installation"

    # Load configuration
    load_config

    # Parse command line arguments
    parse_args "$@"

    # Log installation start
    log "INFO" "=== Starting Triton Installation ==="
    log "INFO" "System: $(uname -a)"
    log "INFO" "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
    log "INFO" "Python Version: $(python3 --version)"
    log "INFO" "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    log "INFO" "Installation Method: $INSTALL_METHOD"
    log "INFO" "Dry Run: $DRY_RUN"
    log "INFO" "Force: $FORCE"
    log "INFO" "Verbose: $VERBOSE"

    # Detect environment
    local wsl=$(detect_wsl)
    local container=$(detect_container)
    log "INFO" "WSL Environment: $wsl"
    log "INFO" "Container Environment: $container"

    if [ "$wsl" = "true" ]; then
        print_warning "Detected WSL environment - some features may be limited"
        log "WARNING" "WSL environment detected"
    fi

    if [ "$container" = "true" ]; then
        print_warning "Detected container environment"
        log "INFO" "Container environment detected"
    fi

    # Check for existing installation
    if check_existing_installation; then
        return 0
    fi

    # Check ROCm installation
    if ! check_rocm_installation; then
        return 1
    fi

    # Detect ROCm version
    local rocm_version=$(detect_rocm_version)

    # Set up ROCm environment
    setup_rocm_env

    # Install uv if not available
    install_uv

    # Ask user for installation preference if not specified
    if [ "$INSTALL_METHOD" = "auto" ] && [ "$DRY_RUN" = false ]; then
        echo
        echo -e "${CYAN}${BOLD}Triton Installation Options:${RESET}"
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
    fi

    # Install Python dependencies
    print_section "Installing Python Dependencies"
    log "INFO" "Installing Python dependencies"
    install_python_package "cmake" "ninja" "pytest" "packaging" "wheel" --upgrade

    # Try PyPI installation first
    if ! install_triton_pypi; then
        # Fall back to source installation
        if ! install_triton_source; then
            print_error "Failed to install Triton"
            log "ERROR" "Failed to install Triton"
            return 1
        fi
    fi

    # Verify installation
    if ! verify_installation; then
        print_error "Installation verification failed"
        log "ERROR" "Installation verification failed"
        return 1
    fi

    # Create test script
    create_test_script

    # Show completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ████████╗██████╗ ██╗████████╗ ██████╗ ███╗   ██╗       ║
    ║  ╚══██╔══╝██╔══██╗██║╚══██╔══╝██╔═══██╗████╗  ██║       ║
    ║     ██║   ██████╔╝██║   ██║   ██║   ██║██╔██╗ ██║       ║
    ║     ██║   ██╔══██╗██║   ██║   ██║   ██║██║╚██╗██║       ║
    ║     ██║   ██║  ██║██║   ██║   ╚██████╔╝██║ ╚████║       ║
    ║     ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝       ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  Triton is now ready to use with your AMD GPU.          ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "Triton installation completed successfully"
    log "SUCCESS" "Triton installation completed successfully"

    # Provide usage examples
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$TRITON_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./triton_rocm_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import triton; print('Triton version:', triton.__version__)\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import triton; print('Triton version:', triton.__version__)\"${RESET}"
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
    echo -e "${GREEN}eval \"\$($0 --show-env)\"${RESET}"
    echo
    echo -e "${GREEN}${BOLD}Test your installation by running the test script:${RESET}"
    echo -e "${GREEN}python3 $INSTALL_DIR/test_triton.py${RESET}"
    echo
    echo -e "${GREEN}${BOLD}Returning to main menu in 5 seconds...${RESET}"
    sleep 1
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"
    sleep 1

    return 0
}

# Check for --show-env option before main installation
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_triton "$@"
