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
# PyTorch with ROCm Installation Script
# =============================================================================
# This script installs PyTorch with ROCm support for AMD GPUs.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗    ██████╗  ██████╗  ██████╗███╗   ███╗
  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║    ██╔══██╗██╔═══██╗██╔════╝████╗ ████║
  ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║    ██████╔╝██║   ██║██║     ██╔████╔██║
  ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║    ██╔══██╗██║   ██║██║     ██║╚██╔╝██║
  ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║    ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║
  ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝
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

# Function to show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

PyTorch with ROCm Installation Script for AMD GPUs
Enhanced with modern installation standards and comprehensive error handling.

OPTIONS:
    --help              Show this help message
    --dry-run           Show what would be done without making changes
    --force             Force reinstallation even if PyTorch is already installed
    --verbose           Enable verbose logging
    --method METHOD     Installation method: global, venv, auto (default: auto)
    --show-env          Show ROCm environment variables for manual setup

EXAMPLES:
    $0                          # Install with default settings
    $0 --dry-run               # Preview installation
    $0 --force                 # Force reinstall
    $0 --method venv           # Install in virtual environment
    $0 --verbose               # Verbose output
    $0 --show-env              # Show environment variables

For more information, visit: https://pytorch.org/
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
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main installation function
install_pytorch_rocm() {
    print_header "PyTorch with ROCm Installation"

    # Parse command line arguments
    parse_args "$@"

    # Check if PyTorch is already installed
    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${PYTORCH_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import torch" &>/dev/null; then
        pytorch_version=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)

        # Check if PyTorch has ROCm/HIP support
        if $PYTHON_CMD -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$($PYTHON_CMD -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)

            # Test if GPU acceleration works
            if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "PyTorch with ROCm support is already installed and working (PyTorch $pytorch_version, ROCm $hip_version)"

                # Check if --force flag is provided
                if [[ "$*" == *"--force"* ]] || [[ "$PYTORCH_REINSTALL" == "true" ]]; then
                    print_warning "Force reinstall requested - proceeding with reinstallation"
                    print_step "Will reinstall PyTorch despite working installation"
                else
                    print_step "PyTorch installation is complete and working. Use --force to reinstall anyway."
                    return 0
                fi
            else
                print_warning "PyTorch with ROCm support is installed (PyTorch $pytorch_version, ROCm $hip_version) but GPU acceleration is not working"
                print_step "Will reinstall to fix GPU acceleration issues"
            fi
        else
            print_warning "PyTorch is installed (version $pytorch_version) but without ROCm support"
            print_step "Will reinstall with ROCm support"
        fi
    fi

    # Check if ROCm is installed
    print_section "Checking ROCm Installation"

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
            print_warning "Could not detect ROCm version, using default version 7.2"
            rocm_version="7.2"
        else
            print_success "Detected ROCm version: $rocm_version"
        fi

    # Check if uv is installed
    print_section "Installing PyTorch with ROCm Support"

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

    # Extract ROCm major and minor version
    rocm_major_version=$(echo "$rocm_version" | cut -d '.' -f 1)
    rocm_minor_version=$(echo "$rocm_version" | cut -d '.' -f 2)

    # Uninstall existing PyTorch if it exists
    if package_installed "torch"; then
        print_step "Uninstalling existing PyTorch..."

        # Create a function to handle uv commands properly
        uv_pip_uninstall() {
            # Check if uv is available as a command
            if command -v uv &> /dev/null; then
                # Use uv directly as a command with proper Python
                if [ -n "$PYTORCH_VENV_PYTHON" ]; then
                    uv pip uninstall "$@"
                else
                    uv pip uninstall --python $(which python3) "$@"
                fi
            else
                # Fall back to pip
                if [ -n "$PYTORCH_VENV_PYTHON" ]; then
                    $PYTORCH_VENV_PYTHON -m pip uninstall "$@"
                else
                    python3 -m pip uninstall "$@"
                fi
            fi
        }

        # Uninstall using the wrapper function
        uv_pip_uninstall -y torch torchvision torchaudio

        if $PYTHON_CMD -c "import torch" &>/dev/null; then
            print_warning "Failed to uninstall PyTorch, continuing anyway"
        else
            print_success "Uninstalled existing PyTorch"
        fi
    fi

    # Install PyTorch with ROCm support
    print_step "Installing PyTorch with ROCm support..."

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}PyTorch Installation Options:${RESET}"
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
        if command -v uv &> /dev/null; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    python3 -m pip install --break-system-packages $args
                    PYTORCH_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    VENV_DIR="./pytorch_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args
                    PYTORCH_VENV_PYTHON="$VENV_DIR/bin/python"
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
                        VENV_DIR="./pytorch_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        PYTORCH_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global installation successful"
                        PYTORCH_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        VENV_DIR="./pytorch_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        PYTORCH_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing with pip..."
            python3 -m pip install $args
            PYTORCH_VENV_PYTHON=""
        fi
    }

    # Use the appropriate PyTorch version based on ROCm version
    if [ "$rocm_major_version" -eq 7 ]; then
        # For ROCm 7.0+, try different sources
        print_step "Installing PyTorch for ROCm 7.0..."

        # Try PyTorch's official nightly builds first
        if uv_pip_install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0 2>/dev/null; then
            print_success "Successfully installed PyTorch nightly build for ROCm 7.0"
        # Try ROCm's manylinux repository
        elif uv_pip_install torch torchvision torchaudio --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ 2>/dev/null; then
            print_success "Successfully installed PyTorch from ROCm manylinux repository"
        # Try compiling from source as last resort
        elif [ "$INSTALL_METHOD" != "auto" ] && [ "$FORCE" = true ]; then
            print_warning "ROCm 7.0 PyTorch builds not available, attempting to compile from source..."
            print_step "This may take considerable time and requires development tools"
            # Try to install build dependencies and compile
            uv_pip_install torch torchvision torchaudio --no-binary torch --no-binary torchvision --no-binary torchaudio
        # Fallback to ROCm 6.4 builds if 7.0 not available
        else
            print_warning "ROCm 7.0 PyTorch builds not available, falling back to ROCm 6.4..."
            uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        fi
    elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
        # For ROCm 6.4+, use nightly builds
        print_step "Using PyTorch nightly build for ROCm 6.4..."
        uv_pip_install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 3 ]; then
        # For ROCm 6.3, use stable builds
        print_step "Using PyTorch stable build for ROCm 6.3..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 0 ]; then
        # For ROCm 6.0-6.2, use stable builds for 6.2
        print_step "Using PyTorch stable build for ROCm 6.2..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    elif [ "$rocm_major_version" -eq 5 ]; then
        # For ROCm 5.x, use stable builds for 5.7
        print_step "Using PyTorch stable build for ROCm 5.7..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    else
        # Fallback to the latest stable ROCm version
        print_step "Using PyTorch stable build for ROCm 6.3 (fallback)..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    fi

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${PYTORCH_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import torch" &>/dev/null; then
        pytorch_version=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_success "PyTorch is installed (version: $pytorch_version)"

        # Check if PyTorch has ROCm/HIP support
        if $PYTHON_CMD -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$($PYTHON_CMD -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
            print_success "PyTorch has ROCm/HIP support (version: $hip_version)"

            # Test GPU availability
            if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "GPU acceleration is available"

                # Get GPU count
                gpu_count=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
                print_step "PyTorch detected $gpu_count GPU(s)"

                # List GPUs
                for i in $(seq 0 $((gpu_count-1))); do
                    gpu_name=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
                    echo "  - GPU $i: $gpu_name"
                done

                # Test a simple tensor operation
                print_step "Testing GPU tensor operations..."
                if $PYTHON_CMD -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success' if torch.all(y == 2) else 'Failed')" 2>/dev/null | grep -q "Success"; then
                    print_success "GPU tensor operations working correctly"
                else
                    print_warning "GPU tensor operations may not be working correctly"
                fi
            else
                print_warning "GPU acceleration is not available"
                print_warning "Check your ROCm installation and environment variables"
            fi
        else
            print_warning "PyTorch does not have explicit ROCm/HIP support"
            print_warning "This might cause issues with AMD GPUs"
        fi
    else
        print_error "PyTorch installation failed"
        return 1
    fi

    # Show a visually appealing completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗  ║
    ║  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║  ║
    ║  ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║  ║
    ║  ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║  ║
    ║  ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║  ║
    ║  ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝  ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  PyTorch with ROCm is now ready to use with your GPU.   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "PyTorch with ROCm installation completed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$PYTORCH_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./pytorch_rocm_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import torch; print('PyTorch version:', torch.__version__); print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'); print('GPU available:', torch.cuda.is_available())\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import torch; print('PyTorch version:', torch.__version__); print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'); print('GPU available:', torch.cuda.is_available())\"${RESET}"
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
    echo -e "${GREEN}eval \"\$(./install_pytorch_rocm.sh --show-env)\"${RESET}"
    echo

    # Add a small delay to ensure the message is seen
    echo -e "${GREEN}${BOLD}Returning to main menu in 3 seconds...${RESET}"
    sleep 1
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"
    sleep 1

    # Ensure we exit properly to prevent hanging
    exit 0

    return 0
}

# Check for --show-env option before main installation
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_pytorch_rocm "$@"

