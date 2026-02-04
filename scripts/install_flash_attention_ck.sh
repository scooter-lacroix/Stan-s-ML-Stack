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
# Enhanced with modern installation standards including:
# - Multi-package-manager support
# - Virtual environment integration
# - Enhanced error handling and recovery
# - Cross-platform compatibility
# - Interactive installation options
#
# Date: $(date +"%Y-%m-%d")
# =============================================================================

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

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

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
        uv pip install --break-system-packages $extra_args "$package"
    else
        print_step "Installing $package with pip..."
        python3 -m pip install --break-system-packages $extra_args "$package"
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

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local percentage=$((current * 100 / total))
    local progress_bar=""

    # Create progress bar
    for ((i=0; i<percentage/2; i++)); do
        progress_bar+="█"
    done
    for ((i=percentage/2; i<50; i++)); do
        progress_bar+="░"
    done

    echo -ne "\r${CYAN}Progress: [${progress_bar}] ${percentage}% - ${description}${RESET}"
}

# Function to complete progress
complete_progress() {
    echo -e "\r${GREEN}Progress: [$(printf '█%.0s' {1..50})] 100% - Complete!${RESET}"
    echo
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

# Main installation function
install_flash_attention_ck() {
    print_header "Flash Attention CK Installation"

    local total_steps=10
    local current_step=0

    show_progress $current_step $total_steps "Initializing installation..."
    sleep 0.5

    # Check if Flash Attention CK is already installed
    ((current_step++))
    show_progress $current_step $total_steps "Checking existing installation..."
    # Use venv Python if available, otherwise installer-selected python
    PYTHON_CMD=${FLASH_ATTENTION_VENV_PYTHON:-$PYTHON_BIN}

    if $PYTHON_CMD -c "import flash_attention_amd" &>/dev/null; then
        flash_attention_version=$($PYTHON_CMD -c "import flash_attention_amd; print(getattr(flash_attention_amd, '__version__', 'Unknown'))" 2>/dev/null)

        # Check if it's working properly
        if $PYTHON_CMD -c "import flash_attention_amd; from flash_attention_amd import FlashAttention; print('Working')" 2>/dev/null | grep -q "Working"; then
            print_success "Flash Attention CK is already installed and working (version: $flash_attention_version)"

            # Check if --force flag is provided
            if [[ "$*" == *"--force"* ]] || [[ "$FLASH_ATTENTION_REINSTALL" == "true" ]]; then
                print_warning "Force reinstall requested - proceeding with reinstallation"
                print_step "Will reinstall Flash Attention CK despite working installation"
            else
                print_step "Flash Attention CK installation is complete and working. Use --force to reinstall anyway."
                return 0
            fi
        else
            print_warning "Flash Attention CK is installed but not working properly"
            print_step "Will reinstall to fix issues"
        fi
    fi

    # Check ROCm installation
    ((current_step++))
    show_progress $current_step $total_steps "Checking ROCm installation..."
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
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    print_step "System: $(uname -a)"
    print_step "Python Version: $(python3 --version)"
    pytorch_version=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')
    print_step "PyTorch Version: $pytorch_version"

    # Check if uv is installed
    ((current_step++))
    show_progress $current_step $total_steps "Installing dependencies..."
    print_section "Installing Dependencies"

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

    # Check for required dependencies
    print_step "Checking system dependencies..."

    # Fix ninja-build detection
    if command_exists ninja && ! command_exists ninja-build; then
        print_step "Creating symlink for ninja-build..."
        sudo ln -sf $(which ninja) /usr/bin/ninja-build
        print_success "Ninja-build symlink created."
    elif command_exists ninja-build; then
        print_success "Ninja-build already available."
    else
        print_step "Installing ninja-build..."
        package_manager=$(detect_package_manager)
        case $package_manager in
            apt)
                sudo apt-get update && sudo apt-get install -y ninja-build
                ;;
            dnf)
                sudo dnf install -y ninja-build
                ;;
            yum)
                sudo yum install -y ninja-build
                ;;
            pacman)
                sudo pacman -S ninja-build
                ;;
            zypper)
                sudo zypper install -y ninja-build
                ;;
            *)
                print_error "Unsupported package manager: $package_manager"
                return 1
                ;;
        esac
        if command_exists ninja-build; then
            print_success "Installed ninja-build"
        else
            print_error "Failed to install ninja-build"
            return 1
        fi
    fi

    # Refresh PATH to include newly installed ninja-build
    export PATH=$PATH:/usr/bin

    DEPS=("git" "python3" "cmake" "ninja-build")
    MISSING_DEPS=()

    for dep in "${DEPS[@]}"; do
        if ! command_exists $dep; then
            MISSING_DEPS+=("$dep")
        fi
    done

    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${MISSING_DEPS[*]}"
        print_step "Attempting to install missing dependencies..."

        package_manager=$(detect_package_manager)
        case $package_manager in
            apt)
                sudo apt-get update && sudo apt-get install -y "${MISSING_DEPS[@]}"
                ;;
            dnf)
                sudo dnf install -y "${MISSING_DEPS[@]}"
                ;;
            yum)
                sudo yum install -y "${MISSING_DEPS[@]}"
                ;;
            pacman)
                sudo pacman -S "${MISSING_DEPS[@]}"
                ;;
            zypper)
                sudo zypper install -y "${MISSING_DEPS[@]}"
                ;;
            *)
                print_error "Unsupported package manager: $package_manager"
                print_step "Please install dependencies manually and run this script again."
                return 1
                ;;
        esac

        # Verify installation
        STILL_MISSING=()
        for dep in "${MISSING_DEPS[@]}"; do
            if ! command_exists $dep; then
                STILL_MISSING+=("$dep")
            fi
        done

        if [ ${#STILL_MISSING[@]} -ne 0 ]; then
            print_error "Failed to install dependencies: ${STILL_MISSING[*]}"
            return 1
        else
            print_success "All dependencies installed successfully"
        fi
    else
        print_success "All system dependencies are available"
    fi

    # Check for Python development headers
    print_step "Checking for Python development headers..."
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    SYSTEM_PYTHON_INCLUDE_DIR="/usr/include/python${PYTHON_VERSION}"

    # For virtual environments created with uv, we need to create a symlink to Python.h
    PYTHON_INCLUDE_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')

    if [ -d "$SYSTEM_PYTHON_INCLUDE_DIR" ]; then
        print_success "Found system Python include directory: $SYSTEM_PYTHON_INCLUDE_DIR"

        if [ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]; then
            print_step "Creating symlink for Python.h in virtual environment..."
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
            print_success "Python headers symlinked successfully"
        else
            print_success "Python.h already exists in virtual environment"
        fi
    else
        print_warning "System Python include directory not found. Python development headers may be missing."
        print_step "Attempting to install Python development headers..."

        package_manager=$(detect_package_manager)
        case $package_manager in
            apt)
                sudo apt-get update && sudo apt-get install -y "python${PYTHON_VERSION}-dev"
                ;;
            dnf)
                sudo dnf install -y "python${PYTHON_VERSION}-devel"
                ;;
            yum)
                sudo yum install -y "python${PYTHON_VERSION}-devel"
                ;;
            pacman)
                sudo pacman -S "python"
                ;;
            zypper)
                sudo zypper install -y "python${PYTHON_VERSION}-devel"
                ;;
            *)
                print_error "Unsupported package manager: $package_manager"
                print_step "Please install python${PYTHON_VERSION}-dev package manually."
                return 1
                ;;
        esac

        # Verify installation
        if [ -d "$SYSTEM_PYTHON_INCLUDE_DIR" ]; then
            print_success "Python development headers installed successfully"
        else
            print_error "Failed to install Python development headers"
            return 1
        fi
    fi

    # Check if PyTorch is installed
    ((current_step++))
    show_progress $current_step $total_steps "Checking PyTorch installation..."
    print_section "Checking PyTorch Installation"

    if ! package_installed "torch"; then
        print_warning "PyTorch is not installed"
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
                PYTORCH_INSTALL_METHOD="global"
                print_step "Using global installation method"
                ;;
            2)
                PYTORCH_INSTALL_METHOD="venv"
                print_step "Using virtual environment method"
                ;;
            3|*)
                PYTORCH_INSTALL_METHOD="auto"
                print_step "Using auto-detect method"
                ;;
        esac

        # Create a function to handle PyTorch installation with venv fallback
        install_pytorch_with_method() {
            local args="$@"

            # Check if uv is available as a command
            if command_exists uv; then
                case $PYTORCH_INSTALL_METHOD in
                    "global")
                        print_step "Installing PyTorch globally with pip..."
                        python3 -m pip install --break-system-packages $args
                        PYTORCH_VENV_PYTHON=""
                        ;;
                    "venv")
                        print_step "Creating uv virtual environment for PyTorch..."
                        PYTORCH_VENV_DIR="./pytorch_rocm_venv"
                        if [ ! -d "$PYTORCH_VENV_DIR" ]; then
                            uv venv "$PYTORCH_VENV_DIR"
                        fi
                        source "$PYTORCH_VENV_DIR/bin/activate"
                        print_step "Installing PyTorch in virtual environment..."
                        uv pip install $args
                        PYTORCH_VENV_PYTHON="$PYTORCH_VENV_DIR/bin/python"
                        print_success "PyTorch installed in virtual environment: $PYTORCH_VENV_DIR"
                        ;;
                    "auto")
                        # Try global install first
                        print_step "Attempting global PyTorch installation with uv..."
                        local install_output
                        install_output=$(uv pip install --python $(which python3) $args 2>&1)
                        local install_exit_code=$?

                        if echo "$install_output" | grep -q "externally managed"; then
                            print_warning "Global installation failed due to externally managed environment"
                            print_step "Creating uv virtual environment for PyTorch installation..."

                            # Create uv venv in project directory
                            PYTORCH_VENV_DIR="./pytorch_rocm_venv"
                            if [ ! -d "$PYTORCH_VENV_DIR" ]; then
                                uv venv "$PYTORCH_VENV_DIR"
                            fi

                            # Activate venv and install
                            source "$PYTORCH_VENV_DIR/bin/activate"
                            print_step "Installing PyTorch in virtual environment..."
                            uv pip install $args

                            # Store venv path for verification
                            PYTORCH_VENV_PYTHON="$PYTORCH_VENV_DIR/bin/python"
                            print_success "PyTorch installed in virtual environment: $PYTORCH_VENV_DIR"
                        elif [ $install_exit_code -eq 0 ]; then
                            print_success "Global PyTorch installation successful"
                            PYTORCH_VENV_PYTHON=""
                        else
                            print_error "Global installation failed with unknown error:"
                            echo "$install_output"
                            print_step "Falling back to virtual environment..."

                            # Create uv venv in project directory
                            PYTORCH_VENV_DIR="./pytorch_rocm_venv"
                            if [ ! -d "$PYTORCH_VENV_DIR" ]; then
                                uv venv "$PYTORCH_VENV_DIR"
                            fi

                            # Activate venv and install
                            source "$PYTORCH_VENV_DIR/bin/activate"
                            print_step "Installing PyTorch in virtual environment..."
                            uv pip install $args

                            # Store venv path for verification
                            PYTORCH_VENV_PYTHON="$PYTORCH_VENV_DIR/bin/python"
                            print_success "PyTorch installed in virtual environment: $PYTORCH_VENV_DIR"
                        fi
                        ;;
                esac
            else
                # Fall back to pip
                print_step "Installing PyTorch with pip..."
                python3 -m pip install $args
                PYTORCH_VENV_PYTHON=""
            fi
        }

        # Use the appropriate PyTorch version based on ROCm version
        rocm_major_version=$(echo "$rocm_version" | cut -d '.' -f 1)
        rocm_minor_version=$(echo "$rocm_version" | cut -d '.' -f 2)

        if [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
            # For ROCm 6.4+, use nightly builds
            print_step "Using PyTorch nightly build for ROCm 6.4..."
            install_pytorch_with_method --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 3 ]; then
            # For ROCm 6.3, use stable builds
            print_step "Using PyTorch stable build for ROCm 6.3..."
            install_pytorch_with_method torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 0 ]; then
            # For ROCm 6.0-6.2, use stable builds for 6.2
            print_step "Using PyTorch stable build for ROCm 6.2..."
            install_pytorch_with_method torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
        elif [ "$rocm_major_version" -eq 5 ]; then
            # For ROCm 5.x, use stable builds for 5.7
            print_step "Using PyTorch stable build for ROCm 5.7..."
            install_pytorch_with_method torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
        else
            # Fallback to the latest stable ROCm version
            print_step "Using PyTorch stable build for ROCm 6.3 (fallback)..."
            install_pytorch_with_method torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        fi

        # Verify PyTorch installation
        PYTHON_CMD=${PYTORCH_VENV_PYTHON:-python3}

        if $PYTHON_CMD -c "import torch" &>/dev/null; then
            pytorch_version=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
            print_success "PyTorch is installed (version: $pytorch_version)"

            # Check if PyTorch has ROCm/HIP support
            if $PYTHON_CMD -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
                hip_version=$($PYTHON_CMD -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
                print_success "PyTorch has ROCm/HIP support (version: $hip_version)"
            else
                print_warning "PyTorch does not have explicit ROCm/HIP support"
            fi
        else
            print_error "PyTorch installation failed"
            return 1
        fi
    else
        print_success "PyTorch is already installed"
        PYTHON_CMD=${PYTORCH_VENV_PYTHON:-python3}
    fi

    # Ask user for Flash Attention CK installation preference
    ((current_step++))
    show_progress $current_step $total_steps "Setting up Flash Attention CK..."
    print_section "Flash Attention CK Installation Options"

    echo
    echo -e "${CYAN}${BOLD}Flash Attention CK Installation Options:${RESET}"
    echo "1) Global installation (recommended for system-wide use)"
    echo "2) Virtual environment (isolated installation)"
    echo "3) Auto-detect (try global, fallback to venv if needed)"
    echo
    read -p "Choose installation method (1-3) [3]: " FLASH_ATTENTION_CHOICE
    FLASH_ATTENTION_CHOICE=${FLASH_ATTENTION_CHOICE:-3}

    case $FLASH_ATTENTION_CHOICE in
        1)
            FLASH_ATTENTION_INSTALL_METHOD="global"
            print_step "Using global installation method for Flash Attention CK"
            ;;
        2)
            FLASH_ATTENTION_INSTALL_METHOD="venv"
            print_step "Using virtual environment method for Flash Attention CK"
            ;;
        3|*)
            FLASH_ATTENTION_INSTALL_METHOD="auto"
            print_step "Using auto-detect method for Flash Attention CK"
            ;;
    esac

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

    print_step "Installation directory: $INSTALL_DIR"
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
((current_step++))
show_progress $current_step $total_steps "Building Flash Attention CK..."
print_step "Building Flash Attention CK implementation..."
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
    print_step "Detecting GPU architecture..."

    if [ -f "/usr/bin/amdgpu-arch" ] || [ -f "$ROCM_PATH/lib/llvm/bin/amdgpu-arch" ]; then
        DETECTED_ARCH=$($ROCM_PATH/lib/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null)
        if [ -n "$DETECTED_ARCH" ]; then
            # Remove any newlines from the architecture string
            GPU_ARCH=$(echo "$DETECTED_ARCH" | tr -d '\n')
            print_success "Detected GPU architecture: $GPU_ARCH"
        else
            print_warning "Could not detect GPU architecture, using default: $GPU_ARCH"
        fi
    else
        print_warning "amdgpu-arch tool not found, using default GPU architecture: $GPU_ARCH"
        print_step "Attempting to install amdgpu-arch..."

        package_manager=$(detect_package_manager)
        case $package_manager in
            apt)
                sudo apt-get update && sudo apt-get install -y llvm-amdgpu
                ;;
            dnf)
                sudo dnf install -y llvm-amdgpu
                ;;
            yum)
                sudo yum install -y llvm-amdgpu
                ;;
            pacman)
                sudo pacman -S llvm-amdgpu
                ;;
            zypper)
                sudo zypper install -y llvm-amdgpu
                ;;
            *)
                print_error "Unsupported package manager: $package_manager"
                ;;
        esac

        if [ -f "/usr/bin/amdgpu-arch" ] || [ -f "$ROCM_PATH/lib/llvm/bin/amdgpu-arch" ]; then
            DETECTED_ARCH=$($ROCM_PATH/lib/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null)
            if [ -n "$DETECTED_ARCH" ]; then
                GPU_ARCH=$(echo "$DETECTED_ARCH" | tr -d '\n')
                print_success "Successfully detected GPU architecture: $GPU_ARCH"
            fi
        fi
    fi

    # Configure CMake with enhanced error handling
    print_step "Configuring CMake..."

    CMAKE_ARGS=(
        -DCMAKE_PREFIX_PATH=$($PYTHON_CMD -c "import torch; print(torch.utils.cmake_prefix_path)")
        -DCMAKE_BUILD_TYPE=Release
        -DGPU_TARGETS="$GPU_ARCH"
        -DCMAKE_CXX_FLAGS="-Wno-error"
    )

    # Add ROCm path if available
    if [ -n "$ROCM_PATH" ]; then
        CMAKE_ARGS+=(-DROCM_PATH="$ROCM_PATH")
    fi

    # Configure CMake with retry mechanism
    MAX_RETRIES=3
    RETRY_COUNT=0
    CMAKE_SUCCESS=false

    while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$CMAKE_SUCCESS" = false ]; do
        if cmake .. "${CMAKE_ARGS[@]}"; then
            CMAKE_SUCCESS=true
            print_success "CMake configuration successful"
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                print_warning "CMake configuration failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
                sleep 2
            else
                print_error "CMake configuration failed after $MAX_RETRIES attempts"
                return 1
            fi
        fi
    done

    # Build with progress indication
    print_step "Building Flash Attention CK..."
    if cmake --build . --config Release -j $(nproc); then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        return 1
    fi

    # Install and verify library
    print_step "Installing library..."

    # Check if the library was built successfully
    if [ -f "flash_attention_amd_cuda.so" ]; then
        print_success "Library built successfully in build directory"
        cp flash_attention_amd_cuda.so ..
    elif [ -f "../flash_attention_amd_cuda.so" ]; then
        print_success "Library already in the correct location"
    else
        # Try to find the library
        LIBRARY_PATH=$(find . -name "flash_attention_amd_cuda.so" 2>/dev/null)
        if [ -n "$LIBRARY_PATH" ]; then
            print_success "Found library at $LIBRARY_PATH"
            cp "$LIBRARY_PATH" ..
        else
            print_warning "Library not found in build directory, checking for alternative names..."
            # Try alternative library names
            ALT_LIBRARY=$(find . -name "*flash_attention*.so" 2>/dev/null | head -n 1)
            if [ -n "$ALT_LIBRARY" ]; then
                print_success "Found alternative library: $ALT_LIBRARY"
                cp "$ALT_LIBRARY" ../flash_attention_amd_cuda.so
            else
                print_error "Could not find Flash Attention library"
                return 1
            fi
        fi
    fi

    # Install the Python package with enhanced method support
    print_section "Installing Python Package"

    cd $INSTALL_DIR

    # Create a function to handle Flash Attention installation with venv fallback
    install_flash_attention_with_method() {
        local args="$@"

        # Check if uv is available as a command
        if command_exists uv; then
            case $FLASH_ATTENTION_INSTALL_METHOD in
                "global")
                    print_step "Installing Flash Attention CK globally with pip..."
                    python3 -m pip install --break-system-packages $args
                    FLASH_ATTENTION_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment for Flash Attention CK..."
                    FLASH_ATTENTION_VENV_DIR="./flash_attention_venv"
                    if [ ! -d "$FLASH_ATTENTION_VENV_DIR" ]; then
                        uv venv "$FLASH_ATTENTION_VENV_DIR"
                    fi
                    source "$FLASH_ATTENTION_VENV_DIR/bin/activate"
                    print_step "Installing Flash Attention CK in virtual environment..."
                    uv pip install $args
                    FLASH_ATTENTION_VENV_PYTHON="$FLASH_ATTENTION_VENV_DIR/bin/python"
                    print_success "Flash Attention CK installed in virtual environment: $FLASH_ATTENTION_VENV_DIR"
                    ;;
                "auto")
                    # Try global install first
                    print_step "Attempting global Flash Attention CK installation with uv..."
                    local install_output
                    install_output=$(uv pip install --python $(which python3) $args 2>&1)
                    local install_exit_code=$?

                    if echo "$install_output" | grep -q "externally managed"; then
                        print_warning "Global installation failed due to externally managed environment"
                        print_step "Creating uv virtual environment for Flash Attention CK installation..."

                        # Create uv venv in project directory
                        FLASH_ATTENTION_VENV_DIR="./flash_attention_venv"
                        if [ ! -d "$FLASH_ATTENTION_VENV_DIR" ]; then
                            uv venv "$FLASH_ATTENTION_VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$FLASH_ATTENTION_VENV_DIR/bin/activate"
                        print_step "Installing Flash Attention CK in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        FLASH_ATTENTION_VENV_PYTHON="$FLASH_ATTENTION_VENV_DIR/bin/python"
                        print_success "Flash Attention CK installed in virtual environment: $FLASH_ATTENTION_VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global Flash Attention CK installation successful"
                        FLASH_ATTENTION_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        FLASH_ATTENTION_VENV_DIR="./flash_attention_venv"
                        if [ ! -d "$FLASH_ATTENTION_VENV_DIR" ]; then
                            uv venv "$FLASH_ATTENTION_VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$FLASH_ATTENTION_VENV_DIR/bin/activate"
                        print_step "Installing Flash Attention CK in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        FLASH_ATTENTION_VENV_PYTHON="$FLASH_ATTENTION_VENV_DIR/bin/python"
                        print_success "Flash Attention CK installed in virtual environment: $FLASH_ATTENTION_VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing Flash Attention CK with pip..."
            python3 -m pip install $args
            FLASH_ATTENTION_VENV_PYTHON=""
        fi
    }

    # Install the package
    if python3 setup_flash_attn_amd.py install; then
        print_success "Python package installation completed"
    else
        print_error "Python package installation failed"
        return 1
    fi

    # Enhanced testing and verification
    ((current_step++))
    show_progress $current_step $total_steps "Verifying installation..."
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise installer-selected python
    PYTHON_CMD=${FLASH_ATTENTION_VENV_PYTHON:-$PYTHON_BIN}

    print_step "Testing Flash Attention CK import..."
    if $PYTHON_CMD -c "
try:
    import flash_attention_amd
    from flash_attention_amd import FlashAttention
    print('✓ Flash Attention CK successfully imported')
    print('Available classes:', [attr for attr in dir(flash_attention_amd) if not attr.startswith('_')])
    
    # Test basic functionality
    import torch
    if torch.cuda.is_available():
        print('✓ GPU acceleration available for testing')
        # Create a simple test
        batch_size, seq_len, heads, head_dim = 1, 128, 8, 64
        q = torch.randn(batch_size, seq_len, heads, head_dim, device='cuda')
        k = torch.randn(batch_size, seq_len, heads, head_dim, device='cuda')
        v = torch.randn(batch_size, seq_len, heads, head_dim, device='cuda')
        
        try:
            flash_attn = FlashAttention()
            out = flash_attn(q, k, v)
            print('✓ Basic Flash Attention computation successful')
            print(f'Output shape: {out.shape}')
        except Exception as e:
            print(f'⚠ Basic computation test failed: {e}')
    else:
        print('⚠ GPU not available, skipping computation test')
        
except Exception as e:
    print(f'✗ Error importing Flash Attention CK: {e}')
    exit(1)
"; then
        print_success "Flash Attention CK verification completed successfully"
    else
        print_error "Flash Attention CK verification failed"
        return 1
    fi

    # Complete progress tracking
    complete_progress

    # Show completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ███████╗██╗     █████╗ ███████╗██╗  ██╗     █████╗ ████████╗████████╗███████╗███╗   ██╗    ║
    ║  ██╔════╝██║    ██╔══██╗██╔════╝██║  ██║    ██╔══██╗╚══██╔══╝╚══██╔══╝██╔════╝████╗  ██║    ║
    ║  █████╗  ██║    ███████║███████╗███████║    ███████║   ██║      ██║   █████╗  ██╔██╗ ██║    ║
    ║  ██╔══╝  ██║    ██╔══██║╚════██║██╔══██║    ██╔══██║   ██║      ██║   ██╔══╝  ██║╚██╗██║    ║
    ║  ██║     ██║    ██║  ██║███████║██║  ██║    ██║  ██║   ██║      ██║   ███████╗██║ ╚████║    ║
    ║  ╚═╝     ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═══╝    ║
    ║                                                         ║
    ║  ██████╗██╗  ██╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     █████╗ ████████╗    ║
    ║  ██╔════╝██║ ██╔╝    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║    ██╔══██╗╚══██╔══╝    ║
    ║  ██║     █████╔╝     ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║    ███████║   ██║       ║
    ║  ██║     ██╔═██╗     ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║    ██╔══██║   ██║       ║
    ║  ╚██████╗██║  ██╗    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗██║    ██║  ██║   ██║       ║
    ║   ╚═════╝╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝    ╚═╝  ╚═╝   ╚═╝       ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "Flash Attention CK installation completed successfully"

    # Provide helpful usage information
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$FLASH_ATTENTION_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./flash_attention_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import flash_attention_amd; from flash_attention_amd import FlashAttention; print('Flash Attention CK ready!')\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import flash_attention_amd; from flash_attention_amd import FlashAttention; print('Flash Attention CK ready!')\"${RESET}"
    fi
    echo
    echo -e "${YELLOW}${BOLD}Installation Details:${RESET}"
    echo -e "${YELLOW}Installation Directory: $INSTALL_DIR${RESET}"
    echo -e "${YELLOW}GPU Architecture: $GPU_ARCH${RESET}"
    echo -e "${YELLOW}ROCm Version: $rocm_version${RESET}"
    echo
    echo -e "${CYAN}${BOLD}Documentation:${RESET}"
    echo -e "${CYAN}$HOME/Prod/Stan-s-ML-Stack/docs/extensions/flash_attention_ck_guide.md${RESET}"
    echo
    echo -e "${CYAN}${BOLD}Environment Variables:${RESET}"
    echo -e "${GREEN}To apply ROCm environment variables to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$(./install_flash_attention_ck.sh --show-env)\"${RESET}"
    echo

    # Add a small delay to ensure the message is seen
    echo -e "${GREEN}${BOLD}Returning to main menu in 3 seconds...${RESET}"
    sleep 1
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"
    sleep 1

    return 0
}

# Function to show help
show_help() {
    cat << EOF
Flash Attention CK Installation Script

USAGE:
    ./install_flash_attention_ck.sh [OPTIONS]

OPTIONS:
    --help              Show this help message
    --show-env          Show ROCm environment variables
    --force             Force reinstallation even if already installed
    --dry-run           Show what would be done without making changes
    --global            Force global installation method
    --venv              Force virtual environment installation method
    --auto              Use auto-detect installation method (default)

EXAMPLES:
    ./install_flash_attention_ck.sh                    # Interactive installation
    ./install_flash_attention_ck.sh --force           # Force reinstall
    ./install_flash_attention_ck.sh --dry-run         # Preview installation
    ./install_flash_attention_ck.sh --show-env        # Show environment variables
    ./install_flash_attention_ck.sh --global          # Force global install

For more information, see:
    $HOME/Prod/Stan-s-ML-Stack/docs/extensions/flash_attention_ck_guide.md
EOF
}

# Parse command line arguments
DRY_RUN=false
FORCE_INSTALL=false
INSTALL_METHOD_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --show-env)
            show_env
            exit 0
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --global)
            INSTALL_METHOD_OVERRIDE="global"
            shift
            ;;
        --venv)
            INSTALL_METHOD_OVERRIDE="venv"
            shift
            ;;
        --auto)
            INSTALL_METHOD_OVERRIDE="auto"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set installation method if overridden
if [ -n "$INSTALL_METHOD_OVERRIDE" ]; then
    FLASH_ATTENTION_INSTALL_METHOD="$INSTALL_METHOD_OVERRIDE"
    PYTORCH_INSTALL_METHOD="$INSTALL_METHOD_OVERRIDE"
fi

# Set force flag
if [ "$FORCE_INSTALL" = true ]; then
    FLASH_ATTENTION_REINSTALL=true
fi

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    print_header "Flash Attention CK Installation (DRY RUN)"
    print_warning "DRY RUN MODE: No actual changes will be made"
    echo
    echo "This would perform the following actions:"
    echo "1. Check ROCm installation and version"
    echo "2. Install system dependencies (git, cmake, ninja-build, etc.)"
    echo "3. Install Python dependencies and development headers"
    echo "4. Install PyTorch with ROCm support"
    echo "5. Clone Flash Attention repository"
    echo "6. Build Flash Attention CK with CMake"
    echo "7. Install Python package"
    echo "8. Verify installation"
    echo
    echo "Installation method: ${FLASH_ATTENTION_INSTALL_METHOD:-auto-detect}"
    echo "Force reinstall: ${FORCE_INSTALL:-false}"
    echo
    print_success "Dry run completed. Use without --dry-run to perform actual installation."
    exit 0
fi

# Run the installation function
install_flash_attention_ck "$@"


