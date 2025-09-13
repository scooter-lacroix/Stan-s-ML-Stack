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
# Flash Attention Build Script for AMD GPUs
# =============================================================================
# This script builds Flash Attention with AMD GPU support using enhanced
# ROCm detection, virtual environment support, and modern package management.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
   ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
  ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
  ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
   ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
  ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
  ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                            Flash Attention Build Script for AMD GPUs
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

# Function to detect GPU architecture
detect_gpu_arch() {
    if command_exists rocminfo; then
        # Try to get GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -o "gfx[0-9]*" | head -n 1)
        if [ -n "$gpu_arch" ]; then
            echo "$gpu_arch"
            return 0
        fi
    fi

    # Fallback to common architectures
    print_warning "Could not detect GPU architecture, using default gfx1100"
    echo "gfx1100"
}

# Function to retry command with exponential backoff
retry_command() {
    local max_attempts=3
    local attempt=1
    local delay=1

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt of $max_attempts: $@"
        if "$@"; then
            return 0
        fi

        if [ $attempt -lt $max_attempts ]; then
            print_warning "Command failed, retrying in $delay seconds..."
            sleep $delay
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done

    print_error "Command failed after $max_attempts attempts"
    return 1
}

# Function to check if running in container
is_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q container /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if running in WSL
is_wsl() {
    if grep -q "microsoft" /proc/version 2>/dev/null || [ -n "$WSL_DISTRO_NAME" ]; then
        return 0
    else
        return 1
    fi
}

# Function to load configuration file
load_config() {
    local config_file="${FLASH_ATTN_CONFIG:-flash_attn_config.sh}"
    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
    else
        print_step "No configuration file found, using defaults"
    fi
}

# Function to save configuration
save_config() {
    local config_file="${FLASH_ATTN_CONFIG:-flash_attn_config.sh}"
    print_step "Saving configuration to $config_file"
    cat > "$config_file" << EOF
# Flash Attention Build Configuration
# Generated on $(date)

# Installation method (global, venv, auto)
FLASH_ATTN_INSTALL_METHOD="${FLASH_ATTN_INSTALL_METHOD:-auto}"

# ROCm version (auto-detected if empty)
FLASH_ATTN_ROCM_VERSION="${FLASH_ATTN_ROCM_VERSION:-}"

# GPU architecture (auto-detected if empty)
FLASH_ATTN_GPU_ARCH="${FLASH_ATTN_GPU_ARCH:-}"

# Virtual environment directory
FLASH_ATTN_VENV_DIR="${FLASH_ATTN_VENV_DIR:-./flash_attn_venv}"

# Force reinstall
FLASH_ATTN_FORCE="${FLASH_ATTN_FORCE:-false}"

# Dry run mode
FLASH_ATTN_DRY_RUN="${FLASH_ATTN_DRY_RUN:-false}"
EOF
    print_success "Configuration saved"
}

# Main build function
build_flash_attention() {
    print_header "Flash Attention Build for AMD GPUs"

    # Load configuration
    load_config

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FLASH_ATTN_FORCE=true
                shift
                ;;
            --dry-run)
                FLASH_ATTN_DRY_RUN=true
                shift
                ;;
            --install-method)
                FLASH_ATTN_INSTALL_METHOD="$2"
                shift 2
                ;;
            --venv-dir)
                FLASH_ATTN_VENV_DIR="$2"
                shift 2
                ;;
            --config)
                FLASH_ATTN_CONFIG="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --force              Force reinstallation"
                echo "  --dry-run           Show what would be done without executing"
                echo "  --install-method    Installation method: global, venv, auto"
                echo "  --venv-dir          Virtual environment directory"
                echo "  --config            Configuration file path"
                echo "  --help              Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Set defaults
    FLASH_ATTN_INSTALL_METHOD="${FLASH_ATTN_INSTALL_METHOD:-auto}"
    FLASH_ATTN_VENV_DIR="${FLASH_ATTN_VENV_DIR:-./flash_attn_venv}"
    FLASH_ATTN_FORCE="${FLASH_ATTN_FORCE:-false}"
    FLASH_ATTN_DRY_RUN="${FLASH_ATTN_DRY_RUN:-false}"

    # Check if dry run
    if [ "$FLASH_ATTN_DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi

    # Detect environment
    print_section "Environment Detection"

    if is_container; then
        print_step "Running in container environment"
        CONTAINER_ENV=true
    fi

    if is_wsl; then
        print_step "Running in WSL environment"
        WSL_ENV=true
    fi

    # Check if Flash Attention is already installed
    if package_installed "flash_attention_amd" && [ "$FLASH_ATTN_FORCE" != true ]; then
        flash_attn_version=$(python3 -c "import flash_attention_amd; print(getattr(flash_attention_amd, '__version__', 'unknown'))" 2>/dev/null)
        print_success "Flash Attention AMD is already installed (version: $flash_attn_version)"
        print_step "Use --force to reinstall"
        return 0
    fi

    # ROCm detection and setup
    print_section "ROCm Detection and Setup"

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH="gfx1100"
        export ROCM_PATH="/opt/rocm"
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

        # Detect GPU architecture
        detected_arch=$(detect_gpu_arch)
        print_step "Detected GPU architecture: $detected_arch"
        export PYTORCH_ROCM_ARCH="$detected_arch"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Try to install rocprofiler
            package_manager=$(detect_package_manager)
            if [ "$package_manager" = "apt" ] && apt-cache show rocprofiler >/dev/null 2>&1; then
                if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                    print_step "Installing rocprofiler for HSA tools support..."
                    retry_command sudo apt-get update && sudo apt-get install -y rocprofiler
                    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
                        export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
                        print_success "ROCm profiler installed and configured"
                    else
                        export HSA_TOOLS_LIB=0
                        print_warning "ROCm profiler installation failed, disabling HSA tools"
                    fi
                else
                    print_step "[DRY RUN] Would install rocprofiler"
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
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        retry_command sudo apt update && sudo apt install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with apt"
                    fi
                    ;;
                dnf)
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        retry_command sudo dnf install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with dnf"
                    fi
                    ;;
                yum)
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        retry_command sudo yum install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with yum"
                    fi
                    ;;
                pacman)
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        retry_command sudo pacman -S rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with pacman"
                    fi
                    ;;
                zypper)
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        retry_command sudo zypper install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with zypper"
                    fi
                    ;;
                *)
                    print_error "Unsupported package manager: $package_manager"
                    return 1
                    ;;
            esac
            if command_exists rocminfo || [ "$FLASH_ATTN_DRY_RUN" = true ]; then
                if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                    print_success "Installed rocminfo"
                else
                    print_step "[DRY RUN] rocminfo would be installed"
                fi
                # Re-run ROCm setup
                export HSA_OVERRIDE_GFX_VERSION=11.0.0
                export PYTORCH_ROCM_ARCH="gfx1100"
                export ROCM_PATH="/opt/rocm"
                export PATH="/opt/rocm/bin:$PATH"
                export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi

    # Check prerequisites
    print_section "Checking Prerequisites"

    # Check if PyTorch with ROCm is installed
    PYTHON_CMD="python3"
    if ! $PYTHON_CMD -c "import torch; print(torch.version.hip)" &> /dev/null; then
        print_error "PyTorch with ROCm support is not installed. Please install PyTorch with ROCm support first."
        print_step "Run: ./install_pytorch_rocm.sh"
        return 1
    fi
    print_success "PyTorch with ROCm support is installed"

    # Check if CUDA is available through ROCm
    if ! $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_warning "CUDA is not available through ROCm. Check your environment variables."
        print_step "Setting environment variables..."
        export HIP_VISIBLE_DEVICES=0,1
        export CUDA_VISIBLE_DEVICES=0,1
        export PYTORCH_ROCM_DEVICE=0,1

        # Check again
        if ! $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
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

    # Installation method selection
    print_section "Installation Method Selection"

    echo
    echo -e "${CYAN}${BOLD}Flash Attention Installation Options:${RESET}"
    echo "1) Global installation (recommended for system-wide use)"
    echo "2) Virtual environment (isolated installation)"
    echo "3) Auto-detect (try global, fallback to venv if needed)"
    echo

    if [ -n "$FLASH_ATTN_INSTALL_METHOD" ]; then
        case $FLASH_ATTN_INSTALL_METHOD in
            global)
                INSTALL_METHOD="global"
                print_step "Using pre-configured global installation method"
                ;;
            venv)
                INSTALL_METHOD="venv"
                print_step "Using pre-configured virtual environment method"
                ;;
            auto)
                INSTALL_METHOD="auto"
                print_step "Using pre-configured auto-detect method"
                ;;
            *)
                print_error "Invalid installation method: $FLASH_ATTN_INSTALL_METHOD"
                return 1
                ;;
        esac
    else
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

    # Install uv if not present
    if ! command_exists uv; then
        print_section "Installing uv Package Manager"
        if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
            print_step "Installing uv package manager..."
            retry_command python3 -m pip install uv

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
            print_step "[DRY RUN] Would install uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Function to handle uv commands properly with venv fallback
    uv_pip_install() {
        local args="$@"

        # Check if uv is available as a command
        if command_exists uv; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        python3 -m pip install --break-system-packages $args
                    else
                        print_step "[DRY RUN] Would install globally: $args"
                    fi
                    FLASH_ATTN_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        if [ ! -d "$FLASH_ATTN_VENV_DIR" ]; then
                            uv venv "$FLASH_ATTN_VENV_DIR"
                        fi
                        source "$FLASH_ATTN_VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args
                        FLASH_ATTN_VENV_PYTHON="$FLASH_ATTN_VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $FLASH_ATTN_VENV_DIR"
                    else
                        print_step "[DRY RUN] Would create venv at $FLASH_ATTN_VENV_DIR and install: $args"
                    fi
                    ;;
                "auto")
                    # Try global install first
                    print_step "Attempting global installation with uv..."
                    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                        local install_output
                        install_output=$(uv pip install --python $(which python3) $args 2>&1)
                        local install_exit_code=$?

                        if echo "$install_output" | grep -q "externally managed"; then
                            print_warning "Global installation failed due to externally managed environment"
                            print_step "Creating uv virtual environment for installation..."

                            # Create uv venv in project directory
                            if [ ! -d "$FLASH_ATTN_VENV_DIR" ]; then
                                uv venv "$FLASH_ATTN_VENV_DIR"
                            fi

                            # Activate venv and install
                            source "$FLASH_ATTN_VENV_DIR/bin/activate"
                            print_step "Installing in virtual environment..."
                            uv pip install $args

                            # Store venv path for verification
                            FLASH_ATTN_VENV_PYTHON="$FLASH_ATTN_VENV_DIR/bin/python"
                            print_success "Installed in virtual environment: $FLASH_ATTN_VENV_DIR"
                        elif [ $install_exit_code -eq 0 ]; then
                            print_success "Global installation successful"
                            FLASH_ATTN_VENV_PYTHON=""
                        else
                            print_error "Global installation failed with unknown error:"
                            echo "$install_output"
                            print_step "Falling back to virtual environment..."

                            # Create uv venv in project directory
                            if [ ! -d "$FLASH_ATTN_VENV_DIR" ]; then
                                uv venv "$FLASH_ATTN_VENV_DIR"
                            fi

                            # Activate venv and install
                            source "$FLASH_ATTN_VENV_DIR/bin/activate"
                            print_step "Installing in virtual environment..."
                            uv pip install $args

                            # Store venv path for verification
                            FLASH_ATTN_VENV_PYTHON="$FLASH_ATTN_VENV_DIR/bin/python"
                            print_success "Installed in virtual environment: $FLASH_ATTN_VENV_DIR"
                        fi
                    else
                        print_step "[DRY RUN] Would attempt global install, fallback to venv at $FLASH_ATTN_VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing with pip..."
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                python3 -m pip install $args
            else
                print_step "[DRY RUN] Would install with pip: $args"
            fi
            FLASH_ATTN_VENV_PYTHON=""
        fi
    }

    # Install dependencies
    print_section "Installing Dependencies"

    print_step "Installing build dependencies..."
    package_manager=$(detect_package_manager)
    case $package_manager in
        apt)
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                retry_command sudo apt-get update
                retry_command sudo apt-get install -y build-essential cmake git python3-dev python3-pip ninja-build
            else
                print_step "[DRY RUN] Would install build dependencies with apt"
            fi
            ;;
        dnf)
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                retry_command sudo dnf install -y gcc gcc-c++ make cmake git python3-devel python3-pip ninja-build
            else
                print_step "[DRY RUN] Would install build dependencies with dnf"
            fi
            ;;
        yum)
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                retry_command sudo yum install -y gcc gcc-c++ make cmake git python3-devel python3-pip ninja-build
            else
                print_step "[DRY RUN] Would install build dependencies with yum"
            fi
            ;;
        pacman)
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                retry_command sudo pacman -S base-devel cmake git python python-pip ninja
            else
                print_step "[DRY RUN] Would install build dependencies with pacman"
            fi
            ;;
        zypper)
            if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
                retry_command sudo zypper install -y gcc gcc-c++ make cmake git python3-devel python3-pip ninja
            else
                print_step "[DRY RUN] Would install build dependencies with zypper"
            fi
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            return 1
            ;;
    esac

    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        print_step "Installing Python dependencies..."
        uv_pip_install packaging ninja wheel setuptools
        print_success "Dependencies installed successfully"
    else
        print_step "[DRY RUN] Would install Python dependencies"
    fi

    # Create AMD implementation
    print_section "Creating AMD Implementation"

    # Create directory for AMD implementation
    print_step "Creating directory for AMD implementation..."
    FLASH_ATTN_DIR="$HOME/flash-attention-amd"
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        mkdir -p $FLASH_ATTN_DIR
        cd $FLASH_ATTN_DIR
    else
        print_step "[DRY RUN] Would create directory $FLASH_ATTN_DIR"
    fi

    # Clone Triton
    print_step "Cloning Triton repository..."
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        if [ ! -d "triton" ]; then
            retry_command git clone https://github.com/ROCm/triton.git
        else
            print_step "Triton already cloned, updating..."
            cd triton
            retry_command git pull
            cd ..
        fi
        cd triton/python
        GPU_ARCHS="$detected_arch" retry_command python3 setup.py install
        cd $FLASH_ATTN_DIR
        uv_pip_install matplotlib pandas
    else
        print_step "[DRY RUN] Would clone Triton and install with GPU_ARCHS=$detected_arch"
    fi

    # Create Python implementation file
    print_step "Creating Python implementation file..."
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        cat > flash_attention_amd.py << 'EOF'
import torch
import torch.nn.functional as F

class FlashAttention(torch.nn.Module):
    """
    Flash Attention implementation for AMD GPUs using PyTorch operations.
    This is a pure PyTorch implementation that works on AMD GPUs.
    """
    def __init__(self, dropout=0.0, causal=False):
        super().__init__()
        self.dropout = dropout
        self.causal = causal

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)

        Returns: (batch_size, seq_len, num_heads, head_dim)
        """
        # Reshape q, k, v for multi-head attention
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, head_dim) @ (batch_size, num_heads, head_dim, seq_len_k)
        # -> (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))

        # Apply attention mask if provided
        if mask is not None:
            # Expand mask to match attention weights shape
            if mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)

            # Apply mask
            attn_weights.masked_fill_(~mask, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Compute attention output
        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_k, head_dim)
        # -> (batch_size, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)

        # Transpose back to (batch_size, seq_len_q, num_heads, head_dim)
        output = output.transpose(1, 2)

        return output

def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, return_attn_probs=False):
    """
    Functional interface for Flash Attention.

    Args:
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        dropout_p: dropout probability
        causal: whether to apply causal masking
        return_attn_probs: whether to return attention probabilities

    Returns:
        output: (batch_size, seq_len, num_heads, head_dim)
        attn_weights: (batch_size, num_heads, seq_len, seq_len) if return_attn_probs=True
    """
    flash_attn = FlashAttention(dropout=dropout_p, causal=causal)
    output = flash_attn(q, k, v)

    if return_attn_probs:
        # Compute attention weights for return
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)

        # Compute attention weights
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        return output, attn_weights

    return output

# For compatibility with the original Flash Attention API
class FlashAttentionInterface:
    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, causal=False):
        output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        return output
EOF
    else
        print_step "[DRY RUN] Would create flash_attention_amd.py"
    fi

    # Create setup file
    print_step "Creating setup file..."
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="flash_attention_amd",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["flash_attention_amd"],
    install_requires=[
        "torch>=2.0.0",
    ],
    author="Stanley Chisango",
    author_email="scooterlacroix@gmail.com",
    description="Flash Attention implementation for AMD GPUs",
    keywords="flash attention, amd, gpu, pytorch, rocm",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
EOF
    else
        print_step "[DRY RUN] Would create setup.py"
    fi

    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        print_success "AMD implementation created successfully"
    fi

    # Install Flash Attention
    print_section "Installing Flash Attention"

    # Install the AMD implementation
    print_step "Installing the AMD implementation..."
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        uv_pip_install -e .
        print_success "Flash Attention installed successfully"
    else
        print_step "[DRY RUN] Would install Flash Attention package"
    fi

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${FLASH_ATTN_VENV_PYTHON:-python3}

    # Create test script
    print_step "Creating test script..."
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        cat > $FLASH_ATTN_DIR/test_flash_attention.py << 'EOF'
import torch
import time
from flash_attention_amd import flash_attn_func

def test_flash_attention():
    print("Testing Flash Attention on AMD GPU...")

    # Create dummy data
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Test basic functionality
    print("Testing basic Flash Attention...")
    output = flash_attn_func(q, k, v, causal=True)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, num_heads, head_dim), f"Expected shape {(batch_size, seq_len, num_heads, head_dim)}, got {output.shape}"

    # Test with attention weights return
    print("Testing with attention weights return...")
    output, attn_weights = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"

    # Performance test
    print("Running performance test...")
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(10):
        _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()

    avg_time = (end_time - start_time) / 10 * 1000
    print(f"Average inference time: {avg_time:.2f} ms")

    print("Flash Attention test passed!")

if __name__ == "__main__":
    test_flash_attention()
EOF

        # Run test script
        print_step "Running test script..."
        cd $FLASH_ATTN_DIR
        if $PYTHON_CMD test_flash_attention.py; then
            print_success "Flash Attention is working correctly"
            verification_success=true
        else
            print_error "Flash Attention test failed"
            verification_success=false
        fi
    else
        print_step "[DRY RUN] Would create and run test script"
        verification_success=true
    fi

    # Save configuration
    save_config

    # Cleanup
    print_section "Cleaning up"
    if [ "$FLASH_ATTN_DRY_RUN" != true ]; then
        print_step "Removing temporary files..."
        rm -f $FLASH_ATTN_DIR/test_flash_attention.py
        print_success "Cleanup completed successfully"
    else
        print_step "[DRY RUN] Would clean up temporary files"
    fi

    # Completion message
    if [ "$verification_success" = true ]; then
        print_header "Flash Attention Build Completed Successfully!"

        echo -e "${GREEN}You can now use Flash Attention in your PyTorch code:${RESET}"
        echo
        if [ -n "$FLASH_ATTN_VENV_PYTHON" ]; then
            echo -e "${YELLOW}source $FLASH_ATTN_VENV_DIR/bin/activate${RESET}"
        fi
        echo -e "${YELLOW}import torch${RESET}"
        echo -e "${YELLOW}from flash_attention_amd import flash_attn_func${RESET}"
        echo
        echo -e "${YELLOW}# Create input tensors${RESET}"
        echo -e "${YELLOW}q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
        echo -e "${YELLOW}k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
        echo -e "${YELLOW}v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
        echo
        echo -e "${YELLOW}# Run Flash Attention${RESET}"
        echo -e "${YELLOW}output = flash_attn_func(q, k, v, causal=True)${RESET}"
        echo

        # Show environment variables
        echo -e "${CYAN}${BOLD}Environment Variables:${RESET}"
        show_env
        echo

        return 0
    else
        print_error "Build completed but verification failed"
        return 1
    fi
}

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the build function with all script arguments
build_flash_attention "$@"
