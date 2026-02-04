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
# PyTorch Profiler Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures PyTorch Profiler for performance analysis
# of PyTorch models on AMD GPUs with enhanced ROCm support.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
   ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗    ██████╗ ██████╗  ██████╗ ███████╗██╗██╗     ███████╗██████╗
   ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║    ██╔══██╗██╔═══██╗██╔═══██╗██╔════╝██║██║     ██╔════╝██╔══██╗
   ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║    ██████╔╝██║   ██║██║   ██║█████╗  ██║██║     █████╗  ██████╔╝
   ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║    ██╔═══╝ ██║   ██║██║   ██║██╔══╝  ██║██║     ██╔══╝  ██╔══██╗
   ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║    ██║     ╚██████╔╝╚██████╔╝██║     ██║███████╗███████╗██║  ██║
   ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝      ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
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

# Function to detect ROCm version with regex patterns
detect_rocm_version() {
    local rocm_version=""

    # Try rocminfo first
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | sed -n 's/.*ROCm Version:\s*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1)
    fi

    # Fallback to directory listing with regex
    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
    fi

    # Fallback to hipcc version
    if [ -z "$rocm_version" ] && command_exists hipcc; then
        hipcc_version=$(hipcc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
        if [ -n "$hipcc_version" ]; then
            rocm_version="$hipcc_version"
        fi
    fi

    echo "$rocm_version"
}

# Function to detect GPU architecture
detect_gpu_architecture() {
    local gpu_arch=""

    if command_exists rocminfo; then
        # Try to get GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -oE 'gfx[0-9]+' | head -n 1)
    fi

    # Fallback to common architectures based on ROCm version
    if [ -z "$gpu_arch" ]; then
        local rocm_version=$(detect_rocm_version)
        local major_version=$(echo "$rocm_version" | cut -d '.' -f 1)

        case $major_version in
            6)
                gpu_arch="gfx1100"  # RDNA3
                ;;
            5)
                gpu_arch="gfx1030"  # RDNA2
                ;;
            *)
                gpu_arch="gfx1100"  # Default to latest
                ;;
        esac
    fi

    echo "$gpu_arch"
}

# Function to install rocminfo if missing
install_rocminfo() {
    local package_manager=$(detect_package_manager)

    print_step "Installing rocminfo using $package_manager..."

    case $package_manager in
        apt)
            sudo apt-get update && sudo apt-get install -y rocminfo
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
        print_success "rocminfo installed successfully"
        return 0
    else
        print_error "Failed to install rocminfo"
        return 1
    fi
}

# Function to setup ROCm environment variables
setup_rocm_environment() {
    print_step "Setting up ROCm environment variables..."

    # Detect GPU architecture for optimal configuration
    local gpu_arch=$(detect_gpu_architecture)
    HSA_OVERRIDE_GFX_VERSION="${gpu_arch#gfx}"  # Remove 'gfx' prefix
    PYTORCH_ROCM_ARCH="$gpu_arch"
    ROCM_PATH="/opt/rocm"
    PATH="/opt/rocm/bin:$PATH"
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Set HSA_TOOLS_LIB if rocprofiler library exists
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
        print_step "ROCm profiler library found and configured"
    else
        # Try to install rocprofiler
        if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
            print_step "Installing rocprofiler for HSA tools support..."
            sudo apt-get update && sudo apt-get install -y rocprofiler
            if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
                HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
                print_success "ROCm profiler installed and configured"
            else
                HSA_TOOLS_LIB=0
                print_warning "ROCm profiler installation failed, disabling HSA tools"
            fi
        else
            HSA_TOOLS_LIB=0
            print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
        fi
    fi

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
        unset PYTORCH_CUDA_ALLOC_CONF
        print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
    fi

    print_success "ROCm environment variables configured"
}

# Function to check for dry run mode
is_dry_run() {
    [[ "$*" == *"--dry-run"* ]]
}

# Function to check for force flag
is_force() {
    [[ "$*" == *"--force"* ]]
}

# Function to handle errors with retry mechanism
retry_command() {
    local max_attempts=3
    local attempt=1
    local command="$1"
    local error_msg="$2"

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt/$max_attempts: $command"
        if eval "$command"; then
            return 0
        else
            print_warning "Attempt $attempt failed"
            if [ $attempt -lt $max_attempts ]; then
                sleep 2
            fi
        fi
        ((attempt++))
    done

    print_error "$error_msg"
    return 1
}

# Function to detect container environment
is_container() {
    # Check for common container indicators
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q container /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to detect WSL environment
is_wsl() {
    if grep -q "microsoft" /proc/version 2>/dev/null || [ -n "$WSL_DISTRO_NAME" ]; then
        return 0
    else
        return 1
    fi
}

# Main installation function
install_pytorch_profiler() {
    print_header "PyTorch Profiler Installation"

    # Check for dry run
    if is_dry_run "$@"; then
        print_warning "DRY RUN MODE - No actual changes will be made"
        echo
    fi

    # Detect environment
    print_section "Environment Detection"

    if is_container; then
        print_step "Container environment detected"
    fi

    if is_wsl; then
        print_step "WSL environment detected"
    fi

    # Check if PyTorch is installed
    print_section "Checking PyTorch Installation"

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
                if is_force "$@"; then
                    print_warning "Force reinstall requested - proceeding with reinstallation"
                    print_step "Will reinstall PyTorch Profiler dependencies"
                else
                    print_step "PyTorch installation is complete. Use --force to reinstall profiler dependencies anyway."
                fi
            else
                print_warning "PyTorch with ROCm support is installed (PyTorch $pytorch_version, ROCm $hip_version) but GPU acceleration is not working"
                print_step "Will install PyTorch Profiler (GPU acceleration may be limited)"
            fi
        else
            print_warning "PyTorch is installed (version $pytorch_version) but without ROCm support"
            print_step "Will install PyTorch Profiler (GPU profiling may not work)"
        fi
    else
        print_error "PyTorch is not installed. Please install PyTorch with ROCm support first."
        print_step "Run: ./install_pytorch_rocm.sh"
        return 1
    fi

    # Check ROCm installation
    print_section "Checking ROCm Installation"

    if command_exists rocminfo; then
        print_success "rocminfo found"
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            if ! is_dry_run "$@"; then
                if ! install_rocminfo; then
                    print_error "Failed to install rocminfo"
                    return 1
                fi
            else
                print_step "[DRY RUN] Would install rocminfo"
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi

    # Detect ROCm version
    rocm_version=$(detect_rocm_version)
    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    # Setup ROCm environment
    if ! is_dry_run "$@"; then
        setup_rocm_environment
    else
        print_step "[DRY RUN] Would setup ROCm environment variables"
    fi

    # Check if uv is installed
    print_section "Installing PyTorch Profiler Dependencies"

    if ! command_exists uv; then
        if ! is_dry_run "$@"; then
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
            print_step "[DRY RUN] Would install uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}PyTorch Profiler Installation Options:${RESET}"
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
                    if ! is_dry_run "$@"; then
                        print_step "Installing globally with pip..."
                        python3 -m pip install --break-system-packages $args
                    else
                        print_step "[DRY RUN] Would install globally with pip: $args"
                    fi
                    PYTORCH_VENV_PYTHON=""
                    ;;
                "venv")
                    if ! is_dry_run "$@"; then
                        print_step "Creating uv virtual environment..."
                        VENV_DIR="./pytorch_profiler_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args
                        PYTORCH_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    else
                        print_step "[DRY RUN] Would create virtual environment and install: $args"
                    fi
                    ;;
                "auto")
                    if ! is_dry_run "$@"; then
                        # Try global install first
                        print_step "Attempting global installation with uv..."
                        local install_output
                        install_output=$(uv pip install --python $(which python3) $args 2>&1)
                        local install_exit_code=$?

                        if echo "$install_output" | grep -q "externally managed"; then
                            print_warning "Global installation failed due to externally managed environment"
                            print_step "Creating uv virtual environment for installation..."

                            # Create uv venv in project directory
                            VENV_DIR="./pytorch_profiler_venv"
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
                            VENV_DIR="./pytorch_profiler_venv"
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
                    else
                        print_step "[DRY RUN] Would attempt auto-detect installation: $args"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            if ! is_dry_run "$@"; then
                print_step "Installing with pip..."
                python3 -m pip install $args
            else
                print_step "[DRY RUN] Would install with pip: $args"
            fi
            PYTORCH_VENV_PYTHON=""
        fi
    }

    # Install PyTorch Profiler dependencies
    print_step "Installing PyTorch Profiler dependencies..."
    install_python_package "torch-tb-profiler" "tensorboard" || true
    
    # Final verification
    if $PYTHON_CMD -c "from torch.profiler import profile; print('✓ Profiler module found')" 2>/dev/null; then
        print_success "PyTorch Profiler is functional"
    fi
}

# Set script options
DRY_RUN=false
FORCE=false
INSTALL_METHOD="auto"

# Parse command line arguments
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
            show_env
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--force] [--show-env]"
            exit 1
            ;;
    esac
done

# Don't exit on error in dry run mode
if [ "$DRY_RUN" = false ]; then
    set -e  # Exit on error
fi

# Run the installation function with all script arguments
install_pytorch_profiler "$@"

exit 0
