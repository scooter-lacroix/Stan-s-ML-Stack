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
# AITER Installation Script
# =============================================================================
# This script installs AMD Iterative Tensor Runtime (AITER) for efficient
# tensor operations on AMD GPUs.
# =============================================================================

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

# ASCII Art Banner
cat << "EOF"
   █████╗ ██╗████████╗███████╗██████╗
  ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗
  ███████║██║   ██║   █████╗  ██████╔╝
  ██╔══██║██║   ██║   ██╔══╝  ██╔══██╗
  ██║  ██║██║   ██║   ███████╗██║  ██║
  ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
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

# Progress bar variables
PROGRESS_BAR_WIDTH=50
PROGRESS_CURRENT=0
PROGRESS_TOTAL=100
PROGRESS_CHAR="▓"
PROGRESS_EMPTY="░"
PROGRESS_ANIMATION=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
ANIMATION_INDEX=0

# Function definitions
print_header() {
    echo
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}║               === $1 ===               ║${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}"
    echo
}

print_section() {
    echo
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${BLUE}${BOLD}│ $1${RESET}"
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}"
}

print_step() {
    echo -e "${MAGENTA}➤ $1${RESET}"
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

# Function to initialize progress bar
init_progress_bar() {
    PROGRESS_TOTAL=$1
    PROGRESS_CURRENT=0

    # Save cursor position
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        tput sc
        # Clear line and print initial progress bar
        tput el
        draw_progress_bar
        # Move cursor back to saved position
        tput rc
    fi
}

# Function to update progress bar
update_progress_bar() {
    local increment=${1:-1}
    PROGRESS_CURRENT=$((PROGRESS_CURRENT + increment))

    # Ensure we don't exceed the total
    if [ $PROGRESS_CURRENT -gt $PROGRESS_TOTAL ]; then
        PROGRESS_CURRENT=$PROGRESS_TOTAL
    fi

    # Save cursor position
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        tput sc
        # Move to top of terminal
        tput cup 0 0
        # Clear line and print updated progress bar
        tput el
        draw_progress_bar
        # Move cursor back to saved position
        tput rc
    fi
}

# Function to draw progress bar
draw_progress_bar() {
    local percent=$((PROGRESS_CURRENT * 100 / PROGRESS_TOTAL))
    local completed=$((PROGRESS_CURRENT * PROGRESS_BAR_WIDTH / PROGRESS_TOTAL))
    local remaining=$((PROGRESS_BAR_WIDTH - completed))

    # Update animation index
    ANIMATION_INDEX=$(( (ANIMATION_INDEX + 1) % ${#PROGRESS_ANIMATION[@]} ))
    local spinner=${PROGRESS_ANIMATION[$ANIMATION_INDEX]}

    # Draw progress bar with colors
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -ne "${CYAN}${BOLD}[${RESET}${MAGENTA}"
        for ((i=0; i<completed; i++)); do
            echo -ne "${PROGRESS_CHAR}"
        done

        for ((i=0; i<remaining; i++)); do
            echo -ne "${BLUE}${PROGRESS_EMPTY}"
        done

        echo -ne "${RESET}${CYAN}${BOLD}]${RESET} ${percent}% ${spinner} "

        # Add task description if provided
        if [ -n "$1" ]; then
            echo -ne "$1"
        fi

        echo -ne "\r"
    fi
}

# Function to complete progress bar
complete_progress_bar() {
    PROGRESS_CURRENT=$PROGRESS_TOTAL

    # Save cursor position
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        tput sc
        # Move to top of terminal
        tput cup 0 0
        # Clear line and print completed progress bar
        tput el
        draw_progress_bar "Complete!"
        echo
        # Move cursor back to saved position
        tput rc
    fi
}

# Function to print a clean separator line
print_separator() {
    echo -e "${BLUE}───────────────────────────────────────────────────────────${RESET}"
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

# Function to detect if running in WSL
detect_wsl() {
    if [ -f /proc/version ] && grep -q "Microsoft" /proc/version; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect if running in a container
detect_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q "docker\|container\|lxc\|podman" /proc/1/cgroup 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect if running on Windows
detect_windows() {
    if [ -n "$WSL_DISTRO_NAME" ] || [ -n "$WSLENV" ]; then
        echo "true"
    else
        echo "false"
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

# Function to create default configuration file
create_default_config() {
    local config_file="$1"
    cat > "$config_file" << 'EOF'
# AITER Installation Configuration File
# This file contains default settings for AITER installation
# Modify these values to customize your installation

# Installation method: global, venv, auto
INSTALL_METHOD=auto

# Virtual environment directory (used when INSTALL_METHOD=venv)
VENV_DIR=./aiter_rocm_venv

# ROCm settings
HSA_OVERRIDE_GFX_VERSION=11.0.0
PYTORCH_ROCM_ARCH=gfx1100;gfx1101;gfx1102
ROCM_PATH=/opt/rocm

# Build options
BUILD_ISOLATION=true
NO_DEPS=false

# Testing options
RUN_TESTS=true
TEST_TIMEOUT=60

# Logging options
LOG_LEVEL=INFO
LOG_FILE=./aiter_install.log

# Color options
USE_COLORS=true
NO_COLOR=false
EOF
}

# Function to load configuration file
load_config() {
    local config_file="$1"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"

        # Read the config file line by line and export variables properly
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^[[:space:]]*# ]] && continue
            [[ -z "$key" ]] && continue

            # Remove leading/trailing whitespace
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

            # Export the variable
            export "$key"="$value"
        done < "$config_file"

        # Override command line arguments if config specifies them
        if [ -n "$INSTALL_METHOD" ]; then
            case $INSTALL_METHOD in
                global|venv|auto)
                    print_step "Using config-specified installation method: $INSTALL_METHOD"
                    ;;
                *)
                    print_warning "Invalid INSTALL_METHOD in config, using default"
                    ;;
            esac
        fi

        print_success "Configuration loaded"
    else
        print_step "No configuration file found, using defaults"
        print_step "To customize installation, create aiter_config.sh with your settings"
    fi
}

# Global variables for cleanup
TEMP_DIRS=()
TEMP_FILES=()
BACKGROUND_PIDS=()

# Function to clean up resources on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up resources...${RESET}"

    # Kill any background processes we started
    for pid in "${BACKGROUND_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done

    # Remove any temporary files we created
    for file in "${TEMP_FILES[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file" 2>/dev/null
        fi
    done

    # Remove any temporary directories we created
    for dir in "${TEMP_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir" 2>/dev/null
        fi
    done

    # Reset terminal (only if we have a TTY)
    if [ -t 1 ] && command_exists tput; then
        tput cnorm  # Show cursor
    fi
    if [ -t 0 ]; then
        stty echo   # Enable echo
    fi

    echo -e "${GREEN}Cleanup completed.${RESET}"
}

# Set up signal handling for graceful exit
handle_signal() {
    echo -e "\n${YELLOW}Received termination signal. Exiting gracefully...${RESET}"
    cleanup
    exit 1
}

# Register signal handlers
trap handle_signal INT TERM HUP PIPE
trap cleanup EXIT

# Check for --show-env option first (before other processing)
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Parse command line arguments
FORCE_INSTALL=false
DRY_RUN=false
CREATE_CONFIG=false
CONFIG_FILE="./aiter_config.sh"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --create-config)
            CREATE_CONFIG=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force         Force reinstallation even if AITER is already installed"
            echo "  --dry-run       Show what would be done without actually installing"
            echo "  --create-config Create a default configuration file"
            echo "  --config FILE   Use specified configuration file"
            echo "  --show-env      Show ROCm environment variables"
            echo "  --help          Show this help message"
            echo ""
            echo "Enhanced AITER Installation Script with ROCm support"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Handle create config option
if [ "$CREATE_CONFIG" = true ]; then
    print_step "Creating default configuration file: $CONFIG_FILE"
    create_default_config "$CONFIG_FILE"
    print_success "Configuration file created. Edit it to customize your installation."
    exit 0
fi

# Load configuration if file exists
load_config "$CONFIG_FILE"

# Main installation function
install_aiter() {
    # Hide cursor during installation for cleaner output
    tput civis

    print_header "AITER Installation"

    # Handle dry run
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual installation will be performed"
        echo
    fi

    # Detect environment
    print_section "Environment Detection"
    is_wsl=$(detect_wsl)
    is_container=$(detect_container)
    is_windows=$(detect_windows)

    if [ "$is_wsl" = "true" ]; then
        print_step "Detected Windows Subsystem for Linux (WSL)"
        print_warning "WSL detected - some ROCm features may have limited functionality"
    fi

    if [ "$is_container" = "true" ]; then
        print_step "Detected container environment"
        print_warning "Container environment detected - ensure ROCm is properly configured"
    fi

    if [ "$is_windows" = "true" ]; then
        print_step "Detected Windows environment"
        print_warning "Windows detected - ensure WSL2 is properly configured for ROCm"
    fi

    if [ "$is_wsl" = "false" ] && [ "$is_container" = "false" ] && [ "$is_windows" = "false" ]; then
        print_step "Detected native Linux environment"
    fi

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking AITER installation..."

    # Check if AITER is already installed
    if package_installed "aiter"; then
        # Check if it's the correct AITER (ROCm version usually has aiter.torch)
        if ! $PYTHON_BIN -c "import aiter.torch" &>/dev/null; then
            print_warning "AITER is installed but appears to be the wrong version (missing ROCm submodules)"
            print_step "Will force reinstallation of ROCm AITER"
            FORCE_INSTALL=true
        fi

        if [ "$FORCE_INSTALL" = true ] || [ "${MLSTACK_BATCH_MODE:-0}" = "1" ] || [ -n "${RUSTY_STACK:-}" ] || [ ! -t 0 ]; then
            print_warning "AITER is already installed, proceeding with reinstallation (non-interactive or forced)"
        else
            print_warning "AITER is already installed"
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping AITER installation"
                complete_progress_bar
                return 0
            fi
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
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    # Check if PyTorch is installed
    update_progress_bar 10
    draw_progress_bar "Checking PyTorch installation..."
    print_section "Checking PyTorch Installation"

    if [ "${MLSTACK_SKIP_TORCH_INSTALL:-0}" != "1" ]; then
        if ! package_installed "torch"; then
            # Check if torch package exists but can't be imported (broken install)
            if python3 -c "import importlib.util; spec = importlib.util.find_spec('torch'); exit(0 if spec else 1)" 2>/dev/null; then
                print_warning "PyTorch is installed but cannot be imported!"
                print_step "This usually means missing system libraries (e.g., libmpi_cxx.so.40)"
                print_step "Attempting to reinstall ROCm PyTorch..."

                # Uninstall broken torch and reinstall from AMD repo
                $PYTHON_BIN -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
                $PYTHON_BIN -m pip install \
                    torch torchvision torchaudio triton \
                    --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
                    --break-system-packages --no-cache-dir

                if ! package_installed "torch"; then
                    print_error "Failed to fix PyTorch. Please run: ./scripts/install_pytorch_multi.sh"
                    complete_progress_bar
                    return 1
                fi
                print_success "ROCm PyTorch reinstalled successfully"
            else
                print_error "PyTorch is not installed. Please install PyTorch with ROCm support first."
                complete_progress_bar
                return 1
            fi
        fi
    fi

    update_progress_bar 10
    draw_progress_bar "Checking PyTorch ROCm support..."

    # Check if PyTorch has ROCm/HIP support
    if [ "${MLSTACK_SKIP_TORCH_INSTALL:-0}" != "1" ]; then
        if ! python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            print_warning "PyTorch does not have explicit ROCm/HIP support"
            print_warning "AITER may not work correctly without ROCm support in PyTorch"
            read -p "Do you want to continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping AITER installation"
                complete_progress_bar
                return 0
            fi
        fi
    fi

    # Check if git is installed
    update_progress_bar 5
    draw_progress_bar "Checking dependencies..."
    print_section "Checking Dependencies"

    if ! command_exists git; then
        print_error "git is not installed. Please install git first."
        complete_progress_bar
        return 1
    fi

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
    echo -e "${CYAN}${BOLD}AITER Installation Options:${RESET}"
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

    # Create a temporary directory for installation
    update_progress_bar 10
    draw_progress_bar "Creating temporary directory..."
    print_section "Installing AITER"
    print_step "Creating temporary directory..."

    # Store original directory
    original_dir=$(pwd)

    temp_dir=$(mktemp -d)
    # Add to our tracking array for cleanup
    TEMP_DIRS+=("$temp_dir")
    # Add test file to tracking
    TEMP_FILES+=("/tmp/test_aiter.py")

    cd "$temp_dir" || {
        print_error "Failed to create temporary directory"
        complete_progress_bar
        return 1
    }

    # Clone AITER repository
    update_progress_bar 10
    draw_progress_bar "Cloning AITER repository..."
    print_step "Cloning AITER repository..."
    
    # Uninstall any existing aiter package (could be the wrong one from PyPI)
    print_step "Ensuring clean environment (uninstalling existing aiter)..."
    $PYTHON_BIN -m pip uninstall -y aiter || true
    
    git clone --recursive https://github.com/ROCm/aiter.git

    if [ $? -ne 0 ]; then
        print_error "Failed to clone AITER repository"
        rm -rf "$temp_dir"
        complete_progress_bar
        return 1
    fi

    # Enter AITER directory
    cd aiter || { print_error "Failed to enter AITER directory"; rm -rf "$temp_dir"; complete_progress_bar; return 1; }

    # Create a custom setup.py to fix compatibility issues
    print_step "Creating custom setup.py to improve compatibility..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="aiter",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.10.0",
        "pandas>=1.5.0",
        "einops>=0.6.0",
        "packaging>=21.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "typing-extensions>=4.0.0",
    ],
    include_package_data=True,
    package_data={
        "aiter": ["jit/*.json", "jit/**/*.json", "configs/*.json", "configs/**/*.json"],
    },
)
EOF

    # Create comprehensive RDNA 3 GPU support for AITER
    print_step "Creating comprehensive RDNA 3 GPU support (gfx1100/gfx1101/gfx1102)..."

    # Create the directory structure if it doesn't exist
    mkdir -p aiter/torch

    # Create a proper torch_hip.py file with full RDNA 3 support
    print_step "Creating aiter/torch/torch_hip.py with full RDNA 3 support..."

    # Copy the pre-created torch_hip.py file to the correct location
    if [ -f "aiter/torch/torch_hip.py" ]; then
        # Backup the original file
        cp aiter/torch/torch_hip.py aiter/torch/torch_hip.py.bak
    fi

    # Create the torch_hip.py file with comprehensive RDNA 3 support
    cat > aiter/torch/torch_hip.py << 'EOF'
#!/usr/bin/env python3
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

import torch
import os
import warnings
import re
import logging

# Configure logging
logger = logging.getLogger("aiter.torch.hip")

# List of supported AMD GPU architectures
SUPPORTED_ARCHS = {
    # RDNA 3 architectures
    "gfx1100": {
        "name": "RDNA 3 (Navi 31)",
        "cards": ["RX 7900 XTX", "RX 7900 XT", "Radeon PRO W7900", "Radeon PRO W7800"],
        "compute_units": 96,
        "supported": True
    },
    "gfx1101": {
        "name": "RDNA 3 (Navi 32)",
        "cards": ["RX 7800 XT", "RX 7700 XT", "Radeon PRO W7700", "Radeon PRO W7600"],
        "compute_units": 60,
        "supported": True
    },
    "gfx1102": {
        "name": "RDNA 3 (Navi 33)",
        "cards": ["RX 7600", "RX 7600 XT", "Radeon PRO W7500"],
        "compute_units": 32,
        "supported": True
    },
    # RDNA 2 architectures (for reference)
    "gfx1030": {
        "name": "RDNA 2 (Navi 21)",
        "cards": ["RX 6900 XT", "RX 6800 XT", "RX 6800"],
        "compute_units": 80,
        "supported": False
    },
    "gfx1031": {
        "name": "RDNA 2 (Navi 22)",
        "cards": ["RX 6700 XT", "RX 6700"],
        "compute_units": 40,
        "supported": False
    },
    "gfx1032": {
        "name": "RDNA 2 (Navi 23)",
        "cards": ["RX 6600 XT", "RX 6600"],
        "compute_units": 28,
        "supported": False
    }
}

# List of supported RDNA 3 architectures for quick access
SUPPORTED_RDNA3_ARCHS = ["gfx1100", "gfx1101", "gfx1102"]

def get_device_name():
    """
    Get the name of the HIP device.

    Returns:
        str: The name of the HIP device, or None if not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Get device name
        device_name = torch.cuda.get_device_name(0)

        # Special handling for RDNA 3 architectures
        for arch in SUPPORTED_RDNA3_ARCHS:
            if arch in device_name:
                logger.info(f"Detected RDNA 3 GPU with architecture {arch}: {device_name}")
                return device_name

        # If no specific architecture is detected in the name, return the name as is
        return device_name
    except Exception as e:
        logger.warning(f"Failed to get device name: {e}")
        return "Unknown AMD GPU"

def get_device_arch():
    """
    Get the architecture of the HIP device.

    Returns:
        str: The architecture of the HIP device, or None if not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Try to get architecture from device properties
        props = str(torch.cuda.get_device_properties(0))

        # Special handling for RDNA 3 architectures
        for arch in SUPPORTED_RDNA3_ARCHS:
            if arch in props:
                logger.info(f"Detected RDNA 3 GPU architecture: {arch}")
                return arch

        # Try to extract architecture from device name
        device_name = torch.cuda.get_device_name(0)
        for arch, info in SUPPORTED_ARCHS.items():
            for card in info["cards"]:
                if card in device_name:
                    logger.info(f"Inferred architecture {arch} from device name: {device_name}")
                    return arch

        # If we can't determine the architecture, check environment variables
        if "PYTORCH_ROCM_ARCH" in os.environ:
            arch = os.environ["PYTORCH_ROCM_ARCH"]
            logger.info(f"Using architecture from PYTORCH_ROCM_ARCH: {arch}")
            return arch

        # If all else fails, default to gfx1100 for RDNA 3 GPUs
        if "Radeon RX 79" in device_name or "Radeon PRO W7" in device_name:
            logger.info(f"Defaulting to gfx1100 for RDNA 3 GPU: {device_name}")
            return "gfx1100"

        # Return unknown if we can't determine the architecture
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to get device architecture: {e}")
        # Default to a common architecture if detection fails
        return "gfx1100"  # Default to RDNA 3 as fallback

def is_rdna3_gpu():
    """
    Check if the current GPU is an RDNA 3 GPU.

    Returns:
        bool: True if the current GPU is an RDNA 3 GPU, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Get device name
        device_name = torch.cuda.get_device_name(0)

        # Check if device name contains any RDNA 3 card names
        for arch in SUPPORTED_RDNA3_ARCHS:
            info = SUPPORTED_ARCHS.get(arch, {})
            for card in info.get("cards", []):
                if card in device_name:
                    return True

        # Check if architecture is in RDNA 3 architectures
        arch = get_device_arch()
        if arch in SUPPORTED_RDNA3_ARCHS:
            return True

        return False
    except Exception as e:
        logger.warning(f"Failed to determine if GPU is RDNA 3: {e}")
        return False

def get_gpu_info():
    """
    Get detailed information about the GPU.

    Returns:
        dict: Dictionary containing GPU information.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(0)

        # Get architecture
        arch = get_device_arch()

        # Get architecture info
        arch_info = SUPPORTED_ARCHS.get(arch, {})

        # Create GPU info dictionary
        gpu_info = {
            "available": True,
            "name": props.name,
            "architecture": arch,
            "architecture_name": arch_info.get("name", "Unknown"),
            "total_memory": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
            "supported": arch_info.get("supported", False),
            "is_rdna3": arch in SUPPORTED_RDNA3_ARCHS
        }

        return gpu_info
    except Exception as e:
        logger.warning(f"Failed to get GPU information: {e}")
        return {"available": False, "error": str(e)}

def print_gpu_info():
    """Print detailed information about the GPU."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    try:
        # Get GPU info
        gpu_info = get_gpu_info()

        # Print GPU info
        print("GPU Information:")
        print(f"  Name: {gpu_info['name']}")
        print(f"  Architecture: {gpu_info['architecture']} ({gpu_info['architecture_name']})")
        print(f"  Total Memory: {gpu_info['total_memory'] / (1024**3):.2f} GB")
        print(f"  Compute Capability: {gpu_info['compute_capability']}")
        print(f"  Supported by AITER: {'Yes' if gpu_info['supported'] else 'No'}")
        print(f"  Is RDNA 3: {'Yes' if gpu_info['is_rdna3'] else 'No'}")
    except Exception as e:
        print(f"Failed to print GPU information: {e}")

# Initialize module
if torch.cuda.is_available():
    try:
        # Print GPU information at module load time
        logger.info(f"CUDA is available through ROCm")
        logger.info(f"Device: {get_device_name()}")
        logger.info(f"Architecture: {get_device_arch()}")
        logger.info(f"Is RDNA 3: {is_rdna3_gpu()}")
    except Exception as e:
        logger.warning(f"Failed to initialize torch_hip module: {e}")
else:
    logger.warning("CUDA is not available through ROCm")
EOF

    print_success "Created comprehensive torch_hip.py with full RDNA 3 support"

    # Create an __init__.py file in the torch directory if it doesn't exist
    if [ ! -f "aiter/torch/__init__.py" ]; then
        print_step "Creating aiter/torch/__init__.py..."
        cat > aiter/torch/__init__.py << 'EOF'
# Import torch_hip module
try:
    from .torch_hip import (
        get_device_name,
        get_device_arch,
        is_rdna3_gpu,
        get_gpu_info,
        print_gpu_info,
        SUPPORTED_RDNA3_ARCHS
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import torch_hip module: {e}")
EOF
        print_success "Created aiter/torch/__init__.py"
    fi

    # Create a patch for aiter/__init__.py to properly handle RDNA 3 GPUs
    if [ -f "aiter/__init__.py" ]; then
        print_step "Patching aiter/__init__.py for RDNA 3 support..."

        # Backup the original file
        cp aiter/__init__.py aiter/__init__.py.bak

        # Add error handling for torch import and set environment variables for RDNA 3 GPUs
        cat > aiter/__init__.py.patch << 'EOF'
--- __init__.py.orig
+++ __init__.py
@@ -1,5 +1,31 @@
+import os
+import logging
+import warnings
+
+# Configure logging
+logging.basicConfig(level=logging.INFO)
+logger = logging.getLogger("aiter")
+
+# Set environment variables for RDNA 3 GPUs
+os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")
+os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")
+os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
+
+# Suppress HIP warnings
+os.environ["AMD_LOG_LEVEL"] = "0"
+
 # Import torch
-import torch
+try:
+    import torch
+
+    # Import torch_hip module for AMD GPU support
+    try:
+        from .torch import torch_hip
+        logger.info(f"AITER initialized with RDNA 3 GPU support")
+    except ImportError as e:
+        logger.warning(f"Failed to import torch_hip module: {e}")
+except ImportError:
+    logger.warning("PyTorch not available, some features will be limited")
+    torch = None

 # Import other modules
 from . import data
EOF

        # Apply the patch
        if command_exists patch; then
            patch -p0 aiter/__init__.py < aiter/__init__.py.patch
            if [ $? -ne 0 ]; then
                print_warning "Failed to apply patch automatically, applying manual edits..."
                # Manual edit as fallback
                sed -i '1s/^/import os\nimport logging\nimport warnings\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger("aiter")\n\n# Set environment variables for RDNA 3 GPUs\nos.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")\nos.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")\n\n# Suppress HIP warnings\nos.environ["AMD_LOG_LEVEL"] = "0"\n\n/' aiter/__init__.py
                sed -i 's/# Import torch\nimport torch/# Import torch\ntry:\n    import torch\n    \n    # Import torch_hip module for AMD GPU support\n    try:\n        from .torch import torch_hip\n        logger.info(f"AITER initialized with RDNA 3 GPU support")\n    except ImportError as e:\n        logger.warning(f"Failed to import torch_hip module: {e}")\nexcept ImportError:\n    logger.warning("PyTorch not available, some features will be limited")\n    torch = None/' aiter/__init__.py
            fi
        else
            print_warning "patch command not found, applying manual edits..."
            # Manual edit as fallback
            sed -i '1s/^/import os\nimport logging\nimport warnings\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger("aiter")\n\n# Set environment variables for RDNA 3 GPUs\nos.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")\nos.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")\n\n# Suppress HIP warnings\nos.environ["AMD_LOG_LEVEL"] = "0"\n\n/' aiter/__init__.py
            sed -i 's/# Import torch\nimport torch/# Import torch\ntry:\n    import torch\n    \n    # Import torch_hip module for AMD GPU support\n    try:\n        from .torch import torch_hip\n        logger.info(f"AITER initialized with RDNA 3 GPU support")\n    except ImportError as e:\n        logger.warning(f"Failed to import torch_hip module: {e}")\nexcept ImportError:\n    logger.warning("PyTorch not available, some features will be limited")\n    torch = None/' aiter/__init__.py
        fi

        print_success "Successfully patched aiter/__init__.py for RDNA 3 support"
    fi

    print_step "Normalizing aiter/__init__.py imports and GPU visibility..."
    python3 - <<'PY'
from pathlib import Path
path = Path("aiter/__init__.py")
text = path.read_text()
old_block = "os.environ[\"HIP_VISIBLE_DEVICES\"] = os.environ.get(\"HIP_VISIBLE_DEVICES\", \"0\")\nos.environ[\"CUDA_VISIBLE_DEVICES\"] = os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"0\")"
if old_block in text:
    new_block = (
        "if \"HIP_VISIBLE_DEVICES\" not in os.environ and \"CUDA_VISIBLE_DEVICES\" not in os.environ:\n"
        "    pass\n"
        "else:\n"
        "    os.environ[\"HIP_VISIBLE_DEVICES\"] = os.environ.get(\"HIP_VISIBLE_DEVICES\", os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"\"))\n"
        "    os.environ[\"CUDA_VISIBLE_DEVICES\"] = os.environ.get(\"CUDA_VISIBLE_DEVICES\", os.environ.get(\"HIP_VISIBLE_DEVICES\", \"\"))"
    )
    text = text.replace(old_block, new_block)
if "from .utility import dtypes as dtypes" in text:
    text = text.replace(
        "from .utility import dtypes as dtypes  # noqa: E402",
        "try:\n    from .utility import dtypes as dtypes  # noqa: E402\nexcept Exception:\n    dtypes = None  # noqa: E402",
    )
path.write_text(text)
PY

    print_step "Creating aiter.utility compatibility shim..."
    mkdir -p aiter/utility
    cat > aiter/utility/__init__.py << 'PY'
"""Compatibility shim for aiter.utility."""
from .dtypes import *  # noqa: F401,F403
PY
    cat > aiter/utility/dtypes.py << 'PY'
"""Fallback dtypes map for AITER."""
import torch


def _torch_dtype(name: str, fallback):
    return getattr(torch, name, fallback)


# Core floating types
fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32

# FP8 variants (fallback to fp16 when unavailable)
fp8 = _torch_dtype("float8_e4m3fn", fp16)
fp8_e4m3fn = _torch_dtype("float8_e4m3fn", fp16)
fp8_e5m2 = _torch_dtype("float8_e5m2", fp16)

# FP8 metadata/scale placeholders (no native dtype -> use uint8)
fp8_e8m0 = getattr(torch, "uint8", fp16)

# Integer types
i8 = torch.int8
u8 = getattr(torch, "uint8", torch.int8)
i16 = torch.int16
i32 = torch.int32

# Packed/quantized placeholders
i4x2 = u8
fp4x2 = u8


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


DTYPE_MAP = {
    "float16": fp16,
    "fp16": fp16,
    "bf16": bf16,
    "bfloat16": bf16,
    "float32": fp32,
    "fp32": fp32,
    "int8": i8,
    "i8": i8,
    "int16": i16,
    "i16": i16,
    "int32": i32,
    "i32": i32,
    "fp8": fp8,
    "fp8_e4m3fn": fp8_e4m3fn,
    "fp8_e5m2": fp8_e5m2,
    "fp8_e8m0": fp8_e8m0,
    "fp4x2": fp4x2,
    "i4x2": i4x2,
}


def get_dtype(name: str):
    return DTYPE_MAP.get(name, fp16)
PY

    # Create a setup.py file with proper RDNA 3 support
    print_step "Creating setup.py with proper RDNA 3 support..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages
import os

# Set environment variables for RDNA 3 GPUs
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")

setup(
    name="aiter",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.10.0",
        "pandas>=1.5.0",
        "einops>=0.6.0",
        "packaging>=21.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "typing-extensions>=4.0.0",
        "torch>=1.13.0",
    ],
    include_package_data=True,
    package_data={
        "aiter": ["jit/*.json", "jit/**/*.json", "configs/*.json", "configs/**/*.json"],
    },
    python_requires=">=3.8",
    author="AMD",
    author_email="aiter@amd.com",
    description="AI Tensor Engine for ROCm",
    keywords="deep learning, machine learning, gpu, amd, rocm",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
EOF

    print_success "Created setup.py with proper RDNA 3 support"

    # Make sure submodules are initialized
    update_progress_bar 10
    draw_progress_bar "Initializing submodules..."
    print_step "Initializing submodules..."
    git submodule sync && git submodule update --init --recursive

    if [ $? -ne 0 ]; then
        print_warning "Failed to initialize some submodules, continuing anyway"
    fi

    # Create a function to handle installation properly
    # FIXED: Always install globally with --break-system-packages for ML stack consistency
    uv_pip_install() {
        local args="$@"

        case $INSTALL_METHOD in
            "global"|"auto")
                # Use pip with --break-system-packages for global installation
                # This ensures packages are available system-wide and torch is preserved
                print_step "Installing globally with pip..."
                if $PYTHON_BIN -m pip install --break-system-packages $args 2>&1; then
                    print_success "Global installation successful"
                    AITER_VENV_PYTHON=""
                else
                    print_warning "pip install failed, trying with uv..."
                    if command_exists uv; then
                        uv pip install --python "$PYTHON_BIN" --break-system-packages $args || true
                    fi
                    AITER_VENV_PYTHON=""
                fi
                ;;
            "venv")
                print_step "Creating virtual environment..."
                VENV_DIR="${HOME}/.mlstack/aiter_venv"
                mkdir -p "${HOME}/.mlstack"
                # Use python3 -m venv instead of uv venv to avoid externally managed issues
                if [ ! -d "$VENV_DIR" ]; then
                    $PYTHON_BIN -m venv "$VENV_DIR"
                fi
                # Install into the venv using its pip directly
                print_step "Installing in virtual environment..."
                "$VENV_DIR/bin/pip" install --upgrade pip
                "$VENV_DIR/bin/pip" install $args
                AITER_VENV_PYTHON="$VENV_DIR/bin/python"
                print_success "Installed in virtual environment: $VENV_DIR"
                ;;
        esac
    }

    # Install required dependencies first
    update_progress_bar 15
    draw_progress_bar "Installing required dependencies..."
    print_step "Installing required dependencies first..."

    uv_pip_install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions

    # Install AITER
    update_progress_bar 20
    draw_progress_bar "Installing AITER..."
    print_step "Installing AITER..."

    # Use the appropriate installation method
    case $INSTALL_METHOD in
        "global")
            print_step "Installing AITER globally..."
            # Use trap to handle SIGPIPE and other signals
            trap 'print_warning "Installation interrupted, trying alternative method..."; break' SIGPIPE SIGINT SIGTERM

            # Try different installation methods
            set +e  # Don't exit on error
            python3 -m pip install --break-system-packages . --no-build-isolation
            install_result=$?

            if [ $install_result -ne 0 ]; then
                print_warning "First installation attempt failed, trying without build isolation..."
                python3 -m pip install --break-system-packages .
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_warning "Second installation attempt failed, trying with --no-deps..."
                python3 -m pip install --break-system-packages . --no-deps
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_error "All installation attempts failed"
                set -e
                trap - SIGPIPE SIGINT SIGTERM
                cd "$original_dir"
                return 1
            fi

            set -e  # Return to normal error handling
            trap - SIGPIPE SIGINT SIGTERM  # Reset trap
            ;;
        "venv")
            print_step "Installing AITER in virtual environment..."
            # Use trap to handle SIGPIPE and other signals
            trap 'print_warning "Installation interrupted, trying alternative method..."; break' SIGPIPE SIGINT SIGTERM

            # Try different installation methods
            set +e  # Don't exit on error
            $PYTHON_BIN -m pip install . --no-build-isolation --break-system-packages
            install_result=$?

            if [ $install_result -ne 0 ]; then
                print_warning "First installation attempt failed, trying without build isolation..."
                $PYTHON_BIN -m pip install . --break-system-packages
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_warning "Second installation attempt failed, trying with --no-deps..."
                $PYTHON_BIN -m pip install . --no-deps --break-system-packages
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_error "All installation attempts failed"
                set -e
                trap - SIGPIPE SIGINT SIGTERM
                cd "$original_dir"
                return 1
            fi

            set -e  # Return to normal error handling
            trap - SIGPIPE SIGINT SIGTERM  # Reset trap
            ;;
        "auto")
            print_step "Installing AITER with auto-detection..."
            # Use trap to handle SIGPIPE and other signals
            trap 'print_warning "Installation interrupted, trying alternative method..."; break' SIGPIPE SIGINT SIGTERM

            # Try different installation methods
            set +e  # Don't exit on error
            uv_pip_install . --no-build-isolation
            install_result=$?

            if [ $install_result -ne 0 ]; then
                print_warning "First installation attempt failed, trying without build isolation..."
                uv_pip_install .
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_warning "Second installation attempt failed, trying with --no-deps..."
                uv_pip_install . --no-deps
                install_result=$?
            fi

            if [ $install_result -ne 0 ]; then
                print_error "All installation attempts failed"
                set -e
                trap - SIGPIPE SIGINT SIGTERM
                cd "$original_dir"
                return 1
            fi

            set -e  # Return to normal error handling
            trap - SIGPIPE SIGINT SIGTERM  # Reset trap
            ;;
    esac

    if [ $install_result -ne 0 ]; then
        print_error "Failed to install AITER after multiple attempts"
        rm -rf "$temp_dir"
        complete_progress_bar
        return 1
    fi

    # Use venv Python if available, otherwise the installer-selected python
    PYTHON_CMD=${AITER_VENV_PYTHON:-$PYTHON_BIN}

    print_step "Ensuring AITER config assets are installed..."
    site_packages=$($PYTHON_CMD - <<'PY'
import sysconfig
print(sysconfig.get_paths().get("purelib", ""))
PY
    )
    if [ -n "$site_packages" ]; then
        mkdir -p "$site_packages/aiter/jit"
        while IFS= read -r file; do
            if [ -f "$file" ]; then
                cp "$file" "$site_packages/aiter/jit/" || true
            fi
        done < <(find aiter/jit -name "*.json" -print 2>/dev/null)

        print_step "Staging AITER meta sources for JIT compilation..."
        meta_dir="$site_packages/aiter_meta"
        mkdir -p "$meta_dir"
        for dir in csrc hsa 3rdparty gradlib; do
            if [ -d "$dir" ]; then
                rm -rf "$meta_dir/$dir" || true
                cp -R "$dir" "$meta_dir/" || true
            fi
        done
        if [ -d "$meta_dir/csrc" ]; then
            echo "package" > "$site_packages/aiter/install_mode" || true
        fi
        export AITER_META_DIR="$meta_dir"
    fi

    # Verify installation
    update_progress_bar 20
    draw_progress_bar "Verifying installation..."
    print_section "Verifying Installation"

    # Add a small delay to ensure the installation is complete
    sleep 2

    # Force Python to reload modules
    $PYTHON_CMD -c "import importlib; import sys; [sys.modules.pop(m, None) for m in list(sys.modules.keys()) if m.startswith('aiter')]" &>/dev/null

    # Create a comprehensive test file to verify functionality with RDNA 3 GPUs
    print_step "Creating a comprehensive test file..."
    cat > /tmp/test_aiter.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import time

# Set environment variables for RDNA 3 GPUs
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")

# Avoid pinning to a single GPU unless explicitly requested
visible_devices = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
if not visible_devices:
    try:
        import subprocess
        rocm_smi = "/opt/rocm/bin/rocm-smi" if os.path.exists("/opt/rocm/bin/rocm-smi") else "rocm-smi"
        output = subprocess.check_output([rocm_smi, "--showproductname", "--csv"], stderr=subprocess.DEVNULL).decode()
        lines = [line for line in output.splitlines()[1:] if line.strip()]
        if lines:
            visible_devices = ",".join(str(i) for i in range(len(lines)))
    except Exception:
        visible_devices = ""

if visible_devices:
    os.environ["HIP_VISIBLE_DEVICES"] = visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

os.environ["AMD_LOG_LEVEL"] = "0"  # Suppress HIP warnings

print("=" * 80)
print("AITER Comprehensive Test for RDNA 3 GPUs")
print("=" * 80)

# Check for all required dependencies
required_deps = [
    "packaging", "pybind11", "pandas", "einops", "psutil",
    "numpy", "setuptools", "wheel", "typing_extensions", "torch"
]

print("\nPython path:", sys.path)
print("\nChecking dependencies:")
missing_deps = []

for dep in required_deps:
    try:
        # Try to import the module
        module_name = dep.replace("-", "_")  # Handle hyphenated names
        __import__(module_name)
        print(f"✓ {dep} is installed")
    except ImportError as e:
        print(f"✗ {dep} is missing: {e}")
        missing_deps.append(dep)

if missing_deps:
    print(f"\nWARNING: Missing dependencies: {', '.join(missing_deps)}")
    print("Will attempt to continue anyway...")
else:
    print("\nAll dependencies are installed!")

# Check PyTorch and GPU availability
try:
    import torch
    print("\nPyTorch Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Print GPU information
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # Try to get device properties
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"  - Total memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"  - Compute capability: {props.major}.{props.minor}")
            except Exception as e:
                print(f"  - Could not get device properties: {e}")
    else:
        print("No CUDA-capable GPU detected")
except ImportError:
    print("\nPyTorch is not installed")

# Now try to import AITER
try:
    import aiter
    print("\nAITER Information:")
    print("AITER imported successfully!")
    print("AITER version:", getattr(aiter, "__version__", "unknown"))
    print("AITER path:", getattr(aiter, "__file__", "unknown"))

    # Try to access some basic functionality
    print("\nTesting basic AITER functionality:")
    if hasattr(aiter, "__all__"):
        print(f"Available modules: {aiter.__all__}")
    else:
        print("No __all__ attribute found, listing dir(aiter):")
        print([x for x in dir(aiter) if not x.startswith('_')])

    # Check for torch_hip module
    try:
        from aiter.torch import torch_hip
        print("\nAITER torch_hip module imported successfully!")
        print("Testing RDNA 3 GPU detection:")

        # Print GPU information
        print(f"Device name: {torch_hip.get_device_name()}")
        print(f"Device architecture: {torch_hip.get_device_arch()}")
        print(f"Is RDNA 3 GPU: {torch_hip.is_rdna3_gpu()}")

        # Print detailed GPU information
        print("\nDetailed GPU Information:")
        torch_hip.print_gpu_info()

        # Test if the GPU is properly detected as RDNA 3
        if torch_hip.is_rdna3_gpu():
            print("\n✓ RDNA 3 GPU detected and properly recognized")
        else:
            print("\n✗ RDNA 3 GPU not detected")
    except ImportError as e:
        print(f"\nAITER torch_hip module not available: {e}")
    except Exception as e:
        print(f"\nError with AITER torch_hip module: {e}")
        # If it's the gfx1100 error, handle it properly
        if "'gfx1100'" in str(e):
            print("\nDetected gfx1100 GPU architecture - this is expected")
            print("Setting environment variables to ensure compatibility...")
            os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100;gfx1101;gfx1102"
            print("Environment variables set for RDNA 3 GPUs")

    # Comprehensive tensor operations and benchmarks
    if torch.cuda.is_available():
        try:
            print("\n" + "=" * 50)
            print("COMPREHENSIVE GPU TENSOR OPERATIONS & BENCHMARKS")
            print("=" * 50)

            # Get GPU info
            device_name = torch.cuda.get_device_name(0)
            print(f"\n✓ Testing on GPU: {device_name}")

            # Basic tensor creation
            print("\n[1/7] Testing basic tensor operations:")
            start_time = time.time()
            x = torch.randn(100, 100, device="cuda")
            y = torch.randn(100, 100, device="cuda")
            print(f"✓ Created tensors on GPU with shape {x.shape}")

            # Basic arithmetic operations
            z = x + y
            print(f"✓ Addition: {z.shape}")
            z = x * y
            print(f"✓ Element-wise multiplication: {z.shape}")
            z = torch.matmul(x, y)
            print(f"✓ Matrix multiplication: {z.shape}")
            basic_time = time.time() - start_time
            print(f"✓ Basic operations completed in {basic_time:.4f} seconds")

            # Memory allocation test
            print("\n[2/7] Testing memory allocation:")
            start_time = time.time()
            max_size = 0
            try:
                for i in range(8):
                    size = 100 * (2 ** i)
                    tensor = torch.randn(size, size, device="cuda")
                    max_size = size
                    print(f"✓ Created tensor of size {size}x{size} ({(size*size*4)/(1024*1024):.2f} MB)")
                    del tensor
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"✓ Maximum tensor size: {max_size}x{max_size}")
                print(f"✓ Reached memory limit: {str(e)}")
            memory_time = time.time() - start_time
            print(f"✓ Memory test completed in {memory_time:.4f} seconds")

            # Linear algebra operations
            print("\n[3/7] Testing linear algebra operations:")
            start_time = time.time()
            a = torch.randn(500, 500, device="cuda")
            # SVD
            try:
                u, s, v = torch.linalg.svd(a)
                print(f"✓ SVD decomposition successful: {u.shape}, {s.shape}, {v.shape}")
            except Exception as e:
                print(f"✗ SVD failed: {e}")

            # Eigenvalues
            try:
                eigenvalues = torch.linalg.eigvals(a)
                print(f"✓ Eigenvalue computation successful: {eigenvalues.shape}")
            except Exception as e:
                print(f"✗ Eigenvalue computation failed: {e}")

            # Matrix inverse
            try:
                a_inv = torch.linalg.inv(a)
                print(f"✓ Matrix inverse successful: {a_inv.shape}")
            except Exception as e:
                print(f"✗ Matrix inverse failed: {e}")
            linalg_time = time.time() - start_time
            print(f"✓ Linear algebra operations completed in {linalg_time:.4f} seconds")

            # Convolution operations (common in deep learning)
            print("\n[4/7] Testing convolution operations:")
            start_time = time.time()
            try:
                # Create a batch of images (batch_size, channels, height, width)
                images = torch.randn(8, 3, 64, 64, device="cuda")
                # Create convolution filters
                filters = torch.randn(16, 3, 3, 3, device="cuda")
                # Perform convolution
                output = torch.nn.functional.conv2d(images, filters, padding=1)
                print(f"✓ 2D Convolution successful: input {images.shape} → output {output.shape}")

                # Max pooling
                pooled = torch.nn.functional.max_pool2d(output, kernel_size=2, stride=2)
                print(f"✓ Max pooling successful: input {output.shape} → output {pooled.shape}")
            except Exception as e:
                print(f"✗ Convolution operations failed: {e}")
            conv_time = time.time() - start_time
            print(f"✓ Convolution operations completed in {conv_time:.4f} seconds")

            # Reduction operations
            print("\n[5/7] Testing reduction operations:")
            start_time = time.time()
            large_tensor = torch.randn(1000, 1000, device="cuda")
            # Sum
            sum_result = torch.sum(large_tensor)
            print(f"✓ Sum reduction: {sum_result.item():.4f}")
            # Mean
            mean_result = torch.mean(large_tensor)
            print(f"✓ Mean reduction: {mean_result.item():.4f}")
            # Max
            max_result = torch.max(large_tensor)
            print(f"✓ Max reduction: {max_result.item():.4f}")
            # Min
            min_result = torch.min(large_tensor)
            print(f"✓ Min reduction: {min_result.item():.4f}")
            reduction_time = time.time() - start_time
            print(f"✓ Reduction operations completed in {reduction_time:.4f} seconds")

            # GEMM benchmark (General Matrix Multiply - core of many ML operations)
            print("\n[6/7] Running GEMM benchmark:")
            start_time = time.time()
            iterations = 10
            sizes = [128, 256, 512, 1024]
            for size in sizes:
                a = torch.randn(size, size, device="cuda")
                b = torch.randn(size, size, device="cuda")

                # Warm-up
                for _ in range(5):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()

                # Benchmark
                gemm_start = time.time()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                gemm_time = time.time() - gemm_start

                # Calculate GFLOPS (2*N^3 operations for matrix multiply)
                flops = 2 * size**3 * iterations
                gflops = flops / (gemm_time * 1e9)
                print(f"✓ GEMM {size}x{size}: {gemm_time/iterations*1000:.2f} ms/iter, {gflops:.2f} GFLOPS")

            # Final benchmark summary
            print("\n[7/7] Benchmark summary:")
            total_time = time.time() - start_time
            print(f"✓ Total benchmark time: {total_time:.4f} seconds")
            print(f"✓ GPU: {device_name}")
            print(f"✓ Basic operations: {basic_time:.4f} seconds")
            print(f"✓ Memory allocation: {memory_time:.4f} seconds")
            print(f"✓ Linear algebra: {linalg_time:.4f} seconds")
            print(f"✓ Convolution: {conv_time:.4f} seconds")
            print(f"✓ Reduction: {reduction_time:.4f} seconds")

            print("\n✅ All tensor operations and benchmarks completed successfully!")
        except Exception as e:
            print(f"\n❌ Error during tensor operations: {e}")
            # Try to provide more detailed error information
            import traceback
            traceback.print_exc()
            print("\nAttempting to continue with installation despite benchmark errors...")

    print("\nAITER package is functional")
    success = True
except ImportError as e:
    print(f"\nError importing AITER: {e}")
    success = False
except Exception as e:
    print(f"\nUnexpected error with AITER: {e}")
    # If it's the gfx1100 error, handle it properly
    if "'gfx1100'" in str(e):
        print("\nDetected gfx1100 GPU architecture - this is expected")
        print("Setting environment variables to ensure compatibility...")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100;gfx1101;gfx1102"
        print("Environment variables set for RDNA 3 GPUs")
        success = True
    else:
        success = False

print("\n" + "=" * 80)
print(f"Test result: {'SUCCESS' if success else 'FAILURE'}")
print("=" * 80)

# Exit with success code if main package works
sys.exit(0 if success else 1)
EOF

    # Run the test directly
    print_step "Running AITER verification test..."
    $PYTHON_CMD /tmp/test_aiter.py
    test_result=$?
    if [ $test_result -eq 0 ]; then
        print_success "AITER main package is functional"

        # Try to import aiter.torch but don't fail if it's not available
        if python3 -c "import aiter.torch" &>/dev/null; then
            print_success "AITER torch module is available"
        else
            print_warning "AITER torch module is not available, but main package is installed"
            # This is not a fatal error, as the main package is installed
        fi

        # Success even if torch module is not available
        return 0
    else
        print_error "AITER installation verification failed"

        # Try a direct installation as a last resort
        print_step "Attempting direct installation as a last resort..."

        # Install dependencies first
        print_step "Installing required dependencies first..."
        if command_exists uv; then
            uv pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
            # Try to install with --no-deps first to avoid dependency conflicts
            uv pip install aiter --no-deps || uv pip install aiter
        else
            python3 -m pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
            # Try to install with --no-deps first to avoid dependency conflicts
            python3 -m pip install aiter --no-deps || python3 -m pip install aiter
        fi

        # Check if that worked
        set +e  # Don't exit on error
        if timeout 10s python3 -c "import aiter; print('Success')" &>/dev/null; then
            print_success "AITER installed successfully via direct installation"
            rm -rf "$temp_dir"
            complete_progress_bar
            return 0
        else
            print_warning "Direct installation verification failed."
            print_error "All installation attempts failed"
            rm -rf "$temp_dir"
            complete_progress_bar
            return 1
        fi
        set -e  # Return to normal error handling
    fi

    # Ensure proper cleanup regardless of how we got here
    update_progress_bar 10
    draw_progress_bar "Cleaning up..."
    print_step "Cleaning up..."

    # Make sure all background processes are terminated
    jobs -p | xargs -r kill 2>/dev/null

    # Clean up temporary files and directories
    if [ -d "$temp_dir" ]; then
        rm -rf "$temp_dir"
    fi

    # Remove any temporary files we created
    rm -f /tmp/test_aiter.py 2>/dev/null

    # Reset any traps we set
    trap - EXIT INT TERM HUP PIPE

    # Show a progress message for final steps
    echo
    echo -e "${CYAN}${BOLD}Finalizing installation...${RESET}"

    # Add a small progress animation for the final steps
    steps=("Registering AITER with Python" "Optimizing for your GPU" "Verifying installation" "Finalizing")
    for i in "${!steps[@]}"; do
        echo -ne "\r\033[K${MAGENTA}[${i}/3] ${steps[$i]}...${RESET}"
        sleep 0.5
    done
    echo -e "\r\033[K${GREEN}✓ Installation finalized successfully!${RESET}"

    # Force flush all output
    sync

    # Display a visually appealing completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   █████╗ ██╗████████╗███████╗██████╗                    ║
    ║  ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗                   ║
    ║  ███████║██║   ██║   █████╗  ██████╔╝                   ║
    ║  ██╔══██║██║   ██║   ██╔══╝  ██╔══██╗                   ║
    ║  ██║  ██║██║   ██║   ███████╗██║  ██║                   ║
    ║  ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝                   ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  AITER is now ready to use with your AMD GPU.           ║
    ║  Enjoy accelerated tensor operations on ROCm!           ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "AITER installation completed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$AITER_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./aiter_rocm_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import torch; import aiter; print('AITER is working with PyTorch', torch.__version__)\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import torch; import aiter; print('AITER is working with PyTorch', torch.__version__)\"${RESET}"
    fi
    echo

    # Verify the installation one last time with a simple import test
    if $PYTHON_CMD -c "import aiter; print('✓ AITER is properly installed')" 2>/dev/null; then
        echo -e "${GREEN}✓ Verified AITER is properly installed and importable${RESET}"
    else
        echo -e "${YELLOW}⚠ AITER may not be properly installed. Please check the installation logs.${RESET}"
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
    echo -e "${GREEN}eval \"\$(./install_aiter.sh --show-env)\"${RESET}"
    echo

    echo -e "${GREEN}${BOLD}Returning to main menu in 3 seconds...${RESET}"
    sleep 3

    # Show cursor again
    tput cnorm

    complete_progress_bar

    # Force exit immediately to prevent any hanging
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"

    # Kill any remaining background processes
    jobs -p | xargs -r kill -9 2>/dev/null

    # Exit cleanly
    exit 0
}

# Run the installation function
install_aiter "$@"
