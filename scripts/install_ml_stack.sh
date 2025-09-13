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
# ML Stack Installation Script
# =============================================================================
# This script installs the ML Stack for AMD GPUs.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# Trap to ensure we exit properly
trap 'echo "Forced exit"; kill -9 $$' EXIT

# Set Python interpreter - prefer virtual environment if available
PYTHON_INTERPRETER="python3"

# Check for virtual environment
if [ -f "$HOME/Prod/Stan-s-ML-Stack/venv/bin/python" ]; then
    PYTHON_INTERPRETER="$HOME/Prod/Stan-s-ML-Stack/venv/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_INTERPRETER="./venv/bin/python"
fi

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                ML Stack Installation Script
EOF
echo

# Enhanced color support with NO_COLOR detection
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

# Function to detect WSL environment
detect_wsl() {
    if [ -f "/proc/version" ] && grep -q "Microsoft" "/proc/version"; then
        echo "true"
    elif [ -f "/proc/version" ] && grep -q "microsoft" "/proc/version"; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect container environment
detect_container() {
    if [ -f "/.dockerenv" ]; then
        echo "docker"
    elif grep -q "container" "/proc/1/cgroup" 2>/dev/null; then
        echo "container"
    elif [ -n "$CONTAINER" ]; then
        echo "container"
    else
        echo "bare-metal"
    fi
}

# Function to detect GPU architecture for optimal configuration
detect_gpu_architecture() {
    if command_exists rocminfo; then
        # Try to detect GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep "Name:" | head -n 1 | grep -o "gfx[0-9a-z]*" || echo "")
        if [ -n "$gpu_arch" ]; then
            echo "$gpu_arch"
            return 0
        fi
    fi

    # Fallback to common architectures based on ROCm version
    case $rocm_version in
        6.4*|6.3*)
            echo "gfx1100"  # RDNA3
            ;;
        6.2*|6.1*|6.0*)
            echo "gfx1030"  # RDNA2
            ;;
        5.*)
            echo "gfx906"   # Vega
            ;;
        *)
            echo "gfx1100"  # Default to latest
            ;;
    esac
}

# Function to retry commands with exponential backoff
retry_command() {
    local cmd="$1"
    local max_attempts="${2:-3}"
    local base_delay="${3:-1}"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt/$max_attempts: $cmd"

        if eval "$cmd"; then
            return 0
        else
            if [ $attempt -eq $max_attempts ]; then
                print_error "Command failed after $max_attempts attempts: $cmd"
                return 1
            fi

            local delay=$((base_delay * (2 ** (attempt - 1))))
            print_warning "Command failed, retrying in $delay seconds..."
            sleep $delay
            ((attempt++))
        fi
    done
}

# Function to check return codes and handle errors gracefully
check_return_code() {
    local return_code=$1
    local error_message="$2"
    local non_critical="${3:-false}"

    if [ $return_code -ne 0 ]; then
        if [ "$non_critical" = "true" ]; then
            print_warning "$error_message (non-critical, continuing)"
            return 0
        else
            print_error "$error_message"
            return 1
        fi
    fi

    return 0
}

# Function to create configuration file
create_config_file() {
    local config_file="$HOME/.ml_stack_config"

    if [ ! -f "$config_file" ]; then
        print_step "Creating configuration file: $config_file"
        cat > "$config_file" << EOF
# ML Stack Configuration File
# Generated by ML Stack Installation Script

# Installation preferences
INSTALL_METHOD="${INSTALL_METHOD:-auto}"
FORCE_REINSTALL="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"

# ROCm configuration
ROCM_VERSION="$rocm_version"
GPU_ARCHITECTURE="$gpu_arch"
HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION"
PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH"

# Environment detection
WSL_DETECTED="$wsl_detected"
CONTAINER_TYPE="$container_type"

# Package manager
PACKAGE_MANAGER="$(detect_package_manager)"

# Virtual environment (if used)
PYTORCH_VENV_PYTHON="$PYTORCH_VENV_PYTHON"
EOF
        print_success "Configuration file created"
    fi
}

# Function to create backup for rollback
create_backup() {
    local component="$1"
    local backup_dir="$HOME/.ml_stack_backup/$(date +%Y%m%d_%H%M%S)_${component}"

    print_step "Creating backup for $component..."
    mkdir -p "$backup_dir"

    # Backup Python packages
    if command_exists uv; then
        uv pip freeze > "$backup_dir/requirements.txt" 2>/dev/null || true
    else
        python3 -m pip freeze > "$backup_dir/requirements.txt" 2>/dev/null || true
    fi

    # Backup environment variables
    env | grep -E "(ROCM|PYTORCH|HSA|CUDA)" > "$backup_dir/environment_vars.txt" 2>/dev/null || true

    # Backup configuration files
    cp -r "$HOME/.ml_stack_config" "$backup_dir/" 2>/dev/null || true

    print_success "Backup created: $backup_dir"
    echo "$backup_dir"
}

# Function to rollback installation
rollback_installation() {
    local component="$1"
    local backup_dir="$2"

    if [ ! -d "$backup_dir" ]; then
        print_error "Backup directory not found: $backup_dir"
        return 1
    fi

    print_warning "Rolling back $component installation..."

    # Restore Python packages
    if [ -f "$backup_dir/requirements.txt" ]; then
        print_step "Restoring Python packages..."
        if command_exists uv; then
            uv pip install -r "$backup_dir/requirements.txt" --quiet
        else
            python3 -m pip install -r "$backup_dir/requirements.txt" --quiet
        fi
    fi

    # Restore environment variables
    if [ -f "$backup_dir/environment_vars.txt" ]; then
        print_step "Restoring environment variables..."
        while IFS='=' read -r key value; do
            export "$key=$value"
        done < "$backup_dir/environment_vars.txt"
    fi

    # Restore configuration
    if [ -f "$backup_dir/.ml_stack_config" ]; then
        cp "$backup_dir/.ml_stack_config" "$HOME/"
    fi

    print_success "Rollback completed for $component"
}

# Function to handle dry run mode
dry_run_command() {
    local cmd="$1"
    if [ "$DRY_RUN" = "true" ]; then
        print_step "[DRY RUN] Would execute: $cmd"
        return 0
    else
        eval "$cmd"
        return $?
    fi
}

check_prerequisites() {
    print_section "Checking prerequisites"

    echo -e "${CYAN}${BOLD}System Requirements Check${RESET}"
    print_separator

    # Enhanced ROCm detection with automatic installation
    print_section "ROCm Detection and Setup"

    if ! command_exists rocminfo; then
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
    print_success "ROCm is installed"

    # Enhanced ROCm version detection with regex patterns
    rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)

    # If rocminfo didn't work, try alternative methods
    if [ -z "$rocm_version" ]; then
        # Try getting version from rocm-smi
        if command_exists rocm-smi; then
            rocm_version=$(rocm-smi --showversion 2>/dev/null | grep -i "ROCm Version" | awk '{print $3}')
        fi
    fi

    # If still no version, try checking the directory name with regex
    if [ -z "$rocm_version" ]; then
        if [ -d "/opt/rocm" ]; then
            # Try to get version from a symlink
            if [ -L "/opt/rocm" ]; then
                rocm_target=$(readlink -f /opt/rocm)
                rocm_version=$(echo "$rocm_target" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo "")
            fi

            # If still no version, check for version file
            if [ -z "$rocm_version" ] && [ -f "/opt/rocm/.info/version" ]; then
                rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo "")
            fi

            # If still no version, check for version-dev file
            if [ -z "$rocm_version" ] && [ -f "/opt/rocm/.info/version-dev" ]; then
                rocm_version=$(cat /opt/rocm/.info/version-dev 2>/dev/null || echo "")
            fi
        fi
    fi

    # If still no version, use a hardcoded version based on common ROCm versions
    if [ -z "$rocm_version" ]; then
        rocm_version="6.4.0"  # Default to a recent version
        print_warning "Could not detect ROCm version automatically. Using default: $rocm_version"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

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

    print_separator
    echo -e "${CYAN}${BOLD}Software Dependencies${RESET}"
    print_separator

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        return 1
    fi
    print_success "Python 3 is installed"

    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"

    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3 first."
        return 1
    fi
    print_success "pip3 is installed"

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

    print_separator
    echo -e "${CYAN}${BOLD}Cross-Platform Detection${RESET}"
    print_separator

    # Detect environment
    wsl_detected=$(detect_wsl)
    container_type=$(detect_container)

    if [ "$wsl_detected" = "true" ]; then
        print_warning "WSL environment detected"
        print_step "WSL-specific configurations will be applied"
    fi

    if [ "$container_type" != "bare-metal" ]; then
        print_warning "Container environment detected: $container_type"
        print_step "Container-specific configurations will be applied"
    fi

    print_separator
    echo -e "${CYAN}${BOLD}GPU Detection${RESET}"
    print_separator

    # Check if AMD GPUs are detected - redirect stderr to avoid "Tool lib 1 failed to load" messages
    if ! rocminfo 2>/dev/null | grep -q "Device Type:.*GPU"; then
        print_error "No AMD GPUs detected. Please check your hardware and ROCm installation."
        return 1
    fi

    # Count AMD GPUs
    gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
    print_success "Detected $gpu_count AMD GPU(s)"

    # Detect GPU architecture for optimal configuration
    gpu_arch=$(detect_gpu_architecture)
    print_success "Detected GPU architecture: $gpu_arch"

    # Update PYTORCH_ROCM_ARCH based on detected architecture
    export PYTORCH_ROCM_ARCH="$gpu_arch"

    # List AMD GPUs
    echo -e "${CYAN}Detected GPU Hardware:${RESET}"

    # Create a temporary file to store GPU information
    tmp_gpu_info=$(mktemp)

    # Extract GPU information to the temporary file
    rocminfo 2>/dev/null > "$tmp_gpu_info"

    # Process GPU information
    gpu_index=0
    while read -r line; do
        gpu_name=$(echo "$line" | awk -F: '{print $2}' | xargs)
        echo -e "  ${GREEN}▸ GPU $gpu_index:${RESET} $gpu_name"
        gpu_index=$((gpu_index+1))
    done < <(grep "Marketing Name" "$tmp_gpu_info")

    # Clean up
    rm -f "$tmp_gpu_info"

    print_separator
    echo -e "${CYAN}${BOLD}Environment Variables${RESET}"
    print_separator

    # Set environment variables if not set
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        print_warning "HIP_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi

    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        print_warning "CUDA_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi

    if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        print_warning "PYTORCH_ROCM_DEVICE is not set. Setting to all GPUs..."
        export PYTORCH_ROCM_DEVICE=$(seq -s, 0 $((gpu_count-1)))
    fi

    # Set ROCm-specific environment variables
    if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
    fi

    if [ -z "$PYTORCH_ROCM_ARCH" ]; then
        export PYTORCH_ROCM_ARCH="$gpu_arch"
    fi

    if [ -z "$ROCM_PATH" ]; then
        export ROCM_PATH="/opt/rocm"
    fi

    # Update PATH and LD_LIBRARY_PATH
    if [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
        export PATH="/opt/rocm/bin:$PATH"
    fi

    if [[ ":$LD_LIBRARY_PATH:" != *":/opt/rocm/lib:"* ]]; then
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
    fi

    echo -e "${CYAN}Current Environment Configuration:${RESET}"
    echo -e "  ${GREEN}▸ HIP_VISIBLE_DEVICES:${RESET} $HIP_VISIBLE_DEVICES"
    echo -e "  ${GREEN}▸ CUDA_VISIBLE_DEVICES:${RESET} $CUDA_VISIBLE_DEVICES"
    echo -e "  ${GREEN}▸ PYTORCH_ROCM_DEVICE:${RESET} $PYTORCH_ROCM_DEVICE"
    echo -e "  ${GREEN}▸ HSA_OVERRIDE_GFX_VERSION:${RESET} $HSA_OVERRIDE_GFX_VERSION"
    echo -e "  ${GREEN}▸ PYTORCH_ROCM_ARCH:${RESET} $PYTORCH_ROCM_ARCH"
    echo -e "  ${GREEN}▸ ROCM_PATH:${RESET} $ROCM_PATH"

    print_separator
    echo -e "${CYAN}${BOLD}Disk Space Check${RESET}"
    print_separator

    # Check disk space
    available_space=$(df -h $HOME | awk 'NR==2 {print $4}')
    total_space=$(df -h $HOME | awk 'NR==2 {print $2}')
    used_space=$(df -h $HOME | awk 'NR==2 {print $3}')
    used_percent=$(df -h $HOME | awk 'NR==2 {print $5}')

    echo -e "${CYAN}Storage Information:${RESET}"
    echo -e "  ${GREEN}▸ Total Space:${RESET}     $total_space"
    echo -e "  ${GREEN}▸ Used Space:${RESET}      $used_space ($used_percent)"
    echo -e "  ${GREEN}▸ Available Space:${RESET} $available_space"

    # Check if there's enough disk space (at least 20GB)
    available_space_kb=$(df -k $HOME | awk 'NR==2 {print $4}')
    if [ $available_space_kb -lt 20971520 ]; then  # 20GB in KB
        print_warning "You have less than 20GB of free disk space. Some components might fail to build."
        read -p "Do you want to continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation aborted by user."
            return 1
        fi
    else
        print_success "Sufficient disk space available for installation"
    fi

    return 0
}
install_rocm_config() {
    print_section "Installing ROCm configuration"

    # Create backup before installation
    local backup_dir=""
    if [ "$DRY_RUN" != "true" ]; then
        backup_dir=$(create_backup "rocm_config")
    fi

    # Create ROCm configuration file
    print_step "Creating ROCm configuration file..."

    # Create .rocmrc file in home directory
    if [ "$DRY_RUN" = "true" ]; then
        print_step "[DRY RUN] Would create ROCm configuration file at $HOME/.rocmrc"
    else
        cat > $HOME/.rocmrc << EOF
# ROCm Configuration File
# Created by ML Stack Installation Script

# Environment Variables
export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE
export HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION
export PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH
export ROCM_PATH=$ROCM_PATH
export HSA_TOOLS_LIB=$HSA_TOOLS_LIB

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# Logging Settings
export HIP_TRACE_API=0
export AMD_LOG_LEVEL=4
EOF

        # Add source to .bashrc if not already there
        if ! grep -q "source \$HOME/.rocmrc" $HOME/.bashrc; then
            echo -e "\n# Source ROCm configuration" >> $HOME/.bashrc
            echo "source \$HOME/.rocmrc" >> $HOME/.bashrc
        fi

        # Source the file
        source $HOME/.rocmrc
    fi

    print_success "ROCm configuration installed successfully"
}

install_pytorch() {
    print_section "Installing PyTorch with ROCm support"

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
        if command_exists uv; then
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
    if [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
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
    print_section "Verifying PyTorch Installation"

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

    return 0
}

install_onnx_runtime() {
    print_section "Installing ONNX Runtime with ROCm support"

    # Create backup before installation
    local backup_dir=""
    if [ "$DRY_RUN" != "true" ]; then
        backup_dir=$(create_backup "onnxruntime")
    fi

    # Check if ONNX Runtime is already installed
    if python3 -c "import onnxruntime" &> /dev/null; then
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
        print_warning "ONNX Runtime is already installed (version $onnx_version)."

        # Check if ROCMExecutionProvider is available
        if python3 -c "import onnxruntime; print('ROCMExecutionProvider' in onnxruntime.get_available_providers())" | grep -q "True"; then
            print_success "ONNX Runtime with ROCm support is already installed."
            if [[ "$*" != *"--force"* ]] && [ "$FORCE" != "true" ]; then
                read -p "Do you want to reinstall? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_step "Skipping ONNX Runtime installation."
                    return 0
                fi
            fi
        else
            print_warning "ONNX Runtime is installed but ROCMExecutionProvider is not available."
            print_step "Reinstalling ONNX Runtime with ROCm support..."
        fi
    fi

    # Build and install ONNX Runtime with ROCm support
    print_step "Building ONNX Runtime with ROCm support..."

    # Run the build script with retry mechanism
    if ! retry_command "$HOME/Prod/Stan-s-ML-Stack/scripts/build_onnxruntime.sh" 2; then
        print_error "ONNX Runtime installation failed after retries."
        # Rollback if installation failed
        if [ -n "$backup_dir" ]; then
            rollback_installation "onnxruntime" "$backup_dir"
        fi
        return 1
    fi

    print_success "ONNX Runtime with ROCm support installed successfully"
    return 0
}

install_migraphx() {
    print_section "Installing MIGraphX"

    # Create backup before installation
    local backup_dir=""
    if [ "$DRY_RUN" != "true" ]; then
        backup_dir=$(create_backup "migraphx")
    fi

    # Check if MIGraphX is already installed
    if python3 -c "import migraphx" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_warning "MIGraphX is already installed (version $migraphx_version)."
        if [[ "$*" != *"--force"* ]] && [ "$FORCE" != "true" ]; then
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping MIGraphX installation."
                return 0
            fi
        fi
    fi

    # Install MIGraphX from ROCm repository
    print_step "Installing MIGraphX from ROCm repository..."

    # Use detected package manager
    package_manager=$(detect_package_manager)
    case $package_manager in
        apt)
            dry_run_command "sudo apt-get update"
            dry_run_command "sudo apt-get install -y migraphx python3-migraphx"
            ;;
        dnf)
            dry_run_command "sudo dnf install -y migraphx python3-migraphx"
            ;;
        yum)
            dry_run_command "sudo yum install -y migraphx python3-migraphx"
            ;;
        pacman)
            dry_run_command "sudo pacman -S migraphx python-migraphx"
            ;;
        zypper)
            dry_run_command "sudo zypper install -y migraphx python3-migraphx"
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            return 1
            ;;
    esac

    # Verify installation
    if [ "$DRY_RUN" != "true" ] && python3 -c "import migraphx; print(migraphx.__version__)" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_success "MIGraphX installed successfully (version $migraphx_version)"
    elif [ "$DRY_RUN" = "true" ]; then
        print_success "[DRY RUN] MIGraphX installation simulated"
    else
        print_error "MIGraphX installation failed."
        # Rollback if installation failed
        if [ -n "$backup_dir" ]; then
            rollback_installation "migraphx" "$backup_dir"
        fi
        return 1
    fi

    return 0
}
install_megatron() {
    print_section "Installing Megatron-LM"

    # Check if Megatron-LM is already installed
    if $PYTHON_INTERPRETER -c "import megatron" &> /dev/null; then
        print_warning "Megatron-LM is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping Megatron-LM installation."
            return 0
        fi
    fi

    # Clone Megatron-LM repository
    print_step "Cloning Megatron-LM repository..."
    cd $HOME
    if [ -d "Megatron-LM" ]; then
        print_warning "Megatron-LM repository already exists. Updating..."
        cd Megatron-LM
        git pull
        cd ..
    else
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM
    fi

    # Create patch file to remove NVIDIA-specific dependencies
    print_step "Creating patch file to remove NVIDIA-specific dependencies..."
    cat > remove_nvidia_deps.patch << 'EOF'
diff --git a/megatron/model/fused_softmax.py b/megatron/model/fused_softmax.py
index 7a5b2e5..3e5c2e5 100644
--- a/megatron/model/fused_softmax.py
+++ b/megatron/model/fused_softmax.py
@@ -15,7 +15,7 @@
 """Fused softmax."""

 import torch
-import torch.nn.functional as F
+import torch.nn.functional as F  # Use PyTorch's native implementation


 class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
@@ -24,8 +24,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             scale (float): scaling factor

         Returns:
@@ -33,10 +32,10 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
-        # Create a mask for the upper triangular part (including the diagonal)
+        # Create a mask for the upper triangular part
         seq_len = inputs.size(2)
         mask = torch.triu(
             torch.ones(seq_len, seq_len, device=inputs.device, dtype=torch.bool),
@@ -59,7 +58,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
@@ -77,8 +76,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, mask, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             mask (Tensor): attention mask (b, 1, sq, sk)
             scale (float): scaling factor

@@ -87,7 +85,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
         # Apply the mask
@@ -110,7 +108,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
diff --git a/megatron/training.py b/megatron/training.py
index 9a5b2e5..3e5c2e5 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -30,7 +30,7 @@ import torch
 from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

 from megatron import get_args
-from megatron import get_timers
+from megatron import get_timers  # Timing utilities
 from megatron import get_tensorboard_writer
 from megatron import mpu
 from megatron import print_rank_0
@@ -38,7 +38,7 @@ from megatron.checkpointing import load_checkpoint
 from megatron.checkpointing import save_checkpoint
 from megatron.model import DistributedDataParallel as LocalDDP
 from megatron.model import Float16Module
-from megatron.model.realm_model import ICTBertModel
+from megatron.model.realm_model import ICTBertModel  # Import model
 from megatron.utils import check_adlr_autoresume_termination
 from megatron.utils import unwrap_model
 from megatron.data.data_samplers import build_pretraining_data_loader
@@ -46,7 +46,7 @@ from megatron.utils import report_memory


 def pretrain(train_valid_test_dataset_provider, model_provider,
-             forward_step_func, extra_args_provider=None, args_defaults={}):
+             forward_step_func, extra_args_provider=None, args_defaults=None):
     """Main training program.

     This function will run the followings in the order provided:
@@ -59,6 +59,9 @@ def pretrain(train_valid_test_dataset_provider, model_provider,
         5) validation

     """
+    if args_defaults is None:
+        args_defaults = {}
+
     # Initalize and get arguments, timers, and Tensorboard writer.
     initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
diff --git a/requirements.txt b/requirements.txt
index 9a5b2e5..3e5c2e5 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,5 @@
 torch>=1.7
 numpy
-apex
 pybind11
 regex
 nltk
EOF

    # Apply patch
    print_step "Applying patch..."
    git apply remove_nvidia_deps.patch

    # Install dependencies using enhanced package management
    print_step "Installing dependencies..."
    install_python_package -r requirements.txt
    install_python_package tensorboard scipy

    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    install_python_package -e .

    # Verify installation
    if python3 -c "import megatron; print('Megatron-LM imported successfully')" &> /dev/null; then
        print_success "Megatron-LM installed successfully"
    else
        print_error "Megatron-LM installation failed."
        return 1
    fi

    return 0
}

install_flash_attention() {
    print_section "Installing Flash Attention with AMD GPU support"

    # Create backup before installation
    local backup_dir=""
    if [ "$DRY_RUN" != "true" ]; then
        backup_dir=$(create_backup "flash_attention")
    fi

    # Check if Flash Attention is already installed
    if $PYTHON_INTERPRETER -c "import flash_attention_amd" &> /dev/null; then
        print_warning "Flash Attention with AMD GPU support is already installed."
        if [[ "$*" != *"--force"* ]] && [ "$FORCE" != "true" ]; then
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping Flash Attention installation."
                return 0
            fi
        fi
    fi

    # Build and install Flash Attention with AMD GPU support
    print_step "Building Flash Attention with AMD GPU support..."

    # Run the build script with retry mechanism
    if ! retry_command "$HOME/Prod/Stan-s-ML-Stack/scripts/build_flash_attn_amd.sh" 2; then
        print_error "Flash Attention installation failed after retries."
        # Rollback if installation failed
        if [ -n "$backup_dir" ]; then
            rollback_installation "flash_attention" "$backup_dir"
        fi
        return 1
    fi

    print_success "Flash Attention with AMD GPU support installed successfully"
    return 0
}

install_rccl() {
    print_section "Installing RCCL"

    # Create backup before installation
    local backup_dir=""
    if [ "$DRY_RUN" != "true" ]; then
        backup_dir=$(create_backup "rccl")
    fi

    # Check if RCCL is already installed
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_warning "RCCL is already installed."
        if [[ "$*" != *"--force"* ]] && [ "$FORCE" != "true" ]; then
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping RCCL installation."
                return 0
            fi
        fi
    fi

    # Install RCCL from ROCm repository
    print_step "Installing RCCL from ROCm repository..."

    # Use detected package manager
    package_manager=$(detect_package_manager)
    case $package_manager in
        apt)
            dry_run_command "sudo apt-get update"
            dry_run_command "sudo apt-get install -y rccl"
            ;;
        dnf)
            dry_run_command "sudo dnf install -y rccl"
            ;;
        yum)
            dry_run_command "sudo yum install -y rccl"
            ;;
        pacman)
            dry_run_command "sudo pacman -S rccl"
            ;;
        zypper)
            dry_run_command "sudo zypper install -y rccl"
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            return 1
            ;;
    esac

    # Verify installation
    if [ "$DRY_RUN" != "true" ] && [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_success "RCCL installed successfully"
    elif [ "$DRY_RUN" = "true" ]; then
        print_success "[DRY RUN] RCCL installation simulated"
    else
        print_error "RCCL installation failed."
        # Rollback if installation failed
        if [ -n "$backup_dir" ]; then
            rollback_installation "rccl" "$backup_dir"
        fi
        return 1
    fi

    return 0
}

install_mpi() {
    print_section "Installing MPI"

    # Check if MPI is already installed
    if command -v mpirun &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_warning "MPI is already installed ($mpi_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping MPI installation."
            return 0
        fi
    fi

    # Install OpenMPI
    print_step "Installing OpenMPI..."
    sudo apt-get update
    sudo apt-get install -y openmpi-bin libopenmpi-dev

    # Configure OpenMPI for ROCm
    print_step "Configuring OpenMPI for ROCm..."

    # Create MPI configuration file
    cat > $HOME/.mpirc << 'EOF'
# MPI Configuration File
# Created by ML Stack Installation Script

# OpenMPI Configuration
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml_ucx_opal_cuda_support=true
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

# Performance Tuning
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib,uct
EOF

    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.mpirc" $HOME/.bashrc; then
        echo -e "\n# Source MPI configuration" >> $HOME/.bashrc
        echo "source \$HOME/.mpirc" >> $HOME/.bashrc
    fi

    # Source the file
    source $HOME/.mpirc

    # Install mpi4py
    print_step "Installing mpi4py..."
    install_python_package mpi4py

    # Verify installation
    if command -v mpirun &> /dev/null && python3 -c "import mpi4py; print('mpi4py imported successfully')" &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_success "MPI installed successfully ($mpi_version)"
    else
        print_error "MPI installation failed."
        return 1
    fi

    return 0
}
install_all_core() {
    print_section "Installing all core components"

    # Create configuration file
    create_config_file

    # Install ROCm configuration
    print_step "Installing ROCm configuration..."
    if ! install_rocm_config; then
        check_return_code 1 "ROCm configuration installation failed" || return 1
    fi

    # Install PyTorch
    print_step "Installing PyTorch..."
    if ! install_pytorch; then
        check_return_code 1 "PyTorch installation failed" || return 1
    fi

    # Install ONNX Runtime
    print_step "Installing ONNX Runtime..."
    if ! install_onnx_runtime; then
        check_return_code 1 "ONNX Runtime installation failed" || return 1
    fi

    # Install MIGraphX
    print_step "Installing MIGraphX..."
    if ! install_migraphx; then
        check_return_code 1 "MIGraphX installation failed" || return 1
    fi

    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    if ! install_megatron; then
        check_return_code 1 "Megatron-LM installation failed" || return 1
    fi

    # Install Flash Attention
    print_step "Installing Flash Attention..."
    if ! install_flash_attention; then
        check_return_code 1 "Flash Attention installation failed" || return 1
    fi

    # Install RCCL
    print_step "Installing RCCL..."
    if ! install_rccl; then
        check_return_code 1 "RCCL installation failed" || return 1
    fi

    # Install MPI
    print_step "Installing MPI..."
    if ! install_mpi; then
        check_return_code 1 "MPI installation failed" || return 1
    fi

    print_success "All core components installed successfully"
    return 0
}

verify_installation() {
    print_section "Verifying installation with comprehensive tests"

    # Create enhanced verification script
    print_step "Creating enhanced verification script..."
    cat > ./verify_ml_stack_enhanced.py << EOF
import sys
import os
import importlib.util
import time
import subprocess

# Color definitions
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"{CYAN}{BOLD}=== {text} ==={RESET}")
    print()

def print_section(text):
    print(f"{BLUE}{BOLD}>>> {text}{RESET}")

def print_step(text):
    print(f"{MAGENTA}>> {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def check_module(module_name, display_name=None):
    if display_name is None:
        display_name = module_name

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print_success(f"{display_name} is installed (version: {version})")
        return module
    except ImportError:
        print_error(f"{display_name} is not installed")
        return None

def benchmark_pytorch():
    """Comprehensive PyTorch benchmarking"""
    print_section("PyTorch Performance Benchmark")

    try:
        import torch

        if not torch.cuda.is_available():
            print_warning("CUDA not available, skipping GPU benchmarks")
            return

        device = torch.device("cuda")

        # Memory bandwidth test
        print_step("Testing memory bandwidth...")
        size = 1024 * 1024 * 256  # 1GB
        x = torch.randn(size, device=device)
        start_time = time.time()
        for _ in range(10):
            y = x * 2
        torch.cuda.synchronize()
        bandwidth = (size * 4 * 10) / (time.time() - start_time) / (1024**3)  # GB/s
        print_success(".2f")

        # Matrix multiplication benchmark
        print_step("Testing matrix multiplication...")
        sizes = [1024, 2048, 4096]
        for n in sizes:
            a = torch.randn(n, n, device=device)
            b = torch.randn(n, n, device=device)
            start_time = time.time()
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            flops = (2 * n**3) / (time.time() - start_time) / (10**9)  # GFLOPS
            print_success(".1f")

        # Flash Attention test (if available)
        print_step("Testing Flash Attention...")
        try:
            from flash_attention_amd import flash_attn_func
            batch_size, seq_len, n_heads, d_head = 2, 1024, 8, 64
            q = torch.randn(batch_size, seq_len, n_heads, d_head, device=device)
            k = torch.randn(batch_size, seq_len, n_heads, d_head, device=device)
            v = torch.randn(batch_size, seq_len, n_heads, d_head, device=device)

            start_time = time.time()
            out = flash_attn_func(q, k, v)
            torch.cuda.synchronize()
            attn_time = time.time() - start_time
            print_success(".3f")
        except ImportError:
            print_warning("Flash Attention not available for benchmarking")

    except Exception as e:
        print_error(f"Benchmark failed: {e}")

def test_distributed_training():
    """Test distributed training capabilities"""
    print_section("Distributed Training Test")

    try:
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp

        if torch.cuda.device_count() < 2:
            print_warning("Need at least 2 GPUs for distributed training test")
            return

        def init_process(rank, size, fn, backend='nccl'):
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend, rank=rank, world_size=size)
            fn(rank, size)

        def run_test(rank, size):
            tensor = torch.ones(1, device=f'cuda:{rank}')
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = torch.ones(1, device=f'cuda:{rank}') * size
            if torch.allclose(tensor, expected):
                if rank == 0:
                    print_success("Distributed training test passed")
            else:
                if rank == 0:
                    print_error("Distributed training test failed")

        print_step("Testing NCCL distributed training...")
        mp.spawn(init_process, args=(2, run_test), nprocs=2, join=True)

    except Exception as e:
        print_error(f"Distributed training test failed: {e}")

def check_system_configuration():
    """Check system configuration and environment"""
    print_section("System Configuration Check")

    # Check ROCm environment variables
    rocm_vars = ['HSA_OVERRIDE_GFX_VERSION', 'PYTORCH_ROCM_ARCH', 'ROCM_PATH', 'HSA_TOOLS_LIB']
    for var in rocm_vars:
        value = os.environ.get(var, 'Not set')
        print_step(f"{var}: {value}")

    # Check GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_free, mem_total = torch.cuda.mem_get_info(i)
                mem_free_gb = mem_free / (1024**3)
                mem_total_gb = mem_total / (1024**3)
                print_step(".1f")
    except:
        pass

def main():
    print_header("Enhanced ML Stack Verification")

    # Basic component checks
    print_section("Component Verification")

    # Check PyTorch
    torch = check_module("torch", "PyTorch")
    if torch:
        has_hip = hasattr(torch.version, 'hip')
        if has_hip:
            print_success(f"PyTorch has ROCm/HIP support (version: {torch.version.hip})")
        else:
            print_warning("PyTorch does not have explicit ROCm/HIP support")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print_step(f"Number of GPUs: {device_count}")

            for i in range(device_count):
                print_step(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # Test basic GPU operations
            try:
                x = torch.ones(10, device="cuda")
                y = x + 1
                print_success("Basic GPU tensor operations working")
            except Exception as e:
                print_error(f"GPU operations failed: {e}")
        else:
            print_error("GPU acceleration is not available")

    # Check other components
    check_module("onnxruntime", "ONNX Runtime")
    check_module("migraphx", "MIGraphX")
    check_module("mpi4py", "MPI4Py")

    # Check Megatron-LM
    try:
        import megatron
        print_success("Megatron-LM is installed")
    except ImportError:
        print_error("Megatron-LM is not installed")

    # Check Flash Attention
    try:
        from flash_attention_amd import flash_attn_func
        print_success("Flash Attention AMD is installed")
    except ImportError:
        print_error("Flash Attention AMD is not installed")

    # System configuration
    check_system_configuration()

    # Performance benchmarks
    benchmark_pytorch()

    # Distributed training test
    test_distributed_training()

    print_header("Verification Complete")

if __name__ == "__main__":
    main()
EOF

    # Run enhanced verification script
    print_step "Running enhanced verification script..."
    if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        $PYTHON_INTERPRETER ./verify_ml_stack_enhanced.py
    else
        print_warning "GPU not available, running basic verification..."
        # Fallback to basic verification
        $PYTHON_INTERPRETER -c "
import sys
import importlib.util

def check_module(name):
    try:
        importlib.import_module(name)
        print(f'✓ {name} is installed')
        return True
    except ImportError:
        print(f'✗ {name} is not installed')
        return False

print('=== Basic ML Stack Verification ===')
check_module('torch')
check_module('onnxruntime')
check_module('migraphx')
check_module('mpi4py')
print('=== Verification Complete ===')
"
    fi

    # Clean up
    print_step "Cleaning up..."
    rm -f ./verify_ml_stack_enhanced.py

    print_success "Enhanced verification completed"
}

show_menu() {
    print_header "ML Stack Installation Menu"

    echo -e "1) Install ROCm Configuration"
    echo -e "2) Install PyTorch with ROCm support"
    echo -e "3) Install ONNX Runtime with ROCm support"
    echo -e "4) Install MIGraphX"
    echo -e "5) Install Megatron-LM"
    echo -e "6) Install Flash Attention with AMD GPU support"
    echo -e "7) Install RCCL"
    echo -e "8) Install MPI"
    echo -e "9) Install All Core Components"
    echo -e "10) Verify Installation"
    echo -e "0) Exit"
    echo

    read -p "Enter your choice: " choice

    case $choice in
        1)
            install_rocm_config
            ;;
        2)
            install_pytorch
            ;;
        3)
            install_onnx_runtime
            ;;
        4)
            install_migraphx
            ;;
        5)
            install_megatron
            ;;
        6)
            install_flash_attention
            ;;
        7)
            install_rccl
            ;;
        8)
            install_mpi
            ;;
        9)
            install_all_core
            ;;
        10)
            verify_installation
            ;;
        0)
            print_header "Exiting ML Stack Installation"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            ;;
    esac

    # Show menu again
    show_menu
}

main() {
    print_header "ML Stack Installation Script"

    # Parse command line arguments
    DRY_RUN=false
    FORCE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                print_warning "DRY RUN MODE: No actual installations will be performed"
                shift
                ;;
            --force)
                FORCE=true
                print_warning "FORCE MODE: Will reinstall existing components"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dry-run    Show what would be installed without actually installing"
                echo "  --force      Force reinstallation of existing components"
                echo "  --show-env   Show ROCm environment variables for manual setup"
                echo "  --help       Show this help message"
                echo ""
                echo "Enhanced ML Stack Installation Script with ROCm support"
                echo "Supports multiple package managers, virtual environments, and cross-platform compatibility"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Start time
    start_time=$(date +%s)

    # Check for --show-env option first (before prerequisites)
    for arg in "$@"; do
        if [ "$arg" = "--show-env" ]; then
            show_env
            exit 0
        fi
    done

    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi

    # Show menu
    show_menu

    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))

    print_header "ML Stack Installation Completed"
    echo -e "${GREEN}Total installation time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"

    # Show environment setup instructions
    echo
    echo -e "${CYAN}${BOLD}Environment Setup:${RESET}"
    echo -e "${YELLOW}To apply ROCm environment variables to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$($0 --show-env)\"${RESET}"
    echo
    echo -e "${CYAN}${BOLD}Configuration:${RESET}"
    echo -e "${YELLOW}Configuration saved to: ${GREEN}$HOME/.ml_stack_config${RESET}"

    return 0
}

# Main script execution
main

# Force exit to prevent hanging
echo "Installation complete. Forcing exit to prevent hanging..."
kill -9 $$ 2>/dev/null
exit 0

