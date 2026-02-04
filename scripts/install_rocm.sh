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
# ROCm Installation Script - v7.2.1-Robust-Purge
# =============================================================================
# This script installs AMD's ROCm platform for GPU computing.
# Fixed GPG signature verification issues on Debian Trixie (13).
# Supports ROCm 7.1 and 7.2 with updated frameworks from manylinux repositories.
# =============================================================================

# Parse command line arguments first to handle --show-env
DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    fi
done

# ASCII Art Banner (skip if --show-env is used)
if [[ "$*" != *"--show-env"* ]]; then
    cat << "EOF"
  ██████╗  ██████╗  ██████╗███╗   ███╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     
  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     
  ██████╔╝██║   ██║██║     ██╔████╔██║    ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     
  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     
  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
EOF
    echo
fi

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
    echo >&2
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}" >&2
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}║               === $1 ===               ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}" >&2
    echo >&2
}

print_section() {
    echo >&2
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}" >&2
    echo -e "${BLUE}${BOLD}│ $1${RESET}" >&2
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}" >&2
}

print_step() {
    echo -e "${MAGENTA}➤ $1${RESET}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}" >&2
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}" >&2
}

print_error() {
    echo -e "${RED}✗ $1${RESET}" >&2
}

# Function to print a clean separator line
print_separator() {
    echo -e "${BLUE}───────────────────────────────────────────────────────────${RESET}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to retry commands with exponential backoff
retry_command() {
    local cmd="$1"
    local max_attempts="${2:-3}"
    local delay="${3:-1}"
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

            print_warning "Command failed, retrying in $delay seconds..."
            sleep $delay
            delay=$((delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        fi
    done
}

# Function to execute a command with logging and dry-run support
execute_command() {
    local cmd="$1"
    local desc="$2"
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        echo -e "${YELLOW}[DRY-RUN]${RESET} ${desc}:" >&2
        echo -e "  ${BOLD}${cmd}${RESET}" >&2
        return 0
    fi
    
    print_step "$desc..."
    if eval "$cmd"; then
        print_success "Done: $desc"
        return 0
    else
        print_error "Failed: $desc"
        return 1
    fi
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

# Function to get ROCm version with better parsing
get_rocm_version() {
    local rocm_version=""

    # Method 1: Check version file first (most reliable for ROCm 7.0+)
    if [ -f "/opt/rocm/.info/version" ]; then
        rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null | grep -o '^[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    # Method 2: Try rocminfo if version file fails
    if [ -z "$rocm_version" ] && command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | sed -n 's/.*ROCm Version\s*:\s*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1)
    fi

    # Method 3: Check ROCm 7.0.0 directory specifically
    if [ -z "$rocm_version" ] && [ -d "/opt/rocm-7.0.0" ]; then
        if [ -f "/opt/rocm-7.0.0/.info/version" ]; then
            rocm_version=$(cat /opt/rocm-7.0.0/.info/version 2>/dev/null | grep -o '^[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        else
            rocm_version="7.0.0"  # Assume if directory exists
        fi
    fi

    # Method 4: Check apt package version
    if [ -z "$rocm_version" ]; then
        rocm_version=$(dpkg -l | grep "^ii.*rocm-dev" | awk '{print $3}' | grep -o '^[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    # Method 5: Directory listing fallback
    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | sort -V | tail -n 1)
    fi

    echo "$rocm_version"
}

# Function to detect GPU architecture
detect_gpu_architecture() {
    local gpu_arch=""

    if command_exists rocminfo; then
        # Try to get GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -o 'gfx[0-9]\+' | head -n 1)
    fi

    # Default to common architecture if not detected
    if [ -z "$gpu_arch" ]; then
        gpu_arch="gfx1100"  # Default for RDNA3
        print_warning "Could not detect GPU architecture, using default: $gpu_arch" >&2
    else
        print_success "Detected GPU architecture: $gpu_arch" >&2
    fi

    echo "$gpu_arch"
}

# Function to detect if running in WSL
detect_wsl() {
    if [ -f /proc/version ] && grep -qi "microsoft\|wsl" /proc/version; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect if running in a container
detect_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q "docker\|container" /proc/1/cgroup 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect ROCm installation path
detect_rocm_path() {
    local rocm_path=""

    # Check common ROCm installation paths
    for path in "/opt/rocm" "/usr/lib/rocm" "/opt/rocm-7.0" "/opt/rocm-6.4" "/opt/rocm-6.3" "/opt/rocm-6.2" "/opt/rocm-6.1" "/opt/rocm-6.0" "/opt/rocm-5.7"; do
        if [ -d "$path" ] && [ -f "$path/bin/rocminfo" ]; then
            rocm_path="$path"
            break
        fi
    done

    # Default fallback
    if [ -z "$rocm_path" ]; then
        rocm_path="/opt/rocm"
    fi

    echo "$rocm_path"
}

# Function to verify ROCm installation
verify_rocm_installation() {
    local expected_version="$1"
    
    print_step "Verifying ROCm installation..."
    
    local detected_version=$(get_rocm_version)
    local system_version=""
    local package_version=""
    
    # Get system version
    if [ -f "/opt/rocm/.info/version" ]; then
        system_version=$(cat /opt/rocm/.info/version 2>/dev/null)
        print_success "System ROCm version: $system_version"
    fi
    
    # Get package version
    package_version=$(dpkg -l | grep "^ii.*rocm-dev" | awk '{print $3}' | grep -o '^[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    if [ -n "$package_version" ]; then
        print_success "Package ROCm version: $package_version"
    fi
    
    # Check PyTorch ROCm version if available
    if command_exists python3; then
        local pytorch_rocm=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')" 2>/dev/null)
        if [ "$pytorch_rocm" != "N/A" ]; then
            print_step "PyTorch built against ROCm: $pytorch_rocm"
            if [[ "$pytorch_rocm" != "$system_version"* ]]; then
                print_warning "PyTorch ROCm version ($pytorch_rocm) differs from system ROCm ($system_version)"
                print_step "This is normal - PyTorch can run on newer ROCm versions than it was built for"
            fi
        fi
    fi
    
    # Verify GPU detection
    if command_exists rocminfo; then
        local gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            print_success "ROCm detected $gpu_count AMD GPU(s)"
            
            # Show GPU architectures
            rocminfo 2>/dev/null | grep -A 10 "Device Type:.*GPU" | grep "gfx" | while read -r line; do
                local gfx_arch=$(echo "$line" | grep -o 'gfx[0-9]\+')
                print_step "Detected GPU architecture: $gfx_arch"
            done
        else
            print_warning "ROCm installed but no GPUs detected"
        fi
    fi
    
    return 0
}

# Function to show environment variables with ASCII art banner
show_env() {
    # Show ASCII art banner first
    cat << "EOF"
  ██████╗  ██████╗  ██████╗███╗   ███╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     
  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     
  ██████╔╝██║   ██║██║     ██╔████╔██║    ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     
  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     
  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
EOF
    echo
    echo "ROCm Environment Variables:"
    echo "=========================="
    
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

# Function to show environment variables without ASCII art (for eval usage)
show_env_clean() {
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
    echo "export ROCM_VERSION=\"$ROCM_VERSION\""
    if [ -n "$ROCM_CHANNEL" ]; then
        echo "export ROCM_CHANNEL=\"$ROCM_CHANNEL\""
    fi
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
}

# Function to purge ROCm and AMDGPU packages
purge_rocm() {
    print_warning "Purging existing ROCm and AMDGPU packages to prevent dependency conflicts..."
    
    local pkg_manager=$(detect_package_manager)
    
    if [ "$pkg_manager" = "apt" ]; then
        # Use dpkg to find all ROCm and AMDGPU packages
        print_step "Identifying all ROCm and AMDGPU packages..."
        local all_pkgs=$(dpkg -l | grep -E "rocm|amdgpu|hsa-rocr|rocminfo|hip-runtime|miopen|rocblas|migraphx|rccl|comgr|mivisionx|rpp|half|librocclr|hsa-ext-rocr" | awk '{print $2}' || true)
        
        if [ -n "$all_pkgs" ]; then
            print_step "Force-purging $(echo "$all_pkgs" | wc -w) packages..."
            # Using dpkg directly avoids apt's dependency resolver which is currently stuck
            # We use --force-all because we want to nuke everything and start over
            sudo dpkg --purge --force-all $all_pkgs || true
        fi
        
        # Clean up any broken package states
        print_step "Fixing potential broken states..."
        sudo dpkg --configure -a || true
        sudo apt-get install -f -y || true
        
        # Clean up the apt state
        print_step "Cleaning up apt state..."
        sudo apt-get autoremove -y --purge || true
        sudo apt-get clean
        
        # Remove all possible ROCm and AMDGPU repository files
        print_step "Removing repository configurations..."
        sudo rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/amdgpu-proprietary.list /etc/apt/sources.list.d/amdgpu-install.list
        sudo rm -f /etc/apt/preferences.d/rocm-pin-600
        
        # Ensure no locks remain
        sudo dpkg --configure -a || true
        
        # Remove ROCm directories to prevent detection of partial installs
        print_step "Removing ROCm system directories..."
        sudo rm -rf /opt/rocm*
        
        sudo apt-get update
    elif [ "$pkg_manager" = "dnf" ] || [ "$pkg_manager" = "yum" ]; then
        execute_command "sudo $pkg_manager remove -y rocm-* amdgpu-*" "Removing ROCm and AMDGPU packages"
        execute_command "sudo $pkg_manager autoremove -y" "Removing unused dependencies"
    elif [ "$pkg_manager" = "zypper" ]; then
        execute_command "sudo zypper remove -y rocm-* amdgpu-*" "Removing ROCm and AMDGPU packages"
    fi
    
    print_success "System-wide ROCm purge completed."
}

# Function to setup GPG fix for Debian Trixie
setup_gpg_fix() {
    print_step "Setting up GPG fix for Debian Trixie..."

    # Remove existing broken setup
    execute_command "sudo rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu-proprietary.list" "Removing existing broken ROCm repository"
    execute_command "sudo rm -f /etc/apt/preferences.d/rocm-pin-600" "Removing existing ROCm preferences"
    execute_command "sudo rm -f /etc/apt/keyrings/rocm.gpg" "Removing existing ROCm GPG key"

    # Create apt configuration to use gpgv instead of sqv for ROCm repos
    execute_command "sudo mkdir -p /etc/apt/apt.conf.d" "Creating apt configuration directory"

    execute_command "sudo tee /etc/apt/apt.conf.d/99rocm-gpg-fix << 'EOF'
APT::Key::gpgvcommand \"/usr/bin/gpgv\";
EOF" "Creating GPG verification override for ROCm repositories"

    # Create keyrings directory
    execute_command "sudo mkdir --parents --mode=0755 /etc/apt/keyrings" "Creating keyrings directory"

    # Download and install ROCm GPG key
    execute_command "wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null" "Installing ROCm GPG key"

    # Set proper permissions on the key
    execute_command "sudo chmod 644 /etc/apt/keyrings/rocm.gpg" "Setting GPG key permissions"

    if [ $? -eq 0 ] && [ "$DRY_RUN" != true ]; then
        print_success "GPG fix applied for Debian Trixie"
    fi
}

# Function to validate system compatibility
validate_system() {
    local distributor="$1"
    local ubuntu_version="$2"

    print_step "Validating system compatibility..."

    # Check for known problematic configurations
    if [ "$distributor" = "Debian" ]; then
        if [ "$ubuntu_version" = "13" ] || [ "$ubuntu_version" = "unstable" ]; then
            print_warning "Debian Trixie (13) detected - GPG verification fix will be applied"
            print_step "This script includes fixes for Debian Trixie GPG signature issues"
        fi
    fi

    # Check for required tools
    local missing_tools=""
    for tool in wget gpg curl; do
        if ! command_exists "$tool"; then
            missing_tools="$missing_tools $tool"
        fi
    done

    if [ -n "$missing_tools" ]; then
        print_warning "Missing required tools:$missing_tools"
        print_step "Installing missing tools..."
        if [ "$package_manager" = "apt" ]; then
            execute_command "sudo apt update && sudo apt install -y wget gnupg curl" "Installing required tools"
        fi
    fi

    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended"
        print_step "Consider running as a regular user with sudo privileges"
    fi

    print_success "System validation completed"
}

# Function to show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

ROCm Installation Script for AMD GPUs
Enhanced with modern installation standards and comprehensive error handling.

OPTIONS:
    --help              Show this help message
    --dry-run           Show what would be done without making changes
    --force             Force reinstallation even if ROCm is already installed
    --show-env          Show ROCm environment variables for manual setup

EXAMPLES:
    $0                          # Install with default settings
    $0 --dry-run               # Preview installation
    $0 --force                 # Force reinstall
    $0 --show-env              # Show environment variables

For more information, visit: https://rocm.docs.amd.com/
EOF
}

# Function to create ROCm virtual environment
create_rocm_venv() {
    local venv_path="${1:-./rocm_venv}"

    if command_exists uv; then
        print_step "Creating ROCm virtual environment with uv..."
        execute_command "uv venv \"$venv_path\"" "Creating virtual environment at $venv_path"
        if [ $? -eq 0 ] && [ "$DRY_RUN" != true ]; then
            print_success "Created virtual environment: $venv_path"
            echo "To activate: source \"$venv_path/bin/activate\""
            return 0
        fi
    else
        print_step "uv not found, falling back to python3 venv..."
        execute_command "python3 -m venv \"$venv_path\"" "Creating virtual environment at $venv_path"
        if [ $? -eq 0 ] && [ "$DRY_RUN" != true ]; then
            print_success "Created virtual environment: $venv_path"
            echo "To activate: source \"$venv_path/bin/activate\""
            return 0
        fi
    fi

    return 1
}

# Function to install ROCm-related Python packages
install_rocm_python_packages() {
    local venv_path="$1"

    if [ -n "$venv_path" ] && [ -d "$venv_path" ]; then
        print_step "Installing ROCm-related Python packages in virtual environment..."

        # Activate venv and install packages
        if [ "$DRY_RUN" != true ]; then
            source "$venv_path/bin/activate"
        fi

        # Install common ROCm-related packages
        execute_command "python3 -m pip install --upgrade pip" "Upgrading pip"
        execute_command "python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4" "Installing PyTorch with ROCm support"
        execute_command "python3 -m pip install rocm-docs-core" "Installing ROCm documentation tools"

        if [ "$DRY_RUN" != true ]; then
            deactivate
        fi

        print_success "ROCm Python packages installed in virtual environment"
    else
        print_warning "Virtual environment not found, installing globally..."
        execute_command "python3 -m pip install --upgrade pip" "Upgrading pip"
        execute_command "python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4" "Installing PyTorch with ROCm support"
    fi
}

# Function to detect OS information robustly
detect_os_info() {
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        OS_ID=$ID
        OS_VERSION_ID=$VERSION_ID
        OS_CODENAME=${VERSION_CODENAME:-}
    elif [ -f /etc/lsb-release ]; then
        # shellcheck source=/dev/null
        . /etc/lsb-release
        OS_ID=$DISTRIB_ID
        OS_VERSION_ID=$DISTRIB_RELEASE
        OS_CODENAME=$DISTRIB_CODENAME
    else
        OS_ID=$(uname -s)
        OS_VERSION_ID=$(uname -r)
        OS_CODENAME=""
    fi
    
    # Normalize ID to lowercase
    OS_ID=$(echo "$OS_ID" | tr '[:upper:]' '[:lower:]')
}

# Main installation function
install_rocm() {
    print_header "ROCm Installation"
    
    # Detect OS information
    detect_os_info
    print_step "Detected OS: $OS_ID $OS_VERSION_ID ($OS_CODENAME)"

    # Detect package manager
    package_manager=$(detect_package_manager)
    print_step "Package manager: $package_manager"
    if command_exists rocminfo; then
        rocm_version=$(get_rocm_version)

        if [ -n "$rocm_version" ]; then
            print_warning "ROCm is already installed (version: $rocm_version)"

            # Check for --force flag
            if [[ "$*" == *"--force"* ]]; then
                print_warning "Force reinstall requested - proceeding with reinstallation"
            else
                echo
                echo -e "${CYAN}${BOLD}ROCm is already installed. What would you like to do?${RESET}"
                echo "1) Skip installation"
                echo "2) Reinstall ROCm"
                echo "3) Install additional components only"
                echo
                read -p "Choose option (1-3) [1]: " REINSTALL_CHOICE
                REINSTALL_CHOICE=${REINSTALL_CHOICE:-1}

                case $REINSTALL_CHOICE in
                    1)
                        print_step "Skipping ROCm installation"
                        return 0
                        ;;
                    2)
                        print_warning "Proceeding with ROCm reinstallation"
                        # Perform a clean purge before reinstallation to avoid dependency hell
                        purge_rocm
                        ;;
                    3)
                        print_step "Installing additional components only"
                        INSTALL_ADDITIONAL=true
                        ;;
                    *)
                        print_step "Skipping ROCm installation"
                        return 0
                        ;;
                esac
            fi
        fi
    fi

    # Check if ROCm is installed but rocminfo is missing
    if ! command_exists rocminfo; then
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_warning "ROCm is installed but rocminfo is missing"
            print_step "Installing rocminfo..."

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
                    print_step "Please install rocminfo manually"
                    return 1
                    ;;
            esac

            if command_exists rocminfo; then
                print_success "Installed rocminfo component"
                # Continue to full installation to ensure all other components are present
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        fi
    fi

    # Detect system and package manager
    print_section "Detecting System"

    # Detect environment
    is_wsl=$(detect_wsl)
    is_container=$(detect_container)

    if [ "$is_wsl" = "true" ]; then
        print_warning "Detected Windows Subsystem for Linux (WSL)"
        print_step "ROCm support in WSL may have limitations"
    fi

    if [ "$is_container" = "true" ]; then
        print_warning "Detected container environment"
        print_step "Ensure GPU passthrough is properly configured"
    fi

    # Map OS to repository details
    if [ "$package_manager" = "apt" ]; then
        repo="ubuntu"
        if [ "$OS_ID" = "ubuntu" ]; then
            ubuntu_codename=$OS_CODENAME
        elif [ "$OS_ID" = "debian" ]; then
            # Map Debian versions to Ubuntu codenames for ROCm repository compatibility
            case "$OS_VERSION_ID" in
                "13"*)
                    ubuntu_codename="noble"
                    print_step "Mapping Debian Trixie (13) to Ubuntu Noble (24.04)"
                    setup_gpg_fix
                    ;;
                "12"*)
                    ubuntu_codename="jammy"
                    print_step "Mapping Debian Bookworm (12) to Ubuntu Jammy (22.04)"
                    ;;
                "11"*)
                    ubuntu_codename="focal"
                    print_step "Mapping Debian Bullseye (11) to Ubuntu Focal (20.04)"
                    ;;
                *)
                    print_warning "Unsupported Debian version: $OS_VERSION_ID. Defaulting to Ubuntu Noble."
                    ubuntu_codename="noble"
                    ;;
            esac
        else
            print_warning "Unknown distributor '$OS_ID', using Ubuntu Noble defaults"
            ubuntu_codename="noble"
        fi
    elif [ "$package_manager" = "dnf" ] || [ "$package_manager" = "yum" ]; then
        repo="rhel"
        # Determine RHEL major version for repo path
        case "$OS_VERSION_ID" in
            10*) rhel_version="10" ;;
            9*)  rhel_version="9" ;;
            8*)  rhel_version="8" ;;
            *)   rhel_version="9" ;;
        esac
        print_step "Using RHEL $rhel_version compatibility"
    elif [ "$package_manager" = "zypper" ]; then
        repo="sle"
        sles_version="15.7" # Default for latest
        print_step "Using SLES $sles_version compatibility"
    else
        print_step "Using $package_manager for package management"
        repo=""
    fi

    # Detect existing ROCm installation path
    existing_rocm_path=$(detect_rocm_path)
    if [ "$existing_rocm_path" != "/opt/rocm" ]; then
        print_step "Detected existing ROCm installation at: $existing_rocm_path"
    fi
    
    # Choose installation method
    if [ "$INSTALL_ADDITIONAL" = true ]; then
        print_section "Additional Components Installation"
        INSTALL_TYPE="additional"
        print_step "Installing additional components only"
    else
        print_section "Installation Options"
        echo
        echo -e "${CYAN}${BOLD}ROCm Installation Methods:${RESET}"
        echo "1) Standard installation (ROCm runtime + basic tools)"
        echo "2) Minimal installation (ROCm runtime only)"
        echo "3) Full installation (ROCm + development tools + libraries)"
        echo "4) Custom installation (select specific components)"
        echo
        read -p "Choose installation method (1-4) [1]: " INSTALL_METHOD
        INSTALL_METHOD=${INSTALL_METHOD:-1}

        case $INSTALL_METHOD in
            1)
                INSTALL_TYPE="standard"
                print_step "Using standard installation method"
                ;;
            2)
                INSTALL_TYPE="minimal"
                print_step "Using minimal installation method"
                ;;
            3)
                INSTALL_TYPE="full"
                print_step "Using full installation method"
                ;;
            4)
                INSTALL_TYPE="custom"
                print_step "Using custom installation method"
                ;;
            *)
                INSTALL_TYPE="standard"
                print_step "Using default standard installation method"
                ;;
        esac
    fi

    echo
    echo -e "${CYAN}${BOLD}Virtual Environment Options:${RESET}"
    echo "1) No virtual environment (install Python packages globally)"
    echo "2) Create ROCm virtual environment (recommended for development)"
    echo "3) Use existing virtual environment (specify path)"
    echo
    read -p "Choose virtual environment option (1-3) [2]: " VENV_METHOD
    VENV_METHOD=${VENV_METHOD:-2}

    case $VENV_METHOD in
        1)
            CREATE_VENV=false
            print_step "Python packages will be installed globally"
            ;;
        2)
            CREATE_VENV=true
            VENV_PATH="./rocm_venv"
            print_step "Will create virtual environment at: $VENV_PATH"
            ;;
        3)
            CREATE_VENV=false
            read -p "Enter path to existing virtual environment: " VENV_PATH
            if [ ! -d "$VENV_PATH" ]; then
                print_warning "Virtual environment path does not exist: $VENV_PATH"
                print_step "Will install Python packages globally instead"
            fi
            ;;
        *)
            CREATE_VENV=true
            VENV_PATH="./rocm_venv"
            print_step "Using default virtual environment option"
            ;;
    esac

    # Download and install amdgpu-install package (skip for additional components)
    if [ "$INSTALL_ADDITIONAL" != true ]; then
        print_section "Installing ROCm"

        # Choose ROCm version (only if not additional components)
        # Check for non-interactive mode (pre-seeded choice)
        if [ -n "$INSTALL_ROCM_PRESEEDED_CHOICE" ]; then
            # Non-interactive mode: use pre-seeded choice
            ROCM_CHOICE="$INSTALL_ROCM_PRESEEDED_CHOICE"

            # Validate the choice is in valid range
            if [[ ! "$ROCM_CHOICE" =~ ^[1-3]$ ]]; then
                print_error "Invalid INSTALL_ROCM_PRESEEDED_CHOICE value: $ROCM_CHOICE"
                echo
                echo -e "${YELLOW}Valid values are: 1 (Legacy 6.4.3), 2 (Stable 7.1), 3 (Latest 7.2)${RESET}"
                echo
                echo -e "${CYAN}For ROCm 7.10.0 Preview (TheRock distribution):${RESET}"
                echo "  https://rocm.docs.amd.com/en/7.10.0-preview/install/rocm.html"
                return 1
            fi

            # Set default if choice is empty
            ROCM_CHOICE=${ROCM_CHOICE:-3}
        else
            # Interactive mode: prompt user
            echo
            echo -e "${CYAN}${BOLD}Choose ROCm Version:${RESET}"

            echo "1) ROCm 6.4.3 (Legacy - Stable)"
            echo "2) ROCm 7.1 (Stable)"
            echo "3) ROCm 7.2 (Latest - Recommended)"
            echo
            echo -e "${YELLOW}NOTE: ROCm 7.10.0 Preview is not available through this installer.${RESET}"
            echo -e "${YELLOW}      ROCm 7.10.0 uses 'TheRock' distribution (pip/tarball only).${RESET}"
            echo -e "${YELLOW}      See: https://rocm.docs.amd.com/en/7.10.0-preview/install/rocm.html${RESET}"
            echo
            read -p "Choose ROCm version (1-3) [3]: " ROCM_CHOICE
            ROCM_CHOICE=${ROCM_CHOICE:-3}
        fi

        case $ROCM_CHOICE in
            1)
                ROCM_VERSION="6.4.3"
                ROCM_INSTALL_VERSION="6.4.60403-1"
                repo="ubuntu"
                ubuntu_codename="noble"
                ROCM_CHANNEL="legacy"
                print_step "Using ROCm 6.4.3 (legacy)"
                ;;
            2)
                ROCM_VERSION="7.1"
                ROCM_INSTALL_VERSION="7.1.70100-1"
                repo="ubuntu"
                ubuntu_codename="noble"
                ROCM_CHANNEL="stable"
                print_step "Using ROCm 7.1 (stable)"
                ;;
            3)
                ROCM_VERSION="7.2"
                ROCM_INSTALL_VERSION="7.2.70200-1"
                repo="ubuntu"
                ubuntu_codename="noble"
                ROCM_CHANNEL="latest"
                print_step "Using ROCm 7.2 (latest - recommended)"
                ;;
            *)
                print_error "Invalid choice: $ROCM_CHOICE"
                echo
                echo -e "${YELLOW}Valid choices are 1-3${RESET}"
                echo
                echo -e "${CYAN}For ROCm 7.10.0 Preview (TheRock distribution):${RESET}"
                echo "  https://rocm.docs.amd.com/en/7.10.0-preview/install/rocm.html"
                return 1
                ;;
        esac

        # Log mode information for debugging
        if [ -n "$INSTALL_ROCM_PRESEEDED_CHOICE" ]; then
            print_step "Non-interactive mode: Using pre-seeded choice $ROCM_CHOICE ($ROCM_CHANNEL channel)"
        fi

        # For ROCm 7.x, offer to install updated framework components
        if [ "$ROCM_CHOICE" = "2" ] || [ "$ROCM_CHOICE" = "3" ]; then
            echo
            echo -e "${CYAN}${BOLD}ROCm 7.x Additional Components:${RESET}"
            echo -e "${YELLOW}ROCm 7.x includes many updated frameworks. Would you like to install them?${RESET}"
            echo "1) Yes - Install updated frameworks (PyTorch 2.7, JAX 0.6.0, ONNX Runtime 1.22.0, etc.)"
            echo "2) No - Install ROCm core only"
            echo
            read -p "Install additional frameworks? (1-2) [1]: " INSTALL_FRAMEWORKS
            INSTALL_FRAMEWORKS=${INSTALL_FRAMEWORKS:-1}

            if [ "$INSTALL_FRAMEWORKS" = "1" ]; then
                INSTALL_ROCM7_FRAMEWORKS=true
                print_step "Will install ROCm 7.x with updated frameworks"
            else
                INSTALL_ROCM7_FRAMEWORKS=false
                print_step "Will install ROCm 7.x core only"
            fi
        fi

        # Determine the appropriate installer package and URL based on OS and ROCm version
        # ROCm 7.2.0 versions:
        # Ubuntu: amdgpu-install_7.2.70200-1_all.deb
        # RHEL 9: amdgpu-install-7.2.70200-1.el9.noarch.rpm
        # SLES 15: amdgpu-install-7.2.70200-1.noarch.rpm
        
        # Determine directory path for ROCm (7.1, 7.2, etc.)
        if [ "$ROCM_VERSION" = "7.1" ]; then
            ROCM_DIR_PATH="7.1"
        elif [ "$ROCM_VERSION" = "7.2" ]; then
            ROCM_DIR_PATH="7.2"
        else
            ROCM_DIR_PATH="$ROCM_VERSION"
        fi

        # Mapping versions to their installer IDs (e.g., 7.2.70200-1)
        case $ROCM_VERSION in
            "6.4.3") ROCM_PKG_VER="6.4.60403-1" ;;
            "7.1")   ROCM_PKG_VER="7.1.70100-1" ;;
            "7.2")   ROCM_PKG_VER="7.2.70200-1" ;;
            *)       ROCM_PKG_VER="7.2.70200-1" ;;
        esac

        if [ "$package_manager" = "apt" ]; then
            installer_file="amdgpu-install_${ROCM_PKG_VER}_all.deb"
            installer_url="https://repo.radeon.com/amdgpu-install/$ROCM_DIR_PATH/ubuntu/$ubuntu_codename/$installer_file"
        elif [ "$package_manager" = "dnf" ] || [ "$package_manager" = "yum" ]; then
            # RHEL naming varies slightly between versions
            if [ "$rhel_version" = "10" ]; then
                installer_file="amdgpu-install-$ROCM_PKG_VER.el10.noarch.rpm"
                installer_url="https://repo.radeon.com/amdgpu-install/$ROCM_DIR_PATH/rhel/10/$installer_file"
            else
                installer_file="amdgpu-install-$ROCM_PKG_VER.el$rhel_version.noarch.rpm"
                installer_url="https://repo.radeon.com/amdgpu-install/$ROCM_DIR_PATH/rhel/$OS_VERSION_ID/$installer_file"
            fi
        elif [ "$package_manager" = "zypper" ]; then
            installer_file="amdgpu-install-$ROCM_PKG_VER.noarch.rpm"
            installer_url="https://repo.radeon.com/amdgpu-install/$ROCM_DIR_PATH/sle/$sles_version/$installer_file"
        fi

        if [ "$DRY_RUN" != true ]; then
            print_step "Downloading installer from: $installer_url"
            if ! retry_command "wget -q \"$installer_url\" -O \"$installer_file\"" 3 2; then
                print_error "Failed to download installer package after retries"
                return 1
            fi
            print_success "Downloaded installer package: $installer_file"
        fi

        # Install the package using the appropriate package manager
        if [ "$package_manager" = "apt" ]; then
            # Fix: Pre-create the file that causes postinst script failures
            print_step "Applying permission fix for amdgpu-install..."
            execute_command "sudo chattr -i /etc/apt/sources.list.d/amdgpu-proprietary.list 2>/dev/null || true" "Removing immutable attribute"
            execute_command "sudo touch /etc/apt/sources.list.d/amdgpu-proprietary.list" "Creating amdgpu-proprietary.list file"
            execute_command "sudo chmod 644 /etc/apt/sources.list.d/amdgpu-proprietary.list" "Setting file permissions"
            
            execute_command "sudo apt install -y ./$installer_file" "Installing amdgpu-install package..."
            
            # Handle postinst script failures gracefully
            if [ $? -ne 0 ] && [ "$DRY_RUN" != true ]; then
                print_warning "amdgpu-install installation had post-install script issues, attempting recovery..."
                execute_command "sudo dpkg --configure -a" "Reconfiguring packages"
            fi
        elif [ "$package_manager" = "dnf" ] || [ "$package_manager" = "yum" ]; then
            execute_command "sudo $package_manager install -y ./$installer_file" "Installing amdgpu-install package..."
            execute_command "sudo $package_manager clean all" "Cleaning package cache"
        elif [ "$package_manager" = "zypper" ]; then
            execute_command "sudo zypper --no-gpg-checks install -y ./$installer_file" "Installing amdgpu-install package..."
        fi

        # Setup ROCm repositories manually for better control (Debian/Ubuntu only)
        if [ "$package_manager" = "apt" ]; then
            # Clean up existing repository files (including the broken proprietary list)
            execute_command "sudo rm -f /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu-proprietary.list" "Cleaning up existing repository files"

            # Ensure GPG key is properly installed
            execute_command "sudo mkdir --parents --mode=0755 /etc/apt/keyrings" "Creating keyrings directory"
            execute_command "wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null" "Installing ROCm GPG key"
            execute_command "sudo chmod 644 /etc/apt/keyrings/rocm.gpg" "Setting GPG key permissions"

            # Map ROCm version to appropriate amdgpu driver repository version
            # These are verified paths on repo.radeon.com
            case $ROCM_VERSION in
                "7.2")   amdgpu_apt_ver="30.30" ;;
                "7.1")   amdgpu_apt_ver="30.20" ;;
                "6.4.3") amdgpu_apt_ver="6.4.3" ;;
                *)       amdgpu_apt_ver="latest" ;;
            esac
            
            execute_command "sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $ubuntu_codename main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$amdgpu_apt_ver/ubuntu $ubuntu_codename main
EOF" "Adding ROCm $ROCM_VERSION repositories with Ubuntu $ubuntu_codename compatibility"

            # Set proper repository priorities
            execute_command "sudo tee /etc/apt/preferences.d/rocm-pin-600 << 'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF" "Setting ROCm repository priorities"
        fi

        # Update package lists
        print_step "Updating package lists..."
        if [ "$package_manager" = "apt" ]; then
            sudo apt clean
            sudo apt update
        elif [ "$package_manager" = "zypper" ]; then
            sudo zypper --gpg-auto-import-keys refresh
        fi

        # Install prerequisites
        print_step "Installing prerequisites..."
        if [ "$package_manager" = "apt" ]; then
            sudo apt install -y python3-setuptools python3-wheel
        elif [ "$package_manager" = "dnf" ] || [ "$package_manager" = "yum" ]; then
            # RHEL prerequisites
            wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-$rhel_version.noarch.rpm
            sudo rpm -ivh epel-release-latest-$rhel_version.noarch.rpm || true
            if [ "$package_manager" = "dnf" ] && [ "$rhel_version" != "10" ]; then
                sudo dnf config-manager --enable codeready-builder-for-rhel-$rhel_version-x86_64-rpms || true
            fi
            sudo $package_manager install -y python3-setuptools python3-wheel
        elif [ "$package_manager" = "zypper" ]; then
            sudo zypper install -y python3-setuptools python3-wheel
        fi

        # Add user to render and video groups
        print_step "Adding user to render and video groups..."
        sudo usermod -a -G render,video $LOGNAME
    fi
    
    # For Debian, install any missing dependencies from Ubuntu Noble
    if [ "$distributor" = "Debian" ]; then
        print_step "Installing any missing dependencies from Ubuntu Noble..."

        # Temporarily add Ubuntu noble for missing dependencies
        execute_command "sudo tee /etc/apt/sources.list.d/ubuntu-deps.list << EOF
deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse
deb [arch=amd64] http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse
EOF" "Adding Ubuntu noble repositories for missing dependencies"

        execute_command "sudo apt update" "Updating package lists with Ubuntu repos"

        # Try to install the missing libstdc++ dependencies
        execute_command "sudo apt install -y libstdc++-11-dev libgcc-11-dev" "Installing missing GCC libraries from Ubuntu"

        # Remove the temporary Ubuntu repos
        execute_command "sudo rm -f /etc/apt/sources.list.d/ubuntu-deps.list" "Removing temporary Ubuntu repositories"
        execute_command "sudo apt update" "Updating package lists after removing Ubuntu repos"
    fi

    # Install ROCm based on selected method
    print_step "Installing ROCm ($INSTALL_TYPE)..."

    case $INSTALL_TYPE in
        "minimal"|"standard"|"full"|"custom")
            if [ "$package_manager" = "apt" ]; then
                # On Debian/Ubuntu, we favor direct apt installation of metapackages
                # as it's more reliable than the amdgpu-install wrapper on non-standard distros.
                print_step "Using direct apt installation for better reliability..."
                
                # Build list of packages to install based on INSTALL_TYPE
                local pkgs_to_install="rocm-core"
                
                case $INSTALL_TYPE in
                    "minimal")
                        pkgs_to_install="$pkgs_to_install rocm-hip-runtime"
                        ;;
                    "standard")
                        pkgs_to_install="$pkgs_to_install rocm"
                        ;;
                    "full")
                        pkgs_to_install="$pkgs_to_install rocm rocm-dev"
                        ;;
                    "custom")
                        # Add specific components based on user selection logic (simplified here to full for custom)
                        pkgs_to_install="$pkgs_to_install rocm rocm-dev"
                        ;;
                esac
                
                # Add common required tools
                pkgs_to_install="$pkgs_to_install rocminfo rocm-smi-lib"
                
                execute_command "sudo apt-get install -y $pkgs_to_install" "Installing ROCm metapackages"
                
                if [ $? -ne 0 ]; then
                    print_warning "Direct apt installation failed, falling back to amdgpu-install wrapper..."
                    # Fallback to amdgpu-install if direct apt fails
                    case $INSTALL_TYPE in
                        "minimal")  execute_command "sudo amdgpu-install --usecase=lrt --accept-eula --no-32 -y" "Fallback: Minimal ROCm runtime" ;;
                        "standard") execute_command "sudo amdgpu-install --usecase=rocm --accept-eula --no-32 -y" "Fallback: Standard ROCm runtime" ;;
                        "full")     execute_command "sudo amdgpu-install --usecase=rocm,rocmdev --accept-eula --no-32 -y" "Fallback: Full ROCm installation" ;;
                    esac
                fi
            else
                # For non-apt managers, use the standard case
                case $INSTALL_TYPE in
                    "minimal")
                        # Install minimal ROCm runtime
                        execute_command "sudo amdgpu-install --usecase=lrt --accept-eula --no-32 -y" "Installing minimal ROCm runtime"
                        ;;
                    "standard")
                        # Install standard ROCm runtime + basic tools
                        execute_command "sudo amdgpu-install --usecase=rocm --accept-eula --no-32 -y" "Installing standard ROCm runtime"
                        ;;
                    "full")
                        # Install full ROCm + development tools
                        execute_command "sudo amdgpu-install --usecase=rocm,rocmdev --accept-eula --no-32 -y" "Installing full ROCm with development tools"
                        ;;
                    "custom")
                        # Custom installation - let user choose components
                        echo
                        echo -e "${CYAN}${BOLD}Available ROCm Use Cases:${RESET}"
                        echo "1) lrt (ROCm runtime - minimal)"
                        echo "2) rocm (full ROCm stack)"
                        echo "3) rocmdev (ROCm + development tools)"
                        echo "4) rocmdevtools (profiling/debugging tools)"
                        echo "5) hip (HIP runtime only)"
                        echo "6) opencl (OpenCL support)"
                        echo "7) mllib (machine learning libraries)"
                        echo
                        read -p "Enter component numbers to install (space-separated) [2]: " COMPONENTS
                        COMPONENTS=${COMPONENTS:-2}

                        # Build usecase string from selected components
                        USECASE=""
                        for comp in $COMPONENTS; do
                            case $comp in
                                1) USECASE="${USECASE}lrt," ;;
                                2) USECASE="${USECASE}rocm," ;;
                                3) USECASE="${USECASE}rocmdev," ;;
                                4) USECASE="${USECASE}rocmdevtools," ;;
                                5) USECASE="${USECASE}hip," ;;
                                6) USECASE="${USECASE}opencl," ;;
                                7) USECASE="${USECASE}mllib," ;;
                            esac
                        done
                        # Remove trailing comma
                        USECASE=${USECASE%,}

                        if [ -n "$USECASE" ]; then
                            execute_command "sudo amdgpu-install --usecase=$USECASE --accept-eula --no-32 -y" "Installing custom ROCm components"
                        fi
                        ;;
                esac
            fi
            ;;
    "additional")
        # Install additional components only
        echo
        echo -e "${CYAN}${BOLD}Available Additional Components:${RESET}"
        echo "1) rocmdev (ROCm development tools)"
        echo "2) rocmdevtools (profiling/debugging tools)"
        echo "3) opencl (OpenCL support)"
        echo "4) hip (HIP runtime)"
        echo "5) mllib (machine learning libraries)"
        echo "6) rccl (ROCm Communication Collectives Library)"
        echo
        read -p "Enter component numbers to install (space-separated) [1]: " COMPONENTS
        COMPONENTS=${COMPONENTS:-1}

        # Build usecase string from selected components
        USECASE=""
        for comp in $COMPONENTS; do
            case $comp in
                1) USECASE="${USECASE}rocmdev," ;;
                2) USECASE="${USECASE}rocmdevtools," ;;
                3) USECASE="${USECASE}opencl," ;;
                4) USECASE="${USECASE}hip," ;;
                5) USECASE="${USECASE}mllib," ;;
                6)
                    USECASE="${USECASE}rccl," ;;
            esac
        done
        # Remove trailing comma
        USECASE=${USECASE%,}

        if [ -n "$USECASE" ]; then
            execute_command "sudo amdgpu-install --usecase=$USECASE --accept-eula --no-32 -y" "Installing additional ROCm components"
        else
            # Fallback for RCCL if usecase not available
            if [ "$package_manager" = "apt" ]; then
                execute_command "sudo apt update && sudo apt install -y librccl-dev librccl1" "Installing RCCL libraries directly"
            elif [ "$package_manager" = "dnf" ] || [ "$package_manager" = "yum" ]; then
                execute_command "sudo $package_manager install -y rccl-devel" "Installing RCCL libraries directly"
            elif [ "$package_manager" = "zypper" ]; then
                execute_command "sudo zypper install -y rccl-devel" "Installing RCCL libraries directly"
            fi
        fi
        ;;
esac

    if [ $? -ne 0 ]; then
        print_error "Failed to install ROCm"

        print_step "Installation failed. Check the error messages above for details."
        print_step "You can try: sudo apt install -f  # to fix broken dependencies"
        print_step "Or check: sudo apt update && sudo apt upgrade  # to update packages"

        return 1
    fi

    print_success "Installed ROCm ($INSTALL_TYPE)"

    # Install ROCm 7.x specific frameworks if selected and ROCm 7.x was installed
    if [ "$INSTALL_ROCM7_FRAMEWORKS" = true ] && ([[ "$ROCM_VERSION" == "7.1" ]] || [[ "$ROCM_VERSION" == "7.2" ]]); then
        print_section "Installing ROCm $ROCM_VERSION Updated Frameworks"
        
        # Use manylinux repo for the specific ROCm version
        repo_ver=$ROCM_VERSION
        [[ "$ROCM_VERSION" == "7.2" ]] && repo_ver="7.2" # Assuming 7.2 repo exists or follows pattern
        # Fallback to 7.1 or latest if 7.2 not yet in manylinux
        
        print_step "Installing PyTorch for ROCm $ROCM_VERSION..."
        # Explicitly use the PyTorch ROCm index URL to prevent NVIDIA variant installation
        local pytorch_index="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
        [[ "$ROCM_VERSION" == "7.2" ]] && pytorch_index="https://download.pytorch.org/whl/nightly/rocm7.2"
        [[ "$ROCM_VERSION" == "7.1" ]] && pytorch_index="https://download.pytorch.org/whl/nightly/rocm7.1"
        
        if [ "$CREATE_VENV" = true ]; then
            source "$VENV_PATH/bin/activate"
            python3 -m pip install --index-url "$pytorch_index" torch torchvision torchaudio
            deactivate
        else
            python3 -m pip install --index-url "$pytorch_index" torch torchvision torchaudio
        fi

        # Install other ROCm frameworks from manylinux repo
        print_step "Installing additional ROCm frameworks..."

        # Install JAX, ONNX Runtime, TensorFlow
        for pkg in "jax jaxlib" "onnxruntime" "tensorflow" "pytorch-triton-rocm"; do
            if [ "$CREATE_VENV" = true ]; then
                source "$VENV_PATH/bin/activate"
                python3 -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-$repo_ver/ $pkg || \
                python3 -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/ $pkg
                deactivate
            else
                python3 -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-$repo_ver/ $pkg || \
                python3 -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/ $pkg
            fi
        done

        print_success "Installed ROCm $ROCM_VERSION updated frameworks"
    fi

    # Set up ROCm environment variables
    print_section "Setting up ROCm Environment"

    # Detect GPU architecture for optimal configuration
    gpu_arch=$(detect_gpu_architecture)

    # Set ROCm environment variables
    print_step "Configuring ROCm environment variables..."

    # HSA_OVERRIDE_GFX_VERSION - Use detected architecture or default
    if [ -n "$gpu_arch" ]; then
        HSA_OVERRIDE_GFX_VERSION="${gpu_arch#gfx}"  # Remove 'gfx' prefix
    else
        HSA_OVERRIDE_GFX_VERSION="11.0.0"  # Default for RDNA3
    fi

    PYTORCH_ROCM_ARCH="$gpu_arch"
    ROCM_PATH="/opt/rocm"
    PATH="/opt/rocm/bin:$PATH"
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Set HSA_TOOLS_LIB if rocprofiler library exists
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
        print_step "ROCm profiler library found and configured"
    else
        HSA_TOOLS_LIB=0
        print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
    fi

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion if present
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
        unset PYTORCH_CUDA_ALLOC_CONF
        print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
    fi

    print_success "ROCm environment variables configured"

    # Create virtual environment if requested
    if [ "$CREATE_VENV" = true ]; then
        print_section "Creating Virtual Environment"
        create_rocm_venv "$VENV_PATH"
        if [ $? -eq 0 ]; then
            print_section "Installing Python Packages"
            install_rocm_python_packages "$VENV_PATH"
        fi
    elif [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        print_section "Installing Python Packages in Existing Virtual Environment"
        install_rocm_python_packages "$VENV_PATH"
    fi

    # Verify installation
    print_section "Verifying Installation"
    verify_rocm_installation "$ROCM_VERSION"

    # Final cleanup and fix for any remaining dependency issues
    if [ "$package_manager" = "apt" ]; then
        print_step "Performing final dependency resolution..."
        sudo apt install -f -y
    fi

    if command_exists rocminfo; then
        rocm_version=$(get_rocm_version)

        if [ -n "$rocm_version" ]; then
            print_success "ROCm is installed (version: $rocm_version)"
        else
            print_warning "ROCm is installed but version could not be determined"
        fi

        # Detect GPU architecture
        gpu_arch=$(detect_gpu_architecture)

        # Check if ROCm can detect GPUs
        print_step "Checking GPU detection..."
        gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)

        if [ "$gpu_count" -gt 0 ]; then
            print_success "Detected $gpu_count AMD GPU(s)"

            # List GPUs with architecture info
            rocminfo 2>/dev/null | grep -A 5 "Device Type:.*GPU" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
                echo -e "  - $gpu (Architecture: $gpu_arch)"
            done
        else
            print_warning "No AMD GPUs detected by ROCm"
        fi
    else
        print_error "ROCm installation verification failed"
        return 1
    fi
    
    # Clean up
    print_step "Cleaning up..."
    rm -f amdgpu-install_${ROCM_INSTALL_VERSION}_all.deb

    # Show completion message with environment variables
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██████╗  ██████╗  ██████╗███╗   ███╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     ║
    ║  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     ║
    ║  ██████╔╝██║   ██║██║     ██╔████╔██║    ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     ║
    ║  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     ║
    ║  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗║
    ║  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  ROCm is now ready to use with your AMD GPU.            ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "ROCm installation completed successfully"
    print_warning "You may need to log out and log back in for group changes to take effect"
    echo
    echo -e "${CYAN}${BOLD}ROCm Environment Variables:${RESET}"
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
    echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}These environment variables are set for this session.${RESET}"
    echo -e "${YELLOW}For future sessions, you may need to run:${RESET}"
    echo -e "${GREEN}eval \"\$(./scripts/install_rocm.sh --show-env)\"${RESET}"
    echo
    echo -e "${CYAN}${BOLD}To apply these settings to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$(./scripts/install_rocm.sh --show-env)\"${RESET}"
    echo
    echo -e "${CYAN}${BOLD}To display environment variables with banner, run:${RESET}"
    echo -e "${GREEN}./scripts/install_rocm.sh --show-env-with-banner${RESET}"

    # Show virtual environment information if created
    if [ "$CREATE_VENV" = true ] && [ -d "$VENV_PATH" ]; then
        echo
        echo -e "${CYAN}${BOLD}Virtual Environment Created:${RESET}"
        echo -e "${GREEN}source \"$VENV_PATH/bin/activate\"${RESET}"
        echo -e "${YELLOW}Python packages have been installed in the virtual environment${RESET}"
    elif [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        echo
        echo -e "${CYAN}${BOLD}Virtual Environment:${RESET}"
        echo -e "${GREEN}source \"$VENV_PATH/bin/activate\"${RESET}"
        echo -e "${YELLOW}Python packages have been installed in the existing virtual environment${RESET}"
    fi

    # Show final component validation summary
    echo
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${BLUE}${BOLD}│ ML STACK COMPONENT VALIDATION SUMMARY                   │${RESET}"
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}"
    
    python3 -c "
import importlib, os
pkgs=['torch', 'flash_attn', 'vllm', 'triton', 'deepspeed', 'onnxruntime', 'bitsandbytes', 'migraphx']
print('='*55)
for p in pkgs:
    try:
        mod = importlib.import_module(p)
        ver = getattr(mod, '__version__', 'Installed')
        print(f'✓ {p:<15} : {ver}')
        if p=='torch':
            print(f'  - ROCm Support  : {getattr(mod.version, \"hip\", \"N/A\")}')
            print(f'  - GPU Available : {mod.cuda.is_available()}')
    except ImportError:
        print(f'✗ {p:<15} : Not Found')
print('-'*55)
if os.path.exists('/opt/rocm/.info/version'):
    print(f'System ROCm Version : {open(\"/opt/rocm/.info/version\").read().strip()}')
print('='*55)
"

    return 0
}

# Global variables for dry-run mode
DRY_RUN=false

# Check for command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_usage
            exit 0
            ;;
        --show-env)
            show_env_clean
            exit 0
            ;;
        --show-env-with-banner)
            show_env
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            print_warning "DRY RUN MODE: No actual changes will be made"
            shift
            ;;
        --force)
            # Already handled in the function
            shift
            ;;
        *)
            # Unknown option
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run the installation function
install_rocm "$@"
