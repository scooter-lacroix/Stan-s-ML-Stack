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
# AMDGPU Drivers Installation Script
# =============================================================================
# This script installs AMD GPU kernel drivers and userspace components.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
   █████╗ ███╗   ███╗██████╗  ██████╗ ██████╗ ██╗   ██╗    ██████╗ ██████╗ ██╗██╗   ██╗███████╗██████╗ ███████╗
  ██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██╔══██╗██║   ██║    ██╔══██╗██╔══██╗██║██║   ██║██╔════╝██╔══██╗██╔════╝
  ███████║██╔████╔██║██║  ██║██║  ███╗██████╔╝██║   ██║    ██║  ██║██████╔╝██║██║   ██║█████╗  ██████╔╝███████╗
  ██╔══██║██║╚██╔╝██║██║  ██║██║   ██║██╔═══╝ ██║   ██║    ██║  ██║██╔══██╗██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║
  ██║  ██║██║ ╚═╝ ██║██████╔╝╚██████╔╝██║     ╚██████╔╝    ██████╔╝██║  ██║██║ ╚████╔╝ ███████╗██║  ██║███████║
  ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚═╝      ╚═════╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝
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

# Function to detect GPU architecture
detect_gpu_architecture() {
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

# Function to detect if running in WSL
detect_wsl() {
    if [ -f /proc/version ] && grep -qi "microsoft\|wsl" /proc/version; then
        return 0
    else
        return 1
    fi
}

# Function to detect if running in container
detect_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q "docker\|container" /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to run sudo commands with password prompt if needed
sudo_with_pass() {
    if command_exists sudo; then
        sudo "$@"
    else
        print_error "sudo command not found"
        return 1
    fi
}

# Function to retry commands with exponential backoff
retry_command() {
    local max_attempts=3
    local attempt=1
    local command="$@"

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt of $max_attempts: $command"

        if eval "$command"; then
            return 0
        fi

        if [ $attempt -lt $max_attempts ]; then
            local delay=$((2 ** (attempt - 1)))
            print_warning "Command failed, retrying in $delay seconds..."
            sleep $delay
        fi

        ((attempt++))
    done

    print_error "Command failed after $max_attempts attempts"
    return 1
}

# Function to check if uv is installed and install if needed
ensure_uv() {
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
            return 1
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi
    return 0
}

# Function to create and manage virtual environment
setup_virtual_environment() {
    local venv_name="${1:-amdgpu_drivers_venv}"
    local venv_path="./$venv_name"

    if [ ! -d "$venv_path" ]; then
        print_step "Creating virtual environment: $venv_path"
        if command_exists uv; then
            uv venv "$venv_path"
        else
            python3 -m venv "$venv_path"
        fi

        if [ ! -d "$venv_path" ]; then
            print_error "Failed to create virtual environment"
            return 1
        fi
    else
        print_step "Virtual environment already exists: $venv_path"
    fi

    # Activate the virtual environment
    source "$venv_path/bin/activate"
    print_success "Activated virtual environment: $venv_path"

    # Store venv path for later use
    AMDGPU_VENV_PYTHON="$venv_path/bin/python"
    export AMDGPU_VENV_PYTHON

    return 0
}

# Function to handle externally managed environments
handle_externally_managed() {
    local command="$1"
    local args="$2"

    # Try the command first
    local output
    output=$(eval "$command $args" 2>&1)
    local exit_code=$?

    if echo "$output" | grep -q "externally managed"; then
        print_warning "Detected externally managed environment"
        print_step "Attempting to use --break-system-packages flag..."

        # Try with --break-system-packages
        output=$(eval "$command --break-system-packages $args" 2>&1)
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            print_success "Installation successful with --break-system-packages"
            return 0
        else
            print_warning "Failed to install globally, falling back to virtual environment"
            return 1
        fi
    elif [ $exit_code -eq 0 ]; then
        print_success "Global installation successful"
        return 0
    else
        print_error "Installation failed: $output"
        return 1
    fi
}

# Function to load configuration from file
load_config() {
    local config_file="${1:-amdgpu_config.sh}"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
        return 0
    else
        print_step "No configuration file found at $config_file, using defaults"
        return 1
    fi
}

# Function to save configuration
save_config() {
    local config_file="${1:-amdgpu_config.sh}"

    print_step "Saving configuration to $config_file"

    cat > "$config_file" << EOF
#!/bin/bash
# AMDGPU Drivers Configuration File
# Generated on $(date)

# Installation settings
export AMDGPU_INSTALL_METHOD="${INSTALL_METHOD:-auto}"
export AMDGPU_DRY_RUN="${DRY_RUN:-false}"
export AMDGPU_FORCE="${FORCE:-false}"

# ROCm settings
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HSA_TOOLS_LIB="${HSA_TOOLS_LIB:-0}"

# Virtual environment settings
export AMDGPU_VENV_PYTHON="${AMDGPU_VENV_PYTHON:-}"

# System information
export AMDGPU_DETECTED_ROCM_VERSION="${rocm_version:-6.4.0}"
export AMDGPU_DETECTED_GPU_ARCH="${gpu_arch:-gfx1100}"
export AMDGPU_PACKAGE_MANAGER="${package_manager:-apt}"
EOF

    print_success "Configuration saved to $config_file"
}

# Function to install package with multiple fallback strategies
install_package_with_fallback() {
    local package="$1"
    local package_manager="${2:-$package_manager}"

    print_step "Installing $package using $package_manager..."

    case $package_manager in
        apt)
            if handle_externally_managed "sudo apt install -y" "$package"; then
                return 0
            fi
            ;;
        dnf)
            if sudo dnf install -y "$package"; then
                return 0
            fi
            ;;
        yum)
            if sudo yum install -y "$package"; then
                return 0
            fi
            ;;
        pacman)
            if sudo pacman -S --noconfirm "$package"; then
                return 0
            fi
            ;;
        zypper)
            if sudo zypper install -y "$package"; then
                return 0
            fi
            ;;
        *)
            print_error "Unsupported package manager: $package_manager"
            return 1
            ;;
    esac

    print_warning "Failed to install $package with $package_manager"
    return 1
}

# Main installation function
install_amdgpu_drivers() {
    print_header "AMDGPU Drivers Installation"

    # Load configuration if available
    load_config

    # Check for command line arguments
    DRY_RUN=${DRY_RUN:-false}
    FORCE=${FORCE:-false}
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                print_warning "DRY RUN MODE: No actual installation will be performed"
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                if [ -n "$CONFIG_FILE" ]; then
                    load_config "$CONFIG_FILE"
                fi
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --dry-run          Show what would be done without making changes"
                echo "  --force            Force reinstallation even if drivers are detected"
                echo "  --config FILE      Load configuration from FILE"
                echo "  --help             Show this help message"
                return 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                return 1
                ;;
        esac
    done

    # Detect environment
    print_section "Detecting Environment"

    # Check if running in WSL
    if detect_wsl; then
        print_warning "Detected Windows Subsystem for Linux (WSL)"
        print_warning "AMD GPU drivers may have limited functionality in WSL"
    fi

    # Check if running in container
    if detect_container; then
        print_warning "Detected container environment"
        print_warning "GPU driver installation may require privileged container or host installation"
    fi

    # Detect package manager
    package_manager=$(detect_package_manager)
    print_step "Detected package manager: $package_manager"

    # Check if AMDGPU drivers are already installed
    if lsmod | grep -q amdgpu && [ "$FORCE" != "true" ]; then
        print_warning "AMDGPU drivers are already loaded"
        echo
        echo -e "${CYAN}${BOLD}Options:${RESET}"
        echo "1) Skip installation"
        echo "2) Force reinstallation"
        echo "3) Verify current installation only"
        echo
        read -p "Choose option (1-3) [1]: " REINSTALL_CHOICE
        REINSTALL_CHOICE=${REINSTALL_CHOICE:-1}

        case $REINSTALL_CHOICE in
            1)
                print_step "Skipping AMDGPU drivers installation"
                return 0
                ;;
            2)
                print_step "Proceeding with forced reinstallation"
                ;;
            3)
                print_step "Verifying current installation only"
                verify_installation
                return $?
                ;;
            *)
                print_error "Invalid choice"
                return 1
                ;;
        esac
    fi

    # Detect system information
    print_section "Detecting System Information"

    if command_exists lsb_release; then
        ubuntu_version=$(lsb_release -rs 2>/dev/null)
        ubuntu_codename=$(lsb_release -cs 2>/dev/null)
        print_step "System: Ubuntu $ubuntu_version ($ubuntu_codename)"
    else
        print_warning "lsb_release not found, cannot detect Ubuntu version"
        ubuntu_version="unknown"
        ubuntu_codename="unknown"
    fi

    # Check ROCm installation
    print_section "Checking ROCm Installation"

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Detect ROCm version
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        fi

        if [ -z "$rocm_version" ]; then
            print_warning "Could not detect ROCm version, using default 6.4.0"
            rocm_version="6.4.0"
        else
            print_success "Detected ROCm version: $rocm_version"
        fi

        # Detect GPU architecture
        gpu_arch=$(detect_gpu_architecture)
        print_step "Detected GPU architecture: $gpu_arch"

    else
        print_step "rocminfo not found, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."

            case $package_manager in
                apt)
                    if [ "$DRY_RUN" != "true" ]; then
                        sudo apt update && sudo apt install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with apt"
                    fi
                    ;;
                dnf)
                    if [ "$DRY_RUN" != "true" ]; then
                        sudo dnf install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with dnf"
                    fi
                    ;;
                yum)
                    if [ "$DRY_RUN" != "true" ]; then
                        sudo yum install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with yum"
                    fi
                    ;;
                pacman)
                    if [ "$DRY_RUN" != "true" ]; then
                        sudo pacman -S rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with pacman"
                    fi
                    ;;
                zypper)
                    if [ "$DRY_RUN" != "true" ]; then
                        sudo zypper install -y rocminfo
                    else
                        print_step "[DRY RUN] Would install rocminfo with zypper"
                    fi
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

    # Installation method selection
    print_section "Installation Method"

    echo
    echo -e "${CYAN}${BOLD}Installation Method Options:${RESET}"
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

    # Ensure uv is available for venv operations
    if [ "$INSTALL_METHOD" = "venv" ] || [ "$INSTALL_METHOD" = "auto" ]; then
        if ! ensure_uv; then
            if [ "$INSTALL_METHOD" = "auto" ]; then
                print_warning "Failed to install uv, falling back to global installation"
                INSTALL_METHOD="global"
            else
                print_error "uv is required for virtual environment installation"
                return 1
            fi
        fi
    fi

    # Set up virtual environment if needed
    if [ "$INSTALL_METHOD" = "venv" ]; then
        if ! setup_virtual_environment "amdgpu_drivers_venv"; then
            print_error "Failed to set up virtual environment"
            return 1
        fi
    fi

    # Install AMDGPU drivers
    print_section "Installing AMDGPU Drivers"

    if [ "$ubuntu_codename" != "unknown" ]; then
        print_step "Downloading amdgpu-install package..."

        if [ "$DRY_RUN" != "true" ]; then
            # Use the latest version (6.4.60400-1)
            if ! retry_command wget -q https://repo.radeon.com/amdgpu-install/6.4/ubuntu/$ubuntu_codename/amdgpu-install_6.4.60400-1_all.deb; then
                print_error "Failed to download amdgpu-install package after retries"
                return 1
            fi

            print_success "Downloaded amdgpu-install package"

            # Install the package
            print_step "Installing amdgpu-install package..."
            if ! retry_command sudo apt install -y ./amdgpu-install_6.4.60400-1_all.deb; then
                print_error "Failed to install amdgpu-install package after retries"
                return 1
            fi

            print_success "Installed amdgpu-install package"
        else
            print_step "[DRY RUN] Would download and install amdgpu-install package"
        fi
    else
        print_error "Cannot determine Ubuntu codename, manual installation required"
        return 1
    fi

    # Update package lists
    print_step "Updating package lists..."
    if [ "$DRY_RUN" != "true" ]; then
        if ! retry_command sudo apt update; then
            print_warning "Failed to update package lists after retries, continuing anyway"
        else
            print_success "Updated package lists"
        fi
    else
        print_step "[DRY RUN] Would update package lists"
    fi

    # Install Linux headers
    print_step "Installing Linux headers..."
    if [ "$DRY_RUN" != "true" ]; then
        if ! retry_command sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"; then
            print_warning "Failed to install Linux headers after retries, continuing anyway"
        else
            print_success "Installed Linux headers"
        fi
    else
        print_step "[DRY RUN] Would install Linux headers"
    fi

    # Install AMDGPU DKMS
    print_step "Installing AMDGPU DKMS..."
    if [ "$DRY_RUN" != "true" ]; then
        if ! retry_command sudo apt install -y amdgpu-dkms; then
            print_error "Failed to install AMDGPU DKMS after retries"
            return 1
        fi

        print_success "Installed AMDGPU DKMS"
    else
        print_step "[DRY RUN] Would install AMDGPU DKMS"
    fi

    # Install additional ROCm components if needed
    if [ "$INSTALL_METHOD" != "global" ] && [ "$DRY_RUN" != "true" ]; then
        print_step "Installing additional ROCm components for enhanced functionality..."

        # Try to install ROCm development packages with fallback
        if ! install_package_with_fallback "rocm-dev" "$package_manager"; then
            print_warning "Failed to install ROCm development packages, continuing anyway"
        fi

        # Install ROCm libraries with fallback
        if ! install_package_with_fallback "rocm-libs" "$package_manager"; then
            print_warning "Failed to install ROCm libraries, continuing anyway"
        fi

        # Install ROCm profiler if available
        if ! install_package_with_fallback "rocprofiler" "$package_manager"; then
            print_warning "Failed to install ROCm profiler, continuing anyway"
        fi
    fi

    # Set up ROCm environment variables
    print_section "Setting up ROCm Environment Variables"

    if [ "$DRY_RUN" != "true" ]; then
        # Set up ROCm environment variables
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH="$gpu_arch"
        export ROCM_PATH="/opt/rocm"
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Try to install rocprofiler
            if command_exists apt-cache && apt-cache show rocprofiler >/dev/null 2>&1; then
                print_step "Installing rocprofiler for HSA tools support..."
                sudo apt-get install -y rocprofiler
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

        # Handle PYTORCH_CUDA_ALLOC_CONF conversion
        if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
            export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
            unset PYTORCH_CUDA_ALLOC_CONF
            print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
        fi

        print_success "ROCm environment variables configured"
    else
        print_step "[DRY RUN] Would set up ROCm environment variables"
    fi

    # Verify installation
    if [ "$DRY_RUN" != "true" ]; then
        verify_installation
        verification_result=$?
    else
        print_step "[DRY RUN] Would verify installation"
        verification_result=0
    fi

    # Clean up
    if [ "$DRY_RUN" != "true" ]; then
        print_step "Cleaning up..."
        rm -f amdgpu-install_6.4.60400-1_all.deb
        print_success "Cleanup completed"
    else
        print_step "[DRY RUN] Would clean up installation files"
    fi

    if [ "$DRY_RUN" != "true" ]; then
        # Save configuration for future use
        save_config

        print_success "AMDGPU drivers installation completed successfully"
        print_warning "You may need to reboot for the drivers to take full effect"

        # Show environment variables for future sessions
        echo
        echo -e "${CYAN}${BOLD}Environment Variables for Future Sessions:${RESET}"
        show_env

        # Show usage instructions
        echo
        echo -e "${CYAN}${BOLD}Usage Instructions:${RESET}"
        echo -e "${GREEN}To apply these settings to your current shell:${RESET}"
        echo -e "${GREEN}source amdgpu_config.sh${RESET}"
        echo
        echo -e "${GREEN}Or run the script with --show-env to see the variables:${RESET}"
        echo -e "${GREEN}./install_amdgpu_drivers.sh --show-env${RESET}"
    else
        print_success "DRY RUN completed - no changes were made"
    fi

    return $verification_result
}

# Verification function
verify_installation() {
    print_section "Verifying Installation"

    local verification_passed=true

    if lsmod | grep -q amdgpu; then
        print_success "AMDGPU drivers are loaded"

        # Check if we can detect GPUs
        print_step "Checking GPU detection..."
        if command_exists lspci; then
            gpu_count=$(lspci | grep -i "amd\|radeon\|advanced micro devices" | grep -i "vga\|3d\|display" | wc -l)

            if [ "$gpu_count" -gt 0 ]; then
                print_success "Detected $gpu_count AMD GPU(s)"

                # List GPUs
                lspci | grep -i "amd\|radeon\|advanced micro devices" | grep -i "vga\|3d\|display" | while read -r gpu; do
                    echo -e "  - $gpu"
                done
            else
                print_warning "No AMD GPUs detected by lspci"
                verification_passed=false
            fi
        else
            print_warning "lspci not found, cannot detect GPUs"
        fi

        # Test ROCm functionality
        if command_exists rocminfo; then
            print_step "Testing ROCm functionality..."
            if rocminfo >/dev/null 2>&1; then
                print_success "ROCm is functioning correctly"
            else
                print_warning "ROCm may not be functioning correctly"
                verification_passed=false
            fi
        fi
    else
        print_error "AMDGPU drivers are not loaded"
        verification_passed=false
    fi

    if [ "$verification_passed" = true ]; then
        print_success "Installation verification passed"
        return 0
    else
        print_warning "Installation verification failed - some components may not be working"
        return 1
    fi
}

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_amdgpu_drivers "$@"

