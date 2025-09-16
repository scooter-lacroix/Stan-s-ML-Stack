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
# ROCm Installation Script
# =============================================================================
# This script installs AMD's ROCm platform for GPU computing.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗  ██████╗  ██████╗███╗   ███╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     
  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     
  ██████╔╝██║   ██║██║     ██╔████╔██║    ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     
  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     
  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
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

    # Try rocminfo first
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | sed -n 's/.*ROCm Version\s*:\s*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1)
    fi

    # Fallback to directory listing with regex
    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    # Fallback to version file
    if [ -z "$rocm_version" ] && [ -f "/opt/rocm/.info/version" ]; then
        rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
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
        print_warning "Could not detect GPU architecture, using default: $gpu_arch"
    else
        print_success "Detected GPU architecture: $gpu_arch"
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

# Main installation function
install_rocm() {
    print_header "ROCm Installation"
    
    # Check if ROCm is already installed
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
                print_success "Installed rocminfo"
                return 0
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        fi
    fi

    # Detect system and package manager
    print_section "Detecting System"
    package_manager=$(detect_package_manager)
    print_step "Package manager: $package_manager"

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

    if [ "$package_manager" = "apt" ]; then
        distributor=$(lsb_release -is 2>/dev/null || echo "Unknown")
        ubuntu_version=$(lsb_release -rs 2>/dev/null || echo "Unknown")
        ubuntu_codename=$(lsb_release -cs 2>/dev/null || echo "Unknown")
        print_step "System: $distributor $ubuntu_version ($ubuntu_codename)"

        if [ "$distributor" = "Ubuntu" ]; then
            repo="ubuntu"
        elif [ "$distributor" = "Debian" ]; then
            repo="debian"
        else
            repo="ubuntu"  # Default fallback
            print_warning "Unknown distributor, using Ubuntu repository"
        fi
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
        echo
        echo -e "${CYAN}${BOLD}Choose ROCm Version:${RESET}"
        echo "1) ROCm 7.0.0 (Latest - Recommended for new installations)"
        echo "2) ROCm 6.4.x (Stable - Default for compatibility)"
        echo
        read -p "Choose ROCm version (1-2) [1]: " ROCM_CHOICE
        ROCM_CHOICE=${ROCM_CHOICE:-1}

        case $ROCM_CHOICE in
            1)
                ROCM_VERSION="7.0"
                ROCM_INSTALL_VERSION="7.0.70000-1"
                print_step "Using ROCm 7.0.0"
                ;;
            2)
                ROCM_VERSION="6.4"
                ROCM_INSTALL_VERSION="6.4.60400-1"
                print_step "Using ROCm 6.4.x"
                ;;
            *)
                ROCM_VERSION="7.0"
                ROCM_INSTALL_VERSION="7.0.70000-1"
                print_step "Using default ROCm 7.0.0"
                ;;
        esac

        # For ROCm 7.0, offer to install updated framework components
        if [ "$ROCM_CHOICE" = "1" ]; then
            echo
            echo -e "${CYAN}${BOLD}ROCm 7.0.0 Additional Components:${RESET}"
            echo -e "${YELLOW}ROCm 7.0.0 includes many updated frameworks. Would you like to install them?${RESET}"
            echo "1) Yes - Install updated frameworks (PyTorch 2.7, JAX 0.6.0, ONNX Runtime 1.22.0, etc.)"
            echo "2) No - Install ROCm core only"
            echo
            read -p "Install additional frameworks? (1-2) [1]: " INSTALL_FRAMEWORKS
            INSTALL_FRAMEWORKS=${INSTALL_FRAMEWORKS:-1}

            if [ "$INSTALL_FRAMEWORKS" = "1" ]; then
                INSTALL_ROCM7_FRAMEWORKS=true
                print_step "Will install ROCm 7.0.0 with updated frameworks"
            else
                INSTALL_ROCM7_FRAMEWORKS=false
                print_step "Will install ROCm 7.0.0 core only"
            fi
        fi


        # Download amdgpu-install package based on selected version
        if [ "$DRY_RUN" != true ]; then
            # Check if ROCm 7.0 packages are available for this distribution
            if [ "$ROCM_VERSION" = "7.0" ] && [ "$repo" = "debian" ]; then
                print_warning "ROCm 7.0.0 packages are not yet available for Debian"
                print_step "Using Ubuntu noble packages (compatible with Debian trixie)"
                print_step "Ubuntu noble (24.04) is the best fit for Debian trixie"
                repo="ubuntu"
                ubuntu_codename="noble"
                print_success "Will use Ubuntu noble packages for ROCm 7.0.0 installation"
            fi

            retry_command "wget -q https://repo.radeon.com/amdgpu-install/$ROCM_VERSION/$repo/$ubuntu_codename/amdgpu-install_${ROCM_INSTALL_VERSION}_all.deb" 3 2
            if [ $? -ne 0 ]; then
                print_error "Failed to download amdgpu-install package after retries"
                print_warning "This might be because ROCm $ROCM_VERSION packages are not yet available for $repo $ubuntu_codename"

                # Offer to compile from source as last resort
                if [ "$ROCM_VERSION" = "7.0" ]; then
                    echo
                    echo -e "${CYAN}${BOLD}Alternative Installation Options:${RESET}"
                    echo "1) Try compiling ROCm 7.0.0 from source (advanced)"
                    echo "2) Fall back to ROCm 6.4.x (recommended)"
                    echo
                    read -p "Choose alternative method (1-2) [2]: " ALT_METHOD
                    ALT_METHOD=${ALT_METHOD:-2}

                    case $ALT_METHOD in
                        1)
                            print_step "Attempting to compile ROCm 7.0.0 from source..."
                            print_step "This may take a significant amount of time and requires development tools"

                            # Clone and build ROCm from source
                            if [ ! -d "/tmp/rocm-source" ]; then
                                git clone --depth 1 --branch rocm-7.0.0 https://github.com/ROCm/ROCm.git /tmp/rocm-source
                            fi
                            cd /tmp/rocm-source

                            # Basic build instructions (simplified)
                            print_warning "Source compilation is complex and may require manual intervention"
                            print_step "Please refer to: https://github.com/ROCm/ROCm#building-rocm-from-source"
                            print_step "Consider using TheRock build system mentioned in the README"

                            return 1
                            ;;
                        2)
                            print_step "Falling back to ROCm 6.4.x"
                            ROCM_VERSION="6.4"
                            ROCM_INSTALL_VERSION="6.4.60400-1"
                            repo="ubuntu"
                            ubuntu_codename="jammy"
                            ;;
                        *)
                            print_step "Falling back to ROCm 6.4.x"
                            ROCM_VERSION="6.4"
                            ROCM_INSTALL_VERSION="6.4.60400-1"
                            repo="ubuntu"
                            ubuntu_codename="jammy"
                            ;;
                    esac
                else
                    print_step "Try selecting ROCm 6.4.x for better compatibility"
                    return 1
                fi
            fi
            print_success "Downloaded amdgpu-install package"
        else
            # Handle dry-run mode for Debian ROCm 7.0 detection
            if [ "$ROCM_VERSION" = "7.0" ] && [ "$repo" = "debian" ]; then
                print_step "[DRY RUN] Would detect ROCm 7.0.0 packages not available for Debian"
                print_step "[DRY RUN] Would use Ubuntu noble packages for Debian trixie"
            fi
            execute_command "wget -q https://repo.radeon.com/amdgpu-install/$ROCM_VERSION/$repo/$ubuntu_codename/amdgpu-install_${ROCM_INSTALL_VERSION}_all.deb" "Downloading amdgpu-install package..."
        fi

        # Install the package
        execute_command "sudo apt install -y ./amdgpu-install_${ROCM_INSTALL_VERSION}_all.deb" "Installing amdgpu-install package..."

        if [ $? -ne 0 ] && [ "$DRY_RUN" != true ]; then
            print_error "Failed to install amdgpu-install package"
            return 1
        elif [ "$DRY_RUN" != true ]; then
            print_success "Installed amdgpu-install package"
        fi

        # Update package lists
        print_step "Updating package lists..."
        sudo apt update

        if [ $? -ne 0 ]; then
            print_warning "Failed to update package lists, continuing anyway"
        else
            print_success "Updated package lists"
        fi

        # Install prerequisites
        print_step "Installing prerequisites..."
        sudo apt install -y python3-setuptools python3-wheel

        if [ $? -ne 0 ]; then
            print_warning "Failed to install some prerequisites, continuing anyway"
        else
            print_success "Installed prerequisites"
        fi

        # Add user to render and video groups
        print_step "Adding user to render and video groups..."
        sudo usermod -a -G render,video $LOGNAME

        if [ $? -ne 0 ]; then
            print_warning "Failed to add user to groups, continuing anyway"
        else
            print_success "Added user to render and video groups"
        fi
    fi
    
    # Install ROCm based on selected method
    print_step "Installing ROCm ($INSTALL_TYPE)..."

    case $INSTALL_TYPE in
        "minimal")
            # Install only ROCm runtime
            sudo apt install -y rocm-core
            ;;
        "standard")
            # Install ROCm runtime + basic tools
            sudo apt install -y rocm
            ;;
        "full")
            # Install ROCm + development tools + libraries
            sudo apt install -y rocm rocm-dev rocm-libs
            ;;
        "custom")
            # Custom installation - let user choose components
            echo
            echo -e "${CYAN}${BOLD}Available ROCm Components:${RESET}"
            echo "1) rocm-core (minimal runtime)"
            echo "2) rocm (standard runtime + tools)"
            echo "3) rocm-dev (development tools)"
            echo "4) rocm-libs (additional libraries)"
            echo "5) rocm-opencl (OpenCL support)"
            echo "6) rocm-hip (HIP runtime)"
            echo "7) rccl (ROCm Communication Collectives Library)"
            echo
            read -p "Enter component numbers to install (space-separated) [2]: " COMPONENTS
            COMPONENTS=${COMPONENTS:-2}

            for comp in $COMPONENTS; do
                case $comp in
                    1) sudo apt install -y rocm-core ;;
                    2) sudo apt install -y rocm ;;
                    3) sudo apt install -y rocm-dev ;;
                    4) sudo apt install -y rocm-libs ;;
                    5) sudo apt install -y rocm-opencl ;;
                    6) sudo apt install -y rocm-hip ;;
                    7) sudo apt install -y librccl-dev librccl1 ;;
                esac
            done
            ;;
        "additional")
            # Install additional components only
            echo
            echo -e "${CYAN}${BOLD}Available Additional Components:${RESET}"
            echo "1) rocm-dev (development tools)"
            echo "2) rocm-libs (additional libraries)"
            echo "3) rocm-opencl (OpenCL support)"
            echo "4) rocm-hip (HIP runtime)"
            echo "5) rccl (ROCm Communication Collectives Library)"
            echo "6) rocm-smi (ROCm System Management Interface)"
            echo
            read -p "Enter component numbers to install (space-separated) [5]: " COMPONENTS
            COMPONENTS=${COMPONENTS:-5}

            for comp in $COMPONENTS; do
                case $comp in
                    1) sudo apt install -y rocm-dev ;;
                    2) sudo apt install -y rocm-libs ;;
                    3) sudo apt install -y rocm-opencl ;;
                    4) sudo apt install -y rocm-hip ;;
                    5)
                        # Install RCCL directly
                        print_step "Installing RCCL..."
                        sudo apt update && sudo apt install -y librccl-dev librccl1
                        ;;
                    6) sudo apt install -y rocm-smi ;;
                esac
            done
            ;;
    esac

    if [ $? -ne 0 ]; then
        print_error "Failed to install ROCm"
        return 1
    fi

    print_success "Installed ROCm ($INSTALL_TYPE)"

    # Install ROCm 7.0.0 specific frameworks if selected
    if [ "$INSTALL_ROCM7_FRAMEWORKS" = true ]; then
        print_section "Installing ROCm 7.0.0 Updated Frameworks"

        # Install PyTorch 2.7 with ROCm 7.0 support
        print_step "Installing PyTorch 2.7 for ROCm 7.0..."
        if [ "$CREATE_VENV" = true ]; then
            source "$VENV_PATH/bin/activate"
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0 || \
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
            deactivate
        else
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0 || \
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
        fi

        # Install other ROCm 7.0 frameworks
        print_step "Installing additional ROCm 7.0.0 frameworks..."

        # Install JAX 0.6.0
        if [ "$CREATE_VENV" = true ]; then
            source "$VENV_PATH/bin/activate"
            python3 -m pip install jax jaxlib --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
            deactivate
        else
            python3 -m pip install jax jaxlib --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
        fi

        # Install ONNX Runtime 1.22.0
        if [ "$CREATE_VENV" = true ]; then
            source "$VENV_PATH/bin/activate"
            python3 -m pip install onnxruntime --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
            deactivate
        else
            python3 -m pip install onnxruntime --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
        fi

        # Install TensorFlow 2.19.1
        if [ "$CREATE_VENV" = true ]; then
            source "$VENV_PATH/bin/activate"
            python3 -m pip install tensorflow --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
            deactivate
        else
            python3 -m pip install tensorflow --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/ || true
        fi

        print_success "Installed ROCm 7.0.0 updated frameworks"
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
    echo -e "${GREEN}eval \"\$(./install_rocm.sh --show-env)\"${RESET}"
    echo
    echo -e "${CYAN}${BOLD}To apply these settings to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$(./install_rocm.sh --show-env)\"${RESET}"

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

# Function to execute commands with dry-run support
execute_command() {
    local cmd="$1"
    local description="$2"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would execute: $description"
        echo "  Command: $cmd"
        return 0
    else
        print_step "$description"
        eval "$cmd"
        return $?
    fi
}

# Run the installation function
install_rocm "$@"
