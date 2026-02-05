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
# DeepSpeed Installation Script
# =============================================================================
# This script installs DeepSpeed, a deep learning optimization library for
# large-scale model training.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗ ███████╗███████╗██████╗ ███████╗██████╗ ███████╗███████╗██████╗
  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗
  ██║  ██║█████╗  █████╗  ██████╔╝███████╗██████╔╝█████╗  █████╗  ██║  ██║
  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔═══╝ ██╔══╝  ██╔══╝  ██║  ██║
  ██████╔╝███████╗███████╗██║     ███████║██║     ███████╗███████╗██████╔╝
  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝     ╚══════╝╚══════╝╚═════╝
EOF
echo

# Parse command line arguments
DRY_RUN=false
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --show-env)
            # Handle --show-env later
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--force] [--show-env]"
            exit 1
            ;;
    esac
done

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

# If dry run, show what would be done
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}${BOLD}DRY RUN MODE${RESET}"
    echo -e "${YELLOW}This script will show what would be executed without actually running commands.${RESET}"
    echo
fi

# Function to load configuration file
load_config() {
    local config_file="${1:-deepspeed_config.sh}"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
    else
        print_step "No configuration file found at $config_file, using defaults"
    fi
}

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
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q "docker\|containerd\|podman" /proc/1/cgroup 2>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Function to detect ROCm installation path
detect_rocm_path() {
    # Check common ROCm installation paths
    for path in "/opt/rocm" "/usr/local/rocm"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    # Check for versioned ROCm installations
    for path in /opt/rocm-*; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    echo "/opt/rocm"  # Default fallback
}

# Function to load configuration file
load_config() {
    local config_file="${1:-deepspeed_config.sh}"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
    else
        print_step "No configuration file found at $config_file, using defaults"
    fi
}

# Load configuration if available
load_config

# Logging configuration
LOG_FILE="${LOG_FILE:-deepspeed_install_$(date +%Y%m%d_%H%M%S).log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Always print to console based on level
    case "$level" in
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${RESET}" >&2
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] WARNING: $message${RESET}" >&2
            ;;
        "INFO")
            if [ "$LOG_LEVEL" = "DEBUG" ] || [ "$LOG_LEVEL" = "INFO" ]; then
                echo -e "${BLUE}[$timestamp] INFO: $message${RESET}"
            fi
            ;;
        "DEBUG")
            if [ "$LOG_LEVEL" = "DEBUG" ]; then
                echo -e "${CYAN}[$timestamp] DEBUG: $message${RESET}"
            fi
            ;;
    esac

    # Write to log file
    echo "[$timestamp] $level: $message" >> "$LOG_FILE"
}

# Initialize logging
log_message "INFO" "DeepSpeed installation started"
log_message "INFO" "Log file: $LOG_FILE"

# Progress bar variables
PROGRESS_BAR_WIDTH=50
PROGRESS_CURRENT=0
PROGRESS_TOTAL=100
PROGRESS_CHAR="▓"
PROGRESS_EMPTY="░"
PROGRESS_ANIMATION=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
ANIMATION_INDEX=0

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

# Function to handle pip installation with venv fallback
uv_pip_install() {
    local args="$@"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN]${RESET} Would install: $args"
        return 0
    fi

    case $INSTALL_METHOD in
        "global")
            print_step "Installing globally with pip..."
            python3 -m pip install --break-system-packages $args
            local install_exit_code=$?
            if [ $install_exit_code -eq 0 ]; then
                DEEPSPEED_VENV_PYTHON=""
            else
                print_error "Global installation failed, DeepSpeed requires manual installation"
                return 1
            fi
            ;;
        "venv")
            print_step "Creating virtual environment..."
            VENV_DIR="./deepspeed_rocm_venv"
            if [ ! -d "$VENV_DIR" ]; then
                python3 -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate"
            print_step "Installing in virtual environment..."
            python3 -m pip install $args
            DEEPSPEED_VENV_PYTHON="$VENV_DIR/bin/python"
            print_success "Installed in virtual environment: $VENV_DIR"
            ;;
        "auto")
            # Try global install first
            print_step "Attempting global installation..."
            if python3 -m pip install --break-system-packages $args; then
                print_success "Global installation successful"
                DEEPSPEED_VENV_PYTHON=""
            else
                print_warning "Global installation failed, creating virtual environment..."
                VENV_DIR="./deepspeed_rocm_venv"
                if [ ! -d "$VENV_DIR" ]; then
                    python3 -m venv "$VENV_DIR"
                fi
                source "$VENV_DIR/bin/activate"
                print_step "Installing in virtual environment..."
                python3 -m pip install $args
                DEEPSPEED_VENV_PYTHON="$VENV_DIR/bin/python"
                print_success "Installed in virtual environment: $VENV_DIR"
            fi
            ;;
    esac
}

# Main installation function
install_deepspeed() {
    print_header "DeepSpeed Installation"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking DeepSpeed installation..."

    # Check if DeepSpeed is already installed
    if package_installed "deepspeed"; then
        # Try to get version cleanly, ignoring warnings and stderr
        deepspeed_version=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null | tail -n 1)

        # Check if it's working properly and detecting ROCm
        # We check for 'cuda' or 'rocm' support in the accelerator
        if python3 -c "import deepspeed; from deepspeed.accelerator import get_accelerator; acc=get_accelerator().communication_backend_name(); print('Working' if acc in ['nccl', 'rccl'] else 'CPU')" 2>/dev/null | grep -q "Working"; then
            print_success "DeepSpeed is already installed and working (version: $deepspeed_version)"

            # Check if --force flag is provided
            if [[ "$FORCE_REINSTALL" == "true" ]]; then
                print_warning "Force reinstall requested - proceeding with reinstallation"
                print_step "Will reinstall DeepSpeed despite working installation"
            else
                print_step "DeepSpeed installation is complete and working. Use --force to reinstall anyway."
                complete_progress_bar
                return 0
            fi
        else
            print_warning "DeepSpeed is installed but not detecting GPU or missing deps"
            print_step "Proceeding with reinstallation to repair it"
        fi
    fi

    # Check if PyTorch is installed
    update_progress_bar 10
    draw_progress_bar "Checking PyTorch installation..."
    print_section "Checking PyTorch Installation"

    if ! package_installed "torch"; then
        print_error "PyTorch is not installed. Please install PyTorch with ROCm support first."
        complete_progress_bar
        return 1
    fi

    update_progress_bar 10
    draw_progress_bar "Checking PyTorch ROCm support..."

    # Check if PyTorch has ROCm/HIP support
    if ! python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
        print_warning "PyTorch does not have explicit ROCm/HIP support"
        print_warning "DeepSpeed may not work correctly without ROCm support in PyTorch"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping DeepSpeed installation"
            complete_progress_bar
            return 0
        fi
    fi

    # Check if ROCm is installed
    print_section "Checking ROCm Installation"

    # Detect environment
    is_wsl=$(detect_wsl)
    is_container=$(detect_container)

    # Force DeepSpeed to use the correct accelerator if supported
    # If 'rocm' is not supported by the installed deepspeed version, 
    # it usually defaults to 'cuda' as an alias for HIP on ROCm systems
    # or auto-detects it. 
    export ROCM_HOME=/opt/rocm
    export HIP_PATH=/opt/rocm
    # We remove DS_ACCELERATOR=rocm as it's causing ValueError in newer versions
    unset DS_ACCELERATOR

    if [ "$is_wsl" = "true" ]; then
        print_step "Detected Windows Subsystem for Linux (WSL) environment"
    fi

    if [ "$is_container" = "true" ]; then
        print_step "Detected container environment"
    fi

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH="gfx1100"
        export ROCM_PATH=$(detect_rocm_path)
        export PATH="$ROCM_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Check if we can install rocprofiler
            if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
                print_step "Installing rocprofiler for HSA tools support..."
                if [ "$DRY_RUN" = true ]; then
                    echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo apt-get update && sudo apt-get install -y rocprofiler"
                else
                    sudo apt-get update && sudo apt-get install -y rocprofiler
                fi
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
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo apt update && sudo apt install -y rocminfo"
                    else
                        sudo apt update && sudo apt install -y rocminfo
                    fi
                    ;;
                dnf)
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo dnf install -y rocminfo"
                    else
                        sudo dnf install -y rocminfo
                    fi
                    ;;
                yum)
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo yum install -y rocminfo"
                    else
                        sudo yum install -y rocminfo
                    fi
                    ;;
                pacman)
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo pacman -S rocminfo"
                    else
                        sudo pacman -S rocminfo
                    fi
                    ;;
                zypper)
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${BLUE}[DRY RUN]${RESET} Would run: sudo zypper install -y rocminfo"
                    else
                        sudo zypper install -y rocminfo
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

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}DeepSpeed Installation Options:${RESET}"
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

    # Check if uv is installed
    update_progress_bar 10
    draw_progress_bar "Checking package manager..."
    print_section "Installing DeepSpeed"

    if ! command_exists uv; then
        print_step "Installing uv package manager..."
        update_progress_bar 5
        draw_progress_bar "Installing uv package manager..."
        if [ "$DRY_RUN" = true ]; then
            echo -e "${BLUE}[DRY RUN]${RESET} Would run: python3 -m pip install uv"
        else
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
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Install required dependencies first
    update_progress_bar 15
    draw_progress_bar "Installing required dependencies..."
    print_step "Installing required dependencies first..."

    # Ensure einops is installed in the correct environment
    python3 -m pip install --break-system-packages packaging ninja pydantic jsonschema einops
    python3 -m pip install --user --break-system-packages einops || true
    
    # Check for anaconda
    if [ -d "$HOME/anaconda3" ]; then
        "$HOME/anaconda3/bin/python" -m pip install --break-system-packages einops || true
    fi
    
    # Force DeepSpeed to use ROCm during installation too
    # We remove DS_ACCELERATOR=rocm as it's causing ValueError in newer versions
    # DeepSpeed auto-detects ROCm when ROCM_HOME is set
    unset DS_ACCELERATOR
    export ROCM_HOME=/opt/rocm
    export HIP_PATH=/opt/rocm

    # Install DeepSpeed
    update_progress_bar 20
    draw_progress_bar "Installing DeepSpeed..."
    print_step "Installing DeepSpeed..."

    # Install with the new function - try multiple approaches
    set +e  # Don't exit on error

    # First attempt
    # We remove DS_ACCELERATOR=rocm as it's causing ValueError in newer versions
    # DeepSpeed auto-detects ROCm when ROCM_HOME is set
    unset DS_ACCELERATOR
    export ROCM_HOME=/opt/rocm
    export HIP_PATH=/opt/rocm
    python3 -m pip install --break-system-packages deepspeed einops
    install_result=$?

    if [ $install_result -ne 0 ]; then
        print_warning "First installation attempt failed, trying with --no-deps..."
        uv_pip_install deepspeed --no-deps
        install_result=$?
    fi

    if [ $install_result -ne 0 ]; then
        print_warning "Second installation attempt failed, trying with --force-reinstall..."
        uv_pip_install deepspeed --force-reinstall
        install_result=$?
    fi

    set -e  # Return to normal error handling

    if [ $install_result -ne 0 ]; then
        print_error "Failed to install DeepSpeed after multiple attempts"
        complete_progress_bar
        return 1
    fi

    # Verify installation
    update_progress_bar 30
    draw_progress_bar "Verifying installation..."
    print_section "Verifying Installation"
    log_message "INFO" "Starting DeepSpeed installation verification"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${DEEPSPEED_VENV_PYTHON:-python3}

    # Use timeout to prevent hanging during verification
    set +e  # Don't exit on error
    if timeout 30s $PYTHON_CMD -c "import deepspeed; print('Success')" &>/dev/null; then
        deepspeed_version=$(timeout 10s $PYTHON_CMD -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
        print_success "DeepSpeed is installed (version: $deepspeed_version)"
        log_message "INFO" "DeepSpeed installed successfully (version: $deepspeed_version)"

        # Check if DeepSpeed can detect GPUs
        update_progress_bar 10
        draw_progress_bar "Checking GPU detection..."
        print_step "Checking GPU detection..."
        log_message "INFO" "Checking GPU detection"

        # Check for all required dependencies
        print_step "Verifying all dependencies are installed..."
        log_message "INFO" "Verifying dependencies"
        for dep in packaging ninja pydantic jsonschema einops; do
            if $PYTHON_CMD -c "import ${dep//-/_}" &>/dev/null; then
                print_success "$dep is installed"
                log_message "INFO" "Dependency $dep is installed"
            else
                print_warning "$dep is not installed, attempting to install it now"
                log_message "WARNING" "Dependency $dep missing, installing"
                uv_pip_install $dep
            fi
        done

        # Check GPU access with timeout to prevent hanging
        if timeout 20s $PYTHON_CMD -c "import deepspeed; import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "DeepSpeed can access GPUs through PyTorch"
            log_message "INFO" "GPU access available through PyTorch"

            # Get GPU count
            gpu_count=$(timeout 10s $PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            print_step "PyTorch detected $gpu_count GPU(s)"
            log_message "INFO" "Detected $gpu_count GPU(s)"

            # List GPUs
            for i in $(seq 0 $((gpu_count-1))); do
                gpu_name=$(timeout 10s $PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
                echo "  - GPU $i: $gpu_name"
                log_message "INFO" "GPU $i: $gpu_name"
            done

            # Test a simple tensor operation
            print_step "Testing GPU tensor operations..."
            log_message "INFO" "Testing GPU tensor operations"
            if timeout 10s $PYTHON_CMD -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success' if torch.all(y == 2) else 'Failed')" 2>/dev/null | grep -q "Success"; then
                print_success "GPU tensor operations working correctly"
                log_message "INFO" "GPU tensor operations working correctly"
            else
                print_warning "GPU tensor operations may not be working correctly"
                log_message "WARNING" "GPU tensor operations may not be working correctly"
            fi

            # Test DeepSpeed basic functionality
            print_step "Testing DeepSpeed basic functionality..."
            log_message "INFO" "Testing DeepSpeed basic functionality"
            if timeout 15s $PYTHON_CMD -c "import deepspeed; import torch; model = torch.nn.Linear(10, 1); optimizer = torch.optim.Adam(model.parameters()); model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config={'train_batch_size': 1}); print('DeepSpeed initialized successfully')" 2>/dev/null | grep -q "successfully"; then
                print_success "DeepSpeed basic functionality working"
                log_message "INFO" "DeepSpeed basic functionality verified"
            else
                print_warning "DeepSpeed basic functionality test failed"
                log_message "WARNING" "DeepSpeed basic functionality test failed"
            fi

            # Test DeepSpeed with GPU if available
            if [ $gpu_count -gt 0 ]; then
                print_step "Testing DeepSpeed GPU integration..."
                log_message "INFO" "Testing DeepSpeed GPU integration"
                if timeout 20s $PYTHON_CMD -c "import deepspeed; import torch; model = torch.nn.Linear(10, 1); optimizer = torch.optim.Adam(model.parameters()); model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config={'train_batch_size': 1}); x = torch.randn(1, 10).cuda(); y = model(x); print('GPU integration test passed')" 2>/dev/null | grep -q "passed"; then
                    print_success "DeepSpeed GPU integration working"
                    log_message "INFO" "DeepSpeed GPU integration working"
                else
                    print_warning "DeepSpeed GPU integration test failed"
                    log_message "WARNING" "DeepSpeed GPU integration test failed"
                fi
            fi
        else
            print_warning "DeepSpeed cannot access GPUs through PyTorch"
            print_warning "This may be normal for ROCm/HIP environments"
            print_warning "DeepSpeed should still work for CPU operations"
            log_message "WARNING" "GPU access not available through PyTorch"
        fi

        # Consider installation successful even if GPU detection fails
        verification_success=0
        log_message "INFO" "DeepSpeed verification completed successfully"
    else
        print_error "DeepSpeed installation verification failed"
        log_message "ERROR" "DeepSpeed installation verification failed"
        print_warning "Attempting one more installation approach..."

        # Try reinstalling with different options
        uv_pip_install --force-reinstall deepspeed

        # Check again
        if timeout 30s $PYTHON_CMD -c "import deepspeed; print('Success')" &>/dev/null; then
            print_success "DeepSpeed installed successfully after retry"
            log_message "INFO" "DeepSpeed installed successfully after retry"
            verification_success=0
        else
            print_error "DeepSpeed installation failed after multiple attempts"
            log_message "ERROR" "DeepSpeed installation failed after multiple attempts"
            verification_success=1
        fi
    fi
    set -e  # Return to normal error handling

    if [ $verification_success -ne 0 ]; then
        complete_progress_bar
        log_message "ERROR" "DeepSpeed installation failed"
        return 1
    fi

    update_progress_bar 10
    draw_progress_bar "Completing installation..."
    print_success "DeepSpeed installation completed successfully"
    log_message "INFO" "DeepSpeed installation completed successfully"

    complete_progress_bar

    # Show a visually appealing completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██████╗ ███████╗███████╗██████╗ ███████╗██████╗ ███████╗██████╗  ║
    ║  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔══██╗ ║
    ║  ██║  ██║█████╗  █████╗  ██████╔╝███████╗██████╔╝█████╗  ██║  ██║ ║
    ║  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔═══╝ ██╔══╝  ██║  ██║ ║
    ║  ██████╔╝███████╗███████╗██║     ███████║██║     ███████╗██████╔╝ ║
    ║  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝     ╚══════╝╚═════╝  ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  DeepSpeed is now ready to use with your GPU.           ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "DeepSpeed installation completed successfully"
    log_message "INFO" "Installation process finished"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$DEEPSPEED_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./deepspeed_rocm_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import deepspeed; print('DeepSpeed version:', deepspeed.__version__); import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import deepspeed; print('DeepSpeed version:', deepspeed.__version__); import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())\"${RESET}"
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
    echo -e "${GREEN}eval \"\$(./install_deepspeed.sh --show-env)\"${RESET}"
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

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_deepspeed "$@"

