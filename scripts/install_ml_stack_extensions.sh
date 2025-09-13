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
# ML Stack Extensions Master Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures additional components to enhance the
# ML stack for AMD GPUs, including:
#
# 1. Triton - Compiler for parallel programming
# 2. BITSANDBYTES - Efficient quantization
# 3. vLLM - High-throughput inference engine
# 4. ROCm SMI - Monitoring and profiling
# 5. PyTorch Profiler - Performance analysis
# 6. WandB - Experiment tracking
# 7. Flash Attention CK - Optimized attention for AMD GPUs
#
# Enhanced Features:
# - Multi-package-manager ROCm detection
# - Installation method choices (global/venv/auto-detect)
# - Virtual environment support with uv
# - Externally managed environment handling
# - Enhanced error handling and recovery
# - Cross-platform compatibility
# - Comprehensive verification and testing
#
# Author: Stanley Chisango (Scooter Lacroix)
# Date: $(date +"%Y-%m-%d")
# =============================================================================

# Enhanced error handling - don't exit immediately on errors
set +e

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
LOG_FILE="$LOG_DIR/ml_stack_extensions_install_$(date +"%Y%m%d_%H%M%S").log"
DRY_RUN=false
FORCE_INSTALL=false
INSTALL_METHOD="auto"
PYTORCH_VENV_PYTHON=""
VENV_DIR=""
ROCM_DETECTED=false
ROCM_VERSION=""
PACKAGE_MANAGER=""

# Color support detection
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

# Enhanced printing functions
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

print_separator() {
    echo "───────────────────────────────────────────────────────────"
}

# Enhanced logging function
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    local python_cmd=${PYTORCH_VENV_PYTHON:-python3}
    $python_cmd -c "import $1" &>/dev/null
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
        log "Installing $package with uv..."
        if [ -n "$PYTORCH_VENV_PYTHON" ]; then
            uv pip install $extra_args "$package"
        else
            uv pip install --python $(which python3) $extra_args "$package"
        fi
    else
        log "Installing $package with pip..."
        local python_cmd=${PYTORCH_VENV_PYTHON:-python3}
        $python_cmd -m pip install $extra_args "$package"
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

# Function to setup ROCm environment variables
setup_rocm_env() {
    print_step "Setting up ROCm environment variables..."

    # Set ROCm environment variables
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
}

# Enhanced ROCm detection function
detect_rocm() {
    print_section "Detecting ROCm Installation"

    PACKAGE_MANAGER=$(detect_package_manager)

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Setup ROCm environment
        setup_rocm_env

        # Detect ROCm version
        ROCM_VERSION=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$ROCM_VERSION" ]; then
            ROCM_VERSION=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        fi

        if [ -z "$ROCM_VERSION" ]; then
            print_warning "Could not detect ROCm version, using default version 6.4.0"
            ROCM_VERSION="6.4.0"
        else
            print_success "Detected ROCm version: $ROCM_VERSION"
        fi

        ROCM_DETECTED=true
        return 0
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            case $PACKAGE_MANAGER in
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
                    print_error "Unsupported package manager: $PACKAGE_MANAGER"
                    return 1
                    ;;
            esac
            if command_exists rocminfo; then
                print_success "Installed rocminfo"
                ROCM_DETECTED=true
                setup_rocm_env
                return 0
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" > /dev/null
}

# Function to handle installation method selection
select_installation_method() {
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
}

# Function to setup virtual environment with uv
setup_virtual_env() {
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
            return 1
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Create virtual environment
    VENV_DIR="./ml_stack_extensions_venv"
    if [ ! -d "$VENV_DIR" ]; then
        print_step "Creating virtual environment..."
        uv venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    PYTORCH_VENV_PYTHON="$VENV_DIR/bin/python"

    print_success "Virtual environment created and activated: $VENV_DIR"
    return 0
}

# Enhanced package installation with fallback handling
install_package_with_fallback() {
    local package="$1"
    local args="$2"

    case $INSTALL_METHOD in
        "global")
            print_step "Installing globally with pip..."
            python3 -m pip install --break-system-packages $args "$package"
            PYTORCH_VENV_PYTHON=""
            ;;
        "venv")
            if [ -z "$VENV_DIR" ]; then
                setup_virtual_env
            fi
            print_step "Installing in virtual environment..."
            uv pip install $args "$package"
            ;;
        "auto")
            # Try global install first
            print_step "Attempting global installation..."
            local install_output
            install_output=$(python3 -m pip install $args "$package" 2>&1)
            local install_exit_code=$?

            if echo "$install_output" | grep -q "externally managed"; then
                print_warning "Global installation failed due to externally managed environment"
                print_step "Creating virtual environment for installation..."

                if [ -z "$VENV_DIR" ]; then
                    setup_virtual_env
                fi

                print_step "Installing in virtual environment..."
                uv pip install $args "$package"

                print_success "Installed in virtual environment: $VENV_DIR"
            elif [ $install_exit_code -eq 0 ]; then
                print_success "Global installation successful"
                PYTORCH_VENV_PYTHON=""
            else
                print_error "Global installation failed with unknown error:"
                echo "$install_output"
                print_step "Falling back to virtual environment..."

                if [ -z "$VENV_DIR" ]; then
                    setup_virtual_env
                fi

                print_step "Installing in virtual environment..."
                uv pip install $args "$package"

                print_success "Installed in virtual environment: $VENV_DIR"
            fi
            ;;
    esac
}


# Enhanced component installation function
install_component() {
    local component="$1"
    local script_path="$PROJECT_ROOT/scripts/install_${component}.sh"

    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would install $component from $script_path"
        return 0
    fi

    if [ -f "$script_path" ]; then
        print_step "Installing $component..."
        chmod +x "$script_path"

        # Run the component installation script
        if "$script_path"; then
            print_success "$component installation completed successfully"
            return 0
        else
            print_error "$component installation failed"
            return 1
        fi
    else
        print_error "Installation script for $component not found at $script_path"
        return 1
    fi
}

# Function to verify component installation
verify_component() {
    local component="$1"
    local python_cmd=${PYTORCH_VENV_PYTHON:-python3}

    case $component in
        "triton")
            if $python_cmd -c "import triton" &>/dev/null; then
                print_success "Triton verification passed"
                return 0
            fi
            ;;
        "bitsandbytes")
            if $python_cmd -c "import bitsandbytes" &>/dev/null; then
                print_success "BitsAndBytes verification passed"
                return 0
            fi
            ;;
        "vllm")
            if $python_cmd -c "import vllm" &>/dev/null; then
                print_success "vLLM verification passed"
                return 0
            fi
            ;;
        "rocm_smi")
            if command_exists rocm-smi; then
                print_success "ROCm SMI verification passed"
                return 0
            fi
            ;;
        "pytorch_profiler")
            if $python_cmd -c "import torch; import torch.profiler" &>/dev/null; then
                print_success "PyTorch Profiler verification passed"
                return 0
            fi
            ;;
        "wandb")
            if $python_cmd -c "import wandb" &>/dev/null; then
                print_success "Weights & Biases verification passed"
                return 0
            fi
            ;;
        "flash_attention_ck")
            if $python_cmd -c "import flash_attn" &>/dev/null; then
                print_success "Flash Attention CK verification passed"
                return 0
            fi
            ;;
    esac

    print_warning "$component verification failed"
    return 1
}

# Function to show usage information
show_usage() {
    cat << EOF
ML Stack Extensions Master Installation Script

USAGE:
    $0 [OPTIONS] [COMPONENTS...]

OPTIONS:
    -h, --help              Show this help message
    -d, --dry-run           Show what would be installed without actually installing
    -f, --force             Force reinstallation of components
    --global                Use global installation method
    --venv                  Use virtual environment installation method
    --auto                  Use auto-detect installation method (default)
    --show-env              Show ROCm environment variables
    --verify-only           Only verify existing installations

COMPONENTS:
    triton                  Triton compiler for parallel programming
    bitsandbytes            Efficient quantization library
    vllm                    High-throughput inference engine
    rocm_smi                ROCm system monitoring interface
    pytorch_profiler        PyTorch performance profiling tools
    wandb                   Weights & Biases experiment tracking
    flash_attention_ck      Optimized attention for AMD GPUs
    all                     Install all components

EXAMPLES:
    $0 --dry-run all                    # Preview installation of all components
    $0 --venv triton bitsandbytes       # Install specific components in venv
    $0 --force --global vllm            # Force reinstall vLLM globally
    $0 --show-env                       # Show ROCm environment variables

EOF
}

# Main installation function
main() {
    print_header "ML Stack Extensions Installation"

    # Parse command line arguments
    local verify_only=false
    local components=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                print_warning "DRY RUN MODE: No actual installations will be performed"
                shift
                ;;
            -f|--force)
                FORCE_INSTALL=true
                print_step "Force reinstallation enabled"
                shift
                ;;
            --global)
                INSTALL_METHOD="global"
                print_step "Using global installation method"
                shift
                ;;
            --venv)
                INSTALL_METHOD="venv"
                print_step "Using virtual environment method"
                shift
                ;;
            --auto)
                INSTALL_METHOD="auto"
                print_step "Using auto-detect method"
                shift
                ;;
            --show-env)
                show_env
                exit 0
                ;;
            --verify-only)
                verify_only=true
                print_step "Verification mode enabled"
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                components+=("$1")
                shift
                ;;
        esac
    done

    # Set default components if none specified
    if [ ${#components[@]} -eq 0 ]; then
        components=("rocm_smi" "pytorch_profiler" "wandb")
    fi

    # Handle 'all' component
    if [[ " ${components[@]} " =~ " all " ]]; then
        components=("triton" "bitsandbytes" "vllm" "rocm_smi" "pytorch_profiler" "wandb" "flash_attention_ck")
    fi

    # Detect ROCm
    if ! detect_rocm; then
        print_error "ROCm detection failed. Cannot proceed with installation."
        exit 1
    fi

    # Check for running processes
    if is_process_running "onnxruntime"; then
        print_warning "ONNX Runtime build is currently running. Will not interrupt it."
    fi

    # Setup installation method
    if [ "$INSTALL_METHOD" = "auto" ]; then
        select_installation_method
    fi

    # Setup virtual environment if needed
    if [ "$INSTALL_METHOD" = "venv" ]; then
        if ! setup_virtual_env; then
            print_error "Failed to setup virtual environment"
            exit 1
        fi
    fi

    # Check dependencies
    print_section "Checking Dependencies"
    local deps=("git" "python3" "pip")
    local missing_deps=()

    for dep in "${deps[@]}"; do
        if ! command_exists "$dep"; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_step "Please install them and run this script again."
        exit 1
    fi

    print_success "All dependencies found"

    # Create installation directory
    local install_dir="$HOME/ml_stack"
    mkdir -p "$install_dir"

    # Process components
    if [ "$verify_only" = true ]; then
        print_section "Verifying Existing Installations"
        for component in "${components[@]}"; do
            verify_component "$component"
        done
    else
        print_section "Installing Components"
        local failed_components=()

        for component in "${components[@]}"; do
            if install_component "$component"; then
                # Verify installation
                if verify_component "$component"; then
                    print_success "$component fully installed and verified"
                else
                    print_warning "$component installed but verification failed"
                fi
            else
                failed_components+=("$component")
                print_error "$component installation failed"
            fi
        done

        # Report results
        print_separator
        if [ ${#failed_components[@]} -eq 0 ]; then
            print_success "All components installed successfully!"
        else
            print_warning "Some components failed to install: ${failed_components[*]}"
        fi

        # Show environment setup instructions
        if [ -n "$VENV_DIR" ]; then
            echo
            echo -e "${CYAN}${BOLD}Virtual Environment Setup:${RESET}"
            echo -e "${GREEN}source $VENV_DIR/bin/activate${RESET}"
            echo -e "${YELLOW}To activate the virtual environment in future sessions${RESET}"
        fi

        # Show ROCm environment variables
        echo
        echo -e "${CYAN}${BOLD}ROCm Environment Variables:${RESET}"
        show_env

        # Final information
        print_separator
        print_success "Installation complete!"
        echo "Installation Directory: $install_dir"
        echo "Log File: $LOG_FILE"
        echo "Documentation: $PROJECT_ROOT/docs/extensions/"
        echo
        echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}Make sure to set the ROCm environment variables in your shell profile${RESET}"
        echo -e "${YELLOW}for future sessions. You can run: eval \"\$($0 --show-env)\"${RESET}"
    fi
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    main "$@"
fi
