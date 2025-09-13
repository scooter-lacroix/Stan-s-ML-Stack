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
# MIGraphX Python Installation Script
# =============================================================================
# This script installs MIGraphX Python bindings for AMD GPUs.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ███╗   ███╗██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗    ██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗
  ████╗ ████║██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║    ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║
  ██╔████╔██║██║██║  ███╗██████╔╝███████║██████╔╝███████║    ██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║
  ██║╚██╔╝██║██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║    ██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║
  ██║ ╚═╝ ██║██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║    ██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║
  ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝    ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
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

# Function to detect ROCm version
detect_rocm_version() {
    local rocm_version=""

    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
    fi

    if [ -z "$rocm_version" ]; then
        rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
    fi

    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    echo "$rocm_version"
}

# Function to detect GPU architecture
detect_gpu_architecture() {
    local gpu_arch=""

    if command_exists rocminfo; then
        # Try to get GPU architecture from rocminfo
        gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -o "gfx[0-9]\+" | head -n 1)
    fi

    if [ -z "$gpu_arch" ]; then
        # Fallback to common architectures based on ROCm version
        local rocm_version=$(detect_rocm_version)
        local rocm_major=$(echo "$rocm_version" | cut -d '.' -f 1)

        if [ "$rocm_major" -ge 6 ]; then
            gpu_arch="gfx1100"  # RDNA3 architecture
        else
            gpu_arch="gfx1030"  # RDNA2 architecture
        fi

        print_warning "Could not detect GPU architecture, using default: $gpu_arch"
    else
        print_success "Detected GPU architecture: $gpu_arch"
    fi

    echo "$gpu_arch"
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

# Function to load configuration file
load_config() {
    local config_file="${MIGRAPHX_CONFIG_FILE:-./migraphx_config.sh}"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
    else
        print_step "No configuration file found at $config_file, using defaults"
    fi
}

# Function to setup logging
setup_logging() {
    local log_file="${MIGRAPHX_LOG_FILE:-./migraphx_install_$(date +%Y%m%d_%H%M%S).log}"

    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$log_file")"

    # Redirect stdout and stderr to log file while still showing on console
    exec > >(tee -a "$log_file") 2>&1

    print_step "Logging to: $log_file"
    echo "=== MIGraphX Python Installation Log $(date) ===" >> "$log_file"
}

# Function to retry command with backoff
retry_command() {
    local max_attempts=3
    local attempt=1
    local exit_code=0

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt of $max_attempts..."

        if "$@"; then
            return 0
        else
            exit_code=$?
            print_warning "Command failed (attempt $attempt/$max_attempts)"

            if [ $attempt -lt $max_attempts ]; then
                local delay=$((attempt * 2))
                print_step "Retrying in $delay seconds..."
                sleep $delay
            fi
        fi

        ((attempt++))
    done

    return $exit_code
}

# Function to install rocminfo if missing
install_rocminfo() {
    local package_manager=$(detect_package_manager)

    print_step "Installing rocminfo..."
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
        return 0
    else
        print_error "Failed to install rocminfo"
        return 1
    fi
}

# Function to check ROCm installation
check_rocm_installation() {
    print_section "Checking ROCm Installation"

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Detect GPU architecture for optimal configuration
        local gpu_arch=$(detect_gpu_architecture)

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
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
        return 0
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            install_rocminfo
            return $?
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi
}

# Function to check MIGraphX installation
check_migraphx_installation() {
    print_section "Checking MIGraphX Installation"

    if ! command_exists migraphx-driver; then
        print_error "MIGraphX is not installed. Please install MIGraphX first."
        print_step "Run the install_migraphx.sh script to install MIGraphX."
        return 1
    fi

    print_success "MIGraphX is installed"
    return 0
}

# Function to check if MIGraphX Python module is installed
check_migraphx_python() {
    local python_cmd="${MIGRAPHX_VENV_PYTHON:-python3}"

    if $python_cmd -c "import migraphx" &>/dev/null; then
        local migraphx_version=$($python_cmd -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
        print_success "MIGraphX Python module is already installed (version: $migraphx_version)"

        # Check if --force flag is provided
        if [[ "$*" == *"--force"* ]] || [[ "$MIGRAPHX_REINSTALL" == "true" ]]; then
            print_warning "Force reinstall requested - proceeding with reinstallation"
            print_step "Will reinstall MIGraphX Python despite working installation"
        else
            print_step "MIGraphX Python installation is complete and working. Use --force to reinstall anyway."
            return 0
        fi
    fi

    return 1
}

# Function to install uv package manager
install_uv() {
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

    return 0
}

# Function to handle uv commands properly with venv fallback
uv_pip_install() {
    local args="$@"

    # Check if uv is available as a command
    if command -v uv &> /dev/null; then
        case $INSTALL_METHOD in
            "global")
                print_step "Installing globally with pip..."
                python3 -m pip install --break-system-packages $args
                MIGRAPHX_VENV_PYTHON=""
                ;;
            "venv")
                print_step "Creating uv virtual environment..."
                VENV_DIR="./migraphx_python_venv"
                if [ ! -d "$VENV_DIR" ]; then
                    uv venv "$VENV_DIR"
                fi
                source "$VENV_DIR/bin/activate"
                print_step "Installing in virtual environment..."
                uv pip install $args
                MIGRAPHX_VENV_PYTHON="$VENV_DIR/bin/python"
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
                    VENV_DIR="./migraphx_python_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args

                    # Store venv path for verification
                    MIGRAPHX_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                elif [ $install_exit_code -eq 0 ]; then
                    print_success "Global installation successful"
                    MIGRAPHX_VENV_PYTHON=""
                else
                    print_error "Global installation failed with unknown error:"
                    echo "$install_output"
                    print_step "Falling back to virtual environment..."

                    # Create uv venv in project directory
                    VENV_DIR="./migraphx_python_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi

                    # Activate venv and install
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args

                    # Store venv path for verification
                    MIGRAPHX_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                fi
                ;;
        esac
    else
        # Fall back to pip
        print_step "Installing with pip..."
        python3 -m pip install $args
        MIGRAPHX_VENV_PYTHON=""
    fi
}

# Function to install MIGraphX Python module
install_migraphx_python() {
    print_header "MIGraphX Python Installation"

    # Load configuration file
    load_config

    # Setup logging
    setup_logging

    # Detect environment
    if detect_wsl; then
        print_step "Detected WSL environment"
    fi

    if detect_container; then
        print_step "Detected container environment"
    fi

    # Check for --dry-run option
    if [[ "$*" == *"--dry-run"* ]]; then
        print_step "DRY RUN MODE - No changes will be made"
        echo
        echo "Would perform the following actions:"
        echo "1. Check ROCm installation"
        echo "2. Check MIGraphX installation"
        echo "3. Install uv package manager if needed"
        echo "4. Install MIGraphX Python module"
        echo "5. Verify installation"
        echo
        print_success "Dry run completed"
        return 0
    fi

    # Check ROCm installation
    if ! check_rocm_installation; then
        return 1
    fi

    # Check MIGraphX installation
    if ! check_migraphx_installation; then
        return 1
    fi

    # Check if MIGraphX Python is already installed
    if check_migraphx_python "$@"; then
        return 0
    fi

    # Install uv if not present
    install_uv

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}MIGraphX Python Installation Options:${RESET}"
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

    # Set environment variables
    export ROCM_PATH=/opt/rocm
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    export PATH=$ROCM_PATH/bin:$PATH

    # Suppress HIP logs
    export AMD_LOG_LEVEL=0
    export HIP_VISIBLE_DEVICES=0,1,2
    export ROCR_VISIBLE_DEVICES=0,1,2

    # Install MIGraphX Python module
    print_section "Installing MIGraphX Python Module"

    print_step "Installing MIGraphX Python module..."
    uv_pip_install migraphx

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    local PYTHON_CMD=${MIGRAPHX_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import migraphx" &>/dev/null; then
        local migraphx_version=$($PYTHON_CMD -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
        print_success "MIGraphX Python module is installed (version: $migraphx_version)"

        # Test basic functionality
        print_step "Testing MIGraphX Python functionality..."
        if $PYTHON_CMD -c "import migraphx; print('MIGraphX import successful')" &>/dev/null; then
            print_success "MIGraphX Python module working correctly"
        else
            print_warning "MIGraphX Python module may not be working correctly"
        fi

        # Performance benchmarking
        print_step "Running performance benchmark..."
        local benchmark_script="
import migraphx as mgx
import numpy as np
import time

# Create a simple model for benchmarking
print('Creating benchmark model...')
x = mgx.argument(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
y = mgx.add(x, x)
prog = mgx.compile(y, offload_copy=False)

# Warm up
print('Warming up...')
for _ in range(10):
    result = prog.run({})[0]

# Benchmark
print('Running benchmark...')
times = []
for _ in range(100):
    start = time.time()
    result = prog.run({})[0]
    end = time.time()
    times.append(end - start)

avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
print(f'Average inference time: {avg_time:.2f} ms')
print(f'Min time: {min(times)*1000:.2f} ms')
print(f'Max time: {max(times)*1000:.2f} ms')
print('Benchmark completed successfully')
"

        if $PYTHON_CMD -c "$benchmark_script" &>/dev/null; then
            print_success "Performance benchmark completed successfully"
        else
            print_warning "Performance benchmark failed, but installation may still be functional"
        fi

    else
        print_error "MIGraphX Python module installation failed"
        return 1
    fi

    # Show completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ███╗   ███╗██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗ ║
    ║  ████╗ ████║██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║ ║
    ║  ██╔████╔██║██║██║  ███╗██████╔╝███████║██████╔╝███████║ ║
    ║  ██║╚██╔╝██║██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║ ║
    ║  ██║ ╚═╝ ██║██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║ ║
    ║  ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  MIGraphX Python is now ready to use with your GPU.     ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "MIGraphX Python installation completed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$MIGRAPHX_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./migraphx_python_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import migraphx; print('MIGraphX version:', getattr(migraphx, '__version__', 'unknown'))\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import migraphx; print('MIGraphX version:', getattr(migraphx, '__version__', 'unknown'))\"${RESET}"
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
    echo -e "${GREEN}eval \"\$(./install_migraphx_python.sh --show-env)\"${RESET}"
    echo

    # Add a small delay to ensure the message is seen
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"

    return 0
}

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_migraphx_python "$@"

exit $?
