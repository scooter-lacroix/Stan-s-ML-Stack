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
# ROCm SMI Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures ROCm System Management Interface (SMI)
# for monitoring and profiling AMD GPUs with enhanced robustness and user experience.
# =============================================================================

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

# ASCII Art Banner
cat << "EOF"
  ██████╗  ██████╗  ██████╗███╗   ███╗    ███████╗███╗   ███╗██╗
  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██╔════╝████╗ ████║██║
  ██████╔╝██║   ██║██║     ██╔████╔██║    ███████╗██╔████╔██║██║
  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ╚════██║██║╚██╔╝██║██║
  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ███████║██║ ╚═╝ ██║██║
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚══════╝╚═╝     ╚═╝╚═╝
EOF
echo

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

# Set script options
set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/rocm_smi_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Parse command line arguments
DRY_RUN=false
FORCE=false
SHOW_ENV=false
INSTALL_METHOD="auto"

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
            SHOW_ENV=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run     Show what would be done without making changes"
            echo "  --force       Force reinstallation even if already installed"
            echo "  --show-env    Show ROCm environment variables"
            echo "  --help        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ROCM_SMI_VENV_PYTHON  Path to Python executable in virtual environment"
            echo "  ROCM_SMI_REINSTALL    Set to 'true' to force reinstallation"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for force reinstall from environment variable
if [[ "$ROCM_SMI_REINSTALL" == "true" ]] || [[ "$FORCE" == "true" ]]; then
    FORCE=true
fi

# Main installation function
install_rocm_smi() {
    print_header "ROCm SMI Installation"

    # Check if ROCm SMI is already installed
    if command_exists rocm-smi; then
        rocm_smi_version=$(rocm-smi --version 2>&1 | head -n 1)
        print_success "ROCm SMI is already installed (version: $rocm_smi_version)"

        # Check if --force flag is provided
        if [[ "$FORCE" == "true" ]]; then
            print_warning "Force reinstall requested - proceeding with reinstallation"
            print_step "Will reinstall ROCm SMI despite working installation"
        else
            print_step "ROCm SMI installation is complete. Use --force to reinstall."
            return 0
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

    # Check if uv is installed
    print_section "Installing ROCm SMI with Python Support"

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
    echo -e "${CYAN}${BOLD}ROCm SMI Installation Options:${RESET}"
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

    # Create installation directory
    INSTALL_DIR="$HOME/ml_stack/rocm_smi"
    mkdir -p $INSTALL_DIR
    cd $INSTALL_DIR

    # Clone ROCm SMI repository
    if [ ! -d "rocm_smi_lib" ]; then
        print_step "Cloning ROCm SMI repository..."
        if [[ "$DRY_RUN" == "true" ]]; then
            print_step "[DRY RUN] Would clone https://github.com/RadeonOpenCompute/rocm_smi_lib.git"
        else
            git clone https://github.com/RadeonOpenCompute/rocm_smi_lib.git
            if [ $? -ne 0 ]; then
                print_error "Failed to clone ROCm SMI repository"
                return 1
            fi
            print_success "Cloned ROCm SMI repository"
        fi
        cd rocm_smi_lib
    else
        print_step "ROCm SMI repository already exists, updating..."
        cd rocm_smi_lib
        if [[ "$DRY_RUN" == "true" ]]; then
            print_step "[DRY RUN] Would update repository"
        else
            git pull
            if [ $? -ne 0 ]; then
                print_warning "Failed to update repository, continuing with existing version"
            else
                print_success "Updated ROCm SMI repository"
            fi
        fi
    fi

    # Install Python wrapper
    print_step "Installing ROCm SMI Python wrapper..."
    if [ -d "python_smi_tools" ]; then
        cd python_smi_tools
        if [ -f "setup.py" ]; then
            print_step "Installing ROCm SMI Python wrapper using enhanced package management..."

            # Create a function to handle uv commands properly with venv fallback
            uv_pip_install() {
                local args="$@"

                # Check if uv is available as a command
                if command -v uv &> /dev/null; then
                    case $INSTALL_METHOD in
                        "global")
                            print_step "Installing globally with pip..."
                            python3 -m pip install --break-system-packages $args
                            ROCM_SMI_VENV_PYTHON=""
                            ;;
                        "venv")
                            print_step "Creating uv virtual environment..."
                            VENV_DIR="./rocm_smi_venv"
                            if [ ! -d "$VENV_DIR" ]; then
                                uv venv "$VENV_DIR"
                            fi
                            source "$VENV_DIR/bin/activate"
                            print_step "Installing in virtual environment..."
                            uv pip install $args
                            ROCM_SMI_VENV_PYTHON="$VENV_DIR/bin/python"
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
                                VENV_DIR="./rocm_smi_venv"
                                if [ ! -d "$VENV_DIR" ]; then
                                    uv venv "$VENV_DIR"
                                fi

                                # Activate venv and install
                                source "$VENV_DIR/bin/activate"
                                print_step "Installing in virtual environment..."
                                uv pip install $args

                                # Store venv path for verification
                                ROCM_SMI_VENV_PYTHON="$VENV_DIR/bin/python"
                                print_success "Installed in virtual environment: $VENV_DIR"
                            elif [ $install_exit_code -eq 0 ]; then
                                print_success "Global installation successful"
                                ROCM_SMI_VENV_PYTHON=""
                            else
                                print_error "Global installation failed with unknown error:"
                                echo "$install_output"
                                print_step "Falling back to virtual environment..."

                                # Create uv venv in project directory
                                VENV_DIR="./rocm_smi_venv"
                                if [ ! -d "$VENV_DIR" ]; then
                                    uv venv "$VENV_DIR"
                                fi

                                # Activate venv and install
                                source "$VENV_DIR/bin/activate"
                                print_step "Installing in virtual environment..."
                                uv pip install $args

                                # Store venv path for verification
                                ROCM_SMI_VENV_PYTHON="$VENV_DIR/bin/python"
                                print_success "Installed in virtual environment: $VENV_DIR"
                            fi
                            ;;
                    esac
                else
                    # Fall back to pip
                    print_step "Installing with pip..."
                    python3 -m pip install $args
                    ROCM_SMI_VENV_PYTHON=""
                fi
            }

            if [[ "$DRY_RUN" == "true" ]]; then
                print_step "[DRY RUN] Would install ROCm SMI Python wrapper"
            else
                uv_pip_install -e .
                if [ $? -ne 0 ]; then
                    print_error "Failed to install ROCm SMI Python wrapper"
                    return 1
                fi
                print_success "Installed ROCm SMI Python wrapper"
            fi
        else
            print_warning "No setup.py found in python_smi_tools. Creating a simple wrapper instead."
        mkdir -p $INSTALL_DIR/python_wrapper
        cat > $INSTALL_DIR/python_wrapper/rocm_smi_lib.py << 'EOF'
#!/usr/bin/env python3
"""
Simple wrapper for ROCm SMI command-line tool
"""

import subprocess
import re
import json

class rsmi:
    @staticmethod
    def rsmi_init(flags=0):
        """Initialize ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_shut_down():
        """Shut down ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_num_monitor_devices():
        """Get number of GPU devices."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            return len(data.keys()), 0
        except:
            return 0, 1

    @staticmethod
    def rsmi_dev_name_get(device_id):
        """Get device name."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            name = data[device_key].get("Card Series", "Unknown")
            return 0, name
        except:
            return 1, "Unknown"

    @staticmethod
    def rsmi_dev_gpu_busy_percent_get(device_id):
        """Get GPU utilization."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            util_str = data[device_key].get("GPU use (%)", "0%")
            util = int(re.sub(r'[^0-9]', '', util_str))
            return 0, util
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_temp_metric_get(device_id, sensor_type, metric_type):
        """Get temperature."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showtemp", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            temp_str = data[device_key].get("Temperature (Sensor edge) (C)", "0.0")
            temp = float(re.sub(r'[^0-9.]', '', temp_str)) * 1000  # Convert to millidegrees
            return 0, int(temp)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_memory_usage_get(device_id, memory_type):
        """Get memory usage."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            used_str = data[device_key].get("GPU Memory Used (B)", "0")
            total_str = data[device_key].get("GPU Memory Total (B)", "0")
            used = int(re.sub(r'[^0-9]', '', used_str))
            total = int(re.sub(r'[^0-9]', '', total_str))
            return 0, used, total
        except:
            return 1, 0, 0

    @staticmethod
    def rsmi_dev_power_ave_get(device_id):
        """Get power consumption."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showpower", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            power_str = data[device_key].get("Average Graphics Package Power (W)", "0.0")
            power = float(re.sub(r'[^0-9.]', '', power_str)) * 1000000  # Convert to microwatts
            return 0, int(power)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_gpu_clk_freq_get(device_id, clock_type):
        """Get clock frequency."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showclocks", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]

            if clock_type == 0:  # GPU clock
                clock_str = data[device_key].get("GPU Clock Level", "0 MHz")
            else:  # Memory clock
                clock_str = data[device_key].get("GPU Memory Clock Level", "0 MHz")

            clock = float(re.sub(r'[^0-9.]', '', clock_str)) * 1000000  # Convert to Hz
            return 0, int(clock)
        except:
            return 1, 0
EOF

        # Create setup.py
        cat > $INSTALL_DIR/python_wrapper/setup.py << 'EOF'
from setuptools import setup

setup(
    name="rocm_smi_lib",
    version="0.1.0",
    py_modules=["rocm_smi_lib"],
    description="Simple wrapper for ROCm SMI command-line tool",
)
EOF

        # Install the wrapper
        cd $INSTALL_DIR/python_wrapper
        if command_exists uv; then
            log "Using uv to install ROCm SMI Python wrapper..."
            if uv pip install -e . --system 2>/dev/null || \
               uv pip install -e . --user 2>/dev/null || \
               uv pip install -e . --break-system-packages 2>/dev/null; then
                log "Successfully installed ROCm SMI Python wrapper with uv"
            else
                log "uv installation failed, falling back to pip..."
                pip install -e . --user 2>/dev/null || pip install -e . --break-system-packages
            fi
        else
            log "Using pip to install ROCm SMI Python wrapper..."
            pip install -e . --user 2>/dev/null || pip install -e . --break-system-packages
        fi
    fi
else
    log "python_smi_tools directory not found. Creating a simple wrapper instead."
    mkdir -p $INSTALL_DIR/python_wrapper
    cat > $INSTALL_DIR/python_wrapper/rocm_smi_lib.py << 'EOF'
#!/usr/bin/env python3
"""
Simple wrapper for ROCm SMI command-line tool
"""

import subprocess
import re
import json

class rsmi:
    @staticmethod
    def rsmi_init(flags=0):
        """Initialize ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_shut_down():
        """Shut down ROCm SMI."""
        return 0

    @staticmethod
    def rsmi_num_monitor_devices():
        """Get number of GPU devices."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            return len(data.keys()), 0
        except:
            return 0, 1

    @staticmethod
    def rsmi_dev_name_get(device_id):
        """Get device name."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showallinfo", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            name = data[device_key].get("Card Series", "Unknown")
            return 0, name
        except:
            return 1, "Unknown"

    @staticmethod
    def rsmi_dev_gpu_busy_percent_get(device_id):
        """Get GPU utilization."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            util_str = data[device_key].get("GPU use (%)", "0%")
            util = int(re.sub(r'[^0-9]', '', util_str))
            return 0, util
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_temp_metric_get(device_id, sensor_type, metric_type):
        """Get temperature."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showtemp", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            temp_str = data[device_key].get("Temperature (Sensor edge) (C)", "0.0")
            temp = float(re.sub(r'[^0-9.]', '', temp_str)) * 1000  # Convert to millidegrees
            return 0, int(temp)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_memory_usage_get(device_id, memory_type):
        """Get memory usage."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            used_str = data[device_key].get("GPU Memory Used (B)", "0")
            total_str = data[device_key].get("GPU Memory Total (B)", "0")
            used = int(re.sub(r'[^0-9]', '', used_str))
            total = int(re.sub(r'[^0-9]', '', total_str))
            return 0, used, total
        except:
            return 1, 0, 0

    @staticmethod
    def rsmi_dev_power_ave_get(device_id):
        """Get power consumption."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showpower", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]
            power_str = data[device_key].get("Average Graphics Package Power (W)", "0.0")
            power = float(re.sub(r'[^0-9.]', '', power_str)) * 1000000  # Convert to microwatts
            return 0, int(power)
        except:
            return 1, 0

    @staticmethod
    def rsmi_dev_gpu_clk_freq_get(device_id, clock_type):
        """Get clock frequency."""
        try:
            output = subprocess.check_output(["rocm-smi", "--showclocks", "--json"], text=True)
            data = json.loads(output)
            device_key = list(data.keys())[device_id]

            if clock_type == 0:  # GPU clock
                clock_str = data[device_key].get("GPU Clock Level", "0 MHz")
            else:  # Memory clock
                clock_str = data[device_key].get("GPU Memory Clock Level", "0 MHz")

            clock = float(re.sub(r'[^0-9.]', '', clock_str)) * 1000000  # Convert to Hz
            return 0, int(clock)
        except:
            return 1, 0
EOF

    # Create setup.py
    cat > $INSTALL_DIR/python_wrapper/setup.py << 'EOF'
from setuptools import setup

setup(
    name="rocm_smi_lib",
    version="0.1.0",
    py_modules=["rocm_smi_lib"],
    description="Simple wrapper for ROCm SMI command-line tool",
)
EOF

    # Install the wrapper
    cd $INSTALL_DIR/python_wrapper
    if command_exists uv; then
        log "Using uv to install ROCm SMI Python wrapper..."
        if uv pip install -e . --system 2>/dev/null || \
           uv pip install -e . --user 2>/dev/null || \
           uv pip install -e . --break-system-packages 2>/dev/null; then
            log "Successfully installed ROCm SMI Python wrapper with uv"
        else
            log "uv installation failed, falling back to pip..."
            pip install -e . --user 2>/dev/null || pip install -e . --break-system-packages
        fi
    else
        log "Using pip to install ROCm SMI Python wrapper..."
        pip install -e . --user 2>/dev/null || pip install -e . --break-system-packages
    fi
fi

    # Verify installation
    print_section "Verifying ROCm SMI Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${ROCM_SMI_VENV_PYTHON:-python3}

    if [[ "$DRY_RUN" == "true" ]]; then
        print_step "[DRY RUN] Would verify ROCm SMI Python wrapper installation"
    else
        print_step "Testing ROCm SMI Python wrapper..."
        if $PYTHON_CMD -c "from rocm_smi_lib import rsmi; print('ROCm SMI Python wrapper installed successfully')" 2>/dev/null; then
            print_success "ROCm SMI Python wrapper installation successful"

            # Test basic functionality
            print_step "Testing basic ROCm SMI functionality..."
            if $PYTHON_CMD -c "from rocm_smi_lib import rsmi; rsmi.rsmi_init(0); count = rsmi.rsmi_num_monitor_devices(); rsmi.rsmi_shut_down(); print(f'Found {count[0]} GPU device(s)')" 2>/dev/null; then
                print_success "ROCm SMI basic functionality working"
            else
                print_warning "ROCm SMI basic functionality test failed, but wrapper is installed"
            fi
        else
            print_error "ROCm SMI Python wrapper installation failed"
            return 1
        fi
    fi

    # Create monitoring script
    print_step "Creating GPU monitoring script..."
    MONITOR_SCRIPT="$INSTALL_DIR/monitor_gpus.py"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_step "[DRY RUN] Would create GPU monitoring script at $MONITOR_SCRIPT"
    else
        cat > $MONITOR_SCRIPT << 'EOF'
#!/usr/bin/env python3
"""
ROCm SMI GPU Monitoring Script

This script provides real-time monitoring of AMD GPUs using the ROCm SMI library.
It displays information such as GPU utilization, temperature, memory usage, and power consumption.
"""

import time
import argparse
import os
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

try:
    from rocm_smi_lib import rsmi
except ImportError:
    print("Error: ROCm SMI Python wrapper not found.")
    print("Please install it with: pip install -e /path/to/rocm_smi_lib/python_smi_tools")
    sys.exit(1)

class GPUMonitor:
    def __init__(self, interval=1.0, log_file=None, plot=False):
        """Initialize the GPU monitor."""
        self.interval = interval
        self.log_file = log_file
        self.plot = plot
        self.history = {
            'timestamp': [],
            'utilization': {},
            'temperature': {},
            'memory': {},
            'power': {}
        }

        # Initialize ROCm SMI
        rsmi.rsmi_init(0)

        # Get number of devices
        self.num_devices = rsmi.rsmi_num_monitor_devices()
        print(f"Found {self.num_devices} GPU device(s)")

        # Initialize history for each device
        for i in range(self.num_devices):
            self.history['utilization'][i] = []
            self.history['temperature'][i] = []
            self.history['memory'][i] = []
            self.history['power'][i] = []

    def __del__(self):
        """Clean up ROCm SMI."""
        rsmi.rsmi_shut_down()

    def get_device_info(self, device_id):
        """Get information for a specific device."""
        info = {}

        try:
            # Get device name
            name = rsmi.rsmi_dev_name_get(device_id)[1]
            info['name'] = name

            # Get GPU utilization
            util = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
            info['utilization'] = util

            # Get temperature
            temp = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C
            info['temperature'] = temp

            # Get memory usage
            mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
            mem_used = mem_info[1] / (1024 * 1024)  # Convert to MB
            mem_total = mem_info[2] / (1024 * 1024)  # Convert to MB
            info['memory_used'] = mem_used
            info['memory_total'] = mem_total
            info['memory_percent'] = (mem_used / mem_total) * 100 if mem_total > 0 else 0

            # Get power consumption
            power = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W
            info['power'] = power

            # Get clock speeds
            gpu_clock = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 0)[1] / 1000000.0  # Convert to GHz
            mem_clock = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 1)[1] / 1000000.0  # Convert to GHz
            info['gpu_clock'] = gpu_clock
            info['memory_clock'] = mem_clock

        except Exception as e:
            print(f"Error getting information for device {device_id}: {e}")

        return info

    def update_history(self, device_id, info):
        """Update history for plotting."""
        if device_id == 0:  # Only add timestamp once
            self.history['timestamp'].append(datetime.now())

        self.history['utilization'][device_id].append(info.get('utilization', 0))
        self.history['temperature'][device_id].append(info.get('temperature', 0))
        self.history['memory'][device_id].append(info.get('memory_percent', 0))
        self.history['power'][device_id].append(info.get('power', 0))

    def plot_history(self):
        """Plot the history of GPU metrics."""
        if not self.plot or len(self.history['timestamp']) < 2:
            return

        # Create figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Convert timestamps to relative seconds
        start_time = self.history['timestamp'][0]
        timestamps = [(t - start_time).total_seconds() for t in self.history['timestamp']]

        # Plot utilization
        for device_id in range(self.num_devices):
            axs[0].plot(timestamps, self.history['utilization'][device_id],
                        label=f"GPU {device_id}")
        axs[0].set_ylabel('Utilization (%)')
        axs[0].set_title('GPU Utilization')
        axs[0].grid(True)
        axs[0].legend()

        # Plot temperature
        for device_id in range(self.num_devices):
            axs[1].plot(timestamps, self.history['temperature'][device_id],
                        label=f"GPU {device_id}")
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title('GPU Temperature')
        axs[1].grid(True)
        axs[1].legend()

        # Plot memory usage
        for device_id in range(self.num_devices):
            axs[2].plot(timestamps, self.history['memory'][device_id],
                        label=f"GPU {device_id}")
        axs[2].set_ylabel('Memory Usage (%)')
        axs[2].set_title('GPU Memory Usage')
        axs[2].grid(True)
        axs[2].legend()

        # Plot power consumption
        for device_id in range(self.num_devices):
            axs[3].plot(timestamps, self.history['power'][device_id],
                        label=f"GPU {device_id}")
        axs[3].set_ylabel('Power (W)')
        axs[3].set_title('GPU Power Consumption')
        axs[3].set_xlabel('Time (s)')
        axs[3].grid(True)
        axs[3].legend()

        plt.tight_layout()
        plt.savefig('gpu_monitoring.png')
        plt.close()

    def log_to_file(self, data):
        """Log data to a file."""
        if not self.log_file:
            return

        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Timestamp: {timestamp}\n")
            for device_id, info in data.items():
                f.write(f"GPU {device_id} ({info.get('name', 'Unknown')}):\n")
                f.write(f"  Utilization: {info.get('utilization', 0):.1f}%\n")
                f.write(f"  Temperature: {info.get('temperature', 0):.1f}°C\n")
                f.write(f"  Memory: {info.get('memory_used', 0):.1f}/{info.get('memory_total', 0):.1f} MB ({info.get('memory_percent', 0):.1f}%)\n")
                f.write(f"  Power: {info.get('power', 0):.1f} W\n")
                f.write(f"  GPU Clock: {info.get('gpu_clock', 0):.2f} GHz\n")
                f.write(f"  Memory Clock: {info.get('memory_clock', 0):.2f} GHz\n")
            f.write("\n")

    def monitor(self, duration=None):
        """Monitor GPUs and display information."""
        try:
            # Initialize log file
            if self.log_file:
                with open(self.log_file, 'w') as f:
                    f.write(f"ROCm SMI GPU Monitoring Log\n")
                    f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            start_time = time.time()
            iteration = 0

            while True:
                # Clear screen
                os.system('clear')

                # Get current time
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"ROCm SMI GPU Monitor - {current_time}")
                print(f"Monitoring interval: {self.interval} seconds")
                if duration:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"Duration: {duration:.1f}s (Remaining: {max(0, remaining):.1f}s)")
                print()

                # Get information for all devices
                data = {}
                table_data = []

                for i in range(self.num_devices):
                    info = self.get_device_info(i)
                    data[i] = info

                    # Update history
                    self.update_history(i, info)

                    # Add to table data
                    table_data.append([
                        i,
                        info.get('name', 'Unknown'),
                        f"{info.get('utilization', 0):.1f}%",
                        f"{info.get('temperature', 0):.1f}°C",
                        f"{info.get('memory_used', 0):.1f}/{info.get('memory_total', 0):.1f} MB ({info.get('memory_percent', 0):.1f}%)",
                        f"{info.get('power', 0):.1f} W",
                        f"{info.get('gpu_clock', 0):.2f} GHz",
                        f"{info.get('memory_clock', 0):.2f} GHz"
                    ])

                # Display table
                headers = ["ID", "Name", "Utilization", "Temperature", "Memory", "Power", "GPU Clock", "Memory Clock"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))

                # Log to file
                self.log_to_file(data)

                # Plot history
                if iteration % 10 == 0:  # Update plot every 10 iterations
                    self.plot_history()

                # Check if duration has elapsed
                if duration and (time.time() - start_time) >= duration:
                    break

                # Wait for next update
                time.sleep(self.interval)
                iteration += 1

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            # Final plot
            self.plot_history()

            if self.log_file:
                print(f"\nLog file saved to: {self.log_file}")

            if self.plot:
                print(f"Plot saved to: gpu_monitoring.png")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Monitor AMD GPUs using ROCm SMI')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Monitoring interval in seconds (default: 1.0)')
    parser.add_argument('-d', '--duration', type=float,
                        help='Monitoring duration in seconds (default: indefinite)')
    parser.add_argument('-l', '--log', type=str,
                        help='Log file path')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Generate plots of GPU metrics')

    args = parser.parse_args()

    # Install tabulate if not available
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate package...")
        os.system('pip install tabulate --break-system-packages')

    # Install matplotlib if plotting is enabled
    if args.plot:
        try:
            import matplotlib
        except ImportError:
            print("Installing matplotlib package...")
            os.system('pip install matplotlib --break-system-packages')

    # Create monitor and start monitoring
    monitor = GPUMonitor(interval=args.interval, log_file=args.log, plot=args.plot)
    monitor.monitor(duration=args.duration)

if __name__ == '__main__':
    main()
EOF

        # Make the script executable
        chmod +x $MONITOR_SCRIPT
        print_success "Created GPU monitoring script at $MONITOR_SCRIPT"

        # Create wrapper script
        print_step "Creating ROCm SMI wrapper example script..."
        WRAPPER_SCRIPT="$INSTALL_DIR/rocm_smi_wrapper.py"

        cat > $WRAPPER_SCRIPT << 'EOF'
#!/usr/bin/env python3
"""
ROCm SMI Python Wrapper Example

This script demonstrates how to use the ROCm SMI Python wrapper to access
GPU information programmatically.
"""

import sys
from rocm_smi_lib import rsmi

def initialize_rsmi():
    """Initialize ROCm SMI."""
    try:
        rsmi.rsmi_init(0)
        return True
    except Exception as e:
        print(f"Error initializing ROCm SMI: {e}")
        return False

def shutdown_rsmi():
    """Shut down ROCm SMI."""
    try:
        rsmi.rsmi_shut_down()
    except Exception as e:
        print(f"Error shutting down ROCm SMI: {e}")

def get_device_count():
    """Get the number of GPU devices."""
    try:
        return rsmi.rsmi_num_monitor_devices()
    except Exception as e:
        print(f"Error getting device count: {e}")
        return 0

def get_device_info(device_id):
    """Get detailed information for a specific device."""
    info = {}

    try:
        # Device name
        info['name'] = rsmi.rsmi_dev_name_get(device_id)[1]

        # Device ID and vendor ID
        info['device_id'] = rsmi.rsmi_dev_id_get(device_id)[1]
        info['vendor_id'] = rsmi.rsmi_dev_vendor_id_get(device_id)[1]

        # PCI info
        info['pci_bandwidth'] = rsmi.rsmi_dev_pci_bandwidth_get(device_id)[1]
        info['pci_address'] = rsmi.rsmi_dev_pci_id_get(device_id)[1]

        # GPU utilization
        info['gpu_utilization'] = rsmi.rsmi_dev_gpu_busy_percent_get(device_id)[1]
        info['memory_utilization'] = rsmi.rsmi_dev_memory_busy_percent_get(device_id)[1]

        # Temperature
        info['temperature'] = rsmi.rsmi_dev_temp_metric_get(device_id, 0, 0)[1] / 1000.0  # Convert to °C

        # Memory info
        mem_info = rsmi.rsmi_dev_memory_usage_get(device_id, 0)
        info['memory_used'] = mem_info[1] / (1024 * 1024)  # Convert to MB
        info['memory_total'] = mem_info[2] / (1024 * 1024)  # Convert to MB

        # Power consumption
        info['power'] = rsmi.rsmi_dev_power_ave_get(device_id)[1] / 1000000.0  # Convert to W

        # Clock speeds
        info['gpu_clock'] = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 0)[1] / 1000000.0  # Convert to GHz
        info['memory_clock'] = rsmi.rsmi_dev_gpu_clk_freq_get(device_id, 1)[1] / 1000000.0  # Convert to GHz

        # Fan speed
        try:
            info['fan_speed'] = rsmi.rsmi_dev_fan_speed_get(device_id, 0)[1]
            info['fan_speed_max'] = rsmi.rsmi_dev_fan_speed_max_get(device_id, 0)[1]
            info['fan_speed_percent'] = (info['fan_speed'] / info['fan_speed_max']) * 100 if info['fan_speed_max'] > 0 else 0
        except:
            info['fan_speed'] = 'N/A'
            info['fan_speed_max'] = 'N/A'
            info['fan_speed_percent'] = 'N/A'

        # Voltage
        try:
            info['voltage'] = rsmi.rsmi_dev_volt_metric_get(device_id, 0)[1] / 1000.0  # Convert to V
        except:
            info['voltage'] = 'N/A'

        # Performance level
        try:
            info['performance_level'] = rsmi.rsmi_dev_perf_level_get(device_id)[1]
        except:
            info['performance_level'] = 'N/A'

        # Overdrive level
        try:
            info['overdrive_level'] = rsmi.rsmi_dev_overdrive_level_get(device_id)[1]
        except:
            info['overdrive_level'] = 'N/A'

        # ECC status
        try:
            info['ecc_status'] = rsmi.rsmi_dev_ecc_status_get(device_id, 0)[1]
        except:
            info['ecc_status'] = 'N/A'

    except Exception as e:
        print(f"Error getting information for device {device_id}: {e}")

    return info

def print_device_info(device_id, info):
    """Print device information in a formatted way."""
    print(f"\n{'=' * 50}")
    print(f"GPU {device_id}: {info.get('name', 'Unknown')}")
    print(f"{'=' * 50}")

    print(f"Device ID: {info.get('device_id', 'N/A')}")
    print(f"Vendor ID: {info.get('vendor_id', 'N/A')}")
    print(f"PCI Address: {info.get('pci_address', 'N/A')}")
    print(f"PCI Bandwidth: {info.get('pci_bandwidth', 'N/A')}")

    print(f"\nUtilization:")
    print(f"  GPU: {info.get('gpu_utilization', 'N/A')}%")
    print(f"  Memory: {info.get('memory_utilization', 'N/A')}%")

    print(f"\nMemory:")
    print(f"  Used: {info.get('memory_used', 'N/A'):.2f} MB")
    print(f"  Total: {info.get('memory_total', 'N/A'):.2f} MB")
    print(f"  Utilization: {(info.get('memory_used', 0) / info.get('memory_total', 1)) * 100:.2f}%")

    print(f"\nTemperature: {info.get('temperature', 'N/A'):.2f}°C")
    print(f"Power: {info.get('power', 'N/A'):.2f} W")

    print(f"\nClock Speeds:")
    print(f"  GPU: {info.get('gpu_clock', 'N/A'):.2f} GHz")
    print(f"  Memory: {info.get('memory_clock', 'N/A'):.2f} GHz")

    print(f"\nFan:")
    if info.get('fan_speed', 'N/A') != 'N/A':
        print(f"  Speed: {info.get('fan_speed', 'N/A')}")
        print(f"  Max Speed: {info.get('fan_speed_max', 'N/A')}")
        print(f"  Percentage: {info.get('fan_speed_percent', 'N/A'):.2f}%")
    else:
        print(f"  Speed: N/A (possibly integrated GPU)")

    print(f"\nVoltage: {info.get('voltage', 'N/A')}")
    print(f"Performance Level: {info.get('performance_level', 'N/A')}")
    print(f"Overdrive Level: {info.get('overdrive_level', 'N/A')}")
    print(f"ECC Status: {info.get('ecc_status', 'N/A')}")

def main():
    """Main function."""
    if not initialize_rsmi():
        sys.exit(1)

    try:
        # Get device count
        device_count = get_device_count()
        print(f"Found {device_count} GPU device(s)")

        # Get information for each device
        for i in range(device_count):
            info = get_device_info(i)
            print_device_info(i, info)

    finally:
        shutdown_rsmi()

if __name__ == '__main__':
    main()
EOF

        # Make the script executable
        chmod +x $WRAPPER_SCRIPT
        print_success "Created ROCm SMI wrapper example at $WRAPPER_SCRIPT"

        # Show completion message
        clear
        cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██████╗  ██████╗  ██████╗███╗   ███╗    ███████╗███╗   ███╗██╗  ║
    ║  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██╔════╝████╗ ████║██║  ║
    ║  ██████╔╝██║   ██║██║     ██╔████╔██║    ███████╗██╔████╔██║██║  ║
    ║  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ╚════██║██║╚██╔╝██║██║  ║
    ║  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ███████║██║ ╚═╝ ██║██║  ║
    ║  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚══════╝╚═╝     ╚═╝╚═╝  ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  ROCm SMI is now ready for GPU monitoring and profiling ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

        print_success "ROCm SMI installation completed successfully"

        # Provide usage examples
        echo
        echo -e "${CYAN}${BOLD}Quick Start Examples:${RESET}"
        if [ -n "$ROCM_SMI_VENV_PYTHON" ]; then
            echo -e "${GREEN}source ./rocm_smi_venv/bin/activate${RESET}"
            echo -e "${GREEN}$ROCM_SMI_VENV_PYTHON $MONITOR_SCRIPT${RESET}"
            echo -e "${GREEN}$ROCM_SMI_VENV_PYTHON $WRAPPER_SCRIPT${RESET}"
        else
            echo -e "${GREEN}python3 $MONITOR_SCRIPT${RESET}"
            echo -e "${GREEN}python3 $WRAPPER_SCRIPT${RESET}"
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
        echo -e "${GREEN}eval \"\$(./install_rocm_smi.sh --show-env)\"${RESET}"
        echo

        # Add a small delay to ensure the message is seen
        echo -e "${GREEN}${BOLD}Returning to main menu in 3 seconds...${RESET}"
        sleep 1
        echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"
        sleep 1

        return 0
    fi
}

# Check for --show-env option
if [[ "$SHOW_ENV" == "true" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_rocm_smi "$@"

