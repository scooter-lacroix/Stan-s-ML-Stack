#!/bin/bash

# ASCII Art Banner
cat << "EOF"
   ███╗   ███╗██████╗ ██╗██╗  ██╗██████╗ ██╗   ██╗
   ████╗ ████║██╔══██╗██║██║  ██║██╔══██╗╚██╗ ██╔╝
   ██╔████╔██║██████╔╝██║███████║██████╔╝ ╚████╔╝
   ██║╚██╔╝██║██╔═══╝ ██║╚════██║██╔═══╝   ╚██╔╝
   ██║ ╚═╝ ██║██║     ██║     ██║██║        ██║
   ╚═╝     ╚═╝╚═╝     ╚═╝     ╚═╝╚═╝        ╚═╝
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

# Function to detect MPI installation path
detect_mpi_path() {
    # Check common MPI installation paths
    local mpi_paths=("/usr/lib64/openmpi" "/opt/openmpi" "/usr/local/openmpi" "/usr/lib/openmpi")

    for path in "${mpi_paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/mpirun" ]; then
            echo "$path"
            return 0
        fi
    done

    # Check if mpirun is in PATH and find its directory
    if command_exists mpirun; then
        local mpirun_path=$(which mpirun)
        local mpi_dir=$(dirname "$mpirun_path")
        mpi_dir=$(dirname "$mpi_dir")  # Go up one level to get the base MPI directory
        if [ -d "$mpi_dir" ]; then
            echo "$mpi_dir"
            return 0
        fi
    fi

    # Default fallback
    echo "/usr/lib64/openmpi"
}

# Function to load configuration file
load_config() {
    local config_file="mpi4py_config.sh"

    if [ -f "$config_file" ]; then
        print_step "Loading configuration from $config_file"
        source "$config_file"
        print_success "Configuration loaded"
    else
        print_step "No configuration file found, using defaults"
    fi
}

# Function to save configuration file
save_config() {
    local config_file="mpi4py_config.sh"

    cat > "$config_file" << EOF
#!/bin/bash
# MPI4Py Installation Configuration
# Generated by install_mpi4py.sh

# Installation method (global, venv, auto)
MPI4PY_INSTALL_METHOD="${INSTALL_METHOD:-auto}"

# Virtual environment directory (if using venv)
MPI4PY_VENV_DIR="${VENV_DIR:-./mpi4py_venv}"

# Force reinstall
MPI4PY_FORCE_INSTALL="${FORCE_INSTALL:-false}"

# Dry run mode
MPI4PY_DRY_RUN="${DRY_RUN:-false}"

# MPI installation path
MPI_PATH="${MPI_PATH:-/usr/lib64/openmpi}"

# ROCm support
ROCM_ENABLED="${ROCM_ENABLED:-true}"
EOF

    print_success "Configuration saved to $config_file"
}

# Function to detect WSL environment
detect_wsl() {
    if [ -f "/proc/version" ] && grep -q "Microsoft" "/proc/version"; then
        return 0
    elif [ -f "/proc/version" ] && grep -q "microsoft" "/proc/version"; then
        return 0
    else
        return 1
    fi
}

# Function to detect container environment
detect_container() {
    if [ -f "/.dockerenv" ]; then
        return 0
    elif grep -q "docker\|containerd\|podman" "/proc/1/cgroup" 2>/dev/null; then
        return 0
    elif [ -n "$CONTAINER" ]; then
        return 0
    else
        return 1
    fi
}

# Function to show system information
show_system_info() {
    print_header "System Information"

    echo "Operating System: $(uname -s)"
    echo "Kernel Version: $(uname -r)"
    echo "Architecture: $(uname -m)"

    if detect_wsl; then
        echo "Environment: WSL (Windows Subsystem for Linux)"
    elif detect_container; then
        echo "Environment: Container"
    else
        echo "Environment: Native Linux"
    fi

    echo "Package Manager: $(detect_package_manager)"
    echo "Python Version: $(python3 --version 2>&1)"

    if command_exists uv; then
        echo "UV Version: $(uv --version)"
    else
        echo "UV: Not installed"
    fi

    if command_exists mpirun; then
        echo "MPI: Installed ($(which mpirun))"
    else
        echo "MPI: Not installed"
    fi

    if command_exists rocminfo; then
        echo "ROCm: Installed"
    else
        echo "ROCm: Not installed"
    fi
}

# Function to check if a Python module exists
python_module_exists() {
    python3 -c "import $1" 2>/dev/null
    return $?
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

# Function to show MPI environment variables
show_mpi_env() {
    # Detect MPI installation path
    MPI_PATH=$(detect_mpi_path)

    PATH="$MPI_PATH/bin:$PATH"
    LD_LIBRARY_PATH="$MPI_PATH/lib:$LD_LIBRARY_PATH"

    echo "export MPI_PATH=\"$MPI_PATH\""
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""

    # Add ROCm variables if ROCm is available
    if command_exists rocminfo; then
        echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0"
        echo "export PYTORCH_ROCM_ARCH=\"gfx1100\""
        echo "export ROCM_PATH=\"/opt/rocm\""
        echo "export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\""
    fi
}

# Function to retry command with backoff
retry_command() {
    local cmd="$1"
    local max_attempts="${2:-3}"
    local delay="${3:-2}"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        print_step "Attempt $attempt/$max_attempts: $cmd"
        if eval "$cmd"; then
            return 0
        else
            print_warning "Command failed (attempt $attempt/$max_attempts)"
            if [ $attempt -lt $max_attempts ]; then
                print_step "Retrying in $delay seconds..."
                sleep $delay
                delay=$((delay * 2))  # Exponential backoff
            fi
        fi
        attempt=$((attempt + 1))
    done

    print_error "Command failed after $max_attempts attempts"
    return 1
}

# Function to detect and setup ROCm environment
setup_rocm_env() {
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
        return 0
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
                # Recursively call setup_rocm_env now that rocminfo is installed
                setup_rocm_env
                return $?
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_warning "ROCm is not installed. MPI will work without GPU acceleration."
            return 1
        fi
    fi
}

# Function to check if MPI is installed
check_mpi() {
    if command -v mpirun >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to install system MPI packages
install_system_mpi() {
    print_step "Installing system MPI packages..."
    
    # Detect package manager and install appropriate MPI packages
    if command -v dnf >/dev/null 2>&1; then
        print_step "Using dnf to install OpenMPI..."
        if sudo dnf install -y openmpi openmpi-devel environment-modules; then
            print_success "System MPI packages installed with dnf"
            # Load the MPI module
            if [ -f /etc/profile.d/modules.sh ]; then
                source /etc/profile.d/modules.sh
                module load mpi/openmpi-x86_64 2>/dev/null || true
            fi
            # Add MPI to PATH
            export PATH="/usr/lib64/openmpi/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH"
            return 0
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        print_step "Using apt-get to install OpenMPI..."
        if sudo apt-get update && sudo apt-get install -y libopenmpi-dev openmpi-bin; then
            print_success "System MPI packages installed with apt-get"
            return 0
        fi
    elif command -v yum >/dev/null 2>&1; then
        print_step "Using yum to install OpenMPI..."
        if sudo yum install -y openmpi openmpi-devel environment-modules; then
            print_success "System MPI packages installed with yum"
            # Load the MPI module
            if [ -f /etc/profile.d/modules.sh ]; then
                source /etc/profile.d/modules.sh
                module load mpi/openmpi-x86_64 2>/dev/null || true
            fi
            # Add MPI to PATH
            export PATH="/usr/lib64/openmpi/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH"
            return 0
        fi
    elif command -v zypper >/dev/null 2>&1; then
        print_step "Using zypper to install OpenMPI..."
        if sudo zypper install -y openmpi openmpi-devel; then
            print_success "System MPI packages installed with zypper"
            return 0
        fi
    elif command -v pacman >/dev/null 2>&1; then
        print_step "Using pacman to install OpenMPI..."
        if sudo pacman -S --noconfirm openmpi; then
            print_success "System MPI packages installed with pacman"
            return 0
        fi
    else
        print_error "Unknown package manager. Cannot auto-install MPI."
        return 1
    fi
    
    print_error "Failed to install system MPI packages"
    return 1
}

# Function to install mpi4py
install_mpi4py() {
    print_header "MPI4Py Installation"

    # Load configuration
    load_config

    # Check for --force flag
    FORCE_INSTALL=false
    if [[ "$*" == *"--force"* ]]; then
        FORCE_INSTALL=true
        print_step "Force reinstall requested"
    fi

    # Check for --dry-run flag
    DRY_RUN=false
    if [[ "$*" == *"--dry-run"* ]]; then
        DRY_RUN=true
        print_warning "DRY RUN MODE - No actual installation will be performed"
    fi

    # Check for --save-config flag
    SAVE_CONFIG_FLAG=false
    if [[ "$*" == *"--save-config"* ]]; then
        SAVE_CONFIG_FLAG=true
    fi

    # Check if mpi4py is already installed
    if package_installed "mpi4py" && [ "$FORCE_INSTALL" = false ]; then
        mpi4py_version=$(python3 -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null)
        print_success "mpi4py is already installed (version: $mpi4py_version)"
        print_step "Use --force to reinstall anyway"
        return 0
    fi

    # Setup ROCm environment if available
    setup_rocm_env

    # Check if MPI is installed
    print_section "Checking MPI Installation"

    if ! check_mpi; then
        print_warning "MPI is not installed. Attempting to install system MPI packages..."

        if [ "$DRY_RUN" = true ]; then
            print_step "[DRY RUN] Would install system MPI packages"
        else
            if install_system_mpi; then
                print_success "System MPI packages installed successfully"

                # Re-check if MPI is now available
                if ! check_mpi; then
                    print_error "MPI installation failed or mpirun is not in PATH"
                    print_step "Try adding MPI to your PATH manually or install it with:"
                    print_step "On Ubuntu/Debian: sudo apt-get install libopenmpi-dev openmpi-bin"
                    print_step "On CentOS/RHEL/Fedora: sudo dnf install openmpi openmpi-devel"
                    return 1
                fi
            else
                print_error "Failed to install system MPI packages automatically"
                print_step "Please install MPI manually:"
                print_step "On Ubuntu/Debian: sudo apt-get install libopenmpi-dev openmpi-bin"
                print_step "On CentOS/RHEL/Fedora: sudo dnf install openmpi openmpi-devel"
                return 1
            fi
        fi
    else
        print_success "MPI is installed"
    fi

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}MPI4Py Installation Options:${RESET}"
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
    print_section "Installing MPI4Py"

    if ! command_exists uv; then
        if [ "$DRY_RUN" = false ]; then
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

    # Create a function to handle uv commands properly with venv fallback
    uv_pip_install() {
        local args="$@"

        if [ "$DRY_RUN" = true ]; then
            print_step "[DRY RUN] Would install mpi4py with: $args"
            return 0
        fi

        # Check if uv is available as a command
        if command_exists uv; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    python3 -m pip install --break-system-packages $args
                    MPI4PY_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    VENV_DIR="./mpi4py_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args
                    MPI4PY_VENV_PYTHON="$VENV_DIR/bin/python"
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
                        VENV_DIR="./mpi4py_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        MPI4PY_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global installation successful"
                        MPI4PY_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        VENV_DIR="./mpi4py_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        MPI4PY_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing with pip..."
            python3 -m pip install $args
            MPI4PY_VENV_PYTHON=""
        fi
    }

    # Install mpi4py
    print_step "Installing mpi4py..."
    uv_pip_install mpi4py

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${MPI4PY_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import mpi4py" &>/dev/null; then
        mpi4py_version=$($PYTHON_CMD -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null)
        print_success "mpi4py is installed (version: $mpi4py_version)"

        # Test basic MPI functionality
        print_step "Testing MPI functionality..."
        if $PYTHON_CMD -c "import mpi4py.MPI as MPI; print('MPI initialized successfully')" 2>/dev/null; then
            print_success "MPI functionality working correctly"
        else
            print_warning "MPI functionality test failed - this may be normal in some environments"
        fi
    else
        print_error "mpi4py installation failed"
        return 1
    fi

    # Show completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ███╗   ███╗██████╗ ██╗██╗  ██╗██████╗ ██╗   ██╗       ║
    ║  ████╗ ████║██╔══██╗██║██║  ██║██╔══██╗╚██╗ ██╔╝       ║
    ║  ██╔████╔██║██████╔╝██║███████║██████╔╝ ╚████╔╝        ║
    ║  ██║╚██╔╝██║██╔═══╝ ██║██╔══██║██╔═══╝   ╚██╔╝         ║
    ║  ██║ ╚═╝ ██║██║     ██║██║  ██║██║        ██║          ║
    ║  ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝          ║
    ║                                                         ║
    ║  Installation Completed Successfully!                  ║
    ║                                                         ║
    ║  MPI4Py is now ready to use with your MPI installation. ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "MPI4Py installation completed successfully"

    # Provide usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    if [ -n "$MPI4PY_VENV_PYTHON" ]; then
        echo -e "${GREEN}source ./mpi4py_venv/bin/activate${RESET}"
        echo -e "${GREEN}python -c \"import mpi4py.MPI as MPI; print('MPI4Py version:', mpi4py.__version__); print('MPI initialized:', MPI.Is_initialized())\"${RESET}"
    else
        echo -e "${GREEN}python3 -c \"import mpi4py.MPI as MPI; print('MPI4Py version:', mpi4py.__version__); print('MPI initialized:', MPI.Is_initialized())\"${RESET}"
    fi
    echo
    echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}MPI environment variables are set for this session.${RESET}"
    echo -e "${YELLOW}For future sessions, you may need to run:${RESET}"

    # Output the actual environment variables that were set
    show_mpi_env
    echo
    echo -e "${CYAN}${BOLD}To apply these settings to your current shell, run:${RESET}"
    echo -e "${GREEN}eval \"\$(./install_mpi4py.sh --show-env)\"${RESET}"
    echo

    # Save configuration if requested
    if [ "$SAVE_CONFIG_FLAG" = true ]; then
        save_config
    fi

    return 0
}

# Main function
main() {
    install_mpi4py "$@"
    return $?
}

# Check for command line options
case "$1" in
    --show-env)
        show_mpi_env
        exit 0
        ;;
    --system-info)
        show_system_info
        exit 0
        ;;
    --help|-h)
        echo "MPI4Py Installation Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --force         Force reinstallation even if already installed"
        echo "  --dry-run       Show what would be done without making changes"
        echo "  --save-config   Save current configuration to file"
        echo "  --show-env      Show MPI environment variables"
        echo "  --system-info   Show system information"
        echo "  --help, -h      Show this help message"
        echo
        exit 0
        ;;
    *)
        # Run main function with all script arguments
        main "$@"
        exit $?
        ;;
esac
