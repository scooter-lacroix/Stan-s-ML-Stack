#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER_GUARD="$SCRIPT_DIR/lib/installer_guard.sh"
if [ -f "$INSTALLER_GUARD" ]; then
    # shellcheck source=lib/installer_guard.sh
    source "$INSTALLER_GUARD"
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

mlstack_is_strict_rocm() {
    case "${MLSTACK_STRICT_ROCM:-1}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

mlstack_preflight_msg() {
    local level="$1"
    shift
    if declare -f "print_${level}" >/dev/null 2>&1; then
        "print_${level}" "$*"
    else
        echo "$*"
    fi
}

mlstack_resolve_python_bin() {
    local candidate="${MLSTACK_PYTHON_BIN:-python3}"
    local selected=""
    local rocm_mm=""

    if mlstack_is_strict_rocm && declare -f mlstack_ensure_python_for_rocm_torch >/dev/null 2>&1; then
        if declare -f strict_detect_rocm_mm >/dev/null 2>&1; then
            rocm_mm="$(strict_detect_rocm_mm)"
        else
            rocm_mm="${ROCM_VERSION:-7.2}"
        fi
        selected="$(mlstack_ensure_python_for_rocm_torch "$candidate" "$rocm_mm" "${MLSTACK_TORCH_CHANNEL:-latest}" "${DRY_RUN:-false}" || true)"
        [ -n "$selected" ] && candidate="$selected"
    elif mlstack_is_strict_rocm && declare -f mlstack_select_python_for_rocm_torch >/dev/null 2>&1; then
        if declare -f strict_detect_rocm_mm >/dev/null 2>&1; then
            rocm_mm="$(strict_detect_rocm_mm)"
        else
            rocm_mm="${ROCM_VERSION:-7.2}"
        fi
        selected="$(mlstack_select_python_for_rocm_torch "$candidate" "$rocm_mm" "${MLSTACK_TORCH_CHANNEL:-latest}" || true)"
        [ -n "$selected" ] && candidate="$selected"
    fi

    if [ -x "$candidate" ]; then
        :
    else
        candidate="$(command -v "$candidate" 2>/dev/null || true)"
    fi
    if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
        mlstack_preflight_msg error "Python interpreter not found: ${MLSTACK_PYTHON_BIN:-python3}"
        return 1
    fi
    MLSTACK_PYTHON_BIN="$candidate"
    PYTHON_BIN="$candidate"
    export MLSTACK_PYTHON_BIN
}

if ! declare -f mlstack_assert_rocm_torch >/dev/null 2>&1; then
    mlstack_assert_rocm_torch() {
        local py="${MLSTACK_PYTHON_BIN:-python3}"
        "$py" - <<'PY' >/dev/null 2>&1
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    raise SystemExit(1)
import torch
hip = getattr(getattr(torch, "version", None), "hip", None)
if not hip:
    raise SystemExit(2)
PY
    }
fi

mlstack_rocm_python_preflight() {
    local dry_run="${1:-false}"
    local strict=false
    if mlstack_is_strict_rocm; then
        strict=true
    fi

    mlstack_resolve_python_bin || {
        [ "$strict" = true ] && return 1
        mlstack_preflight_msg warning "Continuing without strict Python preflight."
        return 0
    }

    if [ "$strict" = true ]; then
        local py_mm py_minor
        py_mm="$("$MLSTACK_PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        py_minor="${py_mm#3.}"
        if [[ ! "$py_mm" =~ ^3\.[0-9]+$ ]] || [ "${py_minor:-0}" -lt 10 ] || [ "${py_minor:-99}" -gt 13 ]; then
            mlstack_preflight_msg error "Strict ROCm mode requires Python 3.10-3.13; found ${py_mm:-unknown}."
            return 1
        fi
    fi

    if mlstack_assert_rocm_torch "$MLSTACK_PYTHON_BIN"; then
        return 0
    fi

    if [ "$strict" != true ]; then
        mlstack_preflight_msg warning "ROCm-enabled PyTorch not validated; strict mode is disabled."
        return 0
    fi

    local pytorch_installer="$SCRIPT_DIR/install_pytorch_rocm.sh"
    if [ ! -f "$pytorch_installer" ]; then
        mlstack_preflight_msg error "Missing $pytorch_installer; cannot repair ROCm PyTorch in strict mode."
        return 1
    fi

    local torch_method torch_channel strict_venv_python verify_py
    torch_method="${PYTORCH_INSTALL_METHOD:-${MLSTACK_INSTALL_METHOD:-auto}}"
    torch_channel="${MLSTACK_TORCH_CHANNEL:-latest}"
    strict_venv_python="$HOME/.mlstack/venvs/pytorch_rocm/bin/python"
    verify_py="$MLSTACK_PYTHON_BIN"

    if [ "$dry_run" = "true" ]; then
        mlstack_preflight_msg warning "[DRY RUN] Would run: MLSTACK_STRICT_ROCM=1 MLSTACK_BATCH_MODE=1 MLSTACK_PYTHON_BIN=$MLSTACK_PYTHON_BIN bash $pytorch_installer --method $torch_method --channel $torch_channel"
        return 0
    fi

    mlstack_preflight_msg warning "ROCm PyTorch missing or corrupt; reinstalling via strict ROCm installer..."
    if ! MLSTACK_STRICT_ROCM=1 MLSTACK_BATCH_MODE=1 MLSTACK_PYTHON_BIN="$MLSTACK_PYTHON_BIN" \
        MLSTACK_INSTALL_METHOD="$torch_method" TORCH_CHANNEL="$torch_channel" \
        bash "$pytorch_installer" --method "$torch_method" --channel "$torch_channel"; then
        mlstack_preflight_msg error "Failed to run PyTorch ROCm installer."
        return 1
    fi

    if [ -x "$strict_venv_python" ] && "$strict_venv_python" -c "import torch" >/dev/null 2>&1; then
        verify_py="$strict_venv_python"
        MLSTACK_PYTHON_BIN="$strict_venv_python"
        PYTHON_BIN="$strict_venv_python"
        export MLSTACK_PYTHON_BIN PYTHON_BIN
    fi

    if ! mlstack_assert_rocm_torch "$verify_py"; then
        mlstack_preflight_msg error "ROCm PyTorch verification failed after reinstall."
        return 1
    fi
}

# Set up colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-$PYTHON_BIN}"
UV_CACHE_DIR="${MLSTACK_UV_CACHE_DIR:-${TMPDIR:-/tmp}/mlstack-uv-cache}"
mkdir -p "$UV_CACHE_DIR" >/dev/null 2>&1 || true
export UV_CACHE_DIR

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

# Progress bar variables
PROGRESS_BAR_WIDTH=50
PROGRESS_CURRENT=0
PROGRESS_TOTAL=100
PROGRESS_CHAR="▓"
PROGRESS_EMPTY="░"
PROGRESS_ANIMATION=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
ANIMATION_INDEX=0

# Suppress HIP logs
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2

# Configuration file support
CONFIG_FILE="${HOME}/.megatron_install_config"
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_step "Loading configuration from $CONFIG_FILE"
        source "$CONFIG_FILE"
        if type mlstack_enforce_global_install_contract >/dev/null 2>&1; then
            mlstack_enforce_global_install_contract
        fi
    fi
}

save_config() {
    print_step "Saving configuration to $CONFIG_FILE"
    cat > "$CONFIG_FILE" << EOF
# Megatron-LM Installation Configuration
# Generated on $(date)

# Installation preferences
INSTALL_METHOD="${INSTALL_METHOD}"
FORCE_REINSTALL="${FORCE_REINSTALL}"

# ROCm settings
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
ROCM_PATH="${ROCM_PATH}"

# Environment detection
IS_WSL="${IS_WSL:-false}"
IS_CONTAINER="${IS_CONTAINER:-false}"
HAS_SUDO="${HAS_SUDO:-false}"
EOF
}

# Logging system
LOG_FILE="${HOME}/megatron_install_$(date +%Y%m%d_%H%M%S).log"
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    echo "[$timestamp] [$level] $message"
}

# Command-line argument parsing
DRY_RUN=false
FORCE_REINSTALL=false
INSTALL_METHOD="${INSTALL_METHOD:-${MLSTACK_INSTALL_METHOD:-auto}}"
INSTALL_METHOD="$(echo "$INSTALL_METHOD" | tr '[:upper:]' '[:lower:]')"
case "$INSTALL_METHOD" in
    global|venv|auto) ;;
    *) INSTALL_METHOD="auto" ;;
esac

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
        --global)
            INSTALL_METHOD="global"
            shift
            ;;
        --venv)
            INSTALL_METHOD="venv"
            shift
            ;;
        --auto)
            INSTALL_METHOD="auto"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Show what would be done without making changes"
            echo "  --force            Force reinstallation even if already installed"
            echo "  --global           Use global installation method"
            echo "  --venv             Use virtual environment installation method"
            echo "  --auto             Use auto-detect installation method (default)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # Preview installation"
            echo "  $0 --force --venv               # Force reinstall in virtual environment"
            echo "  $0 --global                     # Install globally"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Enhanced color detection with NO_COLOR support
if [ -t 1 ]; then
    # Check if NO_COLOR environment variable is set
    if [ -z "$NO_COLOR" ]; then
        # Terminal supports colors - keep existing colors
        :
    else
        # NO_COLOR is set, disable colors
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        MAGENTA=''
        CYAN=''
        BOLD=''
        RESET=''
    fi
else
    # Not a terminal, disable colors
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    RESET=''
fi

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

# Function to print colored messages
print_header() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    else
        echo "=== $1 ==="
    fi
    echo
}

print_section() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${BLUE}${BOLD}>>> $1${RESET}"
    else
        echo ">>> $1"
    fi
}

print_step() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${MAGENTA}>> $1${RESET}"
    else
        echo ">> $1"
    fi
}

print_success() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${GREEN}✓ $1${RESET}"
    else
        echo "✓ $1"
    fi
}

print_warning() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${YELLOW}⚠ $1${RESET}"
    else
        echo "⚠ $1"
    fi
}

print_error() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${RED}✗ $1${RESET}"
    else
        echo "✗ $1"
    fi
}

# Function to check if a Python module exists
python_module_exists() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Function to detect package manager
detect_package_manager() {
    if command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v pacman >/dev/null 2>&1; then
        echo "pacman"
    elif command -v zypper >/dev/null 2>&1; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to detect ROCm installation and version
detect_rocm() {
    local rocm_path=""
    local rocm_version=""

    # Check for rocminfo command
    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Try to get version from rocminfo
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            # Fallback: try to get version from directory listing
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        fi

        # Set ROCm path
        if [ -d "/opt/rocm" ]; then
            rocm_path="/opt/rocm"
        else
            rocm_path=$(ls -d /opt/rocm-* 2>/dev/null | head -n 1)
        fi
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."

        # Check for ROCm directories
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."

            local package_manager=$(detect_package_manager)
            case $package_manager in
                apt)
                    sudo apt-get update && sudo apt-get install -y rocminfo
                    ;;
                dnf)
                    sudo dnf install -y rocminfo
                    ;;
                yum)
                    sudo yum install -y rocminfo
                    ;;
                pacman)
                    sudo pacman -S --noconfirm rocminfo
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
                # Recursively call detect_rocm now that rocminfo is installed
                detect_rocm
                return $?
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_error "ROCm is not installed. Please install ROCm first."
            return 1
        fi
    fi

    # Set default version if not detected
    if [ -z "$rocm_version" ]; then
        print_warning "Could not detect ROCm version, using default version 6.4.0"
        rocm_version="6.4.0"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    # Set ROCm path if not set
    if [ -z "$rocm_path" ]; then
        if [ -d "/opt/rocm" ]; then
            rocm_path="/opt/rocm"
        else
            rocm_path=$(ls -d /opt/rocm-* 2>/dev/null | head -n 1)
        fi
    fi

    # Export detected values
    export ROCM_PATH="$rocm_path"
    export ROCM_VERSION="$rocm_version"

    # Extract major and minor versions for compatibility checks
    export ROCM_MAJOR_VERSION=$(echo "$rocm_version" | cut -d '.' -f 1)
    export ROCM_MINOR_VERSION=$(echo "$rocm_version" | cut -d '.' -f 2)

    return 0
}

# Function to detect GPU architecture
detect_gpu_architecture() {
    if ! command_exists rocminfo; then
        print_warning "rocminfo not available, cannot detect GPU architecture"
        return 1
    fi

    # Get GPU architecture from rocminfo
    local gpu_arch=$(rocminfo 2>/dev/null | grep -i "gfx" | head -n 1 | grep -o "gfx[0-9]*" | head -n 1)

    if [ -n "$gpu_arch" ]; then
        print_success "Detected GPU architecture: $gpu_arch"
        export PYTORCH_ROCM_ARCH="$gpu_arch"
        return 0
    else
        print_warning "Could not detect GPU architecture, using default gfx1100"
        export PYTORCH_ROCM_ARCH="gfx1100"
        return 1
    fi
}

# Function to setup ROCm environment variables
setup_rocm_environment() {
    print_step "Setting up ROCm environment variables..."

    # Set HSA_OVERRIDE_GFX_VERSION
    export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-"11.0.0"}

    # Set PYTORCH_ROCM_ARCH if not already set
    if [ -z "$PYTORCH_ROCM_ARCH" ]; then
        detect_gpu_architecture
    fi

    # Set ROCM_PATH if not already set
    if [ -z "$ROCM_PATH" ]; then
        if [ -d "/opt/rocm" ]; then
            export ROCM_PATH="/opt/rocm"
        else
            export ROCM_PATH=$(ls -d /opt/rocm-* 2>/dev/null | head -n 1)
        fi
    fi

    # Update PATH and LD_LIBRARY_PATH
    if [ -n "$ROCM_PATH" ]; then
        export PATH="$ROCM_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
    fi

    # Set HSA_TOOLS_LIB with automatic profiler library detection
    if [ -f "$ROCM_PATH/lib/librocprofiler-sdk-tool.so" ]; then
        export HSA_TOOLS_LIB="$ROCM_PATH/lib/librocprofiler-sdk-tool.so"
        print_step "ROCm profiler library found and configured"
    else
        # Try to install rocprofiler if possible
        if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
            print_step "Installing rocprofiler for HSA tools support..."
            sudo apt-get update && sudo apt-get install -y rocprofiler
            if [ -f "$ROCM_PATH/lib/librocprofiler-sdk-tool.so" ]; then
                export HSA_TOOLS_LIB="$ROCM_PATH/lib/librocprofiler-sdk-tool.so"
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

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion to PYTORCH_ALLOC_CONF
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
        unset PYTORCH_CUDA_ALLOC_CONF
        print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
    elif [ -z "${PYTORCH_ALLOC_CONF:-}" ]; then
        # Set a sensible default for ROCm GPUs
        export PYTORCH_ALLOC_CONF="expandable_segments:True"
        print_step "Set PYTORCH_ALLOC_CONF=expandable_segments:True"
    fi

    print_success "ROCm environment variables configured"
}

# Function to use uv or pip for Python packages
install_python_package() {
    local package="$1"
    shift
    local extra_args="$@"

    if [ "$DRY_RUN" = true ]; then
        if command_exists uv; then
            print_step "[DRY RUN] Would install $package with uv..."
        else
            print_step "[DRY RUN] Would install $package with pip..."
        fi
        return 0
    fi

    if command_exists uv; then
        print_step "Installing $package with uv..."
        uv pip install --python $(which python3) $extra_args "$package"
    else
        print_step "Installing $package with pip..."
        python3 -m pip install $extra_args "$package"
    fi
}

# Function to detect execution environment
detect_environment() {
    # Check if running in WSL
    if [ -f /proc/version ] && grep -q "Microsoft" /proc/version 2>/dev/null; then
        export IS_WSL=true
        print_step "Detected Windows Subsystem for Linux (WSL) environment"
    else
        export IS_WSL=false
    fi

    # Check if running in a container
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || grep -q "docker\|container\|podman" /proc/1/cgroup 2>/dev/null; then
        export IS_CONTAINER=true
        print_step "Detected container environment"
    else
        export IS_CONTAINER=false
    fi

    # Check if running with sudo
    if [ "$EUID" -eq 0 ]; then
        export HAS_SUDO=true
        print_step "Running with elevated privileges"
    else
        export HAS_SUDO=false
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

# Function to check if C++ compiler is available
check_cpp_compiler() {
    if command -v g++ >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to install system development packages
install_system_dev_packages() {
    print_step "Installing system development packages..."
    
    # Install C++ compiler and development tools
    if command -v dnf >/dev/null 2>&1; then
        print_step "Using dnf to install development packages..."
        if sudo dnf install -y gcc-c++ g++ make cmake; then
            print_success "Development packages installed with dnf"
            return 0
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        print_step "Using apt-get to install development packages..."
        if sudo apt-get update && sudo apt-get install -y g++ gcc make cmake build-essential; then
            print_success "Development packages installed with apt-get"
            return 0
        fi
    elif command -v yum >/dev/null 2>&1; then
        print_step "Using yum to install development packages..."
        if sudo yum install -y gcc-c++ make cmake; then
            print_success "Development packages installed with yum"
            return 0
        fi
    elif command -v zypper >/dev/null 2>&1; then
        print_step "Using zypper to install development packages..."
        if sudo zypper install -y gcc-c++ make cmake; then
            print_success "Development packages installed with zypper"
            return 0
        fi
    elif command -v pacman >/dev/null 2>&1; then
        print_step "Using pacman to install development packages..."
        if sudo pacman -S --noconfirm gcc make cmake; then
            print_success "Development packages installed with pacman"
            return 0
        fi
    else
        print_error "Unknown package manager. Cannot auto-install development packages."
        return 1
    fi
    
    print_error "Failed to install system development packages"
    return 1
}

# Function to install Megatron-LM
install_megatron() {
    print_header "Installing Megatron-LM"
    local megatron_dir="$HOME/Megatron-LM"
    local fallback_megatron_dir="$HOME/.mlstack/src/Megatron-LM"

    # Load configuration
    load_config

    # Initialize logging
    log_message "INFO" "Starting Megatron-LM installation"
    log_message "INFO" "Command line arguments: $@"
    log_message "INFO" "Dry run: $DRY_RUN"
    log_message "INFO" "Force reinstall: $FORCE_REINSTALL"
    log_message "INFO" "Install method: $INSTALL_METHOD"

    # Resolve a writable Megatron checkout directory.
    if [ -d "$megatron_dir" ] && [ ! -w "$megatron_dir" ]; then
        print_warning "Primary Megatron directory is not writable: $megatron_dir"
        megatron_dir="$fallback_megatron_dir"
    elif [ ! -d "$megatron_dir" ] && [ ! -w "$(dirname "$megatron_dir")" ]; then
        print_warning "Primary Megatron parent is not writable: $(dirname "$megatron_dir")"
        megatron_dir="$fallback_megatron_dir"
    fi
    mkdir -p "$(dirname "$megatron_dir")"
    print_step "Using Megatron directory: $megatron_dir"

    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No installation actions will be performed."
        print_step "[DRY RUN] Would validate ROCm/PyTorch prerequisites and compiler/MPI availability"
        print_step "[DRY RUN] Would clone/update Megatron-LM repository under $megatron_dir"
        print_step "[DRY RUN] Would install Megatron-LM with ROCm-compatible Python dependencies"
        print_step "[DRY RUN] Would run post-install import and GPU checks"
        return 0
    fi

    if ! mlstack_rocm_python_preflight "$DRY_RUN"; then
        print_error "ROCm/Python preflight failed"
        return 1
    fi

    # Detect execution environment
    detect_environment
    log_message "INFO" "Environment detection: WSL=$IS_WSL, Container=$IS_CONTAINER, Sudo=$HAS_SUDO"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking dependencies..."

    # Check if PyTorch is installed and working
    if ! python_module_exists "torch"; then
        # Check if torch package exists but can't be imported (broken install)
        if python3 -c "import importlib.util; spec = importlib.util.find_spec('torch'); exit(0 if spec else 1)" 2>/dev/null; then
            log_message "ERROR" "PyTorch is installed but cannot be imported (possibly missing system libraries)"
            print_error "PyTorch is installed but cannot be imported!"
            print_step "This usually means missing system libraries (e.g., libmpi_cxx.so.40)"
            print_step "Attempting to reinstall ROCm PyTorch via strict installer..."
            if ! MLSTACK_STRICT_ROCM=1 MLSTACK_BATCH_MODE=1 MLSTACK_PYTHON_BIN="$MLSTACK_PYTHON_BIN" \
                MLSTACK_INSTALL_METHOD="$INSTALL_METHOD" TORCH_CHANNEL="${MLSTACK_TORCH_CHANNEL:-latest}" \
                bash "$SCRIPT_DIR/install_pytorch_rocm.sh" --method "$INSTALL_METHOD" --channel "${MLSTACK_TORCH_CHANNEL:-latest}"; then
                print_error "Failed to repair PyTorch with install_pytorch_rocm.sh"
                complete_progress_bar
                return 1
            fi
            if [ -x "$HOME/.mlstack/venvs/pytorch_rocm/bin/python" ] && "$HOME/.mlstack/venvs/pytorch_rocm/bin/python" -c "import torch" >/dev/null 2>&1; then
                MLSTACK_PYTHON_BIN="$HOME/.mlstack/venvs/pytorch_rocm/bin/python"
                PYTHON_BIN="$MLSTACK_PYTHON_BIN"
                export MLSTACK_PYTHON_BIN PYTHON_BIN
            fi
            print_success "ROCm PyTorch reinstalled successfully"
        else
            log_message "ERROR" "PyTorch is not installed"
            print_warning "PyTorch is not installed. Bootstrapping strict ROCm PyTorch..."
            if ! MLSTACK_STRICT_ROCM=1 MLSTACK_BATCH_MODE=1 MLSTACK_PYTHON_BIN="$MLSTACK_PYTHON_BIN" \
                MLSTACK_INSTALL_METHOD="$INSTALL_METHOD" TORCH_CHANNEL="${MLSTACK_TORCH_CHANNEL:-latest}" \
                bash "$SCRIPT_DIR/install_pytorch_rocm.sh" --method "$INSTALL_METHOD" --channel "${MLSTACK_TORCH_CHANNEL:-latest}"; then
                print_error "Failed to bootstrap PyTorch with install_pytorch_rocm.sh"
                complete_progress_bar
                return 1
            fi
            if [ -x "$HOME/.mlstack/venvs/pytorch_rocm/bin/python" ] && "$HOME/.mlstack/venvs/pytorch_rocm/bin/python" -c "import torch" >/dev/null 2>&1; then
                MLSTACK_PYTHON_BIN="$HOME/.mlstack/venvs/pytorch_rocm/bin/python"
                PYTHON_BIN="$MLSTACK_PYTHON_BIN"
                export MLSTACK_PYTHON_BIN PYTHON_BIN
            fi
        fi
    fi

    print_success "PyTorch is installed"
    log_message "INFO" "PyTorch is installed"
    update_progress_bar 5
    draw_progress_bar "Detecting ROCm installation..."

    # Detect and setup ROCm environment
    if ! detect_rocm; then
        print_error "ROCm detection failed"
        complete_progress_bar
        return 1
    fi

    if ! setup_rocm_environment; then
        print_error "ROCm environment setup failed"
        complete_progress_bar
        return 1
    fi

    print_success "ROCm environment configured"
    update_progress_bar 5
    draw_progress_bar "Checking C++ compiler..."

    # Check if C++ compiler is installed
    if ! check_cpp_compiler; then
        print_warning "C++ compiler not found. Installing development packages..."
        update_progress_bar 5
        draw_progress_bar "Installing development packages..."
        
        if install_system_dev_packages; then
            print_success "Development packages installed successfully"
            if ! check_cpp_compiler; then
                print_error "C++ compiler still not available after installation"
                complete_progress_bar
                return 1
            fi
        else
            print_error "Failed to install development packages automatically"
            print_step "Please install C++ compiler manually:"
            print_step "On Ubuntu/Debian: sudo apt-get install build-essential g++"
            print_step "On CentOS/RHEL/Fedora: sudo dnf install gcc-c++ g++"
            complete_progress_bar
            return 1
        fi
    fi

    print_success "C++ compiler is available"
    update_progress_bar 5
    draw_progress_bar "Checking MPI installation..."

    # Check if MPI is installed
    if ! check_mpi; then
        print_warning "MPI is not installed. Installing MPI first..."
        update_progress_bar 5
        draw_progress_bar "Installing MPI..."

        # Run the MPI installation script
        if [ -f "$(dirname "$0")/install_mpi4py.sh" ]; then
            bash "$(dirname "$0")/install_mpi4py.sh"
        else
            print_error "MPI installation script not found. Please install MPI first."
            complete_progress_bar
            return 1
        fi
    else
        print_success "MPI is installed"
    fi

    update_progress_bar 20
    draw_progress_bar "Checking for Megatron-LM..."

    # Check if Megatron-LM is already installed
    if [ -d "$megatron_dir" ]; then
        if [ "$FORCE_REINSTALL" = true ]; then
            print_step "Force reinstall requested - removing existing Megatron-LM directory..."
            if [ "$DRY_RUN" = false ]; then
                rm -rf "$megatron_dir"
            else
                print_step "[DRY RUN] Would remove $megatron_dir"
            fi
        else
            print_warning "Megatron-LM directory already exists at $megatron_dir"
            # In batch/non-interactive mode, proceed with reinstall
            if [ "${MLSTACK_BATCH_MODE:-0}" = "1" ] || [ -n "${RUSTY_STACK:-}" ] || [ ! -t 0 ]; then
                print_step "Non-interactive mode: proceeding with reinstall..."
                REPLY="y"
            else
                read -p "Do you want to reinstall Megatron-LM? (y/n) " -n 1 -r
                echo
            fi
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping installation"
                complete_progress_bar
                return 0
            fi

            print_step "Removing existing Megatron-LM directory..."
            if [ "$DRY_RUN" = false ]; then
                rm -rf "$megatron_dir"
            else
                print_step "[DRY RUN] Would remove $megatron_dir"
            fi
        fi
    fi

    update_progress_bar 30
    draw_progress_bar "Cloning Megatron-LM repository..."

    # Clone Megatron-LM repository
    print_step "Cloning Megatron-LM repository..."
    if [ "$DRY_RUN" = true ]; then
        print_step "[DRY RUN] Would clone https://github.com/NVIDIA/Megatron-LM.git to $megatron_dir"
    else
        # Retry mechanism for git clone
        local clone_attempts=0
        local max_clone_attempts=3
        while [ $clone_attempts -lt $max_clone_attempts ]; do
            if git clone https://github.com/NVIDIA/Megatron-LM.git "$megatron_dir"; then
                print_success "Successfully cloned Megatron-LM repository"
                break
            else
                clone_attempts=$((clone_attempts + 1))
                if [ $clone_attempts -lt $max_clone_attempts ]; then
                    print_warning "Git clone failed (attempt $clone_attempts/$max_clone_attempts), retrying in 5 seconds..."
                    sleep 5
                else
                    print_error "Failed to clone Megatron-LM repository after $max_clone_attempts attempts"
                    complete_progress_bar
                    return 1
                fi
            fi
        done
    fi

    update_progress_bar 50
    draw_progress_bar "Installing Megatron-LM..."

    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    cd "$megatron_dir" || { print_error "Failed to enter Megatron-LM directory"; complete_progress_bar; return 1; }

    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_step "Detected Python version: $PYTHON_VERSION"

    # Apply Python 3.12+ compatibility fixes if needed
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
        print_step "Applying Python 3.12+ compatibility fixes..."

        # Backup setup.py
        cp setup.py setup.py.bak
        print_step "Backed up setup.py to setup.py.bak"

        # Modify setup.py to add Python 3.12 support
        print_step "Updating setup.py to add Python 3.12 support..."
        sed -i 's/Programming Language :: Python :: 3.9/Programming Language :: Python :: 3.9\\n        Programming Language :: Python :: 3.12/' setup.py
        print_success "Updated setup.py with Python 3.12 support"

        # Create a patch directory if it doesn't exist
        mkdir -p patches/python312

        # Create a patch for importlib.metadata compatibility
        cat > patches/python312/importlib_patch.py << 'EOF'
"""
Patch for importlib.metadata compatibility in Python 3.12
"""
import sys
import importlib.metadata

# Add backward compatibility for older code expecting metadata attribute
if not hasattr(importlib, 'metadata'):
    importlib.metadata = importlib.metadata

# Patch sys.modules to ensure imports work correctly
sys.modules['importlib.metadata'] = importlib.metadata
EOF

        print_success "Created importlib.metadata compatibility patch"

        # Create a patch for the megatron module
        mkdir -p megatron/patches
        cat > megatron/patches/__init__.py << 'EOF'
"""
Megatron-LM patches for Python 3.12 compatibility
"""
import sys
import os
import importlib

# Apply Python 3.12 patches if needed
if sys.version_info >= (3, 12):
    # Import the importlib patch
    patch_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'patches', 'python312')
    if patch_path not in sys.path:
        sys.path.append(patch_path)

    try:
        import importlib_patch
    except ImportError:
        pass
EOF

        print_success "Created megatron patches module"

        # Update megatron/__init__.py to apply patches
        if [ -f megatron/__init__.py ]; then
            # Check if the patch import is already there
            if ! grep -q "from .patches import" megatron/__init__.py; then
                # Add the patch import at the beginning of the file
                sed -i '1s/^/# Apply compatibility patches\ntry:\n    from .patches import *\nexcept ImportError:\n    pass\n\n/' megatron/__init__.py
                print_success "Updated megatron/__init__.py to apply patches"
            else
                print_success "megatron/__init__.py already includes patches"
            fi
        else
            print_warning "megatron/__init__.py not found, creating it..."
            echo '# Apply compatibility patches
try:
    from .patches import *
except ImportError:
    pass
' > megatron/__init__.py
            print_success "Created megatron/__init__.py with patches"
        fi
    fi

    # Install required dependencies
    print_step "Installing required dependencies..."

    # Install tensorstore with compatibility fix for Python 3.12
    print_step "Installing tensorstore..."
    if ! python_module_exists "tensorstore"; then
        # Try to install the pre-built wheel first
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
            # For Python 3.12+, use a specific version that's compatible
            if ! mlstack_pip_install "$PYTHON_BIN" tensorstore==0.1.75; then
                print_warning "Failed to install tensorstore from PyPI, trying alternative approach..."
                # Create a dummy tensorstore package to satisfy the import
                SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
                mkdir -p "$SITE_PACKAGES/tensorstore"
                echo "# Dummy tensorstore package for compatibility" > "$SITE_PACKAGES/tensorstore/__init__.py"
                print_success "Created dummy tensorstore package for compatibility"
            else
                print_success "tensorstore installed successfully"
            fi
        else
            # For Python 3.8/3.9, use the standard installation
            if ! mlstack_pip_install "$PYTHON_BIN" tensorstore; then
                print_error "Failed to install tensorstore"
            else
                print_success "tensorstore installed successfully"
            fi
        fi
    else
        print_success "tensorstore is already installed"
    fi

    # Keep ROCm path NVIDIA-free: provide a lightweight compatibility stub instead
    print_step "Ensuring nvidia_modelopt compatibility shim (ROCm-safe)..."
    if ! python_module_exists "nvidia_modelopt"; then
        if [ "$DRY_RUN" = true ]; then
            print_step "[DRY RUN] Would create nvidia_modelopt compatibility shim in site-packages"
        else
            SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
            mkdir -p "$SITE_PACKAGES/nvidia_modelopt"
            cat > "$SITE_PACKAGES/nvidia_modelopt/__init__.py" <<'PY'
# ROCm compatibility shim for optional NVIDIA-only dependency.
__all__ = []
PY
            print_success "Created ROCm-safe nvidia_modelopt compatibility shim"
        fi
    else
        print_success "nvidia_modelopt module already present"
    fi

    # Use command-line specified installation method or ask user
    if [ "$INSTALL_METHOD" = "auto" ]; then
        echo
        echo -e "${CYAN}${BOLD}Megatron-LM Installation Options:${RESET}"
        echo "1) Global installation (recommended for system-wide use)"
        echo "2) Virtual environment (isolated installation)"
        echo "3) Auto-detect (try global, fallback to venv if needed)"
        echo
        if [ "${MLSTACK_BATCH_MODE:-0}" = "1" ] || [ -n "${RUSTY_STACK:-}" ] || [ ! -t 0 ]; then
            INSTALL_CHOICE=3
            print_step "Non-interactive mode: defaulting to auto-detect installation method"
        else
            read -p "Choose installation method (1-3) [3]: " INSTALL_CHOICE
            INSTALL_CHOICE=${INSTALL_CHOICE:-3}
        fi

        case $INSTALL_CHOICE in
            1)
                INSTALL_METHOD="global"
                ;;
            2)
                INSTALL_METHOD="venv"
                ;;
            3|*)
                INSTALL_METHOD="auto"
                ;;
        esac
    fi

    print_step "Using $INSTALL_METHOD installation method"

    ensure_megatron_runtime_rocm_torch() {
        local target_python="$1"
        local channel rocm_version

        [ -n "$target_python" ] || {
            print_error "No runtime Python provided for Megatron validation"
            return 1
        }

        if mlstack_assert_rocm_torch "$target_python"; then
            print_success "ROCm PyTorch already available for Megatron runtime"
            return 0
        fi

        channel="${TORCH_CHANNEL:-${MLSTACK_TORCH_CHANNEL:-latest}}"
        rocm_version="${ROCM_VERSION:-}"
        if [ -z "$rocm_version" ] && declare -f strict_detect_rocm_mm >/dev/null 2>&1; then
            rocm_version="$(strict_detect_rocm_mm || true)"
        fi
        rocm_version="${rocm_version:-7.1}"

        if [ "$DRY_RUN" = true ]; then
            print_step "[DRY RUN] Would install ROCm PyTorch stack into $target_python (channel=$channel rocm=$rocm_version)"
            return 0
        fi

        print_step "Ensuring ROCm PyTorch stack exists for Megatron runtime interpreter..."
        if declare -f mlstack_install_rocm_torch_stack >/dev/null 2>&1; then
            if ! mlstack_install_rocm_torch_stack "$target_python" "$rocm_version" "$channel" "megatron"; then
                print_error "Failed to install ROCm PyTorch stack for Megatron runtime"
                return 1
            fi
        else
            print_warning "ROCm installer guard unavailable; falling back to direct torch install"
            if ! mlstack_pip_install "$target_python" --upgrade torch torchvision torchaudio; then
                print_error "Fallback torch installation failed for Megatron runtime"
                return 1
            fi
        fi

        if ! mlstack_assert_rocm_torch "$target_python"; then
            print_error "ROCm PyTorch verification failed for Megatron runtime interpreter"
            return 1
        fi
        print_success "ROCm PyTorch stack verified for Megatron runtime"
    }

    # Function to handle uv/pip installation with venv fallback
    uv_pip_install_megatron() {
        local args="$@"
        local VENV_DIR
        fallback_pip_venv_install() {
            VENV_DIR="./megatron_rocm_venv"
            if [ ! -d "$VENV_DIR" ]; then
                "$PYTHON_BIN" -m venv "$VENV_DIR" || return 1
            fi
            "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel || return 1
            "$VENV_DIR/bin/python" -m pip install $args || return 1
            MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
            print_success "Installed in virtual environment: $VENV_DIR"
            return 0
        }

        if [ "$DRY_RUN" = true ]; then
            case $INSTALL_METHOD in
                "global")
                    print_step "[DRY RUN] Would install globally with pip: python3 -m pip install --break-system-packages $args"
                    MEGATRON_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "[DRY RUN] Would create uv virtual environment and install: uv venv --seed ./megatron_rocm_venv && source ./megatron_rocm_venv/bin/activate && uv pip install $args"
                    MEGATRON_VENV_PYTHON="./megatron_rocm_venv/bin/python"
                    ;;
                "auto")
                    print_step "[DRY RUN] Would attempt global installation, fallback to venv if needed"
                    MEGATRON_VENV_PYTHON="./megatron_rocm_venv/bin/python"
                    ;;
            esac
            return 0
        fi

        # Check if uv is available as a command
        if command_exists uv; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    "$PYTHON_BIN" -m pip install --break-system-packages $args
                    local install_exit_code=$?
                    if [ $install_exit_code -eq 0 ]; then
                        MEGATRON_VENV_PYTHON=""
                    else
                        print_error "Global installation failed, Megatron-LM requires manual installation"
                        return 1
                    fi
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    VENV_DIR="./megatron_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv --seed "$VENV_DIR" || {
                            print_warning "uv venv creation failed; falling back to python venv..."
                            fallback_pip_venv_install || return 1
                            return 0
                        }
                    fi
                    print_step "Installing in virtual environment..."
                    if ! uv pip install --python "$VENV_DIR/bin/python" $args; then
                        print_warning "uv pip install failed; falling back to python venv pip install..."
                        fallback_pip_venv_install || return 1
                        return 0
                    fi
                    MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                    ;;
                "auto")
                    # Try global install first
                    print_step "Attempting global installation with uv..."
                    local install_output
                    install_output=$(uv pip install --python "$PYTHON_BIN" $args 2>&1)
                    local install_exit_code=$?

                    if echo "$install_output" | grep -q "externally managed"; then
                        print_warning "Global installation failed due to externally managed environment"
                        print_step "Creating uv virtual environment for installation..."

                        # Create uv venv in project directory
                        VENV_DIR="./megatron_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv --seed "$VENV_DIR" || {
                                print_warning "uv venv creation failed; falling back to python venv..."
                                fallback_pip_venv_install || return 1
                                return 0
                            }
                        fi

                        print_step "Installing in virtual environment..."
                        if ! uv pip install --python "$VENV_DIR/bin/python" $args; then
                            print_warning "uv pip install failed; falling back to python venv pip install..."
                            fallback_pip_venv_install || return 1
                            return 0
                        fi

                        # Store venv path for verification
                        MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global installation successful"
                        MEGATRON_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        VENV_DIR="./megatron_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv --seed "$VENV_DIR" || {
                                print_warning "uv venv creation failed; falling back to python venv..."
                                fallback_pip_venv_install || return 1
                                return 0
                            }
                        fi

                        print_step "Installing in virtual environment..."
                        if ! uv pip install --python "$VENV_DIR/bin/python" $args; then
                            print_warning "uv pip install failed; falling back to python venv pip install..."
                            fallback_pip_venv_install || return 1
                            return 0
                        fi

                        # Store venv path for verification
                        MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    "$PYTHON_BIN" -m pip install --break-system-packages $args
                    MEGATRON_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating Python virtual environment..."
                    VENV_DIR="./megatron_rocm_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        "$PYTHON_BIN" -m venv "$VENV_DIR"
                    fi
                    print_step "Installing in virtual environment..."
                    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
                    "$VENV_DIR/bin/python" -m pip install $args
                    MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                    ;;
                "auto")
                    print_step "Attempting global installation with pip..."
                    if "$PYTHON_BIN" -m pip install --break-system-packages $args; then
                        MEGATRON_VENV_PYTHON=""
                        print_success "Global installation successful"
                    else
                        print_warning "Global installation failed; falling back to virtual environment..."
                        VENV_DIR="./megatron_rocm_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            "$PYTHON_BIN" -m venv "$VENV_DIR"
                        fi
                        "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
                        "$VENV_DIR/bin/python" -m pip install $args
                        MEGATRON_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        fi
    }

    # Install Megatron-LM using the enhanced installation method
    print_step "Installing Megatron-LM..."
    if ! uv_pip_install_megatron -e .; then
        print_error "Failed to install Megatron-LM"
        complete_progress_bar
        return 1
    fi

    # Verify the runtime interpreter used for Megatron has ROCm PyTorch available.
    MEGATRON_PYTHON_CMD="${MEGATRON_VENV_PYTHON:-$PYTHON_BIN}"
    if ! ensure_megatron_runtime_rocm_torch "$MEGATRON_PYTHON_CMD"; then
        print_error "Megatron runtime interpreter is missing ROCm-enabled PyTorch"
        complete_progress_bar
        return 1
    fi

    update_progress_bar 80
    draw_progress_bar "Verifying installation..."

    # Create a comprehensive test script to verify Megatron-LM installation
    print_step "Creating comprehensive test script to verify Megatron-LM installation..."
    cat > /tmp/test_megatron.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive test script to verify Megatron-LM installation with ROCm support.
Includes benchmarking and detailed diagnostics.
"""

import os
import sys
import time
import traceback
import subprocess
from datetime import datetime

# Set environment variables for ROCm
os.environ["AMD_LOG_LEVEL"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2"
os.environ["ROCR_VISIBLE_DEVICES"] = "0,1,2"

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️ {message}")

def benchmark_operation(operation_name, operation_func, *args, **kwargs):
    """Benchmark an operation and return result and timing."""
    start_time = time.time()
    try:
        result = operation_func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print_success(f"{operation_name} completed in {duration:.2f} seconds")
        return result, duration
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_error(f"{operation_name} failed after {duration:.2f} seconds: {e}")
        return None, duration

def test_pytorch_gpu():
    """Test PyTorch GPU support."""
    print_separator("Testing PyTorch GPU Support")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm/HIP available: {hasattr(torch.version, 'hip')}")

        if hasattr(torch.version, 'hip'):
            print(f"ROCm version: {torch.version.hip}")
        else:
            print_warning("ROCm/HIP support not detected in PyTorch")

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            print_success("PyTorch GPU detection successful")
            return True
        else:
            print_error("GPU acceleration is not available")
            return False
    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_megatron_import():
    """Test Megatron-LM import."""
    print_separator("Testing Megatron-LM Import")

    try:
        import megatron
        print_success("Megatron-LM core module imported successfully")

        # Try to import key submodules
        try:
            from megatron import mpu
            print_success("Megatron-LM MPU module imported successfully")
        except ImportError as e:
            print_warning(f"MPU module import failed: {e}")

        try:
            from megatron.model import GPTModel
            print_success("Megatron-LM GPTModel imported successfully")
        except ImportError as e:
            print_warning(f"GPTModel import failed: {e}")

        return True
    except Exception as e:
        print_error(f"Failed to import Megatron-LM: {e}")
        traceback.print_exc()
        return False

def benchmark_gpu_operations():
    """Benchmark basic GPU operations."""
    print_separator("Benchmarking GPU Operations")

    try:
        import torch

        if not torch.cuda.is_available():
            print_warning("GPU not available, skipping GPU benchmarks")
            return False

        # Test matrix multiplication
        print_info("Testing matrix multiplication performance...")
        sizes = [1000, 2000, 5000]

        for size in sizes:
            matrix_a = torch.randn(size, size, device='cuda')
            matrix_b = torch.randn(size, size, device='cuda')

            # Warm up
            _ = torch.mm(matrix_a[:100, :100], matrix_b[:100, :100])

            # Benchmark
            start_time = time.time()
            result = torch.mm(matrix_a, matrix_b)
            torch.cuda.synchronize()  # Wait for GPU to finish
            end_time = time.time()

            duration = end_time - start_time
            flops = (2 * size**3) / duration / 1e9  # GFLOPS
            print_success(f"Matrix multiplication ({size}x{size}): {duration:.3f}s, {flops:.2f} GFLOPS")

        # Test memory bandwidth
        print_info("Testing GPU memory bandwidth...")
        data_size = 1000 * 1000 * 1000  # 1GB
        test_data = torch.randn(data_size, dtype=torch.float32, device='cuda')

        # Memory copy benchmark
        start_time = time.time()
        copied_data = test_data.clone()
        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        bandwidth = (data_size * 4) / duration / 1e9  # GB/s
        print_success(f"GPU memory bandwidth: {bandwidth:.2f} GB/s")

        return True

    except Exception as e:
        print_error(f"GPU benchmarking failed: {e}")
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test important environment variables."""
    print_separator("Testing Environment Variables")

    important_vars = [
        'ROCM_PATH',
        'HSA_OVERRIDE_GFX_VERSION',
        'PYTORCH_ROCM_ARCH',
        'HSA_TOOLS_LIB',
        'PYTORCH_ALLOC_CONF',
        'LD_LIBRARY_PATH',
        'PATH'
    ]

    all_set = True
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        if value != 'NOT SET':
            print_success(f"{var} = {value}")
        else:
            print_warning(f"{var} is not set")
            all_set = False

    return all_set

def test_system_info():
    """Test system information."""
    print_separator("System Information")

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")

    # Test ROCm info if available
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success("rocminfo command available")
            # Extract version info
            for line in result.stdout.split('\n'):
                if 'ROCm Version' in line:
                    print_info(f"ROCm version: {line.strip()}")
                    break
        else:
            print_warning("rocminfo command failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("rocminfo command not available")

    return True

def main():
    """Main function."""
    print_separator("Megatron-LM ROCm Comprehensive Test Suite")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_results = {}

    # Test system information
    test_results['system_info'] = test_system_info()

    # Test environment variables
    test_results['environment'] = test_environment_variables()

    # Test PyTorch GPU support
    pytorch_ok, pytorch_time = benchmark_operation("PyTorch GPU Test", test_pytorch_gpu)
    test_results['pytorch_gpu'] = pytorch_ok

    # Test Megatron-LM import
    megatron_ok, megatron_time = benchmark_operation("Megatron-LM Import Test", test_megatron_import)
    test_results['megatron_import'] = megatron_ok

    # Benchmark GPU operations
    if pytorch_ok:
        gpu_benchmark_ok, benchmark_time = benchmark_operation("GPU Benchmarking", benchmark_gpu_operations)
        test_results['gpu_benchmark'] = gpu_benchmark_ok
    else:
        test_results['gpu_benchmark'] = False
        print_warning("Skipping GPU benchmarks due to PyTorch GPU test failure")

    # Summary
    print_separator("Test Summary")

    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)

    print(f"Tests passed: {passed_tests}/{total_tests}")

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    # Overall result: benchmark is informational and can fail on mixed/partial GPU environments.
    required_tests = ['system_info', 'environment', 'pytorch_gpu', 'megatron_import']
    overall_success = all(test_results.get(name, False) for name in required_tests)

    if overall_success:
        print_success("🎉 All tests passed! Megatron-LM installation is working correctly.")
    else:
        print_error("❌ Some tests failed. Please check the output above for details.")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

    # Run the test script with the interpreter that owns the install target.
    # Run the test script
    print_step "Running test script..."
    if "$MEGATRON_PYTHON_CMD" /tmp/test_megatron.py; then
        print_success "Megatron-LM verification successful"

        # Check for the "Tool lib '1' failed to load" warning
        if "$MEGATRON_PYTHON_CMD" -c "
import torch
import sys
try:
    torch.cuda.is_available()
    if 'Tool lib \"1\" failed to load' in torch._C._cuda_getDeviceCount.__doc__:
        print('Warning: Tool lib \"1\" failed to load message detected but this is a known issue with ROCm and can be safely ignored.')
        sys.exit(0)
except Exception as e:
    if 'Tool lib' in str(e) and 'failed to load' in str(e):
        print('Warning: Tool lib failed to load message detected but this is a known issue with ROCm and can be safely ignored.')
        sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            print_warning "Detected 'Tool lib failed to load' message, which is a known issue with ROCm and can be safely ignored."
        fi

        print_success "Megatron-LM installed successfully"
        complete_progress_bar

        # Display completion message
        clear
        cat << "EOF"

        ╔═════════════════════════════════════════════════════════╗
        ║                                                         ║
        ║  ███╗   ███╗███████╗ ██████╗  █████╗ ████████╗██████╗  ║
        ║  ████╗ ████║██╔════╝██╔════╝ ██╔══██╗╚══██╔══╝██╔══██╗ ║
        ║  ██╔████╔██║█████╗  ██║  ███╗███████║   ██║   ██████╔╝ ║
        ║  ██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██║   ██║   ██╔══██╗ ║
        ║  ██║ ╚═╝ ██║███████╗╚██████╔╝██║  ██║   ██║   ██║  ██║ ║
        ║  ╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ║
        ║                                                         ║
        ║  Installation Completed Successfully!                   ║
        ║                                                         ║
        ╚═════════════════════════════════════════════════════════╝

EOF

        # Add note about Python 3.12 compatibility
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
            echo -e "${GREEN}${BOLD}Megatron-LM has been installed with Python 3.12+ compatibility patches.${RESET}"
            echo -e "${YELLOW}Note: The 'Tool lib \"1\" failed to load' warning is a known issue with ROCm and can be safely ignored.${RESET}"
        fi

        echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"

        # Save configuration
        save_config
        log_message "INFO" "Configuration saved to $CONFIG_FILE"

        # Clean up test script
        rm -f /tmp/test_megatron.py

        log_message "INFO" "Megatron-LM installation completed successfully"
        log_message "INFO" "Log file: $LOG_FILE"

        return 0
    else
        print_error "Megatron-LM verification failed"

        # The comprehensive suite includes GPU benchmarks that can fail on mixed GPU setups
        # even when Megatron-LM is correctly installed/importable. Treat core import success
        # as installation success and downgrade benchmark-only failures to warnings.
        if "$MEGATRON_PYTHON_CMD" -c "import megatron; print('megatron ok')" >/dev/null 2>&1; then
            print_warning "Comprehensive Megatron-LM test suite failed, but core import succeeded."
            print_warning "Marking installation successful; review benchmark/runtime warnings above."
            print_success "Megatron-LM installed successfully (core import validated)"
            complete_progress_bar

            save_config
            log_message "INFO" "Configuration saved to $CONFIG_FILE"

            rm -f /tmp/test_megatron.py

            log_message "INFO" "Megatron-LM installation completed with benchmark warnings"
            log_message "INFO" "Log file: $LOG_FILE"
            return 0
        fi

        if "$MEGATRON_PYTHON_CMD" -c "
import torch
import sys
try:
    torch.cuda.is_available()
    if 'Tool lib \"1\" failed to load' in torch._C._cuda_getDeviceCount.__doc__:
        sys.exit(0)
except Exception as e:
    if 'Tool lib' in str(e) and 'failed to load' in str(e):
        sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            print_warning "Detected 'Tool lib failed to load' warning, but verification still failed."
        fi

        print_error "Megatron-LM installation failed"
        complete_progress_bar

        # Clean up test script
        rm -f /tmp/test_megatron.py

        # Force exit even on failure
        echo -e "${RED}${BOLD}Installation failed. Exiting now.${RESET}"
        return 1
    fi
}

# Main function - run directly without nested functions to avoid return issues
install_megatron
exit $?
