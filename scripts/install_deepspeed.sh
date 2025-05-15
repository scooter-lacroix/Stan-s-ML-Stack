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

# Color definitions
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

# Progress bar variables
PROGRESS_BAR_WIDTH=50
PROGRESS_CURRENT=0
PROGRESS_TOTAL=100
PROGRESS_CHAR="▓"
PROGRESS_EMPTY="░"
PROGRESS_ANIMATION=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
ANIMATION_INDEX=0

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

# Main installation function
install_deepspeed() {
    print_header "DeepSpeed Installation"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking DeepSpeed installation..."

    # Check if DeepSpeed is already installed
    if package_installed "deepspeed"; then
        deepspeed_version=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
        print_warning "DeepSpeed is already installed (version: $deepspeed_version)"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping DeepSpeed installation"
            complete_progress_bar
            return 0
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

    # Check if uv is installed
    update_progress_bar 10
    draw_progress_bar "Checking package manager..."
    print_section "Installing DeepSpeed"

    if ! command_exists uv; then
        print_step "Installing uv package manager..."
        update_progress_bar 5
        draw_progress_bar "Installing uv package manager..."
        python3 -m pip install uv

        if ! command_exists uv; then
            print_error "Failed to install uv package manager"
            print_step "Falling back to pip"
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Install required dependencies first
    update_progress_bar 15
    draw_progress_bar "Installing required dependencies..."
    print_step "Installing required dependencies first..."

    if command_exists uv; then
        # Install dependencies with uv
        uv pip install packaging ninja pydantic jsonschema
    else
        # Install dependencies with pip
        python3 -m pip install packaging ninja pydantic jsonschema
    fi

    # Install DeepSpeed
    update_progress_bar 20
    draw_progress_bar "Installing DeepSpeed..."
    print_step "Installing DeepSpeed..."

    if command_exists uv; then
        # Install with uv - try multiple approaches
        set +e  # Don't exit on error
        uv pip install deepspeed
        install_result=$?

        if [ $install_result -ne 0 ]; then
            print_warning "First installation attempt failed, trying with --no-deps..."
            uv pip install deepspeed --no-deps
            install_result=$?
        fi

        if [ $install_result -ne 0 ]; then
            print_warning "Second installation attempt failed, trying with --force-reinstall..."
            uv pip install deepspeed --force-reinstall
            install_result=$?
        fi
        set -e  # Return to normal error handling
    else
        # Install with pip - try multiple approaches
        set +e  # Don't exit on error
        python3 -m pip install deepspeed
        install_result=$?

        if [ $install_result -ne 0 ]; then
            print_warning "First installation attempt failed, trying with --no-deps..."
            python3 -m pip install deepspeed --no-deps
            install_result=$?
        fi

        if [ $install_result -ne 0 ]; then
            print_warning "Second installation attempt failed, trying with --force-reinstall..."
            python3 -m pip install deepspeed --force-reinstall
            install_result=$?
        fi
        set -e  # Return to normal error handling
    fi

    if [ $install_result -ne 0 ]; then
        print_error "Failed to install DeepSpeed after multiple attempts"
        complete_progress_bar
        return 1
    fi

    # Verify installation
    update_progress_bar 30
    draw_progress_bar "Verifying installation..."
    print_section "Verifying Installation"

    # Use timeout to prevent hanging during verification
    set +e  # Don't exit on error
    if timeout 30s python3 -c "import deepspeed; print('Success')" &>/dev/null; then
        deepspeed_version=$(timeout 10s python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
        print_success "DeepSpeed is installed (version: $deepspeed_version)"

        # Check if DeepSpeed can detect GPUs
        update_progress_bar 10
        draw_progress_bar "Checking GPU detection..."
        print_step "Checking GPU detection..."

        # Check for all required dependencies
        print_step "Verifying all dependencies are installed..."
        for dep in packaging ninja pydantic jsonschema; do
            if python3 -c "import ${dep//-/_}" &>/dev/null; then
                print_success "$dep is installed"
            else
                print_warning "$dep is not installed, attempting to install it now"
                if command_exists uv; then
                    uv pip install $dep
                else
                    python3 -m pip install $dep
                fi
            fi
        done

        # Check GPU access with timeout to prevent hanging
        if timeout 20s python3 -c "import deepspeed; import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "DeepSpeed can access GPUs through PyTorch"

            # Get GPU count
            gpu_count=$(timeout 10s python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            print_step "PyTorch detected $gpu_count GPU(s)"
        else
            print_warning "DeepSpeed cannot access GPUs through PyTorch"
            print_warning "This may be normal for ROCm/HIP environments"
            print_warning "DeepSpeed should still work for CPU operations"
        fi

        # Consider installation successful even if GPU detection fails
        verification_success=0
    else
        print_error "DeepSpeed installation verification failed"
        print_warning "Attempting one more installation approach..."

        # Try reinstalling with different options
        if command_exists uv; then
            uv pip install --force-reinstall deepspeed
        else
            python3 -m pip install --force-reinstall deepspeed
        fi

        # Check again
        if timeout 30s python3 -c "import deepspeed; print('Success')" &>/dev/null; then
            print_success "DeepSpeed installed successfully after retry"
            verification_success=0
        else
            print_error "DeepSpeed installation failed after multiple attempts"
            verification_success=1
        fi
    fi
    set -e  # Return to normal error handling

    if [ $verification_success -ne 0 ]; then
        complete_progress_bar
        return 1
    fi

    update_progress_bar 10
    draw_progress_bar "Completing installation..."
    print_success "DeepSpeed installation completed successfully"

    complete_progress_bar
    return 0
}

# Run the installation function
install_deepspeed
