#!/bin/bash

# Set up colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

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

# Function to check if MPI is installed
check_mpi() {
    if command -v mpirun >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to install Megatron-LM
install_megatron() {
    print_header "Installing Megatron-LM"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking dependencies..."

    # Check if PyTorch is installed
    if ! python_module_exists "torch"; then
        print_error "PyTorch is not installed. Please install PyTorch first."
        print_step "Run the install_pytorch_rocm.sh script to install PyTorch."
        complete_progress_bar
        return 1
    fi

    print_success "PyTorch is installed"
    update_progress_bar 10
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
    if [ -d "$HOME/Megatron-LM" ]; then
        print_warning "Megatron-LM directory already exists at $HOME/Megatron-LM"
        read -p "Do you want to reinstall Megatron-LM? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping installation"
            complete_progress_bar
            return 0
        fi

        print_step "Removing existing Megatron-LM directory..."
        rm -rf "$HOME/Megatron-LM"
    fi

    update_progress_bar 30
    draw_progress_bar "Cloning Megatron-LM repository..."

    # Clone Megatron-LM repository
    print_step "Cloning Megatron-LM repository..."
    if ! git clone https://github.com/NVIDIA/Megatron-LM.git "$HOME/Megatron-LM"; then
        print_error "Failed to clone Megatron-LM repository"
        complete_progress_bar
        return 1
    fi

    update_progress_bar 50
    draw_progress_bar "Installing Megatron-LM..."

    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    cd "$HOME/Megatron-LM" || { print_error "Failed to enter Megatron-LM directory"; complete_progress_bar; return 1; }

    # Install Megatron-LM with uv
    if command -v uv >/dev/null 2>&1; then
        if ! uv pip install -e .; then
            print_error "Failed to install Megatron-LM with uv"
            print_step "Trying with pip..."
            if ! pip install -e .; then
                print_error "Failed to install Megatron-LM"
                complete_progress_bar
                return 1
            fi
        fi
    else
        if ! pip install -e .; then
            print_error "Failed to install Megatron-LM"
            complete_progress_bar
            return 1
        fi
    fi

    update_progress_bar 80
    draw_progress_bar "Verifying installation..."

    # Verify installation
    if python_module_exists "megatron"; then
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

        echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"

        # Kill any remaining background processes and force exit
        jobs -p | xargs -r kill -9 2>/dev/null
        kill -9 $$ 2>/dev/null
        exit 0
    else
        print_error "Megatron-LM installation failed"
        complete_progress_bar

        # Force exit even on failure
        echo -e "${RED}${BOLD}Installation failed. Exiting now.${RESET}"
        jobs -p | xargs -r kill -9 2>/dev/null
        kill -9 $$ 2>/dev/null
        exit 1
    fi
}

# Trap to ensure we exit properly
trap 'echo "Forced exit"; kill -9 $$' EXIT

# Main function - run directly without nested functions to avoid return issues
install_megatron

# Force exit regardless of what happened above
echo "Forcing exit to prevent hanging..."
kill -9 $$
exit 0
