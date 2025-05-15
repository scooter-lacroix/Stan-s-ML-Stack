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

# Function to install MIGraphX Python module
install_migraphx_python() {
    print_header "Installing MIGraphX Python Module"
    
    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking MIGraphX installation..."
    
    # Check if MIGraphX is installed
    if ! command -v migraphx-driver >/dev/null 2>&1; then
        print_error "MIGraphX is not installed. Please install MIGraphX first."
        print_step "Run the install_migraphx.sh script to install MIGraphX."
        complete_progress_bar
        return 1
    fi
    
    print_success "MIGraphX is installed"
    update_progress_bar 10
    draw_progress_bar "Checking for MIGraphX Python module..."
    
    # Check if MIGraphX Python module is already installed
    if python_module_exists "migraphx"; then
        print_success "MIGraphX Python module is already installed"
        migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
        print_step "MIGraphX Python module version: $migraphx_version"
        complete_progress_bar
        return 0
    fi
    
    print_step "Installing MIGraphX Python module..."
    update_progress_bar 20
    draw_progress_bar "Installing MIGraphX Python module..."
    
    # Set environment variables
    export ROCM_PATH=/opt/rocm
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    export PATH=$ROCM_PATH/bin:$PATH
    
    # Suppress HIP logs
    export AMD_LOG_LEVEL=0
    export HIP_VISIBLE_DEVICES=0,1,2
    export ROCR_VISIBLE_DEVICES=0,1,2
    
    # Try to install with uv first
    if command -v uv >/dev/null 2>&1; then
        print_step "Installing with uv..."
        update_progress_bar 10
        draw_progress_bar "Installing with uv..."
        
        if uv pip install migraphx; then
            print_success "Successfully installed MIGraphX Python module with uv"
            update_progress_bar 40
            draw_progress_bar "Verifying installation..."
            
            if python_module_exists "migraphx"; then
                migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
                print_success "MIGraphX Python module is installed (version: $migraphx_version)"
                complete_progress_bar
                return 0
            else
                print_warning "MIGraphX Python module installation with uv failed, trying with pip..."
            fi
        else
            print_warning "Failed to install MIGraphX Python module with uv, trying with pip..."
        fi
    fi
    
    # Fallback to pip
    update_progress_bar 10
    draw_progress_bar "Installing with pip..."
    
    if pip install migraphx; then
        print_success "Successfully installed MIGraphX Python module with pip"
        update_progress_bar 40
        draw_progress_bar "Verifying installation..."
        
        if python_module_exists "migraphx"; then
            migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
            print_success "MIGraphX Python module is installed (version: $migraphx_version)"
            complete_progress_bar
            return 0
        else
            print_error "MIGraphX Python module installation failed"
            complete_progress_bar
            return 1
        fi
    else
        print_error "Failed to install MIGraphX Python module with pip"
        complete_progress_bar
        return 1
    fi
}

# Main function
main() {
    install_migraphx_python
    return $?
}

# Run main function
main
exit $?
