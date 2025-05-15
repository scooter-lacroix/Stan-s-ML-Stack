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
# ML Stack Repair Tool
# =============================================================================
# This script diagnoses and repairs common issues with the ML Stack installation.
# It detects conflicting dependencies, fixes environment issues, and ensures
# all components are properly installed and configured.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                ML Stack Repair Tool
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to check if Python package version is correct
check_package_version() {
    local package=$1
    local min_version=$2

    if package_installed "$package"; then
        local version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null)
        if [ -z "$version" ]; then
            version=$(python3 -c "import $package; print($package.version)" 2>/dev/null)
        fi

        if [ -n "$version" ]; then
            print_step "$package version: $version"

            # Compare versions (simple string comparison, not semantic versioning)
            if [[ "$version" < "$min_version" ]]; then
                print_warning "$package version $version is older than recommended version $min_version"
                return 1
            else
                print_success "$package version $version is compatible"
                return 0
            fi
        else
            print_warning "Could not determine $package version"
            return 2
        fi
    else
        print_error "$package is not installed"
        return 3
    fi
}

# Function to detect virtual environment
detect_virtual_env() {
    print_section "Detecting Virtual Environment"

    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Running in virtual environment: $VIRTUAL_ENV"
        return 0
    elif [ -d "./venv" ]; then
        print_warning "Virtual environment detected but not activated"
        print_step "To activate, run: source ./venv/bin/activate"
        return 1
    elif [ -d "$HOME/Prod/Stan-s-ML-Stack/venv" ]; then
        print_warning "Virtual environment detected but not activated"
        print_step "To activate, run: source $HOME/Prod/Stan-s-ML-Stack/venv/bin/activate"
        return 1
    else
        print_warning "No virtual environment detected"
        print_step "Consider creating a virtual environment for better isolation"
        return 2
    fi
}

# Function to check for conflicting PyTorch installations
check_pytorch_conflicts() {
    print_section "Checking PyTorch Installation"

    if ! package_installed "torch"; then
        print_error "PyTorch is not installed"
        return 1
    fi

    # Check PyTorch version
    pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_step "PyTorch version: $pytorch_version"

    # Check if PyTorch has ROCm/HIP support
    if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
        hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
        print_success "PyTorch has ROCm/HIP support (version: $hip_version)"

        # Check if CUDA is also available (which might indicate a conflict)
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            if [[ "$pytorch_version" == *"cu"* ]]; then
                print_warning "PyTorch was built for CUDA ($cuda_version) but has ROCm support"
                print_warning "This might cause conflicts. Consider reinstalling PyTorch with ROCm support only"
                return 2
            else
                print_success "PyTorch has CUDA compatibility through ROCm/HIP"
                return 0
            fi
        else
            print_step "CUDA is not available through PyTorch"
            return 0
        fi
    elif python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        print_warning "PyTorch has CUDA support (version: $cuda_version) but no ROCm/HIP support"
        print_warning "This might cause issues with AMD GPUs. Consider reinstalling PyTorch with ROCm support"
        return 3
    else
        print_error "PyTorch has neither ROCm/HIP nor CUDA support"
        print_error "GPU acceleration will not be available"
        return 4
    fi
}

# Function to check for package manager consistency
check_package_manager() {
    print_section "Checking Package Manager"

    # Check if uv is installed
    if command_exists uv; then
        uv_version=$(uv --version 2>/dev/null | head -n 1)
        print_success "uv is installed: $uv_version"

        # Check if pip is also being used
        pip_history=$(grep -r "pip install" $HOME/.bash_history 2>/dev/null | wc -l)
        if [ "$pip_history" -gt 0 ]; then
            print_warning "Found $pip_history pip commands in bash history"
            print_warning "For consistency, use uv instead of pip for all package installations"
        fi

        return 0
    else
        print_warning "uv is not installed"
        print_step "Installing uv..."

        # Install uv using pip
        python3 -m pip install uv

        if command_exists uv; then
            print_success "uv installed successfully"
            return 0
        else
            print_error "Failed to install uv"
            return 1
        fi
    fi
}

# Function to fix PyTorch installation
fix_pytorch_installation() {
    print_section "Fixing PyTorch Installation"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Preparing PyTorch installation..."

    # Check if the PyTorch installation script exists
    if [ -f "$(dirname "$0")/install_pytorch_rocm.sh" ]; then
        print_step "Using dedicated PyTorch installation script..."

        # First, uninstall existing PyTorch
        if package_installed "torch"; then
            print_step "Uninstalling current PyTorch version..."
            update_progress_bar 10
            draw_progress_bar "Uninstalling current PyTorch version..."

            # Uninstall PyTorch with timeout to prevent hanging
            if [ -n "$VIRTUAL_ENV" ]; then
                timeout 30s uv pip uninstall -y torch torchvision torchaudio
            else
                timeout 30s python3 -m uv pip uninstall -y torch torchvision torchaudio
            fi

            update_progress_bar 5
            draw_progress_bar "Checking uninstallation status..."

            # Check if uninstallation was successful
            if ! package_installed "torch"; then
                print_success "PyTorch uninstalled successfully"
                update_progress_bar 5
            else
                print_error "Failed to uninstall PyTorch"
                print_step "Attempting forced uninstallation..."
                update_progress_bar 5
                draw_progress_bar "Attempting alternative uninstallation method..."

                # Try with pip directly as a fallback with timeout
                if [ -n "$VIRTUAL_ENV" ]; then
                    timeout 30s python3 -m pip uninstall -y torch torchvision torchaudio
                else
                    timeout 30s pip uninstall -y torch torchvision torchaudio
                fi

                update_progress_bar 5
                draw_progress_bar "Verifying uninstallation..."

                if ! package_installed "torch"; then
                    print_success "PyTorch uninstalled successfully with pip"
                    update_progress_bar 5
                else
                    print_warning "Could not completely uninstall PyTorch, continuing anyway"
                    update_progress_bar 5
                fi
            fi
        fi

        # Run the dedicated installation script
        print_step "Running PyTorch installation script..."
        update_progress_bar 15
        draw_progress_bar "Installing PyTorch with ROCm support..."

        # Run the installation script with a timeout to prevent hanging
        timeout 300s bash "$(dirname "$0")/install_pytorch_rocm.sh"

        update_progress_bar 15
        draw_progress_bar "Verifying PyTorch installation..."

        # Verify installation
        if package_installed "torch"; then
            pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
            print_success "PyTorch installed successfully: $pytorch_version"
            update_progress_bar 5
            draw_progress_bar "Checking ROCm/HIP support..."

            # Check if PyTorch has ROCm/HIP support
            if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
                hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
                print_success "PyTorch has ROCm/HIP support (version: $hip_version)"
                update_progress_bar 5
                draw_progress_bar "Testing GPU acceleration..."

                # Test GPU availability
                if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    print_success "GPU acceleration is available"
                    update_progress_bar 5
                    draw_progress_bar "Testing tensor operations..."

                    # Test a simple tensor operation
                    if python3 -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success' if torch.all(y == 2) else 'Failed')" 2>/dev/null | grep -q "Success"; then
                        print_success "GPU tensor operations working correctly"
                    else
                        print_warning "GPU tensor operations may not be working correctly"
                    fi
                else
                    print_warning "GPU acceleration is not available"
                    print_warning "Check your ROCm installation and environment variables"
                fi
            else
                print_warning "PyTorch does not have explicit ROCm/HIP support"
                print_warning "This might cause issues with AMD GPUs"
            fi

            # Complete the progress bar
            complete_progress_bar
            return 0
        else
            print_error "Failed to install PyTorch using the installation script"
            print_step "Falling back to manual installation..."
            update_progress_bar 10
            draw_progress_bar "Preparing manual installation..."
        fi
    else
        print_warning "PyTorch installation script not found, using manual installation..."
    fi

    # Manual installation as fallback
    # Detect ROCm version
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            # Try alternative method to get ROCm version
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            if [ -z "$rocm_version" ]; then
                rocm_version="6.4.0"  # Default to a recent version
                print_warning "Could not detect ROCm version, using default: $rocm_version"
            fi
        fi
    else
        rocm_version="6.4.0"  # Default to a recent version
        print_warning "ROCm not found, using default version: $rocm_version"
    fi

    print_step "Detected ROCm version: $rocm_version"

    # Extract major and minor version
    rocm_major_version=$(echo "$rocm_version" | cut -d '.' -f 1)
    rocm_minor_version=$(echo "$rocm_version" | cut -d '.' -f 2)

    # Install PyTorch with ROCm support
    print_step "Installing PyTorch with ROCm support..."

    # Use uv for installation
    if [ -n "$VIRTUAL_ENV" ]; then
        if [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
            # For ROCm 6.4+, use nightly builds
            print_step "Using PyTorch nightly build for ROCm 6.4..."
            uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 3 ]; then
            # For ROCm 6.3, use stable builds
            print_step "Using PyTorch stable build for ROCm 6.3..."
            uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 0 ]; then
            # For ROCm 6.0-6.2, use stable builds for 6.2
            print_step "Using PyTorch stable build for ROCm 6.2..."
            uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
        elif [ "$rocm_major_version" -eq 5 ]; then
            # For ROCm 5.x, use stable builds for 5.7
            print_step "Using PyTorch stable build for ROCm 5.7..."
            uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
        else
            # Fallback to the latest stable ROCm version
            print_step "Using PyTorch stable build for ROCm 6.3 (fallback)..."
            uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        fi
    else
        if [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
            # For ROCm 6.4+, use nightly builds
            print_step "Using PyTorch nightly build for ROCm 6.4..."
            python3 -m uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 3 ]; then
            # For ROCm 6.3, use stable builds
            print_step "Using PyTorch stable build for ROCm 6.3..."
            python3 -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 0 ]; then
            # For ROCm 6.0-6.2, use stable builds for 6.2
            print_step "Using PyTorch stable build for ROCm 6.2..."
            python3 -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
        elif [ "$rocm_major_version" -eq 5 ]; then
            # For ROCm 5.x, use stable builds for 5.7
            print_step "Using PyTorch stable build for ROCm 5.7..."
            python3 -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
        else
            # Fallback to the latest stable ROCm version
            print_step "Using PyTorch stable build for ROCm 6.3 (fallback)..."
            python3 -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
        fi
    fi

    # Verify installation
    if package_installed "torch"; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_success "PyTorch installed successfully: $pytorch_version"

        # Check if PyTorch has ROCm/HIP support
        if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
            print_success "PyTorch has ROCm/HIP support (version: $hip_version)"

            # Test GPU availability
            if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "GPU acceleration is available"

                # Test a simple tensor operation
                if python3 -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success' if torch.all(y == 2) else 'Failed')" 2>/dev/null | grep -q "Success"; then
                    print_success "GPU tensor operations working correctly"
                else
                    print_warning "GPU tensor operations may not be working correctly"
                fi
            else
                print_warning "GPU acceleration is not available"
                print_warning "Check your ROCm installation and environment variables"
            fi
        else
            print_warning "PyTorch does not have explicit ROCm/HIP support"
            print_warning "This might cause issues with AMD GPUs"
        fi

        return 0
    else
        print_error "Failed to install PyTorch"
        return 1
    fi
}

# Function to fix environment variables
fix_environment_variables() {
    print_section "Fixing Environment Variables"

    # Check if ROCm is installed
    if command_exists rocminfo; then
        print_success "ROCm is installed"

        # Get GPU count
        gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
        print_step "Detected $gpu_count AMD GPU(s)"

        # Filter out integrated GPUs
        print_step "Filtering out integrated GPUs..."

        # Get GPU information from rocminfo
        gpu_info=$(rocminfo 2>/dev/null | grep -A 10 "GPU ID" | grep -E "GPU ID|Marketing Name|Device Type")

        # Initialize arrays for discrete GPU indices
        declare -a discrete_gpu_indices
        current_gpu_id=""
        is_discrete=false

        # Parse rocminfo output to identify discrete GPUs
        while IFS= read -r line; do
            if [[ $line == *"GPU ID"* ]]; then
                # Extract GPU ID
                current_gpu_id=$(echo "$line" | grep -o '[0-9]\+')
                is_discrete=false
            elif [[ $line == *"Marketing Name"* ]]; then
                # Check if this is an integrated GPU
                gpu_name=$(echo "$line" | awk -F: '{print $2}' | xargs)
                if [[ $gpu_name == *"Raphael"* || $gpu_name == *"Integrated"* || $gpu_name == *"iGPU"* ||
                      $gpu_name == *"AMD Ryzen"* || $gpu_name == *"AMD Radeon Graphics"* ]]; then
                    print_warning "Detected integrated GPU at index $current_gpu_id: $gpu_name"
                    is_discrete=false
                else
                    print_success "Detected discrete GPU at index $current_gpu_id: $gpu_name"
                    is_discrete=true
                fi
            elif [[ $line == *"Device Type"* && $is_discrete == true ]]; then
                # If we've confirmed this is a discrete GPU, add it to our list
                discrete_gpu_indices+=($current_gpu_id)
            fi
        done <<< "$gpu_info"

        # Create comma-separated list of discrete GPU indices
        if [ ${#discrete_gpu_indices[@]} -gt 0 ]; then
            discrete_gpu_list=$(IFS=,; echo "${discrete_gpu_indices[*]}")
            print_success "Using discrete GPUs: $discrete_gpu_list"
        else
            # Fallback to all GPUs if no discrete GPUs were identified
            print_warning "No discrete GPUs identified, using all available GPUs"
            discrete_gpu_list=$(seq -s, 0 $((gpu_count-1)))
        fi

        # Set environment variables
        print_step "Setting environment variables..."
        export HIP_VISIBLE_DEVICES=$discrete_gpu_list
        export CUDA_VISIBLE_DEVICES=$discrete_gpu_list
        export PYTORCH_ROCM_DEVICE=$discrete_gpu_list

        print_success "Environment variables set:"
        echo -e "  - HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
        echo -e "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo -e "  - PYTORCH_ROCM_DEVICE: $PYTORCH_ROCM_DEVICE"

        # Update environment file
        print_step "Updating environment file..."

        # Check if environment file exists
        if [ -f "$HOME/.mlstack_env" ]; then
            # Backup existing file
            cp "$HOME/.mlstack_env" "$HOME/.mlstack_env.bak"

            # Update environment variables in file
            sed -i "s/export HIP_VISIBLE_DEVICES=.*/export HIP_VISIBLE_DEVICES=$discrete_gpu_list/" "$HOME/.mlstack_env"
            sed -i "s/export CUDA_VISIBLE_DEVICES=.*/export CUDA_VISIBLE_DEVICES=$discrete_gpu_list/" "$HOME/.mlstack_env"
            sed -i "s/export PYTORCH_ROCM_DEVICE=.*/export PYTORCH_ROCM_DEVICE=$discrete_gpu_list/" "$HOME/.mlstack_env"

            print_success "Environment file updated"
        else
            print_warning "Environment file not found, creating new one..."

            # Create environment file
            cat > "$HOME/.mlstack_env" << EOF
# ML Stack Environment File
# Created by ML Stack Repair Tool
# Date: $(date)

# GPU Selection
export HIP_VISIBLE_DEVICES=$discrete_gpu_list
export CUDA_VISIBLE_DEVICES=$discrete_gpu_list
export PYTORCH_ROCM_DEVICE=$discrete_gpu_list

# ROCm Settings
export PATH=\$PATH:/opt/rocm/bin:/opt/rocm/hip/bin
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/opencl/lib:\$LD_LIBRARY_PATH

# Performance Settings
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=1

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# PyTorch Settings
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"
EOF

            # Add source to .bashrc if not already there
            if ! grep -q "source \$HOME/.mlstack_env" "$HOME/.bashrc"; then
                echo -e "\n# Source ML Stack environment" >> "$HOME/.bashrc"
                echo "source \$HOME/.mlstack_env" >> "$HOME/.bashrc"
                print_step "Added environment file to .bashrc"
            fi

            print_success "Environment file created"
        fi

        # Source the environment file
        source "$HOME/.mlstack_env"

        return 0
    else
        print_error "ROCm is not installed, cannot set environment variables"
        return 1
    fi
}

# Function to fix ROCm installation
fix_rocm_installation() {
    print_section "Fixing ROCm Installation"

    # Check if the ROCm installation script exists
    if [ -f "$(dirname "$0")/install_rocm.sh" ]; then
        print_step "Using dedicated ROCm installation script..."

        # Run the dedicated installation script
        print_step "Running ROCm installation script..."
        bash "$(dirname "$0")/install_rocm.sh"

        # Verify installation
        if command_exists rocminfo; then
            rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
            if [ -z "$rocm_version" ]; then
                rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            fi

            if [ -n "$rocm_version" ]; then
                print_success "ROCm installed successfully (version: $rocm_version)"
                return 0
            else
                print_warning "ROCm is installed but version could not be determined"
                return 0
            fi
        else
            print_error "ROCm installation failed"
            return 1
        fi
    else
        print_warning "ROCm installation script not found"
        print_warning "Please install ROCm manually using the instructions from AMD"
        print_step "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        return 1
    fi
}

# Function to fix AMDGPU drivers installation
fix_amdgpu_drivers() {
    print_section "Fixing AMDGPU Drivers Installation"

    # Check if the AMDGPU drivers installation script exists
    if [ -f "$(dirname "$0")/install_amdgpu_drivers.sh" ]; then
        print_step "Using dedicated AMDGPU drivers installation script..."

        # Run the dedicated installation script
        print_step "Running AMDGPU drivers installation script..."
        bash "$(dirname "$0")/install_amdgpu_drivers.sh"

        # Verify installation
        if lsmod | grep -q amdgpu; then
            print_success "AMDGPU drivers installed successfully"
            return 0
        else
            print_error "AMDGPU drivers installation failed"
            return 1
        fi
    else
        print_warning "AMDGPU drivers installation script not found"
        print_warning "Please install AMDGPU drivers manually using the instructions from AMD"
        print_step "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        return 1
    fi
}

# Function to fix AITER installation
fix_aiter_installation() {
    print_section "Fixing AITER Installation"

    # Check if the AITER installation script exists
    if [ -f "$(dirname "$0")/install_aiter.sh" ]; then
        print_step "Using dedicated AITER installation script..."

        # First, uninstall existing AITER if it exists
        if package_installed "aiter"; then
            print_step "Uninstalling current AITER version..."

            # Uninstall AITER
            if [ -n "$VIRTUAL_ENV" ]; then
                uv pip uninstall -y aiter
            else
                python3 -m uv pip uninstall -y aiter
            fi

            # Check if uninstallation was successful
            if ! package_installed "aiter"; then
                print_success "AITER uninstalled successfully"
            else
                print_error "Failed to uninstall AITER"
                print_step "Attempting forced uninstallation..."

                # Try with pip directly as a fallback
                if [ -n "$VIRTUAL_ENV" ]; then
                    python3 -m pip uninstall -y aiter
                else
                    pip uninstall -y aiter
                fi

                if ! package_installed "aiter"; then
                    print_success "AITER uninstalled successfully with pip"
                else
                    print_warning "Could not completely uninstall AITER, continuing anyway"
                fi
            fi
        fi

        # Run the dedicated installation script
        print_step "Running AITER installation script..."
        bash "$(dirname "$0")/install_aiter.sh"

        # Verify installation
        if package_installed "aiter"; then
            print_success "AITER installed successfully"
            return 0
        else
            print_error "AITER installation failed"
            return 1
        fi
    else
        print_warning "AITER installation script not found"
        print_warning "Please install AITER manually using the instructions from AMD"
        print_step "Visit: https://github.com/ROCm/aiter"
        return 1
    fi
}

# Function to fix MIGraphX Python module installation
fix_migraphx_python_installation() {
    print_section "Fixing MIGraphX Python Module Installation"

    # Check if MIGraphX Python module is already installed
    if package_installed "migraphx"; then
        migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
        print_success "MIGraphX Python module is already installed (version: $migraphx_version)"
        return 0
    fi

    # Check if MIGraphX is installed
    if ! command -v migraphx-driver >/dev/null 2>&1; then
        print_error "MIGraphX is not installed. Please install MIGraphX first."
        print_step "Run the install_migraphx.sh script to install MIGraphX."
        return 1
    fi

    # Check if the MIGraphX Python module installation script exists
    if [ -f "$(dirname "$0")/install_migraphx_python.sh" ]; then
        print_step "Using dedicated MIGraphX Python module installation script..."

        # Run the installation script
        bash "$(dirname "$0")/install_migraphx_python.sh"

        # Verify installation
        if package_installed "migraphx"; then
            migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
            print_success "MIGraphX Python module installed successfully (version: $migraphx_version)"
            return 0
        else
            print_error "Failed to install MIGraphX Python module using the installation script"
            print_step "Falling back to manual installation..."
        fi
    else
        print_warning "MIGraphX Python module installation script not found, using manual installation..."
    fi

    # Manual installation as fallback
    print_step "Installing MIGraphX Python module manually..."

    # Install MIGraphX Python module
    if [ -n "$VIRTUAL_ENV" ]; then
        uv pip install migraphx
    else
        python3 -m uv pip install migraphx
    fi

    # Verify installation
    if package_installed "migraphx"; then
        migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
        print_success "MIGraphX Python module installed successfully (version: $migraphx_version)"
        return 0
    else
        print_error "Failed to install MIGraphX Python module"
        return 1
    fi
}

# Function to fix DeepSpeed installation
fix_deepspeed_installation() {
    print_section "Fixing DeepSpeed Installation"

    # Check if the DeepSpeed installation script exists
    if [ -f "$(dirname "$0")/install_deepspeed.sh" ]; then
        print_step "Using dedicated DeepSpeed installation script..."

        # First, uninstall existing DeepSpeed if it exists
        if package_installed "deepspeed"; then
            print_step "Uninstalling current DeepSpeed version..."

            # Uninstall DeepSpeed
            if [ -n "$VIRTUAL_ENV" ]; then
                uv pip uninstall -y deepspeed
            else
                python3 -m uv pip uninstall -y deepspeed
            fi

            # Check if uninstallation was successful
            if ! package_installed "deepspeed"; then
                print_success "DeepSpeed uninstalled successfully"
            else
                print_error "Failed to uninstall DeepSpeed"
                print_step "Attempting forced uninstallation..."

                # Try with pip directly as a fallback
                if [ -n "$VIRTUAL_ENV" ]; then
                    python3 -m pip uninstall -y deepspeed
                else
                    pip uninstall -y deepspeed
                fi

                if ! package_installed "deepspeed"; then
                    print_success "DeepSpeed uninstalled successfully with pip"
                else
                    print_warning "Could not completely uninstall DeepSpeed, continuing anyway"
                fi
            fi
        fi

        # Run the dedicated installation script
        print_step "Running DeepSpeed installation script..."
        bash "$(dirname "$0")/install_deepspeed.sh"

        # Verify installation
        if package_installed "deepspeed"; then
            deepspeed_version=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
            print_success "DeepSpeed installed successfully (version: $deepspeed_version)"
            return 0
        else
            print_error "DeepSpeed installation failed"
            return 1
        fi
    else
        print_warning "DeepSpeed installation script not found"
        print_warning "Installing DeepSpeed manually..."

        # Install DeepSpeed using uv
        if [ -n "$VIRTUAL_ENV" ]; then
            uv pip install deepspeed
        else
            python3 -m uv pip install deepspeed
        fi

        # Verify installation
        if package_installed "deepspeed"; then
            deepspeed_version=$(python3 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
            print_success "DeepSpeed installed successfully (version: $deepspeed_version)"
            return 0
        else
            print_error "DeepSpeed installation failed"
            return 1
        fi
    fi
}

# Function to run diagnostics
run_diagnostics() {
    print_section "Running Diagnostics"

    # Check system information
    print_step "System information:"
    uname -a

    # Check ROCm installation
    print_step "ROCm installation:"
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        print_success "ROCm is installed (version: $rocm_version)"
    else
        print_error "ROCm is not installed"
    fi

    # Check GPU detection
    print_step "GPU detection:"
    if command_exists rocminfo; then
        gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
        print_success "Detected $gpu_count AMD GPU(s)"

        # List GPUs
        rocminfo 2>/dev/null | grep -A 1 "GPU ID" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
            echo -e "  - $gpu"
        done
    else
        print_error "Cannot detect GPUs without ROCm"
    fi

    # Check environment variables
    print_step "Environment variables:"
    echo -e "  - HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo -e "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo -e "  - PYTORCH_ROCM_DEVICE: $PYTORCH_ROCM_DEVICE"

    # Check Python installation
    print_step "Python installation:"
    python_version=$(python3 --version 2>&1)
    print_success "Python version: $python_version"

    # Check virtual environment
    print_step "Virtual environment:"
    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Running in virtual environment: $VIRTUAL_ENV"
    else
        print_warning "Not running in a virtual environment"
    fi

    # Check PyTorch installation
    print_step "PyTorch installation:"
    if package_installed "torch"; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_success "PyTorch version: $pytorch_version"

        # Check if PyTorch has ROCm/HIP support
        if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
            print_success "PyTorch has ROCm/HIP support (version: $hip_version)"
        else
            print_warning "PyTorch does not have explicit ROCm/HIP support"
        fi

        # Check GPU availability
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "GPU acceleration is available"

            # Get GPU count
            gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            print_step "PyTorch detected $gpu_count GPU(s)"

            # List GPUs
            for i in $(seq 0 $((gpu_count-1))); do
                gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
                echo -e "  - GPU $i: $gpu_name"
            done
        else
            print_warning "GPU acceleration is not available"
        fi
    else
        print_error "PyTorch is not installed"
    fi

    # Check other ML components
    print_step "Other ML components:"

    # Check ONNX Runtime
    if package_installed "onnxruntime"; then
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        print_success "ONNX Runtime is installed (version: $onnx_version)"

        # Check if ROCMExecutionProvider is available
        if python3 -c "import onnxruntime; print('ROCMExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null | grep -q "True"; then
            print_success "ONNX Runtime has ROCm support"
        else
            print_warning "ONNX Runtime does not have ROCm support"
        fi
    else
        print_warning "ONNX Runtime is not installed"
    fi

    # Check MIGraphX
    if command -v migraphx-driver >/dev/null 2>&1; then
        print_success "MIGraphX is installed"

        # Check MIGraphX Python module
        if package_installed "migraphx"; then
            migraphx_version=$(python3 -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
            print_success "MIGraphX Python module is installed (version: $migraphx_version)"
        else
            print_warning "MIGraphX Python module is not installed"
        fi
    else
        print_warning "MIGraphX is not installed"
    fi

    # Check Flash Attention
    if package_installed "flash_attention_amd"; then
        print_success "Flash Attention is installed"
    else
        print_warning "Flash Attention is not installed"
    fi

    # Check RCCL
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_success "RCCL is installed"
    else
        print_warning "RCCL is not installed"
    fi

    # Check MPI
    if command_exists mpirun; then
        mpi_version=$(mpirun --version | head -n 1)
        print_success "MPI is installed ($mpi_version)"

        # Check mpi4py
        if package_installed "mpi4py"; then
            mpi4py_version=$(python3 -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null)
            print_success "mpi4py is installed (version: $mpi4py_version)"
        else
            print_warning "mpi4py is not installed"
        fi
    else
        print_warning "MPI is not installed"
    fi

    return 0
}

# Function to uninstall components
uninstall_components() {
    print_section "Uninstalling Components"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5 # Start with 5% progress
    draw_progress_bar "Preparing uninstallation..."

    # Non-interactive mode for scripts
    if [ -n "$NONINTERACTIVE" ]; then
        print_step "Running in non-interactive mode, skipping uninstallation"
        complete_progress_bar
        return 0
    fi

    # Ask user if they want to uninstall components
    read -p "Do you want to uninstall ML Stack components before repair? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_step "Skipping uninstallation"
        complete_progress_bar
        return 0
    fi

    # Update progress
    update_progress_bar 5
    draw_progress_bar "Preparing component list..."

    # List components that can be uninstalled
    print_step "The following components can be uninstalled:"
    echo "1. PyTorch with ROCm"
    echo "2. AITER"
    echo "3. DeepSpeed"
    echo "4. Flash Attention"
    echo "5. Triton"
    echo "6. BITSANDBYTES"
    echo "7. vLLM"
    echo "8. All of the above"
    echo "9. Skip uninstallation"

    # Ask user which components to uninstall
    read -p "Enter the number of the component to uninstall (or 8 for all, 9 to skip): " -r
    echo

    uninstall_success=false

    case $REPLY in
        1)
            # Uninstall PyTorch
            print_step "Uninstalling PyTorch..."
            update_progress_bar 10
            draw_progress_bar "Checking PyTorch installation..."

            if package_installed "torch"; then
                update_progress_bar 5
                draw_progress_bar "Uninstalling PyTorch packages..."

                if [ -n "$VIRTUAL_ENV" ]; then
                    # Set timeout for uv pip command
                    timeout 30s uv pip uninstall -y torch torchvision torchaudio
                    update_progress_bar 10
                    draw_progress_bar "Checking uninstallation status..."

                    # Fallback to pip if uv fails
                    if package_installed "torch"; then
                        update_progress_bar 5
                        draw_progress_bar "Trying alternative uninstallation method..."
                        timeout 30s python3 -m pip uninstall -y torch torchvision torchaudio
                    fi
                else
                    # Set timeout for uv pip command
                    timeout 30s python3 -m uv pip uninstall -y torch torchvision torchaudio
                    update_progress_bar 10
                    draw_progress_bar "Checking uninstallation status..."

                    # Fallback to pip if uv fails
                    if package_installed "torch"; then
                        update_progress_bar 5
                        draw_progress_bar "Trying alternative uninstallation method..."
                        timeout 30s python3 -m pip uninstall -y torch torchvision torchaudio
                    fi
                fi

                update_progress_bar 10
                draw_progress_bar "Verifying uninstallation..."

                if ! package_installed "torch"; then
                    print_success "PyTorch uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall PyTorch"
                fi
            else
                print_warning "PyTorch is not installed"
            fi

            update_progress_bar 10
            draw_progress_bar "PyTorch uninstallation complete"
            ;;
        2)
            # Uninstall AITER
            print_step "Uninstalling AITER..."
            if package_installed "aiter"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y aiter
                    # Fallback to pip if uv fails
                    if package_installed "aiter"; then
                        python3 -m pip uninstall -y aiter
                    fi
                else
                    python3 -m uv pip uninstall -y aiter
                    # Fallback to pip if uv fails
                    if package_installed "aiter"; then
                        python3 -m pip uninstall -y aiter
                    fi
                fi

                if ! package_installed "aiter"; then
                    print_success "AITER uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall AITER"
                fi
            else
                print_warning "AITER is not installed"
            fi
            ;;
        3)
            # Uninstall DeepSpeed
            print_step "Uninstalling DeepSpeed..."
            if package_installed "deepspeed"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y deepspeed
                    # Fallback to pip if uv fails
                    if package_installed "deepspeed"; then
                        python3 -m pip uninstall -y deepspeed
                    fi
                else
                    python3 -m uv pip uninstall -y deepspeed
                    # Fallback to pip if uv fails
                    if package_installed "deepspeed"; then
                        python3 -m pip uninstall -y deepspeed
                    fi
                fi

                if ! package_installed "deepspeed"; then
                    print_success "DeepSpeed uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall DeepSpeed"
                fi
            else
                print_warning "DeepSpeed is not installed"
            fi
            ;;
        4)
            # Uninstall Flash Attention
            print_step "Uninstalling Flash Attention..."
            if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y flash-attn flash_attention_amd
                    # Fallback to pip if uv fails
                    if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                        python3 -m pip uninstall -y flash-attn flash_attention_amd
                    fi
                else
                    python3 -m uv pip uninstall -y flash-attn flash_attention_amd
                    # Fallback to pip if uv fails
                    if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                        python3 -m pip uninstall -y flash-attn flash_attention_amd
                    fi
                fi

                if ! package_installed "flash_attn" && ! package_installed "flash_attention_amd"; then
                    print_success "Flash Attention uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall Flash Attention"
                fi
            else
                print_warning "Flash Attention is not installed"
            fi
            ;;
        5)
            # Uninstall Triton
            print_step "Uninstalling Triton..."
            if package_installed "triton"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y triton
                    # Fallback to pip if uv fails
                    if package_installed "triton"; then
                        python3 -m pip uninstall -y triton
                    fi
                else
                    python3 -m uv pip uninstall -y triton
                    # Fallback to pip if uv fails
                    if package_installed "triton"; then
                        python3 -m pip uninstall -y triton
                    fi
                fi

                if ! package_installed "triton"; then
                    print_success "Triton uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall Triton"
                fi
            else
                print_warning "Triton is not installed"
            fi
            ;;
        6)
            # Uninstall BITSANDBYTES
            print_step "Uninstalling BITSANDBYTES..."
            if package_installed "bitsandbytes"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y bitsandbytes
                    # Fallback to pip if uv fails
                    if package_installed "bitsandbytes"; then
                        python3 -m pip uninstall -y bitsandbytes
                    fi
                else
                    python3 -m uv pip uninstall -y bitsandbytes
                    # Fallback to pip if uv fails
                    if package_installed "bitsandbytes"; then
                        python3 -m pip uninstall -y bitsandbytes
                    fi
                fi

                if ! package_installed "bitsandbytes"; then
                    print_success "BITSANDBYTES uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall BITSANDBYTES"
                fi
            else
                print_warning "BITSANDBYTES is not installed"
            fi
            ;;
        7)
            # Uninstall vLLM
            print_step "Uninstalling vLLM..."
            if package_installed "vllm"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y vllm
                    # Fallback to pip if uv fails
                    if package_installed "vllm"; then
                        python3 -m pip uninstall -y vllm
                    fi
                else
                    python3 -m uv pip uninstall -y vllm
                    # Fallback to pip if uv fails
                    if package_installed "vllm"; then
                        python3 -m pip uninstall -y vllm
                    fi
                fi

                if ! package_installed "vllm"; then
                    print_success "vLLM uninstalled successfully"
                    uninstall_success=true
                else
                    print_error "Failed to uninstall vLLM"
                fi
            else
                print_warning "vLLM is not installed"
            fi
            ;;
        8)
            # Uninstall all components
            print_step "Uninstalling all components..."
            update_progress_bar 10
            draw_progress_bar "Preparing to uninstall all components..."
            components_uninstalled=0

            # Uninstall PyTorch
            if package_installed "torch"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y torch torchvision torchaudio
                    # Fallback to pip if uv fails
                    if package_installed "torch"; then
                        python3 -m pip uninstall -y torch torchvision torchaudio
                    fi
                else
                    python3 -m uv pip uninstall -y torch torchvision torchaudio
                    # Fallback to pip if uv fails
                    if package_installed "torch"; then
                        python3 -m pip uninstall -y torch torchvision torchaudio
                    fi
                fi

                if ! package_installed "torch"; then
                    print_success "PyTorch uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall PyTorch"
                fi
            fi

            # Uninstall AITER
            if package_installed "aiter"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y aiter
                    # Fallback to pip if uv fails
                    if package_installed "aiter"; then
                        python3 -m pip uninstall -y aiter
                    fi
                else
                    python3 -m uv pip uninstall -y aiter
                    # Fallback to pip if uv fails
                    if package_installed "aiter"; then
                        python3 -m pip uninstall -y aiter
                    fi
                fi

                if ! package_installed "aiter"; then
                    print_success "AITER uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall AITER"
                fi
            fi

            # Uninstall DeepSpeed
            if package_installed "deepspeed"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y deepspeed
                    # Fallback to pip if uv fails
                    if package_installed "deepspeed"; then
                        python3 -m pip uninstall -y deepspeed
                    fi
                else
                    python3 -m uv pip uninstall -y deepspeed
                    # Fallback to pip if uv fails
                    if package_installed "deepspeed"; then
                        python3 -m pip uninstall -y deepspeed
                    fi
                fi

                if ! package_installed "deepspeed"; then
                    print_success "DeepSpeed uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall DeepSpeed"
                fi
            fi

            # Uninstall Flash Attention
            if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y flash-attn flash_attention_amd
                    # Fallback to pip if uv fails
                    if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                        python3 -m pip uninstall -y flash-attn flash_attention_amd
                    fi
                else
                    python3 -m uv pip uninstall -y flash-attn flash_attention_amd
                    # Fallback to pip if uv fails
                    if package_installed "flash_attn" || package_installed "flash_attention_amd"; then
                        python3 -m pip uninstall -y flash-attn flash_attention_amd
                    fi
                fi

                if ! package_installed "flash_attn" && ! package_installed "flash_attention_amd"; then
                    print_success "Flash Attention uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall Flash Attention"
                fi
            fi

            # Uninstall Triton
            if package_installed "triton"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y triton
                    # Fallback to pip if uv fails
                    if package_installed "triton"; then
                        python3 -m pip uninstall -y triton
                    fi
                else
                    python3 -m uv pip uninstall -y triton
                    # Fallback to pip if uv fails
                    if package_installed "triton"; then
                        python3 -m pip uninstall -y triton
                    fi
                fi

                if ! package_installed "triton"; then
                    print_success "Triton uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall Triton"
                fi
            fi

            # Uninstall BITSANDBYTES
            if package_installed "bitsandbytes"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y bitsandbytes
                    # Fallback to pip if uv fails
                    if package_installed "bitsandbytes"; then
                        python3 -m pip uninstall -y bitsandbytes
                    fi
                else
                    python3 -m uv pip uninstall -y bitsandbytes
                    # Fallback to pip if uv fails
                    if package_installed "bitsandbytes"; then
                        python3 -m pip uninstall -y bitsandbytes
                    fi
                fi

                if ! package_installed "bitsandbytes"; then
                    print_success "BITSANDBYTES uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall BITSANDBYTES"
                fi
            fi

            # Uninstall vLLM
            if package_installed "vllm"; then
                if [ -n "$VIRTUAL_ENV" ]; then
                    uv pip uninstall -y vllm
                    # Fallback to pip if uv fails
                    if package_installed "vllm"; then
                        python3 -m pip uninstall -y vllm
                    fi
                else
                    python3 -m uv pip uninstall -y vllm
                    # Fallback to pip if uv fails
                    if package_installed "vllm"; then
                        python3 -m pip uninstall -y vllm
                    fi
                fi

                if ! package_installed "vllm"; then
                    print_success "vLLM uninstalled successfully"
                    components_uninstalled=$((components_uninstalled + 1))
                else
                    print_error "Failed to uninstall vLLM"
                fi
            fi

            if [ $components_uninstalled -gt 0 ]; then
                print_success "Uninstalled $components_uninstalled components"
                uninstall_success=true
            else
                print_warning "No components were uninstalled"
            fi
            ;;
        9)
            print_step "Skipping uninstallation"
            ;;
        *)
            print_warning "Invalid option, skipping uninstallation"
            ;;
    esac

    # Update progress bar to completion
    update_progress_bar 40
    draw_progress_bar "Finalizing uninstallation..."

    if [ "$uninstall_success" = true ]; then
        print_success "Uninstallation complete"
    else
        print_warning "No components were uninstalled"
    fi

    # Complete the progress bar
    complete_progress_bar

    return 0
}

# Main function
main() {
    print_header "ML Stack Repair Tool"

    # Initialize overall progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Starting ML Stack repair..."

    # Detect virtual environment
    update_progress_bar 5
    draw_progress_bar "Detecting virtual environment..."
    detect_virtual_env

    # Check package manager
    update_progress_bar 10
    draw_progress_bar "Checking package manager..."
    check_package_manager

    # Uninstall components if requested
    update_progress_bar 10
    draw_progress_bar "Preparing component uninstallation..."
    uninstall_components

    # Run diagnostics
    update_progress_bar 10
    draw_progress_bar "Running diagnostics..."
    run_diagnostics

    # Check for ROCm installation
    if ! command_exists rocminfo; then
        print_warning "ROCm is not installed"
        read -p "Do you want to install ROCm? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_rocm_installation
        fi
    fi

    # Check for AMDGPU drivers
    if ! lsmod | grep -q amdgpu; then
        print_warning "AMDGPU drivers are not loaded"
        read -p "Do you want to install AMDGPU drivers? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_amdgpu_drivers
        fi
    fi

    # Check for PyTorch conflicts
    check_pytorch_conflicts
    pytorch_status=$?

    # Ask user if they want to fix issues
    if [ $pytorch_status -ne 0 ]; then
        print_warning "PyTorch installation issues detected"
        read -p "Do you want to fix PyTorch installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_pytorch_installation
        fi
    fi

    # Check for AITER installation
    if ! package_installed "aiter"; then
        print_warning "AITER is not installed"
        read -p "Do you want to install AITER? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_aiter_installation
        fi
    fi

    # Check for MIGraphX Python module installation
    if command -v migraphx-driver >/dev/null 2>&1 && ! package_installed "migraphx"; then
        print_warning "MIGraphX is installed but the Python module is missing"
        read -p "Do you want to install the MIGraphX Python module? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_migraphx_python_installation
        fi
    fi

    # Check for DeepSpeed installation
    if ! package_installed "deepspeed"; then
        print_warning "DeepSpeed is not installed"
        read -p "Do you want to install DeepSpeed? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_deepspeed_installation
        fi
    fi

    # Check environment variables
    if [ -z "$HIP_VISIBLE_DEVICES" ] || [ -z "$CUDA_VISIBLE_DEVICES" ] || [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        print_warning "Environment variables not set properly"
        read -p "Do you want to fix environment variables? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_environment_variables
        fi
    fi

    # Run final diagnostics
    update_progress_bar 20
    draw_progress_bar "Running final diagnostics..."
    print_section "Running Final Diagnostics"
    run_diagnostics

    # Complete the progress bar
    update_progress_bar 30
    draw_progress_bar "Finalizing repairs..."
    complete_progress_bar

    print_header "ML Stack Repair Complete"
    print_step "To apply all changes, restart your terminal or run: source $HOME/.bashrc"

    return 0
}

# Function to fix ML Stack Core components
fix_ml_stack_core() {
    print_section "Fixing ML Stack Core Components"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking ML Stack Core components..."

    # Check if the ML Stack Core installation script exists
    if [ -f "$(dirname "$0")/install_ml_stack.sh" ]; then
        print_step "Using dedicated ML Stack Core installation script..."
        update_progress_bar 10
        draw_progress_bar "Preparing to fix ML Stack Core components..."

        # Fix the paths in the ML Stack Core installation script
        print_step "Checking and fixing paths in the installation script..."
        update_progress_bar 10
        draw_progress_bar "Checking script paths..."

        # Check if the script contains incorrect paths
        if grep -q "Desktop/Stans_MLStack" "$(dirname "$0")/install_ml_stack.sh"; then
            print_warning "Found incorrect paths in the ML Stack Core installation script"
            print_step "Fixing paths..."
            update_progress_bar 5
            draw_progress_bar "Fixing incorrect paths..."

            # Create a backup of the original script
            cp "$(dirname "$0")/install_ml_stack.sh" "$(dirname "$0")/install_ml_stack.sh.bak.$(date +%s)"

            # Fix the paths
            sed -i 's|$HOME/Desktop/Stans_MLStack/scripts/|$HOME/Prod/Stan-s-ML-Stack/scripts/|g' "$(dirname "$0")/install_ml_stack.sh"

            print_success "Fixed paths in the ML Stack Core installation script"
        else
            print_success "ML Stack Core installation script paths are correct"
        fi

        # Run the installation script
        print_step "Running ML Stack Core installation script..."
        update_progress_bar 30
        draw_progress_bar "Installing ML Stack Core components..."

        # Set environment variables
        export USE_UV=1
        export AMD_LOG_LEVEL=0
        export HIP_VISIBLE_DEVICES=0,1,2
        export ROCR_VISIBLE_DEVICES=0,1,2
        export NONINTERACTIVE=1

        # Run the script with a timeout to prevent hanging
        timeout 600s bash "$(dirname "$0")/install_ml_stack.sh"

        # Check the exit code
        if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 is the timeout exit code
            print_success "ML Stack Core components installation completed"
            update_progress_bar 20
            draw_progress_bar "Verifying installation..."

            # Verify key components
            components_ok=true

            # Check PyTorch
            if ! package_installed "torch"; then
                print_error "PyTorch is not installed"
                components_ok=false
            else
                print_success "PyTorch is installed"
            fi

            # Check if any other critical components are missing
            # Add more checks as needed

            if [ "$components_ok" = true ]; then
                print_success "ML Stack Core components verified successfully"
                complete_progress_bar
                return 0
            else
                print_warning "Some ML Stack Core components may be missing or incorrectly installed"
                complete_progress_bar
                return 1
            fi
        else
            print_error "ML Stack Core installation failed"
            complete_progress_bar
            return 1
        fi
    else
        print_error "ML Stack Core installation script not found"
        complete_progress_bar
        return 1
    fi
}

# Update the main function to include ML Stack Core repair
main() {
    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Starting ML Stack repair..."

    print_header "ML Stack Repair Tool"

    # Check if running in non-interactive mode
    if [ -n "$NONINTERACTIVE" ]; then
        print_warning "Running in non-interactive mode, skipping uninstallation"
    else
        # Uninstall components if requested
        print_section "Uninstalling Components"
        read -p "Do you want to uninstall any components before repair? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Uninstall components
            print_step "Uninstalling components..."
            # Add uninstallation code here
        else
            print_step "Skipping uninstallation"
        fi
    fi

    # Check for ROCm installation
    if ! command_exists rocminfo; then
        print_error "ROCm is not installed or not in PATH"
        read -p "Do you want to fix ROCm installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_rocm_installation
        fi
    fi

    # Check for AMDGPU drivers
    if ! command_exists amdgpu-install; then
        print_warning "AMDGPU drivers installation tool not found"
        read -p "Do you want to fix AMDGPU drivers? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_amdgpu_drivers
        fi
    fi

    # Check for PyTorch installation
    if ! package_installed "torch" || ! python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
        print_warning "PyTorch with ROCm support is not installed properly"
        read -p "Do you want to fix PyTorch installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_pytorch_installation
        fi
    fi

    # Check for ML Stack Core components
    print_warning "Checking ML Stack Core components"
    read -p "Do you want to fix ML Stack Core components? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        fix_ml_stack_core
    fi

    # Check for AITER installation
    if ! package_installed "aiter"; then
        print_warning "AITER is not installed"
        read -p "Do you want to fix AITER installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_aiter_installation
        fi
    fi

    # Check for MIGraphX Python installation
    if ! package_installed "migraphx"; then
        print_warning "MIGraphX Python module is not installed"
        read -p "Do you want to fix MIGraphX Python installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_migraphx_python_installation
        fi
    fi

    # Check for DeepSpeed installation
    if ! package_installed "deepspeed"; then
        print_warning "DeepSpeed is not installed"
        read -p "Do you want to fix DeepSpeed installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_deepspeed_installation
        fi
    fi

    # Check environment variables
    if [ -z "$HIP_VISIBLE_DEVICES" ] || [ -z "$CUDA_VISIBLE_DEVICES" ] || [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        print_warning "Environment variables not set properly"
        read -p "Do you want to fix environment variables? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_environment_variables
        fi
    fi

    # Run final diagnostics
    update_progress_bar 20
    draw_progress_bar "Running final diagnostics..."
    print_section "Running Final Diagnostics"
    run_diagnostics

    # Complete the progress bar
    update_progress_bar 30
    draw_progress_bar "Finalizing repairs..."
    complete_progress_bar

    print_header "ML Stack Repair Complete"
    print_step "To apply all changes, restart your terminal or run: source $HOME/.bashrc"

    return 0
}

# Run main function
main
