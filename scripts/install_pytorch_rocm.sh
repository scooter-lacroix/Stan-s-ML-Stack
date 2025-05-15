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
# PyTorch with ROCm Installation Script
# =============================================================================
# This script installs PyTorch with ROCm support for AMD GPUs.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗    ██████╗  ██████╗  ██████╗███╗   ███╗
  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║    ██╔══██╗██╔═══██╗██╔════╝████╗ ████║
  ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║    ██████╔╝██║   ██║██║     ██╔████╔██║
  ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║    ██╔══██╗██║   ██║██║     ██║╚██╔╝██║
  ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║    ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║
  ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝
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

# Main installation function
install_pytorch_rocm() {
    print_header "PyTorch with ROCm Installation"

    # Check if PyTorch is already installed
    if package_installed "torch"; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)

        # Check if PyTorch has ROCm/HIP support
        if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
            print_warning "PyTorch with ROCm support is already installed (PyTorch $pytorch_version, ROCm $hip_version)"
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping PyTorch installation"
                return 0
            fi
        else
            print_warning "PyTorch is installed (version $pytorch_version) but without ROCm support"
            print_step "Will reinstall with ROCm support"
        fi
    fi

    # Check if ROCm is installed
    print_section "Checking ROCm Installation"

    if ! command_exists rocminfo; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
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
    print_section "Installing PyTorch with ROCm Support"

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

    # Extract ROCm major and minor version
    rocm_major_version=$(echo "$rocm_version" | cut -d '.' -f 1)
    rocm_minor_version=$(echo "$rocm_version" | cut -d '.' -f 2)

    # Uninstall existing PyTorch if it exists
    if package_installed "torch"; then
        print_step "Uninstalling existing PyTorch..."

        # Create a function to handle uv commands properly
        uv_pip_uninstall() {
            # Check if uv is available as a command
            if command -v uv &> /dev/null; then
                # Use uv directly as a command
                uv pip uninstall "$@"
            else
                # Fall back to pip
                python3 -m pip uninstall "$@"
            fi
        }

        # Uninstall using the wrapper function
        uv_pip_uninstall -y torch torchvision torchaudio

        if package_installed "torch"; then
            print_warning "Failed to uninstall PyTorch, continuing anyway"
        else
            print_success "Uninstalled existing PyTorch"
        fi
    fi

    # Install PyTorch with ROCm support
    print_step "Installing PyTorch with ROCm support..."

    # Create a function to handle uv commands properly
    uv_pip_install() {
        # Check if uv is available as a command
        if command -v uv &> /dev/null; then
            # Use uv directly as a command
            uv pip install "$@"
        else
            # Fall back to pip
            python3 -m pip install "$@"
        fi
    }

    # Use the appropriate PyTorch version based on ROCm version
    if [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 4 ]; then
        # For ROCm 6.4+, use nightly builds
        print_step "Using PyTorch nightly build for ROCm 6.4..."
        uv_pip_install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
    elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 3 ]; then
        # For ROCm 6.3, use stable builds
        print_step "Using PyTorch stable build for ROCm 6.3..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    elif [ "$rocm_major_version" -eq 6 ] && [ "$rocm_minor_version" -ge 0 ]; then
        # For ROCm 6.0-6.2, use stable builds for 6.2
        print_step "Using PyTorch stable build for ROCm 6.2..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    elif [ "$rocm_major_version" -eq 5 ]; then
        # For ROCm 5.x, use stable builds for 5.7
        print_step "Using PyTorch stable build for ROCm 5.7..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    else
        # Fallback to the latest stable ROCm version
        print_step "Using PyTorch stable build for ROCm 6.3 (fallback)..."
        uv_pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    fi

    # Verify installation
    print_section "Verifying Installation"

    if package_installed "torch"; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_success "PyTorch is installed (version: $pytorch_version)"

        # Check if PyTorch has ROCm/HIP support
        if python3 -c "import torch; print(hasattr(torch.version, 'hip'))" 2>/dev/null | grep -q "True"; then
            hip_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'None')" 2>/dev/null)
            print_success "PyTorch has ROCm/HIP support (version: $hip_version)"

            # Test GPU availability
            if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                print_success "GPU acceleration is available"

                # Get GPU count
                gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
                print_step "PyTorch detected $gpu_count GPU(s)"

                # List GPUs
                for i in $(seq 0 $((gpu_count-1))); do
                    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
                    echo "  - GPU $i: $gpu_name"
                done

                # Test a simple tensor operation
                print_step "Testing GPU tensor operations..."
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
    else
        print_error "PyTorch installation failed"
        return 1
    fi

    # Show a visually appealing completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║  ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗  ║
    ║  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║  ║
    ║  ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║  ║
    ║  ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║  ║
    ║  ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║  ║
    ║  ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝  ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  PyTorch with ROCm is now ready to use with your GPU.   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "PyTorch with ROCm installation completed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    echo -e "${GREEN}python3 -c \"import torch; print('PyTorch version:', torch.__version__); print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'); print('GPU available:', torch.cuda.is_available())\"${RESET}"
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

# Run the installation function
install_pytorch_rocm
