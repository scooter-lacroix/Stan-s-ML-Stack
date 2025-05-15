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
# Package Manager Utilities
# =============================================================================
# This script provides utilities for working with Python package managers,
# particularly uv, which is the preferred package manager for the ML Stack.
#
# Date: $(date +"%Y-%m-%d")
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗  █████╗  ██████╗██╗  ██╗ █████╗  ██████╗ ███████╗    ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗
  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔════╝ ██╔════╝    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗
  ██████╔╝███████║██║     █████╔╝ ███████║██║  ███╗█████╗      ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝
  ██╔═══╝ ██╔══██║██║     ██╔═██╗ ██╔══██║██║   ██║██╔══╝      ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗
  ██║     ██║  ██║╚██████╗██║  ██╗██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║
  ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝

                                        Python Package Manager Utilities
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
RESET='\033[0m'

# Function to print colored messages
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}"
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to ensure uv is installed and available
ensure_uv_installed() {
    print_section "Checking uv package manager"

    # Check if uv is already installed
    if command_exists uv; then
        uv_version=$(uv --version 2>/dev/null)
        print_success "uv is already installed (version: $uv_version)"

        # Add uv to PATH if not already there
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            print_step "Adding uv to PATH..."
            export PATH="$HOME/.local/bin:$PATH"

            # Add to .bashrc if not already there
            if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.bashrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
                print_success "Added uv to PATH in .bashrc"
            fi
        fi

        # Create symlink in /usr/local/bin if needed (requires sudo)
        if ! command -v uv &>/dev/null; then
            print_step "Creating symlink to uv in /usr/local/bin..."
            if [ -f "$HOME/.local/bin/uv" ]; then
                sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
                print_success "Created symlink to uv in /usr/local/bin"
            fi
        fi

        # Verify uv is in PATH
        if ! command -v uv &>/dev/null; then
            print_warning "uv is still not in PATH, trying alternative approach..."

            # Find uv executable
            UV_PATH=$(find $HOME -name uv -type f -executable 2>/dev/null | head -n 1)

            if [ -n "$UV_PATH" ]; then
                print_step "Found uv at $UV_PATH"

                # Create alias
                alias uv="$UV_PATH"

                # Add alias to .bashrc if not already there
                if ! grep -q "alias uv=\"$UV_PATH\"" "$HOME/.bashrc"; then
                    echo "alias uv=\"$UV_PATH\"" >> "$HOME/.bashrc"
                    print_success "Added uv alias to .bashrc"
                fi

                print_success "uv is now available via alias"
            else
                print_error "Could not find uv executable"
            fi
        fi

        return 0
    fi

    print_warning "uv is not installed, installing now..."

    # Install uv
    print_step "Installing uv package manager..."

    # Check if pip is installed
    if ! command_exists pip3 && ! command_exists pip; then
        print_error "pip is not installed, cannot install uv"
        print_step "Installing pip..."
        sudo apt-get update
        sudo apt-get install -y python3-pip

        if ! command_exists pip3 && ! command_exists pip; then
            print_error "Failed to install pip, cannot install uv"
            return 1
        fi
    fi

    # Install uv
    print_step "Installing uv using pip..."
    python3 -m pip install uv

    # Check if installation was successful
    if command_exists uv; then
        uv_version=$(uv --version 2>/dev/null)
        print_success "uv installed successfully (version: $uv_version)"

        # Add uv to PATH if not already there
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            print_step "Adding uv to PATH..."
            export PATH="$HOME/.local/bin:$PATH"

            # Add to .bashrc if not already there
            if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.bashrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
                print_success "Added uv to PATH in .bashrc"
            fi
        fi

        # Create symlink in /usr/local/bin if needed (requires sudo)
        if ! command -v uv &>/dev/null; then
            print_step "Creating symlink to uv in /usr/local/bin..."
            if [ -f "$HOME/.local/bin/uv" ]; then
                sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
                print_success "Created symlink to uv in /usr/local/bin"
            fi
        fi

        # Verify uv is in PATH
        if ! command -v uv &>/dev/null; then
            print_warning "uv is still not in PATH, trying alternative approach..."

            # Find uv executable
            UV_PATH=$(find $HOME -name uv -type f -executable 2>/dev/null | head -n 1)

            if [ -n "$UV_PATH" ]; then
                print_step "Found uv at $UV_PATH"

                # Create alias
                alias uv="$UV_PATH"

                # Add alias to .bashrc if not already there
                if ! grep -q "alias uv=\"$UV_PATH\"" "$HOME/.bashrc"; then
                    echo "alias uv=\"$UV_PATH\"" >> "$HOME/.bashrc"
                    print_success "Added uv alias to .bashrc"
                fi

                print_success "uv is now available via alias"
            else
                print_error "Could not find uv executable"
            fi
        fi

        return 0
    else
        print_error "Failed to install uv"
        return 1
    fi
}

# Function to install a package using uv or pip
install_package() {
    local package_name=$1
    local package_version=$2
    local extra_args=$3

    print_section "Installing package: $package_name"

    # Check if package is already installed
    if package_installed "$package_name"; then
        print_warning "Package $package_name is already installed"
        return 0
    fi

    # Ensure uv is installed
    ensure_uv_installed

    # Install package
    if command_exists uv; then
        print_step "Installing $package_name using uv..."

        if [ -n "$package_version" ]; then
            print_step "Installing version $package_version..."
            uv pip install "$package_name==$package_version" $extra_args
        else
            uv pip install "$package_name" $extra_args
        fi
    else
        print_warning "uv is not available, falling back to pip..."

        if [ -n "$package_version" ]; then
            print_step "Installing version $package_version..."
            python3 -m pip install "$package_name==$package_version" $extra_args
        else
            python3 -m pip install "$package_name" $extra_args
        fi
    fi

    # Check if installation was successful
    if package_installed "$package_name"; then
        print_success "Package $package_name installed successfully"
        return 0
    else
        print_error "Failed to install package $package_name"
        return 1
    fi
}

# Function to uninstall a package
uninstall_package() {
    local package_name=$1

    print_section "Uninstalling package: $package_name"

    # Check if package is installed
    if ! package_installed "$package_name"; then
        print_warning "Package $package_name is not installed"
        return 0
    fi

    # Ensure uv is installed
    ensure_uv_installed

    # Uninstall package
    if command_exists uv; then
        print_step "Uninstalling $package_name using uv..."
        uv pip uninstall -y "$package_name"
    else
        print_warning "uv is not available, falling back to pip..."
        python3 -m pip uninstall -y "$package_name"
    fi

    # Check if uninstallation was successful
    if ! package_installed "$package_name"; then
        print_success "Package $package_name uninstalled successfully"
        return 0
    else
        print_error "Failed to uninstall package $package_name"
        return 1
    fi
}

# If script is run directly, show usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_header "Package Manager Utilities"
    echo "This script provides utilities for working with Python package managers."
    echo
    echo "Usage:"
    echo "  source $(basename "${BASH_SOURCE[0]}") # To load functions"
    echo "  ensure_uv_installed # To ensure uv is installed"
    echo "  install_package <package_name> [<package_version>] [<extra_args>] # To install a package"
    echo "  uninstall_package <package_name> # To uninstall a package"
    echo
fi
