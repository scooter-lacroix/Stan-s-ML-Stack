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
# AMDGPU Drivers Installation Script
# =============================================================================
# This script installs AMD GPU kernel drivers and userspace components.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
   █████╗ ███╗   ███╗██████╗  ██████╗ ██████╗ ██╗   ██╗    ██████╗ ██████╗ ██╗██╗   ██╗███████╗██████╗ ███████╗
  ██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██╔══██╗██║   ██║    ██╔══██╗██╔══██╗██║██║   ██║██╔════╝██╔══██╗██╔════╝
  ███████║██╔████╔██║██║  ██║██║  ███╗██████╔╝██║   ██║    ██║  ██║██████╔╝██║██║   ██║█████╗  ██████╔╝███████╗
  ██╔══██║██║╚██╔╝██║██║  ██║██║   ██║██╔═══╝ ██║   ██║    ██║  ██║██╔══██╗██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║
  ██║  ██║██║ ╚═╝ ██║██████╔╝╚██████╔╝██║     ╚██████╔╝    ██████╔╝██║  ██║██║ ╚████╔╝ ███████╗██║  ██║███████║
  ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚═╝      ╚═════╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝
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

# Function to print a clean separator line
print_separator() {
    echo -e "${BLUE}───────────────────────────────────────────────────────────${RESET}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main installation function
install_amdgpu_drivers() {
    print_header "AMDGPU Drivers Installation"
    
    # Check if AMDGPU drivers are already installed
    if lsmod | grep -q amdgpu; then
        print_warning "AMDGPU drivers are already loaded"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping AMDGPU drivers installation"
            return 0
        fi
    fi
    
    # Detect Ubuntu version
    print_section "Detecting System"
    ubuntu_version=$(lsb_release -rs)
    ubuntu_codename=$(lsb_release -cs)
    print_step "Ubuntu version: $ubuntu_version ($ubuntu_codename)"
    
    # Download and install amdgpu-install package
    print_section "Installing AMDGPU Drivers"
    print_step "Downloading amdgpu-install package..."
    
    # Use the latest version (6.4.60400-1)
    wget -q https://repo.radeon.com/amdgpu-install/6.4/ubuntu/$ubuntu_codename/amdgpu-install_6.4.60400-1_all.deb
    
    if [ $? -ne 0 ]; then
        print_error "Failed to download amdgpu-install package"
        return 1
    fi
    
    print_success "Downloaded amdgpu-install package"
    
    # Install the package
    print_step "Installing amdgpu-install package..."
    sudo apt install -y ./amdgpu-install_6.4.60400-1_all.deb
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install amdgpu-install package"
        return 1
    fi
    
    print_success "Installed amdgpu-install package"
    
    # Update package lists
    print_step "Updating package lists..."
    sudo apt update
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to update package lists, continuing anyway"
    else
        print_success "Updated package lists"
    fi
    
    # Install Linux headers
    print_step "Installing Linux headers..."
    sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to install Linux headers, continuing anyway"
    else
        print_success "Installed Linux headers"
    fi
    
    # Install AMDGPU DKMS
    print_step "Installing AMDGPU DKMS..."
    sudo apt install -y amdgpu-dkms
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install AMDGPU DKMS"
        return 1
    fi
    
    print_success "Installed AMDGPU DKMS"
    
    # Verify installation
    print_section "Verifying Installation"
    
    if lsmod | grep -q amdgpu; then
        print_success "AMDGPU drivers are loaded"
        
        # Check if we can detect GPUs
        print_step "Checking GPU detection..."
        if command_exists lspci; then
            gpu_count=$(lspci | grep -i "amd\|radeon\|advanced micro devices" | grep -i "vga\|3d\|display" | wc -l)
            
            if [ "$gpu_count" -gt 0 ]; then
                print_success "Detected $gpu_count AMD GPU(s)"
                
                # List GPUs
                lspci | grep -i "amd\|radeon\|advanced micro devices" | grep -i "vga\|3d\|display" | while read -r gpu; do
                    echo -e "  - $gpu"
                done
            else
                print_warning "No AMD GPUs detected by lspci"
            fi
        else
            print_warning "lspci not found, cannot detect GPUs"
        fi
    else
        print_error "AMDGPU drivers installation verification failed"
        return 1
    fi
    
    # Clean up
    print_step "Cleaning up..."
    rm -f amdgpu-install_6.4.60400-1_all.deb
    
    print_success "AMDGPU drivers installation completed successfully"
    print_warning "You may need to reboot for the drivers to take full effect"
    
    return 0
}

# Run the installation function
install_amdgpu_drivers
