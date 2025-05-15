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
# ROCm Installation Script
# =============================================================================
# This script installs AMD's ROCm platform for GPU computing.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗  ██████╗  ██████╗███╗   ███╗    ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     
  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     
  ██████╔╝██║   ██║██║     ██╔████╔██║    ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     
  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     
  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
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
install_rocm() {
    print_header "ROCm Installation"
    
    # Check if ROCm is already installed
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        fi
        
        if [ -n "$rocm_version" ]; then
            print_warning "ROCm is already installed (version: $rocm_version)"
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping ROCm installation"
                return 0
            fi
        fi
    fi
    
    # Detect Ubuntu version
    print_section "Detecting System"
    ubuntu_version=$(lsb_release -rs)
    ubuntu_codename=$(lsb_release -cs)
    print_step "Ubuntu version: $ubuntu_version ($ubuntu_codename)"
    
    # Download and install amdgpu-install package
    print_section "Installing ROCm"
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
    sudo_with_pass apt install -y ./amdgpu-install_6.4.60400-1_all.deb
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install amdgpu-install package"
        return 1
    fi
    
    print_success "Installed amdgpu-install package"
    
    # Update package lists
    print_step "Updating package lists..."
    sudo_with_pass apt update
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to update package lists, continuing anyway"
    else
        print_success "Updated package lists"
    fi
    
    # Install prerequisites
    print_step "Installing prerequisites..."
    sudo_with_pass apt install -y python3-setuptools python3-wheel
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to install some prerequisites, continuing anyway"
    else
        print_success "Installed prerequisites"
    fi
    
    # Add user to render and video groups
    print_step "Adding user to render and video groups..."
    sudo_with_pass usermod -a -G render,video $LOGNAME
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to add user to groups, continuing anyway"
    else
        print_success "Added user to render and video groups"
    fi
    
    # Install ROCm
    print_step "Installing ROCm..."
    sudo_with_pass apt install -y rocm
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install ROCm"
        return 1
    fi
    
    print_success "Installed ROCm"
    
    # Verify installation
    print_section "Verifying Installation"
    
    if command_exists rocminfo; then
        rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        fi
        
        if [ -n "$rocm_version" ]; then
            print_success "ROCm is installed (version: $rocm_version)"
        else
            print_warning "ROCm is installed but version could not be determined"
        fi
        
        # Check if ROCm can detect GPUs
        print_step "Checking GPU detection..."
        gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
        
        if [ "$gpu_count" -gt 0 ]; then
            print_success "Detected $gpu_count AMD GPU(s)"
            
            # List GPUs
            rocminfo 2>/dev/null | grep -A 5 "Device Type:.*GPU" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
                echo -e "  - $gpu"
            done
        else
            print_warning "No AMD GPUs detected by ROCm"
        fi
    else
        print_error "ROCm installation verification failed"
        return 1
    fi
    
    # Clean up
    print_step "Cleaning up..."
    rm -f amdgpu-install_6.4.60400-1_all.deb
    
    print_success "ROCm installation completed successfully"
    print_warning "You may need to log out and log back in for group changes to take effect"
    
    return 0
}

# Run the installation function
install_rocm
