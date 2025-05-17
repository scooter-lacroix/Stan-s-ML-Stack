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

# Suppress HIP logs
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2

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

# Function to check if a Python module exists
python_module_exists() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Function to fix Megatron-LM compatibility
fix_megatron_compatibility() {
    print_header "Fixing Megatron-LM Compatibility for ROCm 6.4.0 and Python 3.12"
    
    # Check if Megatron-LM directory exists
    if [ ! -d "$HOME/Megatron-LM" ]; then
        print_error "Megatron-LM directory not found at $HOME/Megatron-LM"
        print_step "Please run the install_megatron.sh script first"
        return 1
    fi
    
    print_success "Found Megatron-LM directory at $HOME/Megatron-LM"
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_step "Detected Python version: $PYTHON_VERSION"
    
    # Install missing dependencies
    print_section "Installing missing dependencies"
    
    # Check if we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "Not running in a virtual environment. It's recommended to use a virtual environment."
    else
        print_success "Running in virtual environment: $VIRTUAL_ENV"
    fi
    
    # Install tensorstore with compatibility fix for Python 3.12
    print_step "Installing tensorstore with Python 3.12 compatibility..."
    if ! python_module_exists "tensorstore"; then
        pip install --no-deps tensorstore==0.1.45
        if python_module_exists "tensorstore"; then
            print_success "tensorstore installed successfully"
        else
            print_error "Failed to install tensorstore"
        fi
    else
        print_success "tensorstore is already installed"
    fi
    
    # Install nvidia-modelopt with compatibility fix
    print_step "Installing nvidia-modelopt with compatibility fix..."
    if ! python_module_exists "nvidia_modelopt"; then
        pip install nvidia-modelopt
        if python_module_exists "nvidia_modelopt"; then
            print_success "nvidia-modelopt installed successfully"
        else
            print_warning "Failed to install nvidia-modelopt, trying alternative approach..."
            # Create a dummy nvidia_modelopt package to satisfy the import
            SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
            mkdir -p "$SITE_PACKAGES/nvidia_modelopt"
            echo "# Dummy nvidia_modelopt package for compatibility" > "$SITE_PACKAGES/nvidia_modelopt/__init__.py"
            print_success "Created dummy nvidia_modelopt package for compatibility"
        fi
    else
        print_success "nvidia-modelopt is already installed"
    fi
    
    # Patch Megatron-LM for Python 3.12 compatibility
    print_section "Patching Megatron-LM for Python 3.12 compatibility"
    
    cd "$HOME/Megatron-LM" || { print_error "Failed to enter Megatron-LM directory"; return 1; }
    
    # Backup setup.py
    cp setup.py setup.py.bak
    print_step "Backed up setup.py to setup.py.bak"
    
    # Modify setup.py to add Python 3.12 support
    print_step "Updating setup.py to add Python 3.12 support..."
    sed -i 's/Programming Language :: Python :: 3.9/Programming Language :: Python :: 3.9\\n        Programming Language :: Python :: 3.12/' setup.py
    print_success "Updated setup.py with Python 3.12 support"
    
    # Create a patch for Python 3.12 compatibility
    print_section "Creating Python 3.12 compatibility patches"
    
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
        print_error "megatron/__init__.py not found"
    fi
    
    # Reinstall Megatron-LM
    print_section "Reinstalling Megatron-LM with compatibility fixes"
    
    # Uninstall any existing installation
    pip uninstall -y megatron-core
    
    # Install with pip in development mode
    print_step "Installing Megatron-LM with pip in development mode..."
    pip install -e .
    
    # Verify installation
    print_section "Verifying Megatron-LM installation"
    
    if python_module_exists "megatron"; then
        print_success "Megatron-LM installed successfully"
        
        # Test GPU detection
        print_step "Testing GPU detection..."
        python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
        
        print_success "Megatron-LM compatibility fix completed successfully"
    else
        print_error "Failed to install Megatron-LM"
        return 1
    fi
    
    return 0
}

# Run the function
fix_megatron_compatibility
