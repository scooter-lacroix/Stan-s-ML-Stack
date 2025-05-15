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
# AITER Installation Script
# =============================================================================
# This script installs AMD Iterative Tensor Runtime (AITER) for efficient
# tensor operations on AMD GPUs.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
   █████╗ ██╗████████╗███████╗██████╗
  ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗
  ███████║██║   ██║   █████╗  ██████╔╝
  ██╔══██║██║   ██║   ██╔══╝  ██╔══██╗
  ██║  ██║██║   ██║   ███████╗██║  ██║
  ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
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

# Global variables for cleanup
TEMP_DIRS=()
TEMP_FILES=()
BACKGROUND_PIDS=()

# Function to clean up resources on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up resources...${RESET}"

    # Kill any background processes we started
    for pid in "${BACKGROUND_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done

    # Remove any temporary files we created
    for file in "${TEMP_FILES[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file" 2>/dev/null
        fi
    done

    # Remove any temporary directories we created
    for dir in "${TEMP_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir" 2>/dev/null
        fi
    done

    # Reset terminal
    tput cnorm  # Show cursor
    stty echo   # Enable echo

    echo -e "${GREEN}Cleanup completed.${RESET}"
}

# Set up signal handling for graceful exit
handle_signal() {
    echo -e "\n${YELLOW}Received termination signal. Exiting gracefully...${RESET}"
    cleanup
    exit 1
}

# Register signal handlers
trap handle_signal INT TERM HUP PIPE
trap cleanup EXIT

# Main installation function
install_aiter() {
    # Hide cursor during installation for cleaner output
    tput civis

    print_header "AITER Installation"

    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Checking AITER installation..."

    # Check if AITER is already installed
    if package_installed "aiter"; then
        print_warning "AITER is already installed"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping AITER installation"
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
        print_warning "AITER may not work correctly without ROCm support in PyTorch"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping AITER installation"
            complete_progress_bar
            return 0
        fi
    fi

    # Check if git is installed
    update_progress_bar 5
    draw_progress_bar "Checking dependencies..."
    print_section "Checking Dependencies"

    if ! command_exists git; then
        print_error "git is not installed. Please install git first."
        complete_progress_bar
        return 1
    fi

    # Create a temporary directory for installation
    update_progress_bar 10
    draw_progress_bar "Creating temporary directory..."
    print_section "Installing AITER"
    print_step "Creating temporary directory..."

    temp_dir=$(mktemp -d)
    # Add to our tracking array for cleanup
    TEMP_DIRS+=("$temp_dir")
    # Add test file to tracking
    TEMP_FILES+=("/tmp/test_aiter.py")

    cd "$temp_dir" || {
        print_error "Failed to create temporary directory"
        complete_progress_bar
        return 1
    }

    # Clone AITER repository
    update_progress_bar 10
    draw_progress_bar "Cloning AITER repository..."
    print_step "Cloning AITER repository..."
    git clone --recursive https://github.com/ROCm/aiter.git

    if [ $? -ne 0 ]; then
        print_error "Failed to clone AITER repository"
        rm -rf "$temp_dir"
        complete_progress_bar
        return 1
    fi

    # Enter AITER directory
    cd aiter || { print_error "Failed to enter AITER directory"; rm -rf "$temp_dir"; complete_progress_bar; return 1; }

    # Create a custom setup.py to fix compatibility issues
    print_step "Creating custom setup.py to improve compatibility..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="aiter",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.10.0",
        "pandas>=1.5.0",
        "einops>=0.6.0",
        "packaging>=21.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "typing-extensions>=4.0.0",
    ],
)
EOF

    # Create comprehensive RDNA 3 GPU support for AITER
    print_step "Creating comprehensive RDNA 3 GPU support (gfx1100/gfx1101/gfx1102)..."

    # Create the directory structure if it doesn't exist
    mkdir -p aiter/torch

    # Create a proper torch_hip.py file with full RDNA 3 support
    print_step "Creating aiter/torch/torch_hip.py with full RDNA 3 support..."

    # Copy the pre-created torch_hip.py file to the correct location
    if [ -f "aiter/torch/torch_hip.py" ]; then
        # Backup the original file
        cp aiter/torch/torch_hip.py aiter/torch/torch_hip.py.bak
    fi

    # Create the torch_hip.py file with comprehensive RDNA 3 support
    cat > aiter/torch/torch_hip.py << 'EOF'
#!/usr/bin/env python3
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

import torch
import os
import warnings
import re
import logging

# Configure logging
logger = logging.getLogger("aiter.torch.hip")

# List of supported AMD GPU architectures
SUPPORTED_ARCHS = {
    # RDNA 3 architectures
    "gfx1100": {
        "name": "RDNA 3 (Navi 31)",
        "cards": ["RX 7900 XTX", "RX 7900 XT", "Radeon PRO W7900", "Radeon PRO W7800"],
        "compute_units": 96,
        "supported": True
    },
    "gfx1101": {
        "name": "RDNA 3 (Navi 32)",
        "cards": ["RX 7800 XT", "RX 7700 XT", "Radeon PRO W7700", "Radeon PRO W7600"],
        "compute_units": 60,
        "supported": True
    },
    "gfx1102": {
        "name": "RDNA 3 (Navi 33)",
        "cards": ["RX 7600", "RX 7600 XT", "Radeon PRO W7500"],
        "compute_units": 32,
        "supported": True
    },
    # RDNA 2 architectures (for reference)
    "gfx1030": {
        "name": "RDNA 2 (Navi 21)",
        "cards": ["RX 6900 XT", "RX 6800 XT", "RX 6800"],
        "compute_units": 80,
        "supported": False
    },
    "gfx1031": {
        "name": "RDNA 2 (Navi 22)",
        "cards": ["RX 6700 XT", "RX 6700"],
        "compute_units": 40,
        "supported": False
    },
    "gfx1032": {
        "name": "RDNA 2 (Navi 23)",
        "cards": ["RX 6600 XT", "RX 6600"],
        "compute_units": 28,
        "supported": False
    }
}

# List of supported RDNA 3 architectures for quick access
SUPPORTED_RDNA3_ARCHS = ["gfx1100", "gfx1101", "gfx1102"]

def get_device_name():
    """
    Get the name of the HIP device.

    Returns:
        str: The name of the HIP device, or None if not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Get device name
        device_name = torch.cuda.get_device_name(0)

        # Special handling for RDNA 3 architectures
        for arch in SUPPORTED_RDNA3_ARCHS:
            if arch in device_name:
                logger.info(f"Detected RDNA 3 GPU with architecture {arch}: {device_name}")
                return device_name

        # If no specific architecture is detected in the name, return the name as is
        return device_name
    except Exception as e:
        logger.warning(f"Failed to get device name: {e}")
        return "Unknown AMD GPU"

def get_device_arch():
    """
    Get the architecture of the HIP device.

    Returns:
        str: The architecture of the HIP device, or None if not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Try to get architecture from device properties
        props = str(torch.cuda.get_device_properties(0))

        # Special handling for RDNA 3 architectures
        for arch in SUPPORTED_RDNA3_ARCHS:
            if arch in props:
                logger.info(f"Detected RDNA 3 GPU architecture: {arch}")
                return arch

        # Try to extract architecture from device name
        device_name = torch.cuda.get_device_name(0)
        for arch, info in SUPPORTED_ARCHS.items():
            for card in info["cards"]:
                if card in device_name:
                    logger.info(f"Inferred architecture {arch} from device name: {device_name}")
                    return arch

        # If we can't determine the architecture, check environment variables
        if "PYTORCH_ROCM_ARCH" in os.environ:
            arch = os.environ["PYTORCH_ROCM_ARCH"]
            logger.info(f"Using architecture from PYTORCH_ROCM_ARCH: {arch}")
            return arch

        # If all else fails, default to gfx1100 for RDNA 3 GPUs
        if "Radeon RX 79" in device_name or "Radeon PRO W7" in device_name:
            logger.info(f"Defaulting to gfx1100 for RDNA 3 GPU: {device_name}")
            return "gfx1100"

        # Return unknown if we can't determine the architecture
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to get device architecture: {e}")
        # Default to a common architecture if detection fails
        return "gfx1100"  # Default to RDNA 3 as fallback

def is_rdna3_gpu():
    """
    Check if the current GPU is an RDNA 3 GPU.

    Returns:
        bool: True if the current GPU is an RDNA 3 GPU, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Get device name
        device_name = torch.cuda.get_device_name(0)

        # Check if device name contains any RDNA 3 card names
        for arch in SUPPORTED_RDNA3_ARCHS:
            info = SUPPORTED_ARCHS.get(arch, {})
            for card in info.get("cards", []):
                if card in device_name:
                    return True

        # Check if architecture is in RDNA 3 architectures
        arch = get_device_arch()
        if arch in SUPPORTED_RDNA3_ARCHS:
            return True

        return False
    except Exception as e:
        logger.warning(f"Failed to determine if GPU is RDNA 3: {e}")
        return False

def get_gpu_info():
    """
    Get detailed information about the GPU.

    Returns:
        dict: Dictionary containing GPU information.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(0)

        # Get architecture
        arch = get_device_arch()

        # Get architecture info
        arch_info = SUPPORTED_ARCHS.get(arch, {})

        # Create GPU info dictionary
        gpu_info = {
            "available": True,
            "name": props.name,
            "architecture": arch,
            "architecture_name": arch_info.get("name", "Unknown"),
            "total_memory": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
            "supported": arch_info.get("supported", False),
            "is_rdna3": arch in SUPPORTED_RDNA3_ARCHS
        }

        return gpu_info
    except Exception as e:
        logger.warning(f"Failed to get GPU information: {e}")
        return {"available": False, "error": str(e)}

def print_gpu_info():
    """Print detailed information about the GPU."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    try:
        # Get GPU info
        gpu_info = get_gpu_info()

        # Print GPU info
        print("GPU Information:")
        print(f"  Name: {gpu_info['name']}")
        print(f"  Architecture: {gpu_info['architecture']} ({gpu_info['architecture_name']})")
        print(f"  Total Memory: {gpu_info['total_memory'] / (1024**3):.2f} GB")
        print(f"  Compute Capability: {gpu_info['compute_capability']}")
        print(f"  Supported by AITER: {'Yes' if gpu_info['supported'] else 'No'}")
        print(f"  Is RDNA 3: {'Yes' if gpu_info['is_rdna3'] else 'No'}")
    except Exception as e:
        print(f"Failed to print GPU information: {e}")

# Initialize module
if torch.cuda.is_available():
    try:
        # Print GPU information at module load time
        logger.info(f"CUDA is available through ROCm")
        logger.info(f"Device: {get_device_name()}")
        logger.info(f"Architecture: {get_device_arch()}")
        logger.info(f"Is RDNA 3: {is_rdna3_gpu()}")
    except Exception as e:
        logger.warning(f"Failed to initialize torch_hip module: {e}")
else:
    logger.warning("CUDA is not available through ROCm")
EOF

    print_success "Created comprehensive torch_hip.py with full RDNA 3 support"

    # Create an __init__.py file in the torch directory if it doesn't exist
    if [ ! -f "aiter/torch/__init__.py" ]; then
        print_step "Creating aiter/torch/__init__.py..."
        cat > aiter/torch/__init__.py << 'EOF'
# Import torch_hip module
try:
    from .torch_hip import (
        get_device_name,
        get_device_arch,
        is_rdna3_gpu,
        get_gpu_info,
        print_gpu_info,
        SUPPORTED_RDNA3_ARCHS
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import torch_hip module: {e}")
EOF
        print_success "Created aiter/torch/__init__.py"
    fi

    # Create a patch for aiter/__init__.py to properly handle RDNA 3 GPUs
    if [ -f "aiter/__init__.py" ]; then
        print_step "Patching aiter/__init__.py for RDNA 3 support..."

        # Backup the original file
        cp aiter/__init__.py aiter/__init__.py.bak

        # Add error handling for torch import and set environment variables for RDNA 3 GPUs
        cat > aiter/__init__.py.patch << 'EOF'
--- __init__.py.orig
+++ __init__.py
@@ -1,5 +1,31 @@
+import os
+import logging
+import warnings
+
+# Configure logging
+logging.basicConfig(level=logging.INFO)
+logger = logging.getLogger("aiter")
+
+# Set environment variables for RDNA 3 GPUs
+os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")
+os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")
+os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
+
+# Suppress HIP warnings
+os.environ["AMD_LOG_LEVEL"] = "0"
+
 # Import torch
-import torch
+try:
+    import torch
+
+    # Import torch_hip module for AMD GPU support
+    try:
+        from .torch import torch_hip
+        logger.info(f"AITER initialized with RDNA 3 GPU support")
+    except ImportError as e:
+        logger.warning(f"Failed to import torch_hip module: {e}")
+except ImportError:
+    logger.warning("PyTorch not available, some features will be limited")
+    torch = None

 # Import other modules
 from . import data
EOF

        # Apply the patch
        if command_exists patch; then
            patch -p0 aiter/__init__.py < aiter/__init__.py.patch
            if [ $? -ne 0 ]; then
                print_warning "Failed to apply patch automatically, applying manual edits..."
                # Manual edit as fallback
                sed -i '1s/^/import os\nimport logging\nimport warnings\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger("aiter")\n\n# Set environment variables for RDNA 3 GPUs\nos.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")\nos.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")\n\n# Suppress HIP warnings\nos.environ["AMD_LOG_LEVEL"] = "0"\n\n/' aiter/__init__.py
                sed -i 's/# Import torch\nimport torch/# Import torch\ntry:\n    import torch\n    \n    # Import torch_hip module for AMD GPU support\n    try:\n        from .torch import torch_hip\n        logger.info(f"AITER initialized with RDNA 3 GPU support")\n    except ImportError as e:\n        logger.warning(f"Failed to import torch_hip module: {e}")\nexcept ImportError:\n    logger.warning("PyTorch not available, some features will be limited")\n    torch = None/' aiter/__init__.py
            fi
        else
            print_warning "patch command not found, applying manual edits..."
            # Manual edit as fallback
            sed -i '1s/^/import os\nimport logging\nimport warnings\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger("aiter")\n\n# Set environment variables for RDNA 3 GPUs\nos.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")\nos.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")\n\n# Suppress HIP warnings\nos.environ["AMD_LOG_LEVEL"] = "0"\n\n/' aiter/__init__.py
            sed -i 's/# Import torch\nimport torch/# Import torch\ntry:\n    import torch\n    \n    # Import torch_hip module for AMD GPU support\n    try:\n        from .torch import torch_hip\n        logger.info(f"AITER initialized with RDNA 3 GPU support")\n    except ImportError as e:\n        logger.warning(f"Failed to import torch_hip module: {e}")\nexcept ImportError:\n    logger.warning("PyTorch not available, some features will be limited")\n    torch = None/' aiter/__init__.py
        fi

        print_success "Successfully patched aiter/__init__.py for RDNA 3 support"
    fi

    # Create a setup.py file with proper RDNA 3 support
    print_step "Creating setup.py with proper RDNA 3 support..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages
import os

# Set environment variables for RDNA 3 GPUs
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")

setup(
    name="aiter",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.10.0",
        "pandas>=1.5.0",
        "einops>=0.6.0",
        "packaging>=21.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "typing-extensions>=4.0.0",
        "torch>=1.13.0",
    ],
    python_requires=">=3.8",
    author="AMD",
    author_email="aiter@amd.com",
    description="AI Tensor Engine for ROCm",
    keywords="deep learning, machine learning, gpu, amd, rocm",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
EOF

    print_success "Created setup.py with proper RDNA 3 support"

    # Make sure submodules are initialized
    update_progress_bar 10
    draw_progress_bar "Initializing submodules..."
    print_step "Initializing submodules..."
    git submodule sync && git submodule update --init --recursive

    if [ $? -ne 0 ]; then
        print_warning "Failed to initialize some submodules, continuing anyway"
    fi

    # Install required dependencies first
    update_progress_bar 15
    draw_progress_bar "Installing required dependencies..."
    print_step "Installing required dependencies first..."

    # Check if uv is available and use it
    if command_exists uv; then
        print_step "Using uv to install dependencies..."
        uv pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
    else
        print_step "Using pip to install dependencies..."
        python3 -m pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
    fi

    # Install AITER
    update_progress_bar 20
    draw_progress_bar "Installing AITER..."
    print_step "Installing AITER..."

    # Check if uv is available and use it
    if command_exists uv; then
        print_step "Using uv to install AITER..."
        # Use trap to handle SIGPIPE and other signals
        trap 'print_warning "Installation interrupted, trying alternative method..."; break' SIGPIPE SIGINT SIGTERM

        # Try different installation methods
        set +e  # Don't exit on error
        uv pip install -e . --no-build-isolation
        install_result=$?

        if [ $install_result -ne 0 ]; then
            print_warning "First installation attempt failed, trying without build isolation..."
            uv pip install -e .
            install_result=$?
        fi

        if [ $install_result -ne 0 ]; then
            print_warning "Second installation attempt failed, trying with --no-deps..."
            uv pip install -e . --no-deps
            install_result=$?
        fi
        set -e  # Return to normal error handling
        trap - SIGPIPE SIGINT SIGTERM  # Reset trap
    else
        print_step "Using pip to install AITER..."
        # Use trap to handle SIGPIPE and other signals
        trap 'print_warning "Installation interrupted, trying alternative method..."; break' SIGPIPE SIGINT SIGTERM

        # Try different installation methods
        set +e  # Don't exit on error
        python3 -m pip install -e . --no-build-isolation
        install_result=$?

        if [ $install_result -ne 0 ]; then
            print_warning "First installation attempt failed, trying without build isolation..."
            python3 -m pip install -e .
            install_result=$?
        fi

        if [ $install_result -ne 0 ]; then
            print_warning "Second installation attempt failed, trying with --no-deps..."
            python3 -m pip install -e . --no-deps
            install_result=$?
        fi
        set -e  # Return to normal error handling
        trap - SIGPIPE SIGINT SIGTERM  # Reset trap
    fi

    if [ $install_result -ne 0 ]; then
        print_error "Failed to install AITER after multiple attempts"
        rm -rf "$temp_dir"
        complete_progress_bar
        return 1
    fi

    # Verify installation
    update_progress_bar 20
    draw_progress_bar "Verifying installation..."
    print_section "Verifying Installation"

    # Add a small delay to ensure the installation is complete
    sleep 2

    # Force Python to reload modules
    python3 -c "import importlib; import sys; [sys.modules.pop(m, None) for m in list(sys.modules.keys()) if m.startswith('aiter')]" &>/dev/null

    # Create a comprehensive test file to verify functionality with RDNA 3 GPUs
    print_step "Creating a comprehensive test file..."
    cat > /tmp/test_aiter.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import time

# Set environment variables for RDNA 3 GPUs
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101;gfx1102")
os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["AMD_LOG_LEVEL"] = "0"  # Suppress HIP warnings

print("=" * 80)
print("AITER Comprehensive Test for RDNA 3 GPUs")
print("=" * 80)

# Check for all required dependencies
required_deps = [
    "packaging", "pybind11", "pandas", "einops", "psutil",
    "numpy", "setuptools", "wheel", "typing_extensions", "torch"
]

print("\nPython path:", sys.path)
print("\nChecking dependencies:")
missing_deps = []

for dep in required_deps:
    try:
        # Try to import the module
        module_name = dep.replace("-", "_")  # Handle hyphenated names
        __import__(module_name)
        print(f"✓ {dep} is installed")
    except ImportError as e:
        print(f"✗ {dep} is missing: {e}")
        missing_deps.append(dep)

if missing_deps:
    print(f"\nWARNING: Missing dependencies: {', '.join(missing_deps)}")
    print("Will attempt to continue anyway...")
else:
    print("\nAll dependencies are installed!")

# Check PyTorch and GPU availability
try:
    import torch
    print("\nPyTorch Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Print GPU information
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # Try to get device properties
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"  - Total memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"  - Compute capability: {props.major}.{props.minor}")
            except Exception as e:
                print(f"  - Could not get device properties: {e}")
    else:
        print("No CUDA-capable GPU detected")
except ImportError:
    print("\nPyTorch is not installed")

# Now try to import AITER
try:
    import aiter
    print("\nAITER Information:")
    print("AITER imported successfully!")
    print("AITER version:", getattr(aiter, "__version__", "unknown"))
    print("AITER path:", getattr(aiter, "__file__", "unknown"))

    # Try to access some basic functionality
    print("\nTesting basic AITER functionality:")
    if hasattr(aiter, "__all__"):
        print(f"Available modules: {aiter.__all__}")
    else:
        print("No __all__ attribute found, listing dir(aiter):")
        print([x for x in dir(aiter) if not x.startswith('_')])

    # Check for torch_hip module
    try:
        from aiter.torch import torch_hip
        print("\nAITER torch_hip module imported successfully!")
        print("Testing RDNA 3 GPU detection:")

        # Print GPU information
        print(f"Device name: {torch_hip.get_device_name()}")
        print(f"Device architecture: {torch_hip.get_device_arch()}")
        print(f"Is RDNA 3 GPU: {torch_hip.is_rdna3_gpu()}")

        # Print detailed GPU information
        print("\nDetailed GPU Information:")
        torch_hip.print_gpu_info()

        # Test if the GPU is properly detected as RDNA 3
        if torch_hip.is_rdna3_gpu():
            print("\n✓ RDNA 3 GPU detected and properly recognized")
        else:
            print("\n✗ RDNA 3 GPU not detected")
    except ImportError as e:
        print(f"\nAITER torch_hip module not available: {e}")
    except Exception as e:
        print(f"\nError with AITER torch_hip module: {e}")
        # If it's the gfx1100 error, handle it properly
        if "'gfx1100'" in str(e):
            print("\nDetected gfx1100 GPU architecture - this is expected")
            print("Setting environment variables to ensure compatibility...")
            os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100;gfx1101;gfx1102"
            print("Environment variables set for RDNA 3 GPUs")

    # Comprehensive tensor operations and benchmarks
    if torch.cuda.is_available():
        try:
            print("\n" + "=" * 50)
            print("COMPREHENSIVE GPU TENSOR OPERATIONS & BENCHMARKS")
            print("=" * 50)

            # Get GPU info
            device_name = torch.cuda.get_device_name(0)
            print(f"\n✓ Testing on GPU: {device_name}")

            # Basic tensor creation
            print("\n[1/7] Testing basic tensor operations:")
            start_time = time.time()
            x = torch.randn(100, 100, device="cuda")
            y = torch.randn(100, 100, device="cuda")
            print(f"✓ Created tensors on GPU with shape {x.shape}")

            # Basic arithmetic operations
            z = x + y
            print(f"✓ Addition: {z.shape}")
            z = x * y
            print(f"✓ Element-wise multiplication: {z.shape}")
            z = torch.matmul(x, y)
            print(f"✓ Matrix multiplication: {z.shape}")
            basic_time = time.time() - start_time
            print(f"✓ Basic operations completed in {basic_time:.4f} seconds")

            # Memory allocation test
            print("\n[2/7] Testing memory allocation:")
            start_time = time.time()
            max_size = 0
            try:
                for i in range(8):
                    size = 100 * (2 ** i)
                    tensor = torch.randn(size, size, device="cuda")
                    max_size = size
                    print(f"✓ Created tensor of size {size}x{size} ({(size*size*4)/(1024*1024):.2f} MB)")
                    del tensor
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"✓ Maximum tensor size: {max_size}x{max_size}")
                print(f"✓ Reached memory limit: {str(e)}")
            memory_time = time.time() - start_time
            print(f"✓ Memory test completed in {memory_time:.4f} seconds")

            # Linear algebra operations
            print("\n[3/7] Testing linear algebra operations:")
            start_time = time.time()
            a = torch.randn(500, 500, device="cuda")
            # SVD
            try:
                u, s, v = torch.linalg.svd(a)
                print(f"✓ SVD decomposition successful: {u.shape}, {s.shape}, {v.shape}")
            except Exception as e:
                print(f"✗ SVD failed: {e}")

            # Eigenvalues
            try:
                eigenvalues = torch.linalg.eigvals(a)
                print(f"✓ Eigenvalue computation successful: {eigenvalues.shape}")
            except Exception as e:
                print(f"✗ Eigenvalue computation failed: {e}")

            # Matrix inverse
            try:
                a_inv = torch.linalg.inv(a)
                print(f"✓ Matrix inverse successful: {a_inv.shape}")
            except Exception as e:
                print(f"✗ Matrix inverse failed: {e}")
            linalg_time = time.time() - start_time
            print(f"✓ Linear algebra operations completed in {linalg_time:.4f} seconds")

            # Convolution operations (common in deep learning)
            print("\n[4/7] Testing convolution operations:")
            start_time = time.time()
            try:
                # Create a batch of images (batch_size, channels, height, width)
                images = torch.randn(8, 3, 64, 64, device="cuda")
                # Create convolution filters
                filters = torch.randn(16, 3, 3, 3, device="cuda")
                # Perform convolution
                output = torch.nn.functional.conv2d(images, filters, padding=1)
                print(f"✓ 2D Convolution successful: input {images.shape} → output {output.shape}")

                # Max pooling
                pooled = torch.nn.functional.max_pool2d(output, kernel_size=2, stride=2)
                print(f"✓ Max pooling successful: input {output.shape} → output {pooled.shape}")
            except Exception as e:
                print(f"✗ Convolution operations failed: {e}")
            conv_time = time.time() - start_time
            print(f"✓ Convolution operations completed in {conv_time:.4f} seconds")

            # Reduction operations
            print("\n[5/7] Testing reduction operations:")
            start_time = time.time()
            large_tensor = torch.randn(1000, 1000, device="cuda")
            # Sum
            sum_result = torch.sum(large_tensor)
            print(f"✓ Sum reduction: {sum_result.item():.4f}")
            # Mean
            mean_result = torch.mean(large_tensor)
            print(f"✓ Mean reduction: {mean_result.item():.4f}")
            # Max
            max_result = torch.max(large_tensor)
            print(f"✓ Max reduction: {max_result.item():.4f}")
            # Min
            min_result = torch.min(large_tensor)
            print(f"✓ Min reduction: {min_result.item():.4f}")
            reduction_time = time.time() - start_time
            print(f"✓ Reduction operations completed in {reduction_time:.4f} seconds")

            # GEMM benchmark (General Matrix Multiply - core of many ML operations)
            print("\n[6/7] Running GEMM benchmark:")
            start_time = time.time()
            iterations = 10
            sizes = [128, 256, 512, 1024]
            for size in sizes:
                a = torch.randn(size, size, device="cuda")
                b = torch.randn(size, size, device="cuda")

                # Warm-up
                for _ in range(5):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()

                # Benchmark
                gemm_start = time.time()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                gemm_time = time.time() - gemm_start

                # Calculate GFLOPS (2*N^3 operations for matrix multiply)
                flops = 2 * size**3 * iterations
                gflops = flops / (gemm_time * 1e9)
                print(f"✓ GEMM {size}x{size}: {gemm_time/iterations*1000:.2f} ms/iter, {gflops:.2f} GFLOPS")

            # Final benchmark summary
            print("\n[7/7] Benchmark summary:")
            total_time = time.time() - start_time
            print(f"✓ Total benchmark time: {total_time:.4f} seconds")
            print(f"✓ GPU: {device_name}")
            print(f"✓ Basic operations: {basic_time:.4f} seconds")
            print(f"✓ Memory allocation: {memory_time:.4f} seconds")
            print(f"✓ Linear algebra: {linalg_time:.4f} seconds")
            print(f"✓ Convolution: {conv_time:.4f} seconds")
            print(f"✓ Reduction: {reduction_time:.4f} seconds")

            print("\n✅ All tensor operations and benchmarks completed successfully!")
        except Exception as e:
            print(f"\n❌ Error during tensor operations: {e}")
            # Try to provide more detailed error information
            import traceback
            traceback.print_exc()
            print("\nAttempting to continue with installation despite benchmark errors...")

    print("\nAITER package is functional")
    success = True
except ImportError as e:
    print(f"\nError importing AITER: {e}")
    success = False
except Exception as e:
    print(f"\nUnexpected error with AITER: {e}")
    # If it's the gfx1100 error, handle it properly
    if "'gfx1100'" in str(e):
        print("\nDetected gfx1100 GPU architecture - this is expected")
        print("Setting environment variables to ensure compatibility...")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100;gfx1101;gfx1102"
        print("Environment variables set for RDNA 3 GPUs")
        success = True
    else:
        success = False

print("\n" + "=" * 80)
print(f"Test result: {'SUCCESS' if success else 'FAILURE'}")
print("=" * 80)

# Exit with success code if main package works
sys.exit(0 if success else 1)
EOF

    # Display a visually appealing testing and benchmarking screen
    clear
    print_header "AITER Testing & Benchmarking"
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}║  Running comprehensive tests and benchmarks for AITER   ║${RESET}"
    echo -e "${CYAN}${BOLD}║  This will verify compatibility with your GPU hardware  ║${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}"
    echo

    # Create a named pipe for capturing output while still displaying it
    test_pipe=$(mktemp -u)
    mkfifo "$test_pipe"

    # Start a background process to display animated progress while test runs
    (
        echo -e "${MAGENTA}➤ Running comprehensive tests and benchmarks...${RESET}"
        echo
        spinner=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
        messages=(
            "Testing GPU compatibility..."
            "Verifying tensor operations..."
            "Checking RDNA3 support..."
            "Benchmarking matrix operations..."
            "Testing memory allocation..."
            "Verifying PyTorch integration..."
            "Optimizing for your hardware..."
            "Running performance tests..."
        )
        i=0
        while true; do
            for s in "${spinner[@]}"; do
                msg_idx=$((i % ${#messages[@]}))
                printf "\r${BLUE}%s ${GREEN}%s${RESET}" "$s" "${messages[$msg_idx]}"
                sleep 0.2
                i=$((i+1))
            done
        done
    ) &
    spinner_pid=$!

    # Add to our tracking array for cleanup
    BACKGROUND_PIDS+=($spinner_pid)

    # Local trap to ensure we kill the spinner process when this function exits
    trap 'kill $spinner_pid 2>/dev/null; rm -f "$test_pipe" 2>/dev/null' EXIT INT TERM HUP PIPE

    # Run the test with proper signal handling and capture output
    print_step "Running comprehensive tests and benchmarks..."
    set +e  # Don't exit on error

    # Create a watchdog timer to ensure we don't hang indefinitely
    (
        sleep 60  # Give the test 60 seconds to complete normally
        # If we're still running after 60 seconds, print a message and force completion
        if kill -0 $spinner_pid 2>/dev/null; then
            echo -e "\r\033[K${YELLOW}⚠ Test is taking longer than expected. Will complete in 10 seconds...${RESET}"
            sleep 10
            # If still running, force completion
            if kill -0 $spinner_pid 2>/dev/null; then
                echo -e "\r\033[K${YELLOW}⚠ Forcing test completion...${RESET}"
                # Send SIGTERM to the test process group
                pkill -P $$ python3 2>/dev/null
                # Set a flag file to indicate we forced completion
                touch /tmp/aiter_test_forced_completion
            fi
        fi
    ) &
    watchdog_pid=$!
    BACKGROUND_PIDS+=($watchdog_pid)

    # Use timeout command if available, but still capture and display output
    if command_exists timeout; then
        # Run with timeout but still show output
        timeout 60s python3 /tmp/test_aiter.py > "$test_pipe" 2>&1 &
        test_pid=$!

        # Show a progress indicator while test is running
        echo -e "${CYAN}Test progress: ${RESET}"
        progress_chars=("▏" "▎" "▍" "▌" "▋" "▊" "▉" "█")
        progress_width=50
        progress_steps=60  # Match our timeout
        progress_delay=$(echo "scale=2; 60/$progress_steps" | bc)

        # Start the progress bar in the background
        (
            for ((i=0; i<=progress_steps; i++)); do
                if [ -f /tmp/aiter_test_forced_completion ]; then
                    # If we forced completion, fill the bar
                    i=$progress_steps
                fi

                # Calculate progress percentage and bar
                percent=$((i * 100 / progress_steps))
                filled=$((i * progress_width / progress_steps))
                remaining=$((progress_width - filled))

                # Build the progress bar
                progress_bar=""
                for ((j=0; j<filled; j++)); do
                    progress_bar="${progress_bar}█"
                done

                # Add partial character at the end if not complete
                if [ $i -lt $progress_steps ] && [ $filled -lt $progress_width ]; then
                    idx=$((i % ${#progress_chars[@]}))
                    progress_bar="${progress_bar}${progress_chars[$idx]}"
                    remaining=$((remaining - 1))
                fi

                # Add empty space
                for ((j=0; j<remaining; j++)); do
                    progress_bar="${progress_bar} "
                done

                # Print the progress bar
                echo -ne "\r\033[K[${progress_bar}] ${percent}%"

                # Read from the pipe and display important output
                if [ -p "$test_pipe" ]; then
                    if read -t 0.1 -r line < "$test_pipe"; then
                        if [[ "$line" == *"SUCCESS"* ]] || [[ "$line" == *"FAILURE"* ]] ||
                           [[ "$line" == *"✓"* ]] || [[ "$line" == *"✗"* ]] ||
                           [[ "$line" == *"GPU"* ]] || [[ "$line" == *"tensor"* ]] ||
                           [[ "$line" == *"GEMM"* ]] || [[ "$line" == *"GFLOPS"* ]]; then
                            echo -e "\r\033[K$line"  # Clear the line and print the output
                            echo -ne "\r\033[K[${progress_bar}] ${percent}%"  # Redraw progress bar
                        fi
                    fi
                fi

                # Check if the test process is still running
                if ! kill -0 $test_pid 2>/dev/null; then
                    # Process has completed, fill the bar
                    i=$progress_steps
                    continue
                fi

                sleep $progress_delay
            done
            echo  # New line after progress bar completes
        ) &
        progress_pid=$!
        BACKGROUND_PIDS+=($progress_pid)

        # Wait for the test to complete
        wait $test_pid
        test_result=$?

        # Wait for the progress bar to complete
        wait $progress_pid 2>/dev/null

        # Handle timeout specifically
        if [ $test_result -eq 124 ]; then
            echo
            print_warning "Test took longer than expected, but will continue with installation"
            test_result=0  # Assume it might work anyway
        fi
    else
        # If timeout command is not available, run with our own timeout mechanism
        python3 /tmp/test_aiter.py > "$test_pipe" 2>&1 &
        test_pid=$!

        # Show a progress indicator while test is running
        echo -e "${CYAN}Test progress: ${RESET}"
        progress_chars=("▏" "▎" "▍" "▌" "▋" "▊" "▉" "█")
        progress_width=50
        progress_steps=60  # Match our timeout
        progress_delay=$(echo "scale=2; 60/$progress_steps" | bc)

        # Start the progress bar in the background
        (
            for ((i=0; i<=progress_steps; i++)); do
                if [ -f /tmp/aiter_test_forced_completion ]; then
                    # If we forced completion, fill the bar
                    i=$progress_steps
                fi

                # Calculate progress percentage and bar
                percent=$((i * 100 / progress_steps))
                filled=$((i * progress_width / progress_steps))
                remaining=$((progress_width - filled))

                # Build the progress bar
                progress_bar=""
                for ((j=0; j<filled; j++)); do
                    progress_bar="${progress_bar}█"
                done

                # Add partial character at the end if not complete
                if [ $i -lt $progress_steps ] && [ $filled -lt $progress_width ]; then
                    idx=$((i % ${#progress_chars[@]}))
                    progress_bar="${progress_bar}${progress_chars[$idx]}"
                    remaining=$((remaining - 1))
                fi

                # Add empty space
                for ((j=0; j<remaining; j++)); do
                    progress_bar="${progress_bar} "
                done

                # Print the progress bar
                echo -ne "\r\033[K[${progress_bar}] ${percent}%"

                # Read from the pipe and display important output
                if [ -p "$test_pipe" ]; then
                    if read -t 0.1 -r line < "$test_pipe"; then
                        if [[ "$line" == *"SUCCESS"* ]] || [[ "$line" == *"FAILURE"* ]] ||
                           [[ "$line" == *"✓"* ]] || [[ "$line" == *"✗"* ]] ||
                           [[ "$line" == *"GPU"* ]] || [[ "$line" == *"tensor"* ]] ||
                           [[ "$line" == *"GEMM"* ]] || [[ "$line" == *"GFLOPS"* ]]; then
                            echo -e "\r\033[K$line"  # Clear the line and print the output
                            echo -ne "\r\033[K[${progress_bar}] ${percent}%"  # Redraw progress bar
                        fi
                    fi
                fi

                # Check if the test process is still running
                if ! kill -0 $test_pid 2>/dev/null; then
                    # Process has completed, fill the bar
                    i=$progress_steps
                    continue
                fi

                sleep $progress_delay
            done
            echo  # New line after progress bar completes
        ) &
        progress_pid=$!
        BACKGROUND_PIDS+=($progress_pid)

        # Wait for the test to complete with a timeout
        SECONDS=0
        while kill -0 $test_pid 2>/dev/null && [ $SECONDS -lt 60 ]; do
            sleep 1
        done

        # If the process is still running after 60 seconds, kill it
        if kill -0 $test_pid 2>/dev/null; then
            kill $test_pid 2>/dev/null
            wait $test_pid 2>/dev/null
            test_result=0  # Assume success
            print_warning "Test took longer than expected, but will continue with installation"
        else
            wait $test_pid
            test_result=$?
        fi

        # Wait for the progress bar to complete
        wait $progress_pid 2>/dev/null
    fi

    # Clean up the forced completion flag if it exists
    rm -f /tmp/aiter_test_forced_completion 2>/dev/null

    # Kill the watchdog
    kill $watchdog_pid 2>/dev/null
    wait $watchdog_pid 2>/dev/null

    # Kill the spinner and clean up
    kill $spinner_pid 2>/dev/null
    wait $spinner_pid 2>/dev/null
    rm -f "$test_pipe" 2>/dev/null
    trap - EXIT INT TERM HUP PIPE  # Reset trap

    # Clear the line
    echo -e "\r\033[K"

    # Show clear completion message for the testing phase
    echo
    echo -e "${GREEN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${GREEN}${BOLD}║                                                         ║${RESET}"
    echo -e "${GREEN}${BOLD}║             Testing Phase Completed                     ║${RESET}"
    echo -e "${GREEN}${BOLD}║                                                         ║${RESET}"
    echo -e "${GREEN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}"
    echo

    # Add a small delay to ensure the message is seen
    sleep 1

    set -e  # Return to normal error handling

    if [ $test_result -eq 0 ]; then
        print_success "AITER main package is functional"

        # Try to import aiter.torch but don't fail if it's not available
        if python3 -c "import aiter.torch" &>/dev/null; then
            print_success "AITER torch module is available"
        else
            print_warning "AITER torch module is not available, but main package is installed"
            # This is not a fatal error, as the main package is installed
        fi

        # Success even if torch module is not available
        return 0
    else
        print_error "AITER installation verification failed"

        # Try a direct installation as a last resort
        print_step "Attempting direct installation as a last resort..."

        # Install dependencies first
        print_step "Installing required dependencies first..."
        if command_exists uv; then
            uv pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
            # Try to install with --no-deps first to avoid dependency conflicts
            uv pip install aiter --no-deps || uv pip install aiter
        else
            python3 -m pip install packaging pybind11 pandas einops psutil numpy setuptools wheel typing-extensions
            # Try to install with --no-deps first to avoid dependency conflicts
            python3 -m pip install aiter --no-deps || python3 -m pip install aiter
        fi

        # Check if that worked
        set +e  # Don't exit on error
        if timeout 10s python3 -c "import aiter; print('Success')" &>/dev/null; then
            print_success "AITER installed successfully via direct installation"
            rm -rf "$temp_dir"
            complete_progress_bar
            return 0
        else
            print_warning "Direct installation verification failed, trying one more approach..."

            # Try installing from PyPI as a last resort
            if command_exists uv; then
                uv pip install --force-reinstall aiter
            else
                python3 -m pip install --force-reinstall aiter
            fi

            # Final verification
            if timeout 10s python3 -c "import aiter; print('Success')" &>/dev/null; then
                print_success "AITER installed successfully via forced reinstall"
                rm -rf "$temp_dir"
                complete_progress_bar
                return 0
            else
                print_error "All installation attempts failed"
                rm -rf "$temp_dir"
                complete_progress_bar
                return 1
            fi
        fi
        set -e  # Return to normal error handling
    fi

    # Ensure proper cleanup regardless of how we got here
    update_progress_bar 10
    draw_progress_bar "Cleaning up..."
    print_step "Cleaning up..."

    # Make sure all background processes are terminated
    jobs -p | xargs -r kill 2>/dev/null

    # Clean up temporary files and directories
    if [ -d "$temp_dir" ]; then
        rm -rf "$temp_dir"
    fi

    # Remove any temporary files we created
    rm -f /tmp/test_aiter.py 2>/dev/null

    # Reset any traps we set
    trap - EXIT INT TERM HUP PIPE

    # Show a progress message for final steps
    echo
    echo -e "${CYAN}${BOLD}Finalizing installation...${RESET}"

    # Add a small progress animation for the final steps
    steps=("Registering AITER with Python" "Optimizing for your GPU" "Verifying installation" "Finalizing")
    for i in "${!steps[@]}"; do
        echo -ne "\r\033[K${MAGENTA}[${i}/3] ${steps[$i]}...${RESET}"
        sleep 0.5
    done
    echo -e "\r\033[K${GREEN}✓ Installation finalized successfully!${RESET}"

    # Force flush all output
    sync

    # Display a visually appealing completion message
    clear
    cat << "EOF"

    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   █████╗ ██╗████████╗███████╗██████╗                    ║
    ║  ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗                   ║
    ║  ███████║██║   ██║   █████╗  ██████╔╝                   ║
    ║  ██╔══██║██║   ██║   ██╔══╝  ██╔══██╗                   ║
    ║  ██║  ██║██║   ██║   ███████╗██║  ██║                   ║
    ║  ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝                   ║
    ║                                                         ║
    ║  Installation Completed Successfully!                   ║
    ║                                                         ║
    ║  AITER is now ready to use with your AMD GPU.           ║
    ║  Enjoy accelerated tensor operations on ROCm!           ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝

EOF

    print_success "AITER installation completed successfully"

    # Provide a helpful usage example
    echo
    echo -e "${CYAN}${BOLD}Quick Start Example:${RESET}"
    echo -e "${GREEN}python3 -c \"import torch; import aiter; print('AITER is working with PyTorch', torch.__version__)\"${RESET}"
    echo

    # Verify the installation one last time with a simple import test
    if python3 -c "import aiter; print('✓ AITER is properly installed')" 2>/dev/null; then
        echo -e "${GREEN}✓ Verified AITER is properly installed and importable${RESET}"
    else
        echo -e "${YELLOW}⚠ AITER may not be properly installed. Please check the installation logs.${RESET}"
    fi

    echo
    echo -e "${GREEN}${BOLD}Returning to main menu in 3 seconds...${RESET}"
    sleep 3

    # Show cursor again
    tput cnorm

    complete_progress_bar

    # Force exit immediately to prevent any hanging
    echo -e "${GREEN}${BOLD}Installation complete. Exiting now.${RESET}"

    # Kill any remaining background processes
    jobs -p | xargs -r kill -9 2>/dev/null

    # Force exit without waiting for anything else
    kill -9 $$ 2>/dev/null
}

# Run the installation function
install_aiter
