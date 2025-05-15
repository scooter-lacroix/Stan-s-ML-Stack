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
# Enhanced ML Stack Environment Setup
# =============================================================================
# This script sets up an advanced environment for the ML Stack on AMD GPUs with
# automatic hardware detection, configuration, and optimization.
#
# WHEN TO USE THIS SCRIPT:
# - You have a complex GPU setup (multiple GPUs, mixed discrete/integrated)
# - You need automatic filtering of integrated GPUs (Raphael, etc.)
# - You want comprehensive system dependency checks
# - You need detailed environment configuration for optimal performance
# - You're experiencing issues with the basic setup script
#
# Key advantages over the basic setup_environment.sh:
# - Automatically detects and filters out integrated GPUs
# - Performs comprehensive system dependency checks
# - Configures environment variables based on detected hardware
# - Sets up performance optimizations for ROCm and PyTorch
# - Creates a more detailed environment file with conditional settings
# - Handles complex multi-GPU configurations more effectively
#
# Author: Stanley Chisango (Scooter Lacroix)
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                Enhanced ML Stack Environment Setup
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

# Function definitions
print_header() {
    echo -e "${BLUE}=== $1 ===${RESET}"
    echo
    # Flush stdout to ensure real-time output
    sleep 0.05
}

print_section() {
    echo -e "${CYAN}>>> $1${RESET}"
    # Flush stdout to ensure real-time output
    sleep 0.05
}

print_step() {
    echo -e "${YELLOW}>> $1${RESET}"
    # Flush stdout to ensure real-time output
    sleep 0.05
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
    # Flush stdout to ensure real-time output
    sleep 0.05
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
    # Flush stdout to ensure real-time output
    sleep 0.05
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
    # Flush stdout to ensure real-time output
    sleep 0.05
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect AMD GPUs
detect_amd_gpus() {
    print_section "Detecting AMD GPUs"

    # Check if lspci is available
    if ! command_exists lspci; then
        print_warning "lspci command not found. Installing pciutils..."
        sudo apt-get update && sudo apt-get install -y pciutils
    fi

    # Detect AMD GPUs using lspci
    print_step "Searching for AMD GPUs..."
    amd_gpus=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display')

    if [ -z "$amd_gpus" ]; then
        print_error "No AMD GPUs detected."
        return 1
    else
        print_success "AMD GPUs detected:"
        echo "$amd_gpus" | while read -r line; do
            echo "  - $line"
        done
    fi

    # Check if ROCm is installed
    if command_exists rocminfo; then
        print_success "ROCm is installed"
        print_step "ROCm version: $(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)"

        # Get GPU count from ROCm
        gpu_count=$(rocminfo | grep "GPU ID" | wc -l)
        print_step "ROCm detected $gpu_count GPU(s)"

        # List AMD GPUs from ROCm
        print_step "ROCm detected GPUs:"
        rocminfo | grep -A 1 "GPU ID" | grep "Marketing Name" | awk -F: '{print $2}' | while read -r gpu; do
            echo "  - $gpu"
        done

        # Check for specific GPU models
        if rocminfo | grep -q "Radeon RX 7900 XTX"; then
            print_success "Detected Radeon RX 7900 XTX"
            has_7900_xtx=true
        fi

        if rocminfo | grep -q "Radeon RX 7800 XT"; then
            print_success "Detected Radeon RX 7800 XT"
            has_7800_xt=true
        fi
    else
        print_warning "ROCm is not installed. Some features may not work correctly."
        print_step "Attempting to detect GPUs using other methods..."

        # Try to detect GPUs using other methods
        if command_exists glxinfo; then
            print_step "GPU information from glxinfo:"
            glxinfo | grep -i "OpenGL renderer" | awk -F: '{print $2}' | xargs
        elif command_exists clinfo; then
            print_step "GPU information from clinfo:"
            clinfo | grep -i "Device Name" | awk -F: '{print $2}' | xargs
        else
            print_warning "Could not detect detailed GPU information. Installing mesa-utils and clinfo..."
            sudo apt-get update && sudo apt-get install -y mesa-utils clinfo

            if command_exists glxinfo; then
                print_step "GPU information from glxinfo:"
                glxinfo | grep -i "OpenGL renderer" | awk -F: '{print $2}' | xargs
            fi

            if command_exists clinfo; then
                print_step "GPU information from clinfo:"
                clinfo | grep -i "Device Name" | awk -F: '{print $2}' | xargs
            fi
        fi

        # Estimate GPU count
        gpu_count=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display' | wc -l)
    fi

    # Set GPU count
    export GPU_COUNT=$gpu_count
    print_success "Detected $GPU_COUNT AMD GPU(s)"

    return 0
}

# Function to detect ROCm installation
detect_rocm() {
    print_section "Detecting ROCm Installation"

    # Check if ROCm is installed
    if command_exists rocminfo; then
        print_success "ROCm is installed"

        # Get ROCm version
        rocm_version=$(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$rocm_version" ]; then
            # Try alternative method to get ROCm version
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            if [ -z "$rocm_version" ]; then
                rocm_version="unknown"
            fi
        fi

        print_step "ROCm version: $rocm_version"
        export ROCM_VERSION=$rocm_version

        # Check ROCm path
        if [ -d "/opt/rocm" ]; then
            rocm_path="/opt/rocm"
        elif [ -d "/opt/rocm-$rocm_version" ]; then
            rocm_path="/opt/rocm-$rocm_version"
        else
            # Try to find ROCm path
            rocm_path=$(dirname $(which rocminfo))/..
        fi

        print_step "ROCm path: $rocm_path"
        export ROCM_PATH=$rocm_path

        # Check if user has proper permissions
        if groups | grep -q -E '(video|render|rocm)'; then
            print_success "User has proper permissions for ROCm"
        else
            print_warning "User may not have proper permissions for ROCm"
            print_step "Recommended: Add user to video and render groups:"
            print_step "  sudo usermod -a -G video,render $USER"
            print_step "  (Requires logout/login to take effect)"
        fi
    else
        print_warning "ROCm is not installed or not in PATH"

        # Check if ROCm is installed in common locations
        if [ -d "/opt/rocm" ]; then
            print_step "Found ROCm in /opt/rocm"
            rocm_path="/opt/rocm"
        else
            # Try to find any rocm installation
            rocm_dirs=$(ls -d /opt/rocm* 2>/dev/null)
            if [ -n "$rocm_dirs" ]; then
                rocm_path=$(echo "$rocm_dirs" | head -n 1)
                print_step "Found ROCm in $rocm_path"
            else
                print_error "Could not find ROCm installation"
                print_step "Please install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
                return 1
            fi
        fi

        # Try to get ROCm version
        rocm_version=$(echo "$rocm_path" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        if [ -z "$rocm_version" ]; then
            rocm_version="unknown"
        fi

        export ROCM_PATH=$rocm_path
        export ROCM_VERSION=$rocm_version

        print_step "Adding ROCm to PATH and LD_LIBRARY_PATH..."
        export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/hip/lib

        print_warning "Please install ROCm properly for full functionality"
    fi

    return 0
}

# Function to configure environment variables based on detected hardware
configure_environment_variables() {
    print_section "Configuring Environment Variables"

    # Set GPU device variables
    if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
        # Filter out integrated GPUs (Raphael and similar)
        print_step "Detecting discrete GPUs and filtering out integrated GPUs..."

        # Get GPU information from rocminfo if available
        if command_exists rocminfo; then
            # Get list of GPUs with their types
            gpu_info=$(rocminfo | grep -A 10 "GPU ID" | grep -E "GPU ID|Marketing Name|Device Type")

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
                    # Check if this is an integrated GPU (contains "Raphael" or other iGPU indicators)
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
                discrete_gpu_list=$(seq -s, 0 $((GPU_COUNT-1)))
            fi
        else
            # Fallback if rocminfo is not available
            print_warning "rocminfo not available, using all GPUs (may include integrated GPUs)"
            discrete_gpu_list=$(seq -s, 0 $((GPU_COUNT-1)))
        fi

        print_step "Setting GPU device variables..."
        # Only set if not already set
        if [ -z "$HIP_VISIBLE_DEVICES" ]; then
            export HIP_VISIBLE_DEVICES=$discrete_gpu_list
        fi
        if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
            export CUDA_VISIBLE_DEVICES=$discrete_gpu_list
        fi
        if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
            export PYTORCH_ROCM_DEVICE=$discrete_gpu_list
        fi

        print_success "GPU device variables set:"
        echo "  - HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
        echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo "  - PYTORCH_ROCM_DEVICE: $PYTORCH_ROCM_DEVICE"
    else
        print_warning "No GPUs detected, not setting GPU device variables"
    fi

    # Set ROCm-related variables
    if [ -n "$ROCM_PATH" ]; then
        print_step "Setting ROCm-related variables..."
        # Only set if not already set
        if [ -z "$ROCM_HOME" ]; then
            export ROCM_HOME=$ROCM_PATH
        fi
        if [ -z "$CUDA_HOME" ]; then
            export CUDA_HOME=$ROCM_PATH  # For compatibility with CUDA-based tools
        fi

        print_success "ROCm-related variables set:"
        echo "  - ROCM_HOME: $ROCM_HOME"
        echo "  - CUDA_HOME: $CUDA_HOME"
    fi

    # Set performance-related variables
    print_step "Setting performance-related variables..."

    # Fix for "Tool lib '1' failed to load" issue
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:$LD_LIBRARY_PATH

    # Set HSA_OVERRIDE_GFX_VERSION for compatibility
    if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
    fi

    # Set performance variables
    if [ -z "$HSA_ENABLE_SDMA" ]; then
        export HSA_ENABLE_SDMA=0
    fi
    if [ -z "$GPU_MAX_HEAP_SIZE" ]; then
        export GPU_MAX_HEAP_SIZE=100
    fi
    if [ -z "$GPU_MAX_ALLOC_PERCENT" ]; then
        export GPU_MAX_ALLOC_PERCENT=100
    fi
    if [ -z "$HSA_TOOLS_LIB" ]; then
        export HSA_TOOLS_LIB=1
    fi

    # Set MIOpen variables
    if [ -z "$MIOPEN_DEBUG_CONV_IMPLICIT_GEMM" ]; then
        export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
    fi
    if [ -z "$MIOPEN_FIND_MODE" ]; then
        export MIOPEN_FIND_MODE=3
    fi
    if [ -z "$MIOPEN_FIND_ENFORCE" ]; then
        export MIOPEN_FIND_ENFORCE=3
    fi

    # Set PyTorch variables
    if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
        export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
    fi
    if [ -z "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    fi
    if [ -z "$PYTORCH_HIP_ALLOC_CONF" ]; then
        export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"
    fi

    # Set MPI variables
    if [ -z "$OMPI_MCA_opal_cuda_support" ]; then
        export OMPI_MCA_opal_cuda_support=true
    fi
    if [ -z "$OMPI_MCA_pml_ucx_opal_cuda_support" ]; then
        export OMPI_MCA_pml_ucx_opal_cuda_support=true
    fi
    if [ -z "$OMPI_MCA_btl_openib_allow_ib" ]; then
        export OMPI_MCA_btl_openib_allow_ib=true
    fi
    if [ -z "$OMPI_MCA_btl_openib_warn_no_device_params_found" ]; then
        export OMPI_MCA_btl_openib_warn_no_device_params_found=0
    fi
    if [ -z "$OMPI_MCA_coll_hcoll_enable" ]; then
        export OMPI_MCA_coll_hcoll_enable=0
    fi
    if [ -z "$OMPI_MCA_pml" ]; then
        export OMPI_MCA_pml=ucx
    fi
    if [ -z "$OMPI_MCA_osc" ]; then
        export OMPI_MCA_osc=ucx
    fi
    if [ -z "$OMPI_MCA_btl" ]; then
        export OMPI_MCA_btl=^openib,uct
    fi

    # Add ONNX Runtime to Python path if it exists and not already in PYTHONPATH
    if [ -d "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release" ]; then
        if ! echo "$PYTHONPATH" | grep -q "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release"; then
            export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
            print_step "Added ONNX Runtime to PYTHONPATH"
        else
            print_step "ONNX Runtime already in PYTHONPATH"
        fi
    fi

    print_success "Performance-related variables set"

    return 0
}

# Function to create environment file
create_environment_file() {
    print_section "Creating Environment File"

    # Create environment file
    print_step "Creating environment file..."

    # Create .mlstack_env file in home directory
    cat > $HOME/.mlstack_env << EOF
# ML Stack Environment File
# Created by Enhanced ML Stack Environment Setup Script
# Date: $(date)

# GPU Selection
# Only set if not already set
if [ -z "\$HIP_VISIBLE_DEVICES" ]; then export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES; fi
if [ -z "\$CUDA_VISIBLE_DEVICES" ]; then export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; fi
if [ -z "\$PYTORCH_ROCM_DEVICE" ]; then export PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE; fi

# ROCm Settings
# Only set if not already set
if [ -z "\$ROCM_HOME" ]; then export ROCM_HOME=$ROCM_HOME; fi
if [ -z "\$CUDA_HOME" ]; then export CUDA_HOME=$CUDA_HOME; fi
export PATH=\$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:\$LD_LIBRARY_PATH

# Fix for "Tool lib '1' failed to load" issue
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:\$LD_LIBRARY_PATH

# Performance Settings
# Only set if not already set
if [ -z "\$HSA_OVERRIDE_GFX_VERSION" ]; then export HSA_OVERRIDE_GFX_VERSION=11.0.0; fi
if [ -z "\$HSA_ENABLE_SDMA" ]; then export HSA_ENABLE_SDMA=0; fi
if [ -z "\$GPU_MAX_HEAP_SIZE" ]; then export GPU_MAX_HEAP_SIZE=100; fi
if [ -z "\$GPU_MAX_ALLOC_PERCENT" ]; then export GPU_MAX_ALLOC_PERCENT=100; fi
if [ -z "\$HSA_TOOLS_LIB" ]; then export HSA_TOOLS_LIB=1; fi

# MIOpen Settings
# Only set if not already set
if [ -z "\$MIOPEN_DEBUG_CONV_IMPLICIT_GEMM" ]; then export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1; fi
if [ -z "\$MIOPEN_FIND_MODE" ]; then export MIOPEN_FIND_MODE=3; fi
if [ -z "\$MIOPEN_FIND_ENFORCE" ]; then export MIOPEN_FIND_ENFORCE=3; fi

# PyTorch Settings
# Only set if not already set
if [ -z "\$TORCH_CUDA_ARCH_LIST" ]; then export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"; fi
if [ -z "\$PYTORCH_CUDA_ALLOC_CONF" ]; then export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"; fi
if [ -z "\$PYTORCH_HIP_ALLOC_CONF" ]; then export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"; fi

# MPI Settings
# Only set if not already set
if [ -z "\$OMPI_MCA_opal_cuda_support" ]; then export OMPI_MCA_opal_cuda_support=true; fi
if [ -z "\$OMPI_MCA_pml_ucx_opal_cuda_support" ]; then export OMPI_MCA_pml_ucx_opal_cuda_support=true; fi
if [ -z "\$OMPI_MCA_btl_openib_allow_ib" ]; then export OMPI_MCA_btl_openib_allow_ib=true; fi
if [ -z "\$OMPI_MCA_btl_openib_warn_no_device_params_found" ]; then export OMPI_MCA_btl_openib_warn_no_device_params_found=0; fi
if [ -z "\$OMPI_MCA_coll_hcoll_enable" ]; then export OMPI_MCA_coll_hcoll_enable=0; fi
if [ -z "\$OMPI_MCA_pml" ]; then export OMPI_MCA_pml=ucx; fi
if [ -z "\$OMPI_MCA_osc" ]; then export OMPI_MCA_osc=ucx; fi
if [ -z "\$OMPI_MCA_btl" ]; then export OMPI_MCA_btl=^openib,uct; fi

# ONNX Runtime Settings
# Only add if not already in PYTHONPATH
if ! echo "\$PYTHONPATH" | grep -q "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release"; then
  export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:\$PYTHONPATH
fi
EOF

    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.mlstack_env" $HOME/.bashrc; then
        echo -e "\n# Source ML Stack environment" >> $HOME/.bashrc
        echo "source \$HOME/.mlstack_env" >> $HOME/.bashrc
    fi

    # Source the file
    source $HOME/.mlstack_env

    print_success "Environment file created successfully"
    print_step "Environment file: $HOME/.mlstack_env"
    print_step "The environment file has been added to your .bashrc file."
    print_step "To apply the changes, run: source $HOME/.bashrc"

    return 0
}

# Function to create directory structure
create_directory_structure() {
    print_section "Creating Directory Structure"

    # Create directory structure
    print_step "Creating directory structure..."

    # Create directories
    mkdir -p $HOME/Prod/Stan-s-ML-Stack/logs
    mkdir -p $HOME/Prod/Stan-s-ML-Stack/data
    mkdir -p $HOME/Prod/Stan-s-ML-Stack/models
    mkdir -p $HOME/Prod/Stan-s-ML-Stack/benchmark_results
    mkdir -p $HOME/Prod/Stan-s-ML-Stack/test_results

    print_success "Directory structure created successfully"

    return 0
}

# Function to check system dependencies
check_system_dependencies() {
    print_section "Checking System Dependencies"

    # List of required packages
    required_packages=(
        "build-essential"
        "cmake"
        "git"
        "python3-dev"
        "python3-pip"
        "libnuma-dev"
        "pciutils"
        "mesa-utils"
        "clinfo"
    )

    missing_packages=()

    # Check each package
    for package in "${required_packages[@]}"; do
        if ! dpkg -l | grep -q "ii  $package "; then
            missing_packages+=("$package")
        fi
    done

    # Install missing packages if any
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing system dependencies: ${missing_packages[*]}"
        print_step "Installing missing dependencies..."

        # Use a more interactive approach for sudo
        echo "Updating package lists with sudo..."

        # Check if we're running in non-interactive mode
        if [ -n "$NONINTERACTIVE" ]; then
            # Use DEBIAN_FRONTEND=noninteractive to avoid prompts
            DEBIAN_FRONTEND=noninteractive sudo -E apt-get update -y
        else
            # Interactive mode
            sudo apt-get update -y
        fi

        # Add a small delay to ensure output is flushed
        sleep 0.1

        echo "Installing missing packages with sudo..."
        for package in "${missing_packages[@]}"; do
            echo "Installing $package..."

            # Check if we're running in non-interactive mode
            if [ -n "$NONINTERACTIVE" ]; then
                # Use DEBIAN_FRONTEND=noninteractive to avoid prompts
                DEBIAN_FRONTEND=noninteractive sudo -E apt-get install -y "$package"
            else
                # Interactive mode
                sudo apt-get install -y "$package"
            fi

            # Add a small delay to ensure output is flushed
            sleep 0.1

            if [ $? -eq 0 ]; then
                print_success "Installed $package"
            else
                print_error "Failed to install $package"
            fi
        done

        # Check if installation was successful
        for package in "${missing_packages[@]}"; do
            if dpkg -l | grep -q "ii  $package "; then
                print_success "Installed $package"
            else
                print_error "Failed to install $package"
            fi
        done
    else
        print_success "All system dependencies are installed"
    fi

    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_step "Python version: $python_version"

    # Check pip
    if command_exists pip3; then
        pip_version=$(pip3 --version | awk '{print $2}')
        print_step "pip version: $pip_version"
    else
        print_warning "pip3 not found, installing..."
        sudo apt-get install -y python3-pip
    fi

    # Check git
    if command_exists git; then
        git_version=$(git --version | awk '{print $3}')
        print_step "git version: $git_version"
    else
        print_warning "git not found, installing..."
        sudo apt-get install -y git
    fi

    # Check cmake
    if command_exists cmake; then
        cmake_version=$(cmake --version | head -n 1 | awk '{print $3}')
        print_step "cmake version: $cmake_version"
    else
        print_warning "cmake not found, installing..."
        sudo apt-get install -y cmake
    fi

    return 0
}

# Main function
main() {
    # Initialize progress bar
    init_progress_bar 100
    update_progress_bar 5
    draw_progress_bar "Starting Enhanced ML Stack Environment Setup..."

    print_header "Enhanced ML Stack Environment Setup"

    # Check system dependencies
    update_progress_bar 10
    draw_progress_bar "Checking system dependencies..."
    check_system_dependencies

    # Detect AMD GPUs
    update_progress_bar 15
    draw_progress_bar "Detecting AMD GPUs..."
    detect_amd_gpus
    if [ $? -ne 0 ]; then
        print_warning "GPU detection encountered issues, but continuing..."
    fi

    # Detect ROCm
    update_progress_bar 20
    draw_progress_bar "Detecting ROCm installation..."
    detect_rocm
    if [ $? -ne 0 ]; then
        print_warning "ROCm detection encountered issues, but continuing..."
    fi

    # Configure environment variables
    update_progress_bar 25
    draw_progress_bar "Configuring environment variables..."
    configure_environment_variables
    if [ $? -ne 0 ]; then
        print_error "Environment variable configuration failed. Exiting."
        complete_progress_bar
        exit 1
    fi

    # Create environment file
    update_progress_bar 20
    draw_progress_bar "Creating environment file..."
    create_environment_file
    if [ $? -ne 0 ]; then
        print_error "Environment file creation failed. Exiting."
        complete_progress_bar
        exit 1
    fi

    # Create directory structure
    update_progress_bar 20
    draw_progress_bar "Creating directory structure..."
    create_directory_structure
    if [ $? -ne 0 ]; then
        print_error "Directory structure creation failed. Exiting."
        complete_progress_bar
        exit 1
    fi

    # Complete progress bar
    update_progress_bar 10
    draw_progress_bar "Finalizing setup..."
    complete_progress_bar

    print_header "Enhanced ML Stack Environment Setup Complete"
    print_step "To apply the changes, run: source $HOME/.bashrc"

    return 0
}

# Run main function
main
