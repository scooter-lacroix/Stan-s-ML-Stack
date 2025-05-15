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
# ML Stack Installation Script
# =============================================================================
# This script installs the ML Stack for AMD GPUs.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# Trap to ensure we exit properly
trap 'echo "Forced exit"; kill -9 $$' EXIT

# Set Python interpreter - prefer virtual environment if available
PYTHON_INTERPRETER="python3"

# Check for virtual environment
if [ -f "$HOME/Prod/Stan-s-ML-Stack/venv/bin/python" ]; then
    PYTHON_INTERPRETER="$HOME/Prod/Stan-s-ML-Stack/venv/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_INTERPRETER="./venv/bin/python"
fi

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                ML Stack Installation Script
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

check_prerequisites() {
    print_section "Checking prerequisites"

    echo -e "${CYAN}${BOLD}System Requirements Check${RESET}"
    print_separator

    # Check if ROCm is installed
    if ! command -v rocminfo &> /dev/null; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
    fi
    print_success "ROCm is installed"

    # Check ROCm version - try multiple methods to ensure we get a version
    rocm_version=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk '{print $3}')

    # If rocminfo didn't work, try alternative methods
    if [ -z "$rocm_version" ]; then
        # Try getting version from rocm-smi
        if command -v rocm-smi &> /dev/null; then
            rocm_version=$(rocm-smi --showversion 2>/dev/null | grep -i "ROCm Version" | awk '{print $3}')
        fi
    fi

    # If still no version, try checking the directory name
    if [ -z "$rocm_version" ]; then
        if [ -d "/opt/rocm" ]; then
            # Try to get version from a symlink
            if [ -L "/opt/rocm" ]; then
                rocm_target=$(readlink -f /opt/rocm)
                rocm_version=$(echo "$rocm_target" | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo "")
            fi

            # If still no version, check for version file
            if [ -z "$rocm_version" ] && [ -f "/opt/rocm/.info/version" ]; then
                rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo "")
            fi

            # If still no version, check for version-dev file
            if [ -z "$rocm_version" ] && [ -f "/opt/rocm/.info/version-dev" ]; then
                rocm_version=$(cat /opt/rocm/.info/version-dev 2>/dev/null || echo "")
            fi
        fi
    fi

    # If still no version, use a hardcoded version based on common ROCm versions
    if [ -z "$rocm_version" ]; then
        rocm_version="6.4.0"  # Default to a recent version
        print_warning "Could not detect ROCm version automatically. Using default: $rocm_version"
    else
        print_success "Detected ROCm version: $rocm_version"
    fi

    print_separator
    echo -e "${CYAN}${BOLD}Software Dependencies${RESET}"
    print_separator

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        return 1
    fi
    print_success "Python 3 is installed"

    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"

    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3 first."
        return 1
    fi
    print_success "pip3 is installed"

    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install git first."
        return 1
    fi
    print_success "Git is installed"

    # Check if cmake is installed
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install cmake first."
        return 1
    fi
    print_success "CMake is installed"

    print_separator
    echo -e "${CYAN}${BOLD}GPU Detection${RESET}"
    print_separator

    # Check if AMD GPUs are detected - redirect stderr to avoid "Tool lib 1 failed to load" messages
    if ! rocminfo 2>/dev/null | grep -q "Device Type:.*GPU"; then
        print_error "No AMD GPUs detected. Please check your hardware and ROCm installation."
        return 1
    fi

    # Count AMD GPUs
    gpu_count=$(rocminfo 2>/dev/null | grep "Device Type:.*GPU" | wc -l)
    print_success "Detected $gpu_count AMD GPU(s)"

    # List AMD GPUs
    echo -e "${CYAN}Detected GPU Hardware:${RESET}"

    # Create a temporary file to store GPU information
    tmp_gpu_info=$(mktemp)

    # Extract GPU information to the temporary file
    rocminfo 2>/dev/null > "$tmp_gpu_info"

    # Process GPU information
    gpu_index=0
    while read -r line; do
        gpu_name=$(echo "$line" | awk -F: '{print $2}' | xargs)
        echo -e "  ${GREEN}▸ GPU $gpu_index:${RESET} $gpu_name"
        gpu_index=$((gpu_index+1))
    done < <(grep "Marketing Name" "$tmp_gpu_info")

    # Clean up
    rm -f "$tmp_gpu_info"

    print_separator
    echo -e "${CYAN}${BOLD}Environment Variables${RESET}"
    print_separator

    # Set environment variables if not set
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        print_warning "HIP_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi

    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        print_warning "CUDA_VISIBLE_DEVICES is not set. Setting to all GPUs..."
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    fi

    if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        print_warning "PYTORCH_ROCM_DEVICE is not set. Setting to all GPUs..."
        export PYTORCH_ROCM_DEVICE=$(seq -s, 0 $((gpu_count-1)))
    fi

    echo -e "${CYAN}Current Environment Configuration:${RESET}"
    echo -e "  ${GREEN}▸ HIP_VISIBLE_DEVICES:${RESET} $HIP_VISIBLE_DEVICES"
    echo -e "  ${GREEN}▸ CUDA_VISIBLE_DEVICES:${RESET} $CUDA_VISIBLE_DEVICES"
    echo -e "  ${GREEN}▸ PYTORCH_ROCM_DEVICE:${RESET} $PYTORCH_ROCM_DEVICE"

    print_separator
    echo -e "${CYAN}${BOLD}Disk Space Check${RESET}"
    print_separator

    # Check disk space
    available_space=$(df -h $HOME | awk 'NR==2 {print $4}')
    total_space=$(df -h $HOME | awk 'NR==2 {print $2}')
    used_space=$(df -h $HOME | awk 'NR==2 {print $3}')
    used_percent=$(df -h $HOME | awk 'NR==2 {print $5}')

    echo -e "${CYAN}Storage Information:${RESET}"
    echo -e "  ${GREEN}▸ Total Space:${RESET}     $total_space"
    echo -e "  ${GREEN}▸ Used Space:${RESET}      $used_space ($used_percent)"
    echo -e "  ${GREEN}▸ Available Space:${RESET} $available_space"

    # Check if there's enough disk space (at least 20GB)
    available_space_kb=$(df -k $HOME | awk 'NR==2 {print $4}')
    if [ $available_space_kb -lt 20971520 ]; then  # 20GB in KB
        print_warning "You have less than 20GB of free disk space. Some components might fail to build."
        read -p "Do you want to continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation aborted by user."
            return 1
        fi
    else
        print_success "Sufficient disk space available for installation"
    fi

    return 0
}
install_rocm_config() {
    print_section "Installing ROCm configuration"

    # Create ROCm configuration file
    print_step "Creating ROCm configuration file..."

    # Create .rocmrc file in home directory
    cat > $HOME/.rocmrc << EOF
# ROCm Configuration File
# Created by ML Stack Installation Script

# Environment Variables
export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=1

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# Logging Settings
export HIP_TRACE_API=0
export AMD_LOG_LEVEL=4
EOF

    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.rocmrc" $HOME/.bashrc; then
        echo -e "\n# Source ROCm configuration" >> $HOME/.bashrc
        echo "source \$HOME/.rocmrc" >> $HOME/.bashrc
    fi

    # Source the file
    source $HOME/.rocmrc

    print_success "ROCm configuration installed successfully"
}

install_pytorch() {
    print_section "Installing PyTorch with ROCm support"

    # Check if PyTorch with ROCm is already installed
    if python3 -c "import torch; print(torch.version.hip)" &> /dev/null; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        rocm_version=$(python3 -c "import torch; print(torch.version.hip)")
        print_warning "PyTorch with ROCm support is already installed (PyTorch $pytorch_version, ROCm $rocm_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping PyTorch installation."
            return 0
        fi
    fi

    # Install PyTorch with ROCm support
    print_step "Installing PyTorch with ROCm support..."

    # Detect ROCm version to use the appropriate PyTorch version
    rocm_major_version=$(echo "$rocm_version" | cut -d '.' -f 1)
    rocm_minor_version=$(echo "$rocm_version" | cut -d '.' -f 2)

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_step "Installing uv package manager..."
        python3 -m pip install uv
    fi

    # Try to use the exact ROCm version first, then fall back to closest available
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

    # Verify installation - check for both HIP and CUDA backends
    if python3 -c "import torch; print(hasattr(torch, 'version') and hasattr(torch.version, 'hip'))" | grep -q "True"; then
        # ROCm/HIP backend is available
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        rocm_version=$(python3 -c "import torch; print(torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')")
        print_success "PyTorch with ROCm support installed successfully (PyTorch $pytorch_version, ROCm $rocm_version)"

        # Print GPU information
        print_step "GPU information:"
        python3 -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

        # Test a simple tensor operation
        print_step "Testing GPU tensor operations..."
        if python3 -c "import torch; x = torch.ones(10, device='cuda' if torch.cuda.is_available() else 'cpu'); y = x + 1; print('Success' if torch.all(y == 2) else 'Failed');" | grep -q "Success"; then
            print_success "GPU tensor operations working correctly"
        else
            print_warning "GPU tensor operations may not be working correctly"
        fi
    elif python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        # CUDA backend is available but not HIP
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        print_warning "PyTorch with CUDA support installed (PyTorch $pytorch_version), but ROCm/HIP support is not detected"
        print_warning "This may cause issues with AMD GPUs. Consider reinstalling with ROCm support."

        # Print GPU information anyway
        print_step "GPU information:"
        python3 -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
        return 0
    else
        print_error "PyTorch installation failed. Neither ROCm/HIP nor CUDA is available."
        return 1
    fi

    return 0
}

install_onnx_runtime() {
    print_section "Installing ONNX Runtime with ROCm support"

    # Check if ONNX Runtime is already installed
    if python3 -c "import onnxruntime" &> /dev/null; then
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
        print_warning "ONNX Runtime is already installed (version $onnx_version)."

        # Check if ROCMExecutionProvider is available
        if python3 -c "import onnxruntime; print('ROCMExecutionProvider' in onnxruntime.get_available_providers())" | grep -q "True"; then
            print_success "ONNX Runtime with ROCm support is already installed."
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping ONNX Runtime installation."
                return 0
            fi
        else
            print_warning "ONNX Runtime is installed but ROCMExecutionProvider is not available."
            print_step "Reinstalling ONNX Runtime with ROCm support..."
        fi
    fi

    # Build and install ONNX Runtime with ROCm support
    print_step "Building ONNX Runtime with ROCm support..."

    # Run the build script
    $HOME/Prod/Stan-s-ML-Stack/scripts/build_onnxruntime.sh

    # Check if installation was successful
    if [ $? -ne 0 ]; then
        print_error "ONNX Runtime installation failed."
        return 1
    fi

    print_success "ONNX Runtime with ROCm support installed successfully"
    return 0
}

install_migraphx() {
    print_section "Installing MIGraphX"

    # Check if MIGraphX is already installed
    if python3 -c "import migraphx" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_warning "MIGraphX is already installed (version $migraphx_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping MIGraphX installation."
            return 0
        fi
    fi

    # Install MIGraphX from ROCm repository
    print_step "Installing MIGraphX from ROCm repository..."
    sudo apt-get update
    sudo apt-get install -y migraphx python3-migraphx

    # Verify installation
    if python3 -c "import migraphx; print(migraphx.__version__)" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_success "MIGraphX installed successfully (version $migraphx_version)"
    else
        print_error "MIGraphX installation failed."
        return 1
    fi

    return 0
}
install_megatron() {
    print_section "Installing Megatron-LM"

    # Check if Megatron-LM is already installed
    if $PYTHON_INTERPRETER -c "import megatron" &> /dev/null; then
        print_warning "Megatron-LM is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping Megatron-LM installation."
            return 0
        fi
    fi

    # Clone Megatron-LM repository
    print_step "Cloning Megatron-LM repository..."
    cd $HOME
    if [ -d "Megatron-LM" ]; then
        print_warning "Megatron-LM repository already exists. Updating..."
        cd Megatron-LM
        git pull
        cd ..
    else
        git clone https://github.com/NVIDIA/Megatron-LM.git
        cd Megatron-LM
    fi

    # Create patch file to remove NVIDIA-specific dependencies
    print_step "Creating patch file to remove NVIDIA-specific dependencies..."
    cat > remove_nvidia_deps.patch << 'EOF'
diff --git a/megatron/model/fused_softmax.py b/megatron/model/fused_softmax.py
index 7a5b2e5..3e5c2e5 100644
--- a/megatron/model/fused_softmax.py
+++ b/megatron/model/fused_softmax.py
@@ -15,7 +15,7 @@
 """Fused softmax."""

 import torch
-import torch.nn.functional as F
+import torch.nn.functional as F  # Use PyTorch's native implementation


 class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
@@ -24,8 +24,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             scale (float): scaling factor

         Returns:
@@ -33,10 +32,10 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
-        # Create a mask for the upper triangular part (including the diagonal)
+        # Create a mask for the upper triangular part
         seq_len = inputs.size(2)
         mask = torch.triu(
             torch.ones(seq_len, seq_len, device=inputs.device, dtype=torch.bool),
@@ -59,7 +58,7 @@ class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
@@ -77,8 +76,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inputs, mask, scale):
         """Forward pass.
-        Args:
-            inputs (Tensor): input tensor (b, np, sq, sk)
+        Args: inputs (Tensor): input tensor (b, np, sq, sk)
             mask (Tensor): attention mask (b, 1, sq, sk)
             scale (float): scaling factor

@@ -87,7 +85,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        scale_t = torch.tensor([scale])
+        scale_t = torch.tensor([scale], device=inputs.device)
         ctx.scale_t = scale_t
         input_scaled = inputs * scale_t
         # Apply the mask
@@ -110,7 +108,7 @@ class ScaledMaskedSoftmax(torch.autograd.Function):
         """
         # Use the native pytorch implementation for compatibility
         import torch.nn.functional as F
-        output_grads_scaled = grad_output * ctx.softmax_results
+        output_grads_scaled = grad_output * ctx.softmax_results  # Element-wise multiplication
         input_grads = output_grads_scaled - torch.sum(
             output_grads_scaled * ctx.softmax_results, dim=-1, keepdim=True
         ) * ctx.softmax_results
diff --git a/megatron/training.py b/megatron/training.py
index 9a5b2e5..3e5c2e5 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -30,7 +30,7 @@ import torch
 from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

 from megatron import get_args
-from megatron import get_timers
+from megatron import get_timers  # Timing utilities
 from megatron import get_tensorboard_writer
 from megatron import mpu
 from megatron import print_rank_0
@@ -38,7 +38,7 @@ from megatron.checkpointing import load_checkpoint
 from megatron.checkpointing import save_checkpoint
 from megatron.model import DistributedDataParallel as LocalDDP
 from megatron.model import Float16Module
-from megatron.model.realm_model import ICTBertModel
+from megatron.model.realm_model import ICTBertModel  # Import model
 from megatron.utils import check_adlr_autoresume_termination
 from megatron.utils import unwrap_model
 from megatron.data.data_samplers import build_pretraining_data_loader
@@ -46,7 +46,7 @@ from megatron.utils import report_memory


 def pretrain(train_valid_test_dataset_provider, model_provider,
-             forward_step_func, extra_args_provider=None, args_defaults={}):
+             forward_step_func, extra_args_provider=None, args_defaults=None):
     """Main training program.

     This function will run the followings in the order provided:
@@ -59,6 +59,9 @@ def pretrain(train_valid_test_dataset_provider, model_provider,
         5) validation

     """
+    if args_defaults is None:
+        args_defaults = {}
+
     # Initalize and get arguments, timers, and Tensorboard writer.
     initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
diff --git a/requirements.txt b/requirements.txt
index 9a5b2e5..3e5c2e5 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,5 @@
 torch>=1.7
 numpy
-apex
 pybind11
 regex
 nltk
EOF

    # Apply patch
    print_step "Applying patch..."
    git apply remove_nvidia_deps.patch

    # Install dependencies
    print_step "Installing dependencies..."
    uv_pip_install -r requirements.txt
    uv_pip_install tensorboard scipy

    # Install Megatron-LM
    print_step "Installing Megatron-LM..."
    uv_pip_install -e .

    # Verify installation
    if python3 -c "import megatron; print('Megatron-LM imported successfully')" &> /dev/null; then
        print_success "Megatron-LM installed successfully"
    else
        print_error "Megatron-LM installation failed."
        return 1
    fi

    return 0
}

install_flash_attention() {
    print_section "Installing Flash Attention with AMD GPU support"

    # Check if Flash Attention is already installed
    if $PYTHON_INTERPRETER -c "import flash_attention_amd" &> /dev/null; then
        print_warning "Flash Attention with AMD GPU support is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping Flash Attention installation."
            return 0
        fi
    fi

    # Build and install Flash Attention with AMD GPU support
    print_step "Building Flash Attention with AMD GPU support..."

    # Run the build script
    $HOME/Prod/Stan-s-ML-Stack/scripts/build_flash_attn_amd.sh

    # Check if installation was successful
    if [ $? -ne 0 ]; then
        print_error "Flash Attention installation failed."
        return 1
    fi

    print_success "Flash Attention with AMD GPU support installed successfully"
    return 0
}

install_rccl() {
    print_section "Installing RCCL"

    # Check if RCCL is already installed
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_warning "RCCL is already installed."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping RCCL installation."
            return 0
        fi
    fi

    # Install RCCL from ROCm repository
    print_step "Installing RCCL from ROCm repository..."
    sudo apt-get update
    sudo apt-get install -y rccl

    # Verify installation
    if [ -f "/opt/rocm/lib/librccl.so" ]; then
        print_success "RCCL installed successfully"
    else
        print_error "RCCL installation failed."
        return 1
    fi

    return 0
}

install_mpi() {
    print_section "Installing MPI"

    # Check if MPI is already installed
    if command -v mpirun &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_warning "MPI is already installed ($mpi_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping MPI installation."
            return 0
        fi
    fi

    # Install OpenMPI
    print_step "Installing OpenMPI..."
    sudo apt-get update
    sudo apt-get install -y openmpi-bin libopenmpi-dev

    # Configure OpenMPI for ROCm
    print_step "Configuring OpenMPI for ROCm..."

    # Create MPI configuration file
    cat > $HOME/.mpirc << 'EOF'
# MPI Configuration File
# Created by ML Stack Installation Script

# OpenMPI Configuration
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml_ucx_opal_cuda_support=true
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

# Performance Tuning
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib,uct
EOF

    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.mpirc" $HOME/.bashrc; then
        echo -e "\n# Source MPI configuration" >> $HOME/.bashrc
        echo "source \$HOME/.mpirc" >> $HOME/.bashrc
    fi

    # Source the file
    source $HOME/.mpirc

    # Install mpi4py
    print_step "Installing mpi4py..."
    uv_pip_install mpi4py

    # Verify installation
    if command -v mpirun &> /dev/null && python3 -c "import mpi4py; print('mpi4py imported successfully')" &> /dev/null; then
        mpi_version=$(mpirun --version | head -n 1)
        print_success "MPI installed successfully ($mpi_version)"
    else
        print_error "MPI installation failed."
        return 1
    fi

    return 0
}
install_all_core() {
    print_section "Installing all core components"

    # Install ROCm configuration
    install_rocm_config
    if [ $? -ne 0 ]; then
        print_error "ROCm configuration installation failed."
        return 1
    fi

    # Install PyTorch
    install_pytorch
    if [ $? -ne 0 ]; then
        print_error "PyTorch installation failed."
        return 1
    fi

    # Install ONNX Runtime
    install_onnx_runtime
    if [ $? -ne 0 ]; then
        print_error "ONNX Runtime installation failed."
        return 1
    fi

    # Install MIGraphX
    install_migraphx
    if [ $? -ne 0 ]; then
        print_error "MIGraphX installation failed."
        return 1
    fi

    # Install Megatron-LM
    install_megatron
    if [ $? -ne 0 ]; then
        print_error "Megatron-LM installation failed."
        return 1
    fi

    # Install Flash Attention
    install_flash_attention
    if [ $? -ne 0 ]; then
        print_error "Flash Attention installation failed."
        return 1
    fi

    # Install RCCL
    install_rccl
    if [ $? -ne 0 ]; then
        print_error "RCCL installation failed."
        return 1
    fi

    # Install MPI
    install_mpi
    if [ $? -ne 0 ]; then
        print_error "MPI installation failed."
        return 1
    fi

    print_success "All core components installed successfully"
    return 0
}

verify_installation() {
    print_section "Verifying installation"

    # Create verification script in the current directory instead of home
    print_step "Creating verification script..."
    cat > ./verify_ml_stack.py << EOF
import sys
import os
import importlib.util

# Color definitions
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"{CYAN}{BOLD}=== {text} ==={RESET}")
    print()

def print_section(text):
    print(f"{BLUE}{BOLD}>>> {text}{RESET}")

def print_step(text):
    print(f"{MAGENTA}>> {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def check_module(module_name, display_name=None):
    if display_name is None:
        display_name = module_name

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print_success(f"{display_name} is installed (version: {version})")
        return module
    except ImportError:
        print_error(f"{display_name} is not installed")
        return None

def main():
    print_header("ML Stack Verification")

    # Check PyTorch
    print_section("Checking PyTorch")
    torch = check_module("torch", "PyTorch")
    if torch:
        # Check for ROCm/HIP support
        has_hip = hasattr(torch.version, 'hip')
        if has_hip:
            print_success(f"PyTorch has ROCm/HIP support (version: {torch.version.hip})")
        else:
            print_warning("PyTorch does not have explicit ROCm/HIP support")

        # Check CUDA availability (works for both CUDA and ROCm through HIP)
        if torch.cuda.is_available():
            if has_hip:
                print_success("GPU acceleration is available through ROCm/HIP")
            else:
                print_warning("GPU acceleration is available through CUDA (not ROCm/HIP)")

            # Check number of GPUs
            device_count = torch.cuda.device_count()
            print_step(f"Number of GPUs: {device_count}")

            # Check GPU information
            for i in range(device_count):
                print_step(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # Run a simple tensor operation
            try:
                x = torch.ones(10, device="cuda")
                y = x + 1
                print_success("Simple tensor operation on GPU successful")
            except Exception as e:
                print_error(f"Simple tensor operation on GPU failed: {e}")
        else:
            print_error("GPU acceleration is not available")

    # Check ONNX Runtime
    print_section("Checking ONNX Runtime")
    ort = check_module("onnxruntime", "ONNX Runtime")
    if ort:
        # Check available providers
        providers = ort.get_available_providers()
        print_step(f"Available providers: {providers}")

        # Check if ROCMExecutionProvider is available
        if 'ROCMExecutionProvider' in providers:
            print_success("ROCMExecutionProvider is available")
        else:
            print_warning("ROCMExecutionProvider is not available")

    # Check MIGraphX
    print_section("Checking MIGraphX")
    check_module("migraphx", "MIGraphX")

    # Check Megatron-LM
    print_section("Checking Megatron-LM")
    try:
        import megatron
        print_success("Megatron-LM is installed")
    except ImportError:
        print_error("Megatron-LM is not installed")

    # Check Flash Attention
    print_section("Checking Flash Attention")
    try:
        from flash_attention_amd import flash_attn_func
        print_success("Flash Attention is installed")
    except ImportError:
        print_error("Flash Attention is not installed")

    # Check RCCL
    print_section("Checking RCCL")
    if os.path.exists("/opt/rocm/lib/librccl.so"):
        print_success("RCCL is installed")
    else:
        print_error("RCCL is not installed")

    # Check MPI
    print_section("Checking MPI")
    if os.system("which mpirun > /dev/null") == 0:
        print_success("MPI is installed")

        # Check mpi4py
        check_module("mpi4py", "mpi4py")
    else:
        print_error("MPI is not installed")

    print_header("Verification Complete")

if __name__ == "__main__":
    main()
EOF

    # Run verification script
    print_step "Running verification script..."
    # Make sure to use python3 to run the script, not bash
    $PYTHON_INTERPRETER ./verify_ml_stack.py

    # Clean up
    print_step "Cleaning up..."
    rm -f ./verify_ml_stack.py

    print_success "Verification completed"
}

show_menu() {
    print_header "ML Stack Installation Menu"

    echo -e "1) Install ROCm Configuration"
    echo -e "2) Install PyTorch with ROCm support"
    echo -e "3) Install ONNX Runtime with ROCm support"
    echo -e "4) Install MIGraphX"
    echo -e "5) Install Megatron-LM"
    echo -e "6) Install Flash Attention with AMD GPU support"
    echo -e "7) Install RCCL"
    echo -e "8) Install MPI"
    echo -e "9) Install All Core Components"
    echo -e "10) Verify Installation"
    echo -e "0) Exit"
    echo

    read -p "Enter your choice: " choice

    case $choice in
        1)
            install_rocm_config
            ;;
        2)
            install_pytorch
            ;;
        3)
            install_onnx_runtime
            ;;
        4)
            install_migraphx
            ;;
        5)
            install_megatron
            ;;
        6)
            install_flash_attention
            ;;
        7)
            install_rccl
            ;;
        8)
            install_mpi
            ;;
        9)
            install_all_core
            ;;
        10)
            verify_installation
            ;;
        0)
            print_header "Exiting ML Stack Installation"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            ;;
    esac

    # Show menu again
    show_menu
}

main() {
    print_header "ML Stack Installation Script"

    # Start time
    start_time=$(date +%s)

    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi

    # Show menu
    show_menu

    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))

    print_header "ML Stack Installation Completed"
    echo -e "${GREEN}Total installation time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"

    return 0
}

# Main script execution
main

# Force exit to prevent hanging
echo "Installation complete. Forcing exit to prevent hanging..."
kill -9 $$ 2>/dev/null
exit 0
