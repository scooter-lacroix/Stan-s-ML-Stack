#!/bin/bash
#
# Script to create persistent environment variables and symlinks
# for Stan's ML Stack
#

# Source the component detector library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DETECTOR_SCRIPT="$PARENT_DIR/scripts/ml_stack_component_detector.sh"

if [ -f "$DETECTOR_SCRIPT" ]; then
    source "$DETECTOR_SCRIPT"
else
    echo "Error: Component detector script not found at $DETECTOR_SCRIPT"
    exit 1
fi

# Print header
print_header "Creating Persistent Environment for ML Stack"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (use sudo)${RESET}"
  exit 1
fi

# Create system-wide environment file
echo -e "${BLUE}>> Creating system-wide environment file...${RESET}"
cat > /etc/profile.d/mlstack.sh << 'EOF'
#!/bin/bash
# ML Stack Environment Variables
# This file is automatically loaded at login

# Check if ROCm exists
if [ -d "/opt/rocm" ]; then
    # ROCm paths
    export ROCM_PATH=/opt/rocm
    export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:$LD_LIBRARY_PATH

    # CUDA compatibility
    export ROCM_HOME=$ROCM_PATH
    export CUDA_HOME=$ROCM_PATH

    # GPU selection (only set if not already set)
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        export HIP_VISIBLE_DEVICES=0,1
    fi
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    fi
    if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        export PYTORCH_ROCM_DEVICE=0,1
    fi

    # Performance settings
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export HSA_ENABLE_SDMA=0
    export GPU_MAX_HEAP_SIZE=100
    export GPU_MAX_ALLOC_PERCENT=100
    export HSA_TOOLS_LIB=1

    # MIOpen settings
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
    export MIOPEN_FIND_MODE=3
    export MIOPEN_FIND_ENFORCE=3

    # PyTorch settings
    export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"

    # ONNX Runtime
    if [ -d "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release" ]; then
        export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
    fi

    # Flash Attention
    if [ -d "/home/stan/ml_stack/flash_attn_amd_direct" ]; then
        export PYTHONPATH=/home/stan/ml_stack/flash_attn_amd_direct:$PYTHONPATH
    fi

    # Megatron-LM
    if [ -d "/home/stan/megatron/Megatron-LM" ]; then
        export PYTHONPATH=/home/stan/megatron/Megatron-LM:$PYTHONPATH
    fi
fi
EOF

# Make the file executable
chmod +x /etc/profile.d/mlstack.sh
echo -e "${GREEN}✓ Created system-wide environment file: /etc/profile.d/mlstack.sh${RESET}"

# Create systemd service to create symlinks at boot
echo -e "${BLUE}>> Creating systemd service for persistent symlinks...${RESET}"
cat > /etc/systemd/system/mlstack-symlinks.service << 'EOF'
[Unit]
Description=Create ML Stack Symlinks
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash /usr/local/bin/mlstack-symlinks.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Create symlink script
echo -e "${BLUE}>> Creating symlink script...${RESET}"
cat > /usr/local/bin/mlstack-symlinks.sh << 'EOF'
#!/bin/bash
# Script to create persistent symlinks for ML Stack

# Create ninja symlinks
if [ -f "/usr/bin/ninja" ] && [ ! -f "/usr/bin/ninja-build" ]; then
    ln -sf /usr/bin/ninja /usr/bin/ninja-build
    echo "Created symlink: /usr/bin/ninja-build -> /usr/bin/ninja"
elif [ -f "/usr/bin/ninja-build" ] && [ ! -f "/usr/bin/ninja" ]; then
    ln -sf /usr/bin/ninja-build /usr/bin/ninja
    echo "Created symlink: /usr/bin/ninja -> /usr/bin/ninja-build"
fi

# Create ROCm symlinks if needed
if [ -d "/opt/rocm" ]; then
    # Create CUDA compatibility symlinks
    if [ ! -d "/usr/local/cuda" ]; then
        ln -sf /opt/rocm /usr/local/cuda
        echo "Created symlink: /usr/local/cuda -> /opt/rocm"
    fi

    # Create RCCL symlinks if needed
    if [ -f "/opt/rocm/lib/librccl.so" ] && [ ! -d "/opt/rocm/rccl" ]; then
        mkdir -p /opt/rocm/rccl/lib
        ln -sf /opt/rocm/lib/librccl.so /opt/rocm/rccl/lib/librccl.so
        echo "Created symlink: /opt/rocm/rccl/lib/librccl.so -> /opt/rocm/lib/librccl.so"
    fi
fi

# Create Python module symlinks if needed
PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Flash Attention symlink
if [ -d "/home/stan/ml_stack/flash_attn_amd_direct" ] && [ ! -d "$PYTHON_SITE_PACKAGES/flash_attention_amd" ]; then
    ln -sf /home/stan/ml_stack/flash_attn_amd_direct "$PYTHON_SITE_PACKAGES/flash_attention_amd"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/flash_attention_amd -> /home/stan/ml_stack/flash_attn_amd_direct"
fi

# Megatron-LM symlink
if [ -d "/home/stan/megatron/Megatron-LM" ] && [ ! -d "$PYTHON_SITE_PACKAGES/megatron" ]; then
    ln -sf /home/stan/megatron/Megatron-LM "$PYTHON_SITE_PACKAGES/megatron"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/megatron -> /home/stan/megatron/Megatron-LM"
fi

# ONNX Runtime symlink
if [ -d "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release" ] && [ ! -d "$PYTHON_SITE_PACKAGES/onnxruntime" ]; then
    ln -sf /home/stan/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime "$PYTHON_SITE_PACKAGES/onnxruntime"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/onnxruntime -> /home/stan/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime"
fi
EOF

# Make the script executable
chmod +x /usr/local/bin/mlstack-symlinks.sh
echo -e "${GREEN}✓ Created symlink script: /usr/local/bin/mlstack-symlinks.sh${RESET}"

# Enable and start the service
echo -e "${BLUE}>> Enabling and starting the service...${RESET}"
systemctl enable mlstack-symlinks.service
systemctl start mlstack-symlinks.service
echo -e "${GREEN}✓ Enabled and started mlstack-symlinks service${RESET}"

# Create a user-specific environment file
echo -e "${BLUE}>> Creating user-specific environment file...${RESET}"
cat > /home/stan/.mlstack_env << 'EOF'
# ML Stack User Environment
# Source this file in your .bashrc or .zshrc

# Source the system-wide environment file
if [ -f "/etc/profile.d/mlstack.sh" ]; then
    source /etc/profile.d/mlstack.sh
fi

# Add any user-specific environment variables here
EOF

# Add to .bashrc if not already there
if ! grep -q "source ~/.mlstack_env" /home/stan/.bashrc; then
    echo -e "\n# Source ML Stack environment" >> /home/stan/.bashrc
    echo "source ~/.mlstack_env" >> /home/stan/.bashrc
    echo -e "${GREEN}✓ Added environment sourcing to .bashrc${RESET}"
else
    echo -e "${YELLOW}⚠ Environment sourcing already in .bashrc${RESET}"
fi

# Change ownership of the user file
chown stan:stan /home/stan/.mlstack_env
echo -e "${GREEN}✓ Created user-specific environment file: /home/stan/.mlstack_env${RESET}"

# Create verification script
echo -e "${BLUE}>> Creating environment verification script...${RESET}"
cat > /usr/local/bin/verify-mlstack-env.sh << 'EOF'
#!/bin/bash
# Script to verify ML Stack environment variables and symlinks

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'
BOLD='\033[1m'

# Print header
echo -e "${BLUE}${BOLD}=== Verifying ML Stack Environment ===${RESET}\n"

# Check environment variables
echo -e "${BLUE}>> Checking environment variables...${RESET}"
ENV_VARS=(
    "ROCM_PATH"
    "ROCM_HOME"
    "CUDA_HOME"
    "HIP_VISIBLE_DEVICES"
    "CUDA_VISIBLE_DEVICES"
    "PYTORCH_ROCM_DEVICE"
    "HSA_OVERRIDE_GFX_VERSION"
    "HSA_ENABLE_SDMA"
    "GPU_MAX_HEAP_SIZE"
    "GPU_MAX_ALLOC_PERCENT"
    "HSA_TOOLS_LIB"
)

for var in "${ENV_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}✓ $var=${!var}${RESET}"
    else
        echo -e "${RED}✗ $var is not set${RESET}"
    fi
done

# Check symlinks
echo -e "\n${BLUE}>> Checking symlinks...${RESET}"
SYMLINKS=(
    "/usr/bin/ninja:/usr/bin/ninja-build"
    "/usr/local/cuda:/opt/rocm"
    "/opt/rocm/rccl/lib/librccl.so:/opt/rocm/lib/librccl.so"
)

for link in "${SYMLINKS[@]}"; do
    src=$(echo $link | cut -d: -f1)
    dst=$(echo $link | cut -d: -f2)

    if [ -L "$src" ] && [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ]; then
        echo -e "${GREEN}✓ $src -> $(readlink -f "$src")${RESET}"
    elif [ -e "$src" ] && [ -e "$dst" ] && [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ]; then
        echo -e "${GREEN}✓ $src and $dst point to the same file${RESET}"
    elif [ -L "$src" ]; then
        echo -e "${YELLOW}⚠ $src -> $(readlink -f "$src") (expected: $dst)${RESET}"
    elif [ -e "$src" ]; then
        echo -e "${YELLOW}⚠ $src exists but is not a symlink to $dst${RESET}"
    else
        echo -e "${RED}✗ $src does not exist${RESET}"
    fi
done

# Check Python modules
echo -e "\n${BLUE}>> Checking Python modules...${RESET}"
PYTHON_MODULES=(
    "torch"
    "onnxruntime"
    "flash_attention_amd"
    "megatron"
    "triton"
    "bitsandbytes"
    "vllm"
    "wandb"
)

for module in "${PYTHON_MODULES[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        version=$(python3 -c "import $module; print($module.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓ $module (version: $version)${RESET}"
    else
        echo -e "${RED}✗ $module is not importable${RESET}"
    fi
done

# Check if ROCm is working with PyTorch
echo -e "\n${BLUE}>> Checking ROCm with PyTorch...${RESET}"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✓ PyTorch can access $gpu_count GPU(s) through ROCm${RESET}"

    for i in $(seq 0 $((gpu_count-1))); do
        gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null)
        echo -e "${GREEN}  - GPU $i: $gpu_name${RESET}"
    done
else
    echo -e "${RED}✗ PyTorch cannot access GPUs through ROCm${RESET}"
fi

echo -e "\n${BLUE}${BOLD}=== ML Stack Environment Verification Complete ===${RESET}"
EOF

# Make the script executable
chmod +x /usr/local/bin/verify-mlstack-env.sh
echo -e "${GREEN}✓ Created environment verification script: /usr/local/bin/verify-mlstack-env.sh${RESET}"

echo -e "\n${GREEN}${BOLD}=== ML Stack Persistent Environment Setup Complete ===${RESET}"
echo -e "To verify the environment, run: ${BLUE}sudo verify-mlstack-env.sh${RESET}"
echo -e "The environment will be automatically loaded on system boot."
echo -e "You may need to log out and log back in for all changes to take effect."
