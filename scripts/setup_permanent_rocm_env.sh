#!/bin/bash
# setup_permanent_rocm_env.sh
# Unified script for permanent ROCm environment configuration targeting Python 3.12.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_utils.sh"

print_header "Permanent ROCm Environment Setup (Python 3.12)"

# 1. Target Python 3.12
# We prioritized /usr/local/bin/python3 which is now symlinked to uv's 3.12
PYTHON_BIN="/usr/local/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(which python3.12 || which python3)
fi

print_step "Targeting Python interpreter: $PYTHON_BIN"

# 2. Detect Hardware
print_step "Detecting AMD hardware..."
GPU_ARCH=$(rocminfo 2>/dev/null | grep -o "gfx[0-9]*" | head -n1 || echo "gfx1100")
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:.*GPU" || echo "1")
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1 || echo "7.2.0")

# 3. Build .mlstack_env content
ENV_FILE="$HOME/.mlstack_env"
print_step "Generating $ENV_FILE..."

cat > "$ENV_FILE" << EOF
# Permanent ROCm Environment Setup (Generated $(date))
export ROCM_VERSION=$ROCM_VERSION
export ROCM_CHANNEL=latest
export GPU_ARCH=$GPU_ARCH
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
export PYTORCH_ROCM_DEVICE=0
export MLSTACK_PYTHON_BIN=$PYTHON_BIN
export UV_PYTHON=$PYTHON_BIN
export PYTHONPATH=/opt/rocm/lib:\${PYTHONPATH:-}

# Path Settings
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin:/opt/rocm/hip/bin:\$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/opencl/lib:\${LD_LIBRARY_PATH:-}"

# Performance & Compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# PyTorch Optimization
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
export PYTORCH_ALLOC_CONF="max_split_size_mb:512"
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"

# vLLM RDNA3 Support (v0.15.0+)
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0

# Global flags for seamless installs
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_PIP_BREAK_SYSTEM_PACKAGES=1
export UV_SYSTEM_PYTHON=1
EOF

# Correct HSA_TOOLS_LIB logic - Append separately to ensure it's not '0'
if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
    echo "export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler-sdk-tool.so" >> "$ENV_FILE"
else
    echo "# HSA_TOOLS_LIB not set (profiler not found)" >> "$ENV_FILE"
fi

print_success "Environment file created at $ENV_FILE"

# 4. Final Shell check
for shell_rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
    if [ -f "$shell_rc" ]; then
        if ! grep -q "source \$HOME/.mlstack_env" "$shell_rc"; then
            print_step "Injecting source into $shell_rc..."
            echo -e "\n# Source ML Stack Environment\nif [ -f \"\$HOME/.mlstack_env\" ]; then set +u 2>/dev/null; source \"\$HOME/.mlstack_env\"; set -u 2>/dev/null; fi" >> "$shell_rc"
        fi
    fi
done

print_success "Permanent ROCm environment configured for Python 3.12!"
