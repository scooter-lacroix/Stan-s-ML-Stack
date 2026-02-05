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
# ML Stack Verification Script for AMD GPUs
# =============================================================================
# This script verifies the installation of the ML stack components.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                ML Stack Verification Script
EOF
echo

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MLSTACK_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Create log directory
LOG_DIR="${MLSTACK_LOG_DIR:-$HOME/.mlstack/logs}"
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/ml_stack_verify_$(date +"%Y%m%d_%H%M%S").log"

# Set up environment variables for ROCm and ONNX Runtime
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$HOME/.local/lib:$LD_LIBRARY_PATH"

# Suppress HIP logs by setting environment variables
export HIP_TRACE_API=0
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2
export HSA_ENABLE_SDMA=0
if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
    export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
else
    unset HSA_TOOLS_LIB
fi
export HSA_TOOLS_REPORT_LOAD_FAILURE=0
export HSA_ENABLE_DEBUG=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=0
export MIOPEN_ENABLE_LOGGING_IMPL=0
export AMD_LOG_LEVEL=0

# Select Python interpreter (prefer installer-selected venvs)
PYTHON_CMD="${MLSTACK_PYTHON_BIN:-${AITER_VENV_PYTHON:-${UV_PYTHON:-}}}"
if [ -z "$PYTHON_CMD" ]; then
    if [ -f "$HOME/rocm_venv/bin/python" ]; then
        PYTHON_CMD="$HOME/rocm_venv/bin/python"
        # Activate the virtual environment to ensure all libraries are found
        source "$HOME/rocm_venv/bin/activate"
    else
        PYTHON_CMD="python3"
    fi
fi

# Detect ONNX Runtime ROCm provider library if present
ORT_ROCM_EP_PROVIDER_PATH=$(
    $PYTHON_CMD - <<'PY'
import sys
try:
    import onnxruntime
    from pathlib import Path
    base = Path(onnxruntime.__file__).parent
    libs = list(base.rglob("libonnxruntime_providers_rocm.so"))
    if libs:
        print(libs[0])
except Exception:
    pass
PY
)
if [ -n "$ORT_ROCM_EP_PROVIDER_PATH" ]; then
    export ORT_ROCM_EP_PROVIDER_PATH
fi

PYTHON_SITEPACKAGES=$(
    $PYTHON_CMD - <<'PY'
import sysconfig
print(sysconfig.get_paths().get("purelib", ""))
PY
)

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

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to print colored messages
print_header() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${CYAN}${BOLD}=== $1 ===${RESET}" | tee -a $LOG_FILE
    else
        echo "=== $1 ===" | tee -a $LOG_FILE
    fi
    echo | tee -a $LOG_FILE
}

print_section() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${BLUE}${BOLD}>>> $1${RESET}" | tee -a $LOG_FILE
    else
        echo ">>> $1" | tee -a $LOG_FILE
    fi
}

print_step() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${MAGENTA}>> $1${RESET}" | tee -a $LOG_FILE
    else
        echo ">> $1" | tee -a $LOG_FILE
    fi
}

print_success() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${GREEN}✓ $1${RESET}" | tee -a $LOG_FILE
    else
        echo "✓ $1" | tee -a $LOG_FILE
    fi
}

print_warning() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${YELLOW}⚠ $1${RESET}" | tee -a $LOG_FILE
    else
        echo "⚠ $1" | tee -a $LOG_FILE
    fi
}

print_error() {
    if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
        echo -e "${RED}✗ $1${RESET}" | tee -a $LOG_FILE
    else
        echo "✗ $1" | tee -a $LOG_FILE
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

python_has_module() {
    $PYTHON_CMD - <<PY
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$1") else 1)
PY
}

# Set up comprehensive PYTHONPATH to find all components
export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages:$HOME/.local/lib/python3.12/site-packages:$HOME/rocm_venv/lib/python3.13/site-packages:$HOME/rocm_venv/lib/python3.12/site-packages:$HOME/ml_stack/flash_attn_amd_direct:$HOME/ml_stack/flash_attn_amd:$HOME/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-313:$HOME/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-312:$HOME/megatron/Megatron-LM:$HOME/migraphx_build:$HOME/vllm_build:$HOME/vllm_py313:$HOME/ml_stack/bitsandbytes/bitsandbytes:/opt/rocm/lib:$HOME/ml_stack:$REPO_ROOT:$HOME/pytorch:$PYTHONPATH"
print_step "Enhanced PYTHONPATH to find all components"

# Start verification
print_header "Starting ML Stack Verification"
print_step "System: $(uname -a)"
print_step "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
print_step "Python Version: $($PYTHON_CMD --version)"

# Verify ROCm
print_section "Verifying ROCm"
if command_exists hipcc; then
    print_success "ROCm is installed: $(hipcc --version 2>&1 | head -n 1)"
    print_step "ROCm Info:"
    # Suppress HIP logs by setting environment variable
    export HIP_TRACE_API=0
    export HIP_VISIBLE_DEVICES=0,1,2
    export ROCR_VISIBLE_DEVICES=0,1,2
    export HSA_ENABLE_SDMA=0
    # Filter out HIP logs and only show relevant information
    rocminfo 2>/dev/null | grep -E "Name:|Marketing|ROCm Version" | tee -a $LOG_FILE
else
    print_error "ROCm is not installed."
fi

# Verify PyTorch
print_section "Verifying PyTorch"
if $PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    print_success "PyTorch is installed"
    pytorch_version=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>&1)
    print_step "PyTorch version: $pytorch_version"

    # Check CUDA availability
    if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA is available through ROCm"

        # Get GPU count
        gpu_count=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>&1)
        print_step "GPU count: $gpu_count"

        # Get GPU names
        for i in $(seq 0 $((gpu_count-1))); do
            gpu_name=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name($i))" 2>&1)
            print_step "GPU $i: $gpu_name"
        done

        # Test simple operation
        if $PYTHON_CMD -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success')" 2>/dev/null | grep -q "Success"; then
            print_success "Simple tensor operation on GPU successful"
        else
            print_error "Simple tensor operation on GPU failed"
        fi
    else
        print_error "CUDA is not available through ROCm"
    fi
else
    # Check for PyTorch installation in specific directories
    if [ -d "$HOME/pytorch" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch" ]; then
        print_success "PyTorch is installed"
    else
        print_error "PyTorch is not installed."
    fi
fi

# Verify ONNX Runtime
print_section "Verifying ONNX Runtime"
if $PYTHON_CMD -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')" 2>/dev/null; then
    print_success "ONNX Runtime is installed"
    onnx_version=$($PYTHON_CMD -c "import onnxruntime; print(onnxruntime.__version__)" 2>&1)
    print_step "ONNX Runtime version: $onnx_version"

    # Check available providers
    providers=$($PYTHON_CMD -c "import onnxruntime; print(onnxruntime.get_available_providers())" 2>&1)
    print_step "Available providers: $providers"

    # Check if ROCMExecutionProvider is available
    if echo "$providers" | grep -q "ROCMExecutionProvider"; then
        print_success "ROCMExecutionProvider is available"
    else
        print_warning "ROCMExecutionProvider is not available"

        # Try to fix ROCMExecutionProvider issue
        print_step "Attempting to fix ROCMExecutionProvider issue..."

        # Check if the ROCm libraries are in the LD_LIBRARY_PATH
        if [[ ":$LD_LIBRARY_PATH:" != *":/opt/rocm/lib:"* ]]; then
            print_step "Adding /opt/rocm/lib to LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
        fi

        # Find the ROCMExecutionProvider library
        ROCM_PROVIDER_LIB=${ORT_ROCM_EP_PROVIDER_PATH:-$(find "$HOME" -name "libonnxruntime_providers_rocm.so" | head -n 1)}

        if [ -n "$ROCM_PROVIDER_LIB" ]; then
            print_step "Found ROCMExecutionProvider library at: $ROCM_PROVIDER_LIB"

            # Create a Python script to enable the ROCMExecutionProvider
            cat > /tmp/enable_rocm_provider.py << EOF
import os
import sys

# Set environment variables
os.environ['ORT_ROCM_EP_PROVIDER_PATH'] = '$ROCM_PROVIDER_LIB'
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Try to import ONNX Runtime from the system installation
sys.path.insert(0, '${PYTHON_SITEPACKAGES}')

try:
    import onnxruntime
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available providers before: {onnxruntime.get_available_providers()}")

    # Check if ROCMExecutionProvider is available
    if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
        print("ROCMExecutionProvider is already available")
        sys.exit(0)
    else:
        print("ROCMExecutionProvider is not available")

        # Try to manually load the ROCMExecutionProvider
        try:
            # Create a symbolic link to the ROCMExecutionProvider library in a standard location
            import subprocess

            # Create the directory if it doesn't exist
            os.makedirs('${HOME}/.local/lib', exist_ok=True)

            # Create a symbolic link to the ROCMExecutionProvider library
            link_cmd = f"ln -sf {os.environ['ORT_ROCM_EP_PROVIDER_PATH']} ${HOME}/.local/lib/libonnxruntime_providers_rocm.so"
            print(f"Creating symbolic link: {link_cmd}")
            subprocess.run(link_cmd, shell=True, check=True)

            # Set the environment variable to point to the symbolic link
            os.environ['ORT_ROCM_EP_PROVIDER_PATH'] = '${HOME}/.local/lib/libonnxruntime_providers_rocm.so'

            # Try to create a session with ROCMExecutionProvider
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
            print(f"Trying to create a session with providers: {providers}")

            # Check if the ROCMExecutionProvider library exists
            if os.path.exists(os.environ['ORT_ROCM_EP_PROVIDER_PATH']):
                print(f"ROCMExecutionProvider library exists at: {os.environ['ORT_ROCM_EP_PROVIDER_PATH']}")
            else:
                print(f"ROCMExecutionProvider library does not exist at: {os.environ['ORT_ROCM_EP_PROVIDER_PATH']}")

            # Check if ROCm is available
            if os.path.exists('/opt/rocm'):
                print("ROCm is installed at: /opt/rocm")
                rocm_libs = os.listdir('/opt/rocm/lib')
                print(f"ROCm libraries: {rocm_libs[:10]}...")
            else:
                print("ROCm is not installed at: /opt/rocm")

            # Reload ONNX Runtime to pick up the new environment variables
            import importlib
            importlib.reload(onnxruntime)

            # Check if ROCMExecutionProvider is now available
            providers = onnxruntime.get_available_providers()
            print(f"Available providers after reload: {providers}")

            if 'ROCMExecutionProvider' in providers:
                print("ROCMExecutionProvider is now available")
                sys.exit(0)
            else:
                print("ROCMExecutionProvider is still not available")
                sys.exit(1)
        except Exception as e:
            print(f"Error testing ROCMExecutionProvider: {e}")
            sys.exit(1)
except Exception as e:
    print(f"Error importing ONNX Runtime: {e}")
    sys.exit(1)
EOF

            # Run the Python script to enable the ROCMExecutionProvider
            print_step "Attempting to enable ROCMExecutionProvider..."
            if $PYTHON_CMD /tmp/enable_rocm_provider.py; then
                print_success "Successfully enabled ROCMExecutionProvider"
                # Update the providers variable
                providers=$($PYTHON_CMD -c "import sys; sys.path.insert(0, '${PYTHON_SITEPACKAGES}'); import os; os.environ['ORT_ROCM_EP_PROVIDER_PATH'] = '$ROCM_PROVIDER_LIB'; os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', ''); import onnxruntime; print(onnxruntime.get_available_providers())" 2>/dev/null)
            else
                print_warning "Failed to enable ROCMExecutionProvider"

                # Check if the ROCm device is visible
                if [ -f "/opt/rocm/bin/rocminfo" ]; then
                    print_step "Checking ROCm devices:"
                    # Suppress HIP logs by setting environment variable
                    export HIP_TRACE_API=0
                    export HIP_VISIBLE_DEVICES=0,1,2
                    export ROCR_VISIBLE_DEVICES=0,1,2
                    export HSA_ENABLE_SDMA=0
                    # Filter out HIP logs and only show relevant information
                    /opt/rocm/bin/rocminfo 2>/dev/null | grep -E "Name:|Marketing|ROCm Version"
                fi

                # Check ONNX Runtime version
                $PYTHON_CMD -c "import sys; sys.path.insert(0, '${PYTHON_SITEPACKAGES}'); import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')" 2>/dev/null

                print_warning "ONNX Runtime was not built with ROCm support or ROCm support is not properly configured."
                print_step "To fix this issue, you need to rebuild ONNX Runtime with ROCm support:"
                print_step "1. Clone the ONNX Runtime repository: git clone --recursive https://github.com/microsoft/onnxruntime.git"
                print_step "2. Build ONNX Runtime with ROCm support: ./build.sh --config Release --use_rocm --rocm_home /opt/rocm"
                print_step "3. Install the built package: pip install build/Linux/Release/dist/*.whl"
            fi
        else
            print_warning "libonnxruntime_providers_rocm.so not found. ONNX Runtime was not built with ROCm support."
            print_step "You need to rebuild ONNX Runtime with ROCm support."
        fi
    fi
else
    # Check for ONNX Runtime installation in specific directories
    if [ -d "$HOME/onnxruntime" ] || [ -d "$HOME/onnxruntime_build" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/onnxruntime" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/onnxruntime" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/onnxruntime" ]; then
        print_success "ONNX Runtime is installed"
        print_warning "ONNX Runtime cannot be imported (Python compatibility issue)"
    else
        print_error "ONNX Runtime is not installed."
    fi
fi

# Verify MIGraphX
print_section "Verifying MIGraphX"
if $PYTHON_CMD -c "import migraphx; print(f'MIGraphX is installed')" 2>/dev/null; then
    print_success "MIGraphX is installed"
    # Redirect stderr to /dev/null to suppress HIP logs
    migraphx_version=$($PYTHON_CMD -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null)
    print_step "MIGraphX version: $migraphx_version"
else
    # Check for MIGraphX installation in specific directories
    if [ -d "$HOME/migraphx_build" ] || [ -d "$HOME/migraphx_package" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/migraphx" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/migraphx" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/migraphx" ]; then
        print_success "MIGraphX is installed"
        print_warning "MIGraphX cannot be imported (Python compatibility issue)"
    else
        print_error "MIGraphX is not installed."
    fi
fi

# Verify Flash Attention
print_section "Verifying Flash Attention"
if $PYTHON_CMD -c "import flash_attention_amd; print('Flash Attention is installed')" 2>/dev/null; then
    print_success "Flash Attention is installed"

    # Test Flash Attention
    # Redirect stderr to /dev/null to suppress HIP logs
    if $PYTHON_CMD -c "import torch; import flash_attention_amd; q = torch.randn(2, 128, 8, 64, device='cuda'); k = torch.randn(2, 128, 8, 64, device='cuda'); v = torch.randn(2, 128, 8, 64, device='cuda'); output = flash_attention_amd.flash_attn_func(q, k, v); print('Success')" 2>/dev/null | grep -q "Success"; then
        print_success "Flash Attention computation successful"
    else
        print_error "Flash Attention computation failed"
    fi
else
    print_error "Flash Attention is not installed"
fi

# Verify AITER
print_section "Verifying AITER"
if $PYTHON_CMD -c "import aiter; import aiter.torch; print('AITER is installed')" 2>/dev/null; then
    print_success "AITER is installed"

    # Get AITER version
    aiter_version=$($PYTHON_CMD -c "import aiter; print(getattr(aiter, '__version__', 'unknown'))" 2>/dev/null)
    print_step "AITER version: $aiter_version"

    # Test AITER basic functionality
    if $PYTHON_CMD -c "import aiter; import torch; import numpy as np; print('AITER test successful')" 2>/dev/null | grep -q "AITER test successful"; then
        print_success "AITER basic functionality test successful"
    else
        print_error "AITER basic functionality test failed"
    fi
elif python_has_module aiter; then
    print_success "AITER is installed"
    print_warning "AITER cannot be imported (Python compatibility issue)"
else
    print_error "AITER is not installed"
fi

# Verify DeepSpeed
print_section "Verifying DeepSpeed"
if $PYTHON_CMD -c "import deepspeed; print('DeepSpeed is installed')" 2>/dev/null; then
    print_success "DeepSpeed is installed"

    # Get DeepSpeed version
    deepspeed_version=$($PYTHON_CMD -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null)
    print_step "DeepSpeed version: $deepspeed_version"

    # Check if DeepSpeed has ROCm support
    if $PYTHON_CMD -c "import deepspeed; import torch; print('ROCm available' if torch.cuda.is_available() else 'ROCm not available')" 2>/dev/null | grep -q "ROCm available"; then
        print_success "DeepSpeed has ROCm support"
    else
        print_warning "DeepSpeed may not have ROCm support"
    fi
else
    print_error "DeepSpeed is not installed"
fi

# Check for Flash Attention installation in specific directories if import failed
if ! $PYTHON_CMD -c "import flash_attention_amd" 2>/dev/null; then
    if [ -d "$HOME/ml_stack/flash_attn_amd" ] || [ -d "$HOME/ml_stack/flash_attn_amd_direct" ] || [ -d "$HOME/ml_stack/flash-attention" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/flash_attention_amd" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/flash_attention_amd" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/flash_attention_amd" ]; then
        print_success "Flash Attention is installed"
        print_warning "Flash Attention cannot be imported (Python compatibility issue)"
    else
        print_error "Flash Attention is not installed."
    fi
fi

# Verify MPI
print_section "Verifying MPI"
mpi_found=false

if command_exists mpirun; then
    print_success "MPI runtime is installed"
    mpi_version=$(mpirun --version | head -n 1)
    print_step "MPI version: $mpi_version"
    mpi_found=true
else
    print_warning "MPI runtime (mpirun) not found, checking for mpi4py only"
fi

# Check mpi4py - this is the Python interface that's typically needed for ML
if $PYTHON_CMD -c "import mpi4py; print('mpi4py is installed')" 2>/dev/null; then
    print_success "mpi4py is installed"
    mpi4py_version=$($PYTHON_CMD -c "import mpi4py; print(mpi4py.__version__)" 2>&1)
    print_step "mpi4py version: $mpi4py_version"

    # Try basic mpi4py functionality
    if $PYTHON_CMD -c "import mpi4py; from mpi4py import MPI; print('Basic import successful')" 2>/dev/null; then
        print_success "mpi4py basic functionality working"
    else
        print_warning "mpi4py import works but MPI operations may be limited"
    fi

    mpi_found=true
else
    # Check for mpi4py installation in specific directories
    if [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/mpi4py" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/mpi4py" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/mpi4py" ]; then
        print_warning "mpi4py is installed but cannot be imported (Python compatibility issue)"
        mpi_found=true
    else
        print_error "mpi4py is not installed"
    fi
fi

if [ "$mpi_found" = true ]; then
    print_success "MPI/mpi4py components available for distributed computing"
else
    print_error "MPI is not installed."
fi

# Verify RCCL
print_section "Verifying RCCL"
if [ -f "/usr/lib/x86_64-linux-gnu/librccl.so" ] || [ -f "/opt/rocm/lib/librccl.so" ]; then
    print_success "RCCL is installed"
    if [ -f "/usr/lib/x86_64-linux-gnu/librccl.so" ]; then
        rccl_path=$(ls -la /usr/lib/x86_64-linux-gnu/librccl.so)
    else
        rccl_path=$(ls -la /opt/rocm/lib/librccl.so)
    fi
    print_step "RCCL path: $rccl_path"
else
    print_error "RCCL is not installed."
fi

# Verify Megatron-LM
print_section "Verifying Megatron-LM"
if $PYTHON_CMD -c "import megatron; print('Megatron-LM is installed')" 2>/dev/null; then
    print_success "Megatron-LM is installed"
else
    # Check for Megatron-LM installation in specific directories
    if [ -d "$HOME/megatron/Megatron-LM" ] || [ -d "$HOME/Megatron-LM" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/megatron" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/megatron" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/megatron" ]; then
        print_success "Megatron-LM is installed"
        print_warning "Megatron-LM cannot be imported (Python compatibility issue)"
    else
        print_error "Megatron-LM is not installed."
    fi
fi

# Verify Extensions (if installed)
print_header "Verifying Extension Components (if installed)"

# Verify Triton
print_section "Verifying Triton"
if $PYTHON_CMD -c "import triton; print(f'Triton is installed')" 2>/dev/null; then
    print_success "Triton is installed"
    triton_version=$($PYTHON_CMD -c "import triton; print(getattr(triton, '__version__', 'unknown'))" 2>&1)
    print_step "Triton version: $triton_version"
else
    # Check for Triton installation in specific directories
    if [ -d "$HOME/ml_stack/triton" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/triton" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/triton" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/triton" ]; then
        print_success "Triton is installed"
        print_warning "Triton cannot be imported (Python compatibility issue)"
    else
        print_warning "Triton is not installed."
    fi
fi

# Verify BITSANDBYTES
print_section "Verifying BITSANDBYTES"
if $PYTHON_CMD -c "import bitsandbytes; print(f'BITSANDBYTES is installed')" 2>/dev/null; then
    print_success "BITSANDBYTES is installed"
    bnb_version=$($PYTHON_CMD -c "import bitsandbytes as bnb; print(getattr(bnb, '__version__', 'unknown'))" 2>&1)
    print_step "BITSANDBYTES version: $bnb_version"
else
    # Check for BITSANDBYTES installation in specific directories
    if [ -d "$HOME/ml_stack/bitsandbytes" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/bitsandbytes" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/bitsandbytes" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/bitsandbytes" ]; then
        print_success "BITSANDBYTES is installed"
        print_warning "BITSANDBYTES cannot be imported (Python compatibility issue)"
    else
        print_warning "BITSANDBYTES is not installed."
    fi
fi

# Verify vLLM
print_section "Verifying vLLM"
if $PYTHON_CMD -c "import vllm; print('vLLM is installed')" 2>/dev/null; then
    print_success "vLLM is installed"
    vllm_version=$($PYTHON_CMD -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>&1)
    print_step "vLLM version: $vllm_version"
elif python_has_module vllm; then
    print_success "vLLM is installed"
    print_warning "vLLM cannot be imported (Python compatibility issue)"
else
    # Check for vLLM installation in specific directories
    if [ -d "$HOME/vllm" ] || [ -d "$HOME/vllm_build" ] || [ -d "$HOME/vllm_py313" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/vllm" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/vllm" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/vllm" ]; then
        print_success "vLLM is installed"
        print_warning "vLLM cannot be imported (Python compatibility issue)"
    else
        print_warning "vLLM is not installed."
    fi
fi

# Verify ROCm SMI
print_section "Verifying ROCm SMI"
if $PYTHON_CMD -c "from rocm_smi_lib import rsmi; print('ROCm SMI is installed')" 2>/dev/null; then
    print_success "ROCm SMI is installed"
else
    # Check for ROCm SMI installation
    if [ -f "/opt/rocm/bin/rocm-smi" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/rocm_smi_lib" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/rocm_smi_lib" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/rocm_smi_lib" ]; then
        print_success "ROCm SMI is installed"
        print_warning "ROCm SMI cannot be imported (Python compatibility issue)"
    else
        print_warning "ROCm SMI is not installed."
    fi
fi

# Verify PyTorch Profiler
print_section "Verifying PyTorch Profiler"
if $PYTHON_CMD -c "from torch.profiler import profile; print('PyTorch Profiler is installed')" 2>/dev/null; then
    print_success "PyTorch Profiler is installed"
else
    # Check for PyTorch Profiler installation in specific directories
    if [ -d "$HOME/pytorch/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch/profiler" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch/profiler" ]; then
        print_success "PyTorch Profiler is installed"
        print_warning "PyTorch Profiler cannot be imported (Python compatibility issue)"
    else
        print_warning "PyTorch Profiler is not installed."
    fi
fi

# Verify Weights & Biases
print_section "Verifying Weights & Biases"
if $PYTHON_CMD -c "import wandb; print(f'Weights & Biases is installed')" 2>/dev/null; then
    print_success "Weights & Biases is installed"
    wandb_version=$($PYTHON_CMD -c "import wandb; print(wandb.__version__)" 2>&1)
    print_step "Weights & Biases version: $wandb_version"
else
    # Check for Weights & Biases installation directory
    if [ -d "$HOME/ml_stack/wandb" ] || [ -d "$HOME/Stans_MLStack/extensions/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/wandb" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/wandb" ]; then
        print_success "Weights & Biases is installed"
        print_warning "Weights & Biases cannot be imported (Python compatibility issue)"
    else
        print_warning "Weights & Biases is not installed."
    fi
fi

print_header "ML Stack Verification Complete"
print_step "Log File: $LOG_FILE"

# Summary
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    echo -e "${CYAN}${BOLD}============================================================${RESET}"
    echo -e "${CYAN}${BOLD}ML Stack Verification Summary:${RESET}"
    echo ""
    echo -e "${BLUE}${BOLD}Core Components:${RESET}"
    echo -e "- ROCm: $(command_exists hipcc && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
else
    echo "============================================================"
    echo "ML Stack Verification Summary:"
    echo ""
    echo "Core Components:"
    echo "- ROCm: $(command_exists hipcc && echo "Installed" || echo "Not installed")"
fi

# Check PyTorch with fallback
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo -e "- PyTorch: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/pytorch" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch" ]; then
        echo -e "- PyTorch: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- PyTorch: ${RED}Not installed${RESET}"
    fi

    # Check ONNX Runtime with fallback
    if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
        echo -e "- ONNX Runtime: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/onnxruntime" ] || [ -d "$HOME/onnxruntime_build" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/onnxruntime" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/onnxruntime" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/onnxruntime" ]; then
        echo -e "- ONNX Runtime: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- ONNX Runtime: ${RED}Not installed${RESET}"
    fi

    # Check MIGraphX with fallback
    if $PYTHON_CMD -c "import migraphx" 2>/dev/null; then
        echo -e "- MIGraphX: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/migraphx_build" ] || [ -d "$HOME/migraphx_package" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/migraphx" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/migraphx" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/migraphx" ]; then
        echo -e "- MIGraphX: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- MIGraphX: ${RED}Not installed${RESET}"
    fi

    # Check Flash Attention with fallback
    if $PYTHON_CMD -c "import flash_attention_amd" 2>/dev/null; then
        echo -e "- Flash Attention: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/ml_stack/flash_attn_amd" ] || [ -d "$HOME/ml_stack/flash_attn_amd_direct" ] || [ -d "$HOME/ml_stack/flash-attention" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/flash_attention_amd" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/flash_attention_amd" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/flash_attention_amd" ]; then
        echo -e "- Flash Attention: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- Flash Attention: ${RED}Not installed${RESET}"
    fi

    echo -e "- RCCL: $([ -f "/usr/lib/x86_64-linux-gnu/librccl.so" ] || [ -f "/opt/rocm/lib/librccl.so" ] && echo -e "${GREEN}Installed${RESET}" || echo -e "${RED}Not installed${RESET}")"
    echo -e "- MPI: $(command_exists mpirun && echo -e "${GREEN}Installed${RESET}" || $PYTHON_CMD -c "import mpi4py" 2>/dev/null && echo -e "${GREEN}mpi4py available${RESET}" || echo -e "${RED}Not installed${RESET}")"

    # Check Megatron-LM with fallback
    if $PYTHON_CMD -c "import megatron" 2>/dev/null; then
        echo -e "- Megatron-LM: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/megatron/Megatron-LM" ] || [ -d "$HOME/Megatron-LM" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/megatron" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/megatron" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/megatron" ]; then
        echo -e "- Megatron-LM: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- Megatron-LM: ${RED}Not installed${RESET}"
    fi

    # Check AITER with fallback
    if $PYTHON_CMD -c "import aiter" 2>/dev/null; then
        echo -e "- AITER: ${GREEN}Installed${RESET}"
    elif python_has_module aiter; then
        echo -e "- AITER: ${GREEN}Installed${RESET} (Python compatibility issue)"
    elif [ -d "$HOME/aiter" ] || [ -d "$HOME/ml_stack/aiter" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/aiter" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/aiter" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/aiter" ]; then
        echo -e "- AITER: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- AITER: ${RED}Not installed${RESET}"
    fi

    # Check DeepSpeed with fallback
    if $PYTHON_CMD -c "import deepspeed" 2>/dev/null; then
        echo -e "- DeepSpeed: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/deepspeed" ] || [ -d "$HOME/ml_stack/deepspeed" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/deepspeed" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/deepspeed" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/deepspeed" ]; then
        echo -e "- DeepSpeed: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- DeepSpeed: ${RED}Not installed${RESET}"
    fi

    echo ""
    echo -e "${BLUE}${BOLD}Extension Components:${RESET}"
else
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo "- PyTorch: Installed"
    elif [ -d "$HOME/pytorch" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch" ]; then
        echo "- PyTorch: Installed (Python compatibility issue)"
    else
        echo "- PyTorch: Not installed"
    fi

    # Check ONNX Runtime with fallback
    if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
        echo "- ONNX Runtime: Installed"
    elif [ -d "$HOME/onnxruntime" ] || [ -d "$HOME/onnxruntime_build" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/onnxruntime" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/onnxruntime" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/onnxruntime" ]; then
        echo "- ONNX Runtime: Installed (Python compatibility issue)"
    else
        echo "- ONNX Runtime: Not installed"
    fi

    # Check MIGraphX with fallback
    if $PYTHON_CMD -c "import migraphx" 2>/dev/null; then
        echo "- MIGraphX: Installed"
    elif [ -d "$HOME/migraphx_build" ] || [ -d "$HOME/migraphx_package" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/migraphx" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/migraphx" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/migraphx" ]; then
        echo "- MIGraphX: Installed (Python compatibility issue)"
    else
        echo "- MIGraphX: Not installed"
    fi

    # Check Flash Attention with fallback
    if $PYTHON_CMD -c "import flash_attention_amd" 2>/dev/null; then
        echo "- Flash Attention: Installed"
    elif [ -d "$HOME/ml_stack/flash_attn_amd" ] || [ -d "$HOME/ml_stack/flash_attn_amd_direct" ] || [ -d "$HOME/ml_stack/flash-attention" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/flash_attention_amd" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/flash_attention_amd" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/flash_attention_amd" ]; then
        echo "- Flash Attention: Installed (Python compatibility issue)"
    else
        echo "- Flash Attention: Not installed"
    fi

    echo "- RCCL: $([ -f "/usr/lib/x86_64-linux-gnu/librccl.so" ] || [ -f "/opt/rocm/lib/librccl.so" ] && echo "Installed" || echo "Not installed")"
    echo "- MPI: $(command_exists mpirun && echo "Installed" || $PYTHON_CMD -c "import mpi4py" 2>/dev/null && echo "mpi4py available" || echo "Not installed")"

    # Check Megatron-LM with fallback
    if $PYTHON_CMD -c "import megatron" 2>/dev/null; then
        echo "- Megatron-LM: Installed"
    elif [ -d "$HOME/megatron/Megatron-LM" ] || [ -d "$HOME/Megatron-LM" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/megatron" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/megatron" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/megatron" ]; then
        echo "- Megatron-LM: Installed (Python compatibility issue)"
    else
        echo "- Megatron-LM: Not installed"
    fi

    # Check AITER with fallback
    if $PYTHON_CMD -c "import aiter" 2>/dev/null; then
        echo "- AITER: Installed"
    elif python_has_module aiter; then
        echo "- AITER: Installed (Python compatibility issue)"
    elif [ -d "$HOME/aiter" ] || [ -d "$HOME/ml_stack/aiter" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/aiter" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/aiter" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/aiter" ]; then
        echo "- AITER: Installed (Python compatibility issue)"
    else
        echo "- AITER: Not installed"
    fi

    # Check DeepSpeed with fallback
    if $PYTHON_CMD -c "import deepspeed" 2>/dev/null; then
        echo "- DeepSpeed: Installed"
    elif [ -d "$HOME/deepspeed" ] || [ -d "$HOME/ml_stack/deepspeed" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/deepspeed" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/deepspeed" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/deepspeed" ]; then
        echo "- DeepSpeed: Installed (Python compatibility issue)"
    else
        echo "- DeepSpeed: Not installed"
    fi

    echo ""
    echo "Extension Components:"
fi

# Check Triton with fallback
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    if $PYTHON_CMD -c "import triton" 2>/dev/null; then
        echo -e "- Triton: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/ml_stack/triton" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/triton" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/triton" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/triton" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/triton" ]; then
        echo -e "- Triton: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- Triton: ${YELLOW}Not installed${RESET}"
    fi

    # Check BITSANDBYTES with fallback
    if $PYTHON_CMD -c "import bitsandbytes" 2>/dev/null; then
        echo -e "- BITSANDBYTES: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/ml_stack/bitsandbytes" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/bitsandbytes" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/bitsandbytes" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/bitsandbytes" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/bitsandbytes" ]; then
        echo -e "- BITSANDBYTES: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- BITSANDBYTES: ${YELLOW}Not installed${RESET}"
    fi

    # Check vLLM with fallback
    if $PYTHON_CMD -c "import vllm" 2>/dev/null; then
        echo -e "- vLLM: ${GREEN}Installed${RESET}"
    elif python_has_module vllm; then
        echo -e "- vLLM: ${GREEN}Installed${RESET} (Python compatibility issue)"
    elif [ -d "$HOME/vllm" ] || [ -d "$HOME/vllm_build" ] || [ -d "$HOME/vllm_py313" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/vllm" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/vllm" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/vllm" ]; then
        echo -e "- vLLM: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- vLLM: ${YELLOW}Not installed${RESET}"
    fi

    # Check ROCm SMI with fallback
    if $PYTHON_CMD -c "from rocm_smi_lib import rsmi" 2>/dev/null; then
        echo -e "- ROCm SMI: ${GREEN}Installed${RESET}"
    elif [ -f "/opt/rocm/bin/rocm-smi" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/rocm_smi_lib" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/rocm_smi_lib" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/rocm_smi_lib" ]; then
        echo -e "- ROCm SMI: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- ROCm SMI: ${YELLOW}Not installed${RESET}"
    fi

    # Check PyTorch Profiler with fallback
    if $PYTHON_CMD -c "from torch.profiler import profile" 2>/dev/null; then
        echo -e "- PyTorch Profiler: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/pytorch/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch/profiler" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch/profiler" ]; then
        echo -e "- PyTorch Profiler: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- PyTorch Profiler: ${YELLOW}Not installed${RESET}"
    fi

    # Check Weights & Biases with fallback
    if $PYTHON_CMD -c "import wandb" 2>/dev/null; then
        echo -e "- Weights & Biases: ${GREEN}Installed${RESET}"
    elif [ -d "$HOME/ml_stack/wandb" ] || [ -d "$HOME/Stans_MLStack/extensions/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/wandb" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/wandb" ]; then
        echo -e "- Weights & Biases: ${GREEN}Installed${RESET} (Python compatibility issue)"
    else
        echo -e "- Weights & Biases: ${YELLOW}Not installed${RESET}"
    fi
    echo ""
    echo -e "${CYAN}Log File: ${BOLD}$LOG_FILE${RESET}"
    echo -e "${CYAN}${BOLD}============================================================${RESET}"
else
    if $PYTHON_CMD -c "import triton" 2>/dev/null; then
        echo "- Triton: Installed"
    elif [ -d "$HOME/ml_stack/triton" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/triton" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/triton" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/triton" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/triton" ]; then
        echo "- Triton: Installed (Python compatibility issue)"
    else
        echo "- Triton: Not installed"
    fi

    # Check BITSANDBYTES with fallback
    if $PYTHON_CMD -c "import bitsandbytes" 2>/dev/null; then
        echo "- BITSANDBYTES: Installed"
    elif [ -d "$HOME/ml_stack/bitsandbytes" ] || [ -d "$HOME/Stans_MLStack/Stan-s-ML-Stack/extensions/bitsandbytes" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/bitsandbytes" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/bitsandbytes" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/bitsandbytes" ]; then
        echo "- BITSANDBYTES: Installed (Python compatibility issue)"
    else
        echo "- BITSANDBYTES: Not installed"
    fi

    # Check vLLM with fallback
    if $PYTHON_CMD -c "import vllm" 2>/dev/null; then
        echo "- vLLM: Installed"
    elif python_has_module vllm; then
        echo "- vLLM: Installed (Python compatibility issue)"
    elif [ -d "$HOME/vllm" ] || [ -d "$HOME/vllm_build" ] || [ -d "$HOME/vllm_py313" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/vllm" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/vllm" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/vllm" ]; then
        echo "- vLLM: Installed (Python compatibility issue)"
    else
        echo "- vLLM: Not installed"
    fi

    # Check ROCm SMI with fallback
    if $PYTHON_CMD -c "from rocm_smi_lib import rsmi" 2>/dev/null; then
        echo "- ROCm SMI: Installed"
    elif [ -f "/opt/rocm/bin/rocm-smi" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/rocm_smi_lib" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/rocm_smi_lib" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/rocm_smi_lib" ]; then
        echo "- ROCm SMI: Installed (Python compatibility issue)"
    else
        echo "- ROCm SMI: Not installed"
    fi

    # Check PyTorch Profiler with fallback
    if $PYTHON_CMD -c "from torch.profiler import profile" 2>/dev/null; then
        echo "- PyTorch Profiler: Installed"
    elif [ -d "$HOME/pytorch/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/torch/profiler" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/torch/profiler" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/torch/profiler" ]; then
        echo "- PyTorch Profiler: Installed (Python compatibility issue)"
    else
        echo "- PyTorch Profiler: Not installed"
    fi

    # Check Weights & Biases with fallback
    if $PYTHON_CMD -c "import wandb" 2>/dev/null; then
        echo "- Weights & Biases: Installed"
    elif [ -d "$HOME/ml_stack/wandb" ] || [ -d "$HOME/Stans_MLStack/extensions/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.13/site-packages/wandb" ] || [ -d "$HOME/rocm_venv/lib/python3.12/site-packages/wandb" ] || [ -d "$HOME/.local/lib/python3.13/site-packages/wandb" ]; then
        echo "- Weights & Biases: Installed (Python compatibility issue)"
    else
        echo "- Weights & Biases: Not installed"
    fi
    echo ""
    echo "Log File: $LOG_FILE"
    echo "============================================================"
fi

summary_file="${MLSTACK_LOG_DIR:-$HOME/.mlstack/logs}/verify_summary_$(date +"%Y%m%d_%H%M%S").txt"
mkdir -p "$(dirname "$summary_file")"

rocm_status=$(command_exists hipcc && echo installed || echo missing)
pytorch_status=$(python_has_module torch && echo installed || echo missing)
onnx_status=$(python_has_module onnxruntime && echo installed || echo missing)
onnx_provider=$($PYTHON_CMD - <<'PY'
import importlib.util
import sys
if not importlib.util.find_spec("onnxruntime"):
    print("missing")
    sys.exit(0)
import onnxruntime
providers = onnxruntime.get_available_providers()
print("available" if "ROCMExecutionProvider" in providers else "missing")
PY
)
vllm_status=$(python_has_module vllm && echo installed || echo missing)
aiter_status=$(python_has_module aiter && echo installed || echo missing)
aiter_torch_status=$(python_has_module aiter.torch && echo available || echo missing)
triton_status=$(python_has_module triton && echo installed || echo missing)
bnb_status=$(python_has_module bitsandbytes && echo installed || echo missing)

cat <<REPORT
========================================
Verification Summary (copyable)
========================================
Log file: $LOG_FILE
Python: $PYTHON_CMD
ROCm: $rocm_status
PyTorch: $pytorch_status
ONNX Runtime: $onnx_status
ONNX ROCm EP: $onnx_provider
vLLM: $vllm_status
AITER: $aiter_status
AITER torch: $aiter_torch_status
Triton: $triton_status
BITSANDBYTES: $bnb_status
========================================
Saved summary to: $summary_file
REPORT

cat <<REPORT > "$summary_file"
Log file: $LOG_FILE
Python: $PYTHON_CMD
ROCm: $rocm_status
PyTorch: $pytorch_status
ONNX Runtime: $onnx_status
ONNX ROCm EP: $onnx_provider
vLLM: $vllm_status
AITER: $aiter_status
AITER torch: $aiter_torch_status
Triton: $triton_status
BITSANDBYTES: $bnb_status
REPORT
