#!/bin/bash
# Stan's ML Stack - vLLM ROCm installer (channel-aware)
# Uses official vLLM ROCm wheels to avoid build complexity
# IMPORTANT: Protects against NVIDIA torch being pulled in

set -euo pipefail

# Source utility scripts if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi
if [ -f "$SCRIPT_DIR/env_validation_utils.sh" ]; then
    source "$SCRIPT_DIR/env_validation_utils.sh"
fi

# Dry run flag check
DRY_RUN=${DRY_RUN:-false}
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

# Require and validate .mlstack_env (fallback to defaults if missing)
if type validate_mlstack_env >/dev/null 2>&1; then
    validate_mlstack_env "$(basename "$0")" || true
elif type require_mlstack_env >/dev/null 2>&1; then
    require_mlstack_env "$(basename "$0")" || true
fi

ROCM_VERSION=${ROCM_VERSION:-$(cat /opt/rocm/.info/version 2>/dev/null | head -n1 || echo "7.2.0")}
ROCM_CHANNEL=${ROCM_CHANNEL:-latest}
GPU_ARCH=${GPU_ARCH:-gfx1100}

# Attempt auto-detection if possible
if command -v rocminfo >/dev/null 2>&1; then
    detected_arch=$(rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -n1 || true)
    if [ -n "$detected_arch" ]; then
        GPU_ARCH="$detected_arch"
        echo "➤ Detected GPU_ARCH: $GPU_ARCH"
    fi
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ Installing vLLM for ROCm $ROCM_VERSION ($ROCM_CHANNEL)"
echo "└─────────────────────────────────────────────────────────┘"

# Set up ROCm environment variables
export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HIP_ROOT_DIR="$ROCM_HOME"
export PYTORCH_ROCM_ARCH="$GPU_ARCH"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export VLLM_TARGET_DEVICE=rocm

# CRITICAL: Verify ROCm PyTorch is installed BEFORE installing vLLM
echo "➤ Verifying ROCm PyTorch installation..."

# First check if torch can even be imported (might fail if wrong MPI libs)
if ! python3 -c "import torch" 2>/dev/null; then
    echo "✗ ERROR: PyTorch cannot be imported!"
    echo "  This usually means missing system libraries (e.g., libmpi_cxx.so.40)"
    echo "  or the wrong PyTorch version is installed."
    echo ""
    echo "  Current torch installation may be corrupted. Attempting to fix..."
    $PYTHON_BIN -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
    echo "  Installing ROCm PyTorch from AMD repo..."
    $PYTHON_BIN -m pip install \
        torch torchvision torchaudio triton \
        --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
        --break-system-packages --no-cache-dir

    if ! python3 -c "import torch" 2>/dev/null; then
        echo "✗ Failed to fix PyTorch. Please run: ./scripts/install_pytorch_multi.sh"
        exit 1
    fi
fi

# Now verify it's actually ROCm torch
if ! python3 -c "import torch; assert hasattr(torch.version, 'hip') or 'rocm' in torch.__version__.lower(), 'Not ROCm torch'" 2>/dev/null; then
    echo "✗ ERROR: ROCm PyTorch not detected!"
    echo "  Found: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')"
    echo "  vLLM installation requires ROCm PyTorch to be installed first."
    echo ""
    echo "  Attempting to install correct ROCm PyTorch..."
    $PYTHON_BIN -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
    $PYTHON_BIN -m pip install \
        torch torchvision torchaudio triton \
        --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
        --break-system-packages --no-cache-dir

    if ! python3 -c "import torch; assert hasattr(torch.version, 'hip') or 'rocm' in torch.__version__.lower()" 2>/dev/null; then
        echo "✗ Failed to install ROCm PyTorch. Please run: ./scripts/install_pytorch_multi.sh"
        exit 1
    fi
fi

# Record current torch version to verify it wasn't changed
TORCH_VERSION_BEFORE=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "✓ ROCm PyTorch detected: $TORCH_VERSION_BEFORE"

# Check if uv is available (preferred) or fall back to pip
if command -v uv >/dev/null 2>&1; then
    echo "➤ Using uv for installation (faster, recommended)"
    INSTALL_CMD="uv pip install"
else
    echo "➤ Using pip for installation"
    INSTALL_CMD="$PYTHON_BIN -m pip install"
fi

# Install vLLM from official ROCm wheels
# This is the simplified command that works reliably for ROCm 7.x
# The ROCm wheel index prioritizes ROCm-compatible packages
echo "➤ Installing vLLM from official ROCm wheels..."

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY-RUN] Would execute: $INSTALL_CMD vllm --extra-index-url https://wheels.vllm.ai/rocm/ --break-system-packages"
else
    # CRITICAL: Use --no-deps to prevent vLLM from replacing ROCm torch with its bundled version
    echo "➤ Installing vLLM with --no-deps to protect ROCm PyTorch..."
    if $INSTALL_CMD vllm --extra-index-url https://wheels.vllm.ai/rocm/ --no-deps --break-system-packages 2>&1 | tee /tmp/vllm_install.log; then
        echo "✓ vLLM core installed successfully"
    else
        echo "✗ vLLM installation failed"
        echo "➤ Attempting fallback with --no-cache-dir..."
        if $INSTALL_CMD vllm --extra-index-url https://wheels.vllm.ai/rocm/ --no-deps --break-system-packages --no-cache-dir; then
            echo "✓ vLLM core installed successfully (fallback)"
        else
            echo "✗ vLLM installation failed. Please check your ROCm installation."
            exit 1
        fi
    fi

    # Now install vLLM's dependencies EXCEPT torch/torchvision/torchaudio
    echo "➤ Installing vLLM dependencies (excluding torch and xformers)..."
    # These are the common vLLM dependencies that don't conflict with ROCm torch
    # NOTE: xformers is EXCLUDED because it pulls in NVIDIA CUDA PyTorch!
    # For ROCm, flash-attention-ck provides equivalent functionality
    VLLM_DEPS="accelerate aiohttp cloudpickle fastapi msgspec prometheus-client psutil py-cpuinfo pyzmq ray requests sentencepiece tiktoken uvicorn"
    for dep in $VLLM_DEPS; do
        $INSTALL_CMD "$dep" --break-system-packages 2>/dev/null || true
    done
    echo "✓ vLLM dependencies installed"

    # CRITICAL: Verify ROCm PyTorch wasn't overwritten
    echo "➤ Verifying ROCm PyTorch wasn't overwritten..."
    TORCH_VERSION_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

    if ! python3 -c "import torch; assert hasattr(torch.version, 'hip') or 'rocm' in torch.__version__.lower()" 2>/dev/null; then
        echo "✗ CRITICAL ERROR: ROCm PyTorch was overwritten!"
        echo "  Before: $TORCH_VERSION_BEFORE"
        echo "  After:  $TORCH_VERSION_AFTER"
        echo ""
        echo "  Attempting to restore ROCm PyTorch from ROCm 7.2 wheels..."
        $PYTHON_BIN -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
        # Use the correct ROCm 7.2 wheel URL
        $PYTHON_BIN -m pip install \
            torch torchvision torchaudio triton \
            --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
            --break-system-packages --no-cache-dir
        echo ""
        echo "  ROCm PyTorch restored. Verifying..."
        if python3 -c "import torch; assert hasattr(torch.version, 'hip') or 'rocm' in torch.__version__.lower()"; then
            echo "✓ ROCm PyTorch restored successfully"
        else
            echo "✗ Failed to restore ROCm PyTorch. Please reinstall manually."
            exit 1
        fi
    fi

    if [ "$TORCH_VERSION_BEFORE" != "$TORCH_VERSION_AFTER" ]; then
        # Check if the new version is a CUDA build (cu128, cu121, etc.)
        if echo "$TORCH_VERSION_AFTER" | grep -qiE "cu[0-9]+|cuda"; then
            echo "✗ CRITICAL ERROR: NVIDIA CUDA PyTorch was installed!"
            echo "  Before: $TORCH_VERSION_BEFORE"
            echo "  After:  $TORCH_VERSION_AFTER"
            echo ""
            echo "  Restoring ROCm PyTorch..."
            $PYTHON_BIN -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
            # Also remove NVIDIA-specific packages that may have been pulled in
            $PYTHON_BIN -m pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
                nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
                nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 2>/dev/null || true
            $PYTHON_BIN -m pip install \
                torch torchvision torchaudio triton \
                --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
                --break-system-packages --no-cache-dir
            TORCH_VERSION_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
            echo "✓ ROCm PyTorch restored: $TORCH_VERSION_AFTER"
        else
            echo "⚠ WARNING: PyTorch version changed from $TORCH_VERSION_BEFORE to $TORCH_VERSION_AFTER"
            echo "  Please verify this is still a ROCm build."
        fi
    else
        echo "✓ ROCm PyTorch preserved: $TORCH_VERSION_AFTER"
    fi
fi

# Install commonly needed dependencies (without torch)
if [ "$DRY_RUN" = "false" ]; then
    echo "➤ Installing additional dependencies..."
    $INSTALL_CMD einops transformers huggingface-hub --break-system-packages 2>/dev/null || true
fi

# Verify installation
if [ "$DRY_RUN" = "false" ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ Verifying vLLM installation"
    echo "└─────────────────────────────────────────────────────────┘"
    $PYTHON_BIN <<'PY'
try:
    import vllm
    print("vLLM version:", vllm.__version__)
    print("✓ vLLM imported successfully")
except Exception as e:
    print(f"✗ Failed to import vllm: {e}")
PY

    # Setup PATH/symlink for vllm command
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ Setting up vLLM CLI access"
    echo "└─────────────────────────────────────────────────────────┘"

    # Find the uv python bin directory where vllm executable is located
    UV_BIN_DIR=""
    if [ -d "$HOME/.local/share/uv/python" ]; then
        UV_BIN_DIR=$(find "$HOME/.local/share/uv/python" -name "vllm" -type f -executable 2>/dev/null | head -n1 | xargs dirname 2>/dev/null || echo "")
    fi

    # Check if vllm is already accessible
    if command -v vllm >/dev/null 2>&1; then
        echo "✓ vllm command is already accessible in PATH"
        VLLM_PATH=$(command -v vllm)
        echo "  Location: $VLLM_PATH"
    elif [ -n "$UV_BIN_DIR" ]; then
        echo "➤ Found vLLM at: $UV_BIN_DIR/vllm"

        # Try to create symlink in /usr/local/bin (requires sudo)
        if sudo ln -sf "$UV_BIN_DIR/vllm" /usr/local/bin/vllm 2>/dev/null; then
            echo "✓ Created symlink: /usr/local/bin/vllm -> $UV_BIN_DIR/vllm"
            VLLM_LINKED=true
        else
            echo "⚠ Could not create symlink in /usr/local/bin (requires sudo)"
            VLLM_LINKED=false
        fi

        # Add uv bin directory to shell configs if not already present
        ADDED_TO_BASHRC=false
        ADDED_TO_ZSHRC=false

        if ! grep -q "uv/python.*bin" "$HOME/.bashrc" 2>/dev/null; then
            echo "export PATH=\"\$HOME/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin:\$PATH\"" >> "$HOME/.bashrc"
            echo "✓ Added uv bin directory to ~/.bashrc"
            ADDED_TO_BASHRC=true
        else
            echo "✓ uv bin directory already in ~/.bashrc"
        fi

        if [ -f "$HOME/.zshrc" ]; then
            if ! grep -q "uv/python.*bin" "$HOME/.zshrc" 2>/dev/null; then
                echo "export PATH=\"\$HOME/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin:\$PATH\"" >> "$HOME/.zshrc"
                echo "✓ Added uv bin directory to ~/.zshrc"
                ADDED_TO_ZSHRC=true
            else
                echo "✓ uv bin directory already in ~/.zshrc"
            fi
        fi
    else
        echo "⚠ Could not find vLLM executable in uv directory"
        VLLM_LINKED=false
    fi
fi

echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ vLLM Installation Summary"
echo "└─────────────────────────────────────────────────────────┘"
if [ "$DRY_RUN" = "false" ]; then
    vllm_version=$($PYTHON_BIN -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "✓ vLLM installed (version: ${vllm_version})"
else
    echo "⚠ Dry run completed (no changes applied)"
fi
echo "➤ GPU_ARCH: ${GPU_ARCH}"
echo "➤ ROCm channel: ${ROCM_CHANNEL}"
echo "➤ Docs: https://docs.vllm.ai/en/latest/"

if [ "$DRY_RUN" = "false" ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ CLI Access Setup"
    echo "└─────────────────────────────────────────────────────────┘"

    if [ "${VLLM_LINKED:-false}" = "true" ] || command -v vllm >/dev/null 2>&1; then
        echo "✓ vllm command is ready to use!"
        echo ""
        echo "  Try: vllm serve --help"
    else
        echo "⚠ vllm command requires PATH setup"
        echo ""
        echo "  ╔════════════════════════════════════════════════════════╗"
        echo "  ║ IMPORTANT: Start a new terminal session OR run:        ║"
        echo "  ╠════════════════════════════════════════════════════════╣"
        echo "  ║                                                         ║"
        if [ -f "$HOME/.bashrc" ]; then
            echo "  ║  For Bash:    source ~/.bashrc                        ║"
        fi
        if [ -f "$HOME/.zshrc" ]; then
            echo "  ║  For Zsh:     source ~/.zshrc                         ║"
        fi
        echo "  ║                                                         ║"
        echo "  ║  Or create a symlink with sudo:                         ║"
        echo "  ║  sudo ln -sf ~/.local/share/uv/python/cpython-3.12.*   ║"
        echo "  ║               */bin/vllm /usr/local/bin/vllm            ║"
        echo "  ║                                                         ║"
        echo "  ╚════════════════════════════════════════════════════════╝"
        echo ""
        echo "  After setup, verify with: vllm --version"
    fi
fi
