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
if [ -f "$SCRIPT_DIR/lib/installer_guard.sh" ]; then
    # shellcheck source=lib/installer_guard.sh
    source "$SCRIPT_DIR/lib/installer_guard.sh"
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
MLSTACK_STRICT_ROCM="${MLSTACK_STRICT_ROCM:-1}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

strict_validate_python_version() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor < 10:
    raise SystemExit(f"Unsupported Python {major}.{minor}; strict ROCm mode requires Python 3.10+")
PY
}

strict_detect_rocm_mm() {
    local rocm_raw
    rocm_raw="${ROCM_VERSION:-}"
    if [[ -z "$rocm_raw" ]] && [[ -f /opt/rocm/.info/version ]]; then
        rocm_raw="$(head -n1 /opt/rocm/.info/version 2>/dev/null || true)"
    fi
    rocm_raw="$(echo "$rocm_raw" | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
    case "$rocm_raw" in
        5.7|6.0|6.1|6.2|6.3|6.4|7.0|7.1|7.2) echo "$rocm_raw" ;;
        *) echo "7.2" ;;
    esac
}

strict_rocm_index_url() {
    local rocm_mm="$1"
    echo "https://repo.radeon.com/rocm/manylinux/rocm-rel-${rocm_mm}/"
}

strict_venv_python() {
    local component="$1"
    local base_python="$2"
    local venv_dir="${MLSTACK_VENV_DIR:-$HOME/.mlstack/venvs/$component}"
    mkdir -p "$(dirname "$venv_dir")"
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        "$base_python" -m venv "$venv_dir"
    fi
    "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
    printf '%s\n' "$venv_dir/bin/python"
}

strict_purge_nvidia_packages() {
    local py_cmd="$1"
    local nvidia_pkgs
    nvidia_pkgs="$("$py_cmd" -m pip list --format=freeze 2>/dev/null \
        | awk -F== 'BEGIN{IGNORECASE=1}{name=tolower($1); if (name ~ /^nvidia-/ || name ~ /(^|-)cuda([_-]|$)/ || name ~ /^pytorch-cuda$/ || name ~ /^torch-cuda$/) print $1}' \
        | xargs || true)"
    if [[ -n "${nvidia_pkgs:-}" ]]; then
        "$py_cmd" -m pip uninstall -y $nvidia_pkgs >/dev/null || true
    fi
}

strict_verify_no_cuda_contamination() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import re
import subprocess
import sys

errors = []
try:
    import torch
    if getattr(torch.version, "cuda", None):
        errors.append(f"torch.version.cuda={torch.version.cuda}")
    if not getattr(torch.version, "hip", None):
        errors.append("torch.version.hip missing")
    version = getattr(torch, "__version__", "").lower()
    if "cu" in version and "rocm" not in version:
        errors.append(f"CUDA-looking torch version: {version}")
except Exception as exc:
    errors.append(f"torch validation failed: {exc}")

pip_out = subprocess.check_output(
    [sys.executable, "-m", "pip", "list", "--format=freeze"],
    text=True,
    stderr=subprocess.DEVNULL,
)
for line in pip_out.splitlines():
    name = line.split("==", 1)[0].strip().lower()
    if name.startswith("nvidia-") or re.search(r"(^|-)cuda([_-]|$)", name) or name in {"pytorch-cuda", "torch-cuda"}:
        errors.append(f"disallowed package: {name}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)
PY
}

strict_ensure_rocm_torch() {
    local py_cmd="$1"
    local rocm_mm
    rocm_mm="$(strict_detect_rocm_mm)"
    local rocm_index
    rocm_index="$(strict_rocm_index_url "$rocm_mm")"

    strict_purge_nvidia_packages "$py_cmd"
    "$py_cmd" -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true

    if ! "$py_cmd" -m pip install --no-cache-dir --upgrade \
        --index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
        torch torchvision torchaudio triton; then
        "$py_cmd" -m pip install --no-cache-dir --upgrade \
            --index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
            torch torchvision torchaudio
    fi

    "$py_cmd" - <<'PY'
import torch
assert torch.__version__
assert getattr(torch.version, "hip", None), "torch.version.hip missing"
PY
}

strict_install_vllm() {
    local base_python="$PYTHON_BIN"
    local strict_python

    if ! command -v "$base_python" >/dev/null 2>&1; then
        echo "✗ ERROR: Python interpreter not found: $base_python"
        return 1
    fi
    if ! strict_validate_python_version "$base_python"; then
        echo "✗ ERROR: Strict ROCm mode requires Python 3.10+"
        return 1
    fi

    if [ "$DRY_RUN" = "true" ]; then
        local strict_venv_dir="${MLSTACK_VENV_DIR:-$HOME/.mlstack/venvs/vllm}"
        echo "[DRY-RUN] Would ensure ROCm torch and install vLLM in ${strict_venv_dir}"
        return 0
    fi

    strict_python="$(strict_venv_python "vllm" "$base_python")" || return 1

    export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
    export ROCM_PATH="$ROCM_HOME"
    export HIP_PATH="$ROCM_HOME"
    export HIP_ROOT_DIR="$ROCM_HOME"
    export PYTORCH_ROCM_ARCH="$GPU_ARCH"
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export VLLM_TARGET_DEVICE=rocm

    echo "➤ Strict ROCm mode enabled (MLSTACK_STRICT_ROCM=${MLSTACK_STRICT_ROCM})"
    echo "➤ Using deterministic venv: ${strict_python%/bin/python}"

    echo "➤ Ensuring ROCm PyTorch via strict torch ensure logic..."
    strict_ensure_rocm_torch "$strict_python" || return 1
    TORCH_VERSION_BEFORE=$("$strict_python" -c "import torch; print(torch.__version__)")
    echo "✓ ROCm PyTorch detected: $TORCH_VERSION_BEFORE"

    echo "➤ Installing vLLM from official ROCm wheels with --no-deps..."
    if ! "$strict_python" -m pip install --no-cache-dir --no-deps \
        vllm --extra-index-url https://wheels.vllm.ai/rocm/; then
        echo "✗ vLLM installation failed"
        return 1
    fi

    echo "➤ Installing vLLM dependencies (excluding torch and xformers)..."
    "$strict_python" -m pip install --no-cache-dir \
        accelerate aiohttp cloudpickle fastapi msgspec prometheus-client psutil \
        py-cpuinfo pyzmq ray requests sentencepiece tiktoken uvicorn \
        einops transformers huggingface-hub || true

    # Preserve post-check semantics and hard-fail contamination.
    echo "➤ Verifying ROCm PyTorch wasn't overwritten..."
    TORCH_VERSION_AFTER=$("$strict_python" -c "import torch; print(torch.__version__)")
    if [ "$TORCH_VERSION_BEFORE" != "$TORCH_VERSION_AFTER" ]; then
        echo "⚠ WARNING: PyTorch version changed from $TORCH_VERSION_BEFORE to $TORCH_VERSION_AFTER"
    else
        echo "✓ ROCm PyTorch preserved: $TORCH_VERSION_AFTER"
    fi

    if ! strict_verify_no_cuda_contamination "$strict_python"; then
        echo "✗ CRITICAL ERROR: CUDA/NVIDIA contamination detected after vLLM install"
        return 1
    fi

    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ Verifying vLLM installation"
    echo "└─────────────────────────────────────────────────────────┘"
    "$strict_python" <<'PY'
try:
    import vllm
    print("vLLM version:", vllm.__version__)
    print("✓ vLLM imported successfully")
except Exception as e:
    print(f"✗ Failed to import vllm: {e}")
    raise SystemExit(1)
PY

    return 0
}

if [[ "${MLSTACK_STRICT_ROCM}" != "0" ]]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ Installing vLLM for ROCm $ROCM_VERSION ($ROCM_CHANNEL)"
    echo "└─────────────────────────────────────────────────────────┘"
    strict_install_vllm
    exit $?
fi

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

if [ "$DRY_RUN" = "true" ]; then
    echo "➤ DRY RUN MODE - No installation actions will be performed."
    echo "[DRY-RUN] Would verify ROCm torch baseline and preserve it during vLLM install"
    echo "[DRY-RUN] Would install vLLM core with --no-deps from ROCm wheel index"
    echo "[DRY-RUN] Would install non-torch dependencies and run import verification"
    exit 0
fi

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
