#!/usr/bin/env bash
# update_helper.sh - Component detection and update helper functions
#
# Provides functions for detecting installed ML stack components, querying
# their versions, and dispatching update operations to installer scripts.
#
# Usage:
#   if [[ -f "$MLSTACK_SCRIPT_DIR/lib/update_helper.sh" ]]; then
#       source "$MLSTACK_SCRIPT_DIR/lib/update_helper.sh"
#   fi

# --- Function: up_user_home ---
# Get the user's home directory.
up_user_home() {
    echo "${HOME:-$(eval echo "~${SUDO_USER:-$USER}")}"
}

# --- Function: up_path_exists ---
# Check if a path exists.
# Usage:
#   up_path_exists "/opt/rocm" && echo "ROCm found"
up_path_exists() {
    [[ -e "$1" ]]
}

# --- Function: up_command_exists ---
# Check if a command exists on PATH.
# Usage:
#   up_command_exists "python3" && echo "Python found"
up_command_exists() {
    command -v "$1" &>/dev/null
}

# --- Function: up_is_python_module ---
# Check if a Python module is importable.
# Usage:
#   up_is_python_module "torch" && echo "PyTorch found"
up_is_python_module() {
    local module="$1"
    local python="${2:-python3}"
    "$python" -c "import $module" &>/dev/null
}

# --- Function: up_get_version ---
# Get the version of an installed component (best effort).
# Returns version string or "unknown".
# Usage:
#   ver=$(up_get_version "pytorch")
up_get_version() {
    local component_id="$1"
    local python="${2:-python3}"
    local home
    home=$(up_user_home)

    case "$component_id" in
        rocm)
            if [[ -f /opt/rocm/.info/version ]]; then
                head -n1 /opt/rocm/.info/version 2>/dev/null
            elif up_command_exists rocminfo; then
                rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs
            else
                echo "unknown"
            fi
            ;;
        pytorch)
            if up_is_python_module "torch" "$python"; then
                "$python" -c "import torch; print(torch.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        triton)
            if up_is_python_module "triton" "$python"; then
                "$python" -c "import triton; print(getattr(triton, '__version__', 'unknown'))" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        deepspeed)
            if up_is_python_module "deepspeed" "$python"; then
                "$python" -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        vllm)
            if up_is_python_module "vllm" "$python"; then
                "$python" -c "import vllm; print(vllm.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        aiter)
            if up_is_python_module "aiter" "$python"; then
                "$python" -c "import aiter; print(getattr(aiter, '__version__', 'unknown'))" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        onnx)
            if up_is_python_module "onnxruntime" "$python"; then
                "$python" -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        bitsandbytes)
            if up_is_python_module "bitsandbytes" "$python"; then
                "$python" -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        migraphx)
            if up_is_python_module "migraphx" "$python"; then
                "$python" -c "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        flash-attn)
            if up_is_python_module "flash_attn" "$python"; then
                "$python" -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        mpi4py)
            if up_is_python_module "mpi4py" "$python"; then
                "$python" -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        wandb)
            if up_is_python_module "wandb" "$python"; then
                "$python" -c "import wandb; print(wandb.__version__)" 2>/dev/null
            else
                echo "not installed"
            fi
            ;;
        comfyui)
            if up_path_exists "$home/ComfyUI/.git"; then
                git -C "$home/ComfyUI" log -1 --format="%h %s" 2>/dev/null || echo "installed"
            else
                echo "not installed"
            fi
            ;;
        vllm-studio)
            if up_path_exists "$home/vllm-studio/.git"; then
                git -C "$home/vllm-studio" log -1 --format="%h %s" 2>/dev/null || echo "installed"
            else
                echo "not installed"
            fi
            ;;
        textgen)
            if up_path_exists "$home/text-generation-webui/.git"; then
                git -C "$home/text-generation-webui" log -1 --format="%h %s" 2>/dev/null || echo "installed"
            else
                echo "not installed"
            fi
            ;;
        rocm-smi)
            if up_command_exists "rocm-smi"; then
                rocm-smi --version 2>/dev/null | head -1 || echo "installed"
            else
                echo "not installed"
            fi
            ;;
        permanent-env)
            if up_path_exists "$home/.mlstack_env"; then
                echo "installed"
            else
                echo "not installed"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# --- Function: up_display_name ---
# Map component IDs to human-readable names.
# Usage:
#   name=$(up_display_name "pytorch")
up_display_name() {
    local component_id="$1"
    case "$component_id" in
        rocm)              echo "ROCm" ;;
        pytorch)           echo "PyTorch" ;;
        triton)            echo "Triton" ;;
        deepspeed)         echo "DeepSpeed" ;;
        vllm)              echo "vLLM" ;;
        aiter)             echo "AITER" ;;
        onnx)              echo "ONNX Runtime" ;;
        bitsandbytes)      echo "bitsandbytes" ;;
        migraphx)          echo "MIGraphX" ;;
        flash-attn)        echo "Flash Attention" ;;
        mpi4py)            echo "MPI4Py" ;;
        wandb)             echo "Weights & Biases" ;;
        comfyui)           echo "ComfyUI" ;;
        vllm-studio)       echo "vLLM Studio" ;;
        textgen)           echo "text-generation-webui" ;;
        rocm-smi)          echo "ROCm SMI" ;;
        permanent-env)     echo "ML Stack Environment" ;;
        *)                 echo "$component_id" ;;
    esac
}

# --- Function: up_detect_installed ---
# Detect all installed components. Outputs newline-separated list of component IDs.
# Usage:
#   installed=$(up_detect_installed)
up_detect_installed() {
    local python="${1:-python3}"
    local home
    home=$(up_user_home)
    local -a installed=()

    # ROCm
    if up_path_exists "/opt/rocm/.info/version" || up_command_exists rocminfo; then
        installed+=("rocm")
    fi

    # Python packages
    local -a py_modules=(pytorch triton deepspeed vllm aiter onnx bitsandbytes migraphx flash-attn mpi4py wandb)
    for mod in "${py_modules[@]}"; do
        if up_is_python_module "$mod" "$python"; then
            installed+=("$mod")
        fi
    done

    # Git-based
    local -a git_dirs=("ComfyUI" "vllm-studio" "text-generation-webui")
    local -a git_ids=("comfyui" "vllm-studio" "textgen")
    for i in "${!git_dirs[@]}"; do
        if up_path_exists "$home/${git_dirs[$i]}/.git"; then
            installed+=("${git_ids[$i]}")
        fi
    done

    # System tools
    if up_command_exists "rocm-smi"; then
        installed+=("rocm-smi")
    fi

    if up_path_exists "$home/.mlstack_env"; then
        installed+=("permanent-env")
    fi

    printf '%s\n' "${installed[@]}"
}

# --- Function: up_update_component ---
# Dispatch update to the correct installer script.
# Returns 0 on success, 1 on failure.
# Usage:
#   up_update_component "pytorch" "/path/to/scripts" "python3"
up_update_component() {
    local component_id="$1"
    local scripts_dir="$2"
    local python="${3:-python3}"

    local installer_script=""

    case "$component_id" in
        rocm)          installer_script="install_rocm.sh" ;;
        pytorch)       installer_script="install_pytorch_rocm.sh" ;;
        triton)        installer_script="install_triton_multi.sh" ;;
        deepspeed)     installer_script="install_deepspeed.sh" ;;
        vllm)          installer_script="install_vllm_multi.sh" ;;
        aiter)         installer_script="install_aiter.sh" ;;
        onnx)          installer_script="build_onnxruntime_multi.sh" ;;
        bitsandbytes)  installer_script="install_bitsandbytes_multi.sh" ;;
        migraphx)      installer_script="install_migraphx_multi.sh" ;;
        flash-attn)    installer_script="install_flash_attention_ck.sh" ;;
        mpi4py)        installer_script="install_mpi.sh" ;;
        wandb)         installer_script="install_wandb.sh" ;;
        comfyui)       installer_script="install_comfyui.sh" ;;
        vllm-studio)   installer_script="install_vllm_studio.sh" ;;
        textgen)       installer_script="install_textgen.sh" ;;
        *)
            echo "Error: No installer script known for component '$component_id'" >&2
            return 1
            ;;
    esac

    # Check for multi-distro variants with fallback
    local script_path="$scripts_dir/$installer_script"
    if [[ ! -f "$script_path" ]]; then
        # Try non-multi variant as fallback
        local base_name="${installer_script/_multi/}"
        if [[ "$base_name" != "$installer_script" ]] && [[ -f "$scripts_dir/$base_name" ]]; then
            script_path="$scripts_dir/$base_name"
        else
            echo "Error: Installer script not found: $script_path" >&2
            return 1
        fi
    fi

    echo "Updating $(up_display_name "$component_id")..."
    MLSTACK_PYTHON_BIN="$python" bash "$script_path" --force
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "Successfully updated $(up_display_name "$component_id")"
    else
        echo "Failed to update $(up_display_name "$component_id") (exit code: $rc)" >&2
    fi
    return $rc
}
