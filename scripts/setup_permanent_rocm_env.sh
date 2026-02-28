#!/bin/bash
# setup_permanent_rocm_env.sh
# Unified script for permanent ROCm environment configuration targeting Python 3.12.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_utils.sh"
if [ -f "$SCRIPT_DIR/lib/installer_guard.sh" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/lib/installer_guard.sh"
fi

print_header "Permanent ROCm Environment Setup (Python 3.12)"

is_valid_gfx_arch() {
    [[ "${1:-}" =~ ^gfx(10|11|12)[0-9]{2}$ ]] || [[ "${1:-}" =~ ^gfx[0-9]{3}$ ]]
}

normalize_pci_bus_id() {
    local raw="${1:-}"
    raw="${raw,,}"
    raw="${raw%% *}"
    raw="${raw#0000:}"

    if [[ "$raw" =~ ^[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ ]]; then
        printf '%s\n' "$raw"
        return 0
    fi
    if [[ "$raw" =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ ]]; then
        printf '%s\n' "${raw#????:}"
        return 0
    fi
    printf '%s\n' ""
    return 1
}

gpu_series_looks_integrated() {
    local text="${1:-}"
    text="$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')"

    # Strong integrated/APU indicators.
    if [[ "$text" =~ ryzen|apu|integrated|raphael|phoenix|rembrandt|renoir|raven|picasso|cezanne|mendocino|hawk[[:space:]_-]*point|strix ]]; then
        return 0
    fi
    # Any "Radeon ... Graphics" label is treated as iGPU.
    if [[ "$text" =~ radeon ]] && [[ "$text" =~ graphics ]] && [[ ! "$text" =~ instinct ]]; then
        return 0
    fi
    return 1
}

gpu_bus_looks_integrated() {
    local bus_raw="${1:-}"
    local bus_norm line_l=""
    bus_norm="$(normalize_pci_bus_id "$bus_raw" || true)"

    if [ -z "$bus_norm" ]; then
        return 1
    fi

    if command -v lspci >/dev/null 2>&1; then
        line_l="$(lspci -s "$bus_norm" 2>/dev/null | tr '[:upper:]' '[:lower:]' || true)"
        if [[ "$line_l" =~ ryzen|apu|integrated|raphael|phoenix|rembrandt|renoir|raven|picasso|cezanne|mendocino|hawk[[:space:]_-]*point|strix ]]; then
            return 0
        fi
        if [[ "$line_l" =~ radeon ]] && [[ "$line_l" =~ graphics ]] && [[ ! "$line_l" =~ instinct ]]; then
            return 0
        fi
    fi

    # iGPUs generally expose very small VRAM in amdgpu sysfs (or none).
    local sysfs="/sys/bus/pci/devices/0000:${bus_norm}/mem_info_vram_total"
    if [ -r "$sysfs" ]; then
        local vram_bytes
        vram_bytes="$(cat "$sysfs" 2>/dev/null || true)"
        if [[ "$vram_bytes" =~ ^[0-9]+$ ]] && [ "$vram_bytes" -lt 4294967296 ]; then
            return 0
        fi
    fi

    return 1
}

sanitize_device_list() {
    local raw="${1:-}"
    local cleaned="" token
    local seen=","
    local -a tokens=()

    IFS=',' read -r -a tokens <<< "$raw"
    for token in "${tokens[@]}"; do
        token="$(printf '%s' "$token" | tr -d '[:space:]')"
        [[ "$token" =~ ^[0-9]+$ ]] || continue
        if [[ "$seen" == *",$token,"* ]]; then
            continue
        fi
        seen="${seen}${token},"
        if [ -n "$cleaned" ]; then
            cleaned="${cleaned},${token}"
        else
            cleaned="${token}"
        fi
    done

    printf '%s\n' "$cleaned"
}

gfx_to_hsa_override() {
    local gfx="${1:-}"
    local raw="${gfx#gfx}"
    local major minor patch

    if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    case "${#raw}" in
        4)
            major="${raw:0:2}"
            minor="${raw:2:1}"
            patch="${raw:3:1}"
            ;;
        3)
            major="${raw:0:1}"
            minor="${raw:1:1}"
            patch="${raw:2:1}"
            ;;
        *)
            return 1
            ;;
    esac

    printf '%s.%s.%s\n' "$major" "$minor" "$patch"
}

detect_best_gpu_arch() {
    local detected=""
    local env_candidate="${GPU_ARCH:-}"
    local best_idx=""
    local best_arch=""
    local idx gfx series discrete bus

    if command -v rocm-smi >/dev/null 2>&1; then
        while IFS='|' read -r idx gfx series discrete bus; do
            if [ "${discrete:-0}" != "1" ]; then
                continue
            fi
            if gpu_series_looks_integrated "$series" || gpu_bus_looks_integrated "$bus"; then
                continue
            fi
            if ! is_valid_gfx_arch "$gfx"; then
                continue
            fi
            if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
                continue
            fi
            if [ -z "$best_idx" ] || [ "$idx" -lt "$best_idx" ]; then
                best_idx="$idx"
                best_arch="$gfx"
            fi
        done < <(collect_rocm_smi_gpu_info || true)
    fi

    if is_valid_gfx_arch "$best_arch"; then
        printf '%s\n' "$best_arch"
        return 0
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        detected="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]+' | awk '!seen[$0]++' | sort -V | tail -n1 || true)"
    fi

    if is_valid_gfx_arch "$detected"; then
        printf '%s\n' "$detected"
        return 0
    fi

    if is_valid_gfx_arch "$env_candidate"; then
        printf '%s\n' "$env_candidate"
        return 0
    fi

    printf '%s\n' "gfx1100"
    return 0
}

collect_rocm_smi_gpu_info() {
    if ! command -v rocm-smi >/dev/null 2>&1; then
        return 1
    fi

    local smi_json
    smi_json="$(rocm-smi --showproductname --showbus --json 2>/dev/null || true)"
    [ -n "$smi_json" ] || return 1

    if command -v python3 >/dev/null 2>&1; then
        python3 - "$smi_json" <<'PY'
import json
import re
import sys

raw = sys.argv[1].strip() if len(sys.argv) > 1 else ""
if not raw:
    raise SystemExit(1)

match = re.search(r"\{.*\}", raw, flags=re.S)
if match is not None:
    raw = match.group(0)

try:
    data = json.loads(raw)
except Exception:
    raise SystemExit(1)

def meaningful(value):
    s = str(value or "").strip()
    if s.lower() in {"", "n/a", "na", "none", "unknown", "not available", "-"}:
        return ""
    return s

for card_key, payload in data.items():
    idx_match = re.search(r"(\d+)$", card_key)
    if idx_match is None:
        continue
    idx = idx_match.group(1)
    series = meaningful(payload.get("Card Series"))
    model = meaningful(payload.get("Card Model"))
    product_name = meaningful(payload.get("Card SKU"))
    bus = str(
        payload.get("PCI Bus")
        or payload.get("PCI Bus Address")
        or payload.get("PCIe Bus")
        or payload.get("Bus")
        or ""
    ).strip()
    combined = " ".join(x for x in (series, model, product_name) if x)
    gfx = str(payload.get("GFX Version", "")).strip()
    series_l = combined.lower()
    integrated = (
        ("ryzen" in series_l)
        or ("apu" in series_l)
        or ("integrated" in series_l)
        or ("raphael" in series_l)
        or ("phoenix" in series_l)
        or ("rembrandt" in series_l)
        or ("renoir" in series_l)
        or ("raven" in series_l)
        or ("picasso" in series_l)
        or ("cezanne" in series_l)
        or ("mendocino" in series_l)
        or ("hawk point" in series_l)
        or ("strix" in series_l)
    )
    if ("radeon" in series_l) and ("graphics" in series_l) and ("instinct" not in series_l):
        integrated = True
    discrete = 0 if integrated else 1
    name = combined if combined else series
    print(f"{idx}|{gfx}|{name}|{discrete}|{bus}")
PY
    fi
}

detect_discrete_gpu_list() {
    local list=""
    local idx gfx series discrete bus
    local sanitized=""
    local lspci_count=0

    if command -v rocm-smi >/dev/null 2>&1; then
        while IFS='|' read -r idx gfx series discrete bus; do
            [ -n "$idx" ] || continue
            [[ "$idx" =~ ^[0-9]+$ ]] || continue
            if [ "${discrete:-0}" != "1" ]; then
                continue
            fi
            if gpu_series_looks_integrated "$series" || gpu_bus_looks_integrated "$bus"; then
                continue
            fi
            if [ -n "$list" ]; then
                list="${list},${idx}"
            else
                list="${idx}"
            fi
        done < <(collect_rocm_smi_gpu_info || true)
    fi

    sanitized="$(sanitize_device_list "$list")"
    if [ -n "$sanitized" ]; then
        printf '%s\n' "$sanitized"
        return 0
    fi

    if command -v lspci >/dev/null 2>&1; then
        while IFS= read -r line; do
            local line_l
            local bus_id
            line_l="$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')"
            bus_id="$(printf '%s' "$line_l" | awk '{print $1}')"
            if [[ ! "$line_l" =~ (vga|3d|display) ]]; then
                continue
            fi
            if [[ ! "$line_l" =~ (amd|radeon|advanced\ micro\ devices) ]]; then
                continue
            fi
            if gpu_series_looks_integrated "$line_l"; then
                continue
            fi
            if gpu_bus_looks_integrated "$bus_id"; then
                continue
            fi
            lspci_count=$((lspci_count + 1))
        done < <(lspci 2>/dev/null || true)

        if [ "$lspci_count" -gt 0 ]; then
            seq -s, 0 $((lspci_count - 1))
            return 0
        fi
    fi

    printf '%s\n' "0"
    return 0
}

# 1. Select a ROCm-compatible Python (3.10-3.13), preferring 3.12.
if type mlstack_select_python_for_rocm_torch >/dev/null 2>&1; then
    PYTHON_BIN="$(mlstack_select_python_for_rocm_torch "python3.12" "${ROCM_VERSION:-7.2}" "${MLSTACK_TORCH_CHANNEL:-latest}" || true)"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    for py in python3.12 python3.11 python3.10 python3.13 python3; do
        if command -v "$py" >/dev/null 2>&1; then
            candidate="$(command -v "$py")"
            version="$("$candidate" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
            major="${version%%.*}"
            minor="${version#*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    print_error "No ROCm-compatible Python interpreter found (required: 3.10-3.13)."
    exit 1
fi

print_step "Targeting Python interpreter: $PYTHON_BIN"

# 2. Detect Hardware (with iGPU filtering)
print_step "Detecting AMD hardware and filtering integrated GPUs..."
GPU_ARCH="$(detect_best_gpu_arch)"
HSA_OVERRIDE_GFX_VERSION="$(gfx_to_hsa_override "$GPU_ARCH" || echo "11.0.0")"
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1 || echo "7.2.0")
print_step "Selected GPU architecture: $GPU_ARCH"
print_step "Selected HSA override: $HSA_OVERRIDE_GFX_VERSION"

DISCRETE_GPU_LIST="$(detect_discrete_gpu_list)"
if [ "$DISCRETE_GPU_LIST" = "0" ]; then
    print_warning "Discrete GPU detection fallback engaged; using GPU index: 0"
else
    print_success "Detected discrete GPU list from rocm-smi: $DISCRETE_GPU_LIST"
fi

INSTALL_METHOD_SELECTED="${MLSTACK_INSTALL_METHOD:-${INSTALL_METHOD:-auto}}"
INSTALL_METHOD_SELECTED="$(echo "$INSTALL_METHOD_SELECTED" | tr '[:upper:]' '[:lower:]')"
case "$INSTALL_METHOD_SELECTED" in
    global|venv|auto) ;;
    *) INSTALL_METHOD_SELECTED="auto" ;;
esac
print_step "Persisting install method: $INSTALL_METHOD_SELECTED"

# 3. Build .mlstack_env content
ENV_FILE="$HOME/.mlstack_env"
print_step "Generating $ENV_FILE..."

cat > "$ENV_FILE" << EOF
# Permanent ROCm Environment Setup (Generated $(date))
export ROCM_VERSION=$ROCM_VERSION
export ROCM_CHANNEL=latest
export GPU_ARCH=$GPU_ARCH
export PYTORCH_ROCM_ARCH=$GPU_ARCH
export GPU_ARCHS=$GPU_ARCH
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
# Discrete GPUs only (iGPUs filtered): $DISCRETE_GPU_LIST
export HIP_VISIBLE_DEVICES=$DISCRETE_GPU_LIST
export CUDA_VISIBLE_DEVICES=$DISCRETE_GPU_LIST
export PYTORCH_ROCM_DEVICE=$(echo "$DISCRETE_GPU_LIST" | cut -d',' -f1)
export MLSTACK_PYTHON_BIN=$PYTHON_BIN
export UV_PYTHON=$PYTHON_BIN
export MLSTACK_INSTALL_METHOD=$INSTALL_METHOD_SELECTED
export INSTALL_METHOD=$INSTALL_METHOD_SELECTED
export PYTHONPATH=/opt/rocm/lib:\$PYTHONPATH
export AMDGPU_ASIC_ID_TABLE_PATH=/usr/share/libdrm/amdgpu.ids
export AMDGPU_ASIC_ID_TABLE_PATHS=/usr/share/libdrm

# Path Settings
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin:/opt/rocm/hip/bin:\$PATH"
export LD_LIBRARY_PATH="\$HOME/.mlstack/libmpi-compat:\$HOME/.mlstack/libmpi-compat-user-\$(id -u):/opt/rocm/lib:/opt/rocm/hip/lib:/opt/rocm/opencl/lib:\$LD_LIBRARY_PATH"

# Performance & Compatibility
export HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MLSTACK_TRITON_HOME=\$HOME/.cache/mlstack/triton
export TRITON_HOME=\$MLSTACK_TRITON_HOME
export TRITON_CACHE_DIR=\$MLSTACK_TRITON_HOME/cache
export TRITON_DUMP_DIR=\$MLSTACK_TRITON_HOME/dump
export TRITON_OVERRIDE_DIR=\$MLSTACK_TRITON_HOME/override
mkdir -p "\$TRITON_CACHE_DIR" "\$TRITON_DUMP_DIR" "\$TRITON_OVERRIDE_DIR" 2>/dev/null || true

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

# 4. Patch shell startup files for bash/zsh/fish
if type mlstack_patch_shell_env_for_mlstack >/dev/null 2>&1; then
    print_step "Patching shell startup files for ML Stack environment..."
    mlstack_patch_shell_env_for_mlstack
else
    print_warning "installer_guard.sh not available; skipped shell startup patching."
fi

print_success "Permanent ROCm environment configured for Python 3.12!"
