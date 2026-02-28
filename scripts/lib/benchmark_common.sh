#!/usr/bin/env bash
# Shared helpers for benchmark wrapper scripts.

BENCHMARK_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$BENCHMARK_LIB_DIR/installer_guard.sh" ]; then
    # shellcheck disable=SC1090
    source "$BENCHMARK_LIB_DIR/installer_guard.sh"
fi

benchmark_enable_colors() {
    if [ -t 1 ] && [ "${NO_COLOR:-}" != "1" ]; then
        BENCH_COLOR_RED='\033[0;31m'
        BENCH_COLOR_GREEN='\033[0;32m'
        BENCH_COLOR_YELLOW='\033[0;33m'
        BENCH_COLOR_BLUE='\033[0;34m'
        BENCH_COLOR_RESET='\033[0m'
    else
        BENCH_COLOR_RED=''
        BENCH_COLOR_GREEN=''
        BENCH_COLOR_YELLOW=''
        BENCH_COLOR_BLUE=''
        BENCH_COLOR_RESET=''
    fi
}

benchmark_log_file='' 

benchmark_set_log_file() {
    benchmark_log_file="$1"
}

benchmark_log() {
    local message="$1"
    local line
    line="[$(date '+%Y-%m-%d %H:%M:%S')] ${message}"
    if [ -n "${benchmark_log_file:-}" ]; then
        printf '%s\n' "$line" | tee -a "$benchmark_log_file"
    else
        printf '%s\n' "$line"
    fi
}

benchmark_info() {
    benchmark_log "${BENCH_COLOR_BLUE}INFO${BENCH_COLOR_RESET} $1"
}

benchmark_warn() {
    benchmark_log "${BENCH_COLOR_YELLOW}WARN${BENCH_COLOR_RESET} $1"
}

benchmark_error() {
    benchmark_log "${BENCH_COLOR_RED}ERROR${BENCH_COLOR_RESET} $1"
}

benchmark_success() {
    benchmark_log "${BENCH_COLOR_GREEN}OK${BENCH_COLOR_RESET} $1"
}

benchmark_is_dry_run() {
    [ "${DRY_RUN:-false}" = "true" ]
}

benchmark_load_global_env() {
    local env_file="${MLSTACK_PERSISTENT_ENV_FILE:-$HOME/.mlstack_env}"
    if [ ! -f "$env_file" ]; then
        benchmark_warn "Persistent env file not found: $env_file"
        return 0
    fi

    set +u 2>/dev/null || true
    # shellcheck disable=SC1090
    source "$env_file"
    set -u 2>/dev/null || true
}

benchmark_normalize_pci_bus_id() {
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
    return 1
}

benchmark_gpu_series_looks_integrated() {
    local text="${1:-}"
    text="$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')"
    if [[ "$text" =~ ryzen|apu|integrated|raphael|phoenix|rembrandt|renoir|raven|picasso|cezanne|mendocino|hawk[[:space:]_-]*point|strix ]]; then
        return 0
    fi
    if [[ "$text" =~ radeon ]] && [[ "$text" =~ graphics ]] && [[ ! "$text" =~ instinct ]]; then
        return 0
    fi
    return 1
}

benchmark_gpu_bus_looks_integrated() {
    local bus_raw="${1:-}"
    local bus_norm line_l=""
    bus_norm="$(benchmark_normalize_pci_bus_id "$bus_raw" || true)"
    [ -n "$bus_norm" ] || return 1

    if command -v lspci >/dev/null 2>&1; then
        line_l="$(lspci -s "$bus_norm" 2>/dev/null | tr '[:upper:]' '[:lower:]' || true)"
        if [[ "$line_l" =~ ryzen|apu|integrated|raphael|phoenix|rembrandt|renoir|raven|picasso|cezanne|mendocino|hawk[[:space:]_-]*point|strix ]]; then
            return 0
        fi
        if [[ "$line_l" =~ radeon ]] && [[ "$line_l" =~ graphics ]] && [[ ! "$line_l" =~ instinct ]]; then
            return 0
        fi
    fi

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

benchmark_sanitize_device_list() {
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

benchmark_detect_discrete_gpu_list() {
    local list="" idx gfx series discrete bus
    local lspci_count=0

    if command -v rocm-smi >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
        while IFS='|' read -r idx gfx series discrete bus; do
            [ -n "$idx" ] || continue
            [[ "$idx" =~ ^[0-9]+$ ]] || continue
            if [ "${discrete:-0}" != "1" ]; then
                continue
            fi
            if benchmark_gpu_series_looks_integrated "$series" || benchmark_gpu_bus_looks_integrated "$bus"; then
                continue
            fi
            if [ -n "$list" ]; then
                list="${list},${idx}"
            else
                list="${idx}"
            fi
        done < <(
            rocm-smi --showproductname --showbus --json 2>/dev/null | python3 - <<'PY'
import json
import re
import sys

raw = sys.stdin.read()
if not raw:
    raise SystemExit(0)

match = re.search(r"\{.*\}", raw, flags=re.S)
if match is not None:
    raw = match.group(0)

try:
    data = json.loads(raw)
except Exception:
    raise SystemExit(0)

def meaningful(value):
    s = str(value or "").strip()
    if s.lower() in {"", "n/a", "na", "none", "unknown", "not available", "-"}:
        return ""
    return s

for card_key, payload in data.items():
    idx_match = re.search(r"(\d+)$", str(card_key))
    if idx_match is None:
        continue
    idx = idx_match.group(1)
    series = meaningful(payload.get("Card Series"))
    model = meaningful(payload.get("Card Model"))
    sku = meaningful(payload.get("Card SKU"))
    bus = str(
        payload.get("PCI Bus")
        or payload.get("PCI Bus Address")
        or payload.get("PCIe Bus")
        or payload.get("Bus")
        or ""
    ).strip()
    name = " ".join(x for x in (series, model, sku) if x)
    lower = name.lower()
    integrated = (
        ("ryzen" in lower)
        or ("apu" in lower)
        or ("integrated" in lower)
        or ("raphael" in lower)
        or ("phoenix" in lower)
        or ("rembrandt" in lower)
        or ("renoir" in lower)
        or ("raven" in lower)
        or ("picasso" in lower)
        or ("cezanne" in lower)
        or ("mendocino" in lower)
        or ("hawk point" in lower)
        or ("strix" in lower)
    )
    if ("radeon" in lower) and ("graphics" in lower) and ("instinct" not in lower):
        integrated = True
    discrete = 0 if integrated else 1
    print(f"{idx}|{payload.get('GFX Version','')}|{name}|{discrete}|{bus}")
PY
        )
    fi

    list="$(benchmark_sanitize_device_list "$list")"
    if [ -n "$list" ]; then
        printf '%s\n' "$list"
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
            if benchmark_gpu_series_looks_integrated "$line_l"; then
                continue
            fi
            if benchmark_gpu_bus_looks_integrated "$bus_id"; then
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
}

benchmark_rocm_runtime_visible_ok() {
    local python_bin="$1"
    local visible_list="$2"
    local primary="${visible_list%%,*}"
    [ -n "$primary" ] || primary="0"

    HIP_VISIBLE_DEVICES="$visible_list" \
    CUDA_VISIBLE_DEVICES="$visible_list" \
    PYTORCH_ROCM_DEVICE="$primary" \
    "$python_bin" - <<'PY' >/dev/null 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
if torch.cuda.device_count() < 1:
    raise SystemExit(1)
torch.cuda.get_device_properties(0)
PY
}

benchmark_select_working_gpu_list() {
    local python_bin="$1"
    local detected_list="$2"
    local -a candidates=()
    local -a _tokens=()
    local token first

    detected_list="$(benchmark_sanitize_device_list "$detected_list")"
    if [ -n "$detected_list" ]; then
        candidates+=("$detected_list")
        first="${detected_list%%,*}"
        if [ -n "$first" ] && [ "$first" != "$detected_list" ]; then
            candidates+=("$first")
        fi
        IFS=',' read -r -a _tokens <<< "$detected_list"
        for token in "${_tokens[@]}"; do
            token="$(printf '%s' "$token" | tr -d '[:space:]')"
            [[ "$token" =~ ^[0-9]+$ ]] || continue
            candidates+=("$token")
        done
    fi
    candidates+=("0")

    local seen=","
    for token in "${candidates[@]}"; do
        token="$(benchmark_sanitize_device_list "$token")"
        [ -n "$token" ] || continue
        if [[ "$seen" == *",$token,"* ]]; then
            continue
        fi
        seen="${seen}${token},"
        if benchmark_rocm_runtime_visible_ok "$python_bin" "$token"; then
            printf '%s\n' "$token"
            return 0
        fi
    done

    local fallback="${detected_list%%,*}"
    if [ -z "$fallback" ]; then
        fallback="0"
    fi
    printf '%s\n' "$fallback"
    return 0
}

benchmark_prepare_triton_cache_env() {
    local triton_home="${MLSTACK_TRITON_HOME:-${TRITON_HOME:-$HOME/.cache/mlstack/triton}}"
    local triton_cache="${TRITON_CACHE_DIR:-$triton_home/cache}"
    local triton_dump="${TRITON_DUMP_DIR:-$triton_home/dump}"
    local triton_override="${TRITON_OVERRIDE_DIR:-$triton_home/override}"
    local probe_file=""
    local fallback_home=""

    mkdir -p "$triton_cache" "$triton_dump" "$triton_override" >/dev/null 2>&1 || true
    probe_file="$triton_cache/.mlstack_write_probe_$$"
    if ! ( : >"$probe_file" ) 2>/dev/null; then
        fallback_home="${TMPDIR:-/tmp}/mlstack-triton-$(id -u 2>/dev/null || echo 0)"
        triton_home="$fallback_home"
        triton_cache="$triton_home/cache"
        triton_dump="$triton_home/dump"
        triton_override="$triton_home/override"
        mkdir -p "$triton_cache" "$triton_dump" "$triton_override" >/dev/null 2>&1 || true
        probe_file="$triton_cache/.mlstack_write_probe_$$"
        if ! ( : >"$probe_file" ) 2>/dev/null; then
            benchmark_warn "Triton cache directory is not writable; vLLM kernel compilation may fail (cache dir: $triton_cache)"
            return 1
        fi
    fi

    rm -f "$probe_file" >/dev/null 2>&1 || true
    export MLSTACK_TRITON_HOME="$triton_home"
    export TRITON_HOME="$triton_home"
    export TRITON_CACHE_DIR="$triton_cache"
    export TRITON_DUMP_DIR="$triton_dump"
    export TRITON_OVERRIDE_DIR="$triton_override"
    return 0
}

benchmark_python_exists() {
    local python_bin="${1:-}"
    [ -n "$python_bin" ] || return 1

    if [ -x "$python_bin" ]; then
        return 0
    fi

    command -v "$python_bin" >/dev/null 2>&1
}

benchmark_normalize_install_method() {
    local method="${1:-auto}"
    method="$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')"
    case "$method" in
        global|venv|auto) printf '%s\n' "$method" ;;
        *) printf '%s\n' "auto" ;;
    esac
}

benchmark_resolve_python_for_method() {
    local method="$1"
    local candidate=""
    local chosen=""
    local var_name=""

    case "$method" in
        global)
            for var_name in MLSTACK_PYTHON_BIN UV_PYTHON; do
                candidate="${!var_name:-}"
                if benchmark_python_exists "$candidate"; then
                    chosen="$candidate"
                    break
                fi
            done
            ;;
        venv)
            for var_name in \
                MLSTACK_BENCHMARK_PYTHON \
                PYTORCH_VENV_PYTHON \
                VLLM_VENV_PYTHON \
                DEEPSPEED_VENV_PYTHON \
                AITER_VENV_PYTHON \
                FLASH_ATTENTION_VENV_PYTHON \
                MPI4PY_VENV_PYTHON \
                MEGATRON_VENV_PYTHON \
                WANDB_VENV_PYTHON \
                MLSTACK_PYTHON_BIN \
                UV_PYTHON; do
                candidate="${!var_name:-}"
                if benchmark_python_exists "$candidate"; then
                    chosen="$candidate"
                    break
                fi
            done
            ;;
        *)
            for var_name in MLSTACK_BENCHMARK_PYTHON MLSTACK_PYTHON_BIN UV_PYTHON python3 python; do
                if [ "$var_name" = "python3" ] || [ "$var_name" = "python" ]; then
                    candidate="$var_name"
                else
                    candidate="${!var_name:-}"
                fi
                if benchmark_python_exists "$candidate"; then
                    chosen="$candidate"
                    break
                fi
            done
            ;;
    esac

    if [ -z "$chosen" ]; then
        chosen="${MLSTACK_PYTHON_BIN:-${UV_PYTHON:-python3}}"
    fi

    printf '%s\n' "$chosen"
}

benchmark_prepare_rocm_runtime() {
    benchmark_load_global_env
    if declare -f mlstack_enforce_global_install_contract >/dev/null 2>&1; then
        mlstack_enforce_global_install_contract
    fi

    # Some ROCm Python wheels bundle libdrm that defaults to /opt/amdgpu/share/libdrm.
    # Force ASIC ID lookup to the distro table so GPU names resolve consistently.
    if [ -f "/usr/share/libdrm/amdgpu.ids" ]; then
        export AMDGPU_ASIC_ID_TABLE_PATH="/usr/share/libdrm/amdgpu.ids"
        export AMDGPU_ASIC_ID_TABLE_PATHS="/usr/share/libdrm"
    fi

    local install_method
    local benchmark_python
    local discrete_gpu_list
    install_method="$(benchmark_normalize_install_method "${MLSTACK_INSTALL_METHOD:-${INSTALL_METHOD:-auto}}")"
    benchmark_python="$(benchmark_resolve_python_for_method "$install_method")"
    discrete_gpu_list="$(benchmark_detect_discrete_gpu_list)"
    discrete_gpu_list="$(benchmark_select_working_gpu_list "$benchmark_python" "$discrete_gpu_list")"

    export MLSTACK_INSTALL_METHOD="$install_method"
    export INSTALL_METHOD="$install_method"
    export MLSTACK_BENCHMARK_PYTHON="$benchmark_python"
    export HIP_VISIBLE_DEVICES="$discrete_gpu_list"
    export CUDA_VISIBLE_DEVICES="$discrete_gpu_list"
    export PYTORCH_ROCM_DEVICE="${discrete_gpu_list%%,*}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
    benchmark_prepare_triton_cache_env || true
    if [ "$install_method" = "global" ]; then
        export MLSTACK_PYTHON_BIN="$benchmark_python"
        export UV_PYTHON="$benchmark_python"
    fi

    : "${MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL:=HuggingFaceTB/SmolLM2-135M-Instruct}"
    export MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL
    if [ -z "${MLSTACK_BENCH_VLLM_GGUF_MODEL_PATH:-}" ] && [ -n "${MLSTACK_TINY_GGUF_MODEL_PATH:-}" ]; then
        export MLSTACK_BENCH_VLLM_GGUF_MODEL_PATH="$MLSTACK_TINY_GGUF_MODEL_PATH"
    fi
    : "${MLSTACK_BENCH_VLLM_GGUF_TOKENIZER:=$MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL}"
    export MLSTACK_BENCH_VLLM_GGUF_TOKENIZER

    local vllm_target_device_raw="${VLLM_TARGET_DEVICE:-}"
    local vllm_target_device_trimmed
    vllm_target_device_trimmed="$(printf '%s' "$vllm_target_device_raw" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    case "$vllm_target_device_trimmed" in
        rocm|cuda|cpu) ;;
        *)
            vllm_target_device_trimmed="rocm"
            ;;
    esac
    export VLLM_TARGET_DEVICE="$vllm_target_device_trimmed"

    benchmark_info "ROCm runtime from global env: install_method=${MLSTACK_INSTALL_METHOD:-unset}, benchmark_python=${MLSTACK_BENCHMARK_PYTHON:-unset}, GPU_ARCH=${GPU_ARCH:-unset}, HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-unset}, PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-unset}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-unset}, VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-unset}, VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-unset}, VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-unset}, TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-unset}, tiny_vllm_safetensors=${MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL:-unset}, tiny_vllm_gguf=${MLSTACK_BENCH_VLLM_GGUF_MODEL_PATH:-unset}"
}

benchmark_json_parser_python() {
    local candidate
    for candidate in \
        "${MLSTACK_BENCHMARK_PYTHON:-}" \
        "${MLSTACK_PYTHON_BIN:-}" \
        "${UV_PYTHON:-}" \
        python3 \
        python; do
        if benchmark_python_exists "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '%s\n' "python3"
}

benchmark_python_module_exists() {
    local python_bin="$1"
    local module_name="$2"

    "$python_bin" - "$module_name" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

name = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(name) is not None else 1)
PY
}

benchmark_python_can_import_module() {
    local python_bin="$1"
    local module_name="$2"

    "$python_bin" - "$module_name" <<'PY' >/dev/null 2>&1
import importlib
import sys

name = sys.argv[1]
importlib.import_module(name)
PY
}

benchmark_pip_install_for_python() {
    local python_bin="$1"
    shift
    if [ "$#" -eq 0 ]; then
        return 0
    fi

    if declare -f mlstack_pip_install >/dev/null 2>&1; then
        mlstack_pip_install "$python_bin" "$@"
        return $?
    fi

    local -a maybe_break=()
    if "$python_bin" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
        maybe_break=(--break-system-packages)
    fi
    "$python_bin" -m pip install "${maybe_break[@]}" "$@"
}

benchmark_vllm_pkg_requires_no_deps() {
    local req="${1:-}"
    local base
    base="$(benchmark_requirement_base_name "$req")"
    case "$base" in
        xgrammar|triton-kernels|conch-triton-kernels)
            return 0
            ;;
    esac
    return 1
}

benchmark_install_vllm_dependency() {
    local python_bin="$1"
    local req_spec="${2:-}"
    local log_file="${3:-/dev/null}"

    [ -n "$req_spec" ] || return 0

    if benchmark_vllm_pkg_requires_no_deps "$req_spec"; then
        benchmark_info "Installing vLLM dependency with --no-deps due ROCm triton constraints: $req_spec"
        benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --no-deps --extra-index-url https://wheels.vllm.ai/rocm/ "$req_spec" >>"$log_file" 2>&1
        return $?
    fi

    benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --extra-index-url https://wheels.vllm.ai/rocm/ "$req_spec" >>"$log_file" 2>&1
}

benchmark_is_safe_vllm_pkg() {
    local pkg="${1,,}"
    pkg="${pkg%%;*}"
    pkg="${pkg%%,*}"
    pkg="${pkg%% *}"
    pkg="${pkg%%[*}"
    pkg="${pkg%%<*}"
    pkg="${pkg%%>*}"
    pkg="${pkg%%=*}"
    pkg="${pkg%%!*}"
    pkg="${pkg%%~*}"
    pkg="${pkg%%(*}"
    pkg="${pkg//_/-}"
    pkg="${pkg// /}"
    case "$pkg" in
        *nvidia*|*cuda*|*cudnn*|*cublas*|*cufft*|*curand*|*cusolver*|*cusparse*|*nccl*|*nvtx*|*nvjitlink*|*tensorrt*)
            return 1
            ;;
    esac
    case "$pkg" in
        torch|torchvision|torchaudio|triton|xformers)
            return 1
            ;;
    esac
    return 0
}

benchmark_requirement_base_name() {
    local req="${1:-}"
    req="${req%%;*}"
    req="${req%%,*}"
    req="${req%% *}"
    req="${req%%[*}"
    req="${req%%<*}"
    req="${req%%>*}"
    req="${req%%=*}"
    req="${req%%!*}"
    req="${req%%~*}"
    req="${req%%(*}"
    req="${req//_/-}"
    req="${req// /}"
    printf '%s\n' "${req,,}"
}

benchmark_requirement_pinned_version() {
    local req="${1:-}"
    printf '%s\n' "$req" | sed -nE 's/^[^=<>!~]+==([^; ,]+).*$/\1/p'
}

benchmark_requirement_has_specifier() {
    local req="${1:-}"
    [[ "$req" == *"=="* || "$req" == *">="* || "$req" == *"<="* || "$req" == *"~="* || "$req" == *"!="* || "$req" == *"<"* || "$req" == *">"* ]]
}

benchmark_python_requirement_satisfied() {
    local python_bin="$1"
    local req_spec="${2:-}"
    local fallback_name="${3:-}"
    "$python_bin" - "$req_spec" "$fallback_name" <<'PY' >/dev/null 2>&1
import importlib.metadata
import re
import sys

req_spec = (sys.argv[1] or "").strip()
fallback_name = (sys.argv[2] or "").strip().lower().replace("_", "-")
if not req_spec and not fallback_name:
    raise SystemExit(1)

try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None

parsed = None
name = fallback_name
if req_spec:
    if Requirement is not None:
        try:
            parsed = Requirement(req_spec)
            name = parsed.name.lower().replace("_", "-")
        except Exception:
            parsed = None
    if not name:
        head = req_spec.split(";", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)", head)
        if match:
            name = match.group(1).lower().replace("_", "-")

if not name:
    raise SystemExit(1)

def requirement_match(version: str) -> bool:
    if parsed is None:
        return True
    spec = parsed.specifier
    if not spec:
        return True
    if spec.contains(version, prereleases=True):
        return True
    base = version.split("+", 1)[0]
    if base != version and spec.contains(base, prereleases=True):
        return True
    return False

try:
    installed = importlib.metadata.version(name)
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if requirement_match(installed) else 1)
PY
}

benchmark_vllm_abi_requirement_for_name() {
    local python_bin="$1"
    local target_name="${2:-}"
    "$python_bin" - "$target_name" <<'PY'
import importlib.metadata as md
import sys

target = sys.argv[1].strip().lower().replace("_", "-")
if not target:
    raise SystemExit(1)

requirements = md.requires("vllm") or []
try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None

for raw in requirements:
    raw = (raw or "").strip()
    if not raw:
        continue
    if Requirement is not None:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:
                continue
        name = req.name.lower().replace("_", "-")
        if name != target:
            continue
        spec = str(req.specifier) if req.specifier else ""
        print(f"{name}{spec}")
        raise SystemExit(0)
    else:
        head = raw.split(";", 1)[0].strip().replace(" ", "")
        name = head.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].split("!", 1)[0].split("~", 1)[0].lower().replace("_", "-")
        if name == target:
            print(head)
            raise SystemExit(0)

raise SystemExit(1)
PY
}

benchmark_collect_vllm_abi_requirements() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import importlib.metadata as md

targets = {"torch", "torchvision", "torchaudio", "triton"}
requirements = md.requires("vllm") or []
try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None

seen = set()
for raw in requirements:
    raw = (raw or "").strip()
    if not raw:
        continue

    if Requirement is not None:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:
                continue
        name = req.name.lower().replace("_", "-")
        if name not in targets:
            continue
        spec = str(req.specifier) if req.specifier else ""
        normalized = f"{name}{spec}"
    else:
        head = raw.split(";", 1)[0].strip().replace(" ", "")
        name = head.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].split("!", 1)[0].split("~", 1)[0].lower().replace("_", "-")
        if name not in targets:
            continue
        normalized = head

    if normalized in seen:
        continue
    seen.add(normalized)
    print(normalized)
PY
}

benchmark_reconcile_vllm_abi_requirements() {
    local python_bin="$1"
    local log_file="${2:-/dev/null}"
    local req_spec req_name

    if ! "$python_bin" -c "import importlib.metadata as md; md.version('vllm')" >/dev/null 2>&1; then
        benchmark_warn "vLLM not installed; skipping ABI requirement reconciliation"
        return 0
    fi

    while IFS= read -r req_spec; do
        req_spec="${req_spec// /}"
        [ -n "$req_spec" ] || continue
        req_name="$(benchmark_requirement_base_name "$req_spec")"
        [ -n "$req_name" ] || continue

        if [ "$req_name" = "triton" ]; then
            if benchmark_rocm_triton_satisfies_requirement "$python_bin" "$req_spec"; then
                benchmark_info "vLLM ABI requirement already satisfied by installed ROCm Triton runtime: ${req_spec}"
                continue
            fi
        elif benchmark_python_requirement_satisfied "$python_bin" "$req_spec" "$req_name"; then
            benchmark_info "vLLM ABI requirement already satisfied: ${req_spec}"
            continue
        fi

        benchmark_info "Reconciling vLLM ABI requirement from metadata: ${req_spec}"
        if ! benchmark_install_vllm_abi_requirement "$python_bin" "$req_spec" "$log_file"; then
            benchmark_error "Failed to reconcile vLLM ABI requirement: ${req_spec}"
            return 1
        fi
    done < <(benchmark_collect_vllm_abi_requirements "$python_bin")

    return 0
}

benchmark_install_vllm_abi_requirement() {
    local python_bin="$1"
    local req_spec="${2:-}"
    local log_file="${3:-/dev/null}"
    local req_name version no_local rocm_mm rocm_index candidate
    local -a pip_dep_flags=()
    local -a triton_candidates

    rocm_mm="$(benchmark_detect_rocm_mm)"
    rocm_index="https://repo.radeon.com/rocm/manylinux/rocm-rel-${rocm_mm}/"

    req_name="$(benchmark_requirement_base_name "$req_spec")"
    version="$(benchmark_requirement_pinned_version "$req_spec")"
    if [[ "$req_name" == "torchvision" || "$req_name" == "torchaudio" || "$req_name" == "triton" ]]; then
        pip_dep_flags=(--no-deps)
    fi

    case "$req_name" in
        triton)
            if [ -n "$version" ]; then
                benchmark_info "Enforcing vLLM ABI-compatible Triton runtime for ${req_spec}"
                triton_candidates=("triton==${version}")
                if [[ "$version" == *+* ]]; then
                    no_local="${version%%+*}"
                    if [ -n "$no_local" ]; then
                        triton_candidates+=("triton==${no_local}")
                    fi
                fi
                triton_candidates+=("triton-rocm==${version}")
                if [[ "$version" == *+* && -n "$no_local" ]]; then
                    triton_candidates+=("triton-rocm==${no_local}")
                fi
                for candidate in "${triton_candidates[@]}"; do
                    if benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                        --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                        "$candidate" >>"$log_file" 2>&1; then
                        return 0
                    fi
                done
                benchmark_error "Unable to install ABI-pinned Triton runtime required by vLLM (${req_spec})"
                return 1
            fi
            for candidate in triton triton-rocm; do
                if benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                    --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                    "$candidate" >>"$log_file" 2>&1; then
                    return 0
                fi
            done
            return 1
            ;;
        torch|torchvision|torchaudio)
            if [ -n "$req_spec" ] && benchmark_requirement_has_specifier "$req_spec"; then
                benchmark_info "Enforcing vLLM ABI-compatible ${req_name}: ${req_spec}"
                if benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                    --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                    "$req_spec" >>"$log_file" 2>&1; then
                    benchmark_verify_torch_runtime_loadable "$python_bin" || return 1
                    return 0
                fi
                if [[ "$version" == *+* ]]; then
                    no_local="${version%%+*}"
                    if [ -n "$no_local" ] && benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                        --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                        "${req_name}==${no_local}" >>"$log_file" 2>&1; then
                        benchmark_verify_torch_runtime_loadable "$python_bin" || return 1
                        return 0
                    fi
                fi
                benchmark_error "Unable to install ABI-pinned ${req_name} required by vLLM (${req_spec})"
                return 1
            fi
            if benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_name" >>"$log_file" 2>&1; then
                benchmark_verify_torch_runtime_loadable "$python_bin" || return 1
                return 0
            fi
            return 1
            ;;
    esac

    return 1
}

benchmark_extract_vllm_requirement_from_pip_check_line() {
    local line="${1:-}"
    local req=""

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? requires (.*), which is not installed\.$/\2/p')"
    if [ -n "$req" ]; then
        printf '%s\n' "$req"
        return 0
    fi

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? requires (.*), but you have .* which is incompatible\.$/\2/p')"
    if [ -n "$req" ]; then
        printf '%s\n' "$req"
        return 0
    fi

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? has requirement (.*), but you have .*$/\2/p')"
    if [ -n "$req" ]; then
        printf '%s\n' "$req"
        return 0
    fi

    return 1
}

benchmark_detect_rocm_mm() {
    local rocm_raw="${ROCM_VERSION:-}"
    if [ -z "$rocm_raw" ] && [ -f /opt/rocm/.info/version ]; then
        rocm_raw="$(head -n1 /opt/rocm/.info/version 2>/dev/null || true)"
    fi
    rocm_raw="$(printf '%s\n' "$rocm_raw" | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
    case "$rocm_raw" in
        5.7|6.0|6.1|6.2|6.3|6.4|7.0|7.1|7.2)
            printf '%s\n' "$rocm_raw"
            ;;
        *)
            printf '%s\n' "7.2"
            ;;
    esac
}

benchmark_is_rocm_torch_pkg() {
    local pkg="${1,,}"
    case "$pkg" in
        torch|torchvision|torchaudio|triton)
            return 0
            ;;
    esac
    return 1
}

benchmark_has_rocm_triton_runtime() {
    local python_bin="$1"
    "$python_bin" - <<'PY' >/dev/null 2>&1
import importlib.util
import importlib.metadata
import sys

if importlib.util.find_spec("triton") is not None:
    raise SystemExit(0)

try:
    importlib.metadata.version("triton-rocm")
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
}

benchmark_shared_lib_exists() {
    local lib_name="$1"
    if command -v ldconfig >/dev/null 2>&1; then
        if ldconfig -p 2>/dev/null | grep -q -- "$lib_name"; then
            return 0
        fi
    fi
    local compat_uid
    compat_uid="$(benchmark_effective_home_uid)"
    local path
    for path in /usr/lib /usr/lib64 /opt/rocm/lib /opt/rocm/hipsparselt/lib \
        "$HOME/.mlstack/libmpi-compat" "$HOME/.mlstack/libmpi-compat-user-${compat_uid}"; do
        if [ -e "$path/$lib_name" ]; then
            return 0
        fi
    done
    return 1
}

benchmark_effective_home_uid() {
    local uid_candidate="${MLSTACK_TARGET_UID:-}"
    if [[ "$uid_candidate" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$uid_candidate"
        return 0
    fi
    if [[ "${SUDO_UID:-}" =~ ^[0-9]+$ ]] && [ "${SUDO_UID}" -gt 0 ]; then
        printf '%s\n' "${SUDO_UID}"
        return 0
    fi
    uid_candidate="$(stat -c '%u' "$HOME" 2>/dev/null || true)"
    if [[ "$uid_candidate" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$uid_candidate"
        return 0
    fi
    printf '%s\n' "${UID:-$(id -u 2>/dev/null || echo 0)}"
}

benchmark_effective_home_gid() {
    local gid_candidate="${MLSTACK_TARGET_GID:-}"
    if [[ "$gid_candidate" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$gid_candidate"
        return 0
    fi
    if [[ "${SUDO_GID:-}" =~ ^[0-9]+$ ]] && [ "${SUDO_GID}" -gt 0 ]; then
        printf '%s\n' "${SUDO_GID}"
        return 0
    fi
    gid_candidate="$(stat -c '%g' "$HOME" 2>/dev/null || true)"
    if [[ "$gid_candidate" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$gid_candidate"
        return 0
    fi
    printf '%s\n' "$(id -g 2>/dev/null || echo 0)"
}

benchmark_ld_prepend_once() {
    local path="$1"
    [ -n "$path" ] || return 1
    if [ ! -d "$path" ]; then
        return 1
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$path:"*) ;;
        *) export LD_LIBRARY_PATH="$path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    return 0
}

benchmark_mpi_cxx_compat_dir_candidates() {
    local compat_uid
    compat_uid="$(benchmark_effective_home_uid)"
    printf '%s\n' "$HOME/.mlstack/libmpi-compat" "$HOME/.mlstack/libmpi-compat-user-${compat_uid}"
}

benchmark_mpi_cxx_stub_has_required_symbols() {
    local stub_so="$1"
    [ -r "$stub_so" ] || return 1
    command -v nm >/dev/null 2>&1 || return 1
    local symbol
    for symbol in \
        "_ZN3MPI3Win4FreeEv" \
        "_ZN3MPI4CommC2Ev" \
        "_ZN3MPI8Datatype4FreeEv" \
        "ompi_mpi_cxx_op_intercept" \
        "ompi_op_set_cxx_callback"; do
        if ! nm -D "$stub_so" 2>/dev/null | grep -q -- "$symbol"; then
            return 1
        fi
    done
    return 0
}

benchmark_activate_existing_mpi_cxx_shim() {
    local compat_dir stub_so
    while IFS= read -r compat_dir; do
        [ -n "$compat_dir" ] || continue
        stub_so="$compat_dir/libmpi_cxx.so.40"
        if [ -e "$stub_so" ] && benchmark_mpi_cxx_stub_has_required_symbols "$stub_so"; then
            benchmark_ld_prepend_once "$compat_dir" || true
            return 0
        fi
    done < <(benchmark_mpi_cxx_compat_dir_candidates)
    return 1
}

benchmark_ensure_libmpi_cxx_compat() {
    local compat_dir stub_so stub_cpp cxx target_uid target_gid
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libmpi_cxx.so.40"; then
        return 0
    fi
    if benchmark_activate_existing_mpi_cxx_shim; then
        benchmark_warn "Using existing MPI C++ compatibility shim"
        return 0
    fi
    if ! benchmark_shared_lib_exists "libmpi.so.40"; then
        benchmark_error "Missing libmpi_cxx.so.40 and libmpi.so.40; cannot create MPI C++ compatibility shim"
        return 1
    fi
    if command -v g++ >/dev/null 2>&1; then
        cxx="g++"
    elif command -v c++ >/dev/null 2>&1; then
        cxx="c++"
    else
        benchmark_error "Missing g++/c++; cannot create MPI C++ compatibility shim for libmpi_cxx.so.40"
        return 1
    fi
    target_uid="$(benchmark_effective_home_uid)"
    target_gid="$(benchmark_effective_home_gid)"

    while IFS= read -r compat_dir; do
        [ -n "$compat_dir" ] || continue
        mkdir -p "$compat_dir" 2>/dev/null || continue
        [ -w "$compat_dir" ] || continue
        stub_so="$compat_dir/libmpi_cxx.so.40"
        stub_cpp="$compat_dir/mpi_cxx_stub.cpp"
        cat > "$stub_cpp" <<'CPP'
extern "C" void mlstack_mpi_win_free(void*) asm("_ZN3MPI3Win4FreeEv");
extern "C" void mlstack_mpi_win_free(void*) {}

extern "C" void mlstack_mpi_datatype_free(void*) asm("_ZN3MPI8Datatype4FreeEv");
extern "C" void mlstack_mpi_datatype_free(void*) {}

extern "C" void mlstack_mpi_comm_ctor(void*) asm("_ZN3MPI4CommC2Ev");
extern "C" void mlstack_mpi_comm_ctor(void*) {}

extern "C" void ompi_mpi_cxx_op_intercept(void) {}
extern "C" void ompi_op_set_cxx_callback(void) {}
CPP
        if ! "$cxx" -shared -fPIC "$stub_cpp" -Wl,-soname,libmpi_cxx.so.40 -Wl,--no-as-needed -lmpi -o "$stub_so"; then
            continue
        fi
        if ! benchmark_mpi_cxx_stub_has_required_symbols "$stub_so"; then
            continue
        fi
        if [ "$(id -u 2>/dev/null || echo 0)" -eq 0 ] && [ "$target_uid" != "0" ] && command -v chown >/dev/null 2>&1; then
            chown "$target_uid:$target_gid" "$stub_so" "$stub_cpp" >/dev/null 2>&1 || true
        fi
        benchmark_ld_prepend_once "$compat_dir" || true
        benchmark_warn "Using generated MPI C++ compatibility shim at $stub_so"
        return 0
    done < <(benchmark_mpi_cxx_compat_dir_candidates)

    benchmark_error "Failed to build a usable MPI C++ compatibility shim"
    return 1
}

benchmark_report_hipsparselt_missing() {
    benchmark_error "Missing libhipsparselt.so.0 required by this vLLM/Torch ABI."
    benchmark_error "Install ROCm hipsparselt runtime package (typically: hipsparselt), then retry."
}

benchmark_try_install_system_hipsparselt() {
    if benchmark_shared_lib_exists "libhipsparselt.so.0"; then
        return 0
    fi

    benchmark_sudo_run() {
        if sudo -n true >/dev/null 2>&1; then
            sudo "$@"
            return $?
        fi
        if [ -n "${MLSTACK_SUDO_PASSWORD:-}" ]; then
            printf '%s\n' "$MLSTACK_SUDO_PASSWORD" | sudo -S -p '' "$@"
            return $?
        fi
        return 1
    }

    if command -v apt-get >/dev/null 2>&1; then
        benchmark_info "Attempting non-interactive install of hipsparselt via apt"
        benchmark_sudo_run apt-get update -y >/dev/null 2>&1 || true
        if benchmark_sudo_run apt-get install -y hipsparselt >/dev/null 2>&1; then
            benchmark_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        benchmark_warn "Automatic apt install for hipsparselt did not complete"
    fi

    if command -v dnf >/dev/null 2>&1; then
        benchmark_info "Attempting non-interactive install of hipsparselt via dnf"
        if benchmark_sudo_run dnf install -y hipsparselt >/dev/null 2>&1; then
            benchmark_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        benchmark_warn "Automatic dnf install for hipsparselt did not complete"
    fi

    if command -v yum >/dev/null 2>&1; then
        benchmark_info "Attempting non-interactive install of hipsparselt via yum"
        if benchmark_sudo_run yum install -y hipsparselt >/dev/null 2>&1; then
            benchmark_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        benchmark_warn "Automatic yum install for hipsparselt did not complete"
    fi

    if command -v zypper >/dev/null 2>&1; then
        benchmark_info "Attempting non-interactive install of hipsparselt via zypper"
        if benchmark_sudo_run zypper --non-interactive install -y hipsparselt >/dev/null 2>&1; then
            benchmark_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        benchmark_warn "Automatic zypper install for hipsparselt did not complete"
    fi

    if command -v pacman >/dev/null 2>&1; then
        benchmark_info "Attempting non-interactive install of hipsparselt via pacman"
        if benchmark_sudo_run pacman -S --needed --noconfirm hipsparselt >/dev/null 2>&1; then
            benchmark_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        benchmark_warn "Automatic pacman install for hipsparselt did not complete"
    fi

    return 1
}

benchmark_verify_torch_runtime_loadable() {
    local python_bin="$1"
    local output
    local pass
    benchmark_activate_existing_mpi_cxx_shim || true
    for pass in 1 2 3; do
        output="$("$python_bin" - <<'PY' 2>&1
import traceback
try:
    import torch  # noqa: F401
    print("ok")
except Exception as exc:
    print(exc)
    traceback.print_exc()
    raise
PY
)"
        if [ $? -eq 0 ]; then
            return 0
        fi

        if printf '%s\n' "$output" | grep -Eq "libmpi_cxx\\.so\\.40|_ZN3MPI|ompi_mpi_cxx_op_intercept|ompi_op_set_cxx_callback"; then
            if benchmark_ensure_libmpi_cxx_compat; then
                continue
            fi
        fi

        if printf '%s\n' "$output" | grep -q "libhipsparselt.so.0"; then
            if benchmark_try_install_system_hipsparselt; then
                continue
            fi
            benchmark_report_hipsparselt_missing
            return 1
        fi

        break
    done

    if printf '%s\n' "$output" | grep -q "libhipsparselt.so.0"; then
        benchmark_report_hipsparselt_missing
    fi
    benchmark_error "Torch runtime import failed after ABI reconciliation"
    benchmark_warn "$(printf '%s\n' "$output" | tail -n 20 | tr '\n' ' ')"
    return 1
}

benchmark_rocm_triton_satisfies_requirement() {
    local python_bin="$1"
    local req_spec="${2:-triton}"
    "$python_bin" - "$req_spec" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys

req = sys.argv[1].strip()
versions = []
for pkg in ("triton", "triton-rocm"):
    try:
        versions.append(importlib.metadata.version(pkg))
    except Exception:
        pass
if not versions:
    raise SystemExit(1)

try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(0 if ("==" not in req and ">=" not in req and "<=" not in req and "~=" not in req and "!=" not in req and "<" not in req and ">" not in req) else 1)

try:
    parsed = Requirement(req)
except Exception:
    raise SystemExit(1)

spec = parsed.specifier
if not spec:
    raise SystemExit(0)

for ver in versions:
    if spec.contains(ver, prereleases=True):
        raise SystemExit(0)
    base = ver.split("+", 1)[0]
    if spec.contains(base, prereleases=True):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

benchmark_pip_check_line_is_incompatible() {
    local line="${1:-}"
    [[ "$line" == *", but you have "* ]] && return 0
    return 1
}

benchmark_install_rocm_torch_requirement() {
    local python_bin="$1"
    local req_spec="$2"
    local log_file="${3:-/dev/null}"
    local rocm_mm rocm_index req_name

    rocm_mm="$(benchmark_detect_rocm_mm)"
    rocm_index="https://repo.radeon.com/rocm/manylinux/rocm-rel-${rocm_mm}/"
    req_name="$(benchmark_requirement_base_name "$req_spec")"
    if [ "$req_name" = "triton" ]; then
        req_spec="triton-rocm"
        req_name="triton-rocm"
    fi
    benchmark_info "Attempting ROCm-aligned install for vLLM dependency: $req_spec (ROCm ${rocm_mm})"

    if benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_spec" >>"$log_file" 2>&1; then
        return 0
    fi

    benchmark_warn "Exact ROCm install failed for '$req_spec'; retrying by base package name '$req_name'"
    benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_name" >>"$log_file" 2>&1
}

benchmark_install_vllm_missing_from_pip_check() {
    local python_bin="$1"
    local log_file="${2:-/dev/null}"
    local pass installs_this_pass pip_output line req_spec req_name
    local failed=0

    for pass in $(seq 1 6); do
        pip_output="$("$python_bin" -m pip check 2>/dev/null || true)"
        if [ -z "$pip_output" ]; then
            return 0
        fi
        installs_this_pass=0

        while IFS= read -r line; do
            req_spec="$(benchmark_extract_vllm_requirement_from_pip_check_line "$line" || true)"
            if [ -z "$req_spec" ]; then
                continue
            fi

            req_name="$(benchmark_requirement_base_name "$req_spec")"
            if [ -z "$req_name" ]; then
                benchmark_warn "Could not parse vLLM requirement from pip check line: $line"
                failed=1
                continue
            fi

            if benchmark_is_rocm_torch_pkg "$req_name" && ! benchmark_requirement_has_specifier "$req_spec"; then
                local meta_req
                meta_req="$(benchmark_vllm_abi_requirement_for_name "$python_bin" "$req_name" 2>/dev/null || true)"
                if [ -n "$meta_req" ]; then
                    benchmark_info "Upgrading pip-check ABI requirement '${req_spec}' to metadata pin '${meta_req}'"
                    req_spec="$meta_req"
                fi
            fi

            if [ "$req_name" = "triton" ]; then
                if benchmark_rocm_triton_satisfies_requirement "$python_bin" "$req_spec"; then
                    benchmark_info "pip-check requirement already satisfied by installed ROCm Triton runtime: $req_spec"
                    continue
                fi
            elif benchmark_python_requirement_satisfied "$python_bin" "$req_spec" "$req_name"; then
                benchmark_info "pip-check requirement already satisfied: $req_spec"
                continue
            fi

            if benchmark_pip_check_line_is_incompatible "$line"; then
                if benchmark_is_rocm_torch_pkg "$req_name"; then
                    benchmark_info "Resolving vLLM ABI requirement from pip check: $req_spec"
                    if ! benchmark_install_vllm_abi_requirement "$python_bin" "$req_spec" "$log_file"; then
                        benchmark_error "Could not enforce ABI-compatible ROCm dependency required by vLLM: $req_spec"
                        failed=1
                    fi
                    installs_this_pass=$((installs_this_pass + 1))
                    continue
                fi
                if ! benchmark_is_safe_vllm_pkg "$req_name"; then
                    benchmark_warn "Skipping blocked incompatible package while repairing vLLM deps: $req_name"
                    failed=1
                    continue
                fi
                benchmark_info "Reconciling incompatible vLLM dependency from pip check (pass $pass): $req_spec"
                if ! benchmark_install_vllm_dependency "$python_bin" "$req_spec" "$log_file"; then
                    benchmark_warn "Requirement reconciliation failed, retrying by package name: $req_name"
                    if ! benchmark_install_vllm_dependency "$python_bin" "$req_name" "$log_file"; then
                        benchmark_error "Could not reconcile incompatible vLLM dependency: $req_spec"
                        failed=1
                    fi
                fi
                installs_this_pass=$((installs_this_pass + 1))
                continue
            fi

            if [ "$req_name" = "triton" ] && benchmark_rocm_triton_satisfies_requirement "$python_bin" "$req_spec"; then
                benchmark_info "Treating vLLM triton requirement as satisfied by installed ROCm Triton runtime (${req_spec})"
                continue
            fi

            if benchmark_is_rocm_torch_pkg "$req_name"; then
                if ! benchmark_install_vllm_abi_requirement "$python_bin" "$req_spec" "$log_file" \
                    && ! benchmark_install_rocm_torch_requirement "$python_bin" "$req_spec" "$log_file"; then
                    benchmark_error "Could not install ROCm-compatible torch dependency required by vLLM: $req_spec"
                    failed=1
                fi
                installs_this_pass=$((installs_this_pass + 1))
                continue
            fi

            if ! benchmark_is_safe_vllm_pkg "$req_name"; then
                benchmark_warn "Skipping blocked package while repairing vLLM deps: $req_name"
                failed=1
                continue
            fi

            benchmark_info "Installing missing/incompatible vLLM dependency from pip check (pass $pass): $req_spec"
            if ! benchmark_install_vllm_dependency "$python_bin" "$req_spec" "$log_file"; then
                benchmark_warn "Requirement install failed, retrying by package name: $req_name"
                if ! benchmark_install_vllm_dependency "$python_bin" "$req_name" "$log_file"; then
                    benchmark_error "Could not install required vLLM dependency: $req_spec"
                    failed=1
                fi
            fi
            installs_this_pass=$((installs_this_pass + 1))
        done <<< "$pip_output"

        if [ "$installs_this_pass" -eq 0 ]; then
            break
        fi
    done

    if [ "$failed" -eq 0 ]; then
        return 0
    fi
    return 1
}

benchmark_vllm_missing_module_to_package() {
    local module_name="${1:-}"
    module_name="${module_name//\"/}"
    module_name="${module_name//\'/}"
    case "$module_name" in
        cachetools) printf '%s\n' "cachetools" ;;
        cbor2) printf '%s\n' "cbor2" ;;
        cloudpickle) printf '%s\n' "cloudpickle" ;;
        aiohttp) printf '%s\n' "aiohttp" ;;
        fastapi) printf '%s\n' "fastapi" ;;
        msgspec) printf '%s\n' "msgspec" ;;
        prometheus_client) printf '%s\n' "prometheus-client" ;;
        psutil) printf '%s\n' "psutil" ;;
        cpuinfo) printf '%s\n' "py-cpuinfo" ;;
        zmq) printf '%s\n' "pyzmq" ;;
        requests) printf '%s\n' "requests" ;;
        sentencepiece) printf '%s\n' "sentencepiece" ;;
        tiktoken) printf '%s\n' "tiktoken" ;;
        uvicorn) printf '%s\n' "uvicorn" ;;
        einops) printf '%s\n' "einops" ;;
        transformers) printf '%s\n' "transformers" ;;
        huggingface_hub) printf '%s\n' "huggingface-hub" ;;
        accelerate) printf '%s\n' "accelerate" ;;
        blake3) printf '%s\n' "blake3" ;;
        lark) printf '%s\n' "lark" ;;
        watchfiles) printf '%s\n' "watchfiles" ;;
        pythonjsonlogger) printf '%s\n' "python-json-logger" ;;
        openai) printf '%s\n' "openai" ;;
        openai_harmony|openai_harmony.*) printf '%s\n' "openai-harmony>=0.0.3" ;;
        partial_json_parser) printf '%s\n' "partial-json-parser" ;;
        compressed_tensors) printf '%s\n' "compressed-tensors" ;;
        depyf) printf '%s\n' "depyf" ;;
        gguf) printf '%s\n' "gguf" ;;
        llguidance|llguidance.*) printf '%s\n' "llguidance>=1.3.0,<1.4.0" ;;
        lmformatenforcer) printf '%s\n' "lm-format-enforcer" ;;
        amdsmi) printf '%s\n' "amdsmi" ;;
        numba) printf '%s\n' "numba" ;;
        outlines) printf '%s\n' "outlines" ;;
        outlines_core|outlines_core.*) printf '%s\n' "outlines-core==0.2.11" ;;
        prometheus_fastapi_instrumentator) printf '%s\n' "prometheus-fastapi-instrumentator" ;;
        mistral_common|mistral_common.*) printf '%s\n' "mistral-common[image]>=1.9.0" ;;
        triton_kernels|triton_kernels.*) printf '%s\n' "triton-kernels==1.0.0" ;;
        xgrammar|xgrammar.*) printf '%s\n' "xgrammar==0.1.29" ;;
        packaging) printf '%s\n' "packaging" ;;
        typing_extensions) printf '%s\n' "typing-extensions" ;;
        importlib_metadata) printf '%s\n' "importlib-metadata" ;;
        attr) printf '%s\n' "attrs" ;;
        yaml) printf '%s\n' "pyyaml" ;;
        google.protobuf|google.protobuf.*) printf '%s\n' "protobuf" ;;
        pkg_resources) printf '%s\n' "setuptools" ;;
        pybase64) printf '%s\n' "pybase64" ;;
        ijson) printf '%s\n' "ijson" ;;
        setproctitle) printf '%s\n' "setproctitle" ;;
        six) printf '%s\n' "six" ;;
        *)
            printf '%s\n' "${module_name%%.*}" | tr '_' '-'
            ;;
    esac
}

benchmark_repair_vllm_import() {
    local python_bin="$1"
    local log_file="${2:-/dev/null}"
    local attempt output missing_module package_name

    for attempt in $(seq 1 20); do
        output="$("$python_bin" - <<'PY' 2>&1
try:
    import vllm
except Exception as exc:
    print(exc)
    raise
print("ok")
PY
)"
        if [ $? -eq 0 ]; then
            return 0
        fi

        missing_module="$(printf '%s\n' "$output" | sed -n "s/.*No module named ['\"]\([^'\"]\+\)['\"].*/\1/p" | tail -n1)"
        if [ -z "$missing_module" ]; then
            if printf '%s\n' "$output" | grep -qiE 'vllm\._C|vllm\._rocm_C|undefined symbol|native extensions failed to load'; then
                benchmark_warn "vLLM import failed with native extension/ABI error; attempting native extension repair"
                if benchmark_repair_vllm_native_extensions "$python_bin" "$log_file"; then
                    return 0
                fi
            fi
            benchmark_error "vLLM import failed with non-module error: $(printf '%s\n' "$output" | tail -n1)"
            return 1
        fi

        package_name="$(benchmark_vllm_missing_module_to_package "$missing_module")"
        if ! benchmark_is_safe_vllm_pkg "$package_name"; then
            benchmark_error "Refusing to install blocked package while repairing vLLM import: $package_name"
            return 1
        fi

        benchmark_warn "vLLM import missing module '$missing_module'; installing '$package_name' (attempt $attempt)"
        if ! benchmark_install_vllm_dependency "$python_bin" "$package_name" "$log_file"; then
            benchmark_error "Failed to install '$package_name' while repairing vLLM import"
            return 1
        fi
    done

    benchmark_error "vLLM import repair exceeded retry limit"
    return 1
}

benchmark_log_vllm_torch_versions() {
    local python_bin="$1"
    "$python_bin" - <<'PY' 2>/dev/null || true
import importlib.metadata as md
import json

meta = {}
for pkg in ("torch", "torchaudio", "torchvision", "vllm", "triton", "triton-rocm"):
    try:
        meta[pkg] = md.version(pkg)
    except Exception:
        meta[pkg] = "missing"
print("package_versions=" + json.dumps(meta, sort_keys=True))
PY
}

benchmark_vllm_native_extensions_ok() {
    local python_bin="$1"
    local output=""
    output="$("$python_bin" - <<'PY' 2>&1
import importlib

errors = []
loaded = []
for ext_name in ("vllm._C", "vllm._rocm_C"):
    try:
        importlib.import_module(ext_name)
        loaded.append(ext_name)
    except Exception as exc:
        errors.append(f"{ext_name} import failed: {exc}")

if loaded:
    print("loaded_native_extensions=" + ",".join(loaded))
if len(errors) == 2:
    print("native_extension_errors=" + " | ".join(errors))
    raise SystemExit(1)
PY
)"
    if [ $? -eq 0 ]; then
        return 0
    fi
    if [ -n "$output" ]; then
        benchmark_warn "vLLM native extension probe failed: $output"
    fi
    return 1
}

benchmark_has_hipsparselt_cmake() {
    local -a candidates=(
        "/opt/rocm/lib/cmake/hipsparselt/hipsparseltConfig.cmake"
        "/opt/rocm/lib/cmake/hipsparselt/hipsparselt-config.cmake"
        "/opt/rocm/hipsparselt/lib/cmake/hipsparselt/hipsparseltConfig.cmake"
        "/opt/rocm/hipsparselt/lib/cmake/hipsparselt/hipsparselt-config.cmake"
    )
    local path
    for path in "${candidates[@]}"; do
        if [ -f "$path" ]; then
            return 0
        fi
    done
    return 1
}

benchmark_repair_vllm_native_extensions() {
    local python_bin="$1"
    local log_file="${2:-/dev/null}"

    benchmark_warn "Attempting automated repair for vLLM native ROCm extensions"
    benchmark_log_vllm_torch_versions "$python_bin"

    benchmark_info "Reinstalling vLLM wheel with --force-reinstall --no-deps"
    if ! benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://wheels.vllm.ai/rocm/ vllm >>"$log_file" 2>&1; then
        benchmark_warn "vLLM wheel reinstall did not complete cleanly; continuing with fallback"
    fi

    if ! benchmark_reconcile_vllm_abi_requirements "$python_bin" "$log_file"; then
        benchmark_error "Failed to reconcile vLLM ABI requirements from metadata"
        return 1
    fi

    if ! benchmark_install_vllm_missing_from_pip_check "$python_bin" "$log_file"; then
        benchmark_warn "pip-check reconciliation reported unresolved entries after vLLM reinstall"
    fi

    if benchmark_vllm_native_extensions_ok "$python_bin"; then
        benchmark_info "vLLM native extensions repaired via wheel reinstall"
        return 0
    fi

    if ! benchmark_has_hipsparselt_cmake; then
        benchmark_warn "Skipping vLLM source rebuild fallback: hipsparselt CMake package not found in ROCm installation"
    else
        benchmark_warn "Wheel reinstall did not resolve ABI issue; attempting source rebuild of vLLM"
        benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir \
            pip setuptools wheel setuptools-scm packaging cmake ninja pybind11 >>"$log_file" 2>&1 || true

        if VLLM_TARGET_DEVICE=rocm VLLM_USE_ROCM=1 USE_ROCM=1 \
            benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall --no-deps \
            --no-build-isolation --no-binary vllm vllm >>"$log_file" 2>&1; then
            if ! benchmark_install_vllm_missing_from_pip_check "$python_bin" "$log_file"; then
                benchmark_warn "pip-check reconciliation reported unresolved entries after source rebuild"
            fi
            if benchmark_vllm_native_extensions_ok "$python_bin"; then
                benchmark_info "vLLM native extensions repaired via source rebuild"
                return 0
            fi
        else
            benchmark_warn "Source rebuild command for vLLM failed"
        fi
    fi

    benchmark_warn "Restoring vLLM wheel/runtime packages after failed native repair fallback"
    benchmark_pip_install_for_python "$python_bin" --upgrade --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://wheels.vllm.ai/rocm/ vllm >>"$log_file" 2>&1 || true
    if ! benchmark_install_vllm_missing_from_pip_check "$python_bin" "$log_file"; then
        benchmark_warn "pip-check reconciliation reported unresolved entries while restoring wheel state"
    fi

    if benchmark_python_module_exists "$python_bin" "vllm"; then
        benchmark_warn "vLLM package is present but native extensions are still not loadable"
    else
        benchmark_error "vLLM package is missing after native repair attempts"
    fi

    benchmark_error "vLLM native extension repair failed; ABI mismatch persists"
    benchmark_log_vllm_torch_versions "$python_bin"
    return 1
}

benchmark_ensure_vllm_runtime_basics() {
    local python_bin="${MLSTACK_BENCHMARK_PYTHON:-${MLSTACK_PYTHON_BIN:-python3}}"
    local missing_modules=()
    local missing_packages=()
    local module=""
    local package=""
    local spec=""
    local log_file="${benchmark_log_file:-/dev/null}"

    if ! benchmark_python_exists "$python_bin"; then
        benchmark_error "Benchmark python not found while validating vLLM runtime basics: $python_bin"
        return 1
    fi

    if ! benchmark_python_module_exists "$python_bin" "vllm"; then
        benchmark_info "vLLM package not present in benchmark interpreter; skipping vLLM runtime dependency checks"
        return 0
    fi

    benchmark_patch_vllm_rocm_amdsmi_handle_probe "$python_bin" "$log_file" || true

    if ! benchmark_verify_torch_runtime_loadable "$python_bin"; then
        benchmark_error "Torch runtime is not loadable for vLLM benchmark interpreter"
        return 1
    fi

    if ! benchmark_reconcile_vllm_abi_requirements "$python_bin" "$log_file"; then
        benchmark_error "Failed to reconcile vLLM ABI requirements from metadata"
        return 1
    fi

    for spec in \
        "cachetools:cachetools" \
        "cbor2:cbor2" \
        "gguf:gguf" \
        "pybase64:pybase64" \
        "ijson:ijson" \
        "lark:lark" \
        "setproctitle:setproctitle" \
        "watchfiles:watchfiles" \
        "amdsmi:amdsmi" \
        "llguidance:llguidance>=1.3.0,<1.4.0" \
        "mistral_common:mistral-common[image]>=1.9.0" \
        "openai_harmony:openai-harmony>=0.0.3" \
        "outlines_core:outlines-core==0.2.11" \
        "xgrammar:xgrammar==0.1.29" \
        "triton_kernels:triton-kernels==1.0.0" \
        "pythonjsonlogger:python-json-logger" \
        "partial_json_parser:partial-json-parser" \
        "lmformatenforcer:lm-format-enforcer" \
        "prometheus_fastapi_instrumentator:prometheus-fastapi-instrumentator"; do
        module="${spec%%:*}"
        package="${spec##*:}"
        if ! benchmark_python_can_import_module "$python_bin" "$module"; then
            missing_modules+=("$module")
            missing_packages+=("$package")
        fi
    done

    if [ "${#missing_modules[@]}" -eq 0 ]; then
        if ! benchmark_is_dry_run; then
            if ! benchmark_install_vllm_missing_from_pip_check "$python_bin" "$log_file"; then
                benchmark_error "Failed to resolve one or more vLLM transitive dependencies from pip check output"
                return 1
            fi
            benchmark_patch_vllm_rocm_amdsmi_handle_probe "$python_bin" "$log_file" || true
            if ! benchmark_repair_vllm_import "$python_bin" "$log_file"; then
                return 1
            fi
            if ! benchmark_vllm_native_extensions_ok "$python_bin"; then
                if ! benchmark_repair_vllm_native_extensions "$python_bin" "$log_file"; then
                    return 1
                fi
            fi
        fi
        benchmark_info "vLLM benchmark runtime dependencies verified for $python_bin"
        return 0
    fi

    benchmark_warn "Missing vLLM runtime modules for benchmark python ($python_bin): ${missing_modules[*]}"
    if benchmark_is_dry_run; then
        benchmark_info "[DRY-RUN] Would run: $python_bin -m pip install --upgrade ${missing_packages[*]}"
        return 0
    fi

    benchmark_info "Installing missing vLLM runtime modules for benchmark python..."
    local idx module_name package_name
    for idx in "${!missing_modules[@]}"; do
        module_name="${missing_modules[$idx]}"
        package_name="${missing_packages[$idx]}"
        benchmark_info "Installing runtime dependency '${package_name}' for missing module '${module_name}'"
        if ! benchmark_install_vllm_dependency "$python_bin" "$package_name" "$log_file"; then
            benchmark_error "Failed to install required runtime package '${package_name}' for module '${module_name}'"
            return 1
        fi
        if ! benchmark_python_can_import_module "$python_bin" "$module_name"; then
            benchmark_error "Failed to import ${module_name} after installation for $python_bin"
            return 1
        fi
    done

    if ! benchmark_install_vllm_missing_from_pip_check "$python_bin" "$log_file"; then
        benchmark_error "Failed to resolve one or more vLLM transitive dependencies from pip check output"
        return 1
    fi

    benchmark_patch_vllm_rocm_amdsmi_handle_probe "$python_bin" "$log_file" || true
    if ! benchmark_repair_vllm_import "$python_bin" "$log_file"; then
        return 1
    fi
    if ! benchmark_vllm_native_extensions_ok "$python_bin"; then
        if ! benchmark_repair_vllm_native_extensions "$python_bin" "$log_file"; then
            return 1
        fi
    fi

    benchmark_success "vLLM runtime modules repaired for benchmark python"
    return 0
}

benchmark_patch_vllm_rocm_amdsmi_handle_probe() {
    local python_bin="$1"
    local log_file="${2:-/dev/null}"

    "$python_bin" - <<'PY' >>"$log_file" 2>&1
import pathlib
import site
import sys

try:
    candidates = []
    for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
        try:
            p = pathlib.Path(base) / "vllm" / "platforms" / "rocm.py"
        except Exception:
            continue
        if p.is_file():
            candidates.append(p)
    if not candidates:
        raise SystemExit(0)
    path = candidates[0]
    text = path.read_text(encoding="utf-8")

    marker = "# mlstack: amdsmi-handle-compat-v2"
    if marker in text:
        raise SystemExit(0)

    new_block = """    handles = amdsmi_get_processor_handles()
    if handles:
        # mlstack: amdsmi-handle-compat-v2
        # Some mixed iGPU+dGPU systems report an unsupported ASIC on handle 0.
        # Probe all handles and use the first valid graphics target.
        for handle in handles:
            try:
                asic_info = amdsmi_get_gpu_asic_info(handle)
            except Exception:
                continue
            target_gfx = (asic_info or {}).get("target_graphics_version", "")
            if target_gfx:
                return target_gfx
        # Fall back to explicit ROCm arch hints exported by the stack.
        arch_hint = (
            os.environ.get("GPU_ARCH")
            or os.environ.get("PYTORCH_ROCM_ARCH")
            or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
            or ""
        )
        if arch_hint:
            if not str(arch_hint).startswith("gfx") and "." in str(arch_hint):
                parts = [p for p in str(arch_hint).split(".") if p]
                if len(parts) >= 2:
                    arch_hint = f"gfx{parts[0]}{parts[1]}"
            return str(arch_hint)
    raise RuntimeError("amdsmi did not return valid GCN arch")
"""
    start = "    handles = amdsmi_get_processor_handles()\\n    if handles:\\n"
    end = "    raise RuntimeError(\\\"amdsmi did not return valid GCN arch\\\")\\n"
    start_idx = text.find(start)
    if start_idx == -1:
        raise SystemExit(0)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        raise SystemExit(0)
    end_idx += len(end)
    current_block = text[start_idx:end_idx]
    if marker in current_block:
        raise SystemExit(0)
    patched = text[:start_idx] + new_block + text[end_idx:]
    path.write_text(patched, encoding="utf-8")
except SystemExit:
    raise
except Exception as exc:
    print(f"[mlstack][WARN] Unable to patch vLLM ROCm amdsmi handle probe: {exc}", file=sys.stderr)
    raise SystemExit(0)
PY
}

benchmark_ensure_deepspeed_runtime_basics() {
    local python_bin="${MLSTACK_BENCHMARK_PYTHON:-${MLSTACK_PYTHON_BIN:-python3}}"
    local missing_modules=()
    local module=""
    local log_file="${benchmark_log_file:-/dev/null}"

    if ! benchmark_python_exists "$python_bin"; then
        benchmark_error "Benchmark python not found while validating DeepSpeed runtime basics: $python_bin"
        return 1
    fi

    if ! benchmark_python_module_exists "$python_bin" "deepspeed"; then
        benchmark_info "DeepSpeed package not present in benchmark interpreter; skipping DeepSpeed runtime dependency checks"
        return 0
    fi

    for module in mpi4py; do
        if ! benchmark_python_can_import_module "$python_bin" "$module"; then
            missing_modules+=("$module")
        fi
    done

    if [ "${#missing_modules[@]}" -eq 0 ]; then
        benchmark_info "DeepSpeed benchmark runtime dependencies verified for $python_bin"
        return 0
    fi

    benchmark_warn "Missing DeepSpeed runtime modules for benchmark python ($python_bin): ${missing_modules[*]}"
    if benchmark_is_dry_run; then
        benchmark_info "[DRY-RUN] Would run: $python_bin -m pip install --upgrade ${missing_modules[*]}"
        return 0
    fi

    benchmark_info "Installing missing DeepSpeed runtime modules for benchmark python..."
    if "$python_bin" -m pip install --upgrade "${missing_modules[@]}" >>"$log_file" 2>&1; then
        for module in "${missing_modules[@]}"; do
            if ! benchmark_python_can_import_module "$python_bin" "$module"; then
                benchmark_error "Failed to import $module after installation for $python_bin"
                return 1
            fi
        done
        benchmark_success "DeepSpeed runtime modules repaired for benchmark python"
        return 0
    fi

    benchmark_error "Failed to install missing DeepSpeed runtime modules (${missing_modules[*]}) for $python_bin"
    return 1
}

benchmark_resolve_log_dir() {
    local preferred="${MLSTACK_LOG_DIR:-$HOME/.rusty-stack/logs}"
    local fallback="${TMPDIR:-/tmp}/rusty-stack/logs"

    if mkdir -p "$preferred" 2>/dev/null && [ -w "$preferred" ]; then
        printf '%s\n' "$preferred"
        return 0
    fi

    mkdir -p "$fallback"
    printf '%s\n' "$fallback"
}

benchmark_resolve_project_root() {
    local from_dir="${1:-}"
    local current

    current="$(cd "$from_dir" && pwd)"
    while [ "$current" != "/" ]; do
        if [ -f "$current/rusty-stack/Cargo.toml" ]; then
            printf '%s\n' "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done

    return 1
}

benchmark_discover_manifest_path() {
    local project_root="$1"
    local manifest="${MLSTACK_BENCHMARK_MANIFEST:-$project_root/rusty-stack/Cargo.toml}"

    if [ ! -f "$manifest" ]; then
        return 1
    fi

    printf '%s\n' "$manifest"
}

benchmark_bootstrap_cargo_env() {
    if command -v cargo >/dev/null 2>&1; then
        return 0
    fi

    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
    fi

    if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.rustup/env" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.rustup/env"
    fi
}

benchmark_require_cargo() {
    benchmark_bootstrap_cargo_env
    if ! command -v cargo >/dev/null 2>&1; then
        benchmark_error "cargo not found in PATH"
        benchmark_error "PATH is: $PATH"
        return 1
    fi
    return 0
}

benchmark_target_dir_from_metadata() {
    local manifest_path="$1"
    local log_file="$2"
    local metadata

    metadata="$(cargo metadata --manifest-path="$manifest_path" --format-version 1 --no-deps 2>>"$log_file" || true)"
    if [ -z "$metadata" ]; then
        return 0
    fi

    local parser_python
    parser_python="$(benchmark_json_parser_python)"
    if benchmark_python_exists "$parser_python"; then
        printf '%s' "$metadata" | "$parser_python" -c 'import json,sys
try:
    print(json.load(sys.stdin).get("target_directory", ""))
except Exception:
    pass
' 2>>"$log_file"
    else
        printf '%s\n' "$metadata" | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' | head -n 1
    fi
}

benchmark_discover_target_dir() {
    local manifest_path="$1"
    local log_file="$2"
    local discovered

    discovered="$(benchmark_target_dir_from_metadata "$manifest_path" "$log_file" | tail -n 1)"

    if [ -n "$discovered" ]; then
        printf '%s\n' "$discovered"
        return 0
    fi

    if [ -n "${CARGO_TARGET_DIR:-}" ]; then
        printf '%s\n' "$CARGO_TARGET_DIR"
        return 0
    fi

    printf '%s/target\n' "$(dirname "$manifest_path")"
}

benchmark_resolve_bench_binary() {
    local manifest_path="$1"
    local log_file="$2"
    local target_dir

    target_dir="$(benchmark_discover_target_dir "$manifest_path" "$log_file")"
    printf '%s/debug/rusty-stack-bench\n' "$target_dir"
}

benchmark_ensure_writable_dir() {
    local dir="$1"
    mkdir -p "$dir" 2>/dev/null || return 1
    [ -w "$dir" ]
}

benchmark_pick_writable_target_dir() {
    local candidate

    if [ -n "${CARGO_TARGET_DIR:-}" ] && benchmark_ensure_writable_dir "$CARGO_TARGET_DIR"; then
        printf '%s\n' "$CARGO_TARGET_DIR"
        return 0
    fi

    candidate="${HOME}/.rusty-stack/cargo-target"
    if benchmark_ensure_writable_dir "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi

    candidate="/tmp/rusty-stack-cargo-target"
    if benchmark_ensure_writable_dir "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi

    return 1
}

benchmark_build_rusty_stack_bench() {
    local manifest_path="$1"
    local log_file="$2"
    local fallback_dir

    if benchmark_is_dry_run; then
        benchmark_info "[DRY-RUN] Would run: cargo build --manifest-path=\"$manifest_path\" --bin rusty-stack-bench"
        return 0
    fi

    if cargo build --manifest-path="$manifest_path" --bin rusty-stack-bench >>"$log_file" 2>&1; then
        return 0
    fi

    if fallback_dir="$(benchmark_pick_writable_target_dir)"; then
        benchmark_warn "Build failed. Retrying with writable CARGO_TARGET_DIR=$fallback_dir"
        export CARGO_TARGET_DIR="$fallback_dir"
        if cargo build --manifest-path="$manifest_path" --bin rusty-stack-bench >>"$log_file" 2>&1; then
            return 0
        fi
    fi

    benchmark_error "Failed to build rusty-stack-bench"
    return 1
}

benchmark_run_named_json() {
    local manifest_path="$1"
    local log_file="$2"
    local benchmark_name="$3"
    local bench_bin

    bench_bin="$(benchmark_resolve_bench_binary "$manifest_path" "$log_file")"

    if benchmark_is_dry_run; then
        if [ -x "$bench_bin" ]; then
            benchmark_info "[DRY-RUN] Would run: $bench_bin $benchmark_name --json"
        else
            benchmark_info "[DRY-RUN] Would run fallback: cargo run --manifest-path=\"$manifest_path\" --bin rusty-stack-bench -- $benchmark_name --json"
        fi
        return 0
    fi

    if [ -x "$bench_bin" ]; then
        benchmark_info "Using benchmark binary: $bench_bin"
        "$bench_bin" "$benchmark_name" --json 2>&1 | tee -a "$log_file"
    else
        benchmark_warn "Binary not found at $bench_bin; using cargo run fallback"
        cargo run --manifest-path="$manifest_path" --bin rusty-stack-bench -- "$benchmark_name" --json 2>&1 | tee -a "$log_file"
    fi
}

benchmark_run_named_json_to_file() {
    local manifest_path="$1"
    local log_file="$2"
    local json_file="$3"
    local benchmark_name="$4"
    local bench_bin

    bench_bin="$(benchmark_resolve_bench_binary "$manifest_path" "$log_file")"

    if benchmark_is_dry_run; then
        if [ -x "$bench_bin" ]; then
            benchmark_info "[DRY-RUN] Would run: $bench_bin $benchmark_name --json"
        else
            benchmark_info "[DRY-RUN] Would run fallback: cargo run --manifest-path=\"$manifest_path\" --bin rusty-stack-bench -- $benchmark_name --json"
        fi
        benchmark_info "[DRY-RUN] Benchmark JSON output path: $json_file"
        return 0
    fi

    if [ -x "$bench_bin" ]; then
        benchmark_info "Using benchmark binary: $bench_bin"
        "$bench_bin" "$benchmark_name" --json 2>&1 | tee -a "$log_file" | tee "$json_file"
    else
        benchmark_warn "Binary not found at $bench_bin; using cargo run fallback"
        cargo run --manifest-path="$manifest_path" --bin rusty-stack-bench -- "$benchmark_name" --json 2>&1 | tee -a "$log_file" | tee "$json_file"
    fi
}
