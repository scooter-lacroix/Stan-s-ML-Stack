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
INSTALL_METHOD="${INSTALL_METHOD:-${MLSTACK_INSTALL_METHOD:-auto}}"
INSTALL_METHOD="$(echo "$INSTALL_METHOD" | tr '[:upper:]' '[:lower:]')"
case "$INSTALL_METHOD" in
    global|venv|auto) ;;
    *)
        echo "⚠ WARNING: Invalid install method '$INSTALL_METHOD'; defaulting to auto"
        INSTALL_METHOD="auto"
        ;;
esac

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

legacy_pip_install() {
    local py_cmd="$1"
    shift
    if declare -f mlstack_pip_install >/dev/null 2>&1; then
        mlstack_pip_install "$py_cmd" "$@"
        return $?
    fi
    if "$py_cmd" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
        "$py_cmd" -m pip install --break-system-packages "$@"
    else
        "$py_cmd" -m pip install "$@"
    fi
}

strict_validate_python_version() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor < 10 or minor > 13:
    raise SystemExit(f"Unsupported Python {major}.{minor}; strict ROCm mode requires Python 3.10-3.13")
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
    local venv_python="${venv_dir}/bin/python"

    if [ "${DRY_RUN:-false}" = "true" ]; then
        printf '%s\n' "$venv_python"
        return 0
    fi

    mkdir -p "$(dirname "$venv_dir")"
    if [[ ! -x "$venv_python" ]]; then
        "$base_python" -m venv "$venv_dir"
    fi
    "$venv_python" -m pip install --upgrade pip setuptools wheel >/dev/null
    printf '%s\n' "$venv_python"
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

strict_shared_lib_exists() {
    local lib_name="$1"
    if command -v ldconfig >/dev/null 2>&1; then
        if ldconfig -p 2>/dev/null | grep -q -- "$lib_name"; then
            return 0
        fi
    fi
    local compat_uid
    compat_uid="$(strict_effective_home_uid)"
    local path
    for path in /usr/lib /usr/lib64 /opt/rocm/lib /opt/rocm/hipsparselt/lib \
        "$HOME/.mlstack/libmpi-compat" "$HOME/.mlstack/libmpi-compat-user-${compat_uid}"; do
        if [ -e "$path/$lib_name" ]; then
            return 0
        fi
    done
    return 1
}

strict_effective_home_uid() {
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

strict_effective_home_gid() {
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

strict_ld_prepend_once() {
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

vllm_prepare_triton_cache_env() {
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
            echo "⚠ WARNING: Triton cache directory is not writable (target: $triton_cache). vLLM runtime may fail during kernel compilation."
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

# Ensure Triton kernel cache is always writable for vLLM runtime/verification.
vllm_prepare_triton_cache_env || true

strict_mpi_cxx_compat_dir_candidates() {
    local compat_uid
    compat_uid="$(strict_effective_home_uid)"
    printf '%s\n' "$HOME/.mlstack/libmpi-compat" "$HOME/.mlstack/libmpi-compat-user-${compat_uid}"
}

strict_mpi_cxx_stub_has_required_symbols() {
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

strict_activate_existing_mpi_cxx_shim() {
    local compat_dir stub_so
    while IFS= read -r compat_dir; do
        [ -n "$compat_dir" ] || continue
        stub_so="$compat_dir/libmpi_cxx.so.40"
        if [ -e "$stub_so" ] && strict_mpi_cxx_stub_has_required_symbols "$stub_so"; then
            strict_ld_prepend_once "$compat_dir" || true
            return 0
        fi
    done < <(strict_mpi_cxx_compat_dir_candidates)
    return 1
}

strict_ensure_libmpi_cxx_compat() {
    local compat_dir stub_so stub_cpp cxx target_uid target_gid
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libmpi_cxx.so.40"; then
        return 0
    fi
    if strict_activate_existing_mpi_cxx_shim; then
        echo "⚠ WARNING: Using existing MPI C++ compatibility shim"
        return 0
    fi
    if ! strict_shared_lib_exists "libmpi.so.40"; then
        echo "✗ ERROR: Missing libmpi_cxx.so.40 and libmpi.so.40; cannot create MPI C++ compatibility shim"
        return 1
    fi
    if command -v g++ >/dev/null 2>&1; then
        cxx="g++"
    elif command -v c++ >/dev/null 2>&1; then
        cxx="c++"
    else
        echo "✗ ERROR: Missing g++/c++; cannot create MPI C++ compatibility shim for libmpi_cxx.so.40"
        return 1
    fi
    target_uid="$(strict_effective_home_uid)"
    target_gid="$(strict_effective_home_gid)"

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
        if ! strict_mpi_cxx_stub_has_required_symbols "$stub_so"; then
            continue
        fi
        if [ "$(id -u 2>/dev/null || echo 0)" -eq 0 ] && [ "$target_uid" != "0" ] && command -v chown >/dev/null 2>&1; then
            chown "$target_uid:$target_gid" "$stub_so" "$stub_cpp" >/dev/null 2>&1 || true
        fi
        strict_ld_prepend_once "$compat_dir" || true
        echo "⚠ WARNING: Using generated MPI C++ compatibility shim at $stub_so"
        return 0
    done < <(strict_mpi_cxx_compat_dir_candidates)

    echo "✗ ERROR: Failed to build a usable MPI C++ compatibility shim"
    return 1
}

strict_report_hipsparselt_missing() {
    echo "✗ ERROR: Missing libhipsparselt.so.0 required by this vLLM/Torch ABI."
    echo "  Install ROCm hipsparselt runtime package (typically: hipsparselt), then retry."
}

strict_try_install_system_hipsparselt() {
    if strict_shared_lib_exists "libhipsparselt.so.0"; then
        return 0
    fi

    strict_sudo_run() {
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
        echo "➤ Attempting non-interactive install of hipsparselt via apt..."
        strict_sudo_run apt-get update -y >/dev/null 2>&1 || true
        if strict_sudo_run apt-get install -y hipsparselt >/dev/null 2>&1; then
            strict_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        echo "⚠ WARNING: Automatic apt install for hipsparselt did not complete"
    fi

    if command -v dnf >/dev/null 2>&1; then
        echo "➤ Attempting non-interactive install of hipsparselt via dnf..."
        if strict_sudo_run dnf install -y hipsparselt >/dev/null 2>&1; then
            strict_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        echo "⚠ WARNING: Automatic dnf install for hipsparselt did not complete"
    fi

    if command -v yum >/dev/null 2>&1; then
        echo "➤ Attempting non-interactive install of hipsparselt via yum..."
        if strict_sudo_run yum install -y hipsparselt >/dev/null 2>&1; then
            strict_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        echo "⚠ WARNING: Automatic yum install for hipsparselt did not complete"
    fi

    if command -v zypper >/dev/null 2>&1; then
        echo "➤ Attempting non-interactive install of hipsparselt via zypper..."
        if strict_sudo_run zypper --non-interactive install -y hipsparselt >/dev/null 2>&1; then
            strict_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        echo "⚠ WARNING: Automatic zypper install for hipsparselt did not complete"
    fi

    if command -v pacman >/dev/null 2>&1; then
        echo "➤ Attempting non-interactive install of hipsparselt via pacman..."
        if strict_sudo_run pacman -S --needed --noconfirm hipsparselt >/dev/null 2>&1; then
            strict_shared_lib_exists "libhipsparselt.so.0" && return 0
        fi
        echo "⚠ WARNING: Automatic pacman install for hipsparselt did not complete"
    fi

    return 1
}

strict_verify_torch_runtime_loadable() {
    local py_cmd="$1"
    local output
    local pass
    strict_activate_existing_mpi_cxx_shim || true
    for pass in 1 2 3; do
        output="$("$py_cmd" - <<'PY' 2>&1
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
            if strict_ensure_libmpi_cxx_compat; then
                continue
            fi
        fi

        if printf '%s\n' "$output" | grep -q "libhipsparselt.so.0"; then
            if strict_try_install_system_hipsparselt; then
                continue
            fi
            strict_report_hipsparselt_missing
            return 1
        fi

        break
    done

    if printf '%s\n' "$output" | grep -q "libhipsparselt.so.0"; then
        strict_report_hipsparselt_missing
    fi
    echo "✗ ERROR: Torch runtime import failed after ABI install:"
    printf '%s\n' "$output" | tail -n 20
    return 1
}

strict_rocm_torch_contract_ok() {
    local py_cmd="$1"
    strict_verify_no_cuda_contamination "$py_cmd" && strict_verify_torch_runtime_loadable "$py_cmd"
}

strict_ensure_rocm_torch() {
    local py_cmd="$1"
    local rocm_mm
    rocm_mm="$(strict_detect_rocm_mm)"
    local rocm_index
    rocm_index="$(strict_rocm_index_url "$rocm_mm")"

    strict_purge_nvidia_packages "$py_cmd"
    "$py_cmd" -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true

    if ! strict_pip_install "$py_cmd" --no-cache-dir --upgrade \
        --index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
        torch torchvision torchaudio triton; then
        strict_pip_install "$py_cmd" --no-cache-dir --upgrade \
            --index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
            torch torchvision torchaudio
    fi

    "$py_cmd" - <<'PY'
import torch
assert torch.__version__
assert getattr(torch.version, "hip", None), "torch.version.hip missing"
PY
    strict_verify_torch_runtime_loadable "$py_cmd"
}

strict_pip_install() {
    local py_cmd="$1"
    shift
    if [ "${STRICT_INSTALL_CONTEXT:-venv}" = "global" ]; then
        "$py_cmd" -m pip install --break-system-packages "$@"
    else
        "$py_cmd" -m pip install "$@"
    fi
}

strict_vllm_pkg_requires_no_deps() {
    local req="${1:-}"
    local base
    base="$(strict_requirement_base_name "$req")"
    case "$base" in
        xgrammar|triton-kernels|conch-triton-kernels)
            return 0
            ;;
    esac
    return 1
}

strict_vllm_dep_install() {
    local py_cmd="$1"
    local req="${2:-}"
    [ -n "$req" ] || return 0

    if strict_vllm_pkg_requires_no_deps "$req"; then
        strict_pip_install "$py_cmd" --no-cache-dir --no-deps --extra-index-url https://wheels.vllm.ai/rocm/ "$req"
        return $?
    fi

    strict_pip_install "$py_cmd" --no-cache-dir --extra-index-url https://wheels.vllm.ai/rocm/ "$req"
}

strict_is_safe_pure_python_pkg() {
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

strict_requirement_base_name() {
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

strict_requirement_pinned_version() {
    local req="${1:-}"
    printf '%s\n' "$req" | sed -nE 's/^[^=<>!~]+==([^; ,]+).*$/\1/p'
}

strict_requirement_has_specifier() {
    local req="${1:-}"
    [[ "$req" == *"=="* || "$req" == *">="* || "$req" == *"<="* || "$req" == *"~="* || "$req" == *"!="* || "$req" == *"<"* || "$req" == *">"* ]]
}

strict_python_requirement_satisfied() {
    local py_cmd="$1"
    local req_spec="${2:-}"
    local fallback_name="${3:-}"
    "$py_cmd" - "$req_spec" "$fallback_name" <<'PY' >/dev/null 2>&1
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

strict_vllm_abi_requirement_for_name() {
    local py_cmd="$1"
    local target_name="${2:-}"
    "$py_cmd" - "$target_name" <<'PY'
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

strict_install_vllm_abi_requirement() {
    local py_cmd="$1"
    local req_spec="${2:-}"
    local req_name version no_local rocm_mm rocm_index candidate
    local -a pip_dep_flags=()
    local -a triton_candidates
    rocm_mm="$(strict_detect_rocm_mm)"
    rocm_index="$(strict_rocm_index_url "$rocm_mm")"

    req_name="$(strict_requirement_base_name "$req_spec")"
    version="$(strict_requirement_pinned_version "$req_spec")"
    if [[ "$req_name" == "torchvision" || "$req_name" == "torchaudio" || "$req_name" == "triton" ]]; then
        pip_dep_flags=(--no-deps)
    fi

    case "$req_name" in
        triton)
            if [[ -n "$version" ]]; then
                echo "➤ Enforcing vLLM ABI-compatible Triton runtime for ${req_spec}"
                triton_candidates=("triton==${version}")
                if [[ "$version" == *+* ]]; then
                    no_local="${version%%+*}"
                    if [[ -n "$no_local" ]]; then
                        triton_candidates+=("triton==${no_local}")
                    fi
                fi
                triton_candidates+=("triton-rocm==${version}")
                if [[ "$version" == *+* && -n "$no_local" ]]; then
                    triton_candidates+=("triton-rocm==${no_local}")
                fi
                for candidate in "${triton_candidates[@]}"; do
                    if strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                        --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                        "$candidate"; then
                        return 0
                    fi
                done
                echo "✗ ERROR: Unable to install ABI-pinned Triton runtime required by vLLM (${req_spec})"
                return 1
            fi
            for candidate in triton triton-rocm; do
                if strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                    --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                    "$candidate"; then
                    return 0
                fi
            done
            return 1
            ;;
        torch|torchvision|torchaudio)
            if [[ -n "$req_spec" ]] && strict_requirement_has_specifier "$req_spec"; then
                echo "➤ Enforcing vLLM ABI-compatible ${req_name}: ${req_spec}"
                if strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                    --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                    "$req_spec"; then
                    strict_verify_torch_runtime_loadable "$py_cmd" || return 1
                    return 0
                fi
                if [[ -n "$version" && "$version" == *+* ]]; then
                    no_local="${version%%+*}"
                    if [[ -n "$no_local" ]] && strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                        --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple \
                        "${req_name}==${no_local}"; then
                        strict_verify_torch_runtime_loadable "$py_cmd" || return 1
                        return 0
                    fi
                fi
                echo "✗ ERROR: Unable to install ABI-pinned ${req_name} required by vLLM (${req_spec})"
                return 1
            fi
            if strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall "${pip_dep_flags[@]}" \
                --index-url https://wheels.vllm.ai/rocm/ --extra-index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_name"; then
                strict_verify_torch_runtime_loadable "$py_cmd" || return 1
                return 0
            fi
            return 1
            ;;
    esac

    return 1
}

strict_extract_vllm_requirement_from_pip_check_line() {
    local line="${1:-}"
    local req=""

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? requires (.*), which is not installed\.$/\2/p')"
    if [[ -n "$req" ]]; then
        printf '%s\n' "$req"
        return 0
    fi

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? requires (.*), but you have .* which is incompatible\.$/\2/p')"
    if [[ -n "$req" ]]; then
        printf '%s\n' "$req"
        return 0
    fi

    req="$(printf '%s\n' "$line" | sed -nE 's/^vllm( [^ ]+)? has requirement (.*), but you have .*$/\2/p')"
    if [[ -n "$req" ]]; then
        printf '%s\n' "$req"
        return 0
    fi

    return 1
}

strict_is_rocm_torch_pkg() {
    local pkg="${1,,}"
    case "$pkg" in
        torch|torchvision|torchaudio|triton)
            return 0
            ;;
    esac
    return 1
}

strict_has_rocm_triton_runtime() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY' >/dev/null 2>&1
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

strict_rocm_triton_satisfies_requirement() {
    local py_cmd="$1"
    local req_spec="${2:-triton}"
    "$py_cmd" - "$req_spec" <<'PY' >/dev/null 2>&1
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
    # If packaging is unavailable, only accept unpinned triton requirements.
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

strict_collect_vllm_abi_requirements() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
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
        # Fallback parser: best-effort split for "name[extras] <spec>; marker"
        head = raw.split(";", 1)[0].strip()
        name = head.split("[", 1)[0].split(" ", 1)[0].lower().replace("_", "-")
        if name not in targets:
            continue
        normalized = head.replace(" ", "")

    if normalized in seen:
        continue
    seen.add(normalized)
    print(normalized)
PY
}

strict_reconcile_vllm_abi_requirements() {
    local py_cmd="$1"
    local req_spec req_name

    if ! "$py_cmd" -c "import importlib.metadata as md; md.version('vllm')" >/dev/null 2>&1; then
        echo "⚠ WARNING: vLLM not installed; skipping ABI requirement reconciliation"
        return 0
    fi

    while IFS= read -r req_spec; do
        req_spec="${req_spec// /}"
        [ -n "$req_spec" ] || continue
        req_name="$(strict_requirement_base_name "$req_spec")"
        [ -n "$req_name" ] || continue

        if [[ "$req_name" == "triton" ]]; then
            if strict_rocm_triton_satisfies_requirement "$py_cmd" "$req_spec"; then
                echo "➤ vLLM ABI requirement already satisfied by installed ROCm Triton runtime: ${req_spec}"
                continue
            fi
        elif strict_python_requirement_satisfied "$py_cmd" "$req_spec" "$req_name"; then
            echo "➤ vLLM ABI requirement already satisfied: ${req_spec}"
            continue
        fi

        echo "➤ Reconciling vLLM ABI requirement from metadata: ${req_spec}"
        if ! strict_install_vllm_abi_requirement "$py_cmd" "$req_spec"; then
            echo "⚠ WARNING: Failed to reconcile vLLM ABI requirement: ${req_spec}"
            return 1
        fi
    done < <(strict_collect_vllm_abi_requirements "$py_cmd")

    return 0
}

strict_pip_check_line_is_incompatible() {
    local line="${1:-}"
    [[ "$line" == *", but you have "* ]] && return 0
    return 1
}

strict_install_rocm_torch_requirement() {
    local py_cmd="$1"
    local req_spec="$2"
    local rocm_mm rocm_index req_name

    rocm_mm="$(strict_detect_rocm_mm)"
    rocm_index="$(strict_rocm_index_url "$rocm_mm")"
    req_name="$(strict_requirement_base_name "$req_spec")"
    if [[ "$req_name" == "triton" ]]; then
        req_spec="triton-rocm"
        req_name="triton-rocm"
    fi

    echo "➤ Installing ROCm-aligned dependency for vLLM: ${req_spec} (ROCm ${rocm_mm})"
    if strict_pip_install "$py_cmd" --no-cache-dir --index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_spec"; then
        return 0
    fi

    echo "⚠ WARNING: Exact ROCm dependency install failed for '${req_spec}'; retrying '${req_name}'"
    strict_pip_install "$py_cmd" --no-cache-dir --index-url "$rocm_index" --extra-index-url https://pypi.org/simple "$req_name"
}

strict_install_vllm_missing_from_pip_check() {
    local py_cmd="$1"
    local pip_output line req_spec req_name
    local meta_req
    local pass installs_this_pass
    local failed=0

    for pass in $(seq 1 6); do
        pip_output="$("$py_cmd" -m pip check 2>/dev/null || true)"
        if [[ -z "${pip_output}" ]]; then
            return 0
        fi
        installs_this_pass=0

        while IFS= read -r line; do
            req_spec="$(strict_extract_vllm_requirement_from_pip_check_line "$line" || true)"
            if [[ -z "${req_spec}" ]]; then
                continue
            fi

            req_name="$(strict_requirement_base_name "$req_spec")"
            if [[ -z "$req_name" ]]; then
                echo "⚠ WARNING: Could not parse vLLM requirement from pip check line: $line"
                failed=1
                continue
            fi

            if strict_is_rocm_torch_pkg "$req_name" && ! strict_requirement_has_specifier "$req_spec"; then
                meta_req="$(strict_vllm_abi_requirement_for_name "$py_cmd" "$req_name" 2>/dev/null || true)"
                if [[ -n "$meta_req" ]]; then
                    echo "➤ Upgrading pip-check ABI requirement '$req_spec' to metadata pin '$meta_req'"
                    req_spec="$meta_req"
                fi
            fi

            if [[ "$req_name" == "triton" ]]; then
                if strict_rocm_triton_satisfies_requirement "$py_cmd" "$req_spec"; then
                    echo "➤ pip-check requirement already satisfied by installed ROCm Triton runtime: $req_spec"
                    continue
                fi
            elif strict_python_requirement_satisfied "$py_cmd" "$req_spec" "$req_name"; then
                echo "➤ pip-check requirement already satisfied: $req_spec"
                continue
            fi

            if strict_pip_check_line_is_incompatible "$line"; then
                if strict_is_rocm_torch_pkg "$req_name"; then
                    echo "➤ Resolving vLLM ABI requirement from pip check: $req_spec"
                    if ! strict_install_vllm_abi_requirement "$py_cmd" "$req_spec"; then
                        echo "⚠ WARNING: Could not enforce ABI-compatible ROCm dependency required by vLLM: $req_spec"
                        failed=1
                    fi
                    installs_this_pass=$((installs_this_pass + 1))
                    continue
                fi
                if ! strict_is_safe_pure_python_pkg "$req_name"; then
                    echo "⚠ WARNING: Skipping blocked incompatible package to protect ROCm stack: $req_name"
                    failed=1
                    continue
                fi
                echo "➤ Reconciling incompatible vLLM dependency from pip check (pass $pass): $req_spec"
                if ! strict_vllm_dep_install "$py_cmd" "$req_spec"; then
                    echo "⚠ WARNING: Requirement reconciliation failed, retrying by package name: $req_name"
                    if ! strict_vllm_dep_install "$py_cmd" "$req_name"; then
                        echo "⚠ WARNING: Could not reconcile incompatible vLLM dependency: $req_spec"
                        failed=1
                    fi
                fi
                installs_this_pass=$((installs_this_pass + 1))
                continue
            fi

            if [[ "$req_name" == "triton" ]] && strict_rocm_triton_satisfies_requirement "$py_cmd" "$req_spec"; then
                echo "➤ Treating vLLM triton requirement as satisfied by installed ROCm Triton runtime"
                continue
            fi

            if strict_is_rocm_torch_pkg "$req_name"; then
                if ! strict_install_vllm_abi_requirement "$py_cmd" "$req_spec" \
                    && ! strict_install_rocm_torch_requirement "$py_cmd" "$req_spec"; then
                    echo "⚠ WARNING: Could not install ROCm torch dependency required by vLLM: $req_spec"
                    failed=1
                fi
                installs_this_pass=$((installs_this_pass + 1))
                continue
            fi

            if ! strict_is_safe_pure_python_pkg "$req_name"; then
                echo "⚠ WARNING: Skipping blocked package to protect ROCm stack: $req_name"
                failed=1
                continue
            fi

            echo "➤ Installing missing vLLM dependency from pip check (pass $pass): $req_spec"
            if ! strict_vllm_dep_install "$py_cmd" "$req_spec"; then
                echo "⚠ WARNING: Requirement install failed, retrying by package name: $req_name"
                if ! strict_vllm_dep_install "$py_cmd" "$req_name"; then
                    echo "⚠ WARNING: Could not install required vLLM dependency: $req_spec"
                    failed=1
                fi
            fi
            installs_this_pass=$((installs_this_pass + 1))
        done <<< "$pip_output"

        if [ "$installs_this_pass" -eq 0 ]; then
            break
        fi
    done

    return "$failed"
}

strict_install_vllm_deps() {
    local py_cmd="$1"
    local dep missing_module missing_pkg output
    local import_ok=0
    local module_name package_name req
    local -a deps=(
        accelerate aiohttp cloudpickle fastapi msgspec prometheus-client psutil
        py-cpuinfo pyzmq requests sentencepiece tiktoken uvicorn einops
        transformers huggingface-hub cachetools cbor2 gguf pybase64 ijson
        python-json-logger setproctitle watchfiles six openai blake3 lark
        amdsmi
        lm-format-enforcer partial-json-parser prometheus-fastapi-instrumentator
        datasets diskcache timm peft numba
        "openai-harmony>=0.0.3" "mistral-common[image]>=1.9.0"
        "triton-kernels==1.0.0" "outlines-core==0.2.11" "xgrammar==0.1.29"
        "llguidance>=1.3.0,<1.4.0"
    )
    local -a optional_deps=(ray)

    for dep in "${deps[@]}"; do
        if ! strict_vllm_dep_install "$py_cmd" "$dep"; then
            echo "⚠ WARNING: Failed to install dependency: $dep"
        fi
    done

    for dep in "${optional_deps[@]}"; do
        if ! strict_pip_install "$py_cmd" --no-cache-dir "$dep"; then
            echo "⚠ WARNING: Optional dependency unavailable on this interpreter: $dep"
        fi
    done

    if ! strict_reconcile_vllm_abi_requirements "$py_cmd"; then
        echo "✗ ERROR: Could not reconcile vLLM ABI requirements from package metadata"
        return 1
    fi

    for req in \
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
        module_name="${req%%:*}"
        package_name="${req##*:}"
        if ! "$py_cmd" -c "import ${module_name}" >/dev/null 2>&1; then
            echo "⚠ WARNING: Required module '${module_name}' missing; installing '${package_name}'"
            strict_vllm_dep_install "$py_cmd" "$package_name" || return 1
        fi
    done

    # Retry import and auto-install missing modules repeatedly.
    for _ in $(seq 1 20); do
        output="$("$py_cmd" - <<'PY' 2>&1
try:
    import vllm
except Exception as exc:
    print(exc)
    raise
print("ok")
PY
)"
        if [ $? -eq 0 ]; then
            import_ok=1
            break
        fi
        missing_module="$(printf '%s\n' "$output" | sed -n "s/.*No module named ['\"]\([^'\"]\+\)['\"].*/\1/p" | tail -n1)"
        if [[ -z "$missing_module" ]] || [[ ! "$missing_module" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
            return 1
        fi

        case "$missing_module" in
            cachetools) missing_pkg="cachetools" ;;
            cbor2) missing_pkg="cbor2" ;;
            cloudpickle) missing_pkg="cloudpickle" ;;
            aiohttp) missing_pkg="aiohttp" ;;
            fastapi) missing_pkg="fastapi" ;;
            msgspec) missing_pkg="msgspec" ;;
            prometheus_client) missing_pkg="prometheus-client" ;;
            psutil) missing_pkg="psutil" ;;
            cpuinfo) missing_pkg="py-cpuinfo" ;;
            zmq) missing_pkg="pyzmq" ;;
            requests) missing_pkg="requests" ;;
            sentencepiece) missing_pkg="sentencepiece" ;;
            tiktoken) missing_pkg="tiktoken" ;;
            uvicorn) missing_pkg="uvicorn" ;;
            einops) missing_pkg="einops" ;;
            transformers) missing_pkg="transformers" ;;
            huggingface_hub) missing_pkg="huggingface-hub" ;;
            accelerate) missing_pkg="accelerate" ;;
            blake3) missing_pkg="blake3" ;;
            lark) missing_pkg="lark" ;;
            watchfiles) missing_pkg="watchfiles" ;;
            pythonjsonlogger) missing_pkg="python-json-logger" ;;
            openai) missing_pkg="openai" ;;
            openai_harmony|openai_harmony.*) missing_pkg="openai-harmony>=0.0.3" ;;
            partial_json_parser) missing_pkg="partial-json-parser" ;;
            compressed_tensors) missing_pkg="compressed-tensors" ;;
            depyf) missing_pkg="depyf" ;;
            gguf) missing_pkg="gguf" ;;
            llguidance|llguidance.*) missing_pkg="llguidance>=1.3.0,<1.4.0" ;;
            lmformatenforcer) missing_pkg="lm-format-enforcer" ;;
            amdsmi) missing_pkg="amdsmi" ;;
            numba) missing_pkg="numba" ;;
            outlines) missing_pkg="outlines" ;;
            prometheus_fastapi_instrumentator) missing_pkg="prometheus-fastapi-instrumentator" ;;
            mistral_common|mistral_common.*) missing_pkg="mistral-common[image]>=1.9.0" ;;
            triton_kernels|triton_kernels.*) missing_pkg="triton-kernels==1.0.0" ;;
            xgrammar|xgrammar.*) missing_pkg="xgrammar==0.1.29" ;;
            outlines_core|outlines_core.*) missing_pkg="outlines-core==0.2.11" ;;
            packaging) missing_pkg="packaging" ;;
            typing_extensions) missing_pkg="typing-extensions" ;;
            importlib_metadata) missing_pkg="importlib-metadata" ;;
            attr) missing_pkg="attrs" ;;
            yaml) missing_pkg="pyyaml" ;;
            google.protobuf|google.protobuf.*) missing_pkg="protobuf" ;;
            pkg_resources) missing_pkg="setuptools" ;;
            pybase64) missing_pkg="pybase64" ;;
            ijson) missing_pkg="ijson" ;;
            setproctitle) missing_pkg="setproctitle" ;;
            six) missing_pkg="six" ;;
            *) missing_pkg="$(printf '%s\n' "${missing_module%%.*}" | tr '_' '-')" ;;
        esac

        if ! strict_is_safe_pure_python_pkg "$missing_pkg"; then
            echo "⚠ WARNING: Refusing fallback install for non-safe package candidate: $missing_pkg"
            return 1
        fi

        echo "⚠ WARNING: vLLM import missing module '$missing_module'; attempting safe fallback install: $missing_pkg"
        strict_vllm_dep_install "$py_cmd" "$missing_pkg" || return 1
    done

    if ! strict_install_vllm_missing_from_pip_check "$py_cmd"; then
        echo "⚠ WARNING: One or more vLLM dependencies from pip check could not be installed automatically"
    fi

    "$py_cmd" - <<'PY' >/dev/null 2>&1
import vllm
PY
    if [ "$import_ok" -eq 0 ]; then
        return 1
    fi
    return 0
}

vllm_multi_pip_install() {
    local py_cmd="$1"
    local package="$2"
    if "$py_cmd" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
        "$py_cmd" -m pip install --no-cache-dir --break-system-packages "$package"
    else
        "$py_cmd" -m pip install --no-cache-dir "$package"
    fi
}

patch_vllm_rocm_amdsmi_handle_probe() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY' >/dev/null 2>&1
import pathlib
import site

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
path.write_text(text[:start_idx] + new_block + text[end_idx:], encoding="utf-8")
PY
}

ensure_vllm_runtime_basics() {
    local py_cmd="$1"
    local req module_name package_name
    vllm_prepare_triton_cache_env || true
    patch_vllm_rocm_amdsmi_handle_probe "$py_cmd" || true
    for req in \
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
        module_name="${req%%:*}"
        package_name="${req##*:}"
        if ! "$py_cmd" -c "import ${module_name}" >/dev/null 2>&1; then
            echo "⚠ WARNING: Required module '${module_name}' missing; forcing install of '${package_name}'"
            if ! vllm_multi_pip_install "$py_cmd" "$package_name"; then
                echo "✗ Failed to install required module package: ${package_name}"
                return 1
            fi
            if ! "$py_cmd" -c "import ${module_name}" >/dev/null 2>&1; then
                echo "✗ Required runtime module still missing: ${module_name}"
                return 1
            fi
        fi
    done
    patch_vllm_rocm_amdsmi_handle_probe "$py_cmd" || true
    return 0
}

ensure_vllm_runtime_basics_for_benchmark_python() {
    local target_python="${MLSTACK_BENCHMARK_PYTHON:-${MLSTACK_PYTHON_BIN:-${UV_PYTHON:-python3}}}"
    local resolved=""
    if [ -x "$target_python" ]; then
        resolved="$target_python"
    else
        resolved="$(command -v "$target_python" 2>/dev/null || true)"
    fi
    if [ -z "$resolved" ] || [ ! -x "$resolved" ]; then
        echo "⚠ WARNING: Benchmark Python is not executable (${target_python}); skipping benchmark runtime dep check"
        return 0
    fi
    if ! ensure_vllm_runtime_basics "$resolved"; then
        echo "✗ Required vLLM runtime dependencies are missing for benchmark interpreter: ${resolved}"
        return 1
    fi
    return 0
}

strict_verify_vllm_native_extensions() {
    local py_cmd="$1"
    local output=""

    output="$("$py_cmd" - <<'PY' 2>&1
import importlib
import sys

errors = []
loaded = []
for name in ("vllm._C", "vllm._rocm_C"):
    try:
        importlib.import_module(name)
        loaded.append(name)
    except Exception as exc:
        errors.append(f"{name} import failed: {exc}")

if loaded:
    print("loaded_native_extensions=" + ",".join(loaded))

if len(errors) == 2:
    print("native_extension_errors=" + " | ".join(errors))
    raise SystemExit(1)
PY
)"
    if [ $? -eq 0 ]; then
        echo "✓ vLLM native ROCm extensions are loadable (${output})"
        return 0
    fi

    if [ -n "$output" ]; then
        echo "⚠ WARNING: vLLM native extension probe output: $output"
    fi
    return 1
}

strict_log_vllm_torch_versions() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY' 2>/dev/null || true
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

strict_has_hipsparselt_cmake() {
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

strict_repair_vllm_native_extensions() {
    local py_cmd="$1"

    echo "⚠ WARNING: vLLM native ROCm extensions are not loadable; attempting automated repair"
    strict_log_vllm_torch_versions "$py_cmd"

    echo "➤ Reinstalling vLLM wheel with --force-reinstall --no-deps..."
    if ! strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://wheels.vllm.ai/rocm/ vllm; then
        echo "⚠ WARNING: Wheel reinstall did not complete cleanly; proceeding with repair fallback"
    fi

    if ! strict_install_vllm_deps "$py_cmd"; then
        echo "⚠ WARNING: Dependency reconciliation after wheel reinstall failed"
    fi

    if strict_verify_vllm_native_extensions "$py_cmd"; then
        echo "✓ vLLM native ROCm extensions repaired via wheel reinstall"
        return 0
    fi

    if ! strict_has_hipsparselt_cmake; then
        echo "⚠ WARNING: Skipping source rebuild fallback because hipsparselt CMake package is not available in ROCm installation"
    else
        echo "⚠ WARNING: Wheel reinstall did not fix ABI mismatch; attempting source rebuild of vLLM"
        strict_pip_install "$py_cmd" --no-cache-dir --upgrade \
            pip setuptools wheel setuptools-scm packaging cmake ninja pybind11 >/dev/null 2>&1 || true

        if VLLM_TARGET_DEVICE=rocm VLLM_USE_ROCM=1 USE_ROCM=1 \
            strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall --no-deps \
            --no-build-isolation --no-binary vllm vllm; then
            if ! strict_install_vllm_deps "$py_cmd"; then
                echo "⚠ WARNING: Dependency reconciliation after source rebuild reported issues"
            fi
            if strict_verify_vllm_native_extensions "$py_cmd"; then
                echo "✓ vLLM native ROCm extensions repaired via source rebuild"
                return 0
            fi
        else
            echo "⚠ WARNING: Source rebuild install command failed"
        fi
    fi

    echo "⚠ WARNING: Restoring vLLM wheel/runtime dependencies after failed native repair fallback"
    strict_pip_install "$py_cmd" --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://wheels.vllm.ai/rocm/ vllm >/dev/null 2>&1 || true
    if ! strict_install_vllm_deps "$py_cmd"; then
        echo "⚠ WARNING: Dependency reconciliation after wheel restore reported issues"
    fi

    if "$py_cmd" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)" >/dev/null 2>&1; then
        echo "⚠ WARNING: vLLM package is present but native ROCm extensions remain unavailable"
    else
        echo "✗ ERROR: vLLM package is missing after native repair attempts"
    fi

    echo "✗ ERROR: vLLM native ROCm extensions still fail after automated repair"
    strict_log_vllm_torch_versions "$py_cmd"
    return 1
}

strict_prefer_global_python() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import os
import site
import sys

paths = []
for getter in (site.getsitepackages, lambda: [site.getusersitepackages()]):
    try:
        paths.extend(getter())
    except Exception:
        pass

paths = [p for p in paths if p]
if any(os.access(p, os.W_OK) for p in paths):
    sys.exit(0)
sys.exit(1)
PY
}

strict_install_vllm() {
    local base_python="$PYTHON_BIN"
    local strict_python
    local strict_context
    local selected_python=""
    local strict_rocm_mm=""

    if ! command -v "$base_python" >/dev/null 2>&1; then
        echo "✗ ERROR: Python interpreter not found: $base_python"
        return 1
    fi

    strict_rocm_mm="$(strict_detect_rocm_mm)"
    if declare -f mlstack_ensure_python_for_rocm_torch >/dev/null 2>&1; then
        selected_python="$(mlstack_ensure_python_for_rocm_torch "$base_python" "$strict_rocm_mm" "${ROCM_CHANNEL:-latest}" "${DRY_RUN:-false}" || true)"
    elif declare -f mlstack_select_python_for_rocm_torch >/dev/null 2>&1; then
        selected_python="$(mlstack_select_python_for_rocm_torch "$base_python" "$strict_rocm_mm" "${ROCM_CHANNEL:-latest}" || true)"
    fi

    if [ -n "$selected_python" ] && [ "$selected_python" != "$base_python" ]; then
        echo "⚠ WARNING: Switching Python interpreter for ROCm wheel compatibility: ${base_python} -> ${selected_python}"
        base_python="$selected_python"
        PYTHON_BIN="$selected_python"
        MLSTACK_PYTHON_BIN="$selected_python"
        export PYTHON_BIN MLSTACK_PYTHON_BIN
    fi

    if ! strict_validate_python_version "$base_python"; then
        if [ "$DRY_RUN" = "true" ]; then
            echo "⚠ WARNING: Dry run: no compatible Python 3.10-3.13 interpreter is currently available."
            echo "⚠ WARNING: Dry run: real install would attempt to provision Python 3.12 via uv."
            return 0
        fi
        echo "✗ ERROR: Strict ROCm mode requires Python 3.10-3.13"
        return 1
    fi

    case "$INSTALL_METHOD" in
        global)
            strict_context="global"
            strict_python="$base_python"
            ;;
        venv)
            strict_context="venv"
            strict_python="$(strict_venv_python "vllm" "$base_python")" || return 1
            ;;
        auto)
            if strict_prefer_global_python "$base_python"; then
                strict_context="global"
                strict_python="$base_python"
            else
                strict_context="venv"
                strict_python="$(strict_venv_python "vllm" "$base_python")" || return 1
            fi
            ;;
        *)
            strict_context="venv"
            strict_python="$(strict_venv_python "vllm" "$base_python")" || return 1
            ;;
    esac

    STRICT_INSTALL_CONTEXT="$strict_context"
    export STRICT_INSTALL_CONTEXT

    if [ "$DRY_RUN" = "true" ]; then
        if [ "$strict_context" = "global" ]; then
            echo "[DRY-RUN] Would ensure ROCm torch and install vLLM using global Python ($strict_python)"
        else
            local strict_venv_dir="${MLSTACK_VENV_DIR:-$HOME/.mlstack/venvs/vllm}"
            echo "[DRY-RUN] Would ensure ROCm torch and install vLLM in ${strict_venv_dir}"
        fi
        return 0
    fi

    export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
    export ROCM_PATH="$ROCM_HOME"
    export HIP_PATH="$ROCM_HOME"
    export HIP_ROOT_DIR="$ROCM_HOME"
    export PYTORCH_ROCM_ARCH="$GPU_ARCH"
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export VLLM_TARGET_DEVICE=rocm

    echo "➤ Strict ROCm mode enabled (MLSTACK_STRICT_ROCM=${MLSTACK_STRICT_ROCM})"
    echo "➤ Install method: ${INSTALL_METHOD} (resolved to ${strict_context})"
    if [ "$strict_context" = "venv" ]; then
        echo "➤ Using venv: ${strict_python%/bin/python}"
    else
        echo "➤ Using global Python: ${strict_python}"
    fi

    echo "➤ Verifying existing PyTorch ROCm contract..."
    if strict_rocm_torch_contract_ok "$strict_python"; then
        echo "✓ Existing ROCm PyTorch contract is valid; reusing current torch installation"
    else
        echo "⚠ WARNING: ROCm PyTorch contract failed; repairing torch installation..."
        if declare -f mlstack_install_rocm_torch_stack >/dev/null 2>&1; then
            mlstack_install_rocm_torch_stack "$strict_python" "$strict_rocm_mm" "${ROCM_CHANNEL:-latest}" "vllm" || return 1
        else
            echo "⚠ WARNING: Shared ROCm torch resolver unavailable; falling back to strict torch installer"
            strict_ensure_rocm_torch "$strict_python" || return 1
        fi
        if ! strict_rocm_torch_contract_ok "$strict_python"; then
            echo "✗ ERROR: ROCm PyTorch contract verification failed after repair"
            return 1
        fi
        echo "✓ ROCm PyTorch repaired and verified"
    fi
    TORCH_VERSION_BEFORE=$("$strict_python" -c "import torch; print(torch.__version__)")
    echo "✓ ROCm PyTorch detected: $TORCH_VERSION_BEFORE"

    echo "➤ Installing vLLM from official ROCm wheels with --no-deps..."
    if ! strict_pip_install "$strict_python" --no-cache-dir --no-deps \
        vllm --extra-index-url https://wheels.vllm.ai/rocm/; then
        echo "✗ vLLM installation failed"
        return 1
    fi

    echo "➤ Installing vLLM dependencies (excluding torch and xformers)..."
    if ! strict_install_vllm_deps "$strict_python"; then
        echo "✗ vLLM dependency install/import validation failed"
        return 1
    fi
    if ! ensure_vllm_runtime_basics "$strict_python"; then
        echo "✗ Required vLLM runtime dependencies are missing for ${strict_python}"
        return 1
    fi
    export MLSTACK_BENCHMARK_PYTHON="$strict_python"
    if ! ensure_vllm_runtime_basics_for_benchmark_python; then
        return 1
    fi
    if [ "$strict_python" != "$PYTHON_BIN" ]; then
        if ! ensure_vllm_runtime_basics "$PYTHON_BIN"; then
            echo "✗ Could not reconcile vLLM runtime deps on ${PYTHON_BIN}"
            return 1
        fi
    fi

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
    if ! strict_verify_vllm_native_extensions "$strict_python"; then
        if ! strict_repair_vllm_native_extensions "$strict_python"; then
            return 1
        fi
    fi

    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ Verifying vLLM installation"
    echo "└─────────────────────────────────────────────────────────┘"
    "$strict_python" <<'PY'
import importlib

try:
    import vllm
    print("vLLM version:", vllm.__version__)
    loaded = []
    errors = []
    for ext_name in ("vllm._C", "vllm._rocm_C"):
        try:
            importlib.import_module(ext_name)
            loaded.append(ext_name)
        except Exception as exc:
            errors.append(f"{ext_name} import failed: {exc}")
    if not loaded and errors:
        print("✗ vLLM native extension load failure:", " | ".join(errors))
        raise SystemExit(1)
    print("✓ vLLM imported successfully")
    print("✓ Native extensions:", ", ".join(loaded) if loaded else "not required")
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
    VLLM_DEPS="accelerate aiohttp cloudpickle fastapi msgspec openai openai-harmony>=0.0.3 mistral-common[image]>=1.9.0 llguidance>=1.3.0,<1.4.0 outlines-core==0.2.11 xgrammar==0.1.29 triton-kernels==1.0.0 prometheus-client psutil py-cpuinfo pyzmq requests sentencepiece tiktoken uvicorn cachetools cbor2 gguf pybase64 ijson python-json-logger setproctitle watchfiles six blake3 lark lm-format-enforcer partial-json-parser prometheus-fastapi-instrumentator datasets diskcache timm peft numba"
    for dep in $VLLM_DEPS; do
        $INSTALL_CMD "$dep" --break-system-packages 2>/dev/null || true
    done
    # Ray wheels are not available for all interpreters/platforms; keep it optional.
    $INSTALL_CMD ray --break-system-packages 2>/dev/null || true

    for req in "cachetools:cachetools" "cbor2:cbor2" "gguf:gguf" "pybase64:pybase64" "llguidance:llguidance>=1.3.0,<1.4.0" "mistral_common:mistral-common[image]>=1.9.0" "openai_harmony:openai-harmony>=0.0.3" "outlines_core:outlines-core==0.2.11" "xgrammar:xgrammar==0.1.29" "triton_kernels:triton-kernels==1.0.0"; do
        module_name="${req%%:*}"
        package_name="${req##*:}"
        if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
            echo "⚠ WARNING: Required module '${module_name}' missing after dependency pass; forcing install of '${package_name}'"
            legacy_pip_install "$PYTHON_BIN" "$package_name" || {
                echo "✗ Failed to install required module package: ${package_name}"
                exit 1
            }
            "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1 || {
                echo "✗ Required runtime module still missing: ${module_name}"
                exit 1
            }
        fi
    done
    if ! ensure_vllm_runtime_basics "$PYTHON_BIN"; then
        echo "✗ Required vLLM runtime dependencies are missing for ${PYTHON_BIN}"
        exit 1
    fi
    if ! ensure_vllm_runtime_basics_for_benchmark_python; then
        exit 1
    fi
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
    if ! $PYTHON_BIN <<'PY'
try:
    import vllm
    print("vLLM version:", vllm.__version__)
    print("✓ vLLM imported successfully")
except Exception as e:
    print(f"✗ Failed to import vllm: {e}")
    raise SystemExit(1)
PY
    then
        exit 1
    fi

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
    if vllm_version=$($PYTHON_BIN -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null); then
        echo "✓ vLLM installed (version: ${vllm_version})"
    else
        echo "✗ vLLM import check failed during summary generation"
        exit 1
    fi
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
