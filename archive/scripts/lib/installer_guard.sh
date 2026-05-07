#!/usr/bin/env bash
# Shared installer guard utilities for Stan's ML Stack.

mlstack_log_info() {
    printf '[mlstack][INFO] %s\n' "$*" >&2
}

mlstack_log_warn() {
    printf '[mlstack][WARN] %s\n' "$*" >&2
}

mlstack_log_error() {
    printf '[mlstack][ERROR] %s\n' "$*" >&2
}

mlstack_load_persistent_env_file() {
    local env_file="${MLSTACK_PERSISTENT_ENV_FILE:-$HOME/.mlstack_env}"
    [ -f "$env_file" ] || return 0

    # shellcheck disable=SC1090
    set +u 2>/dev/null || true
    . "$env_file" >/dev/null 2>&1 || true
    set -u 2>/dev/null || true
}

mlstack_enforce_global_install_contract() {
    local method="${MLSTACK_INSTALL_METHOD:-${INSTALL_METHOD:-auto}}"
    method="$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')"
    [ "$method" = "global" ] || return 0

    export MLSTACK_INSTALL_METHOD="global"
    export INSTALL_METHOD="global"

    if [ -n "${MLSTACK_PYTHON_BIN:-}" ]; then
        export UV_PYTHON="${UV_PYTHON:-$MLSTACK_PYTHON_BIN}"
    fi

    export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
    export UV_PIP_BREAK_SYSTEM_PACKAGES="${UV_PIP_BREAK_SYSTEM_PACKAGES:-1}"
    export UV_SYSTEM_PYTHON="${UV_SYSTEM_PYTHON:-1}"
}

mlstack_python_version() {
    local python_bin="${1:-python3}"
    "$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

mlstack_python_supported() {
    local version="${1:-}"
    local major="${version%%.*}"
    local minor="${version#*.}"

    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    [[ "$minor" =~ ^[0-9]+$ ]] || return 1

    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
        return 0
    fi
    return 1
}

mlstack_python_supported_for_rocm_torch() {
    local version="${1:-}"
    local major="${version%%.*}"
    local minor="${version#*.}"

    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    [[ "$minor" =~ ^[0-9]+$ ]] || return 1

    # ROCm wheel availability is currently bounded to Python 3.10-3.13.
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
        return 0
    fi
    return 1
}

mlstack_python_candidates() {
    local preferred="${1:-python3}"
    local cand resolved
    local -a raw=()

    raw+=("$preferred")
    raw+=(python3.12 python3.11 python3.10 python3.13 python3)

    for cand in "${raw[@]}"; do
        [ -n "$cand" ] || continue
        if [ -x "$cand" ]; then
            resolved="$cand"
        else
            resolved="$(command -v "$cand" 2>/dev/null || true)"
        fi
        [ -n "$resolved" ] || continue
        printf '%s\n' "$resolved"
    done | awk '!seen[$0]++'
}

mlstack_select_python_for_rocm_torch() {
    local preferred="${1:-${MLSTACK_PYTHON_BIN:-python3}}"
    local rocm_version="${2:-${ROCM_VERSION:-7.2}}"
    local channel="${3:-${MLSTACK_TORCH_CHANNEL:-latest}}"
    local cand version fallback=""

    while IFS= read -r cand; do
        version="$(mlstack_python_version "$cand" 2>/dev/null || true)"
        [ -n "$version" ] || continue

        if ! mlstack_python_supported_for_rocm_torch "$version"; then
            continue
        fi

        [ -n "$fallback" ] || fallback="$cand"
        if mlstack_select_torch_index "$cand" "$rocm_version" "$channel" >/dev/null 2>&1; then
            printf '%s\n' "$cand"
            return 0
        fi
    done < <(mlstack_python_candidates "$preferred")

    if [ -n "$fallback" ]; then
        mlstack_log_warn "No wheel-index probe succeeded; falling back to compatible Python interpreter: $fallback"
        printf '%s\n' "$fallback"
        return 0
    fi

    mlstack_log_error "No Python 3.10-3.13 interpreter found for ROCm torch installation."
    return 1
}

mlstack_ensure_python_for_rocm_torch() {
    local preferred="${1:-${MLSTACK_PYTHON_BIN:-python3}}"
    local rocm_version="${2:-${ROCM_VERSION:-7.2}}"
    local channel="${3:-${MLSTACK_TORCH_CHANNEL:-latest}}"
    local dry_run="${4:-false}"
    local selected=""
    local uv_python=""
    local uv_cache_dir=""

    selected="$(mlstack_select_python_for_rocm_torch "$preferred" "$rocm_version" "$channel" || true)"
    if [ -n "$selected" ]; then
        printf '%s\n' "$selected"
        return 0
    fi

    if [ "$dry_run" = "true" ]; then
        mlstack_log_warn "Dry run: would provision Python 3.12 for ROCm torch compatibility (via uv)."
        return 1
    fi

    if command -v uv >/dev/null 2>&1; then
        mlstack_log_warn "Provisioning Python 3.12 via uv for ROCm torch compatibility..."
        uv_cache_dir="${MLSTACK_UV_CACHE_DIR:-${TMPDIR:-/tmp}/mlstack-uv-cache}"
        mkdir -p "$uv_cache_dir" >/dev/null 2>&1 || true
        if UV_CACHE_DIR="$uv_cache_dir" uv python install 3.12 >/dev/null 2>&1 \
            || UV_CACHE_DIR="$uv_cache_dir" uv python install 3.12; then
            uv_python="$(UV_CACHE_DIR="$uv_cache_dir" uv python find 3.12 2>/dev/null | head -n1 || true)"
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                selected="$(mlstack_select_python_for_rocm_torch "$uv_python" "$rocm_version" "$channel" || true)"
                if [ -n "$selected" ]; then
                    printf '%s\n' "$selected"
                    return 0
                fi
            fi
        fi
    fi

    mlstack_log_error "Unable to provision a compatible Python interpreter (3.10-3.13) for ROCm torch."
    return 1
}

mlstack_assert_python_supported() {
    local python_bin="${1:-python3}"
    local version

    if ! command -v "$python_bin" >/dev/null 2>&1 && [ ! -x "$python_bin" ]; then
        mlstack_log_error "Python interpreter not found: $python_bin"
        return 1
    fi

    version="$(mlstack_python_version "$python_bin")" || {
        mlstack_log_error "Failed to read Python version from: $python_bin"
        return 1
    }

    if ! mlstack_python_supported "$version"; then
        mlstack_log_error "Unsupported Python $version from $python_bin (supported: 3.10+)"
        return 1
    fi
}

mlstack_default_venv_base() {
    printf '%s\n' "${MLSTACK_VENV_BASE:-$HOME/.mlstack/venvs}"
}

mlstack_prepare_venv() {
    local name="${1:-}"
    local python_bin="${2:-${MLSTACK_PYTHON_BIN:-python3}}"
    local venv_base
    local venv_dir
    local venv_python

    if [ -z "$name" ]; then
        mlstack_log_error "mlstack_prepare_venv requires a venv name"
        return 1
    fi

    mlstack_assert_python_supported "$python_bin"
    venv_base="$(mlstack_default_venv_base)"
    venv_dir="${venv_base}/${name}"
    venv_python="${venv_dir}/bin/python"

    mkdir -p "$venv_base"
    if [ ! -x "$venv_python" ]; then
        mlstack_log_info "Creating virtualenv: $venv_dir"
        "$python_bin" -m venv "$venv_dir"
    fi

    mlstack_assert_python_supported "$venv_python"
    mlstack_pip_install "$venv_python" --upgrade pip setuptools wheel
    printf '%s\n' "$venv_dir"
}

mlstack_venv_python() {
    local name="${1:-}"
    local venv_python

    if [ -z "$name" ]; then
        mlstack_log_error "mlstack_venv_python requires a venv name"
        return 1
    fi

    venv_python="$(mlstack_default_venv_base)/${name}/bin/python"
    if [ ! -x "$venv_python" ]; then
        mlstack_log_error "Virtualenv python not found: $venv_python"
        return 1
    fi
    printf '%s\n' "$venv_python"
}

_mlstack_python_in_venv() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else 1)
PY
}

mlstack_pip_install() {
    local python_bin="${1:-}"
    local -a maybe_break=()

    if [ -z "$python_bin" ] || [ $# -lt 2 ]; then
        mlstack_log_error "Usage: mlstack_pip_install <python_bin> <args...>"
        return 1
    fi
    shift

    if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
        mlstack_log_warn "pip not found for ${python_bin}; attempting bootstrap via ensurepip"
        if ! "$python_bin" -m ensurepip --upgrade >/dev/null 2>&1 &&
           ! "$python_bin" -m ensurepip >/dev/null 2>&1; then
            mlstack_log_error "Unable to bootstrap pip for ${python_bin}"
            return 1
        fi
    fi

    if ! mlstack_guard_install_request "pip" "$@"; then
        return 1
    fi

    if ! _mlstack_python_in_venv "$python_bin"; then
        if "$python_bin" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
            maybe_break=(--break-system-packages)
        fi
    fi

    # Prevent host-level pip hash requirement settings from breaking direct package installs.
    if env -u PIP_REQUIRE_HASHES true >/dev/null 2>&1; then
        env -u PIP_REQUIRE_HASHES -u PIP_REQUIREMENT -u PIP_REQUIREMENTS_FILE \
            PIP_DISABLE_PIP_VERSION_CHECK=1 "$python_bin" -m pip install "${maybe_break[@]}" "$@"
    else
        PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_REQUIRE_HASHES=0 PIP_REQUIREMENT= \
            "$python_bin" -m pip install "${maybe_break[@]}" "$@"
    fi
}

mlstack_normalize_pkg_name() {
    local raw="${1:-}"
    raw="${raw#./}"
    raw="${raw%%[*}"
    raw="${raw%%[<>=!~@; ]*}"
    printf '%s\n' "${raw,,}"
}

mlstack_is_disallowed_pkg_name() {
    local pkg
    pkg="$(mlstack_normalize_pkg_name "${1:-}")"
    [ -n "$pkg" ] || return 1
    case "$pkg" in
        nvidia-*|pytorch-cuda|torch-cuda|cuda-python|cuda-bindings|cuda-pathfinder|cupy-cuda*|cupy-cuda1*|cupy-cuda2*|triton)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

mlstack_guard_install_request() {
    local component="${1:-unknown}"
    shift || true
    local arg pkg
    local -a blocked=()

    for arg in "$@"; do
        [[ "$arg" == -* ]] && continue
        pkg="$(mlstack_normalize_pkg_name "$arg")"
        [ -n "$pkg" ] || continue
        if mlstack_is_disallowed_pkg_name "$pkg"; then
            blocked+=("$pkg")
        fi
    done

    if [ "${#blocked[@]}" -gt 0 ]; then
        mlstack_log_error "Blocked NVIDIA/CUDA package request in ${component}: ${blocked[*]}"
        return 1
    fi
    return 0
}

_mlstack_list_nvidia_packages() {
    local python_bin="${1:-}"

    "$python_bin" -m pip list --format=freeze 2>/dev/null | awk -F'==' '
BEGIN { IGNORECASE=1 }
$1 ~ /^nvidia-/ { print $1; next }
$1 ~ /^pytorch-cuda$/ { print $1; next }
$1 ~ /^torch-cuda$/ { print $1; next }
$1 ~ /^cuda-python$/ { print $1; next }
$1 ~ /^cuda-bindings$/ { print $1; next }
$1 ~ /^cuda-pathfinder$/ { print $1; next }
$1 ~ /^cupy-cuda/ { print $1; next }
'
}

mlstack_collect_nvidia_contamination() {
    local python_bin="${1:-}"
    "$python_bin" - <<'PY'
import re
import subprocess
import sys

blocked = []
try:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        stderr=subprocess.DEVNULL,
        text=True,
    )
except Exception as exc:
    print(f"pip-list-error:{exc}")
    raise SystemExit(0)

for line in output.splitlines():
    name, _, version = line.partition("==")
    n = name.strip().lower()
    v = version.strip().lower()
    if (
        n.startswith("nvidia-")
        or n in {"pytorch-cuda", "torch-cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder"}
        or n.startswith("cupy-cuda")
    ):
        blocked.append(name.strip())
        continue
    if n == "triton" and "+rocm" not in v:
        blocked.append(f"{name.strip()}=={version.strip()}")

try:
    import torch  # noqa: F401
    cuda_tag = getattr(getattr(torch, "version", None), "cuda", None)
    hip_tag = getattr(getattr(torch, "version", None), "hip", None)
    version = getattr(torch, "__version__", "")
    if cuda_tag:
        blocked.append(f"torch.version.cuda={cuda_tag}")
    if not hip_tag and "rocm" not in str(version).lower():
        blocked.append("torch-not-rocm")
except Exception as exc:
    blocked.append(f"torch-import-error:{exc}")

for item in blocked:
    print(item)
PY
}

mlstack_purge_nvidia_packages() {
    local python_bin="${1:-}"
    local -a nvidia_pkgs=()

    if [ -z "$python_bin" ]; then
        mlstack_log_error "Usage: mlstack_purge_nvidia_packages <python_bin>"
        return 1
    fi

    mapfile -t nvidia_pkgs < <(_mlstack_list_nvidia_packages "$python_bin")
    if [ "${#nvidia_pkgs[@]}" -gt 0 ]; then
        mlstack_log_warn "Removing NVIDIA Python packages: ${nvidia_pkgs[*]}"
        "$python_bin" -m pip uninstall -y "${nvidia_pkgs[@]}"
    fi

    # Defensive cleanup for CUDA triton variants when present.
    "$python_bin" -m pip uninstall -y triton >/dev/null 2>&1 || true
}

mlstack_has_nvidia_packages() {
    local python_bin="${1:-}"
    local pkg_list

    if [ -z "$python_bin" ]; then
        mlstack_log_error "Usage: mlstack_has_nvidia_packages <python_bin>"
        return 1
    fi

    pkg_list="$(_mlstack_list_nvidia_packages "$python_bin")"
    [ -n "$pkg_list" ]
}

mlstack_assert_rocm_torch() {
    local python_bin="${1:-}"

    if [ -z "$python_bin" ]; then
        mlstack_log_error "Usage: mlstack_assert_rocm_torch <python_bin>"
        return 1
    fi

    if ! "$python_bin" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Unable to import torch: {exc}")

print("PyTorch version:", torch.__version__)
print("HIP version:", torch.version.hip)
print("CUDA version tag:", torch.version.cuda)
print("CUDA/HIP available:", torch.cuda.is_available())

if not torch.version.hip:
    raise SystemExit("Installed torch is not a ROCm/HIP build.")
if torch.version.cuda:
    raise SystemExit("Installed torch reports a CUDA build, expected ROCm/HIP.")
if not torch.cuda.is_available():
    raise SystemExit("ROCm torch installed but no HIP device is available.")

print("Device:", torch.cuda.get_device_name(0))
PY
    then
        return 1
    fi

    if mlstack_has_nvidia_packages "$python_bin"; then
        mlstack_log_error "NVIDIA packages remain installed in the target Python environment."
        return 1
    fi

    return 0
}

mlstack_guard_python_env() {
    local component="${1:-unknown}"
    local python_bin="${2:-}"
    local purge_mode="${3:-}"
    local contamination

    if [ -z "$python_bin" ]; then
        mlstack_log_error "Usage: mlstack_guard_python_env <component> <python_bin> [--purge]"
        return 1
    fi

    contamination="$(mlstack_collect_nvidia_contamination "$python_bin" | sed '/^[[:space:]]*$/d' || true)"
    if [ -n "$contamination" ]; then
        mlstack_log_error "Detected CUDA/NVIDIA contamination after ${component}:"
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            mlstack_log_error "  - $line"
        done <<< "$contamination"

        if [ "$purge_mode" = "--purge" ]; then
            mlstack_purge_nvidia_packages "$python_bin" || true
            contamination="$(mlstack_collect_nvidia_contamination "$python_bin" | sed '/^[[:space:]]*$/d' || true)"
            if [ -n "$contamination" ]; then
                mlstack_log_error "Contamination remains after purge for ${component}."
                return 1
            fi
        else
            return 1
        fi
    fi
    return 0
}

mlstack_torch_channel_normalize() {
    local channel="${1:-${MLSTACK_TORCH_CHANNEL:-latest}}"
    channel="${channel,,}"
    case "$channel" in
        stable|latest|nightly) printf '%s\n' "$channel" ;;
        *)
            mlstack_log_error "Invalid torch channel '$channel' (expected: stable, latest, nightly)"
            return 1
            ;;
    esac
}

mlstack_rocm_mm_from_version() {
    local rocm_version="${1:-${ROCM_VERSION:-7.1}}"
    local mm
    mm="$(printf '%s' "$rocm_version" | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
    if [ -z "$mm" ]; then
        mm="7.1"
    fi
    printf '%s\n' "$mm"
}

mlstack_rocm_series_candidates() {
    local detected_mm
    detected_mm="$(mlstack_rocm_mm_from_version "${1:-}")"
    {
        printf '%s\n' "$detected_mm"
        printf '%s\n' 7.2 7.1 7.0 6.4 6.3 6.2 6.1 6.0 5.7
    } | awk '!seen[$0]++'
}

mlstack_torch_index_url() {
    local channel="${1:-latest}"
    local series="${2:-7.1}"
    case "$channel" in
        nightly) printf 'https://download.pytorch.org/whl/nightly/rocm%s\n' "$series" ;;
        *) printf 'https://download.pytorch.org/whl/rocm%s\n' "$series" ;;
    esac
}

mlstack_torch_radeon_index_url() {
    local series="${1:-7.1}"
    printf 'https://repo.radeon.com/rocm/manylinux/rocm-rel-%s/\n' "$series"
}

mlstack_torch_index_has_compatible_wheel() {
    local python_bin="${1:-}"
    local index_url="${2:-}"
    local package="${3:-torch}"
    [ -n "$python_bin" ] && [ -n "$index_url" ] || return 1

    "$python_bin" - "$index_url" "$package" <<'PY'
import re
import sys
import urllib.request
import urllib.parse

index, package = sys.argv[1], sys.argv[2]
major, minor = sys.version_info[:2]
cp = f"cp{major}{minor}"
url = index.rstrip("/") + f"/{package}/"
try:
    html = urllib.request.urlopen(url, timeout=15).read().decode("utf-8", "ignore")
except Exception:
    raise SystemExit(1)

wheel_links = re.findall(r'href="([^"]+\.whl[^"]*)"', html)
pattern = re.compile(rf"-{cp}-(?:{cp}|{cp}t)-.*linux", re.IGNORECASE)
ok = any(pattern.search(urllib.parse.unquote(link)) for link in wheel_links)
raise SystemExit(0 if ok else 1)
PY
}

mlstack_select_torch_index() {
    local python_bin="${1:-}"
    local rocm_version="${2:-${ROCM_VERSION:-7.1}}"
    local requested_channel
    local selected
    selected="$(mlstack_torch_index_candidates "$python_bin" "$rocm_version" "${3:-${MLSTACK_TORCH_CHANNEL:-latest}}" | head -n1)"
    if [ -n "$selected" ]; then
        printf '%s\n' "$selected"
        return 0
    fi
    requested_channel="$(mlstack_torch_channel_normalize "${3:-${MLSTACK_TORCH_CHANNEL:-latest}}")" || return 1
    mlstack_log_error "No compatible ROCm torch wheels found for $(mlstack_python_version "$python_bin") and channel '$requested_channel'"
    return 1
}

mlstack_torch_index_candidates() {
    local python_bin="${1:-}"
    local rocm_version="${2:-${ROCM_VERSION:-7.1}}"
    local requested_channel
    local ch series index
    local -a channels=()

    requested_channel="$(mlstack_torch_channel_normalize "${3:-${MLSTACK_TORCH_CHANNEL:-latest}}")" || return 1

    case "$requested_channel" in
        latest) channels=(stable nightly) ;;
        stable|nightly) channels=("$requested_channel") ;;
    esac

    for ch in "${channels[@]}"; do
        for series in $(mlstack_rocm_series_candidates "$rocm_version"); do
            if [ "$ch" != "nightly" ]; then
                index="$(mlstack_torch_radeon_index_url "$series")"
                if mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torch &&
                   mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torchvision &&
                   mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torchaudio; then
                    printf '%s|%s|stable\n' "$index" "$series"
                fi
            fi
            index="$(mlstack_torch_index_url "$ch" "$series")"
            if mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torch &&
               mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torchvision &&
               mlstack_torch_index_has_compatible_wheel "$python_bin" "$index" torchaudio; then
                printf '%s|%s|%s\n' "$index" "$series" "$ch"
            fi
        done
    done | awk '!seen[$0]++'
}

mlstack_torch_index_latest_version() {
    local python_bin="${1:-}"
    local index_url="${2:-}"
    local package="${3:-torch}"
    local channel="${4:-stable}"
    local out version
    local -a cmd=("$python_bin" -m pip index versions "$package" --index-url "$index_url")
    if [ "$channel" = "nightly" ]; then
        cmd=( "$python_bin" -m pip index versions --pre "$package" --index-url "$index_url" )
    fi
    out="$("${cmd[@]}" 2>/dev/null || true)"
    version="$(printf '%s\n' "$out" | awk -F'[()]' -v pkg="$package" '$1 ~ ("^" pkg " "){print $2; exit}')"
    printf '%s\n' "$version"
}

mlstack_install_rocm_torch_stack() {
    local python_bin="${1:-}"
    local rocm_version="${2:-${ROCM_VERSION:-7.1}}"
    local requested_channel="${3:-${MLSTACK_TORCH_CHANNEL:-latest}}"
    local component="${4:-pytorch}"
    local index_url series effective_channel
    local torch_v vision_v audio_v triton_v
    local use_pinned=1
    local tried_any=0

    [ -n "$python_bin" ] || {
        mlstack_log_error "Usage: mlstack_install_rocm_torch_stack <python_bin> [rocm_version] [channel] [component]"
        return 1
    }

    while IFS='|' read -r index_url series effective_channel; do
        tried_any=1
        torch_v="$(mlstack_torch_index_latest_version "$python_bin" "$index_url" torch "$effective_channel")"
        vision_v="$(mlstack_torch_index_latest_version "$python_bin" "$index_url" torchvision "$effective_channel")"
        audio_v="$(mlstack_torch_index_latest_version "$python_bin" "$index_url" torchaudio "$effective_channel")"
        triton_v="$(mlstack_torch_index_latest_version "$python_bin" "$index_url" pytorch-triton-rocm "$effective_channel")"

        use_pinned=1
        if [ -z "$torch_v" ] || [ -z "$vision_v" ] || [ -z "$audio_v" ]; then
            use_pinned=0
            mlstack_log_warn "Unable to resolve exact versions for ${index_url}; falling back to latest compatible wheels from index."
        fi

        mlstack_log_info "Resolved ROCm torch channel=${effective_channel} series=${series} index=${index_url}"
        if [ "$use_pinned" -eq 1 ]; then
            mlstack_log_info "Resolved versions: torch=$torch_v torchvision=$vision_v torchaudio=$audio_v${triton_v:+ pytorch-triton-rocm=$triton_v}"
        fi

        mlstack_purge_nvidia_packages "$python_bin" || true

        if [ "$use_pinned" -eq 1 ]; then
            if ! mlstack_pip_install "$python_bin" --index-url "$index_url" --extra-index-url https://pypi.org/simple \
                --upgrade --force-reinstall "torch==$torch_v" "torchvision==$vision_v" "torchaudio==$audio_v"; then
                mlstack_log_warn "ROCm torch install failed for ${index_url}; trying next compatible candidate."
                continue
            fi
        else
            if ! mlstack_pip_install "$python_bin" --index-url "$index_url" --extra-index-url https://pypi.org/simple \
                --upgrade --force-reinstall torch torchvision torchaudio; then
                mlstack_log_warn "ROCm torch fallback install failed for ${index_url}; trying next compatible candidate."
                continue
            fi
        fi

        if [ -n "$triton_v" ]; then
            mlstack_pip_install "$python_bin" --index-url "$index_url" --extra-index-url https://pypi.org/simple \
                --upgrade "pytorch-triton-rocm==$triton_v" || true
        fi

        if ! mlstack_guard_python_env "$component" "$python_bin" --purge; then
            mlstack_log_warn "Python environment guard failed for ${index_url}; trying next compatible candidate."
            continue
        fi

        if mlstack_assert_rocm_torch "$python_bin"; then
            return 0
        fi

        mlstack_log_warn "ROCm torch validation failed for ${index_url}; trying next compatible candidate."
    done < <(mlstack_torch_index_candidates "$python_bin" "$rocm_version" "$requested_channel")

    if [ "$tried_any" -eq 0 ]; then
        mlstack_log_error "No compatible ROCm torch index candidates available."
    else
        mlstack_log_error "Failed to install ROCm torch stack across all compatible index candidates."
    fi
    return 1
}

mlstack_write_nvidia_constraint_file() {
    local dest="${1:-}"
    local dir
    local stem suffix
    [ -n "$dest" ] || return 1
    dir="$(dirname "$dest")"
    mkdir -p "$dir"
    {
        printf '# Auto-generated NVIDIA/CUDA blocker constraints for ROCm installs\n'
        printf 'pytorch-cuda===0\n'
        printf 'torch-cuda===0\n'
        printf 'cuda-python===0\n'
        printf 'cuda-bindings===0\n'
        printf 'cuda-pathfinder===0\n'
        printf 'cupy-cuda11x===0\n'
        printf 'cupy-cuda12x===0\n'
        for suffix in cu10 cu11 cu12 cu13 cu14; do
            for stem in \
                nvidia-cublas \
                nvidia-cuda-cupti \
                nvidia-cuda-nvrtc \
                nvidia-cuda-runtime \
                nvidia-cudnn \
                nvidia-cufft \
                nvidia-cufile \
                nvidia-curand \
                nvidia-cusolver \
                nvidia-cusparse \
                nvidia-cusparselt \
                nvidia-nccl \
                nvidia-nvjitlink \
                nvidia-nvshmem \
                nvidia-nvtx
            do
                printf '%s-%s===0\n' "$stem" "$suffix"
            done
        done
    } > "$dest"
}

mlstack_activate_nvidia_dependency_blocker() {
    local constraint_file="${MLSTACK_NVIDIA_CONSTRAINT_FILE:-$HOME/.mlstack/pip/no-nvidia-constraints.txt}"
    if ! mlstack_write_nvidia_constraint_file "$constraint_file"; then
        local fallback_file="${TMPDIR:-/tmp}/mlstack-no-nvidia-constraints.txt"
        mlstack_log_warn "Failed to write constraint file at $constraint_file; falling back to $fallback_file"
        mlstack_write_nvidia_constraint_file "$fallback_file" || return 1
        constraint_file="$fallback_file"
    fi
    export PIP_CONSTRAINT="$constraint_file"
    export UV_CONSTRAINT="$constraint_file"
    export MLSTACK_NVIDIA_BLOCKER_ACTIVE=1
}

mlstack_detect_pkg_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        printf 'apt\n'
    elif command -v dnf >/dev/null 2>&1; then
        printf 'dnf\n'
    elif command -v yum >/dev/null 2>&1; then
        printf 'yum\n'
    elif command -v zypper >/dev/null 2>&1; then
        printf 'zypper\n'
    elif command -v pacman >/dev/null 2>&1; then
        printf 'pacman\n'
    elif command -v apk >/dev/null 2>&1; then
        printf 'apk\n'
    else
        mlstack_log_error "No supported package manager found."
        return 1
    fi
}

_mlstack_run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        mlstack_log_error "Root privileges required but sudo is unavailable."
        return 1
    fi
}

mlstack_pm_update() {
    local pm
    pm="$(mlstack_detect_pkg_manager)"

    case "$pm" in
        apt) _mlstack_run_as_root apt-get update ;;
        dnf) _mlstack_run_as_root dnf -y makecache ;;
        yum) _mlstack_run_as_root yum -y makecache ;;
        zypper) _mlstack_run_as_root zypper --gpg-auto-import-keys refresh ;;
        pacman) _mlstack_run_as_root pacman -Sy --noconfirm ;;
        apk) _mlstack_run_as_root apk update ;;
        *) mlstack_log_error "Unsupported package manager: $pm"; return 1 ;;
    esac
}

_mlstack_map_pkg_name() {
    local pm="$1"
    local pkg="$2"

    case "$pkg" in
        libprotobuf-dev)
            case "$pm" in
                apt) printf 'libprotobuf-dev\n' ;;
                dnf|yum|zypper) printf 'protobuf-devel\n' ;;
                pacman) printf 'protobuf\n' ;;
                apk) printf 'protobuf-dev\n' ;;
            esac
            ;;
        protobuf-compiler)
            case "$pm" in
                apt|dnf|yum|zypper) printf 'protobuf-compiler\n' ;;
                pacman|apk) printf 'protobuf\n' ;;
            esac
            ;;
        migraphx-dev)
            case "$pm" in
                apt) printf 'migraphx-dev\n' ;;
                dnf|yum|zypper) printf 'migraphx-devel\n' ;;
                pacman|apk) printf 'migraphx\n' ;;
            esac
            ;;
        python3-migraphx)
            case "$pm" in
                apt|dnf|yum|zypper) printf 'python3-migraphx\n' ;;
                pacman) printf 'python-migraphx\n' ;;
                apk) printf '\n' ;;
            esac
            ;;
        *)
            printf '%s\n' "$pkg"
            ;;
    esac
}

mlstack_pm_has_package() {
    local pm
    local pkg="${1:-}"
    local mapped_pkg

    if [ -z "$pkg" ]; then
        mlstack_log_error "Usage: mlstack_pm_has_package <pkg>"
        return 1
    fi

    pm="$(mlstack_detect_pkg_manager)"
    mapped_pkg="$(_mlstack_map_pkg_name "$pm" "$pkg")"
    [ -n "$mapped_pkg" ] || return 1

    case "$pm" in
        apt) apt-cache show "$mapped_pkg" >/dev/null 2>&1 ;;
        dnf) dnf -q list --available "$mapped_pkg" >/dev/null 2>&1 || dnf -q list "$mapped_pkg" >/dev/null 2>&1 ;;
        yum) yum -q list available "$mapped_pkg" >/dev/null 2>&1 || yum -q list "$mapped_pkg" >/dev/null 2>&1 ;;
        zypper) zypper --quiet search --match-exact "$mapped_pkg" 2>/dev/null | grep -Eq '(^v |^\|)' ;;
        pacman) pacman -Si "$mapped_pkg" >/dev/null 2>&1 ;;
        apk) apk search -x "$mapped_pkg" 2>/dev/null | grep -Fxq "$mapped_pkg" ;;
        *) return 1 ;;
    esac
}

mlstack_pm_install() {
    local pm
    local pkg
    local mapped_pkg
    local -a mapped_pkgs=()

    if [ $# -lt 1 ]; then
        mlstack_log_error "Usage: mlstack_pm_install <pkgs...>"
        return 1
    fi

    pm="$(mlstack_detect_pkg_manager)"
    for pkg in "$@"; do
        mapped_pkg="$(_mlstack_map_pkg_name "$pm" "$pkg")"
        if [ -n "$mapped_pkg" ]; then
            mapped_pkgs+=("$mapped_pkg")
        fi
    done

    case "$pm" in
        apt) _mlstack_run_as_root apt-get install -y "${mapped_pkgs[@]}" ;;
        dnf) _mlstack_run_as_root dnf install -y "${mapped_pkgs[@]}" ;;
        yum) _mlstack_run_as_root yum install -y "${mapped_pkgs[@]}" ;;
        zypper) _mlstack_run_as_root zypper --non-interactive install --no-recommends "${mapped_pkgs[@]}" ;;
        pacman) _mlstack_run_as_root pacman -S --needed --noconfirm "${mapped_pkgs[@]}" ;;
        apk) _mlstack_run_as_root apk add --no-cache "${mapped_pkgs[@]}" ;;
        *) mlstack_log_error "Unsupported package manager: $pm"; return 1 ;;
    esac
}

mlstack_patch_shell_env_for_mlstack() {
    local env_file="$HOME/.mlstack_env"
    local rc_file
    local fish_conf_dir="$HOME/.config/fish/conf.d"
    local fish_env_file="${fish_conf_dir}/mlstack_env.fish"

    for rc_file in "$HOME/.bashrc" "$HOME/.zshrc"; do
        [ -f "$rc_file" ] || touch "$rc_file"
        if ! grep -Fq 'ML Stack environment loader' "$rc_file"; then
            {
                printf '\n# ML Stack environment loader\n'
                printf 'if [ -f "$HOME/.mlstack_env" ]; then\n'
                printf '  . "$HOME/.mlstack_env"\n'
                printf 'fi\n'
            } >> "$rc_file"
        fi
    done

    mkdir -p "$fish_conf_dir"
    cat > "$fish_env_file" <<'EOF'
# ML Stack environment loader
if test -f "$HOME/.mlstack_env"
    for line in (bash -lc 'source "$HOME/.mlstack_env" >/dev/null 2>&1; env')
        set -l key (string split -m1 '=' -- $line)[1]
        set -l val (string split -m1 '=' -- $line)[2]
        if string match -rq '^[A-Za-z_][A-Za-z0-9_]*$' -- $key
            # Skip read-only/shell-managed vars in fish.
            if test "$key" = "PWD"; or test "$key" = "SHLVL"; or test "$key" = "_"; or test "$key" = "pipestatus"; or test "$key" = "status"; or test "$key" = "fish_pid"
                continue
            end
            set -gx $key $val
        end
    end
end
EOF

    if [ ! -f "$env_file" ]; then
        mlstack_log_warn "$env_file does not exist yet; shell hooks were still installed."
    fi
}

mlstack_load_persistent_env_file
mlstack_enforce_global_install_contract

if [ "${MLSTACK_ENABLE_NVIDIA_BLOCKER:-1}" = "1" ] && [ -z "${MLSTACK_NVIDIA_BLOCKER_ACTIVE:-}" ]; then
    mlstack_activate_nvidia_dependency_blocker || mlstack_log_warn "Failed to activate NVIDIA dependency blocker constraints."
fi
