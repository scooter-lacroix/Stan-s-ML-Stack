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

    if ! _mlstack_python_in_venv "$python_bin"; then
        if "$python_bin" -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'; then
            maybe_break=(--break-system-packages)
        fi
    fi

    PIP_DISABLE_PIP_VERSION_CHECK=1 "$python_bin" -m pip install "${maybe_break[@]}" "$@"
}

_mlstack_list_nvidia_packages() {
    local python_bin="${1:-}"

    "$python_bin" -m pip list --format=freeze 2>/dev/null | awk -F'==' '
BEGIN { IGNORECASE=1 }
$1 ~ /^nvidia-/ { print $1; next }
$1 ~ /^pytorch-cuda$/ { print $1; next }
$1 ~ /^cuda-python$/ { print $1; next }
$1 ~ /^cupy-cuda/ { print $1; next }
'
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

    "$python_bin" - <<'PY'
import sys
import torch

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

    if mlstack_has_nvidia_packages "$python_bin"; then
        mlstack_log_error "NVIDIA packages remain installed in the target Python environment."
        return 1
    fi
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
        *)
            printf '%s\n' "$pkg"
            ;;
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
            set -gx $key $val
        end
    end
end
EOF

    if [ ! -f "$env_file" ]; then
        mlstack_log_warn "$env_file does not exist yet; shell hooks were still installed."
    fi
}
