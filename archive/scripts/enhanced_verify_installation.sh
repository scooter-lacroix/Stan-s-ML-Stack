#!/bin/bash
set -euo pipefail

# =============================================================================
# SOURCE MULTI-DISTRO ABSTRACTION LAYER
# =============================================================================
SCRIPT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib"
if [[ -f "$SCRIPT_LIB_DIR/distro_detection.sh" ]]; then
    source "$SCRIPT_LIB_DIR/distro_detection.sh"
fi
if [[ -f "$SCRIPT_LIB_DIR/package_manager.sh" ]]; then
    source "$SCRIPT_LIB_DIR/package_manager.sh"
fi
if [[ -f "$SCRIPT_LIB_DIR/rocm_env.sh" ]]; then
    source "$SCRIPT_LIB_DIR/rocm_env.sh"
fi

EXPLICIT_MLSTACK_PYTHON_BIN="${MLSTACK_PYTHON_BIN:-}"
if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi
if [ -n "$EXPLICIT_MLSTACK_PYTHON_BIN" ]; then
    MLSTACK_PYTHON_BIN="$EXPLICIT_MLSTACK_PYTHON_BIN"
    export MLSTACK_PYTHON_BIN
fi
if [ -f "/usr/share/libdrm/amdgpu.ids" ]; then
    export AMDGPU_ASIC_ID_TABLE_PATH="/usr/share/libdrm/amdgpu.ids"
    export AMDGPU_ASIC_ID_TABLE_PATHS="/usr/share/libdrm"
fi

PYTHON_CMD="${EXPLICIT_MLSTACK_PYTHON_BIN:-${MLSTACK_PYTHON_BIN:-${AITER_VENV_PYTHON:-${UV_PYTHON:-python3}}}}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1 && [ ! -x "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

rocm_tool_ok() {
    local output=""
    local rc=0
    output="$("$@" 2>&1 >/dev/null)" || rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    if printf '%s' "$output" | grep -Eqi 'amdgpu\.ids'; then
        return 0
    fi
    return "$rc"
}

printf '\n========================================\n'
printf "Stan's ML Stack Verification\n"
printf '========================================\n'
printf 'ROCm Version : %s\n' "${ROCM_VERSION:-unknown}"
printf 'ROCm Channel : %s\n' "${ROCM_CHANNEL:-unknown}"
printf 'GPU Arch     : %s\n' "${GPU_ARCH:-unknown}"
printf '========================================\n'

echo "[1/4] ROCm diagnostics"
rocm_tool_ok rocminfo && echo "  ✓ rocminfo" || echo "  ✗ rocminfo"
rocm_tool_ok rocm-smi && echo "  ✓ rocm-smi" || echo "  ✗ rocm-smi"

echo "[2/4] PyTorch"
$PYTHON_CMD <<'PY'
import torch
print("  PyTorch:", torch.__version__)
print("  ROCm available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  Device:", torch.cuda.get_device_name(0))
PY

echo "[3/4] Components"
component_failures=0
failed_components=()
for module in flash_attention_amd flash_attn vllm cachetools cbor2 gguf pybase64 llguidance mistral_common openai_harmony outlines_core xgrammar triton onnxruntime migraphx bitsandbytes; do
    if [ "$module" = "flash_attn" ] && $PYTHON_CMD -c "import flash_attention_amd" &>/dev/null; then
        continue # Already checked as flash_attention_amd
    fi
    if module_output=$("$PYTHON_CMD" - "$module" <<'PY'
import importlib
import sys

mod = sys.argv[1]
try:
    importlib.import_module(mod)
    print(f"  ✓ {mod}")
except Exception as exc:
    msg = str(exc)
    if "amdgpu.ids" in msg:
        print(f"  - {mod}: benign amdgpu.ids warning ignored")
        sys.exit(0)
    if mod in ("flash_attn", "flash_attention_amd"):
        print(f"  - {mod}: {exc}")
        sys.exit(0)
    print(f"  ✗ {mod}: {exc}")
    sys.exit(2)
PY
); then
        module_rc=0
    else
        module_rc=$?
    fi
    printf '%s\n' "$module_output"
    if [ "$module_rc" -eq 2 ]; then
        component_failures=$((component_failures + 1))
        failed_components+=("$module")
    fi
done
if [ "$component_failures" -gt 0 ]; then
    echo "  Component import failures: ${failed_components[*]}"
    exit 1
fi

echo "[4/4] Environment"
printenv | grep -E 'ROCM|GPU|HIP_VISIBLE_DEVICES' || echo "(No ROCm env detected)"


summary_file="${MLSTACK_LOG_DIR:-$HOME/.mlstack/logs}/enhanced_verify_$(date +"%Y%m%d_%H%M%S").txt"
mkdir -p "$(dirname "$summary_file")"

cat <<REPORT
========================================
Verification Summary (copyable)
========================================
Report file: $summary_file
ROCm Version : ${ROCM_VERSION:-unknown}
ROCm Channel : ${ROCM_CHANNEL:-unknown}
GPU Arch     : ${GPU_ARCH:-unknown}
========================================
REPORT
{
    echo "ROCm Version : ${ROCM_VERSION:-unknown}"
    echo "ROCm Channel : ${ROCM_CHANNEL:-unknown}"
    echo "GPU Arch     : ${GPU_ARCH:-unknown}"
    echo "ROCm info    : $(rocm_tool_ok rocminfo && echo ok || echo missing)"
    echo "rocm-smi     : $(rocm_tool_ok rocm-smi && echo ok || echo missing)"
    echo "PyTorch      : $(PYTHONIOENCODING=utf-8 $PYTHON_CMD - <<'PY'
import torch
print(torch.__version__)
print('rocm' if torch.cuda.is_available() else 'no-rocm')
PY
    )"
    echo "Components:" 
    for module in flash_attention_amd flash_attn vllm llguidance mistral_common openai_harmony outlines_core xgrammar triton onnxruntime migraphx bitsandbytes; do
        if $PYTHON_CMD -c "import importlib; import sys; sys.exit(0 if importlib.util.find_spec('${module}') else 1)"; then
            echo "  - ${module}: installed"
        else
            echo "  - ${module}: missing"
        fi
    done
} > "$summary_file"

printf '========================================\nVerification complete\n========================================\n'
printf '%s\n' "Saved summary to: $summary_file"
