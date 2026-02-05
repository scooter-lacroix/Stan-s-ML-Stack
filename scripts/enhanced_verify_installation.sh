#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

PYTHON_CMD="${MLSTACK_PYTHON_BIN:-${AITER_VENV_PYTHON:-${UV_PYTHON:-python3}}}"

printf '\n========================================\n'
printf "Stan's ML Stack Verification\n"
printf '========================================\n'
printf 'ROCm Version : %s\n' "${ROCM_VERSION:-unknown}"
printf 'ROCm Channel : %s\n' "${ROCM_CHANNEL:-unknown}"
printf 'GPU Arch     : %s\n' "${GPU_ARCH:-unknown}"
printf '========================================\n'

echo "[1/4] ROCm diagnostics"
rocminfo >/dev/null 2>&1 && echo "  ✓ rocminfo" || echo "  ✗ rocminfo"
rocm-smi >/dev/null 2>&1 && echo "  ✓ rocm-smi" || echo "  ✗ rocm-smi"

echo "[2/4] PyTorch"
$PYTHON_CMD <<'PY'
import torch
print("  PyTorch:", torch.__version__)
print("  ROCm available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  Device:", torch.cuda.get_device_name(0))
PY

echo "[3/4] Components"
for module in flash_attention_amd flash_attn vllm triton onnxruntime migraphx bitsandbytes; do
    if [ "$module" = "flash_attn" ] && $PYTHON_CMD -c "import flash_attention_amd" &>/dev/null; then
        continue # Already checked as flash_attention_amd
    fi
    $PYTHON_CMD -c "
import importlib
import sys
mod = '${module}'
try:
    importlib.import_module(mod)
    print(f'  ✓ {mod}')
except Exception as exc:
    if mod == 'flash_attn' or mod == 'flash_attention_amd':
        # Don't fail if one of the flash attention variants is missing
        print(f'  - {mod}: {exc}')
    else:
        print(f'  ✗ {mod}: {exc}')
        sys.exit(1)
"
done

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
    echo "ROCm info    : $(rocminfo >/dev/null 2>&1 && echo ok || echo missing)"
    echo "rocm-smi     : $(rocm-smi >/dev/null 2>&1 && echo ok || echo missing)"
    echo "PyTorch      : $(PYTHONIOENCODING=utf-8 $PYTHON_CMD - <<'PY'
import torch
print(torch.__version__)
print('rocm' if torch.cuda.is_available() else 'no-rocm')
PY
    )"
    echo "Components:" 
    for module in flash_attention_amd flash_attn vllm triton onnxruntime migraphx bitsandbytes; do
        if $PYTHON_CMD -c "import importlib; import sys; sys.exit(0 if importlib.util.find_spec('${module}') else 1)"; then
            echo "  - ${module}: installed"
        else
            echo "  - ${module}: missing"
        fi
    done
} > "$summary_file"

printf '========================================\nVerification complete\n========================================\n'
printf '%s\n' "Saved summary to: $summary_file"
