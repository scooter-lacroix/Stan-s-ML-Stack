#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

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
python3 <<'PY'
import torch
print("  PyTorch:", torch.__version__)
print("  ROCm available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  Device:", torch.cuda.get_device_name(0))
PY

echo "[3/4] Components"
for module in flash_attn vllm triton onnxruntime migraphx bitsandbytes; do
    python3 -c "
import importlib
import sys
mod = '${module}'
try:
    importlib.import_module(mod)
    print(f'  ✓ {mod}')
except Exception as exc:
    print(f'  ✗ {mod}: {exc}')
    sys.exit(1)
"
done

echo "[4/4] Environment"
printenv | grep -E 'ROCM|GPU|HIP_VISIBLE_DEVICES' || echo "(No ROCm env detected)"

printf '========================================\nVerification complete\n========================================\n'
