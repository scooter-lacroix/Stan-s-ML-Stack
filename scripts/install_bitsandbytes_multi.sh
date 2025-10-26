#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

echo "Installing bitsandbytes for ROCm"
pip3 install --upgrade bitsandbytes
python3 - <<'PY'
import bitsandbytes as bnb
print("bitsandbytes version:", bnb.__version__)
PY
