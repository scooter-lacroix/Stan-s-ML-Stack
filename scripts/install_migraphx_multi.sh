#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

echo "Installing MIGraphX (ROCm ${ROCM_VERSION:-unknown})"
sudo apt-get update
sudo apt-get install -y migraphx migraphx-dev half

python3 - <<'PY'
try:
    import migraphx
    print("MIGraphX available")
except Exception as exc:
    print("Warning: MIGraphX python binding import failed:", exc)
PY
