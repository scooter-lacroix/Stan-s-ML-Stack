#!/usr/bin/env bash
# Analyze installer scripts and runtime logs for NVIDIA/CUDA dependency contamination.

set -euo pipefail

LOG_FILE="${1:-$HOME/.mlstack/logs/rusty-stack.log}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PATTERN='(nvidia-|pytorch-cuda|torch-cuda|cuda-bindings|cuda-pathfinder|cuda-python|cupy-cuda)'

echo "== NVIDIA dependency risk analyzer =="
echo "Project root: $ROOT_DIR"
echo "Log file: $LOG_FILE"
echo

echo "## 1) Installer script command risk scan"
if grep -RInE "(pip|uv pip) install[^\n]*${PATTERN}" "$ROOT_DIR/scripts" --include='*.sh'; then
    echo
    echo "Found explicit risky install commands above."
else
    echo "No explicit risky install commands detected."
fi
echo

if [ -f "$LOG_FILE" ]; then
    echo "## 2) Runtime log contamination scan"
    if grep -nEi "$PATTERN" "$LOG_FILE" | tail -n 80; then
        echo
        echo "Contamination evidence detected in runtime log."
    else
        echo "No runtime contamination lines detected."
    fi
else
    echo "## 2) Runtime log contamination scan"
    echo "Log file not found: $LOG_FILE"
fi
echo

echo "## 3) Active Python environment contamination check (best effort)"
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import re
import subprocess

try:
    out = subprocess.check_output(
        ["python3", "-m", "pip", "list", "--format=freeze"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    print("Could not run pip list for python3.")
    raise SystemExit(0)

flagged = []
for line in out.splitlines():
    name = line.split("==", 1)[0].lower()
    if (
        name.startswith("nvidia-")
        or name in {"pytorch-cuda", "torch-cuda", "cuda-bindings", "cuda-pathfinder", "cuda-python"}
        or name.startswith("cupy-cuda")
    ):
        flagged.append(line)

if flagged:
    print("Detected NVIDIA/CUDA packages in python3 environment:")
    for item in flagged:
        print("  -", item)
else:
    print("No NVIDIA/CUDA packages detected in python3 environment.")
PY
else
    echo "python3 not found; skipping active environment check."
fi
