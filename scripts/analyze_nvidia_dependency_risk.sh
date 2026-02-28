#!/usr/bin/env bash
# Analyze installer scripts and runtime logs for NVIDIA/CUDA dependency contamination.

set -euo pipefail

LOG_FILE="${1:-$HOME/.mlstack/logs/rusty-stack.log}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRITON_TOKEN='(^|[^[:alnum:]_-])triton([^[:alnum:]_-]|$)'
PATTERN="(nvidia-|pytorch-cuda|torch-cuda|cuda-bindings|cuda-pathfinder|cuda-python|cupy-cuda|${TRITON_TOKEN})"

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

echo "## 2) Guard coverage scan for install commands"
mapfile -t install_scripts < <(
    grep -RIlE "(^|[[:space:]])(uv[[:space:]]+pip[[:space:]]+install|pip[[:space:]]+install|python[0-9.]*[[:space:]]+-m[[:space:]]+pip[[:space:]]+install)" \
        "$ROOT_DIR/scripts" --include='*.sh' || true
)

missing_guard=0
for script in "${install_scripts[@]}"; do
    if [[ "$script" == "$ROOT_DIR/scripts/lib/installer_guard.sh" ]]; then
        continue
    fi

    has_direct_guard_source=0
    has_variable_guard_source=0

    if grep -Eq "(^|[[:space:]])(source|\\.)[[:space:]]+.*installer_guard\\.sh" "$script"; then
        has_direct_guard_source=1
    fi

    if grep -Eq "installer_guard\\.sh" "$script" \
        && grep -Eq "(^|[[:space:]])(source|\\.)[[:space:]]+[\"']?\\$\\{?[A-Za-z_][A-Za-z0-9_]*\\}?[\"']?" "$script"; then
        has_variable_guard_source=1
    fi

    if [ "$has_direct_guard_source" -eq 0 ] && [ "$has_variable_guard_source" -eq 0 ]; then
        echo "Missing installer_guard sourcing: ${script#$ROOT_DIR/}"
        missing_guard=1
    fi
done

if [ "$missing_guard" -eq 0 ]; then
    echo "All install-command scripts appear to source installer_guard.sh."
fi
echo

if [ -f "$LOG_FILE" ]; then
    echo "## 3) Runtime log contamination scan"
    if grep -nEi "$PATTERN" "$LOG_FILE" | tail -n 80; then
        echo
        echo "Contamination evidence detected in runtime log."
    else
        echo "No runtime contamination lines detected."
    fi
else
    echo "## 3) Runtime log contamination scan"
    echo "Log file not found: $LOG_FILE"
fi
echo

echo "## 4) Active Python environment contamination check (best effort)"
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
    version = ""
    if "==" in line:
        version = line.split("==", 1)[1].lower()

    if (
        name.startswith("nvidia-")
        or name in {"pytorch-cuda", "torch-cuda", "cuda-bindings", "cuda-pathfinder", "cuda-python"}
        or name.startswith("cupy-cuda")
        or (name == "triton" and "+rocm" not in version)
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
