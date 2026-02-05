#!/bin/bash
# scorched_earth_cleanup.sh
# Final cleanup of all non-3.12 Python artifacts to prevent pollution.

set -euo pipefail

echo "➤ Aggressively purging Python 3.13 and 3.14 local artifacts..."
rm -rf $HOME/.local/lib/python3.13 || true
rm -rf $HOME/.local/lib/python3.14 || true
rm -rf $HOME/.mlstack/aiter_venv || true
rm -rf $HOME/.mlstack/*_venv || true
rm -rf $HOME/anaconda3 || true
rm -rf $HOME/miniconda3 || true

# Clean up /tmp build artifacts
echo "➤ Cleaning up temporary build artifacts..."
rm -rf /tmp/vllm-rocm || true
rm -rf /tmp/onnxruntime-rocm || true

echo "✓ Local Python pollution cleared."
