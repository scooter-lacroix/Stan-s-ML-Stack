#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.mlstack_env" ]; then
    source "$HOME/.mlstack_env"
fi

echo "Installing RCCL packages"
sudo apt-get update
sudo apt-get install -y rccl rccl-dev

rccl-version || true
