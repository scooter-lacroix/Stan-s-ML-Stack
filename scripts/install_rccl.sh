#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/lib/installer_guard.sh" ]]; then
    # shellcheck source=lib/installer_guard.sh
    source "$SCRIPT_DIR/lib/installer_guard.sh"
fi

# RCCL Installation Script
echo "Installing RCCL..."

# Update package list
sudo apt-get update

# Install RCCL packages
sudo apt-get install -y librccl-dev librccl1

# Verify installation
if dpkg -l | grep -q librccl; then
    echo "RCCL installation completed successfully"
else
    echo "RCCL installation failed"
    exit 1
fi
