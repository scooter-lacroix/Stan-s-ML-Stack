#!/bin/bash

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