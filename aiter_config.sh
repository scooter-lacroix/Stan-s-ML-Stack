# AITER Installation Configuration File
# This file contains default settings for AITER installation
# Modify these values to customize your installation

# Installation method: global, venv, auto
INSTALL_METHOD=auto

# Virtual environment directory (used when INSTALL_METHOD=venv)
VENV_DIR=./aiter_rocm_venv

# ROCm settings
HSA_OVERRIDE_GFX_VERSION=11.0.0
PYTORCH_ROCM_ARCH=gfx1100;gfx1101;gfx1102
ROCM_PATH=/opt/rocm

# Build options
BUILD_ISOLATION=true
NO_DEPS=false

# Testing options
RUN_TESTS=true
TEST_TIMEOUT=60

# Logging options
LOG_LEVEL=INFO
LOG_FILE=./aiter_install.log

# Color options
USE_COLORS=true
NO_COLOR=false
