#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# vLLM Runner Script for AMD GPUs
# =============================================================================
# This script sets up the correct environment variables for running vLLM
# with AMD GPUs through ROCm, handling the specific requirements of Ray.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Date: $(date +"%Y-%m-%d")
# =============================================================================

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_STACK_DIR="$(dirname "$SCRIPT_DIR")"

# Create log directory
LOG_DIR="$ML_STACK_DIR/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/vllm_run_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Save original environment variables
log "Saving original environment variables..."
ORIGINAL_ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
ORIGINAL_HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
ORIGINAL_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ORIGINAL_PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE

# Log the original environment variables
log "Original ROCR_VISIBLE_DEVICES: $ORIGINAL_ROCR_VISIBLE_DEVICES"
log "Original HIP_VISIBLE_DEVICES: $ORIGINAL_HIP_VISIBLE_DEVICES"
log "Original CUDA_VISIBLE_DEVICES: $ORIGINAL_CUDA_VISIBLE_DEVICES"
log "Original PYTORCH_ROCM_DEVICE: $ORIGINAL_PYTORCH_ROCM_DEVICE"

# Temporarily adjust environment variables for vLLM/Ray compatibility
log "Adjusting environment variables for vLLM/Ray compatibility..."

# Step 1: Handle ROCR_VISIBLE_DEVICES conflict
# Ray requires HIP_VISIBLE_DEVICES and will error with ROCR_VISIBLE_DEVICES
if [ -n "$ORIGINAL_ROCR_VISIBLE_DEVICES" ]; then
    log "Unsetting ROCR_VISIBLE_DEVICES and setting HIP_VISIBLE_DEVICES to $ORIGINAL_ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES=$ORIGINAL_ROCR_VISIBLE_DEVICES
elif [ -z "$HIP_VISIBLE_DEVICES" ]; then
    log "Setting HIP_VISIBLE_DEVICES to default value (0,1)"
    export HIP_VISIBLE_DEVICES=0,1
fi

# Step 2: Handle inconsistency between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES
# Ray requires consistent values for both variables or one must be unset
if [ -n "$HIP_VISIBLE_DEVICES" ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    if [ "$HIP_VISIBLE_DEVICES" != "$CUDA_VISIBLE_DEVICES" ]; then
        log "WARNING: Inconsistent values detected between HIP_VISIBLE_DEVICES ($HIP_VISIBLE_DEVICES) and CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES)"
        log "Ray requires consistent values for both variables. Temporarily unsetting CUDA_VISIBLE_DEVICES."
        unset CUDA_VISIBLE_DEVICES
    else
        log "HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES have consistent values ($HIP_VISIBLE_DEVICES)"
    fi
elif [ -n "$HIP_VISIBLE_DEVICES" ]; then
    log "Only HIP_VISIBLE_DEVICES is set ($HIP_VISIBLE_DEVICES), which is the preferred configuration for Ray/vLLM"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    log "Only CUDA_VISIBLE_DEVICES is set ($CUDA_VISIBLE_DEVICES), setting HIP_VISIBLE_DEVICES to match"
    export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi

# Set other environment variables for AMD GPUs
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=$(python3 -c "import torch; print(','.join(torch.cuda.get_arch_list()))" 2>/dev/null || echo "gfx90a")
export AMD_LOG_LEVEL=0

# Ensure ROCR_VISIBLE_DEVICES is unset to prevent Ray initialization errors
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    log "WARNING: ROCR_VISIBLE_DEVICES is set, which can cause Ray initialization errors"
    log "Unsetting ROCR_VISIBLE_DEVICES for vLLM"
    unset ROCR_VISIBLE_DEVICES
fi

log "Current environment variables:"
log "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
log "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"
log "PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"

# Check if vLLM is installed
if ! python3 -c "import vllm" &>/dev/null; then
    log "ERROR: vLLM is not installed. Please run the install_vllm.sh script first."
    exit 1
fi

# Run the provided command with the adjusted environment variables
if [ $# -eq 0 ]; then
    log "No command provided. Running a simple vLLM test..."
    python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
    RESULT=$?
else
    log "Running command: $@"
    "$@"
    RESULT=$?
fi

# Restore original environment variables
log "Restoring original environment variables..."
if [ -n "$ORIGINAL_ROCR_VISIBLE_DEVICES" ]; then
    log "Restoring ROCR_VISIBLE_DEVICES to $ORIGINAL_ROCR_VISIBLE_DEVICES"
    export ROCR_VISIBLE_DEVICES=$ORIGINAL_ROCR_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_HIP_VISIBLE_DEVICES" ]; then
    log "Restoring HIP_VISIBLE_DEVICES to $ORIGINAL_HIP_VISIBLE_DEVICES"
    export HIP_VISIBLE_DEVICES=$ORIGINAL_HIP_VISIBLE_DEVICES
elif [ "$ORIGINAL_HIP_VISIBLE_DEVICES" = "" ]; then
    log "Unsetting HIP_VISIBLE_DEVICES as it was originally unset"
    unset HIP_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]; then
    log "Restoring CUDA_VISIBLE_DEVICES to $ORIGINAL_CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES
elif [ "$ORIGINAL_CUDA_VISIBLE_DEVICES" = "" ]; then
    log "Unsetting CUDA_VISIBLE_DEVICES as it was originally unset"
    unset CUDA_VISIBLE_DEVICES
fi

if [ -n "$ORIGINAL_PYTORCH_ROCM_DEVICE" ]; then
    log "Restoring PYTORCH_ROCM_DEVICE to $ORIGINAL_PYTORCH_ROCM_DEVICE"
    export PYTORCH_ROCM_DEVICE=$ORIGINAL_PYTORCH_ROCM_DEVICE
elif [ "$ORIGINAL_PYTORCH_ROCM_DEVICE" = "" ]; then
    log "Unsetting PYTORCH_ROCM_DEVICE as it was originally unset"
    unset PYTORCH_ROCM_DEVICE
fi

log "Command completed with exit code: $RESULT"
exit $RESULT
