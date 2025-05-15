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
# ML Stack Extensions Master Installation Script for AMD GPUs
# =============================================================================
# This script installs and configures additional components to enhance the
# ML stack for AMD GPUs, including:
#
# 1. Triton - Compiler for parallel programming
# 2. BITSANDBYTES - Efficient quantization
# 3. vLLM - High-throughput inference engine
# 4. ROCm SMI - Monitoring and profiling
# 5. PyTorch Profiler - Performance analysis
# 6. WandB - Experiment tracking
#
# Author: User
# Date: $(date +"%Y-%m-%d")
# =============================================================================

set -e  # Exit on error

# Create log directory
LOG_DIR="$HOME/Prod/Stan-s-ML-Stack/logs/extensions"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/ml_stack_extensions_install_$(date +"%Y%m%d_%H%M%S").log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" > /dev/null
}

# Start installation
log "=== Starting ML Stack Extensions Installation ==="
log "System: $(uname -a)"
log "ROCm Path: $(which hipcc 2>/dev/null || echo 'Not found')"
log "Python Version: $(python3 --version)"
log "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check if ONNX build is running
if is_process_running "onnxruntime"; then
    log "ONNX Runtime build is currently running. Will not interrupt it."
    ONNX_RUNNING=true
else
    log "ONNX Runtime build is not running."
    ONNX_RUNNING=false
fi

# Check for required dependencies
log "Checking dependencies..."
    # Fix ninja-build detection
    fix_ninja_detection
DEPS=("git" "python3" "pip")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if ! command_exists $dep; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log "Missing dependencies: ${MISSING_DEPS[*]}"
    log "Please install them and run this script again."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/ml_stack"
mkdir -p $INSTALL_DIR

# Function to install a component
install_component() {
    component=$1
    script_path="$HOME/Prod/Stan-s-ML-Stack/scripts/install_${component}.sh"

    if [ -f "$script_path" ]; then
        log "Installing $component..."
        chmod +x "$script_path"
        "$script_path"
        if [ $? -eq 0 ]; then
            log "$component installation completed successfully."
        else
            log "Error: $component installation failed."
        fi
    else
        log "Error: Installation script for $component not found at $script_path"
    fi
}

# Install components
COMPONENTS=("rocm_smi" "pytorch_profiler" "wandb")

# Ask user which components to install
echo "Which components would you like to install?"
echo "1. Triton - Compiler for parallel programming"
echo "2. BITSANDBYTES - Efficient quantization"
echo "3. vLLM - High-throughput inference engine"
echo "4. ROCm SMI - Monitoring and profiling"
echo "5. PyTorch Profiler - Performance analysis"
echo "6. WandB - Experiment tracking"
echo "7. Flash Attention CK - Optimized attention for AMD GPUs"
echo "8. All components"
echo "Enter numbers separated by spaces (e.g., '1 3 5'): "
read -a selections

# Process selections
if [[ " ${selections[@]} " =~ " 8 " ]]; then
    # Install all components
    COMPONENTS=("triton" "bitsandbytes" "vllm" "rocm_smi" "pytorch_profiler" "wandb" "flash_attention_ck")
else
    COMPONENTS=()
    for selection in "${selections[@]}"; do
        case $selection in
            1) COMPONENTS+=("triton") ;;
            2) COMPONENTS+=("bitsandbytes") ;;
            3) COMPONENTS+=("vllm") ;;
            4) COMPONENTS+=("rocm_smi") ;;
            5) COMPONENTS+=("pytorch_profiler") ;;
            6) COMPONENTS+=("wandb") ;;
            7) COMPONENTS+=("flash_attention_ck") ;;
            *) log "Invalid selection: $selection" ;;
        esac
    done
fi

# Install selected components
for component in "${COMPONENTS[@]}"; do
    install_component "$component"
done

log "=== ML Stack Extensions Installation Complete ==="
log "Installation Directory: $INSTALL_DIR"
log "Log File: $LOG_FILE"
log "Documentation: $HOME/Prod/Stan-s-ML-Stack/docs/extensions/"

# Final message
echo "============================================================"
echo "ML Stack Extensions Installation Complete!"
echo "Documentation is available in $HOME/Prod/Stan-s-ML-Stack/docs/extensions/"
echo "Installation logs are available in $LOG_FILE"
echo "============================================================"

# Fix ninja-build detection
fix_ninja_detection() {
    if command -v ninja &>/dev/null && ! command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating symlink for ninja-build..."
        sudo ln -sf $(which ninja) /usr/bin/ninja-build
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build symlink created."
        return 0
    elif command -v ninja-build &>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ninja-build already available."
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing ninja-build..."
        sudo apt-get update && sudo apt-get install -y ninja-build
        return $?
    fi
}
