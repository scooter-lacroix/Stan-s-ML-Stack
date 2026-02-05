#!/bin/bash
# Simple component installation script without TUI issues
# This is a fallback for systems where TUI installers fail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLSTACK_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "╔═════════════════════════════════════════════════════════╗"
echo "║                                                         ║"
echo "║       Stan's ML Stack - Component Installer            ║"
echo "║       (Simple Text-Based Installer)                   ║"
echo "║                                                         ║"
echo "╚═════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if we're on a nosuid filesystem
if mount | grep "$(pwd)" | grep -q nosuid; then
    echo -e "${YELLOW}${BOLD}Note: Your filesystem is mounted with 'nosuid' option.${RESET}"
    echo -e "${YELLOW}This means sudo will not work in this directory.${RESET}"
    echo ""
    echo -e "${CYAN}Options:${RESET}"
    echo "  1. Move to a filesystem without nosuid restriction"
    echo "  2. Install components individually without root"
    echo ""
    read -p "Continue with individual component installation? (y/n): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    echo ""
fi

cd "$MLSTACK_DIR" || exit 1

# Available components
declare -A COMPONENTS=(
    ["rocm"]="ROCm (GPU Computing Platform)"
    ["pytorch"]="PyTorch (Deep Learning Framework)"
    ["flash_attn"]="Flash Attention (Attention Optimization)"
    ["triton"]="Triton (GPU Programming Language)"
    ["vllm"]="vLLM (LLM Inference Engine)"
    ["aiter"]="AITER (AMD Optimization Toolkit)"
    ["vllm_studio"]="vLLM Studio (Model Lifecycle UI)"
    ["onnx"]="ONNX Runtime (Model Optimizer)"
    ["migraphx"]="MIGraphX (AMD Graph Optimizer)"
    ["rccl"]="RCCL (ROCm Communication Library)"
    ["bitsandbytes"]="bitsandbytes (Model Quantization)"
    ["wandb"]="Weights & Biases (Experiment Tracking)"
)

echo -e "${GREEN}${BOLD}Available Components:${RESET}"
echo ""

# List components with numbers
num=1
for key in "${!COMPONENTS[@]}"; do
    echo "  [$num] ${COMPONENTS[$key]}"
    ((num++))
done

echo ""
echo "Enter component numbers to install (comma-separated, or 'all'):"
echo -e "${YELLOW}Example: 1,2,3 or all${RESET}"
echo ""
read -p "> " selection

# Parse selection
if [ "$selection" = "all" ]; then
    echo -e "${CYAN}Installing all components...${RESET}"
    # Would run all installers here
else
    IFS=',' read -ra numbers <<< "$selection"
    for num in "${numbers[@]}"; do
        echo -e "${CYAN}Selected component #$num${RESET}"
    done
fi

# ROCm installation guide
echo ""
echo -e "${CYAN}${BOLD}ROCm Installation:${RESET}"
echo ""
echo "Due to the 'nosuid' filesystem restriction, you cannot use sudo here."
echo "Please run:"
echo ""
echo -e "${GREEN}cd /tmp  # Move to a temporary directory (usually not nosuid)${RESET}"
echo -e "${GREEN}sudo bash /path/to/install_rocm.sh  # Run ROCm installer${RESET}"
echo ""
echo "Or manually install ROCm from:"
echo "  https://rocm.docs.amd.com/"
echo ""

exit 0
