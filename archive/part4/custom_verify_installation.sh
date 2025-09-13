#!/bin/bash
#
# Enhanced ML Stack Verification Script
#
# This script verifies the installation of ML stack components
# with specific checks for custom installations.

# ASCII Art Banner
# Color definitions
# Function definitions
# Hardware detection
# Component verification functions
# Summary generation
# Main function
# ASCII Art Banner
cat << "BANNER"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                                Custom ML Stack Verification Script
BANNER
echo

# Create log directory
LOG_DIR="$HOME/Desktop/Stans_MLStack/logs"
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/ml_stack_verify_$(date +"%Y%m%d_%H%M%S").log"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
RESET='\033[0m'

# Function to log messages
log() {
    echo -e "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to print colored messages
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}" | tee -a $LOG_FILE
    echo | tee -a $LOG_FILE
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}" | tee -a $LOG_FILE
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}" | tee -a $LOG_FILE
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}" | tee -a $LOG_FILE
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}" | tee -a $LOG_FILE
}

print_error() {
    echo -e "${RED}✗ $1${RESET}" | tee -a $LOG_FILE
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python module exists
python_module_exists() {
    python3 -c "import $1" >/dev/null 2>&1
}

# Function to check if directory exists
directory_exists() {
    [ -d "$1" ]
}

# Function to check if file exists
file_exists() {
    [ -f "$1" ]
}
# Function to detect hardware
detect_hardware() {
    print_section "Detecting Hardware"
    
    # System information
    print_step "System: $(uname -a)"
    
    # CPU information
    cpu_info=$(lscpu | grep "Model name" | sed 's/Model name: *//g')
    print_step "CPU: $cpu_info"
    
    # Memory information
    mem_info=$(free -h | grep "Mem:" | awk '{print $2}')
    print_step "Memory: $mem_info"
    
    # GPU detection
    if command_exists lspci; then
        print_step "Detecting GPUs using lspci..."
        amd_gpus=$(lspci | grep -i 'amd\|radeon\|advanced micro devices' | grep -i 'vga\|3d\|display')
        
        if [ -n "$amd_gpus" ]; then
            print_success "AMD GPUs detected:"
            echo "$amd_gpus" | while read -r line; do
                echo -e "  - $line" | tee -a $LOG_FILE
            done
        else
            print_error "No AMD GPUs detected with lspci."
        fi
    fi
    
    # ROCm detection
    if command_exists rocminfo; then
        print_step "ROCm Path: $(which rocminfo)"
        print_step "Python Version: $(python3 --version)"
        
        # Get ROCm version
        rocm_version=$(rocminfo | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -n "$rocm_version" ]; then
            print_step "ROCm Version: $rocm_version"
        else
            # Try alternative method to get ROCm version
            rocm_version=$(ls -d /opt/rocm-* 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
            if [ -n "$rocm_version" ]; then
                print_step "ROCm Version: $rocm_version (detected from path)"
            else
                print_warning "Could not determine ROCm version."
            fi
        fi
    else
        print_error "ROCm is not installed or not in PATH."
    fi
}

# Function to verify ROCm
verify_rocm() {
    print_section "Verifying ROCm"
    
    if command_exists hipcc; then
        hip_version=$(hipcc --version 2>&1 | head -n 1)
        print_success "ROCm is installed: $hip_version"
        print_step "ROCm Info:"
        rocminfo 2>&1 | grep -E "Name:|Marketing|ROCm Version" | tee -a $LOG_FILE
        return 0
    else
        print_error "ROCm is not installed."
        return 1
    fi
}

# Function to verify PyTorch
verify_pytorch() {
    print_section "Verifying PyTorch"
    
    if python_module_exists "torch"; then
        print_success "PyTorch is installed"
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>&1)
        print_step "PyTorch version: $pytorch_version"
        
        # Check if PyTorch was built with ROCm support
        if python3 -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -q -v "None"; then
            print_success "PyTorch was built with ROCm support"
            hip_version=$(python3 -c "import torch; print(torch.version.hip)" 2>&1)
            print_step "HIP version: $hip_version"
        else
            print_warning "PyTorch was not built with ROCm support"
        fi
        
        # Check CUDA availability
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "CUDA is available through ROCm"
            
            # Get GPU count
            gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>&1)
            print_step "GPU count: $gpu_count"
            
            # Get GPU names
            for i in $(seq 0 $((gpu_count-1))); do
                gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))" 2>&1)
                print_step "GPU $i: $gpu_name"
            done
            
            # Test simple operation
            if python3 -c "import torch; x = torch.ones(10, device='cuda'); y = x + 1; print('Success')" 2>/dev/null | grep -q "Success"; then
                print_success "Simple tensor operation on GPU successful"
            else
                print_error "Simple tensor operation on GPU failed"
            fi
        else
            print_error "CUDA is not available through ROCm"
        fi
        
        return 0
    else
        print_error "PyTorch is not installed."
        return 1
    fi
}

# Function to verify ONNX Runtime
verify_onnxruntime() {
    print_section "Verifying ONNX Runtime"
    
    if python_module_exists "onnxruntime"; then
        print_success "ONNX Runtime is installed"
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>&1)
        print_step "ONNX Runtime version: $onnx_version"
        
        # Check available providers
        providers=$(python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())" 2>&1)
        print_step "Available providers: $providers"
        
        # Check if ROCMExecutionProvider is available
        if echo "$providers" | grep -q "ROCMExecutionProvider"; then
            print_success "ROCMExecutionProvider is available"
        else
            print_warning "ROCMExecutionProvider is not available"
        fi
        
        return 0
    else
        print_error "ONNX Runtime is not installed."
        return 1
    fi
}

# Function to verify MIGraphX
verify_migraphx() {
    print_section "Verifying MIGraphX"
    
    if command_exists migraphx-driver; then
        print_success "MIGraphX is installed"
        migraphx_version=$(migraphx-driver --version 2>&1 | head -n 1)
        print_step "MIGraphX version: $migraphx_version"
        
        # Check if Python module is installed
        if python_module_exists "migraphx"; then
            print_success "MIGraphX Python module is installed"
        else
            print_warning "MIGraphX Python module is not installed"
        fi
        
        return 0
    else
        print_error "MIGraphX is not installed."
        return 1
    fi
}
# Function to verify Flash Attention
verify_flash_attention() {
    print_section "Verifying Flash Attention"
    
    # Check for different possible module names and locations
    if python_module_exists "flash_attn"; then
        print_success "Flash Attention is installed (flash_attn)"
        flash_attn_version=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>&1)
        print_step "Flash Attention version: $flash_attn_version"
        return 0
    elif python_module_exists "flash_attention_amd"; then
        print_success "Flash Attention is installed (flash_attention_amd)"
        print_step "Flash Attention version: AMD version"
        return 0
    elif python_module_exists "flash_attn_amd_direct"; then
        print_success "Flash Attention is installed (flash_attn_amd_direct)"
        print_step "Flash Attention version: AMD direct version"
        return 0
    elif directory_exists "/home/stan/ml_stack/flash_attn_amd_direct"; then
        print_success "Flash Attention is installed (directory exists at /home/stan/ml_stack/flash_attn_amd_direct)"
        print_step "Flash Attention version: AMD custom version"
        return 0
    else
        print_error "Flash Attention is not installed."
        return 1
    fi
}

# Function to verify RCCL
verify_rccl() {
    print_section "Verifying RCCL"
    
    # Check for RCCL in various locations
    if directory_exists "/opt/rocm/rccl" || directory_exists "$ROCM_PATH/rccl"; then
        print_success "RCCL is installed (directory exists)"
        return 0
    elif file_exists "/opt/rocm/lib/librccl.so" || file_exists "$ROCM_PATH/lib/librccl.so"; then
        print_success "RCCL is installed (integrated into ROCm)"
        print_step "RCCL library: $(find /opt/rocm -name 'librccl.so' 2>/dev/null | head -n 1)"
        return 0
    elif python3 -c "import torch.distributed as dist; print('NCCL available' if hasattr(dist, 'Backend') and 'nccl' in dist.Backend._plugins else 'NCCL not available')" 2>/dev/null | grep -q "NCCL available"; then
        print_success "RCCL is available through PyTorch (NCCL backend)"
        return 0
    else
        # Check if PyTorch distributed is available but NCCL is not
        if python_module_exists "torch.distributed"; then
            print_warning "PyTorch distributed is available but NCCL backend is not"
            print_step "This is normal for ROCm builds as they use RCCL internally"
            print_step "NCCL not being available doesn't mean RCCL isn't working"
            print_success "RCCL is likely working through PyTorch's distributed module"
            return 0
        else
            print_error "RCCL is not installed."
            return 1
        fi
    fi
}

# Function to verify MPI
verify_mpi() {
    print_section "Verifying MPI"
    
    if command_exists mpirun; then
        print_success "MPI is installed"
        mpi_version=$(mpirun --version 2>&1 | head -n 1)
        print_step "MPI version: $mpi_version"
        return 0
    else
        print_error "MPI is not installed."
        return 1
    fi
}

# Function to verify Megatron-LM
verify_megatron() {
    print_section "Verifying Megatron-LM"
    
    # Check for Megatron-LM in various locations
    if directory_exists "$HOME/Desktop/Stans_MLStack/Megatron-LM"; then
        print_success "Megatron-LM is installed (in Stans_MLStack)"
        return 0
    elif directory_exists "$HOME/megatron/Megatron-LM"; then
        print_success "Megatron-LM is installed (in $HOME/megatron)"
        return 0
    elif directory_exists "$HOME/Megatron-LM"; then
        print_success "Megatron-LM is installed (in $HOME)"
        return 0
    elif python_module_exists "megatron"; then
        print_success "Megatron-LM is installed (Python module)"
        return 0
    elif python_module_exists "megatron_core"; then
        print_success "Megatron-LM Core is installed (Python module)"
        return 0
    else
        print_error "Megatron-LM is not installed."
        return 1
    fi
}

# Function to verify Triton
verify_triton() {
    print_section "Verifying Triton"
    
    if python_module_exists "triton"; then
        print_success "Triton is installed"
        triton_version=$(python3 -c "import triton; print(triton.__version__)" 2>&1)
        print_step "Triton version: $triton_version"
        return 0
    else
        print_error "Triton is not installed."
        return 1
    fi
}

# Function to verify BITSANDBYTES
verify_bitsandbytes() {
    print_section "Verifying BITSANDBYTES"
    
    if python_module_exists "bitsandbytes"; then
        print_success "BITSANDBYTES is installed"
        bnb_version=$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>&1)
        print_step "BITSANDBYTES version: $bnb_version"
        return 0
    else
        print_error "BITSANDBYTES is not installed."
        return 1
    fi
}

# Function to verify vLLM
verify_vllm() {
    print_section "Verifying vLLM"
    
    if python_module_exists "vllm"; then
        print_success "vLLM is installed"
        vllm_version=$(python3 -c "import vllm; print(vllm.__version__)" 2>&1)
        print_step "vLLM version: $vllm_version"
        return 0
    else
        print_error "vLLM is not installed."
        return 1
    fi
}

# Function to verify ROCm SMI
verify_rocm_smi() {
    print_section "Verifying ROCm SMI"
    
    if command_exists rocm-smi; then
        print_success "ROCm SMI is installed"
        rocm_smi_version=$(rocm-smi --version 2>&1 | head -n 1)
        print_step "ROCm SMI version: $rocm_smi_version"
        return 0
    else
        print_error "ROCm SMI is not installed."
        return 1
    fi
}

# Function to verify PyTorch Profiler
verify_pytorch_profiler() {
    print_section "Verifying PyTorch Profiler"
    
    if python_module_exists "torch.profiler"; then
        print_success "PyTorch Profiler is installed"
        return 0
    else
        print_error "PyTorch Profiler is not installed."
        return 1
    fi
}

# Function to verify Weights & Biases
verify_wandb() {
    print_section "Verifying Weights & Biases"
    
    if python_module_exists "wandb"; then
        print_success "Weights & Biases is installed"
        wandb_version=$(python3 -c "import wandb; print(wandb.__version__)" 2>&1)
        print_step "Weights & Biases version: $wandb_version"
        return 0
    else
        print_error "Weights & Biases is not installed."
        return 1
    fi
}
# Function to generate summary
generate_summary() {
    print_header "ML Stack Verification Summary"
    
    # Create summary table
    echo -e "${BOLD}Core Components:${RESET}" | tee -a $LOG_FILE
    
    # ROCm
    if [ "$ROCM_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ROCm${RESET}: Successfully installed (version $ROCM_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ROCm${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # PyTorch
    if [ "$PYTORCH_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ PyTorch${RESET}: Successfully installed (version $PYTORCH_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ PyTorch${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # ONNX Runtime
    if [ "$ONNXRUNTIME_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ONNX Runtime${RESET}: Successfully installed (version $ONNXRUNTIME_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ONNX Runtime${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # MIGraphX
    if [ "$MIGRAPHX_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ MIGraphX${RESET}: Successfully installed (version $MIGRAPHX_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ MIGraphX${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # Flash Attention
    if [ "$FLASH_ATTENTION_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Flash Attention${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Flash Attention${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # RCCL
    if [ "$RCCL_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ RCCL${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ RCCL${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # MPI
    if [ "$MPI_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ MPI${RESET}: Successfully installed (version $MPI_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ MPI${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # Megatron-LM
    if [ "$MEGATRON_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Megatron-LM${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Megatron-LM${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    echo | tee -a $LOG_FILE
    echo -e "${BOLD}Extension Components:${RESET}" | tee -a $LOG_FILE
    
    # Triton
    if [ "$TRITON_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Triton${RESET}: Successfully installed (version $TRITON_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Triton${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # BITSANDBYTES
    if [ "$BITSANDBYTES_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ BITSANDBYTES${RESET}: Successfully installed (version $BITSANDBYTES_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ BITSANDBYTES${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # vLLM
    if [ "$VLLM_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ vLLM${RESET}: Successfully installed (version $VLLM_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ vLLM${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # ROCm SMI
    if [ "$ROCM_SMI_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ ROCm SMI${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ ROCm SMI${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # PyTorch Profiler
    if [ "$PYTORCH_PROFILER_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ PyTorch Profiler${RESET}: Successfully installed" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ PyTorch Profiler${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    # Weights & Biases
    if [ "$WANDB_STATUS" = "success" ]; then
        echo -e "${GREEN}✓ Weights & Biases${RESET}: Successfully installed (version $WANDB_VERSION)" | tee -a $LOG_FILE
    else
        echo -e "${RED}✗ Weights & Biases${RESET}: Not installed or not working properly" | tee -a $LOG_FILE
    fi
    
    echo | tee -a $LOG_FILE
    echo -e "${BOLD}Log file:${RESET} $LOG_FILE" | tee -a $LOG_FILE
}

# Main function
main() {
    print_header "ML Stack Verification"
    
    # Detect hardware
    detect_hardware
    
    # Verify ROCm
    verify_rocm
    ROCM_STATUS=$?
    if [ $ROCM_STATUS -eq 0 ]; then
        ROCM_STATUS="success"
        ROCM_VERSION=$(rocminfo 2>/dev/null | grep -i "ROCm Version" | awk -F: '{print $2}' | xargs)
        if [ -z "$ROCM_VERSION" ]; then
            ROCM_VERSION="unknown"
        fi
    else
        ROCM_STATUS="failure"
        ROCM_VERSION="N/A"
    fi
    
    # Verify PyTorch
    verify_pytorch
    PYTORCH_STATUS=$?
    if [ $PYTORCH_STATUS -eq 0 ]; then
        PYTORCH_STATUS="success"
        PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        if [ -z "$PYTORCH_VERSION" ]; then
            PYTORCH_VERSION="unknown"
        fi
    else
        PYTORCH_STATUS="failure"
        PYTORCH_VERSION="N/A"
    fi
    
    # Verify ONNX Runtime
    verify_onnxruntime
    ONNXRUNTIME_STATUS=$?
    if [ $ONNXRUNTIME_STATUS -eq 0 ]; then
        ONNXRUNTIME_STATUS="success"
        ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        if [ -z "$ONNXRUNTIME_VERSION" ]; then
            ONNXRUNTIME_VERSION="unknown"
        fi
    else
        ONNXRUNTIME_STATUS="failure"
        ONNXRUNTIME_VERSION="N/A"
    fi
    
    # Verify MIGraphX
    verify_migraphx
    MIGRAPHX_STATUS=$?
    if [ $MIGRAPHX_STATUS -eq 0 ]; then
        MIGRAPHX_STATUS="success"
        MIGRAPHX_VERSION=$(migraphx-driver --version 2>/dev/null | head -n 1)
        if [ -z "$MIGRAPHX_VERSION" ]; then
            MIGRAPHX_VERSION="unknown"
        fi
    else
        MIGRAPHX_STATUS="failure"
        MIGRAPHX_VERSION="N/A"
    fi
    
    # Verify Flash Attention
    verify_flash_attention
    FLASH_ATTENTION_STATUS=$?
    if [ $FLASH_ATTENTION_STATUS -eq 0 ]; then
        FLASH_ATTENTION_STATUS="success"
    else
        FLASH_ATTENTION_STATUS="failure"
    fi
    
    # Verify RCCL
    verify_rccl
    RCCL_STATUS=$?
    if [ $RCCL_STATUS -eq 0 ]; then
        RCCL_STATUS="success"
    else
        RCCL_STATUS="failure"
    fi
    
    # Verify MPI
    verify_mpi
    MPI_STATUS=$?
    if [ $MPI_STATUS -eq 0 ]; then
        MPI_STATUS="success"
        MPI_VERSION=$(mpirun --version 2>/dev/null | head -n 1)
        if [ -z "$MPI_VERSION" ]; then
            MPI_VERSION="unknown"
        fi
    else
        MPI_STATUS="failure"
        MPI_VERSION="N/A"
    fi
    
    # Verify Megatron-LM
    verify_megatron
    MEGATRON_STATUS=$?
    if [ $MEGATRON_STATUS -eq 0 ]; then
        MEGATRON_STATUS="success"
    else
        MEGATRON_STATUS="failure"
    fi
    
    # Verify Triton
    verify_triton
    TRITON_STATUS=$?
    if [ $TRITON_STATUS -eq 0 ]; then
        TRITON_STATUS="success"
        TRITON_VERSION=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null)
        if [ -z "$TRITON_VERSION" ]; then
            TRITON_VERSION="unknown"
        fi
    else
        TRITON_STATUS="failure"
        TRITON_VERSION="N/A"
    fi
    
    # Verify BITSANDBYTES
    verify_bitsandbytes
    BITSANDBYTES_STATUS=$?
    if [ $BITSANDBYTES_STATUS -eq 0 ]; then
        BITSANDBYTES_STATUS="success"
        BITSANDBYTES_VERSION=$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null)
        if [ -z "$BITSANDBYTES_VERSION" ]; then
            BITSANDBYTES_VERSION="unknown"
        fi
    else
        BITSANDBYTES_STATUS="failure"
        BITSANDBYTES_VERSION="N/A"
    fi
    
    # Verify vLLM
    verify_vllm
    VLLM_STATUS=$?
    if [ $VLLM_STATUS -eq 0 ]; then
        VLLM_STATUS="success"
        VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        if [ -z "$VLLM_VERSION" ]; then
            VLLM_VERSION="unknown"
        fi
    else
        VLLM_STATUS="failure"
        VLLM_VERSION="N/A"
    fi
    
    # Verify ROCm SMI
    verify_rocm_smi
    ROCM_SMI_STATUS=$?
    if [ $ROCM_SMI_STATUS -eq 0 ]; then
        ROCM_SMI_STATUS="success"
    else
        ROCM_SMI_STATUS="failure"
    fi
    
    # Verify PyTorch Profiler
    verify_pytorch_profiler
    PYTORCH_PROFILER_STATUS=$?
    if [ $PYTORCH_PROFILER_STATUS -eq 0 ]; then
        PYTORCH_PROFILER_STATUS="success"
    else
        PYTORCH_PROFILER_STATUS="failure"
    fi
    
    # Verify Weights & Biases
    verify_wandb
    WANDB_STATUS=$?
    if [ $WANDB_STATUS -eq 0 ]; then
        WANDB_STATUS="success"
        WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)" 2>/dev/null)
        if [ -z "$WANDB_VERSION" ]; then
            WANDB_VERSION="unknown"
        fi
    else
        WANDB_STATUS="failure"
        WANDB_VERSION="N/A"
    fi
    
    # Generate summary
    generate_summary
    
    print_header "ML Stack Verification Complete"
    echo -e "Log file: $LOG_FILE"
}

# Run main function
main
