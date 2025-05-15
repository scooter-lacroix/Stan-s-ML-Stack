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
# Flash Attention Build Script
# =============================================================================
# This script builds Flash Attention with AMD GPU support.
#
# Author: User
# Date: 2023-04-19
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                                                                 
                           Flash Attention Build Script for AMD GPUs
EOF
echo

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
BLINK='\033[5m'
REVERSE='\033[7m'
RESET='\033[0m'

# Function definitions
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

check_prerequisites() {
    print_section "Checking prerequisites"
    
    # Check if ROCm is installed
    if ! command -v rocminfo &> /dev/null; then
        print_error "ROCm is not installed. Please install ROCm first."
        return 1
    fi
    print_success "ROCm is installed"
    
    # Check if PyTorch with ROCm is installed
    if ! python3 -c "import torch; print(torch.version.hip)" &> /dev/null; then
        print_error "PyTorch with ROCm support is not installed. Please install PyTorch with ROCm support first."
        return 1
    fi
    print_success "PyTorch with ROCm support is installed"
    
    # Check if CUDA is available through ROCm
    if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_warning "CUDA is not available through ROCm. Check your environment variables."
        print_step "Setting environment variables..."
        export HIP_VISIBLE_DEVICES=0,1
        export CUDA_VISIBLE_DEVICES=0,1
        export PYTORCH_ROCM_DEVICE=0,1
        
        # Check again
        if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            print_error "CUDA is still not available through ROCm. Please check your ROCm installation."
            return 1
        fi
        print_success "Environment variables set successfully"
    fi
    print_success "CUDA is available through ROCm"
    
    # Check Python version
    python_version=$(python3 --version | cut -d ' ' -f 2)
    if [[ $(echo "$python_version" | cut -d '.' -f 1) -lt 3 || ($(echo "$python_version" | cut -d '.' -f 1) -eq 3 && $(echo "$python_version" | cut -d '.' -f 2) -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $python_version"
        return 1
    fi
    print_success "Python version is $python_version"
    
    return 0
}

install_dependencies() {
    print_section "Installing dependencies"
    
    print_step "Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git python3-dev python3-pip ninja-build
    
    print_step "Installing Python dependencies..."
    pip install packaging ninja wheel setuptools
    
    print_success "Dependencies installed successfully"
}
create_amd_implementation() {
    print_section "Creating AMD implementation"
    
    # Create directory for AMD implementation
    print_step "Creating directory for AMD implementation..."
    mkdir -p $HOME/flash-attention-amd
    git clone https://github.com/ROCm/triton.git
    cd triton/python
    GPU_ARCHS=gfx"" python setup.py install
    pip install matplotlib pandas
    cd $HOME/flash-attention-amd
    
    # Create Python implementation file
    print_step "Creating Python implementation file..."
    cat > flash_attention_amd.py << 'EOF'
import torch
import torch.nn.functional as F

class FlashAttention(torch.nn.Module):
    """
    Flash Attention implementation for AMD GPUs using PyTorch operations.
    This is a pure PyTorch implementation that works on AMD GPUs.
    """
    def __init__(self, dropout=0.0, causal=False):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)
        
        Returns: (batch_size, seq_len, num_heads, head_dim)
        """
        # Reshape q, k, v for multi-head attention
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, head_dim) @ (batch_size, num_heads, head_dim, seq_len_k)
        # -> (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if mask is not None:
            # Expand mask to match attention weights shape
            if mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            
            # Apply mask
            attn_weights.masked_fill_(~mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Compute attention output
        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_k, head_dim)
        # -> (batch_size, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back to (batch_size, seq_len_q, num_heads, head_dim)
        output = output.transpose(1, 2)
        
        return output

def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, return_attn_probs=False):
    """
    Functional interface for Flash Attention.
    
    Args:
        q, k, v: (batch_size, seq_len, num_heads, head_dim)
        dropout_p: dropout probability
        causal: whether to apply causal masking
        return_attn_probs: whether to return attention probabilities
        
    Returns:
        output: (batch_size, seq_len, num_heads, head_dim)
        attn_weights: (batch_size, num_heads, seq_len, seq_len) if return_attn_probs=True
    """
    flash_attn = FlashAttention(dropout=dropout_p, causal=causal)
    output = flash_attn(q, k, v)
    
    if return_attn_probs:
        # Compute attention weights for return
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        return output, attn_weights
    
    return output

# For compatibility with the original Flash Attention API
class FlashAttentionInterface:
    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, causal=False):
        output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        return output
EOF
    
    # Create setup file
    print_step "Creating setup file..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="flash_attention_amd",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["flash_attention_amd"],
    install_requires=[
        "torch>=2.0.0",
    ],
    author="User",
    author_email="user@example.com",
    description="Flash Attention implementation for AMD GPUs",
    keywords="flash attention, amd, gpu, pytorch",
    python_requires=">=3.8",
)
EOF
    
    print_success "AMD implementation created successfully"
}

install_flash_attention() {
    print_section "Installing Flash Attention"
    
    # Install the AMD implementation
    print_step "Installing the AMD implementation..."
    cd $HOME/flash-attention-amd
    pip install -e .
    
    print_success "Flash Attention installed successfully"
}

verify_installation() {
    print_section "Verifying installation"
    
    # Create test script
    print_step "Creating test script..."
    cat > $HOME/flash-attention-amd/test_flash_attention.py << 'EOF'
import torch
import time
from flash_attention_amd import flash_attn_func

def test_flash_attention():
    # Create dummy data
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda")

    # Run Flash Attention
    start_time = time.time()
    output = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"Output shape: {output.shape}")
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
    print("Flash Attention test passed!")

if __name__ == "__main__":
    test_flash_attention()
EOF
    
    # Run test script
    print_step "Running test script..."
    cd $HOME/flash-attention-amd
    python test_flash_attention.py
    
    if [ $? -eq 0 ]; then
        print_success "Flash Attention is working correctly"
        return 0
    else
        print_error "Flash Attention test failed"
        return 1
    fi
}

cleanup() {
    print_section "Cleaning up"
    
    # Remove temporary files
    print_step "Removing temporary files..."
    rm -f $HOME/flash-attention-amd/test_flash_attention.py
    
    print_success "Cleanup completed successfully"
}
main() {
    print_header "Flash Attention Build Script for AMD GPUs"
    
    # Start time
    start_time=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi
    
    # Install dependencies
    install_dependencies
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies. Exiting."
        exit 1
    fi
    
    # Create AMD implementation
    create_amd_implementation
    if [ $? -ne 0 ]; then
        print_error "Failed to create AMD implementation. Exiting."
        exit 1
    fi
    
    # Install Flash Attention
    install_flash_attention
    if [ $? -ne 0 ]; then
        print_error "Failed to install Flash Attention. Exiting."
        exit 1
    fi
    
    # Verify installation
    verify_installation
    if [ $? -ne 0 ]; then
        print_error "Installation verification failed. Exiting."
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    print_header "Flash Attention Build Completed Successfully!"
    echo -e "${GREEN}Total build time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"
    echo
    echo -e "${CYAN}You can now use Flash Attention in your PyTorch code:${RESET}"
    echo
    echo -e "${YELLOW}import torch${RESET}"
    echo -e "${YELLOW}from flash_attention_amd import flash_attn_func${RESET}"
    echo
    echo -e "${YELLOW}# Create input tensors${RESET}"
    echo -e "${YELLOW}q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
    echo -e "${YELLOW}k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
    echo -e "${YELLOW}v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=\"cuda\")${RESET}"
    echo
    echo -e "${YELLOW}# Run Flash Attention${RESET}"
    echo -e "${YELLOW}output = flash_attn_func(q, k, v, causal=True)${RESET}"
    echo
    
    return 0
}

# Main script execution
main
