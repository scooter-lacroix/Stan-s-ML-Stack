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
install_rocm_config() {
    print_section "Installing ROCm configuration"
    
    # Create ROCm configuration file
    print_step "Creating ROCm configuration file..."
    
    # Create .rocmrc file in home directory
    cat > $HOME/.rocmrc << EOF
# ROCm Configuration File
# Created by ML Stack Installation Script

# Environment Variables
export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_ROCM_DEVICE=$PYTORCH_ROCM_DEVICE

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=1

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# Logging Settings
export HIP_TRACE_API=0
export AMD_LOG_LEVEL=4
EOF
    
    # Add source to .bashrc if not already there
    if ! grep -q "source \$HOME/.rocmrc" $HOME/.bashrc; then
        echo -e "\n# Source ROCm configuration" >> $HOME/.bashrc
        echo "source \$HOME/.rocmrc" >> $HOME/.bashrc
    fi
    
    # Source the file
    source $HOME/.rocmrc
    
    print_success "ROCm configuration installed successfully"
}

install_pytorch() {
    print_section "Installing PyTorch with ROCm support"
    
    # Check if PyTorch with ROCm is already installed
    if python3 -c "import torch; print(torch.version.hip)" &> /dev/null; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        rocm_version=$(python3 -c "import torch; print(torch.version.hip)")
        print_warning "PyTorch with ROCm support is already installed (PyTorch $pytorch_version, ROCm $rocm_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping PyTorch installation."
            return 0
        fi
    fi
    
    # Install PyTorch with ROCm support
    print_step "Installing PyTorch with ROCm support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    
    # Verify installation
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
        rocm_version=$(python3 -c "import torch; print(torch.version.hip)")
        print_success "PyTorch with ROCm support installed successfully (PyTorch $pytorch_version, ROCm $rocm_version)"
        
        # Print GPU information
        print_step "GPU information:"
        python3 -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    else
        print_error "PyTorch installation failed. CUDA is not available."
        return 1
    fi
    
    return 0
}

install_onnx_runtime() {
    print_section "Installing ONNX Runtime with ROCm support"
    
    # Check if ONNX Runtime is already installed
    if python3 -c "import onnxruntime" &> /dev/null; then
        onnx_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
        print_warning "ONNX Runtime is already installed (version $onnx_version)."
        
        # Check if ROCMExecutionProvider is available
        if python3 -c "import onnxruntime; print('ROCMExecutionProvider' in onnxruntime.get_available_providers())" | grep -q "True"; then
            print_success "ONNX Runtime with ROCm support is already installed."
            read -p "Do you want to reinstall? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_step "Skipping ONNX Runtime installation."
                return 0
            fi
        else
            print_warning "ONNX Runtime is installed but ROCMExecutionProvider is not available."
            print_step "Reinstalling ONNX Runtime with ROCm support..."
        fi
    fi
    
    # Build and install ONNX Runtime with ROCm support
    print_step "Building ONNX Runtime with ROCm support..."
    
    # Run the build script
    $HOME/Desktop/Stans_MLStack/scripts/build_onnxruntime.sh
    
    # Check if installation was successful
    if [ $? -ne 0 ]; then
        print_error "ONNX Runtime installation failed."
        return 1
    fi
    
    print_success "ONNX Runtime with ROCm support installed successfully"
    return 0
}

install_migraphx() {
    print_section "Installing MIGraphX"
    
    # Check if MIGraphX is already installed
    if python3 -c "import migraphx" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_warning "MIGraphX is already installed (version $migraphx_version)."
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping MIGraphX installation."
            return 0
        fi
    fi
    
    # Install MIGraphX from ROCm repository
    print_step "Installing MIGraphX from ROCm repository..."
    sudo apt-get update
    sudo apt-get install -y migraphx python3-migraphx
    
    # Verify installation
    if python3 -c "import migraphx; print(migraphx.__version__)" &> /dev/null; then
        migraphx_version=$(python3 -c "import migraphx; print(migraphx.__version__)")
        print_success "MIGraphX installed successfully (version $migraphx_version)"
    else
        print_error "MIGraphX installation failed."
        return 1
    fi
    
    return 0
}
