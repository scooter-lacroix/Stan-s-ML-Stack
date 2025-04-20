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
install_all_core() {
    print_section "Installing all core components"
    
    # Install ROCm configuration
    install_rocm_config
    if [ $? -ne 0 ]; then
        print_error "ROCm configuration installation failed."
        return 1
    fi
    
    # Install PyTorch
    install_pytorch
    if [ $? -ne 0 ]; then
        print_error "PyTorch installation failed."
        return 1
    fi
    
    # Install ONNX Runtime
    install_onnx_runtime
    if [ $? -ne 0 ]; then
        print_error "ONNX Runtime installation failed."
        return 1
    fi
    
    # Install MIGraphX
    install_migraphx
    if [ $? -ne 0 ]; then
        print_error "MIGraphX installation failed."
        return 1
    fi
    
    # Install Megatron-LM
    install_megatron
    if [ $? -ne 0 ]; then
        print_error "Megatron-LM installation failed."
        return 1
    fi
    
    # Install Flash Attention
    install_flash_attention
    if [ $? -ne 0 ]; then
        print_error "Flash Attention installation failed."
        return 1
    fi
    
    # Install RCCL
    install_rccl
    if [ $? -ne 0 ]; then
        print_error "RCCL installation failed."
        return 1
    fi
    
    # Install MPI
    install_mpi
    if [ $? -ne 0 ]; then
        print_error "MPI installation failed."
        return 1
    fi
    
    print_success "All core components installed successfully"
    return 0
}

verify_installation() {
    print_section "Verifying installation"
    
    # Create verification script
    print_step "Creating verification script..."
    cat > $HOME/verify_ml_stack.py << 'EOF'
import sys
import os
import importlib.util

# Color definitions
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"{CYAN}{BOLD}=== {text} ==={RESET}")
    print()

def print_section(text):
    print(f"{BLUE}{BOLD}>>> {text}{RESET}")

def print_step(text):
    print(f"{MAGENTA}>> {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def check_module(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print_success(f"{display_name} is installed (version: {version})")
        return module
    except ImportError:
        print_error(f"{display_name} is not installed")
        return None

def main():
    print_header("ML Stack Verification")
    
    # Check PyTorch
    print_section("Checking PyTorch")
    torch = check_module("torch", "PyTorch")
    if torch:
        # Check CUDA availability
        if torch.cuda.is_available():
            print_success("CUDA is available through ROCm")
            
            # Check number of GPUs
            device_count = torch.cuda.device_count()
            print_step(f"Number of GPUs: {device_count}")
            
            # Check GPU information
            for i in range(device_count):
                print_step(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Run a simple tensor operation
            try:
                x = torch.ones(10, device="cuda")
                y = x + 1
                print_success("Simple tensor operation on GPU successful")
            except Exception as e:
                print_error(f"Simple tensor operation on GPU failed: {e}")
        else:
            print_error("CUDA is not available through ROCm")
    
    # Check ONNX Runtime
    print_section("Checking ONNX Runtime")
    ort = check_module("onnxruntime", "ONNX Runtime")
    if ort:
        # Check available providers
        providers = ort.get_available_providers()
        print_step(f"Available providers: {providers}")
        
        # Check if ROCMExecutionProvider is available
        if 'ROCMExecutionProvider' in providers:
            print_success("ROCMExecutionProvider is available")
        else:
            print_warning("ROCMExecutionProvider is not available")
    
    # Check MIGraphX
    print_section("Checking MIGraphX")
    check_module("migraphx", "MIGraphX")
    
    # Check Megatron-LM
    print_section("Checking Megatron-LM")
    try:
        import megatron
        print_success("Megatron-LM is installed")
    except ImportError:
        print_error("Megatron-LM is not installed")
    
    # Check Flash Attention
    print_section("Checking Flash Attention")
    try:
        from flash_attention_amd import flash_attn_func
        print_success("Flash Attention is installed")
    except ImportError:
        print_error("Flash Attention is not installed")
    
    # Check RCCL
    print_section("Checking RCCL")
    if os.path.exists("/opt/rocm/lib/librccl.so"):
        print_success("RCCL is installed")
    else:
        print_error("RCCL is not installed")
    
    # Check MPI
    print_section("Checking MPI")
    if os.system("which mpirun > /dev/null") == 0:
        print_success("MPI is installed")
        
        # Check mpi4py
        check_module("mpi4py", "mpi4py")
    else:
        print_error("MPI is not installed")
    
    print_header("Verification Complete")

if __name__ == "__main__":
    main()
EOF
    
    # Run verification script
    print_step "Running verification script..."
    python3 $HOME/verify_ml_stack.py
    
    # Clean up
    print_step "Cleaning up..."
    rm -f $HOME/verify_ml_stack.py
    
    print_success "Verification completed"
}

show_menu() {
    print_header "ML Stack Installation Menu"
    
    echo -e "1) Install ROCm Configuration"
    echo -e "2) Install PyTorch with ROCm support"
    echo -e "3) Install ONNX Runtime with ROCm support"
    echo -e "4) Install MIGraphX"
    echo -e "5) Install Megatron-LM"
    echo -e "6) Install Flash Attention with AMD GPU support"
    echo -e "7) Install RCCL"
    echo -e "8) Install MPI"
    echo -e "9) Install All Core Components"
    echo -e "10) Verify Installation"
    echo -e "0) Exit"
    echo
    
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            install_rocm_config
            ;;
        2)
            install_pytorch
            ;;
        3)
            install_onnx_runtime
            ;;
        4)
            install_migraphx
            ;;
        5)
            install_megatron
            ;;
        6)
            install_flash_attention
            ;;
        7)
            install_rccl
            ;;
        8)
            install_mpi
            ;;
        9)
            install_all_core
            ;;
        10)
            verify_installation
            ;;
        0)
            print_header "Exiting ML Stack Installation"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            ;;
    esac
    
    # Show menu again
    show_menu
}

main() {
    print_header "ML Stack Installation Script"
    
    # Start time
    start_time=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    if [ $? -ne 0 ]; then
        print_error "Prerequisites check failed. Exiting."
        exit 1
    fi
    
    # Show menu
    show_menu
    
    # End time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    print_header "ML Stack Installation Completed"
    echo -e "${GREEN}Total installation time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"
    
    return 0
}

# Main script execution
main
