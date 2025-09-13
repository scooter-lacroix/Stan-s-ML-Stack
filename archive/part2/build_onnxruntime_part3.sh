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
cleanup() {
    print_section "Cleaning up"
    
    # Remove temporary files
    print_step "Removing temporary files..."
    cd $HOME/onnxruntime-build
    rm -f test_onnxruntime.py simple_model.onnx
    
    print_success "Cleanup completed successfully"
}

main() {
    print_header "ONNX Runtime Build Script for AMD GPUs"
    
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
    
    # Clone repository
    clone_repository
    if [ $? -ne 0 ]; then
        print_error "Failed to clone repository. Exiting."
        exit 1
    fi
    
    # Configure build
    configure_build
    if [ $? -ne 0 ]; then
        print_error "Failed to configure build. Exiting."
        exit 1
    fi
    
    # Build ONNX Runtime
    build_onnxruntime
    if [ $? -ne 0 ]; then
        print_error "Failed to build ONNX Runtime. Exiting."
        exit 1
    fi
    
    # Install ONNX Runtime
    install_onnxruntime
    if [ $? -ne 0 ]; then
        print_error "Failed to install ONNX Runtime. Exiting."
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
    
    print_header "ONNX Runtime Build Completed Successfully!"
    echo -e "${GREEN}Total build time: ${BOLD}${hours}h ${minutes}m ${seconds}s${RESET}"
    echo
    echo -e "${CYAN}You can now use ONNX Runtime in your Python code:${RESET}"
    echo
    echo -e "${YELLOW}import onnxruntime as ort${RESET}"
    echo -e "${YELLOW}import numpy as np${RESET}"
    echo
    echo -e "${YELLOW}# Create ONNX Runtime session${RESET}"
    echo -e "${YELLOW}session = ort.InferenceSession(\"model.onnx\", providers=['ROCMExecutionProvider'])${RESET}"
    echo
    echo -e "${YELLOW}# Run inference${RESET}"
    echo -e "${YELLOW}input_name = session.get_inputs()[0].name${RESET}"
    echo -e "${YELLOW}output_name = session.get_outputs()[0].name${RESET}"
    echo -e "${YELLOW}ort_inputs = {input_name: np.random.randn(1, 10).astype(np.float32)}${RESET}"
    echo -e "${YELLOW}ort_output = session.run([output_name], ort_inputs)[0]${RESET}"
    echo
    
    return 0
}

# Main script execution
main
