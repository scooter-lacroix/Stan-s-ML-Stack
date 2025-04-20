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
