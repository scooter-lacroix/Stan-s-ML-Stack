#!/bin/bash
#
# Script to create persistent environment variables and symlinks
# for Stan's ML Stack - Enhanced Version with User Choice
#

# Source the component detector library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DETECTOR_SCRIPT="$PARENT_DIR/scripts/ml_stack_component_detector.sh"

if [ -f "$DETECTOR_SCRIPT" ]; then
    source "$DETECTOR_SCRIPT"
else
    echo "Error: Component detector script not found at $DETECTOR_SCRIPT"
    exit 1
fi

# Enhanced color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Enhanced print functions
print_header() {
    echo
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}║               === $1 ===               ║${RESET}"
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}"
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}"
    echo
}

print_section() {
    echo
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${BLUE}${BOLD}│ $1${RESET}"
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}"
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

print_info() {
    echo -e "${CYAN}ℹ $1${RESET}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (use sudo)${RESET}"
  exit 1
fi

# Detect existing customizations
detect_existing_setup() {
    local bashrc_path="$HOME/.bashrc"
    local has_custom_prompt=false
    local has_ml_stack_env=false
    local has_enhanced_functions=false
    local has_fastfetch=false
    local has_chalk=false
    local has_lolcat=false
    local has_figlet=false

    if [ -f "$bashrc_path" ]; then
        # Check for custom prompt
        if grep -q "get_git_branch\|get_venv_info\|get_node_info" "$bashrc_path"; then
            has_custom_prompt=true
        fi

        # Check for ML Stack environment
        if grep -q "mlstack_env" "$bashrc_path"; then
            has_ml_stack_env=true
        fi

        # Check for enhanced functions
        if grep -q "success()\|error()\|warning()" "$bashrc_path"; then
            has_enhanced_functions=true
        fi
    fi

    # Check for installed tools (more robust detection)
    if command -v fastfetch &> /dev/null; then
        has_fastfetch=true
    fi

    # Check for chalk in multiple locations (also check as original user if running as sudo)
    if command -v chalk &> /dev/null || [ -f "/usr/local/bin/chalk" ] || [ -f "/usr/bin/chalk" ] || [ -L "$HOME/.npm-global/bin/chalk" ] || find "$HOME/.nvm/versions/node" -name "chalk" -type l 2>/dev/null | grep -q chalk || npm list -g 2>/dev/null | grep -q chalk-cli; then
        has_chalk=true
    elif [ -n "$SUDO_USER" ]; then
        # If running as sudo, check as the original user with proper nvm sourcing
        if sudo -u "$SUDO_USER" -i bash -c "source ~/.nvm/nvm.sh 2>/dev/null && nvm use --lts >/dev/null 2>&1 && command -v chalk" &>/dev/null || sudo -u "$SUDO_USER" -i bash -c "[ -L \$HOME/.npm-global/bin/chalk ]" || sudo -u "$SUDO_USER" -i bash -c "find \$HOME/.nvm/versions/node -name chalk -type l 2>/dev/null | grep -q chalk"; then
            has_chalk=true
        fi
    fi

    # Check for lolcat in multiple locations
    if command -v lolcat &> /dev/null || [ -f "/snap/bin/lolcat" ] || [ -f "/usr/bin/lolcat" ]; then
        has_lolcat=true
    fi

    if command -v figlet &> /dev/null; then
        has_figlet=true
    fi

    # Return results
    echo "$has_custom_prompt:$has_ml_stack_env:$has_enhanced_functions:$has_fastfetch:$has_chalk:$has_lolcat:$has_figlet"
}

# Install required dependencies
install_dependencies() {
    print_section "Installing Required Dependencies"

    local deps_to_install=""

    # Check what we need to install
    if ! command -v fastfetch &> /dev/null; then
        deps_to_install="$deps_to_install fastfetch"
    fi
    if ! command -v lolcat &> /dev/null; then
        deps_to_install="$deps_to_install lolcat"
    fi
    if ! command -v figlet &> /dev/null; then
        deps_to_install="$deps_to_install figlet"
    fi

    # Handle npm packages separately
    npm_packages=""
    if ! command -v chalk &> /dev/null && ! [ -f "/usr/local/bin/chalk" ] && ! [ -f "/usr/bin/chalk" ] && ! [ -L "$HOME/.npm-global/bin/chalk" ] && ! find "$HOME/.nvm/versions/node" -name "chalk" -type l 2>/dev/null | grep -q chalk && ! npm list -g 2>/dev/null | grep -q chalk-cli; then
        # Also check as original user if running as sudo
        if [ -n "$SUDO_USER" ]; then
            if ! sudo -u "$SUDO_USER" -i bash -c "source ~/.nvm/nvm.sh 2>/dev/null && nvm use --lts >/dev/null 2>&1 && command -v chalk" &>/dev/null && ! sudo -u "$SUDO_USER" -i bash -c "[ -L \$HOME/.npm-global/bin/chalk ]" && ! sudo -u "$SUDO_USER" -i bash -c "find \$HOME/.nvm/versions/node -name chalk -type l 2>/dev/null | grep -q chalk"; then
                npm_packages="$npm_packages chalk-cli"
            fi
        else
            npm_packages="$npm_packages chalk-cli"
        fi
    fi

    if [ -n "$deps_to_install" ]; then
        print_info "Installing missing system dependencies: $deps_to_install"
        apt update && apt install -y $deps_to_install
        print_success "System dependencies installed"
    fi

    if [ -n "$npm_packages" ]; then
        print_info "Installing missing npm packages: $npm_packages"
        if command -v npm &> /dev/null; then
            npm install -g $npm_packages
            print_success "NPM packages installed"
        elif [ -n "$SUDO_USER" ]; then
            # Try to install as the original user with nvm
            print_info "Attempting to install npm packages as user $SUDO_USER..."
            if sudo -u "$SUDO_USER" -i bash -c "source ~/.nvm/nvm.sh 2>/dev/null && nvm use --lts >/dev/null 2>&1 && npm install -g $npm_packages"; then
                print_success "NPM packages installed successfully"
            else
                print_warning "Failed to install npm packages as user"
                print_info "Please install manually:"
                print_info "  source ~/.nvm/nvm.sh && nvm use --lts && npm install -g $npm_packages"
                print_info "Then re-run this script"
            fi
        else
            print_warning "npm not found"
            print_info "Please install manually:"
            print_info "  npm install -g $npm_packages"
            print_info "Then re-run this script"
        fi
    fi

    # Check if wrong chalk package is installed and offer to fix
    if npm list -g 2>/dev/null | grep -q "chalk@" && ! npm list -g 2>/dev/null | grep -q "chalk-cli"; then
        print_warning "Found 'chalk' package but need 'chalk-cli' for command-line usage"
        if command -v npm &> /dev/null; then
            print_info "Installing correct chalk-cli package..."
            npm uninstall -g chalk
            npm install -g chalk-cli
            print_success "Installed chalk-cli"
        fi
    fi

    if [ -z "$deps_to_install" ] && [ -z "$npm_packages" ]; then
        print_success "All dependencies already installed"
    fi
}

# Main setup function
main_setup() {
    print_header "Creating Persistent Environment for ML Stack"

    # Update PATH to include npm global bin and snap bin for better detection
    export PATH="$PATH:/usr/local/bin:/snap/bin"
    if [ -d "$HOME/.npm-global/bin" ]; then
        export PATH="$PATH:$HOME/.npm-global/bin"
    fi
    # Also check for nvm paths
    if [ -d "$HOME/.nvm" ]; then
        for node_version in "$HOME/.nvm/versions/node"/*/; do
            if [ -d "${node_version}bin" ]; then
                export PATH="$PATH:${node_version}bin"
            fi
        done
    fi

    # Detect existing setup
    print_section "Analyzing Current Setup"
    local setup_info=$(detect_existing_setup)
    IFS=':' read -r has_custom_prompt has_ml_stack_env has_enhanced_functions has_fastfetch has_chalk has_lolcat has_figlet <<< "$setup_info"

    # Re-check chalk detection after PATH updates (also check as original user with nvm)
    if ! [ "$has_chalk" = "true" ]; then
        if command -v chalk &> /dev/null || [ -f "/usr/local/bin/chalk" ] || [ -f "/usr/bin/chalk" ] || [ -L "$HOME/.npm-global/bin/chalk" ] || find "$HOME/.nvm/versions/node" -name "chalk" -type l 2>/dev/null | grep -q chalk; then
            has_chalk=true
        else
            # Try to detect chalk as the original user with nvm sourcing
            if [ -n "$SUDO_USER" ]; then
                if sudo -u "$SUDO_USER" -i bash -c "source ~/.nvm/nvm.sh 2>/dev/null && nvm use --lts >/dev/null 2>&1 && command -v chalk" &>/dev/null || sudo -u "$SUDO_USER" -i bash -c "[ -L \$HOME/.npm-global/bin/chalk ]" || sudo -u "$SUDO_USER" -i bash -c "find \$HOME/.nvm/versions/node -name chalk -type l 2>/dev/null | grep -q chalk"; then
                    has_chalk=true
                fi
            fi
        fi
    fi

    echo "Current setup analysis:"
    echo "  Custom prompt: $([ "$has_custom_prompt" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  ML Stack env: $([ "$has_ml_stack_env" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  Enhanced functions: $([ "$has_enhanced_functions" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  Fastfetch: $([ "$has_fastfetch" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  Chalk: $([ "$has_chalk" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  Lolcat: $([ "$has_lolcat" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo "  Figlet: $([ "$has_figlet" = "true" ] && echo "✅ Yes" || echo "❌ No")"
    echo

    # Ask user what they want to do
    if [ "$has_custom_prompt" = "true" ] || [ "$has_enhanced_functions" = "true" ]; then
        print_warning "Detected existing custom terminal setup"
        echo
        echo "What would you like to do?"
        echo "1) Keep existing setup and just add ROCm paths"
        echo "2) Integrate ML Stack branding into existing setup"
        echo "3) Replace with complete ML Stack terminal customization"
        echo "4) Skip terminal customization (just ROCm paths)"
        echo
        read -p "Choose option (1-4) [1]: " CUSTOMIZATION_CHOICE
        CUSTOMIZATION_CHOICE=${CUSTOMIZATION_CHOICE:-1}
    else
        echo "No existing customizations detected."
        echo "What would you like to do?"
        echo "1) Install complete ML Stack terminal customization"
        echo "2) Just add ROCm paths (minimal setup)"
        echo
        read -p "Choose option (1-2) [1]: " CUSTOMIZATION_CHOICE
        CUSTOMIZATION_CHOICE=${CUSTOMIZATION_CHOICE:-1}

        # Adjust choice for the simpler menu
        if [ "$CUSTOMIZATION_CHOICE" = "2" ]; then
            CUSTOMIZATION_CHOICE=4
        else
            CUSTOMIZATION_CHOICE=3
        fi
    fi

    # Install dependencies first
    install_dependencies

    # Process user's choice
    case $CUSTOMIZATION_CHOICE in
        1)
            print_info "Adding ROCm paths to existing setup..."
            setup_rocm_paths_only
            ;;
        2)
            print_info "Integrating ML Stack branding..."
            integrate_ml_stack_branding
            ;;
        3)
            print_info "Installing complete ML Stack terminal customization..."
            setup_complete_customization
            ;;
        4)
            print_info "Setting up minimal ROCm paths only..."
            setup_rocm_paths_only
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac

    print_success "ML Stack persistent environment setup complete!"

    # Create verification script
    create_verification_script
}

create_verification_script() {
    print_section "Creating Verification Script"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VERIFY_SCRIPT="$SCRIPT_DIR/verify-mlstack-env.sh"

    cat > "$VERIFY_SCRIPT" << 'EOF'
#!/bin/bash
# ML Stack Environment Verification Script

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Print header
echo -e "${BLUE}${BOLD}=== ML Stack Environment Verification ===${RESET}\n"

# Check ROCm installation
echo -e "${BLUE}>> Checking ROCm installation...${RESET}"
if command -v rocminfo &> /dev/null; then
    rocm_version=$(rocminfo 2>/dev/null | grep "ROCm Version" | sed 's/.*ROCm Version\s*:\s*\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/' | head -n1)
    if [ -n "$rocm_version" ]; then
        echo -e "${GREEN}✓ ROCm installed (version: $rocm_version)${RESET}"
    else
        echo -e "${GREEN}✓ ROCm installed (version unknown)${RESET}"
    fi

    # Check GPU detection
    gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type:.*GPU")
    if [ "$gpu_count" -gt 0 ]; then
        echo -e "${GREEN}✓ Detected $gpu_count AMD GPU(s)${RESET}"
        rocminfo 2>/dev/null | grep "Marketing Name" | sed 's/.*Marketing Name\s*:\s*//' | while read -r gpu; do
            echo -e "${GREEN}  - $gpu${RESET}"
        done
    else
        echo -e "${YELLOW}⚠ No AMD GPUs detected by ROCm${RESET}"
    fi
else
    echo -e "${RED}✗ ROCm not installed or not in PATH${RESET}"
fi

# Check environment variables
echo -e "\n${BLUE}>> Checking environment variables...${RESET}"
ENV_VARS=(
    "ROCM_PATH:/opt/rocm"
    "LD_LIBRARY_PATH:/opt/rocm/lib"
    "PATH:/opt/rocm/bin"
    "HIP_VISIBLE_DEVICES:0,1"
)

for env_var in "${ENV_VARS[@]}"; do
    var_name=$(echo $env_var | cut -d: -f1)
    expected_value=$(echo $env_var | cut -d: -f2)

    if [ -n "${!var_name}" ]; then
        if [[ "${!var_name}" == *"$expected_value"* ]]; then
            echo -e "${GREEN}✓ $var_name is set correctly${RESET}"
        else
            echo -e "${YELLOW}⚠ $var_name is set but may not contain expected path${RESET}"
        fi
    else
        echo -e "${RED}✗ $var_name is not set${RESET}"
    fi
done

# Check PyTorch ROCm support
echo -e "\n${BLUE}>> Checking PyTorch ROCm support...${RESET}"
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✓ PyTorch can access $gpu_count GPU(s) through ROCm${RESET}"
else
    echo -e "${RED}✗ PyTorch cannot access GPUs through ROCm${RESET}"
fi

# Check symlinks
echo -e "\n${BLUE}>> Checking symlinks...${RESET}"
SYMLINKS=(
    "/usr/local/cuda:/opt/rocm"
)

for link in "${SYMLINKS[@]}"; do
    src=$(echo $link | cut -d: -f1)
    dst=$(echo $link | cut -d: -f2)

    if [ -L "$src" ]; then
        if [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ]; then
            echo -e "${GREEN}✓ $src -> $dst${RESET}"
        else
            echo -e "${YELLOW}⚠ $src points to $(readlink -f "$src") (expected: $dst)${RESET}"
        fi
    else
        echo -e "${RED}✗ $src is not a symlink to $dst${RESET}"
    fi
done

echo -e "\n${BLUE}${BOLD}=== Verification Complete ===${RESET}"
echo -e "To verify again: ${GREEN}./verify-mlstack-env.sh${RESET}"
echo -e "Run benchmarks with: ${GREEN}python3 ../benchmarks/pytorch_gpu_benchmark.py${RESET}"
EOF

    chmod +x "$VERIFY_SCRIPT"
    print_success "Created verification script: $VERIFY_SCRIPT"
}

# Setup functions for different customization levels
setup_rocm_paths_only() {
    print_section "Setting up ROCm Paths Only"

    # Create or update system-wide environment file
    if [ -f "/etc/profile.d/mlstack.sh" ]; then
        print_info "Updating existing ML Stack environment file..."
    else
        print_info "Creating new ML Stack environment file..."
    fi

    cat > /etc/profile.d/mlstack.sh << 'EOF'
#!/bin/bash
# ML Stack Environment Variables
# This file is automatically loaded at login

# Check if ROCm exists
if [ -d "/opt/rocm" ]; then
    # ROCm paths
    export ROCM_PATH=/opt/rocm
    
    # Safe path addition function
    safe_add_path() {
        if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
            export PATH="$1:$PATH"
        fi
    }

    safe_add_path "/opt/rocm/bin"
    safe_add_path "/opt/rocm/hip/bin"
    safe_add_path "/usr/local/bin"
    safe_add_path "/usr/bin"
    safe_add_path "/bin"

    # Safe LD_LIBRARY_PATH addition
    safe_add_ld_path() {
        if [ -d "$1" ] && [[ ":$LD_LIBRARY_PATH:" != *":$1:"* ]]; then
            export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
        fi
    }

    safe_add_ld_path "/opt/rocm/lib"
    safe_add_ld_path "/opt/rocm/hip/lib"
    safe_add_ld_path "/opt/rocm/opencl/lib"

    # CUDA compatibility
    export ROCM_HOME=$ROCM_PATH
    export CUDA_HOME=$ROCM_PATH

    # GPU selection (only set if not already set)
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        export HIP_VISIBLE_DEVICES=0,1
    fi
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    fi
    if [ -z "$PYTORCH_ROCM_DEVICE" ]; then
        export PYTORCH_ROCM_DEVICE=0,1
    fi

    # Performance settings
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export HSA_ENABLE_SDMA=0
    export GPU_MAX_HEAP_SIZE=100
    export GPU_MAX_ALLOC_PERCENT=100
    # HSA_TOOLS_LIB must be a library path or 0, not 1
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
    else
        export HSA_TOOLS_LIB=0
    fi

    # MIOpen settings
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
    export MIOPEN_FIND_MODE=3
    export MIOPEN_FIND_ENFORCE=3

    # PyTorch settings
    export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"

    # ONNX Runtime
    if [ -d "$HOME/onnxruntime_build/onnxruntime/build/Linux/Release" ]; then
        export PYTHONPATH=$HOME/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
    fi

    # Flash Attention
    if [ -d "$HOME/ml_stack/flash_attn_amd_direct" ]; then
        export PYTHONPATH=$HOME/ml_stack/flash_attn_amd_direct:$PYTHONPATH
    fi

    # Megatron-LM
    if [ -d "$HOME/megatron/Megatron-LM" ]; then
        export PYTHONPATH=$HOME/megatron/Megatron-LM:$PYTHONPATH
    fi
fi
EOF

    # Make the file executable
    chmod +x /etc/profile.d/mlstack.sh
    print_success "ROCm paths added to system environment"

    # Create symlinks
    create_symlinks

    # Create user environment file if it doesn't exist
    if [ ! -f "$HOME/.mlstack_env" ]; then
        create_user_env_file
    fi
}

integrate_ml_stack_branding() {
    print_section "Integrating ML Stack Branding"

    # Backup existing .bashrc
    cp $HOME/.bashrc $HOME/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    print_success "Backed up existing .bashrc"

    # Add ML Stack branding to existing welcome message
    if grep -q "show_welcome" $HOME/.bashrc; then
        # Replace existing welcome function to include ML Stack branding
        sed -i 's/WELCOME BACK/WELCOME TO ML STACK/g' $HOME/.bashrc
    fi

    # Update the figlet welcome message
    sed -i 's/"WELCOME TO ML STACK"/"WELCOME TO YOUR ML STACK"/g' $HOME/.bashrc

    # Update fastfetch config if it exists
    if [ -f "$HOME/.config/fastfetch/config.jsonc" ]; then
        sed -i 's/" 🚀 ",/" 🎯 ML Stack ",' $HOME/.config/fastfetch/config.jsonc
        sed -i 's/" 🚀 CPU",/" 🎯 ML Stack CPU",/g' $HOME/.config/fastfetch/config.jsonc
        print_success "Updated Fastfetch config with ML Stack branding"
    fi

    # Add ROCm paths
    setup_rocm_paths_only
}

setup_complete_customization() {
    print_section "Setting up Complete ML Stack Terminal Customization"

    # Backup existing .bashrc
    cp $HOME/.bashrc $HOME/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    print_success "Backed up existing .bashrc"

    # Create enhanced fastfetch config with ML Stack branding
    mkdir -p $HOME/.config/fastfetch/
    cat > $HOME/.config/fastfetch/config.jsonc << 'EOF'
{
    "$schema": "https://github.com/fastfetch-cli/fastfetch/raw/dev/doc/json_schema.json",
    "logo": {
        "type": "small",
        "color": {
            "1": "blue",
            "2": "cyan"
        }
    },
    "display": {
        "separator": "  "
    },
    "modules": [
        {
            "type": "custom",
            "format": "╔═════════════════════════════════════════════════════════╗",
            "keyColor": "blue"
        },
        {
            "type": "custom",
            "format": "║                                                         ║",
            "keyColor": "blue"
        },
        {
            "type": "custom",
            "format": "║              🎯 STAN'S ML STACK              ║",
            "keyColor": "cyan"
        },
        {
            "type": "custom",
            "format": "║                                                         ║",
            "keyColor": "blue"
        },
        {
            "type": "custom",
            "format": "╚═════════════════════════════════════════════════════════╝",
            "keyColor": "blue"
        },
        {
            "type": "title",
            "key": " 🚀 ",
            "keyColor": "blue"
        },
        {
            "type": "separator"
        },
        {
            "type": "os",
            "key": " 🖥️  OS"
        },
        {
            "type": "kernel",
            "key": " 🧠 Kernel"
        },
        {
            "type": "uptime",
            "key": " ⏰ Uptime"
        },
        {
            "type": "packages",
            "key": " 📦 Packages"
        },
        {
            "type": "shell",
            "key": " 🐚 Shell"
        },
        {
            "type": "terminal",
            "key": " 💻 Terminal"
        },
        {
            "type": "cpu",
            "key": " 🚀 CPU"
        },
        {
            "type": "gpu",
            "key": " 🎮 GPU"
        },
        {
            "type": "memory",
            "key": " 💾 Memory"
        },
        {
            "type": "disk",
            "key": " 💿 Disk"
        },
        {
            "type": "localip",
            "key": " 🌐 Local IP"
        },
        {
            "type": "colors",
            "paddingLeft": 2,
            "symbol": "circle"
        }
    ]
}
EOF
    print_success "Created enhanced Fastfetch config with ML Stack branding"

    # Add ROCm paths
    setup_rocm_paths_only
}

create_symlinks() {
    print_section "Creating Symlinks"

    # Create ninja symlinks
    if [ -f "/usr/bin/ninja" ] && [ ! -f "/usr/bin/ninja-build" ]; then
        ln -sf /usr/bin/ninja /usr/bin/ninja-build
        echo "✓ Created symlink: /usr/bin/ninja-build -> /usr/bin/ninja"
    elif [ -f "/usr/bin/ninja-build" ] && [ ! -f "/usr/bin/ninja" ]; then
        ln -sf /usr/bin/ninja-build /usr/bin/ninja
        echo "✓ Created symlink: /usr/bin/ninja -> /usr/bin/ninja-build"
    fi

    # Create ROCm symlinks if needed
    if [ -d "/opt/rocm" ]; then
        # Create CUDA compatibility symlinks
        if [ ! -d "/usr/local/cuda" ]; then
            ln -sf /opt/rocm /usr/local/cuda
            echo "✓ Created symlink: /usr/local/cuda -> /opt/rocm"
        fi

        # Create RCCL symlinks if needed
        if [ -f "/opt/rocm/lib/librccl.so" ] && [ ! -d "/opt/rocm/rccl" ]; then
            mkdir -p /opt/rocm/rccl/lib
            ln -sf /opt/rocm/lib/librccl.so /opt/rocm/rccl/lib/librccl.so
            echo "✓ Created symlink: /opt/rocm/rccl/lib/librccl.so -> /opt/rocm/lib/librccl.so"
        fi
    fi

    # Create Python module symlinks if needed
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

    # Flash Attention symlink
    if [ -d "$HOME/ml_stack/flash_attn_amd_direct" ] && [ ! -d "$PYTHON_SITE_PACKAGES/flash_attention_amd" ]; then
        ln -sf $HOME/ml_stack/flash_attn_amd_direct "$PYTHON_SITE_PACKAGES/flash_attention_amd"
        echo "✓ Created symlink: $PYTHON_SITE_PACKAGES/flash_attention_amd -> $HOME/ml_stack/flash_attn_amd_direct"
    fi

    # Megatron-LM symlink
    if [ -d "$HOME/megatron/Megatron-LM" ] && [ ! -d "$PYTHON_SITE_PACKAGES/megatron" ]; then
        ln -sf $HOME/megatron/Megatron-LM "$PYTHON_SITE_PACKAGES/megatron"
        echo "✓ Created symlink: $PYTHON_SITE_PACKAGES/megatron -> $HOME/megatron/Megatron-LM"
    fi

    # ONNX Runtime symlink
    if [ -d "$HOME/onnxruntime_build/onnxruntime/build/Linux/Release" ] && [ ! -d "$PYTHON_SITE_PACKAGES/onnxruntime" ]; then
        ln -sf $HOME/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime "$PYTHON_SITE_PACKAGES/onnxruntime"
        echo "✓ Created symlink: $PYTHON_SITE_PACKAGES/onnxruntime -> $HOME/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime"
    fi

    print_success "Symlinks created"
}

create_user_env_file() {
    print_section "Creating User Environment File"

    cat > $HOME/.mlstack_env << 'EOF'
# ML Stack User Environment
# Source this file in your .bashrc or .zshrc

# Source the system-wide environment file
if [ -f "/etc/profile.d/mlstack.sh" ]; then
    source /etc/profile.d/mlstack.sh
fi

# Add any user-specific environment variables here
EOF

    # Change ownership of the user file
    chown stan:stan $HOME/.mlstack_env
    print_success "Created user-specific environment file: $HOME/.mlstack_env"

    # Add to .bashrc if not already there
    if ! grep -q "source ~/.mlstack_env" $HOME/.bashrc; then
        echo -e "\n# Source ML Stack environment" >> $HOME/.bashrc
        echo "source ~/.mlstack_env" >> $HOME/.bashrc
        print_success "Added environment sourcing to .bashrc"
    else
        print_info "Environment sourcing already in .bashrc"
    fi
}

# Create systemd service to create symlinks at boot
create_systemd_service() {
    print_section "Creating Systemd Service"

    cat > /etc/systemd/system/mlstack-symlinks.service << 'EOF'
[Unit]
Description=Create ML Stack Symlinks
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash /usr/local/bin/mlstack-symlinks.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Create symlink script
    cat > /usr/local/bin/mlstack-symlinks.sh << 'EOF'
#!/bin/bash
# Script to create persistent symlinks for ML Stack

# Create ninja symlinks
if [ -f "/usr/bin/ninja" ] && [ ! -f "/usr/bin/ninja-build" ]; then
    ln -sf /usr/bin/ninja /usr/bin/ninja-build
    echo "Created symlink: /usr/bin/ninja-build -> /usr/bin/ninja"
elif [ -f "/usr/bin/ninja-build" ] && [ ! -f "/usr/bin/ninja" ]; then
    ln -sf /usr/bin/ninja-build /usr/bin/ninja
    echo "Created symlink: /usr/bin/ninja -> /usr/bin/ninja-build"
fi

# Create ROCm symlinks if needed
if [ -d "/opt/rocm" ]; then
    # Create CUDA compatibility symlinks
    if [ ! -d "/usr/local/cuda" ]; then
        ln -sf /opt/rocm /usr/local/cuda
        echo "Created symlink: /usr/local/cuda -> /opt/rocm"
    fi

    # Create RCCL symlinks if needed
    if [ -f "/opt/rocm/lib/librccl.so" ] && [ ! -d "/opt/rocm/rccl" ]; then
        mkdir -p /opt/rocm/rccl/lib
        ln -sf /opt/rocm/lib/librccl.so /opt/rocm/rccl/lib/librccl.so
        echo "Created symlink: /opt/rocm/rccl/lib/librccl.so -> /opt/rocm/lib/librccl.so"
    fi
fi

# Create Python module symlinks if needed
PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Flash Attention symlink
if [ -d "$HOME/ml_stack/flash_attn_amd_direct" ] && [ ! -d "$PYTHON_SITE_PACKAGES/flash_attention_amd" ]; then
    ln -sf $HOME/ml_stack/flash_attn_amd_direct "$PYTHON_SITE_PACKAGES/flash_attention_amd"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/flash_attention_amd -> $HOME/ml_stack/flash_attn_amd_direct"
fi

# Megatron-LM symlink
if [ -d "$HOME/megatron/Megatron-LM" ] && [ ! -d "$PYTHON_SITE_PACKAGES/megatron" ]; then
    ln -sf $HOME/megatron/Megatron-LM "$PYTHON_SITE_PACKAGES/megatron"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/megatron -> $HOME/megatron/Megatron-LM"
fi

# ONNX Runtime symlink
if [ -d "$HOME/onnxruntime_build/onnxruntime/build/Linux/Release" ] && [ ! -d "$PYTHON_SITE_PACKAGES/onnxruntime" ]; then
    ln -sf $HOME/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime "$PYTHON_SITE_PACKAGES/onnxruntime"
    echo "Created symlink: $PYTHON_SITE_PACKAGES/onnxruntime -> $HOME/onnxruntime_build/onnxruntime/build/Linux/Release/onnxruntime"
fi
EOF

    # Make the script executable
    chmod +x /usr/local/bin/mlstack-symlinks.sh
    print_success "Created symlink script: /usr/local/bin/mlstack-symlinks.sh"

    # Enable and start the service
    systemctl enable mlstack-symlinks.service
    systemctl start mlstack-symlinks.service
    print_success "Enabled and started mlstack-symlinks service"
}

# Main execution
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (use sudo)${RESET}"
  exit 1
fi

# Run main setup
main_setup

# Create systemd service (this is common to all setups)
create_systemd_service
