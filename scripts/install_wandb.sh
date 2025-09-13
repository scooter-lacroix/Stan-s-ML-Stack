#!/bin/bash
#
# Weights & Biases (WandB) Installation Script for ML Stack
#
# This script installs Weights & Biases (WandB), a tool for experiment tracking,
# model visualization, and collaboration.
#
# Enhanced with modern installation standards including ROCm support,
# multiple installation methods, and comprehensive error handling.
#

# ASCII Art Banner
cat << "EOF"
██╗    ██╗ █████╗ ███╗   ██╗██████╗ ██████╗
██║    ██║██╔══██╗████╗  ██║██╔══██╗██╔══██╗
██║ █╗ ██║███████║██╔██╗ ██║██║  ██║██████╔╝
██║███╗██║██╔══██║██║╚██╗██║██║  ██║██╔══██╗
╚███╔███╔╝██║  ██║██║ ╚████║██████╔╝██████╔╝
 ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═════╝
EOF
echo

# Check if terminal supports colors
if [ -t 1 ]; then
    # Check if NO_COLOR environment variable is set
    if [ -z "$NO_COLOR" ]; then
        # Terminal supports colors
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
    else
        # NO_COLOR is set, don't use colors
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        MAGENTA=''
        CYAN=''
        BOLD=''
        UNDERLINE=''
        BLINK=''
        REVERSE=''
        RESET=''
    fi
else
    # Not a terminal, don't use colors
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    UNDERLINE=''
    BLINK=''
    REVERSE=''
    RESET=''
fi

# Function definitions
print_header() {
    echo
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║                                                         ║"
    echo "║               === $1 ===               ║"
    echo "║                                                         ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    echo
}

print_section() {
    echo
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ $1"
    echo "└─────────────────────────────────────────────────────────┘"
}

print_step() {
    echo "➤ $1"
}

print_success() {
    echo "✓ $1"
}

print_warning() {
    echo "⚠ $1"
}

print_error() {
    echo "✗ $1"
}

# Function to print a clean separator line
print_separator() {
    echo "───────────────────────────────────────────────────────────"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" &>/dev/null
}

# Function to detect package manager
detect_package_manager() {
    if command_exists dnf; then
        echo "dnf"
    elif command_exists apt-get; then
        echo "apt"
    elif command_exists yum; then
        echo "yum"
    elif command_exists pacman; then
        echo "pacman"
    elif command_exists zypper; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Function to use uv or pip for Python packages
install_python_package() {
    local package="$1"
    shift
    local extra_args="$@"

    if command_exists uv; then
        print_step "Installing $package with uv..."
        uv pip install --python $(which python3) $extra_args "$package"
    else
        print_step "Installing $package with pip..."
        python3 -m pip install $extra_args "$package"
    fi
}

# Function to show environment variables
show_env() {
    # Set up minimal ROCm environment for WandB GPU operations
    HSA_TOOLS_LIB=0
    HSA_OVERRIDE_GFX_VERSION=11.0.0
    PYTORCH_ROCM_ARCH="gfx1100"
    ROCM_PATH="/opt/rocm"
    PATH="/opt/rocm/bin:$PATH"
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

    # Check if rocprofiler library exists and update HSA_TOOLS_LIB accordingly
    if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
        HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
    fi

    # Handle PYTORCH_CUDA_ALLOC_CONF conversion
    if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
        PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    fi

    echo "export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\""
    echo "export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\""
    if [ -n "$PYTORCH_ALLOC_CONF" ]; then
        echo "export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\""
    fi
    echo "export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\""
    echo "export ROCM_PATH=\"$ROCM_PATH\""
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
}

# Function to detect ROCm and GPU architecture
detect_rocm() {
    print_section "ROCm Detection and Configuration"

    if command_exists rocminfo; then
        print_success "rocminfo found"

        # Set up ROCm environment variables
        print_step "Setting up ROCm environment variables..."
        export HSA_OVERRIDE_GFX_VERSION=11.0.0
        export PYTORCH_ROCM_ARCH="gfx1100"
        export ROCM_PATH="/opt/rocm"
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

        # Set HSA_TOOLS_LIB if rocprofiler library exists
        if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
            export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
            print_step "ROCm profiler library found and configured"
        else
            # Check if we can install rocprofiler
            if command_exists apt-get && apt-cache show rocprofiler >/dev/null 2>&1; then
                print_step "Installing rocprofiler for HSA tools support..."
                sudo apt-get update && sudo apt-get install -y rocprofiler
                if [ -f "/opt/rocm/lib/librocprofiler-sdk-tool.so" ]; then
                    export HSA_TOOLS_LIB="/opt/rocm/lib/librocprofiler-sdk-tool.so"
                    print_success "ROCm profiler installed and configured"
                else
                    export HSA_TOOLS_LIB=0
                    print_warning "ROCm profiler installation failed, disabling HSA tools"
                fi
            else
                export HSA_TOOLS_LIB=0
                print_warning "ROCm profiler library not found, disabling HSA tools (this may cause warnings but won't affect functionality)"
            fi
        fi

        # Fix deprecated PYTORCH_CUDA_ALLOC_CONF warning
        if [ -n "$PYTORCH_CUDA_ALLOC_CONF" ]; then
            export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
            unset PYTORCH_CUDA_ALLOC_CONF
            print_step "Converted deprecated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF"
        fi

        print_success "ROCm environment variables configured"
        return 0
    else
        print_step "rocminfo not found in PATH, checking for ROCm installation..."
        if [ -d "/opt/rocm" ] || ls /opt/rocm-* >/dev/null 2>&1; then
            print_step "ROCm directory found, attempting to install rocminfo..."
            package_manager=$(detect_package_manager)
            case $package_manager in
                apt)
                    sudo apt update && sudo apt install -y rocminfo
                    ;;
                dnf)
                    sudo dnf install -y rocminfo
                    ;;
                yum)
                    sudo yum install -y rocminfo
                    ;;
                pacman)
                    sudo pacman -S rocminfo
                    ;;
                zypper)
                    sudo zypper install -y rocminfo
                    ;;
                *)
                    print_error "Unsupported package manager: $package_manager"
                    return 1
                    ;;
            esac
            if command_exists rocminfo; then
                print_success "Installed rocminfo"
                # Recursively call detect_rocm now that rocminfo is installed
                detect_rocm
                return $?
            else
                print_error "Failed to install rocminfo"
                return 1
            fi
        else
            print_warning "ROCm is not installed. WandB will work without GPU acceleration."
            return 1
        fi
    fi
}

# Function to create WandB test script
create_test_script() {
    local test_script="$1"
    cat > "$test_script" << 'EOF'
#!/usr/bin/env python3
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

def test_wandb_logging():
    """Test basic WandB logging."""
    print("=== Testing WandB Logging ===")

    # Initialize WandB
    run = wandb.init(project="ml_stack_test", name="test_run", mode="offline")

    # Log some metrics
    for i in range(10):
        wandb.log({
            "loss": 1.0 - i * 0.1,
            "accuracy": i * 0.1,
            "step": i
        })

    # Log a table
    data = [[i, i**2, i**3] for i in range(10)]
    table = wandb.Table(data=data, columns=["x", "x^2", "x^3"])
    wandb.log({"table": table})

    # Log a plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    data = [[x[i], y[i]] for i in range(len(x))]
    table = wandb.Table(data=data, columns=["x", "sin(x)"])
    wandb.log({"plot": wandb.plot.line(table, "x", "sin(x)", title="Sin Wave")})

    # Finish the run
    run.finish()

    print("WandB logging test completed successfully!")
    return True

def test_wandb_model_tracking():
    """Test WandB model tracking."""
    print("\n=== Testing WandB Model Tracking ===")

    # Initialize WandB
    run = wandb.init(project="ml_stack_test", name="model_tracking", mode="offline")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleModel()

    # Watch the model
    wandb.watch(model, log="all")

    # Create some data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # Train the model
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "loss": loss.item()
        })

    # Save the model
    torch.save(model.state_dict(), "simple_model.pt")
    wandb.save("simple_model.pt")

    # Finish the run
    run.finish()

    print("WandB model tracking test completed successfully!")
    return True

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test WandB")
    parser.add_argument("--test-logging", action="store_true", help="Run logging test")
    parser.add_argument("--test-model", action="store_true", help="Run model tracking test")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If no specific test is selected, run all tests
    if not (args.test_logging or args.test_model) or args.all:
        args.test_logging = args.test_model = True

    # Run tests
    results = []

    if args.test_logging:
        try:
            result = test_wandb_logging()
            results.append(("Logging Test", result))
        except Exception as e:
            print(f"Error in logging test: {e}")
            results.append(("Logging Test", False))

    if args.test_model:
        try:
            result = test_wandb_model_tracking()
            results.append(("Model Tracking Test", result))
        except Exception as e:
            print(f"Error in model tracking test: {e}")
            results.append(("Model Tracking Test", False))

    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the logs.")

if __name__ == "__main__":
    main()
EOF
}

# Function to create WandB documentation
create_documentation() {
    local docs_file="$1"
    cat > "$docs_file" << 'EOF'
# Weights & Biases (WandB) Guide

## Overview

Weights & Biases (WandB) is a tool for experiment tracking, model visualization, and collaboration. It helps you track your machine learning experiments, visualize model performance, and share your results with your team.

## Features

- **Experiment Tracking**: Track hyperparameters, metrics, and artifacts for your machine learning experiments.
- **Visualization**: Visualize model performance with interactive plots and dashboards.
- **Collaboration**: Share your results with your team and collaborate on machine learning projects.
- **Model Management**: Version and manage your models and datasets.
- **Reports**: Create and share reports with your team.

## Usage

### Basic Logging

```python
import wandb

# Initialize WandB
wandb.init(project="my_project", name="my_run")

# Log metrics
for i in range(10):
    wandb.log({
        "loss": 1.0 - i * 0.1,
        "accuracy": i * 0.1,
        "step": i
    })

# Finish the run
wandb.finish()
```

### Model Tracking

```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize WandB
wandb.init(project="my_project", name="model_tracking")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()

# Watch the model
wandb.watch(model, log="all")

# Train the model
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    wandb.log({
        "epoch": epoch,
        "loss": loss.item()
    })

# Save the model
torch.save(model.state_dict(), "simple_model.pt")
wandb.save("simple_model.pt")

# Finish the run
wandb.finish()
```

## Resources

- [WandB Documentation](https://docs.wandb.ai/)
- [WandB GitHub Repository](https://github.com/wandb/wandb)
- [WandB Python API Reference](https://docs.wandb.ai/ref/python)
EOF
}

# Main installation function
install_wandb() {
    print_header "WandB Installation"

    # Parse command line arguments
    DRY_RUN=false
    FORCE=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                print_step "Dry run mode enabled - no changes will be made"
                shift
                ;;
            --force)
                FORCE=true
                print_step "Force reinstall enabled"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Check for required dependencies
    print_section "Checking Dependencies"
    DEPS=("python3" "pip")
    MISSING_DEPS=()

    for dep in "${DEPS[@]}"; do
        if ! command_exists $dep; then
            MISSING_DEPS+=("$dep")
        fi
    done

    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${MISSING_DEPS[*]}"
        print_step "Please install them and run this script again."
        return 1
    fi

    print_success "All dependencies found"

    # Detect ROCm for GPU support
    detect_rocm

    # Check if WandB is already installed
    print_section "Checking Existing Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${WANDB_VENV_PYTHON:-python3}

    if package_installed "wandb"; then
        wandb_version=$($PYTHON_CMD -c "import wandb; print(wandb.__version__)" 2>/dev/null)
        print_success "WandB is already installed (version: $wandb_version)"

        # Check if --force flag is provided
        if [[ "$FORCE" == "true" ]]; then
            print_warning "Force reinstall requested - proceeding with reinstallation"
        else
            print_step "WandB installation is complete. Use --force to reinstall anyway."
            return 0
        fi
    fi

    # Check if uv is installed
    print_section "Package Manager Setup"

    if ! command_exists uv; then
        print_step "Installing uv package manager..."
        python3 -m pip install uv

        # Add uv to PATH if it was installed in a user directory
        if [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi

        # Add uv to PATH if it was installed via cargo
        if [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        if ! command_exists uv; then
            print_error "Failed to install uv package manager"
            print_step "Falling back to pip"
        else
            print_success "Installed uv package manager"
        fi
    else
        print_success "uv package manager is already installed"
    fi

    # Ask user for installation preference
    echo
    echo -e "${CYAN}${BOLD}WandB Installation Options:${RESET}"
    echo "1) Global installation (recommended for system-wide use)"
    echo "2) Virtual environment (isolated installation)"
    echo "3) Auto-detect (try global, fallback to venv if needed)"
    echo
    read -p "Choose installation method (1-3) [3]: " INSTALL_CHOICE
    INSTALL_CHOICE=${INSTALL_CHOICE:-3}

    case $INSTALL_CHOICE in
        1)
            INSTALL_METHOD="global"
            print_step "Using global installation method"
            ;;
        2)
            INSTALL_METHOD="venv"
            print_step "Using virtual environment method"
            ;;
        3|*)
            INSTALL_METHOD="auto"
            print_step "Using auto-detect method"
            ;;
    esac

    # Create a function to handle uv commands properly with venv fallback
    uv_pip_install() {
        local args="$@"

        if [[ "$DRY_RUN" == "true" ]]; then
            print_step "[DRY RUN] Would install with: uv pip install $args"
            return 0
        fi

        # Check if uv is available as a command
        if command_exists uv; then
            case $INSTALL_METHOD in
                "global")
                    print_step "Installing globally with pip..."
                    python3 -m pip install --break-system-packages $args
                    WANDB_VENV_PYTHON=""
                    ;;
                "venv")
                    print_step "Creating uv virtual environment..."
                    VENV_DIR="./wandb_venv"
                    if [ ! -d "$VENV_DIR" ]; then
                        uv venv "$VENV_DIR"
                    fi
                    source "$VENV_DIR/bin/activate"
                    print_step "Installing in virtual environment..."
                    uv pip install $args
                    WANDB_VENV_PYTHON="$VENV_DIR/bin/python"
                    print_success "Installed in virtual environment: $VENV_DIR"
                    ;;
                "auto")
                    # Try global install first
                    print_step "Attempting global installation with uv..."
                    local install_output
                    install_output=$(uv pip install --python $(which python3) $args 2>&1)
                    local install_exit_code=$?

                    if echo "$install_output" | grep -q "externally managed"; then
                        print_warning "Global installation failed due to externally managed environment"
                        print_step "Creating uv virtual environment for installation..."

                        # Create uv venv in project directory
                        VENV_DIR="./wandb_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        WANDB_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    elif [ $install_exit_code -eq 0 ]; then
                        print_success "Global installation successful"
                        WANDB_VENV_PYTHON=""
                    else
                        print_error "Global installation failed with unknown error:"
                        echo "$install_output"
                        print_step "Falling back to virtual environment..."

                        # Create uv venv in project directory
                        VENV_DIR="./wandb_venv"
                        if [ ! -d "$VENV_DIR" ]; then
                            uv venv "$VENV_DIR"
                        fi

                        # Activate venv and install
                        source "$VENV_DIR/bin/activate"
                        print_step "Installing in virtual environment..."
                        uv pip install $args

                        # Store venv path for verification
                        WANDB_VENV_PYTHON="$VENV_DIR/bin/python"
                        print_success "Installed in virtual environment: $VENV_DIR"
                    fi
                    ;;
            esac
        else
            # Fall back to pip
            print_step "Installing with pip..."
            python3 -m pip install $args
            WANDB_VENV_PYTHON=""
        fi
    }

    # Install WandB
    print_section "Installing WandB"
    uv_pip_install wandb

    # Verify installation
    print_section "Verifying Installation"

    # Use venv Python if available, otherwise system python3
    PYTHON_CMD=${WANDB_VENV_PYTHON:-python3}

    if $PYTHON_CMD -c "import wandb" &>/dev/null; then
        wandb_version=$($PYTHON_CMD -c "import wandb; print(wandb.__version__)" 2>/dev/null)
        print_success "WandB is installed (version: $wandb_version)"

        # Test basic functionality
        print_step "Testing WandB basic functionality..."
        if $PYTHON_CMD -c "import wandb; print('WandB import successful')" &>/dev/null; then
            print_success "WandB basic functionality verified"
        else
            print_warning "WandB basic functionality test failed"
        fi
    else
        print_error "WandB installation failed"
        return 1
    fi

    # Create installation directory
    INSTALL_DIR="$HOME/ml_stack/wandb"
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p $INSTALL_DIR
    else
        print_step "[DRY RUN] Would create directory: $INSTALL_DIR"
    fi

    # Create test script
    TEST_SCRIPT="$INSTALL_DIR/test_wandb.py"
    if [[ "$DRY_RUN" == "false" ]]; then
        create_test_script "$TEST_SCRIPT"
        print_success "Created test script at $TEST_SCRIPT"
    else
        print_step "[DRY RUN] Would create test script at $TEST_SCRIPT"
    fi

    # Create documentation directory
    DOCS_DIR="$HOME/Desktop/ml_stack_extensions/docs"
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p $DOCS_DIR
    else
        print_step "[DRY RUN] Would create docs directory: $DOCS_DIR"
    fi

    # Create documentation
    DOCS_FILE="$DOCS_DIR/wandb_guide.md"
    if [[ "$DRY_RUN" == "false" ]]; then
        create_documentation "$DOCS_FILE"
        print_success "Created documentation at $DOCS_FILE"
    else
        print_step "[DRY RUN] Would create documentation at $DOCS_FILE"
    fi

    # Show completion message
    if [[ "$DRY_RUN" == "false" ]]; then
        clear
        cat << "EOF"

        ╔═════════════════════════════════════════════════════════╗
        ║                                                         ║
        ║  ██╗    ██╗ █████╗ ███╗   ██╗██████╗ ██████╗           ║
        ║  ██║    ██║██╔══██╗████╗  ██║██╔══██╗██╔══██╗          ║
        ║  ██║ █╗ ██║███████║██╔██╗ ██║██║  ██║██████╔╝          ║
        ║  ██║███╗██║██╔══██║██║╚██╗██║██║  ██║██╔══██╗          ║
        ║  ╚███╔███╔╝██║  ██║██║ ╚████║██████╔╝██████╔╝          ║
        ║   ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═════╝           ║
        ║                                                         ║
        ║  Installation Completed Successfully!                   ║
        ║                                                         ║
        ║  WandB is now ready for experiment tracking and         ║
        ║  model visualization.                                   ║
        ║                                                         ║
        ╚═════════════════════════════════════════════════════════╝

EOF

        print_success "WandB installation completed successfully"

        # Provide usage examples
        echo
        echo -e "${CYAN}${BOLD}Quick Start Examples:${RESET}"
        if [ -n "$WANDB_VENV_PYTHON" ]; then
            echo -e "${GREEN}source ./wandb_venv/bin/activate${RESET}"
            echo -e "${GREEN}python -c \"import wandb; print('WandB version:', wandb.__version__)\"${RESET}"
        else
            echo -e "${GREEN}python3 -c \"import wandb; print('WandB version:', wandb.__version__)\"${RESET}"
        fi
        echo -e "${GREEN}python3 $TEST_SCRIPT --all${RESET}"
        echo
        echo -e "${YELLOW}${BOLD}Note:${RESET} ${YELLOW}ROCm environment variables are set for GPU acceleration.${RESET}"
        echo -e "${YELLOW}For future sessions, you may need to run:${RESET}"

        # Output the actual environment variables that were set
        echo -e "${GREEN}export HSA_TOOLS_LIB=\"$HSA_TOOLS_LIB\"${RESET}"
        echo -e "${GREEN}export HSA_OVERRIDE_GFX_VERSION=\"$HSA_OVERRIDE_GFX_VERSION\"${RESET}"
        if [ -n "$PYTORCH_ALLOC_CONF" ]; then
            echo -e "${GREEN}export PYTORCH_ALLOC_CONF=\"$PYTORCH_ALLOC_CONF\"${RESET}"
        fi
        echo -e "${GREEN}export PYTORCH_ROCM_ARCH=\"$PYTORCH_ROCM_ARCH\"${RESET}"
        echo -e "${GREEN}export ROCM_PATH=\"$ROCM_PATH\"${RESET}"
        echo -e "${GREEN}export PATH=\"$PATH\"${RESET}"
        echo -e "${GREEN}export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"${RESET}"
        echo
        echo -e "${CYAN}${BOLD}To apply these settings to your current shell, run:${RESET}"
        echo -e "${GREEN}eval \"\$(./install_wandb.sh --show-env)\"${RESET}"
        echo
    else
        print_success "Dry run completed - no changes were made"
    fi

    return 0
}

# Check for --show-env option
if [[ "$1" == "--show-env" ]]; then
    show_env
    exit 0
fi

# Run the installation function with all script arguments
install_wandb "$@"
