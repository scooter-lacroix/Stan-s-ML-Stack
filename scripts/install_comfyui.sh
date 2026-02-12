#!/bin/bash
# Stan's ML Stack - ComfyUI installer

set -euo pipefail

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    "$PYTHON_BIN" "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source common utilities if available
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi

DRY_RUN=${DRY_RUN:-false}
COMFYUI_DIR=${COMFYUI_DIR:-"$HOME/ComfyUI"}
REPO_URL="https://github.com/comfyanonymous/ComfyUI.git"
WEB_PORT=8188

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dir)
            COMFYUI_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--dir <path>]"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

print_header "ComfyUI Installer"
print_step "Install directory: $COMFYUI_DIR"

# Dependency checks
missing_deps=()
for dep in git python3 pip; do
    if ! command_exists "$dep"; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -ne 0 ]; then
    print_error "Missing dependencies: ${missing_deps[*]}"
    exit 1
fi

# Check for PyTorch with ROCm support
if ! python3 -c "import torch" &>/dev/null; then
    print_warning "PyTorch is not detected. ComfyUI requires PyTorch with ROCm support for GPU acceleration."
    print_warning "Install ROCm PyTorch first for optimal performance."
fi

# Clone or update repo
# Check for existing models before proceeding
MODELS_DIR="$COMFYUI_DIR/models"
INPUT_DIR="$COMFYUI_DIR/input"
OUTPUT_DIR="$COMFYUI_DIR/output"
USER_DIR="$COMFYUI_DIR/user"
HAS_MODELS=false

check_dir_has_content() {
    [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]
}

if [ -d "$COMFYUI_DIR/.git" ]; then
    # Check if any user directories have content
    if check_dir_has_content "$MODELS_DIR" || \
       check_dir_has_content "$INPUT_DIR" || \
       check_dir_has_content "$OUTPUT_DIR" || \
       check_dir_has_content "$USER_DIR"; then
        HAS_MODELS=true
        print_warning "Existing user data detected - will preserve during update"
    fi

    print_step "Updating ComfyUI..."
    # Ensure we are on a branch before pulling
    git -C "$COMFYUI_DIR" remote set-head origin -a || true
    DEFAULT_BRANCH=$(git -C "$COMFYUI_DIR" symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')
    DEFAULT_BRANCH=${DEFAULT_BRANCH:-master}

    execute_command "git -C \"$COMFYUI_DIR\" checkout \"$DEFAULT_BRANCH\"" "Checking out $DEFAULT_BRANCH branch"
    execute_command "git -C \"$COMFYUI_DIR\" fetch --all" "Fetching ComfyUI updates"

    if [ "$HAS_MODELS" = true ]; then
        # Preserve user data by stashing it before reset
        execute_command "git -C \"$COMFYUI_DIR\" stash push -u -m \"rusty-stack-preserve-user-data\" -- \"$MODELS_DIR\" \"$INPUT_DIR\" \"$OUTPUT_DIR\" \"$USER_DIR\"" "Stashing user data"
        execute_command "git -C \"$COMFYUI_DIR\" reset --hard \"origin/$DEFAULT_BRANCH\"" "Resetting to origin/$DEFAULT_BRANCH"
        execute_command "git -C \"$COMFYUI_DIR\" stash pop" "Restoring user data"
        print_success "User data preserved during update"
    else
        execute_command "git -C \"$COMFYUI_DIR\" reset --hard \"origin/$DEFAULT_BRANCH\"" "Resetting to origin/$DEFAULT_BRANCH"
    fi
else
    execute_command "git clone \"$REPO_URL\" \"$COMFYUI_DIR\"" "Cloning ComfyUI repository"
fi

# Ensure correct ownership if run with sudo
if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER:$SUDO_USER" "$COMFYUI_DIR"
fi

# Install Python dependencies (excluding torch/torchvision/torchaudio since ROCm version is installed)
if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
    print_step "Installing ComfyUI Python dependencies..."

    if [ "$DRY_RUN" = "false" ]; then
        # Create a filtered requirements file, excluding torch packages
        FILTERED_REQS=$(mktemp)
        grep -v -E '^(torch|torchvision|torchaudio|torchsde|sentencepiece)' "$COMFYUI_DIR/requirements.txt" > "$FILTERED_REQS" || true

        if execute_command "python3 -m pip install -r \"$FILTERED_REQS\"" "Installing filtered requirements"; then
            print_success "Python dependencies installed"
        else
            print_warning "Some dependencies may have failed. Check output above."
        fi
        rm -f "$FILTERED_REQS"
    else
        print_step "[DRY-RUN] Would install Python dependencies (excluding torch/torchvision/torchaudio)"
    fi
else
    print_warning "requirements.txt not found, skipping Python dependency installation"
fi

# Create launcher script
LAUNCHER_DIR="$HOME/.local/bin"
mkdir -p "$LAUNCHER_DIR" 2>/dev/null || true

print_step "Creating ComfyUI launcher script..."

if [ "$DRY_RUN" = "false" ]; then
    LAUNCHER_PATH="$LAUNCHER_DIR/comfy"
    cat > "$LAUNCHER_PATH" << EOF
#!/bin/bash
# ComfyUI launcher for Stan's ML Stack
cd "$COMFYUI_DIR" && \\
    HIP_VISIBLE_DEVICES=0,1 \\
    CUDA_VISIBLE_DEVICES=0,1 \\
    python3 main.py --enable-manager "\$@"
EOF
    chmod +x "$LAUNCHER_PATH"
    print_success "Launcher script created at $LAUNCHER_PATH"

    # Ensure ~/.local/bin is in PATH if not already
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        print_warning "~/.local/bin is not in your PATH. Add it to your shell profile:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
else
    print_step "[DRY-RUN] Would create launcher at $LAUNCHER_DIR/comfy"
fi

# Create systemd user service file (optional)
SERVICE_DIR="$HOME/.config/systemd/user"
if [ "$DRY_RUN" = "false" ] && [ -d "$SERVICE_DIR" ]; then
    print_step "Creating systemd user service file..."
    cat > "$SERVICE_DIR/comfyui.service" << EOF
[Unit]
Description=ComfyUI - Node-based UI for Stable Diffusion
After=network.target

[Service]
Type=simple
WorkingDirectory=$COMFYUI_DIR
Environment="HIP_VISIBLE_DEVICES=0,1"
Environment="CUDA_VISIBLE_DEVICES=0,1"
ExecStart=$PYTHON_BIN $COMFYUI_DIR/main.py --enable-manager
Restart=on-failure

[Install]
WantedBy=default.target
EOF
    print_step "To enable ComfyUI as a service: systemctl --user enable --now comfyui.service"
fi

# Print installation summary
print_section "ComfyUI Installation Summary"
print_success "ComfyUI installed at: $COMFYUI_DIR"
echo
echo -e "${BOLD}Run Commands:${RESET}"
echo "  comfy                    # Start ComfyUI with manager"
echo "  comfy --listen 0.0.0.0   # Allow network access"
echo "  comfy --port 8188        # Custom port"
echo
echo -e "${BOLD}Web Interface:${RESET}"
echo "  http://localhost:$WEB_PORT"
echo
echo -e "${BOLD}GPU Acceleration:${RESET}"
print_success "Uses ROCm for AMD GPU acceleration"
echo
echo -e "${BOLD}Service (optional):${RESET}"
echo "  systemctl --user start comfyui.service   # Start as background service"
echo "  systemctl --user enable comfyui.service  # Enable auto-start"
echo
echo -e "${BOLD}Documentation:${RESET}"
echo "  https://github.com/comfyanonymous/ComfyUI"
echo
