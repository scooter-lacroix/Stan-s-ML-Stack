#!/bin/bash
# Stan's ML Stack - text-generation-webui (oobabooga) installer
# Installs with ROCm-only dependencies, filtering out nvidia/CUDA packages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/lib/installer_guard.sh" ]]; then
    # shellcheck source=lib/installer_guard.sh
    source "$SCRIPT_DIR/lib/installer_guard.sh"
fi

# Source UI installer helper if available
if [[ -f "$SCRIPT_DIR/lib/ui_installer_helper.sh" ]]; then
    # shellcheck source=lib/ui_installer_helper.sh
    source "$SCRIPT_DIR/lib/ui_installer_helper.sh"
fi

PYTHON_BIN="${MLSTACK_PYTHON_BIN:-python3}"

# Wrapper for python3 to ensure we use the correct interpreter
python3() {
    command "$PYTHON_BIN" "$@"
}

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source common utilities if available
if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi

DRY_RUN=${DRY_RUN:-false}
TEXTGEN_DIR=${TEXTGEN_DIR:-"$HOME/text-generation-webui"}
REPO_URL="https://github.com/oobabooga/text-generation-webui.git"
WEB_PORT=7860

# Parse CLI arguments
if declare -f ui_parse_common_args &>/dev/null; then
    ui_parse_common_args DRY_RUN TEXTGEN_DIR "$@"
    local _rc=$?
    if [[ "$_rc" -eq 2 ]]; then exit 0; fi
    if [[ "$_rc" -ne 0 ]]; then exit "$_rc"; fi
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                shift
                ;;
            --dir)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --dir requires a path argument" >&2
                    exit 1
                fi
                if [[ "$2" != /* ]]; then
                    echo "Error: --dir requires an absolute path" >&2
                    exit 1
                fi
                TEXTGEN_DIR="$(cd "$2" 2>/dev/null && pwd)" || TEXTGEN_DIR="$2"
                case "$TEXTGEN_DIR" in
                    /|/usr|/bin|/sbin|/etc|/var|/boot|/dev|/proc|/sys|/opt/rocm)
                        echo "Error: --dir targets a system directory: $TEXTGEN_DIR" >&2
                        exit 1
                        ;;
                esac
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--dry-run] [--dir <path>] [--force]"
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done
fi

print_header "text-generation-webui Installer"
print_step "Install directory: $TEXTGEN_DIR"

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

# Verify PyTorch is installed and is a ROCm build
if ! python3 -c "import torch" &>/dev/null; then
    print_error "PyTorch is not detected. text-generation-webui requires PyTorch with ROCm support."
    print_error "Install ROCm PyTorch first, then re-run this installer."
    exit 1
fi

if ! python3 -c "import torch; assert hasattr(torch.version, 'hip') and torch.version.hip" &>/dev/null; then
    print_warning "PyTorch is installed but does not appear to be a ROCm build (torch.version.hip not set)."
    print_warning "GPU acceleration may not work. Consider installing ROCm PyTorch first."
fi

# Clone or update repo (preserve user data: models, loras, embeddings, presets, characters, training)
if declare -f ui_git_clone_or_update &>/dev/null; then
    ui_git_clone_or_update "$TEXTGEN_DIR" "$REPO_URL" "models" "loras" "embeddings" "presets" "characters" "training"
else
    if [ -d "$TEXTGEN_DIR/.git" ]; then
        print_step "Updating text-generation-webui..."
        git -C "$TEXTGEN_DIR" remote set-head origin -a || true
        DEFAULT_BRANCH=$(git -C "$TEXTGEN_DIR" symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')
        DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

        # Check for user data in preserve directories
        HAS_USER_DATA=false
        for d in models loras embeddings presets characters training; do
            if [ -d "$TEXTGEN_DIR/$d" ] && [ "$(ls -A "$TEXTGEN_DIR/$d" 2>/dev/null)" ]; then
                HAS_USER_DATA=true
                break
            fi
        done

        execute_command "git -C \"$TEXTGEN_DIR\" checkout \"$DEFAULT_BRANCH\"" "Checking out $DEFAULT_BRANCH branch"
        execute_command "git -C \"$TEXTGEN_DIR\" fetch --all" "Fetching text-generation-webui updates"

        if [ "$HAS_USER_DATA" = true ]; then
            execute_command "git -C \"$TEXTGEN_DIR\" stash push -u -m \"rusty-stack-preserve-user-data\" -- models loras embeddings presets characters training" "Stashing user data"
            execute_command "git -C \"$TEXTGEN_DIR\" reset --hard \"origin/$DEFAULT_BRANCH\"" "Resetting to origin/$DEFAULT_BRANCH"
            execute_command "git -C \"$TEXTGEN_DIR\" stash pop" "Restoring user data"
            print_success "User data preserved during update"
        else
            execute_command "git -C \"$TEXTGEN_DIR\" reset --hard \"origin/$DEFAULT_BRANCH\"" "Resetting to origin/$DEFAULT_BRANCH"
        fi
    else
        execute_command "git clone \"$REPO_URL\" \"$TEXTGEN_DIR\"" "Cloning text-generation-webui repository"
    fi
fi

# Ensure correct ownership if run with sudo
if declare -f ui_fix_ownership &>/dev/null; then
    ui_fix_ownership "$TEXTGEN_DIR"
else
    if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$TEXTGEN_DIR"
    fi
fi

# Install Python dependencies with ROCm-only filtering
# Exclude: nvidia-*, cuda*, tensorrt*, triton[versioned], xformers, flash-attn, torch/torchvision/torchaudio
if [ -f "$TEXTGEN_DIR/requirements.txt" ]; then
    print_step "Installing text-generation-webui Python dependencies (ROCm-only)..."

    if [ "$DRY_RUN" = "false" ]; then
        FILTERED_REQS=$(mktemp)
        grep -v -E '^(nvidia-|cuda|tensorrt|triton([=<>!\[]|$)|xformers|flash-attn|torch([=<>! ]|$)|torchvision|torchaudio)' "$TEXTGEN_DIR/requirements.txt" > "$FILTERED_REQS" || true

        # Also check for AMD-specific requirements file
        if [ -f "$TEXTGEN_DIR/requirements_amd.txt" ]; then
            print_step "Found requirements_amd.txt — installing AMD-specific dependencies"
            if execute_command "python3 -m pip install -r \"$TEXTGEN_DIR/requirements_amd.txt\"" "Installing AMD-specific requirements"; then
                print_success "AMD requirements installed"
            else
                print_warning "Some AMD dependencies may have failed. Check output above."
            fi
        fi

        if execute_command "python3 -m pip install -r \"$FILTERED_REQS\"" "Installing filtered requirements (ROCm-only)"; then
            print_success "Python dependencies installed (nvidia/CUDA packages excluded)"
        else
            print_warning "Some dependencies may have failed. Check output above."
        fi
        rm -f "$FILTERED_REQS"
    else
        print_step "[DRY-RUN] Would install Python dependencies (excluding nvidia/CUDA packages)"
    fi
else
    print_warning "requirements.txt not found, skipping Python dependency installation"
fi

# Detect GPU count for device selection
GPU_DEVICES=""
if declare -f ui_detect_gpu_devices &>/dev/null; then
    GPU_DEVICES=$(ui_detect_gpu_devices) || true
else
    if command -v rocm-smi &>/dev/null; then
        GPU_COUNT=$(rocm-smi --showproductname | grep -c "GPU\[" || true)
        if [ "$GPU_COUNT" -gt 0 ]; then
            GPU_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
            print_step "Detected $GPU_COUNT AMD GPU(s)"
        fi
    fi
fi

# Default to 0,1 if GPU detection failed
if [ -z "$GPU_DEVICES" ]; then
    GPU_DEVICES="0,1"
    print_warning "Could not detect GPU count, using default: $GPU_DEVICES"
fi

# Create launcher script
LAUNCHER_DIR="$HOME/.local/bin"
if [ "$DRY_RUN" = "false" ]; then
    if declare -f ui_create_launcher_shim &>/dev/null; then
        ui_create_launcher_shim "textgen" "$LAUNCHER_DIR" \
            "cd \"$TEXTGEN_DIR\" && HIP_VISIBLE_DEVICES=$GPU_DEVICES CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3 server.py --chat" \
            "text-generation-webui launcher for Stan's ML Stack"
        print_success "Launcher script created at $LAUNCHER_DIR/textgen"
    else
        mkdir -p "$LAUNCHER_DIR" 2>/dev/null || true
        LAUNCHER_PATH="$LAUNCHER_DIR/textgen"
        cat > "$LAUNCHER_PATH" << EOF
#!/bin/bash
# text-generation-webui launcher for Stan's ML Stack
# Auto-detected GPU devices: $GPU_DEVICES
cd "$TEXTGEN_DIR" && \\
    HIP_VISIBLE_DEVICES=$GPU_DEVICES \\
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES \\
    python3 server.py --chat "\$@"
EOF
        chmod +x "$LAUNCHER_PATH"
        print_success "Launcher script created at $LAUNCHER_PATH"
    fi

    # Ensure ~/.local/bin is in PATH if not already
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        print_warning "~/.local/bin is not in your PATH. Add it to your shell profile:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
else
    print_step "[DRY-RUN] Would create launcher at $LAUNCHER_DIR/textgen"
fi

# Create systemd user service file (optional)
SERVICE_DIR="$HOME/.config/systemd/user"
if [ "$DRY_RUN" = "false" ] && [ -d "$SERVICE_DIR" ]; then
    print_step "Creating systemd user service file..."
    if declare -f ui_create_systemd_service &>/dev/null; then
        ui_create_systemd_service "textgen" "$TEXTGEN_DIR" \
            "$PYTHON_BIN $TEXTGEN_DIR/server.py --chat" \
            "HIP_VISIBLE_DEVICES=$GPU_DEVICES" \
            "CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
    else
        cat > "$SERVICE_DIR/textgen.service" << EOF
[Unit]
Description=text-generation-webui - LLM chat interface with ROCm support
After=network.target

[Service]
Type=simple
WorkingDirectory="${TEXTGEN_DIR}"
Environment="HIP_VISIBLE_DEVICES=${GPU_DEVICES}"
Environment="CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
ExecStart=${PYTHON_BIN} ${TEXTGEN_DIR}/server.py --chat
Restart=on-failure

[Install]
WantedBy=default.target
EOF
    fi
    print_step "To enable text-generation-webui as a service: systemctl --user enable --now textgen.service"
fi

# Print installation summary
if declare -f ui_print_summary &>/dev/null; then
    COMMANDS_ARRAY=(
        "textgen                    # Start text-generation-webui in chat mode"
        "textgen --notebook         # Start in notebook mode"
        "web: http://localhost:$WEB_PORT"
    )
    ui_print_summary "text-generation-webui" "$TEXTGEN_DIR"
    echo
    echo -e "${BOLD}GPU Acceleration:${RESET}"
    print_success "Uses ROCm for AMD GPU acceleration (HIP_VISIBLE_DEVICES=$GPU_DEVICES)"
    echo
    echo -e "${BOLD}Service (optional):${RESET}"
    echo "  systemctl --user start textgen.service   # Start as background service"
    echo "  systemctl --user enable textgen.service  # Enable auto-start"
    echo
    echo -e "${BOLD}Documentation:${RESET}"
    echo "  https://github.com/oobabooga/text-generation-webui"
else
    print_section "text-generation-webui Installation Summary"
    print_success "text-generation-webui installed at: $TEXTGEN_DIR"
    echo
    echo -e "${BOLD}Run Commands:${RESET}"
    echo "  textgen                    # Start text-generation-webui in chat mode"
    echo "  textgen --notebook         # Start in notebook mode"
    echo
    echo -e "${BOLD}Web Interface:${RESET}"
    echo "  http://localhost:$WEB_PORT"
    echo
    echo -e "${BOLD}GPU Acceleration:${RESET}"
    print_success "Uses ROCm for AMD GPU acceleration (HIP_VISIBLE_DEVICES=$GPU_DEVICES)"
    echo
    echo -e "${BOLD}Service (optional):${RESET}"
    echo "  systemctl --user start textgen.service   # Start as background service"
    echo "  systemctl --user enable textgen.service  # Enable auto-start"
    echo
    echo -e "${BOLD}Documentation:${RESET}"
    echo "  https://github.com/oobabooga/text-generation-webui"
fi
