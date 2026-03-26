#!/bin/bash
# Stan's ML Stack - vLLM Studio installer

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
VLLM_STUDIO_DIR=${VLLM_STUDIO_DIR:-"$HOME/vllm-studio"}
REPO_URL="https://github.com/0xSero/vllm-studio.git"

# Parse CLI arguments
if declare -f ui_parse_common_args &>/dev/null; then
    ui_parse_common_args DRY_RUN VLLM_STUDIO_DIR "$@"
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
                VLLM_STUDIO_DIR="$(cd "$2" 2>/dev/null && pwd)" || VLLM_STUDIO_DIR="$2"
                case "$VLLM_STUDIO_DIR" in
                    /|/usr|/bin|/sbin|/etc|/var|/boot|/dev|/proc|/sys|/opt/rocm)
                        echo "Error: --dir targets a system directory: $VLLM_STUDIO_DIR" >&2
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

print_header "vLLM Studio Installer"
print_step "Install directory: $VLLM_STUDIO_DIR"

# Dependency checks
missing_deps=()
for dep in git python3; do
    if ! command_exists "$dep"; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -ne 0 ]; then
    print_error "Missing dependencies: ${missing_deps[*]}"
    exit 1
fi

# Detect package manager once (bun preferred, npm fallback)
PKG_MGR=""
if command_exists bun; then
    PKG_MGR="bun"
elif command_exists npm; then
    PKG_MGR="npm"
else
    print_error "Need bun or npm for vLLM Studio dependencies"
    exit 1
fi
print_step "Using package manager: $PKG_MGR"

if ! python3 -c "import vllm" &>/dev/null; then
    print_warning "vLLM is not detected. Install vLLM first for full functionality."
fi

# Clone or update repo (no preserve directories for vLLM Studio)
if declare -f ui_git_clone_or_update &>/dev/null; then
    ui_git_clone_or_update "$VLLM_STUDIO_DIR" "$REPO_URL"
else
    if [ -d "$VLLM_STUDIO_DIR/.git" ]; then
        print_step "Updating vLLM Studio..."
        # Ensure we are on a branch before pulling
        git -C "$VLLM_STUDIO_DIR" remote set-head origin -a || true
        DEFAULT_BRANCH=$(git -C "$VLLM_STUDIO_DIR" symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')
        DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

        execute_command "git -C \"$VLLM_STUDIO_DIR\" checkout \"$DEFAULT_BRANCH\"" "Checking out $DEFAULT_BRANCH branch"
        execute_command "git -C \"$VLLM_STUDIO_DIR\" fetch --all" "Fetching vLLM Studio updates"
        execute_command "git -C \"$VLLM_STUDIO_DIR\" reset --hard \"origin/$DEFAULT_BRANCH\"" "Resetting to origin/$DEFAULT_BRANCH"
    else
        execute_command "git clone \"$REPO_URL\" \"$VLLM_STUDIO_DIR\"" "Cloning vLLM Studio repository"
    fi
fi

# Ensure correct ownership if run with sudo
if declare -f ui_fix_ownership &>/dev/null; then
    ui_fix_ownership "$VLLM_STUDIO_DIR"
else
    if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$VLLM_STUDIO_DIR"
    fi
fi

# Install backend/controller
if [ -d "$VLLM_STUDIO_DIR/controller" ]; then
    if [ "$DRY_RUN" = "false" ]; then
        cd "$VLLM_STUDIO_DIR/controller"
    fi

    execute_command "$PKG_MGR install" "Installing vLLM Studio controller dependencies with $PKG_MGR"

    if [ "$DRY_RUN" = "false" ]; then
        cd "$VLLM_STUDIO_DIR"
    fi
fi

# Build frontend
if [ -d "$VLLM_STUDIO_DIR/frontend" ]; then
    if [ "$DRY_RUN" = "false" ]; then
        cd "$VLLM_STUDIO_DIR/frontend"
    fi

    # Fix broken logs page if missing components
    if [ -f "src/app/logs/page.tsx" ]; then
        if grep -q "LogsView" "src/app/logs/page.tsx" && [ ! -d "src/app/logs/_components" ]; then
            print_warning "Fixing broken logs page (missing upstream components)"
            mv src/app/logs/page.tsx src/app/logs/page.tsx.bak || true
            cat > src/app/logs/page.tsx << 'EOF'
"use client";
export default function LogsPage() {
  return <div className="p-8">Logs viewer is currently unavailable (missing upstream components).</div>;
}
EOF
        fi
    fi

    execute_command "$PKG_MGR install" "Installing vLLM Studio frontend dependencies with $PKG_MGR"
    local build_cmd="run build"
    if [ "$PKG_MGR" = "npm" ]; then build_cmd="run build"; fi
    if execute_command "$PKG_MGR $build_cmd" "Building vLLM Studio frontend with $PKG_MGR"; then
        print_success "Frontend built successfully with $PKG_MGR"
    else
        print_warning "Frontend build failed. You can still run the backend or use '$PKG_MGR dev' in the frontend directory."
    fi
fi

print_section "vLLM Studio Installation Summary"
print_success "Controller installed from $VLLM_STUDIO_DIR"
print_success "Frontend built in $VLLM_STUDIO_DIR/frontend"

# Create a shim for vllm-studio
SHIM_PATH="/usr/local/bin/vllm-studio"
print_step "Creating shim at $SHIM_PATH..."
if [ "$DRY_RUN" = "false" ]; then
    local shim_tmp
    shim_tmp="$(mktemp /tmp/vllm-studio-shim.XXXXXX)"
    cat > "$shim_tmp" << EOF
#!/bin/bash
cd "$VLLM_STUDIO_DIR/controller" && $PKG_MGR run start
EOF
    chmod +x "$shim_tmp"
    sudo mv "$shim_tmp" "$SHIM_PATH" || { print_error "Failed to install shim at $SHIM_PATH"; rm -f "$shim_tmp"; }
fi

print_step "Run the controller: vllm-studio"
print_step "Frontend: $PKG_MGR run dev (inside $VLLM_STUDIO_DIR/frontend)"
print_step "Docs: https://github.com/0xSero/vllm-studio"
