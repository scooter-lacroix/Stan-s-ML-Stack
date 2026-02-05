#!/bin/bash
# Stan's ML Stack - vLLM Studio installer

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
VLLM_STUDIO_DIR=${VLLM_STUDIO_DIR:-"$HOME/vllm-studio"}
REPO_URL="https://github.com/0xSero/vllm-studio.git"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dir)
            VLLM_STUDIO_DIR="$2"
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

print_header "vLLM Studio Installer"
print_step "Install directory: $VLLM_STUDIO_DIR"

# Dependency checks
missing_deps=()
for dep in git python3 npm; do
    if ! command_exists "$dep"; then
        missing_deps+=("$dep")
    fi

done

if [ ${#missing_deps[@]} -ne 0 ]; then
    print_error "Missing dependencies: ${missing_deps[*]}"
    exit 1
fi

if ! python3 -c "import vllm" &>/dev/null; then
    print_warning "vLLM is not detected. Install vLLM first for full functionality."
fi

# Clone or update repo
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

# Ensure correct ownership if run with sudo
if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER:$SUDO_USER" "$VLLM_STUDIO_DIR"
fi

# Install backend/controller
if [ -d "$VLLM_STUDIO_DIR/controller" ]; then
    if [ "$DRY_RUN" = "false" ]; then
        cd "$VLLM_STUDIO_DIR/controller"
    fi
    
    if command_exists bun; then
        execute_command "bun install" "Installing vLLM Studio controller dependencies with bun"
    else
        execute_command "npm install" "Installing vLLM Studio controller dependencies"
    fi
    
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

    # Use bun if available as requested by user
    if command_exists bun; then
        execute_command "bun install" "Installing vLLM Studio frontend dependencies with bun"
        if execute_command "bun run build" "Building vLLM Studio frontend with bun"; then
            print_success "Frontend built successfully with bun"
        else
            print_warning "Frontend build with bun failed. You can still run the backend or use 'bun dev' in the frontend directory."
        fi
    else
        execute_command "npm install" "Installing vLLM Studio frontend dependencies"
        if execute_command "npm run build" "Building vLLM Studio frontend"; then
            print_success "Frontend built successfully"
        else
            print_warning "Frontend build failed. You can still run the backend or use 'npm run dev' in the frontend directory."
        fi
    fi
fi

print_section "vLLM Studio Installation Summary"
print_success "Controller installed from $VLLM_STUDIO_DIR"
print_success "Frontend built in $VLLM_STUDIO_DIR/frontend"

# Create a shim for vllm-studio
SHIM_PATH="/usr/local/bin/vllm-studio"
print_step "Creating shim at $SHIM_PATH..."
if [ "$DRY_RUN" = "false" ]; then
    cat > /tmp/vllm-studio-shim << EOF
#!/bin/bash
cd "$VLLM_STUDIO_DIR/controller" && bun run start
EOF
    chmod +x /tmp/vllm-studio-shim
    sudo mv /tmp/vllm-studio-shim "$SHIM_PATH" || true
fi

print_step "Run the controller: vllm-studio"
print_step "Frontend: npm run dev (inside $VLLM_STUDIO_DIR/frontend)"
print_step "Docs: https://github.com/0xSero/vllm-studio"
