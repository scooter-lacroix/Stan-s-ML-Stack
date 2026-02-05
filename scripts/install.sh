#!/bin/bash
# Rusty Stack - One-line Install Script
#
# This script installs Rusty Stack by cloning the repository,
# installing the Rust toolchain if needed, building the Rusty-Stack TUI,
# and launching the installer.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash
#
# Or with options:
#   curl -fsSL https://.../install.sh | bash -s -- --no-build --branch main

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/scooter-lacroix/Stan-s-ML-Stack.git"
REPO_DIR="${REPO_DIR:-$HOME/Stan-s-ML-Stack}"
BUILD_FLAG=true
BRANCH="${BRANCH:-main}"

# Parse arguments
SKIP_BUILD=false
for arg in "$@"; do
    case $arg in
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --branch=*)
            BRANCH="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Rusty Stack One-Line Install Script"
            echo ""
            echo "Usage:"
            echo "  curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-build    Skip building Rusty-Stack (assumes already built)"
            echo "  --branch=X    Clone specific branch (default: main)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  REPO_DIR      Installation directory (default: \$HOME/Stan-s-ML-Stack)"
            exit 0
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    command -v "$1" &> /dev/null
}

# Header
echo -e "${GREEN}"
cat << 'EOF'
██████╗ ██╗   ██╗███████╗████████╗██╗   ██╗    ███████╗████████╗ █████╗  ██████╗██╗  ██╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝    ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
██████╔╝██║   ██║███████╗   ██║    ╚████╔╝     ███████╗   ██║   ███████║██║     █████╔╝ 
██╔══██╗██║   ██║╚════██║   ██║     ╚██╔╝      ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
██║  ██║╚██████╔╝███████║   ██║      ██║       ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝       ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                                        
EOF
echo -e "${NC}"
log_info "Rusty Stack - AMD GPU ML Environment Installer"
echo ""

# Check dependencies
log_info "Checking dependencies..."

if ! check_command git; then
    log_error "git is not installed. Please install git first."
    log_info "  Ubuntu/Debian: sudo apt install git"
    log_info "  Fedora/RHEL: sudo dnf install git"
    exit 1
fi

# Check for Rust toolchain
if ! check_command rustc || ! check_command cargo; then
    log_warning "Rust toolchain not found. Installing via rustup..."

    if ! check_command curl; then
        log_error "curl is required to install Rust. Please install curl first."
        exit 1
    fi

    # Install Rust via rustup
    log_info "Downloading rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    # Source rust environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
        log_success "Rust toolchain installed"
    else
        log_error "Failed to install Rust toolchain"
        exit 1
    fi
else
    RUST_VERSION=$(rustc --version)
    log_success "Rust toolchain found: $RUST_VERSION"
fi

# Clone or update repository
if [ -d "$REPO_DIR/.git" ]; then
    log_info "Repository exists at $REPO_DIR. Updating..."
    cd "$REPO_DIR"
    git fetch --all
    git reset --hard "origin/$BRANCH"
    log_success "Repository updated"
else
    log_info "Cloning repository from $REPO_URL..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    log_success "Repository cloned to $REPO_DIR"
fi

# Build Rusty-Stack TUI
if [ "$SKIP_BUILD" = false ]; then
    log_info "Building Rusty-Stack TUI..."
    cd "$REPO_DIR/rusty-stack"

    if cargo build --release; then
        log_success "Rusty-Stack TUI built successfully"
    else
        log_error "Failed to build Rusty-Stack TUI"
        log_info "Please check the error messages above."
        log_info "For troubleshooting, see: https://github.com/scooter-lacroix/Stan-s-ML-Stack/blob/main/docs/rusty_stack_guide.md"
        exit 1
    fi
else
    log_info "Skipping build (using existing binary)"
fi

# Launch installer
echo ""
log_success "Installation preparation complete!"
echo ""
log_info "Launching Rusty-Stack TUI..."
echo ""

exec "$REPO_DIR/rusty-stack/target/release/Rusty-Stack"
