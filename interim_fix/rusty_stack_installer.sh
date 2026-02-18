#!/usr/bin/env bash
# Rusty Stack - One-line Install Script (Multi-Shell & Multi-Distro)
#
# This script installs Rusty Stack by cloning the repository,
# installing the Rust toolchain if needed, building the Rusty-Stack TUI,
# and launching the installer.
#
# Supports: Bash, Zsh, Fish shells
# Supports: Arch, Debian, Ubuntu, Fedora, RHEL, openSUSE
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
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/scooter-lacroix/Stan-s-ML-Stack.git"
REPO_DIR="${REPO_DIR:-$HOME/Stan-s-ML-Stack}"
BUILD_FLAG=true
BRANCH="${BRANCH:-main}"

# System detection variables
USER_SHELL="bash"
SHELL_RC=""
SHELL_PROFILE=""
DISTRO_ID="unknown"
DISTRO_NAME="Unknown"
DISTRO_VERSION="unknown"
PKG_MANAGER="unknown"
IS_ARCH=false
IS_DEBIAN=false
IS_FEDORA=false

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
            echo ""
            echo "Supported Systems:"
            echo "  Shells:        bash, zsh, fish"
            echo "  Distributions: Arch Linux, Debian, Ubuntu, Fedora, RHEL, openSUSE"
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

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

detect_shell() {
    # Detect the user's default shell
    USER_SHELL=$(basename "$SHELL" 2>/dev/null || echo "bash")
    
    log_info "Detected shell: $USER_SHELL"
    
    # Determine shell config files
    case "$USER_SHELL" in
        bash)
            SHELL_RC="$HOME/.bashrc"
            SHELL_PROFILE="$HOME/.bash_profile"
            ;;
        zsh)
            SHELL_RC="$HOME/.zshrc"
            SHELL_PROFILE="$HOME/.zprofile"
            ;;
        fish)
            SHELL_RC="$HOME/.config/fish/config.fish"
            SHELL_PROFILE="$HOME/.config/fish/config.fish"
            # Create fish config directory if it doesn't exist
            mkdir -p "$HOME/.config/fish"
            ;;
        *)
            SHELL_RC="$HOME/.profile"
            SHELL_PROFILE="$HOME/.profile"
            ;;
    esac
}

detect_distribution() {
    # Detect Linux distribution
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_NAME="${NAME:-Unknown}"
        DISTRO_VERSION="${VERSION_ID:-unknown}"
    elif [[ -f /etc/arch-release ]]; then
        DISTRO_ID="arch"
        DISTRO_NAME="Arch Linux"
        DISTRO_VERSION="rolling"
    else
        DISTRO_ID="unknown"
        DISTRO_NAME="Unknown"
        DISTRO_VERSION="unknown"
    fi
    
    log_info "Detected distribution: $DISTRO_NAME ($DISTRO_ID) $DISTRO_VERSION"
    
    # Detect package manager and set distribution flags
    if check_command pacman; then
        PKG_MANAGER="pacman"
        IS_ARCH=true
        log_info "Package manager: pacman (Arch Linux)"
    elif check_command apt || check_command apt-get; then
        PKG_MANAGER="apt"
        IS_DEBIAN=true
        log_info "Package manager: apt (Debian/Ubuntu)"
    elif check_command dnf; then
        PKG_MANAGER="dnf"
        IS_FEDORA=true
        log_info "Package manager: dnf (Fedora)"
    elif check_command yum; then
        PKG_MANAGER="yum"
        IS_FEDORA=true
        log_info "Package manager: yum (RHEL/CentOS)"
    elif check_command zypper; then
        PKG_MANAGER="zypper"
        log_info "Package manager: zypper (openSUSE)"
    else
        PKG_MANAGER="unknown"
        log_warning "No recognized package manager found"
    fi
}

install_dependency() {
    local dep_name="$1"
    local pkg_name="$2"
    
    log_info "Installing $dep_name..."
    
    case "$PKG_MANAGER" in
        pacman)
            if sudo pacman -Sy --noconfirm "$pkg_name"; then
                log_success "$dep_name installed"
                return 0
            fi
            ;;
        apt)
            sudo apt-get update -qq
            if sudo apt-get install -y "$pkg_name"; then
                log_success "$dep_name installed"
                return 0
            fi
            ;;
        dnf)
            if sudo dnf install -y "$pkg_name"; then
                log_success "$dep_name installed"
                return 0
            fi
            ;;
        yum)
            if sudo yum install -y "$pkg_name"; then
                log_success "$dep_name installed"
                return 0
            fi
            ;;
        zypper)
            if sudo zypper install -y "$pkg_name"; then
                log_success "$dep_name installed"
                return 0
            fi
            ;;
        *)
            log_error "Cannot install $dep_name: unknown package manager"
            return 1
            ;;
    esac
    
    log_error "Failed to install $dep_name"
    return 1
}

add_rust_to_shell_config() {
    log_info "Configuring Rust environment for $USER_SHELL..."
    
    case "$USER_SHELL" in
        fish)
            # Fish shell uses different syntax
            if [[ -f "$SHELL_RC" ]]; then
                # Check if already configured
                if ! grep -q "set -gx PATH.*cargo/bin" "$SHELL_RC" 2>/dev/null; then
                    echo "" >> "$SHELL_RC"
                    echo "# Rust cargo environment" >> "$SHELL_RC"
                    echo "set -gx PATH \$HOME/.cargo/bin \$PATH" >> "$SHELL_RC"
                    log_success "Added Rust to $SHELL_RC"
                else
                    log_info "Rust already configured in $SHELL_RC"
                fi
            fi
            ;;
        *)
            # Bash/Zsh/others
            if [[ -f "$SHELL_RC" ]]; then
                # Check if already configured
                if ! grep -q "cargo/env" "$SHELL_RC" 2>/dev/null; then
                    echo "" >> "$SHELL_RC"
                    echo "# Rust cargo environment" >> "$SHELL_RC"
                    echo ". \"\$HOME/.cargo/env\"" >> "$SHELL_RC"
                    log_success "Added Rust to $SHELL_RC"
                else
                    log_info "Rust already configured in $SHELL_RC"
                fi
            fi
            ;;
    esac
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Detect system first
detect_shell
detect_distribution

# Header
echo ""
echo -e "${GREEN}"
cat << 'EOF'
█████╗ ██╗   ██╗███████╗████████╗██╗   ██╗    ███████╗████████╗ █████╗  ██████╗██╗  ██╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝    ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
███████║██║   ██║███████╗   ██║    ╚████╔╝     ███████╗   ██║   ███████║██║     █████╔╝ 
██╔══██║██║   ██║╚════██║   ██║     ╚██╔╝      ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
██║  ██║╚██████╔╝███████║   ██║      ██║       ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝       ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                                        
EOF
echo -e "${NC}"
echo -e "${BOLD}${CYAN}Rusty Stack - AMD GPU ML Environment Installer${NC}"
echo ""
echo -e "  ${BOLD}Shell:${NC}         $USER_SHELL"
echo -e "  ${BOLD}Distribution:${NC}  $DISTRO_NAME ($DISTRO_ID)"
echo -e "  ${BOLD}Package Mgr:${NC}   $PKG_MANAGER"
echo -e "  ${BOLD}Repository:${NC}    $REPO_URL"
echo -e "  ${BOLD}Branch:${NC}        $BRANCH"
echo -e "  ${BOLD}Install Dir:${NC}   $REPO_DIR"
echo ""

# Check dependencies
log_info "Checking dependencies..."

# Check for git
if ! check_command git; then
    log_warning "git is not installed"
    
    case "$PKG_MANAGER" in
        pacman|apt|dnf|yum|zypper)
            echo ""
            read -p "Would you like to install git now? [Y/n] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                if ! install_dependency "git" "git"; then
                    log_error "Failed to install git. Please install it manually."
                    exit 1
                fi
            else
                log_error "git is required to continue"
                exit 1
            fi
            ;;
        *)
            log_error "git is not installed. Please install git first."
            log_info "  Ubuntu/Debian: sudo apt install git"
            log_info "  Fedora/RHEL:   sudo dnf install git"
            log_info "  Arch Linux:    sudo pacman -S git"
            log_info "  openSUSE:      sudo zypper install git"
            exit 1
            ;;
    esac
else
    GIT_VERSION=$(git --version)
    log_success "git found: $GIT_VERSION"
fi

# Check for curl (needed for Rust installation)
if ! check_command curl; then
    log_warning "curl is not installed"
    
    case "$PKG_MANAGER" in
        pacman|apt|dnf|yum|zypper)
            echo ""
            read -p "Would you like to install curl now? [Y/n] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                if ! install_dependency "curl" "curl"; then
                    log_error "Failed to install curl. Please install it manually."
                    exit 1
                fi
            else
                log_error "curl is required to install Rust"
                exit 1
            fi
            ;;
        *)
            log_error "curl is not installed. Please install curl first."
            log_info "  Ubuntu/Debian: sudo apt install curl"
            log_info "  Fedora/RHEL:   sudo dnf install curl"
            log_info "  Arch Linux:    sudo pacman -S curl"
            log_info "  openSUSE:      sudo zypper install curl"
            exit 1
            ;;
    esac
else
    log_success "curl found"
fi

# Check for Rust toolchain
if ! check_command rustc || ! check_command cargo; then
    log_warning "Rust toolchain not found"
    echo ""
    read -p "Would you like to install Rust now? [Y/n] " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Downloading rustup..."
        
        if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
            # Source rust environment
            if [[ -f "$HOME/.cargo/env" ]]; then
                source "$HOME/.cargo/env"
                log_success "Rust toolchain installed"
                
                # Add to shell config for future sessions
                add_rust_to_shell_config
            else
                log_error "Failed to install Rust toolchain"
                exit 1
            fi
        else
            log_error "Failed to install Rust toolchain"
            exit 1
        fi
    else
        log_error "Rust is required to build Rusty Stack"
        log_info "Please install Rust manually: https://rustup.rs/"
        exit 1
    fi
else
    RUST_VERSION=$(rustc --version)
    log_success "Rust toolchain found: $RUST_VERSION"
fi

echo ""
log_info "All dependencies satisfied"
echo ""

# Clone or update repository
if [[ -d "$REPO_DIR/.git" ]]; then
    log_info "Repository exists at $REPO_DIR. Updating..."
    cd "$REPO_DIR"
    
    if git fetch --all && git reset --hard "origin/$BRANCH"; then
        log_success "Repository updated to latest $BRANCH"
    else
        log_error "Failed to update repository"
        exit 1
    fi
else
    log_info "Cloning repository from $REPO_URL..."
    
    if git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"; then
        cd "$REPO_DIR"
        log_success "Repository cloned to $REPO_DIR"
    else
        log_error "Failed to clone repository"
        exit 1
    fi
fi

echo ""

# Build Rusty-Stack TUI
if [[ "$SKIP_BUILD" = false ]]; then
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

echo ""

# Launch installer
BINARY_PATH="$REPO_DIR/rusty-stack/target/release/rusty-stack"

if [[ ! -f "$BINARY_PATH" ]]; then
    log_error "Binary not found at: $BINARY_PATH"
    log_info "This may indicate a build failure. Please check the output above."
    exit 1
fi

if [[ ! -x "$BINARY_PATH" ]]; then
    log_info "Making binary executable..."
    chmod +x "$BINARY_PATH"
fi

log_success "Installation preparation complete!"
echo ""

# Show shell-specific instructions for future use
case "$USER_SHELL" in
    fish)
        log_info "To run Rusty Stack in the future, ensure Rust is in your PATH:"
        echo -e "  ${CYAN}set -gx PATH \$HOME/.cargo/bin \$PATH${NC}"
        echo -e "  ${CYAN}$BINARY_PATH${NC}"
        ;;
    bash)
        log_info "To run Rusty Stack in the future:"
        echo -e "  ${CYAN}source ~/.bashrc${NC}  # If you just installed Rust"
        echo -e "  ${CYAN}$BINARY_PATH${NC}"
        ;;
    zsh)
        log_info "To run Rusty Stack in the future:"
        echo -e "  ${CYAN}source ~/.zshrc${NC}  # If you just installed Rust"
        echo -e "  ${CYAN}$BINARY_PATH${NC}"
        ;;
    *)
        log_info "To run Rusty Stack in the future:"
        echo -e "  ${CYAN}$BINARY_PATH${NC}"
        ;;
esac

echo ""
log_info "Launching Rusty-Stack TUI..."
echo ""

exec "$BINARY_PATH"