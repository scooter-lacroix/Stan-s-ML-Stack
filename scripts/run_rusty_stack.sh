#!/bin/bash
# Build and run the ML Stack TUI installer

set -euo pipefail

# Color definitions
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

# Function definitions
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_step() {
    echo -e "${YELLOW}${BOLD}→ $1${RESET}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_MODE=${1:-release}

print_header "ML Stack TUI Installer"
print_step "Build mode: $BUILD_MODE"

cd "$PROJECT_ROOT"

if [ "$BUILD_MODE" = "debug" ]; then
    cargo build -p rusty-stack
    EXEC_PATH="target/debug/Rusty-Stack"
else
    cargo build -p rusty-stack --release
    EXEC_PATH="target/release/Rusty-Stack"
fi

print_step "Launching $EXEC_PATH"
"$EXEC_PATH"
