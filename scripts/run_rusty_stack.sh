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
RUSTY_STACK_DIR="$PROJECT_ROOT/rusty-stack"

BUILD_MODE="release"
if [ "$#" -gt 0 ]; then
    case "$1" in
        debug|release)
            BUILD_MODE="$1"
            shift
            ;;
    esac
fi
BINARY_ARGS=("$@")

print_header "ML Stack TUI Installer"
print_step "Build mode: $BUILD_MODE"

if [ ! -d "$RUSTY_STACK_DIR" ]; then
    echo -e "${RED}${BOLD}✗ Rust project directory not found: $RUSTY_STACK_DIR${RESET}" >&2
    exit 1
fi

cd "$RUSTY_STACK_DIR"

case "$BUILD_MODE" in
    debug)
        cargo build
        EXEC_PATH="target/debug/rusty-stack"
        ;;
    release)
        cargo build --release
        EXEC_PATH="target/release/rusty-stack"
        ;;
    *)
        echo -e "${RED}${BOLD}✗ Invalid build mode: $BUILD_MODE (expected: debug|release)${RESET}" >&2
        exit 1
        ;;
esac

print_step "Launching $EXEC_PATH"
exec "./$EXEC_PATH" "${BINARY_ARGS[@]}"
