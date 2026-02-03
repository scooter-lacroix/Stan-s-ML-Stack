#!/bin/bash
# Build and run the ML Stack TUI installer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$SCRIPT_DIR/common_utils.sh" ]; then
    source "$SCRIPT_DIR/common_utils.sh"
fi

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
