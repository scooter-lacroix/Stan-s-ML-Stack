#!/bin/bash
# =============================================================================
# Rusty-Stack System Integration
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INSTALL_DIR="/usr/local/bin"

cd "$ROOT_DIR/rusty-stack"

cargo build --release

sudo install -m 0755 target/release/Rusty-Stack "$INSTALL_DIR/Rusty-Stack"

cat << EOF
Rusty-Stack installed to $INSTALL_DIR/Rusty-Stack
Run with: sudo Rusty-Stack
EOF
