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

sudo install -m 0755 target/release/rusty-stack "$INSTALL_DIR/rusty-stack"

cat << EOF
Rusty-Stack installed to $INSTALL_DIR/rusty-stack
Run with: sudo rusty-stack
EOF
