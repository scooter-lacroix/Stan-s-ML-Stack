#!/bin/bash
# Rusty-Stack build helper

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"

cd "$ROOT_DIR/rusty-stack"

cargo build --release

cp target/release/rusty-stack "$BUILD_DIR/rusty-stack"

echo "Built rusty-stack -> $BUILD_DIR/rusty-stack"
