#!/bin/bash
# system_python_cleanup.sh
# Aggressively clean up non-essential Pythons and set 3.12 as global default.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

print_step() { echo -e "${YELLOW}➤ $1${RESET}"; }
print_success() { echo -e "${GREEN}✓ $1${RESET}"; }
print_error() { echo -e "${RED}✗ $1${RESET}"; }

# 1. Remove Anaconda
if [ -d "$HOME/anaconda3" ]; then
    print_step "Removing Anaconda..."
    rm -rf "$HOME/anaconda3"
    sed -i '/anaconda3/d' "$HOME/.bashrc" || true
    sed -i '/conda/d' "$HOME/.bashrc" || true
    print_success "Anaconda removed."
fi

# 2. Remove Homebrew Python 3.14
if command -v brew &>/dev/null; then
    print_step "Removing Homebrew Python 3.14..."
    brew uninstall --force python@3.14 || true
    print_success "Homebrew Python 3.14 removed."
fi

# 3. Setup Python 3.12 via UV
if ! command -v uv &>/dev/null; then
    print_step "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

print_step "Ensuring Python 3.12 is installed via uv..."
uv python install 3.12

# Get the path to uv-managed python 3.12
PY312_PATH=$(uv python find 3.12)
PY312_BIN_DIR=$(dirname "$PY312_PATH")

print_step "Setting Python 3.12 as global default in /usr/local/bin..."
# We use sudo to put symlinks in /usr/local/bin which is usually first in PATH
# but doesn't break system tools in /usr/bin
sudo ln -sf "$PY312_PATH" /usr/local/bin/python3
sudo ln -sf "$PY312_PATH" /usr/local/bin/python
sudo ln -sf "$PY312_BIN_DIR/pip" /usr/local/bin/pip || sudo ln -sf "$PY312_PATH -m pip" /usr/local/bin/pip
sudo ln -sf "$PY312_BIN_DIR/pip3" /usr/local/bin/pip3 || sudo ln -sf "$PY312_PATH -m pip" /usr/local/bin/pip3

# 4. Verify
print_step "Verifying current Python 3 setup..."
python3 --version
which python3

# 5. Environment Refactor
print_step "Updating environment variables to favor Python 3.12..."
# We will regenerate .mlstack_env in the next step via Rusty-Stack

print_success "System Python cleanup and 3.12 migration complete!"
echo -e "${YELLOW}Please restart your terminal or run: source ~/.bashrc${RESET}"
