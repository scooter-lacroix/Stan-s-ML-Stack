#!/bin/bash
# fix_python_symlinks.sh
# Fix symlinks to point to the user's uv python instead of root's.

set -euo pipefail

USER_PY312="$HOME/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
USER_BIN_DIR=$(dirname "$USER_PY312")

if [ ! -x "$USER_PY312" ]; then
    echo "Error: User Python 3.12 not found at $USER_PY312"
    # Try to find it
    USER_PY312=$(find $HOME/.local/share/uv/python/ -name "python3.12" | head -n 1)
    if [ -z "$USER_PY312" ]; then
        echo "Could not find any uv-managed python 3.12 for user stan."
        exit 1
    fi
    USER_BIN_DIR=$(dirname "$USER_PY312")
fi

echo "Found user Python 3.12 at: $USER_PY312"

echo "Updating /usr/local/bin symlinks..."
sudo ln -sf "$USER_PY312" /usr/local/bin/python3
sudo ln -sf "$USER_PY312" /usr/local/bin/python
sudo ln -sf "$USER_BIN_DIR/pip" /usr/local/bin/pip || sudo ln -sf "$USER_PY312 -m pip" /usr/local/bin/pip
sudo ln -sf "$USER_BIN_DIR/pip3" /usr/local/bin/pip3 || sudo ln -sf "$USER_PY312 -m pip" /usr/local/bin/pip3

echo "Verifying..."
/usr/local/bin/python3 --version
which python3
