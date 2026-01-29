#!/bin/bash

# Debugging wrapper for the Go UI installer
# This helps identify why the installer shows a blank screen

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLSTACK_DIR="$(dirname "$SCRIPT_DIR")"
GO_INSTALLER_DIR="$MLSTACK_DIR/mlstack-installer"

echo "=== Go UI Installer Debug Wrapper ==="
echo ""
echo "Path diagnostics:"
echo "  SCRIPT_DIR: $SCRIPT_DIR"
echo "  MLSTACK_DIR: $MLSTACK_DIR"
echo "  GO_INSTALLER_DIR: $GO_INSTALLER_DIR"
echo ""

# Check if installer directory exists
if [ ! -d "$GO_INSTALLER_DIR" ]; then
    echo "ERROR: Go installer directory not found at: $GO_INSTALLER_DIR"
    exit 1
fi

cd "$GO_INSTALLER_DIR" || exit 1

echo "Current directory: $(pwd)"
echo ""

# Find executable
EXECUTABLE=""
for exe in mlstack-installer-fixed mlstack-installer-rebuilt mlstack-installer build/mlstack-installer-fixed build/mlstack-installer dist/mlstack-installer-linux-amd64; do
    if [ -f "$exe" ]; then
        EXECUTABLE="$exe"
        echo "Found executable: $EXECUTABLE"
        break
    fi
done

if [ -z "$EXECUTABLE" ]; then
    echo "ERROR: No executable found"
    exit 1
fi
echo ""

echo "Environment diagnostics:"
echo "  EUID: $EUID"
echo "  USER: $USER"
echo "  TERM: $TERM"
echo "  TTY available: $(test -t 0 && echo 'Yes' || echo 'No')"
echo "  stdin TTY: $(test -t 0 && echo 'Yes' || echo 'No')"
echo "  stdout TTY: $(test -t 1 && echo 'Yes' || echo 'No')"
echo "  stderr TTY: $(test -t 2 && echo 'Yes' || echo 'No')"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Not running as root - will use sudo"
    SUDO_PREFIX="sudo -E"
else
    echo "Running as root"
    SUDO_PREFIX=""
fi
echo ""

echo "Running installer with timeout and strace..."
echo "This will help identify where it hangs"
echo ""

# Run with timeout and strace to see what's happening
timeout 5 strace -e trace=ioctl,read,write,open $SUDO_PREFIX ./$EXECUTABLE 2>&1 | head -100 || echo "Installer timed out or exited"

exit $?
