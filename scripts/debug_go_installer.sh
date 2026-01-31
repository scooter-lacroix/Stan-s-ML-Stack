#!/bin/bash

# Debug wrapper for the Go installer to see why it shows a blank screen

INSTALLER_DIR="/home/stan/Documents/Stan-s-ML-Stack/mlstack-installer"
cd "$INSTALLER_DIR" || exit 1

echo "=== Go Installer Debug Wrapper ==="
echo ""

# Find executable
EXECUTABLE=""
for exe in mlstack-installer-fixed mlstack-installer build/mlstack-installer-fixed build/mlstack-installer; do
    if [ -f "$exe" ]; then
        EXECUTABLE="$exe"
        break
    fi
done

if [ -z "$EXECUTABLE" ]; then
    echo "ERROR: No installer found"
    exit 1
fi

echo "Executable: $EXECUTABLE"
echo "Current directory: $(pwd)"
echo ""

# Test 1: Check if installer binary works at all
echo "=== Test 1: --help flag ==="
timeout 2 ./"$EXECUTABLE" --help 2>&1 | head -5
echo ""

# Test 2: Check if installer binary works with --version
echo "=== Test 2: --version flag ==="
timeout 2 ./"$EXECUTABLE" --version 2>&1
echo ""

# Test 3: Run installer directly without sudo to see what happens
echo "=== Test 3: Run without sudo (shows demo mode prompt) ==="
echo "Will timeout after 3 seconds..."
timeout 3 ./"$EXECUTABLE" 2>&1 || echo "Timed out (expected - waiting for input)"
echo ""

# Test 4: Try with strace to see system calls
echo "=== Test 4: Run with strace to see system calls ==="
echo "Will timeout after 3 seconds..."
timeout 3 strace -e trace=ioctl,select,read,write,open,exit ./"$EXECUTABLE" 2>&1 | head -50
echo ""

echo "=== Debug Complete ==="
echo ""
echo "The blank screen issue is likely a Bubble Tea TUI initialization problem."
echo "The Python curses UI works because it handles terminals differently."
