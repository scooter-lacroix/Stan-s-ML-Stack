#!/bin/bash

# Stan's ML Stack - Go Bubble Tea UI Launcher
# Simple launcher that runs the Go installer with sudo

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALLER_DIR="$PROJECT_DIR/mlstack-installer"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "╔═════════════════════════════════════════════════════════╗"
echo "║                                                         ║"
echo "║       ✵ Stan's ML Stack Installer (Go UI) © ✵        ║"
echo "║       Modern Bubble Tea Terminal Interface              ║"
echo "║                                                         ║"
echo "╚═════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if installer directory exists
if [ ! -d "$INSTALLER_DIR" ]; then
    echo -e "${RED}${BOLD}Error: Installer directory not found:${RESET} $INSTALLER_DIR"
    exit 1
fi

# Find the Go installer executable
EXECUTABLE=""
cd "$INSTALLER_DIR" || exit 1

# Try different possible locations
for exe in mlstack-installer-fixed mlstack-installer build/mlstack-installer-fixed build/mlstack-installer dist/mlstack-installer-linux-amd64; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        EXECUTABLE="$exe"
        break
    fi
done

if [ -z "$EXECUTABLE" ]; then
    # Try any executable file
    for exe in mlstack-installer*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            EXECUTABLE="$exe"
            break
        fi
    done
fi

if [ -z "$EXECUTABLE" ]; then
    echo -e "${RED}${BOLD}Error: No Go installer executable found in:${RESET} $INSTALLER_DIR"
    echo "Available files:"
    ls -la | grep "mlstack-installer"
    exit 1
fi

echo -e "${GREEN}${BOLD}Found installer: $EXECUTABLE${RESET}"
echo ""

# Run the installer with sudo
echo -e "${CYAN}Launching installer with sudo...${RESET}"
echo ""

# Run with sudo -E to preserve environment
sudo -E ./"$EXECUTABLE" "$@"
exit $?
