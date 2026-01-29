#!/bin/bash

# Stan's ML Stack - Go Bubble Tea UI Launcher
# This script launches the modern Go-based Bubble Tea installer UI

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLSTACK_DIR="$(dirname "$SCRIPT_DIR")"
GO_INSTALLER_DIR="$MLSTACK_DIR/mlstack-installer"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Print fancy header
echo -e "${MAGENTA}${BOLD}"
echo "╔═════════════════════════════════════════════════════════╗"
echo "║                                                         ║"
echo "║       ✵ Stan's ML Stack Installer (Go UI) © ✵        ║"
echo "║       Modern Bubble Tea Terminal Interface              ║"
echo "║                                                         ║"
echo "╚═════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if the installer directory exists
if [ ! -d "$GO_INSTALLER_DIR" ]; then
    echo -e "${RED}${BOLD}Error: Go installer directory not found at: $GO_INSTALLER_DIR${RESET}"
    exit 1
fi

# Change to the installer directory
cd "$GO_INSTALLER_DIR" || exit 1

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    # Already running as root
    echo -e "${GREEN}${BOLD}Running with root privileges${RESET}"
    echo ""
else
    # Not running as root - check if we should use sudo
    echo -e "${YELLOW}${BOLD}This installer requires root privileges for full functionality.${RESET}"
    echo ""

    # Check if sudo is available
    if ! command -v sudo >/dev/null 2>&1; then
        echo -e "${RED}${BOLD}Error: sudo not found. Please run as root.${RESET}"
        exit 1
    fi

    # Check if we can sudo without password or prompt once
    echo -e "${CYAN}The installer will now restart with sudo...${RESET}"
    echo ""

    # Re-run this script with sudo, preserving environment
    exec sudo -E "$0" "$@"
fi

# Try to find a working executable
# Priority: mlstack-installer-fixed > mlstack-installer-rebuilt > mlstack-installer > build/mlstack-installer
EXECUTABLE=""

# Check for various executables in different locations
for exe in \
    "mlstack-installer-fixed" \
    "mlstack-installer-rebuilt" \
    "mlstack-installer" \
    "build/mlstack-installer" \
    "build/mlstack-installer-fixed" \
    "dist/mlstack-installer-linux-amd64"
do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        EXECUTABLE="$exe"
        break
    fi
done

# If no executable found, try to find any executable file
if [ -z "$EXECUTABLE" ]; then
    for exe in mlstack-installer*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            EXECUTABLE="$exe"
            echo -e "${YELLOW}Found executable: $EXECUTABLE${RESET}"
            break
        fi
    done
fi

# If still no executable, check build directory
if [ -z "$EXECUTABLE" ] && [ -d "build" ]; then
    for exe in build/*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            EXECUTABLE="$exe"
            echo -e "${YELLOW}Found executable: $EXECUTABLE${RESET}"
            break
        fi
    done
fi

if [ -z "$EXECUTABLE" ]; then
    echo -e "${RED}${BOLD}Error: No Go installer executable found!${RESET}"
    echo "Please build the installer first with:"
    echo "  cd $GO_INSTALLER_DIR"
    echo "  make build"
    echo ""
    echo "Or check if an executable exists in the build directory."
    exit 1
fi

echo -e "${GREEN}${BOLD}Found Go installer: $EXECUTABLE${RESET}"
echo ""

# Display terminal diagnostics
echo -e "${CYAN}Terminal Diagnostics:${RESET}"
echo "  TERM: $TERM"
echo "  TTY available: $(test -t 0 && echo 'Yes' || echo 'No')"
echo "  stdin is TTY: $(test -t 0 && echo 'Yes' || echo 'No')"
echo "  stdout is TTY: $(test -t 1 && echo 'Yes' || echo 'No')"
echo ""

# Check for required terminal capabilities
if [ ! -t 0 ] || [ ! -t 1 ]; then
    echo -e "${RED}${BOLD}Error: This installer requires an interactive terminal (TTY).${RESET}"
    echo ""
    echo "Please run this directly in a terminal, not through:"
    echo "  • Pipes or redirects"
    echo "  • Background jobs"
    echo "  • Cron jobs"
    echo "  • Non-interactive shells"
    echo ""
    echo "Try running in a real terminal with:"
    echo "  sudo ./scripts/run_go_ui.sh"
    exit 1
fi

# Ensure proper TERM setting
if [ -z "$TERM" ] || [ "$TERM" = "dumb" ]; then
    export TERM=xterm-256color
    echo -e "${YELLOW}Setting TERM to xterm-256color for better compatibility${RESET}"
fi

# Launch the installer
echo -e "${CYAN}${BOLD}Launching Stan's ML Stack Installer...${RESET}"
echo ""
echo -e "${YELLOW}Tip: Press Ctrl+C to exit the installer at any time${RESET}"
echo ""

# Run the executable
if [ -x "./$EXECUTABLE" ]; then
    ./"$EXECUTABLE" "$@"
else
    # Make executable if it exists but isn't executable
    chmod +x "$EXECUTABLE"
    ./"$EXECUTABLE" "$@"
fi

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}${BOLD}The installer exited with code: $EXIT_CODE${RESET}"
    echo ""
    echo "Common issues:"
    echo "  • Terminal not in interactive mode - run directly in a terminal"
    echo "  • Missing TTY - ensure you're not piping output"
    echo "  • Display issues - try setting TERM=xterm-256color"
    echo ""
    echo "To debug, run with: bash -x $0"
    exit $EXIT_CODE
else
    echo ""
    echo -e "${GREEN}${BOLD}Installer exited successfully.${RESET}"
fi

exit 0
