#!/bin/bash

# ML Stack Installation UI Runner
# This script runs the curses-based UI for ML Stack installation

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLSTACK_DIR="$(dirname "$SCRIPT_DIR")"

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
echo "║             ✵ Stan's ML Stack Installer © ✵            ║"
echo "║                                                         ║"
echo "╚═════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}${BOLD}Error: Python 3 is not installed.${RESET}"
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if curses is available
if ! python3 -c "import curses" &> /dev/null; then
    echo -e "${RED}${BOLD}Error: Python curses module is not available.${RESET}"
    echo "Please install the curses module and try again."
    echo "On Debian/Ubuntu: sudo apt-get install python3-dev"
    exit 1
fi

# Run the UI
echo -e "${CYAN}${BOLD}Starting the ML Stack Installation UI...${RESET}"
cd "$MLSTACK_DIR" || exit 1
python3 "$SCRIPT_DIR/install_ml_stack_curses.py"

# Check the exit code
if [ $? -ne 0 ]; then
    echo -e "${RED}${BOLD}The UI exited with an error.${RESET}"
    echo "Check the logs for more information."
    exit 1
fi

echo -e "${GREEN}${BOLD}ML Stack Installation UI exited successfully.${RESET}"
