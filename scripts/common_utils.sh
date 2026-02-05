#!/bin/bash
# Common Utilities for Stan's ML Stack
# Provides reusable functions for all scripts including dry-run support

# Check if terminal supports colors
if [ -t 1 ]; then
    if [ -z "${NO_COLOR:-}" ]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[0;33m'
        BLUE='\033[0;34m'
        MAGENTA='\033[0;35m'
        CYAN='\033[0;36m'
        BOLD='\033[1m'
        RESET='\033[0m'
    else
        RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' RESET=''
    fi
else
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' RESET=''
fi

# Print functions (all output to stderr to avoid breaking command captures)
print_header() {
    echo >&2
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════╗${RESET}" >&2
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}║               === $1 ===               ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}║                                                         ║${RESET}" >&2
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════╝${RESET}" >&2
    echo >&2
}

print_section() {
    echo >&2
    echo -e "${BLUE}${BOLD}┌─────────────────────────────────────────────────────────┐${RESET}" >&2
    echo -e "${BLUE}${BOLD}│ $1${RESET}" >&2
    echo -e "${BLUE}${BOLD}└─────────────────────────────────────────────────────────┘${RESET}" >&2
}

print_step() {
    echo -e "${MAGENTA}➤ $1${RESET}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}" >&2
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}" >&2
}

print_error() {
    echo -e "${RED}✗ $1${RESET}" >&2
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Execute command with dry-run support and transient progress
# Usage: execute_command "command" "description"
execute_command() {
    local cmd="$1"
    local desc="$2"
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        echo -e "${YELLOW}[DRY-RUN]${RESET} ${desc}:" >&2
        echo -e "  ${BOLD}${cmd}${RESET}" >&2
        return 0
    fi
    
    print_step "$desc..."
    
    # We want to capture output and preserve \r for the TUI to handle as transient logs
    # Using 'stdbuf -oL' to keep it line-buffered
    if eval "$cmd" 2>&1; then
        print_success "Done: $desc"
        return 0
    else
        print_error "Failed: $desc"
        return 1
    fi
}

# Source .mlstack_env if it exists
if [ -f "$HOME/.mlstack_env" ]; then
    # Disable unbound variable check during source to prevent crash if LD_LIBRARY_PATH is empty
    set +u 2>/dev/null || true
    # shellcheck source=/dev/null
    source "$HOME/.mlstack_env"
    set -u 2>/dev/null || true
fi
