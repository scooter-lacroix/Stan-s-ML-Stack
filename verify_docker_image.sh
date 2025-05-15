#!/bin/bash
# Script to verify the ML Stack Docker image

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Function to print colored messages
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}! $1${RESET}"
}

# Set variables
VERSION="0.1.4-secure"
IMAGE_NAME="bartholemewii/stans-ml-stack"

print_header "Verifying Stan's ML Stack Docker Image v${VERSION}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if the image exists
if ! docker images | grep -q "${IMAGE_NAME}"; then
    print_error "Docker image ${IMAGE_NAME} not found. Please build it first."
    exit 1
fi

# Run the verification script inside the container
print_section "Running verification script"
docker run --rm ${IMAGE_NAME}:${VERSION} /workspace/verify_ml_stack.sh

# Verify pip package installation
print_section "Verifying Stan's ML Stack pip package installation"
docker run --rm ${IMAGE_NAME}:${VERSION} python3 -c "import importlib.util; print('Stan\'s ML Stack package is ' + ('installed' if importlib.util.find_spec('scripts') else 'not installed'))"

print_section "Verifying entry points"
docker run --rm ${IMAGE_NAME}:${VERSION} which ml-stack-install ml-stack-verify ml-stack-repair

print_header "Verification complete!"
print_success "The Docker image is ready to use!"
echo "You can run it with:"
echo "  docker run --device=/dev/kfd --device=/dev/dri --group-add video -it ${IMAGE_NAME}:latest"
