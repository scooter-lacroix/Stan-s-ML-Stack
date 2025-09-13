#!/bin/bash
# Script to push the ML Stack Docker image to Docker Hub

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
VERSION="0.1.4-Sotapanna"
ROCM_VERSION="latest"
IMAGE_NAME="bartholemewii/stans-ml-stack"

print_header "Pushing Stan's ML Stack Docker Image v${VERSION} to Docker Hub"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if user is logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username"; then
    print_warning "You are not logged in to Docker Hub. Please login first."
    docker login

    # Check if login was successful
    if [ $? -ne 0 ]; then
        print_error "Docker login failed. Please try again."
        exit 1
    fi
fi

# Check if the image exists
if ! docker images | grep -q "${IMAGE_NAME}"; then
    print_error "Docker image ${IMAGE_NAME} not found. Please build it first with ./build_and_push_docker.sh"
    exit 1
fi

# Push the images to Docker Hub
print_section "Pushing Docker images to Docker Hub"
docker push ${IMAGE_NAME}:${VERSION}
docker push ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:rocm-${ROCM_VERSION}

# Check if the push was successful
if [ $? -ne 0 ]; then
    print_error "Docker push failed!"
    exit 1
fi

print_success "Images pushed to Docker Hub:"
echo "  ${IMAGE_NAME}:${VERSION}"
echo "  ${IMAGE_NAME}:latest"
echo "  ${IMAGE_NAME}:rocm-${ROCM_VERSION}"

print_header "Push complete!"
echo "Users can pull the image with:"
echo "  docker pull ${IMAGE_NAME}:latest"
echo ""
echo "And run it with:"
echo "  docker run --device=/dev/kfd --device=/dev/dri --group-add video -it ${IMAGE_NAME}:latest"
