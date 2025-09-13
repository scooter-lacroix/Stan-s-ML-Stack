#!/bin/bash
# Script to build and push the lightweight ML Stack Docker image

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

print_header "Building Stan's ML Stack Lightweight Docker Image v${VERSION}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "You can install Docker with:"
    echo "  sudo apt install docker.io"
    echo "  sudo systemctl start docker"
    echo "  sudo systemctl enable docker"
    echo "  sudo usermod -aG docker $USER"
    exit 1
fi

# Check if user is logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username"; then
    print_warning "You are not logged in to Docker Hub. You may need to login with 'docker login' before pushing."
fi

# Remove any existing images
print_section "Removing existing images"
docker rmi -f ${IMAGE_NAME}:latest ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:rocm-${ROCM_VERSION} 2>/dev/null || true

# Build the Docker image
print_section "Building Docker image"
docker build -t ${IMAGE_NAME}:${VERSION} -f Dockerfile.lightweight .

# Check if the build was successful
if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

print_success "Docker image built successfully!"

# Tag the image with additional tags
print_section "Tagging image with additional tags"
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:rocm-${ROCM_VERSION}

print_success "Image tagged as:"
echo "  ${IMAGE_NAME}:${VERSION}"
echo "  ${IMAGE_NAME}:latest"
echo "  ${IMAGE_NAME}:rocm-${ROCM_VERSION}"

# Verify the Docker image
if [ -f "./verify_docker_image.sh" ]; then
    print_section "Verifying Docker image"
    ./verify_docker_image.sh
else
    print_warning "verify_docker_image.sh not found, skipping verification"
fi

# Ask user if they want to push the image
read -p "Do you want to push the images to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_section "Pushing Docker images to Docker Hub"
    docker push ${IMAGE_NAME}:${VERSION}
    docker push ${IMAGE_NAME}:latest
    docker push ${IMAGE_NAME}:rocm-${ROCM_VERSION}
    print_success "Images pushed to Docker Hub:"
    echo "  ${IMAGE_NAME}:${VERSION}"
    echo "  ${IMAGE_NAME}:latest"
    echo "  ${IMAGE_NAME}:rocm-${ROCM_VERSION}"
else
    print_section "Skipping push to Docker Hub"
fi

print_header "Build process complete!"
echo "You can run the image with:"
echo "  docker run --device=/dev/kfd --device=/dev/dri --group-add video -it ${IMAGE_NAME}:latest"
