#!/bin/bash
set -ex

# This script builds and installs FastVideo with ROCm support.
# It assumes that the FastVideo repository has been forked to scooter-lacroix/FastVideo
# and the feature/rocm-gfx11-support branch exists.

FASTVIDEO_REPO="https://github.com/scooter-lacroix/FastVideo.git"
FASTVIDEO_BRANCH="feature/rocm-gfx11-support"
FASTVIDEO_DIR="/tmp/FastVideo_ROCm_build"

mkdir -p "${FASTVIDEO_DIR}"
cd "${FASTVIDEO_DIR}"

# Clone the forked FastVideo repository
git clone "${FASTVIDEO_REPO}" .
git checkout "${FASTVIDEO_BRANCH}"

# Navigate to the fastvideo-kernel directory and build with ROCm support
cd fastvideo-kernel
./build.sh --rocm

# Install the built package
pip install .

# Clean up the build directory
rm -rf "${FASTVIDEO_DIR}"

echo "FastVideo with ROCm support built and installed successfully."
