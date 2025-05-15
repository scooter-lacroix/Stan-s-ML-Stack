# Base image with ROCm support
FROM rocm/dev-ubuntu-22.04:6.4.43482

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV AMD_LOG_LEVEL=0
ENV ROCM_PATH=/opt/rocm
ENV PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib
ENV HIP_VISIBLE_DEVICES=0,1
ENV CUDA_VISIBLE_DEVICES=0,1
ENV PYTORCH_ROCM_DEVICE=0,1
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV HSA_ENABLE_SDMA=0
ENV GPU_MAX_HEAP_SIZE=100
ENV GPU_MAX_ALLOC_PERCENT=100
ENV HSA_TOOLS_LIB=1
ENV ROCM_HOME=$ROCM_PATH
ENV CUDA_HOME=$ROCM_PATH

# Label the image
LABEL maintainer="Stanley Chisango (Scooter Lacroix) <scooterlacroix@gmail.com>"
LABEL version="0.1.2"
LABEL description="Stan's ML Stack - A comprehensive machine learning environment optimized for AMD GPUs"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    python3-dev \
    python3-pip \
    ninja-build \
    libopenmpi-dev \
    openmpi-bin \
    libnuma-dev \
    pciutils \
    mesa-utils \
    clinfo \
    lsb-release \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -sSf https://astral.sh/uv/install.sh | sh

# Create workspace directory
WORKDIR /workspace

# Copy ML Stack files
COPY . /workspace/Stans_MLStack/

# Create symbolic links for compatibility
RUN ln -sf /usr/bin/ninja /usr/bin/ninja-build || true \
    && ln -sf /opt/rocm /usr/local/cuda || true

# Install Python dependencies using UV
RUN /root/.cargo/bin/uv pip install --upgrade pip \
    && /root/.cargo/bin/uv pip install torch==2.6.0+rocm6.4.43482 torchvision==0.21.0+rocm6.4.43482 --index-url https://download.pytorch.org/whl/rocm6.4.43482 \
    && /root/.cargo/bin/uv pip install numpy scipy matplotlib pandas scikit-learn jupyter \
    && /root/.cargo/bin/uv pip install onnx \
    && /root/.cargo/bin/uv pip install wandb \
    && /root/.cargo/bin/uv pip install psutil tqdm requests pyyaml

# MIGraphX installation is skipped as it requires a more complex setup
# It needs to be built from source for ROCm compatibility

# Set up working directory
WORKDIR /workspace/Stans_MLStack

# Make scripts executable
RUN chmod +x /workspace/Stans_MLStack/scripts/*.sh /workspace/Stans_MLStack/scripts/*.py

# Run basic verification script to check installation
RUN echo "Running basic verification..." \
    && python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" \
    && python3 -c "import onnx; print('ONNX version:', onnx.__version__)" \
    && python3 -c "import wandb; print('Weights & Biases version:', wandb.__version__)" \
    && python3 -c "import psutil; print('psutil version:', psutil.__version__)" \
    && python3 -c "import tqdm; print('tqdm version:', tqdm.__version__)" \
    && python3 -c "import yaml; print('PyYAML version:', yaml.__version__)"

# Expose ports for services (Jupyter, TensorBoard, etc.)
EXPOSE 8888 6006 8080

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["./scripts/install_ml_stack_curses.py"]
