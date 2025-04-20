# Base image with ROCm support
FROM rocm/dev-ubuntu-22.04:6.2.4

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:$LD_LIBRARY_PATH
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
ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
ENV MIOPEN_FIND_MODE=3
ENV MIOPEN_FIND_ENFORCE=3
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    python3-dev \
    python3-pip \
    ninja-build \
    libopenmpi-dev \
    openmpi-bin \
    libomp-dev \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy ML Stack files
COPY . /workspace/Stans_MLStack/

# Create symbolic links for compatibility
RUN ln -sf /usr/bin/ninja /usr/bin/ninja-build || true \
    && ln -sf /opt/rocm /usr/local/cuda || true

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install torch==2.6.0+rocm6.2.4 torchvision==0.17.0+rocm6.2.4 --index-url https://download.pytorch.org/whl/rocm6.2.4 \
    && pip3 install numpy scipy matplotlib pandas scikit-learn jupyter \
    && pip3 install onnx onnxruntime-rocm \
    && pip3 install wandb bitsandbytes-rocm triton-rocm

# Install MIGraphX
RUN pip3 install migraphx

# Set up working directory
WORKDIR /workspace/Stans_MLStack

# Make scripts executable
RUN chmod +x /workspace/Stans_MLStack/scripts/*.sh

# Run verification script to check installation
RUN /workspace/Stans_MLStack/scripts/custom_verify_installation.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
