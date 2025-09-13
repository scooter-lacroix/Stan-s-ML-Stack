# Base image with ROCm support
FROM rocm/dev-ubuntu-22.04:latest

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
LABEL version="0.1.4-Sotapanna"
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

# Create workspace directory
WORKDIR /workspace

# Copy ML Stack files
COPY . /workspace/Stans_MLStack/

# Create symbolic links for compatibility
RUN ln -sf /usr/bin/ninja /usr/bin/ninja-build || true \
    && ln -sf /opt/rocm /usr/local/cuda || true

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4 \
    && pip3 install numpy scipy matplotlib pandas scikit-learn jupyter \
    && pip3 install onnxruntime wandb psutil tqdm requests pyyaml \
    && pip3 install -e /workspace/Stans_MLStack/

# MIGraphX installation is skipped as it requires a more complex setup
# It needs to be built from source for ROCm compatibility

# Set up working directory
WORKDIR /workspace/Stans_MLStack

# Make scripts executable
RUN chmod +x /workspace/Stans_MLStack/scripts/*.sh /workspace/Stans_MLStack/scripts/*.py

# Create a verification script
RUN echo '#!/bin/bash\n\
echo "=== Stan'\''s ML Stack Verification ===="\n\
echo "Note: This verification should be run with GPU access enabled"\n\
echo "PyTorch version: $(python3 -c "import torch; print(torch.__version__)")" \n\
echo "Checking CUDA availability..."\n\
python3 -c "import torch; print('\''CUDA available:'\'', torch.cuda.is_available())" \n\
echo "Checking ROCm version..."\n\
python3 -c "import torch; print('\''ROCm version:'\'', torch.version.hip if hasattr(torch.version, '\''hip'\'') else '\''Not available'\'')" \n\
echo ""\n\
echo "Stan'\''s ML Stack is installed as a pip package. You can run:"\n\
echo "ml-stack-install    # To launch the installer UI"\n\
echo "ml-stack-verify     # To verify your installation"\n\
echo "ml-stack-repair     # To repair a broken installation"\n\
echo ""\n\
echo "To use this container with GPU access, run:"\n\
echo "docker run --device=/dev/kfd --device=/dev/dri --group-add video -it stans-ml-stack:0.1.4-Sotapanna"\n\
' > /workspace/verify_ml_stack.sh \
    && chmod +x /workspace/verify_ml_stack.sh

# Run basic verification script to check installation
RUN echo "Running basic verification..." \
    && python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" \
    && python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)" \
    && python3 -c "import wandb; print('Weights & Biases version:', wandb.__version__)" \
    && python3 -c "import psutil; print('psutil version:', psutil.__version__)" \
    && python3 -c "import tqdm; print('tqdm version:', tqdm.__version__)" \
    && python3 -c "import yaml; print('PyYAML version:', yaml.__version__)" \
    && python3 -c "import stans_ml_stack; print('Stan\'s ML Stack imported successfully')"

# Run basic tests during build
RUN echo "Running basic tests..." \
    && python3 -m pytest tests/pytorch_simple_test.py -v || echo "PyTorch test failed, but continuing..." \
    && python3 -c "import torch; x = torch.randn(10); print('Basic PyTorch operations work:', x.mean().item())"

# Run a quick benchmark during build
RUN echo "Running quick benchmark..." \
    && python3 benchmarks/matrix_multiplication_benchmark.py --sizes 512 --num-runs 1 || echo "Benchmark failed, but continuing..."

# Expose ports for services (Jupyter, TensorBoard, etc.)
EXPOSE 8888 6006 8080

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Welcome to Stan'\''s ML Stack! Run ml-stack-install to launch the installer UI or ml-stack-verify to verify installation.'"]
