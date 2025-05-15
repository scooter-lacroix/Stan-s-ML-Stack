# Stan's ML Stack Docker Image

This Docker image provides a lightweight environment for Stan's ML Stack, a comprehensive machine learning environment optimized for AMD GPUs.

## Features

- Based on the latest ROCm 6.3.4 image
- Pre-installed PyTorch with ROCm support
- Stan's ML Stack installer included as a pip package
- Minimal size with only essential dependencies
- Ready for AMD GPU acceleration

## Quick Start

To run the container with GPU access:

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it bartholemewii/stans-ml-stack:latest
```

## Available Commands

Once inside the container, you can:

- Run `./verify_ml_stack.sh` to verify your installation
- Run `ml-stack-install` to launch the ML Stack installer UI
- Run `ml-stack-verify` to verify your ML Stack installation
- Run `ml-stack-repair` to repair a broken installation

## Installing Additional Components

This image is intentionally lightweight. You can install additional components as needed:

```bash
# Inside the container
ml-stack-install

# Select the components you want to install from the UI
```

## Available Tags

- `latest`: The latest version of Stan's ML Stack
- `0.1.3`: Version 0.1.3 of Stan's ML Stack
- `rocm-6.3.4`: Built with ROCm 6.3.4

## Building the Image Locally

To build the image locally:

```bash
git clone https://github.com/scooter-lacroix/Stan-s-ML-Stack.git
cd Stan-s-ML-Stack
docker build -t stans-ml-stack:local -f Dockerfile.lightweight .
```

## Environment Variables

The following environment variables are set in the container:

- `ROCM_PATH=/opt/rocm`
- `PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin`
- `LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib`
- `HIP_VISIBLE_DEVICES=0`
- `CUDA_VISIBLE_DEVICES=0`
- `PYTORCH_ROCM_DEVICE=0`
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `HSA_ENABLE_SDMA=0`
- `GPU_MAX_HEAP_SIZE=100`
- `GPU_MAX_ALLOC_PERCENT=100`
- `HSA_TOOLS_LIB=1`
- `ROCM_HOME=$ROCM_PATH`
- `CUDA_HOME=$ROCM_PATH`
- `AMD_LOG_LEVEL=0`

## Exposed Ports

- 8888: Jupyter Notebook
- 6006: TensorBoard
- 8080: Web services

## Troubleshooting

If you encounter issues with GPU access, make sure:

1. Your host has ROCm installed and configured
2. You're using the correct Docker run command with device access
3. Your user has permissions to access the GPU devices

For more detailed troubleshooting, run `ml-stack-verify` inside the container.

## License

MIT License

## Maintainer

Stanley Chisango (Scooter Lacroix) <scooterlacroix@gmail.com>
