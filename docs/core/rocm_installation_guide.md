# ROCm Installation Guide

## Introduction

This guide provides detailed instructions for installing the ROCm (Radeon Open Compute) platform on Ubuntu for AMD GPUs. ROCm is the foundation for GPU computing on AMD hardware and is required for running machine learning workloads on AMD GPUs.

## Prerequisites

Before installing ROCm, ensure your system meets the following requirements:

1. **Supported AMD GPU**: RX 7900 XTX, RX 7800 XT, or other supported AMD GPU
2. **Supported Linux Distribution**: Ubuntu 25.04 (or compatible)
3. **System Requirements**:
   - 64-bit x86_64 CPU (preferably with PCIe 4.0 support)
   - At least 16GB of system memory
   - At least 50GB of free disk space
   - Internet connection for downloading packages

## Installation Steps

### 1. Update System Packages

First, update your system packages:

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Required Dependencies

Install the necessary dependencies:

```bash
sudo apt install -y \
    libnuma-dev \
    cmake \
    pkg-config \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools
```

### 3. Add ROCm Repository

Add the ROCm repository to your system:

```bash
# Add the ROCm repository key
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# Add the ROCm repository
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update package lists
sudo apt update
```

### 4. Install ROCm

Install the ROCm packages:

```bash
sudo apt install -y rocm-dev
```

### 5. Set Up User Permissions

Add your user to the required groups:

```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
```

### 6. Configure Environment Variables

Add the following to your `~/.bashrc` file:

```bash
# ROCm Path
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin

# GPU Selection
export HIP_VISIBLE_DEVICES=0,1  # Adjust based on your GPU configuration
export CUDA_VISIBLE_DEVICES=0,1  # For CUDA compatibility layer
export PYTORCH_ROCM_DEVICE=0,1  # For PyTorch

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=1
```

Apply the changes:

```bash
source ~/.bashrc
```

### 7. Verify Installation

Verify that ROCm is installed correctly:

```bash
# Check ROCm version
rocminfo | grep "ROCm Version"

# List available GPUs
rocm-smi

# Check if HIP is working
hipconfig --full
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   - Check if the GPU is properly seated in the PCIe slot
   - Verify that the GPU is supported by ROCm
   - Check if the user has proper permissions

2. **Installation Fails**:
   - Check for dependency issues
   - Verify that your system meets the requirements
   - Check for conflicts with existing GPU drivers

3. **DKMS Module Build Failures**:
   - Ensure you have the correct kernel headers installed
   - Try installing with `--no-dkms` option

### Useful Commands

```bash
# Check ROCm installation
rocminfo

# Monitor GPU status
rocm-smi

# Check GPU temperature and usage
rocm-smi --showtemp --showuse

# Reset GPU if it's in a bad state
sudo rocm-smi --resetgpu
```

## Additional Resources

- [Official ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm GitHub Repository](https://github.com/RadeonOpenCompute/ROCm)
- [ROCm Community Forum](https://community.amd.com/t5/ROCm/bd-p/rocm)

## Next Steps

After installing ROCm, you can proceed to install PyTorch with ROCm support, which is covered in the [PyTorch ROCm Guide](/docs/core/pytorch_rocm_guide.md).


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

