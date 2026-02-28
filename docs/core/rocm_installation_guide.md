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

## Rusty-Stack Managed ROCm Flow (Recommended)

For production installs, use the Rusty-Stack ROCm component instead of ad-hoc manual steps. The installer now implements a strict force-reinstall workflow that is designed for broken or partial ROCm states.

### Supported Linux Package Families

- Debian/Ubuntu (`apt`/`dpkg`)
- Fedora/RHEL (`dnf`/`yum`)
- openSUSE (`zypper`)
- Arch/CachyOS/Manjaro (`pacman` + `yay`/`paru` when AUR packages are required)

### Force Reinstall Workflow

When `force reinstall` is enabled in preinstall configuration, the installer performs:

1. Full ROCm/AMDGPU purge with dependency-cycle handling.
2. Purge validation pass to ensure packages are gone.
3. 10-second reboot countdown and `sudo reboot`.
4. Resume install on next login/session (autostart where available).
5. ROCm installation pass.
6. Mandatory second reboot to load drivers cleanly.

This process intentionally separates uninstall and reinstall phases to avoid mixed-driver states.

### Arch-Specific Notes

- AUR helpers are run as non-root user.
- Repository packages are installed via `pacman`.
- AUR package availability is checked before install.
- A sudo ticket keepalive is used so long-running AUR builds do not fail on auth timeout.

### Persistent Environment Setup (bash/zsh/fish)

After ROCm installation, run:

```bash
./scripts/setup_permanent_rocm_env.sh
```

The generated `~/.mlstack_env` now:

- Is sourceable from bash, zsh, and fish startup files.
- Exports ROCm runtime variables and Triton cache paths.
- Filters integrated GPUs and only exports discrete GPU indices in `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

### Post-Install Validation

Run the following before benchmarking:

```bash
source ~/.mlstack_env
rocminfo | head -n 40
rocm-smi
python3 - <<'PY'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
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
