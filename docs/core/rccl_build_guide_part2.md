## Installation Options

There are two ways to install RCCL:

1. **Install from ROCm Repository**: Easier but may not have the latest version
2. **Build from Source**: More complex but provides the latest version and customization options

### Install from ROCm Repository

The easiest way to install RCCL is from the ROCm repository:

```bash
# Update package lists
sudo apt update

# Install RCCL
sudo apt install rccl
```

Verify the installation:

```bash
# Check if RCCL is installed
ls -la /opt/rocm/lib/librccl*
```

You should see output similar to:

```
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2.0.0
```

## Building from Source

Building RCCL from source provides the latest features and optimizations.

### Clone the Repository

First, clone the RCCL repository:

```bash
# Create a directory for the build
mkdir -p $HOME/rccl-build
cd $HOME/rccl-build

# Clone the repository
git clone https://github.com/ROCmSoftwarePlatform/rccl.git
cd rccl
```

### Configure the Build

Configure the build with CMake:

```bash
# Create a build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$ROCM_PATH \
      -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc \
      -DCMAKE_C_COMPILER=$ROCM_PATH/bin/hipcc \
      ..
```

### Compile and Install

Compile and install RCCL:

```bash
# Compile
make -j$(nproc)

# Install (requires sudo)
sudo make install
```

### Verify Installation

Verify that RCCL is installed correctly:

```bash
# Check if RCCL is installed
ls -la $ROCM_PATH/lib/librccl*
```

You should see output similar to:

```
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2
-rwxr-xr-x 1 root root 12345678 Jan 1 2023 /opt/rocm/lib/librccl.so.2.0.0
```

### Build and Run Tests

Build and run the RCCL tests to verify functionality:

```bash
# Build tests
cd $HOME/rccl-build/rccl/build
make test

# Run tests
cd test
./all_reduce_test
```

You should see output indicating that the tests passed.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

