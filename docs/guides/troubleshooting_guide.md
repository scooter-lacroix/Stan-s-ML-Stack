# ML Stack Troubleshooting Guide

## Introduction
## Hardware Issues
## ROCm Issues
## PyTorch Issues
## ONNX Runtime Issues
## Environment Variable Issues
## Python Compatibility Issues
## Build and Installation Issues
## Performance Issues
## Common Error Messages
# ML Stack Troubleshooting Guide

## Introduction

This guide provides solutions to common issues encountered when using Stan's ML Stack with AMD GPUs. It covers hardware detection, ROCm installation, PyTorch compatibility, ONNX Runtime, environment variables, and more.

If you encounter an issue not covered in this guide, please check the [GitHub Issues](https://github.com/scooter-lacroix/Stans_MLStack/issues) or create a new issue.

## Hardware Issues

### AMD GPU Not Detected

**Symptoms**:
- `lspci` doesn't show your AMD GPU
- ROCm tools don't detect your GPU
- PyTorch reports 0 available GPUs

**Solutions**:

1. **Check physical installation**:
   - Ensure the GPU is properly seated in the PCIe slot
   - Check power connections
   - Verify the GPU is powered on (fans spinning, lights on)

2. **Check system recognition**:
   ```bash
   lspci | grep -i 'amd\|radeon\|advanced micro devices'
   ```
   If your GPU is not listed, it may not be properly connected or recognized by the system.

3. **Update PCI IDs database**:
   ```bash
   sudo update-pciids
   ```

4. **Check BIOS settings**:
   - Ensure PCIe is enabled
   - Set primary display to PCIe or discrete GPU
   - Disable CSM if using UEFI

5. **Check kernel modules**:
   ```bash
   lsmod | grep amdgpu
   ```
   If not loaded, load the module:
   ```bash
   sudo modprobe amdgpu
   ```

6. **Check user permissions**:
   ```bash
   sudo usermod -a -G video,render $USER
   ```
   Log out and log back in for changes to take effect.

### Multiple GPUs Not All Detected

**Symptoms**:
- Only some of your GPUs are detected
- `rocminfo` shows fewer GPUs than physically installed

**Solutions**:

1. **Check PCIe lanes and power**:
   - Ensure all GPUs have sufficient PCIe lanes
   - Verify all GPUs have proper power connections

2. **Set environment variables**:
   ```bash
   export HIP_VISIBLE_DEVICES=0,1  # Adjust based on your GPU count
   export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU count
   export PYTORCH_ROCM_DEVICE=0,1  # Adjust based on your GPU count
   ```

3. **Check ROCm compatibility**:
   - Verify all GPUs are supported by your ROCm version
   - Check the [ROCm Hardware Compatibility Matrix](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)

## ROCm Issues

### ROCm Installation Fails

**Symptoms**:
- Installation script exits with errors
- ROCm tools are not available after installation

**Solutions**:

1. **Check system requirements**:
   - Ensure your system meets the [ROCm system requirements](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
   - Verify your kernel version is supported

2. **Clean previous installations**:
   ```bash
   sudo apt purge rocm-* hip-* miopen-* migraphx-*
   sudo apt autoremove
   ```

3. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install libnuma-dev
   ```

4. **Try alternative installation method**:
   - Use the ROCm package repository
   - Try the tarball installation
   - Use the ML Stack installation script

### ROCm Version Compatibility

**Symptoms**:
- Components fail with version mismatch errors
- Libraries cannot be found

**Solutions**:

1. **Check ROCm version**:
   ```bash
   rocminfo | grep -i "ROCm Version"
   ```

2. **Ensure consistent versions**:
   - PyTorch should match ROCm version (e.g., PyTorch 2.6.0+rocm6.2.4 for ROCm 6.2.4)
   - ONNX Runtime should be built against the same ROCm version

3. **Set ROCm path**:
   ```bash
   export ROCM_PATH=/opt/rocm
   export PATH=$PATH:$ROCM_PATH/bin:$ROCM_PATH/hip/bin
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH
   ```

### "Tool lib '1' failed to load" Warning

**Symptoms**:
- Warning message appears when running ROCm applications
- Message: "Tool lib '1' failed to load"

**Solutions**:

1. **Set HSA_TOOLS_LIB environment variable**:
   ```bash
   export HSA_TOOLS_LIB=1
   ```

2. **Add to your environment file**:
   Add the above line to your `~/.mlstack_env` file.

3. **Ignore the warning**:
   This warning is harmless and doesn't affect functionality. It's related to ROCm's profiling tools.
## PyTorch Issues

### PyTorch Not Using GPU

**Symptoms**:
- `torch.cuda.is_available()` returns `False`
- Models run slowly on CPU instead of GPU
- `torch.cuda.device_count()` returns 0

**Solutions**:

1. **Check ROCm installation**:
   ```bash
   rocminfo
   ```
   Ensure ROCm is properly installed and your GPU is detected.

2. **Verify PyTorch ROCm build**:
   ```python
   import torch
   print(torch.version.hip)  # Should not be None
   ```
   If it's `None`, you don't have a ROCm-enabled PyTorch build.

3. **Install ROCm-enabled PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```
   Adjust the ROCm version as needed.

4. **Set environment variables**:
   ```bash
   export HIP_VISIBLE_DEVICES=0,1
   export CUDA_VISIBLE_DEVICES=0,1
   export PYTORCH_ROCM_DEVICE=0,1
   ```

5. **Check GPU architecture compatibility**:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```
   This overrides the GPU architecture version for compatibility.

### PyTorch Out of Memory Errors

**Symptoms**:
- CUDA out of memory errors
- Models crash with memory errors
- Training fails with large batch sizes

**Solutions**:

1. **Increase memory allocation limits**:
   ```bash
   export GPU_MAX_HEAP_SIZE=100
   export GPU_MAX_ALLOC_PERCENT=100
   ```

2. **Set PyTorch memory allocation parameters**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
   export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"
   ```

3. **Reduce batch size**:
   Decrease the batch size in your training code.

4. **Enable gradient checkpointing**:
   Use gradient checkpointing to reduce memory usage at the cost of increased computation.

5. **Use mixed precision training**:
   Implement mixed precision training (FP16/BF16) to reduce memory usage.

### PyTorch Performance Issues

**Symptoms**:
- Models run slower than expected
- GPU utilization is low
- Training takes longer than on NVIDIA GPUs

**Solutions**:

1. **Set performance environment variables**:
   ```bash
   export HSA_ENABLE_SDMA=0
   export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
   export MIOPEN_FIND_MODE=3
   export MIOPEN_FIND_ENFORCE=3
   ```

2. **Use optimized operators**:
   - Use Flash Attention for transformer models
   - Use optimized convolution implementations

3. **Benchmark different batch sizes**:
   Find the optimal batch size for your model and GPU.

4. **Update to latest ROCm and PyTorch versions**:
   Newer versions often include performance improvements.

5. **Profile your code**:
   Use PyTorch Profiler to identify bottlenecks.

## ONNX Runtime Issues

### ROCMExecutionProvider Not Available

**Symptoms**:
- `onnxruntime.get_available_providers()` doesn't include `ROCMExecutionProvider`
- ONNX Runtime falls back to CPU execution
- Error: "ROCMExecutionProvider is not in the available providers"

**Solutions**:

1. **Build ONNX Runtime with ROCm support**:
   ```bash
   ./scripts/build_onnxruntime.sh
   ```
   This script builds ONNX Runtime from source with ROCm support.

2. **Set PYTHONPATH to include ONNX Runtime build**:
   ```bash
   export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
   ```

3. **Verify ROCm support**:
   ```python
   import onnxruntime
   print(onnxruntime.get_available_providers())
   ```
   It should include `ROCMExecutionProvider`.

4. **Set environment variables**:
   ```bash
   export CUDA_HOME=/opt/rocm
   export ROCM_HOME=/opt/rocm
   ```

### ONNX Runtime Performance Issues

**Symptoms**:
- Inference is slower than expected
- GPU utilization is low
- CPU is used instead of GPU

**Solutions**:

1. **Ensure ROCMExecutionProvider is used**:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession(model_path, providers=['ROCMExecutionProvider'])
   ```
   Explicitly specify the provider.

2. **Optimize the ONNX model**:
   ```python
   import onnxruntime as ort
   from onnxruntime.transformers import optimizer
   optimized_model = optimizer.optimize_model(model_path)
   ```

3. **Use MIGraphX for graph optimization**:
   ```bash
   migraphx-driver optimize model.onnx optimized_model.onnx
   ```

4. **Set performance environment variables**:
   ```bash
   export HSA_ENABLE_SDMA=0
   export GPU_MAX_HEAP_SIZE=100
   export GPU_MAX_ALLOC_PERCENT=100
   ```

5. **Use appropriate input shapes**:
   Avoid dynamic shapes when possible, as they can reduce optimization opportunities.
## Environment Variable Issues

### Environment Variables Not Persisting After Reboot

**Symptoms**:
- Environment variables are lost after system reboot
- ML Stack components stop working after reboot
- Symlinks are missing after system updates

**Solutions**:

1. **Use the persistent environment setup script**:
   ```bash
   sudo ./scripts/create_persistent_env.sh
   ```
   This script creates a system-wide environment file and sets up a systemd service to ensure all necessary symlinks are created at boot time.

2. **Verify the environment setup**:
   ```bash
   sudo verify-mlstack-env.sh
   ```
   This command checks if all environment variables and symlinks are correctly set up.

3. **Check the systemd service status**:
   ```bash
   sudo systemctl status mlstack-symlinks.service
   ```
   If the service is not running, start it:
   ```bash
   sudo systemctl start mlstack-symlinks.service
   ```

### Environment Variables Not Persisting Between Terminals

**Symptoms**:
- Environment variables are lost after logging out
- Variables set in one terminal don't affect other terminals
- Applications can't find libraries or paths

**Solutions**:

1. **Use the ML Stack environment file**:
   ```bash
   source ~/.mlstack_env
   ```
   This file is created by the `enhanced_setup_environment.sh` script.

2. **Add to .bashrc or .zshrc**:
   ```bash
   echo 'source ~/.mlstack_env' >> ~/.bashrc
   ```
   This ensures the environment is loaded in every new terminal.

3. **Create a systemwide environment file**:
   ```bash
   sudo bash -c 'cat > /etc/profile.d/mlstack.sh << EOF
   # ML Stack Environment Variables
   export ROCM_PATH=/opt/rocm
   export PATH=\$PATH:\$ROCM_PATH/bin:\$ROCM_PATH/hip/bin
   export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$ROCM_PATH/hip/lib:\$LD_LIBRARY_PATH
   # Add other variables as needed
   EOF'
   ```
   This makes the environment available to all users.

### CUDA_HOME Not Set

**Symptoms**:
- Applications fail with "CUDA_HOME not set" errors
- Build processes fail to find CUDA
- Libraries can't locate CUDA dependencies

**Solutions**:

1. **Set CUDA_HOME to ROCm path**:
   ```bash
   export CUDA_HOME=/opt/rocm
   ```
   This provides compatibility with CUDA-based applications.

2. **Add to environment file**:
   Add the above line to your `~/.mlstack_env` file.

3. **Create a symlink**:
   ```bash
   sudo ln -sf /opt/rocm /usr/local/cuda
   ```
   This creates a symlink that some applications look for.

### LD_LIBRARY_PATH Issues

**Symptoms**:
- "Library not found" errors
- Applications can't find shared libraries
- Conflicts between different library versions

**Solutions**:

1. **Set LD_LIBRARY_PATH correctly**:
   ```bash
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/opencl/lib:$LD_LIBRARY_PATH
   ```

2. **Check library locations**:
   ```bash
   ldconfig -p | grep libname
   ```
   Replace `libname` with the name of the missing library.

3. **Create symlinks if needed**:
   ```bash
   sudo ln -sf /path/to/existing/lib.so /path/to/expected/lib.so
   ```

4. **Update ldconfig cache**:
   ```bash
   sudo ldconfig
   ```

## Python Compatibility Issues

### Python 3.13 Compatibility

**Symptoms**:
- Packages fail to install with Python 3.13
- Error messages about unsupported Python version
- ImportError or ModuleNotFoundError

**Solutions**:

1. **Use vLLM Python 3.13 Workaround**:
   ```bash
   ./scripts/install_vllm.sh
   ```
   This script installs a custom version of vLLM compatible with Python 3.13.

2. **Modify package requirements**:
   For other packages, you can modify their setup files to accept Python 3.13:
   ```bash
   sed -i 's/python_requires=">=3.8,<3.13"/python_requires=">=3.8,<3.14"/' setup.py
   ```

3. **Use a virtual environment with an older Python version**:
   ```bash
   sudo apt install python3.10
   python3.10 -m venv venv_py310
   source venv_py310/bin/activate
   ```

4. **Check for updated versions**:
   Many packages are actively adding Python 3.13 support. Check for newer versions.

### Package Version Conflicts

**Symptoms**:
- Dependency resolution errors during installation
- Packages fail to import due to version conflicts
- Error messages about incompatible dependencies

**Solutions**:

1. **Use virtual environments**:
   ```bash
   python -m venv mlstack_env
   source mlstack_env/bin/activate
   ```
   This isolates your ML Stack from other Python environments.

2. **Pin specific versions**:
   Create a requirements.txt file with exact versions of all dependencies.

3. **Use compatibility layers**:
   For some packages, compatibility layers or forks are available that work with newer versions.

4. **Install with --no-deps and manage dependencies manually**:
   ```bash
   pip install --no-deps package_name
   ```
   Then manually install compatible versions of dependencies.
## Build and Installation Issues

### Ninja Build Issues

**Symptoms**:
- Build fails with "ninja-build: command not found"
- Error: "Could not find ninja. Looked for: ninja, ninja-build"
- CMake configuration fails

**Solutions**:

1. **Install ninja**:
   ```bash
   sudo apt install ninja-build
   ```

2. **Create symlink**:
   ```bash
   sudo ln -sf /usr/bin/ninja-build /usr/bin/ninja
   ```
   or
   ```bash
   sudo ln -sf /usr/bin/ninja /usr/bin/ninja-build
   ```
   depending on which one is installed.

3. **Add to PATH**:
   ```bash
   export PATH=/path/to/ninja:$PATH
   ```

### CMake Issues

**Symptoms**:
- Build fails with CMake errors
- "CMake version X.X or higher is required"
- Configuration or generation errors

**Solutions**:

1. **Install latest CMake**:
   ```bash
   sudo apt remove --purge cmake
   wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
   chmod +x cmake-3.28.0-linux-x86_64.sh
   sudo ./cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --skip-license
   ```

2. **Set CMake path**:
   ```bash
   export PATH=/usr/local/bin:$PATH
   ```

3. **Clear CMake cache**:
   ```bash
   rm -rf CMakeCache.txt CMakeFiles/
   ```
   Run this in the build directory before retrying.

### Compilation Errors

**Symptoms**:
- Build fails with compilation errors
- "error: unknown type name" or similar errors
- Linker errors

**Solutions**:

1. **Install build dependencies**:
   ```bash
   sudo apt install build-essential g++ python3-dev
   ```

2. **Check compiler version**:
   ```bash
   g++ --version
   ```
   Ensure you have a compatible compiler version.

3. **Set compiler flags**:
   ```bash
   export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
   ```
   This may be needed for some libraries.

4. **Clean build directory**:
   ```bash
   rm -rf build/
   mkdir build
   ```
   Start with a clean build directory.

### Installation Path Issues

**Symptoms**:
- Libraries installed but not found
- "No module named X" errors
- Import errors despite successful installation

**Solutions**:

1. **Check installation path**:
   ```bash
   pip show package_name
   ```
   This shows where the package is installed.

2. **Set PYTHONPATH**:
   ```bash
   export PYTHONPATH=/path/to/package:$PYTHONPATH
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .
   ```
   This creates links instead of copying files.

4. **Use absolute paths in scripts**:
   ```python
   import os
   import sys
   sys.path.append(os.path.abspath('/path/to/package'))
   ```

## Performance Issues

### Low GPU Utilization

**Symptoms**:
- GPU utilization is low (check with `rocm-smi`)
- Models run slower than expected
- CPU usage is high while GPU usage is low

**Solutions**:

1. **Set performance environment variables**:
   ```bash
   export HSA_ENABLE_SDMA=0
   export GPU_MAX_HEAP_SIZE=100
   export GPU_MAX_ALLOC_PERCENT=100
   export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
   export MIOPEN_FIND_MODE=3
   export MIOPEN_FIND_ENFORCE=3
   ```

2. **Increase batch size**:
   Larger batch sizes often lead to better GPU utilization.

3. **Use optimized operators**:
   - Flash Attention for transformer models
   - Optimized convolution implementations

4. **Check data loading**:
   Ensure your data loading pipeline isn't bottlenecking the GPU.

5. **Profile your code**:
   Use PyTorch Profiler to identify bottlenecks.

### Memory Leaks

**Symptoms**:
- Memory usage increases over time
- Out of memory errors after running for a while
- Performance degrades over time

**Solutions**:

1. **Clear PyTorch cache periodically**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Delete unused tensors**:
   ```python
   del tensor
   torch.cuda.empty_cache()
   ```

3. **Use context managers for large operations**:
   ```python
   with torch.no_grad():
       # operations here
   ```

4. **Check for reference cycles**:
   Use tools like `objgraph` to find reference cycles.

5. **Monitor memory usage**:
   ```bash
   watch -n 1 rocm-smi
   ```

### Slow Data Loading

**Symptoms**:
- GPU is idle waiting for data
- Training is bottlenecked by data loading
- CPU usage is high during data loading

**Solutions**:

1. **Use PyTorch DataLoader with multiple workers**:
   ```python
   dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
   ```

2. **Prefetch data**:
   ```python
   prefetch_factor=2
   ```

3. **Use memory mapping for large datasets**:
   ```python
   import numpy as np
   data = np.memmap('data.npy', dtype='float32', mode='r', shape=(1000, 1000))
   ```

4. **Cache processed data**:
   Save preprocessed data to disk to avoid redundant processing.

5. **Use efficient data formats**:
   HDF5, Parquet, or other columnar formats can be more efficient than CSV.

## Common Error Messages

### "Tool lib '1' failed to load"

**Error**:
```
Tool lib '1' failed to load
```

**Solution**:
This warning is harmless and doesn't affect functionality. Set the following environment variable:
```bash
export HSA_TOOLS_LIB=1
```

### "CUDA_HOME is not set"

**Error**:
```
AssertionError: CUDA_HOME is not set
```

**Solution**:
Set CUDA_HOME to point to your ROCm installation:
```bash
export CUDA_HOME=/opt/rocm
```

### "No module named 'onnxruntime'"

**Error**:
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Solution**:
Ensure ONNX Runtime is installed and in your PYTHONPATH:
```bash
export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
```

### "CUDA out of memory"

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solution**:
Increase memory allocation limits and reduce batch size:
```bash
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
```

### "RuntimeError: HIP error: no ROCm devices"

**Error**:
```
RuntimeError: HIP error: no ROCm devices
```

**Solution**:
Ensure ROCm is properly installed and your GPU is detected:
```bash
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_DEVICE=0,1
```

### "Python version not supported"

**Error**:
```
ERROR: Package 'X' requires a different Python: 3.13.X not in '>=3.8,<3.13'
```

**Solution**:
Use the provided workarounds or modify the package requirements:
```bash
sed -i 's/python_requires=">=3.8,<3.13"/python_requires=">=3.8,<3.14"/' setup.py
```

### "ninja: command not found"

**Error**:
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja". CMAKE_MAKE_PROGRAM is not set.
```

**Solution**:
Install ninja and create the necessary symlinks:
```bash
sudo apt install ninja-build
sudo ln -sf /usr/bin/ninja-build /usr/bin/ninja
```

### "Library not found: librocm_smi64.so"

**Error**:
```
ImportError: librocm_smi64.so: cannot open shared object file: No such file or directory
```

**Solution**:
Add ROCm libraries to LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH
```

### "ROCMExecutionProvider is not in the available providers"

**Error**:
```
ValueError: 'ROCMExecutionProvider' is not in the available providers
```

**Solution**:
Build ONNX Runtime with ROCm support and set the correct environment variables:
```bash
./scripts/build_onnxruntime.sh
export PYTHONPATH=/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release:$PYTHONPATH
```
