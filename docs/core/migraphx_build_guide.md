# MIGraphX Build Guide for AMD GPUs

## Introduction

This guide provides detailed instructions for building and installing MIGraphX for AMD GPUs. MIGraphX is AMD's graph optimization library for deep learning, providing performance optimizations for neural network models.

## Prerequisites

Before building MIGraphX, ensure you have:

1. **ROCm Installed**: Follow the [ROCm Installation Guide](/docs/core/rocm_installation_guide.md)
2. **Python Environment**: Python 3.8+ with pip
3. **System Dependencies**: Required system libraries and tools
4. **GPU Access**: Proper permissions to access AMD GPUs
5. **Disk Space**: At least 5GB of free disk space for the build

## Installation Options

There are two ways to install MIGraphX:

1. **Install from ROCm Repository**: Easier but may not have the latest version
2. **Build from Source**: More complex but provides the latest version and customization options

## Option 1: Install from ROCm Repository

### 1. Install MIGraphX Package

```bash
sudo apt-get update
sudo apt-get install -y migraphx
```

### 2. Install Python Wrapper

```bash
sudo apt-get install -y python3-migraphx
```

### 3. Verify Installation

```bash
python3 -c "import migraphx; print(migraphx.__version__)"
```

## Option 2: Build from Source

### 1. Install Build Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    libpython3-dev \
    libnuma-dev \
    half \
    libprotobuf-dev \
    protobuf-compiler \
    libopenblas-dev
```

### 2. Clone MIGraphX Repository

```bash
cd $HOME
git clone https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git
cd AMDMIGraphX
```

### 3. Configure Build

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/migraphx \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DGPU_TARGETS="gfx1100;gfx1101;gfx1102" \
    -DBUILD_DEV=On
```

Note: Adjust `GPU_TARGETS` based on your GPU architecture. For RX 7900 XTX and RX 7800 XT, use `gfx1100;gfx1101;gfx1102`.

### 4. Build and Install

```bash
make -j$(nproc)
sudo make install
```

### 5. Build Python Wrapper

```bash
cd ../src/py
python3 setup.py build
sudo python3 setup.py install
```

### 6. Configure Environment

Add the following to your `~/.bashrc` file:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/migraphx/lib
export PYTHONPATH=$PYTHONPATH:/opt/rocm/migraphx/lib/python3/site-packages
```

Apply the changes:

```bash
source ~/.bashrc
```

### 7. Verify Installation

```bash
python3 -c "import migraphx; print(migraphx.__version__)"
```

## Usage Examples

### Basic Example: ONNX Model Import

```python
import migraphx
import numpy as np

# Create a MIGraphX model
model = migraphx.parse_onnx("path/to/model.onnx")

# Compile the model for GPU
model.compile(migraphx.get_target("gpu"))

# Prepare input data
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Create parameter map
param_map = {"input": input_data}

# Run inference
results = model.run(param_map)

# Get output
output = results[0]
print(output.shape)
```

### Optimizing a Model

```python
import migraphx

# Create a MIGraphX model
model = migraphx.parse_onnx("path/to/model.onnx")

# Print the original model
print("Original model:")
print(model)

# Optimize the model
model.compile(migraphx.get_target("gpu"))

# Print the optimized model
print("Optimized model:")
print(model)

# Save the optimized model
model.save("optimized_model.mxr")
```

### Loading a Saved Model

```python
import migraphx

# Load a saved model
model = migraphx.load("optimized_model.mxr")

# Compile the model for GPU
model.compile(migraphx.get_target("gpu"))

# Run inference
# ...
```

## Troubleshooting

### Common Issues

1. **Import Error**:
   ```
   ImportError: No module named migraphx
   ```
   
   Solutions:
   - Check if MIGraphX is installed: `pip list | grep migraphx`
   - Verify PYTHONPATH: `echo $PYTHONPATH`
   - Reinstall the Python wrapper

2. **Library Not Found**:
   ```
   ImportError: libmigraphx.so: cannot open shared object file: No such file or directory
   ```
   
   Solutions:
   - Check LD_LIBRARY_PATH: `echo $LD_LIBRARY_PATH`
   - Verify installation path: `ls -la /opt/rocm/migraphx/lib`
   - Reinstall MIGraphX

3. **Compilation Errors**:
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify GPU targets match your hardware
   - Update to the latest ROCm version

4. **ONNX Import Errors**:
   
   Solutions:
   - Check ONNX model version compatibility
   - Try simplifying the model
   - Check for unsupported operators

## Performance Optimization

### Model Optimization

1. **Quantization**: Quantize models to improve performance
   ```python
   import migraphx
   
   # Load model
   model = migraphx.parse_onnx("model.onnx")
   
   # Quantize model
   model.quantize_fp16()
   
   # Compile and save
   model.compile(migraphx.get_target("gpu"))
   model.save("model_fp16.mxr")
   ```

2. **Operator Fusion**: MIGraphX automatically fuses operators during compilation
   ```python
   # Compile with fusion enabled
   model.compile(migraphx.get_target("gpu"))
   ```

3. **Memory Optimization**: Optimize memory usage
   ```python
   # Enable memory optimization
   model.compile(migraphx.get_target("gpu"), offload_copy=True)
   ```

## Additional Resources

- [MIGraphX GitHub Repository](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
- [MIGraphX Documentation](https://rocmsoftwareplatform.github.io/AMDMIGraphX/doc/html/)
- [MIGraphX API Reference](https://rocmsoftwareplatform.github.io/AMDMIGraphX/doc/html/reference/cpp.html)

## Next Steps

After installing MIGraphX, you can proceed to install other components of the ML stack, such as Megatron-LM and Flash Attention.


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

