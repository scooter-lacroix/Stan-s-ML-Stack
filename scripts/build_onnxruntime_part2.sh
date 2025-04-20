#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
clone_repository() {
    print_section "Cloning ONNX Runtime repository"
    
    # Create build directory
    print_step "Creating build directory..."
    mkdir -p $HOME/onnxruntime-build
    cd $HOME/onnxruntime-build
    
    # Clone repository
    print_step "Cloning repository..."
    if [ -d "onnxruntime" ]; then
        print_warning "ONNX Runtime repository already exists. Updating..."
        cd onnxruntime
        git pull
        cd ..
    else
        git clone --recursive https://github.com/microsoft/onnxruntime.git
    fi
    
    print_success "Repository cloned successfully"
}

configure_build() {
    print_section "Configuring build"
    
    # Set ROCm path
    print_step "Setting ROCm path..."
    export ROCM_PATH=/opt/rocm
    
    # Set Python path
    print_step "Setting Python path..."
    export PYTHON_BIN_PATH=$(which python3)
    
    # Set build type
    print_step "Setting build type..."
    export BUILD_TYPE=Release
    
    # Create build configuration script
    print_step "Creating build configuration script..."
    cd $HOME/onnxruntime-build
    
    cat > build_rocm.sh << 'EOF'
#!/bin/bash

# Set build directory
BUILD_DIR="$HOME/onnxruntime-build/onnxruntime/build/Linux/Release"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure build with ROCm support
cmake ../../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/onnxruntime-build/onnxruntime/install \
    -Donnxruntime_RUN_ONNX_TESTS=OFF \
    -Donnxruntime_GENERATE_TEST_REPORTS=OFF \
    -Donnxruntime_USE_ROCM=ON \
    -Donnxruntime_ROCM_HOME=/opt/rocm \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_ENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$PYTHON_BIN_PATH \
    -Donnxruntime_USE_COMPOSABLE_KERNEL=ON \
    -Donnxruntime_USE_MIMALLOC=OFF \
    -Donnxruntime_ENABLE_ROCM_PROFILING=OFF

# Build ONNX Runtime
cmake --build . --config Release -- -j$(nproc)

# Install ONNX Runtime
cmake --install .

# Build Python wheel
cd ../../../
$PYTHON_BIN_PATH setup.py bdist_wheel
EOF
    
    chmod +x build_rocm.sh
    
    print_success "Build configured successfully"
}

build_onnxruntime() {
    print_section "Building ONNX Runtime"
    
    # Run build script
    print_step "Running build script..."
    cd $HOME/onnxruntime-build
    
    # Start the build in the background and show progress
    ./build_rocm.sh > build.log 2>&1 &
    build_pid=$!
    
    print_step "Building ONNX Runtime (this may take a while)..."
    show_progress $build_pid
    
    # Check if build was successful
    if wait $build_pid; then
        print_success "ONNX Runtime built successfully"
    else
        print_error "ONNX Runtime build failed. Check build.log for details."
        return 1
    fi
    
    return 0
}

install_onnxruntime() {
    print_section "Installing ONNX Runtime"
    
    # Install the Python wheel
    print_step "Installing Python wheel..."
    cd $HOME/onnxruntime-build/onnxruntime
    
    # Find the wheel file
    wheel_file=$(find dist -name "onnxruntime_rocm-*.whl")
    
    if [ -z "$wheel_file" ]; then
        print_error "Could not find ONNX Runtime wheel file."
        return 1
    fi
    
    # Install the wheel
    pip install $wheel_file
    
    print_success "ONNX Runtime installed successfully"
}

verify_installation() {
    print_section "Verifying installation"
    
    # Create test script
    print_step "Creating test script..."
    cd $HOME/onnxruntime-build
    
    cat > test_onnxruntime.py << 'EOF'
import onnxruntime as ort
import torch
import numpy as np

def test_onnxruntime():
    # Print ONNX Runtime version
    print(f"ONNX Runtime version: {ort.__version__}")
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(5, 2)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Create input data
    x = torch.randn(1, 10)
    
    # Export model to ONNX
    onnx_file = "simple_model.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_file,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Run inference with PyTorch
    with torch.no_grad():
        torch_output = model(x).numpy()
    
    # Create ONNX Runtime session
    if 'ROCMExecutionProvider' in providers:
        session = ort.InferenceSession(onnx_file, providers=['ROCMExecutionProvider'])
        print("Using ROCMExecutionProvider")
    else:
        session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        print("Using CPUExecutionProvider")
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference with ONNX Runtime
    ort_inputs = {input_name: x.numpy()}
    ort_output = session.run([output_name], ort_inputs)[0]
    
    # Compare PyTorch and ONNX Runtime outputs
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
    print("PyTorch and ONNX Runtime outputs match")
    
    print("ONNX Runtime test passed!")

if __name__ == "__main__":
    test_onnxruntime()
EOF
    
    # Run test script
    print_step "Running test script..."
    python3 test_onnxruntime.py
    
    if [ $? -eq 0 ]; then
        print_success "ONNX Runtime is working correctly"
        return 0
    else
        print_error "ONNX Runtime test failed"
        return 1
    fi
}
