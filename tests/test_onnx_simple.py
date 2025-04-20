import os
import sys

# Add the ONNX Runtime build directory to the Python path
sys.path.insert(0, "/home/stan/onnxruntime_build/onnxruntime/build/Linux/Release")

# Set environment variables for ROCm
os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ROCM_DEVICE"] = "0,1"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["GPU_MAX_HEAP_SIZE"] = "100"
os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
os.environ["HSA_TOOLS_LIB"] = "1"
os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "3"
os.environ["MIOPEN_FIND_ENFORCE"] = "3"

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    if "ROCMExecutionProvider" in ort.get_available_providers():
        print("ROCMExecutionProvider is available")
    else:
        print("ROCMExecutionProvider is not available")
        
    # Create a simple model
    import torch
    import numpy as np
    
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
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Model exported to ONNX successfully")
    
    # Run inference with PyTorch
    with torch.no_grad():
        torch_output = model(x).numpy()
    
    # Create ONNX Runtime session
    if "ROCMExecutionProvider" in ort.get_available_providers():
        session = ort.InferenceSession(onnx_file, providers=["ROCMExecutionProvider"])
        print("Created ONNX Runtime session with ROCMExecutionProvider")
    else:
        session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        print("Created ONNX Runtime session with CPUExecutionProvider")
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference with ONNX Runtime
    ort_inputs = {input_name: x.numpy()}
    ort_output = session.run([output_name], ort_inputs)[0]
    print("ONNX Runtime inference successful")
    
    # Compare PyTorch and ONNX Runtime outputs
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
    print("PyTorch and ONNX Runtime outputs match")
    
    print("All tests passed!")
except Exception as e:
    print(f"Error: {e}")
