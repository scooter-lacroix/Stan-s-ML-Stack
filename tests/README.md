# ML Stack Tests

This directory contains tests for various components of the ML Stack on AMD GPUs.

## Available Tests

1. **GPU Detection Test**: Tests if AMD GPUs are properly detected by PyTorch.
2. **Flash Attention Test**: Tests if Flash Attention is working correctly on AMD GPUs.
3. **MPI Test**: Tests if MPI is working correctly with AMD GPUs.
4. **ONNX Runtime Test**: Tests if ONNX Runtime is working correctly with AMD GPUs.

## Running Tests

### GPU Detection Test

```bash
python test_gpu_detection.py
```

This test checks if:
- CUDA (ROCm) is available
- AMD GPUs are detected
- Environment variables are set correctly
- Simple tensor operations work on GPU

### Flash Attention Test

```bash
python test_flash_attention.py
```

This test checks if:
- Flash Attention module can be imported
- Flash Attention computation works
- Flash Attention output matches standard attention output
- Flash Attention works with different sequence lengths, batch sizes, and head dimensions

### MPI Test

```bash
mpirun -np 2 python test_mpi.py
```

This test checks if:
- MPI is properly initialized
- Basic MPI operations (broadcast, reduce, allreduce) work
- GPU operations work with MPI
- MPI works with GPU data

### ONNX Runtime Test

```bash
python test_onnx.py
```

This test checks if:
- ONNX Runtime can be imported
- ROCMExecutionProvider is available
- Model can be exported to ONNX
- ONNX Runtime inference works
- PyTorch and ONNX Runtime outputs match
- ONNX Runtime works with different batch sizes

## Interpreting Results

Each test will print colorful output indicating success or failure:

- **Green**: Success messages
- **Blue**: Information messages
- **Yellow**: Warning messages
- **Red**: Error messages

The tests will also exit with a status code:
- **0**: All tests passed
- **1**: One or more tests failed

## Example Output

### GPU Detection Test

```
=== GPU Detection Test ===
INFO: PyTorch version: 2.6.0+rocm6.2.4
SUCCESS: CUDA (ROCm) is available
SUCCESS: Number of GPUs: 2
SUCCESS: GPU 0: AMD Radeon RX 7900 XTX
SUCCESS: GPU 1: AMD Radeon RX 7800 XT
INFO: Environment Variables:
  HIP_VISIBLE_DEVICES: 0,1
  CUDA_VISIBLE_DEVICES: 0,1
  PYTORCH_ROCM_DEVICE: 0,1
SUCCESS: Simple tensor operation on GPU successful
SUCCESS: Matrix multiplication on GPU successful
SUCCESS: All GPU detection tests passed
```

### Flash Attention Test

```
=== Flash Attention Test ===
SUCCESS: Flash Attention module imported successfully
SUCCESS: Standard attention computation successful
SUCCESS: Flash Attention computation successful
INFO: Maximum difference between standard and Flash Attention: 0.000123
SUCCESS: Flash Attention output matches standard attention output
INFO: Testing with sequence length 128
SUCCESS: Flash Attention computation successful for sequence length 128
INFO: Testing with sequence length 256
SUCCESS: Flash Attention computation successful for sequence length 256
INFO: Testing with sequence length 512
SUCCESS: Flash Attention computation successful for sequence length 512
INFO: Testing with batch size 1
SUCCESS: Flash Attention computation successful for batch size 1
INFO: Testing with batch size 4
SUCCESS: Flash Attention computation successful for batch size 4
INFO: Testing with batch size 8
SUCCESS: Flash Attention computation successful for batch size 8
INFO: Testing with head dimension 32
SUCCESS: Flash Attention computation successful for head dimension 32
INFO: Testing with head dimension 128
SUCCESS: Flash Attention computation successful for head dimension 128
SUCCESS: All Flash Attention tests passed
```

## Notes

- The tests are designed to run on AMD GPUs with ROCm support.
- Some tests may take a few minutes to complete, especially the Flash Attention and ONNX Runtime tests.
- The MPI test requires at least 2 processes to run properly.
- The tests use the CUDA compatibility layer provided by ROCm, so they use CUDA terminology (e.g., `torch.cuda.is_available()`).


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

