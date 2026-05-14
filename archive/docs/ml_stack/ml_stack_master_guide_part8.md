## Troubleshooting

Common issues and their solutions.

### GPU Detection Issues

1. **GPU Not Detected**:
   ```
   No CUDA GPUs are available
   ```
   
   Solutions:
   - Check ROCm installation: `rocminfo`
   - Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
   - Check permissions: `groups` (should include video or render)
   - Update drivers: `sudo apt update && sudo apt upgrade`

2. **Multiple GPUs Not Detected**:
   ```
   Only one GPU is visible
   ```
   
   Solutions:
   - Set environment variables: `export HIP_VISIBLE_DEVICES=0,1`
   - Check PCIe configuration: `lspci | grep -i amd`
   - Verify ROCm multi-GPU support: `rocminfo`

### Memory Issues

1. **Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce batch size
   - Use mixed precision training
   - Use gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`
   - Monitor memory usage: `torch.cuda.memory_summary()`

2. **Memory Fragmentation**:
   ```
   RuntimeError: CUDA out of memory (fragmented memory)
   ```
   
   Solutions:
   - Clear cache periodically
   - Allocate tensors in order of size (largest first)
   - Use persistent RNN for recurrent models
   - Restart the process if fragmentation is severe

### Performance Issues

1. **Slow Training**:
   ```
   Training is slower than expected
   ```
   
   Solutions:
   - Profile with PyTorch Profiler
   - Check GPU utilization with ROCm SMI
   - Optimize data loading (more workers, pin_memory)
   - Use optimized kernels and operations
   - Check for CPU bottlenecks

2. **Low GPU Utilization**:
   ```
   GPU utilization is below 50%
   ```
   
   Solutions:
   - Increase batch size
   - Use more workers for data loading
   - Check for CPU bottlenecks
   - Use non-blocking transfers: `tensor.to('cuda', non_blocking=True)`
   - Profile with PyTorch Profiler to identify bottlenecks

### Compatibility Issues

1. **PyTorch Version Mismatch**:
   ```
   ImportError: cannot import name 'X' from 'torch'
   ```
   
   Solutions:
   - Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Install compatible version: `pip install torch==X.Y.Z+rocmA.B.C`
   - Check ROCm compatibility: `rocminfo`

2. **CUDA Compatibility Layer Issues**:
   ```
   AttributeError: module 'torch.cuda' has no attribute 'X'
   ```
   
   Solutions:
   - Use ROCm-specific APIs when available
   - Check PyTorch ROCm documentation
   - Update to latest PyTorch version with ROCm support

### Extension Component Issues

1. **Triton Compilation Errors**:
   ```
   Error: Failed to compile kernel
   ```
   
   Solutions:
   - Check ROCm version compatibility
   - Use simpler kernel implementations
   - Check for AMD-specific limitations

2. **BITSANDBYTES Quantization Errors**:
   ```
   RuntimeError: CUDA error: device-side assert triggered
   ```
   
   Solutions:
   - Check model compatibility with 8-bit quantization
   - Use different quantization parameters
   - Try different quantization methods

3. **vLLM Memory Errors**:
   ```
   RuntimeError: CUDA out of memory
   ```
   
   Solutions:
   - Reduce model size or sequence length
   - Adjust PagedAttention parameters
   - Use tensor parallelism across multiple GPUs

## References

1. [ROCm Documentation](https://rocm.docs.amd.com/)
2. [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
3. [AMD GPU Architecture Guide](https://www.amd.com/en/technologies/rdna-2)
4. [ONNX Runtime Documentation](https://onnxruntime.ai/)
5. [MIGraphX Documentation](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
6. [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
7. [Flash Attention Documentation](https://github.com/Dao-AILab/flash-attention)
8. [Triton Documentation](/docs/extensions/triton_guide.md)
9. [BITSANDBYTES Documentation](/docs/extensions/bitsandbytes_guide.md)
10. [vLLM Documentation](/docs/extensions/vllm_guide.md)
11. [ROCm SMI Documentation](/docs/extensions/rocm_smi_guide.md)
12. [PyTorch Profiler Documentation](/docs/extensions/pytorch_profiler_guide.md)
13. [WandB Documentation](/docs/extensions/wandb_guide.md)


## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

