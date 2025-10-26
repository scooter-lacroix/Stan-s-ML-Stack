# Stan's ML Stack - Multi-Channel ROCm Guide

Stan's ML Stack supports four ROCm channels:

| Channel | ROCm Version | Recommended For |
|---------|--------------|-----------------|
| Legacy  | 6.4.3        | Maximum stability (RDNA 1/2) |
| Stable  | 7.0.0        | Production RDNA 3 environments |
| Latest  | 7.0.2        | Default choice, RDNA 3/4 |
| Preview | 7.9.0        | Experimental testing via TheRock |

## Installing ROCm

Interactive mode:

```bash
./scripts/install_rocm.sh
```

Non-interactive mode:

```bash
./scripts/install_rocm_channel.sh latest   # legacy|stable|latest|preview
```

## Component Installers

```bash
./scripts/install_pytorch_multi.sh
./scripts/install_triton_multi.sh
./scripts/build_flash_attn_amd.sh
./scripts/install_vllm_multi.sh
./scripts/build_onnxruntime_multi.sh
./scripts/install_migraphx_multi.sh
./scripts/install_bitsandbytes_multi.sh
./scripts/install_rccl_multi.sh
```

## Verification

```bash
./scripts/enhanced_verify_installation.sh
```

## Environment Variables

`~/.mlstack_env` now includes:

- `ROCM_VERSION`
- `ROCM_CHANNEL`
- `GPU_ARCH`

These variables are respected by the helper scripts above.
