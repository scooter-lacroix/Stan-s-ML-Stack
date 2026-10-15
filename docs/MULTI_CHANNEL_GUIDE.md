# Stan's ML Stack - Multi-Channel ROCm Guide

Stan's ML Stack supports three ROCm channels:

| Channel | ROCm Version | Recommended For |
|---------|--------------|-----------------|
| Legacy  | 6.4.3        | Conservative compatibility (RDNA 1/2) |
| Stable  | 7.2.3        | Production RDNA 3/4 environments |
| Latest  | 7.2.4        | Default choice, current ROCm production release |

**Note:** AMD's newest releases (7.14.x, 10.0, 10.1.0 as of Oct 2026) use the 'TheRock' distribution (pip/tarball/runfile). The ROCm system installer does not yet offer them as a channel, but PyTorch installs for ROCm 10.x resolve to AMD's wheel index (`https://stable.repo.amd.com/rocm/whl-next/`, `torch[device-all]==2.14.0+rocm10.1.0`). See: https://rocm.docs.amd.com/en/latest/install/rocm.html

## Installing ROCm

Interactive mode:

```bash
./scripts/install_rocm.sh
```

Non-interactive mode:

```bash
./scripts/install_rocm_channel.sh latest   # legacy|stable|latest
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
