# Stan's ML Stack – ROCm Multi-Version Guide

This guide explains how to install Stan's ML Stack using the new ROCm version options that are available inside `scripts/install_rocm.sh`.

> **Important**
> - ROCm 6.4.3, 7.0.0, and 7.0.2 are installed via the traditional `amdgpu-install` packages.
> - ROCm 7.9.0 is a *technology preview* delivered via AMD's **TheRock** tarball distribution. It is **not** intended for production workloads.
> - All changes were made without removing any existing logic from the original installer.

## 1. Running the Installer

```bash
cd /path/to/stan_ml_stack
chmod +x scripts/install_rocm.sh
./scripts/install_rocm.sh
```

When prompted, choose from the following options:

| Option | ROCm Version | Status    | Notes |
|--------|--------------|-----------|-------|
| 1      | 6.4.3        | Legacy    | Maximum stability, legacy GPUs |
| 2      | 7.0.0        | Stable    | Production-ready for RDNA 3 |
| 3      | 7.0.2        | Latest    | **Recommended** (adds RDNA 4 support) |
| 4      | 7.9.0        | Preview   | Experimental via TheRock |

The installer now exports two additional environment variables:

- `ROCM_VERSION`
- `ROCM_CHANNEL`

These variables are written to `~/.mlstack_env` so that helper scripts (PyTorch, Triton, Flash Attention, etc.) can tailor their behaviour to the selected channel.

## 2. Installing ROCm 7.9.0 Preview (TheRock)

The preview build (option 4) relies on AMD's **TheRock** distribution. Because AMD may change the download URLs or bundle contents, the helper script intentionally stops with a descriptive message and links to the official documentation. To install the preview version:

1. Visit [https://repo.amd.com/rocm/tarball/](https://repo.amd.com/rocm/tarball/).
2. Download the tarball that matches your GPU architecture (gfx1100, gfx1101, gfx1200, etc.).
3. Extract the tarball into `/opt/rocm-preview-7.9.0` (or a path of your choice).
4. Update your environment by setting `ROCM_PATH` to the extracted directory and re-running `./scripts/enhanced_setup_environment.sh`.

> **Reminder**: Preview builds are not guaranteed to work with every helper script. Use them for experimentation only.

## 3. Environment Configuration

After installation, run:

```bash
./scripts/enhanced_setup_environment.sh
source ~/.mlstack_env
```

The generated environment now includes:

```bash
export ROCM_PATH="/opt/rocm"
export ROCM_VERSION="7.0.2"
export ROCM_CHANNEL="latest"
export GPU_ARCH="gfx1100"
```

These values are consumed by helper scripts such as:

- `scripts/install_pytorch.sh`
- `scripts/build_flash_attn_amd.sh`
- `scripts/install_triton.sh`
- `scripts/install_vllm.sh`

## 4. Helper Script Expectations

Each helper script should read `~/.mlstack_env` (e.g., `source ~/.mlstack_env`) and adjust its behaviour based on `ROCM_CHANNEL` and `GPU_ARCH`. For example:

- Option 1 (legacy) installs PyTorch wheels from `rocm6.4`.
- Option 3 (latest) uses the nightly ROCm 7.0 pip index.
- Option 4 (preview) includes instructions for pulling TheRock-specific nightlies.

## 5. Summary

- The original `install_rocm.sh` remains intact with all previous features.
- Additional ROCm versions are now selectable via the interactive menu.
- Environment exports (`ROCM_VERSION`, `ROCM_CHANNEL`, `GPU_ARCH`) make downstream scripting safer and more robust.
- Preview builds require manual steps and should be used for testing only.

For any questions or issues, consult the project README or open an issue with details about your selected ROCm channel.
