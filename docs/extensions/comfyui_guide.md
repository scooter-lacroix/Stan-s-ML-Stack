# ComfyUI ROCm Guide

ComfyUI is a powerful node-based UI for Stable Diffusion and other AI image generation models. This guide covers installing and using ComfyUI with AMD GPU acceleration via ROCm.

## Overview

ComfyUI provides a node-based interface for:
- Stable Diffusion image generation
- Image-to-image transformations
- Inpainting and outpainting
- Custom workflows via node graphs
- Model management via ComfyUI Manager

## Installation via Rusty-Stack

The easiest way to install ComfyUI with ROCm support is through the Rusty-Stack TUI:

```bash
./scripts/run_rusty_stack.sh
```

Then navigate to **UI/UX** category and select **ComfyUI**.

### Manual Installation

```bash
./scripts/install_comfyui.sh
```

This script:
1. Clones ComfyUI from GitHub
2. Installs Python dependencies (excluding torch since ROCm version is already installed)
3. Creates a `comfy` launcher script
4. Sets up ROCm environment variables

## Running ComfyUI

### Basic Usage

```bash
# Start ComfyUI with the manager
comfy
```

This will:
- Launch ComfyUI on http://localhost:8188
- Enable the ComfyUI Manager for easy model installation
- Use ROCm GPU acceleration

### Advanced Options

```bash
# Allow network access (useful for remote access)
comfy --listen 0.0.0.0

# Use a custom port
comfy --port 8080

# Enable CUDA device selection (maps to ROCm GPUs)
comfy --cuda-device 0

# Disable the manager
comfy --disable-manager
```

### Environment Variables

ComfyUI automatically uses the following ROCm environment variables:

```bash
# GPU selection (set in your .mlstack_env)
HIP_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1
```

## Directory Structure

ComfyUI is installed to `$HOME/ComfyUI` by default:

```
ComfyUI/
├── models/           # Download your models here
│   ├── checkpoints/  # SD models
│   ├── lora/         # LoRA weights
│   ├── embeddings/   # Textual inversions
│   └── ...
├── input/            # Input images
├── output/           # Generated images
├── custom_nodes/     # Custom nodes
└── main.py           # Main entry point
```

## Model Installation

### Via ComfyUI Manager

1. Open http://localhost:8188
2. Click "Manager" button
3. Browse and install models directly

### Manual Installation

Download models and place them in the appropriate directories:

```bash
# Stable Diffusion models
~/ComfyUI/models/checkpoints/

# LoRA weights
~/ComfyUI/models/lora/

# Textual inversions
~/ComfyUI/models/embeddings/

# VAE models
~/ComfyUI/models/vae/
```

## Troubleshooting

### PyTorch Not Detected

If you see a warning about PyTorch not being detected:

```bash
# Install ROCm PyTorch first
source ~/.mlstack_env
python3 -c "import torch; print(torch.cuda.is_available())"
```

### GPU Not Recognized

Ensure your ROCm environment is properly configured:

```bash
# Check ROCm is working
rocm-smi

# Verify GPU visibility
echo $HIP_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Models Not Found

If ComfyUI can't find your models:

1. Check the `models/` directory structure
2. Verify model files are in the correct subdirectories
3. Restart ComfyUI after adding new models

## Performance Tips

1. **Use ROCm PyTorch**: Ensure you're using the ROCm-optimized PyTorch from this stack
2. **GPU Selection**: Use `HIP_VISIBLE_DEVICES` to select specific GPUs
3. **Batch Size**: Adjust based on your GPU memory (7900 XTX: 24GB, 7800 XT: 16GB)
4. **Image Resolution**: Higher resolutions require more VRAM

## Systemd Service (Optional)

Run ComfyUI as a background service:

```bash
# Enable the service
systemctl --user enable comfyui.service

# Start the service
systemctl --user start comfyui.service

# Check status
systemctl --user status comfyui.service
```

## Links

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Wiki](https://github.com/comfyanonymous/ComfyUI/wiki)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
