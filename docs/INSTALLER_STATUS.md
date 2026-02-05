# Installer Status

This document describes the current status of all installers for Rusty Stack (formerly Stan's ML Stack).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    User-Facing Installers                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   Rusty-Stack TUI   │    │  PyPI Package       │             │
│  │   (PRIMARY)         │    │  (Compatibility)    │             │
│  │   Rust + Ratatui    │    │  stans-ml-stack     │             │
│  └──────────┬──────────┘    └──────────┬──────────┘             │
│             │                          │                         │
└─────────────┼──────────────────────────┼─────────────────────────┘
              │                          │
              │                          │
              ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend Layer                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│         Shell Scripts (scripts/install_*.sh)                     │
│         - install_rocm.sh                                        │
│         - install_pytorch*.sh                                    │
│         - install_vllm*.sh                                       │
│         - install_flash_attention*.sh                            │
│         - ... (and many more)                                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Current Installer Status

### Primary Installers

| Installer | Status | Technology | Description |
|-----------|--------|------------|-------------|
| **Rusty-Stack TUI** | ✅ **ACTIVE** | Rust + Ratatui | The primary system management interface. Handles installation, hardware detection, and persistent performance benchmarking. |
| **PyPI Package** | ✅ **ACTIVE** | Python | `stans-ml-stack` package on PyPI. Maintained for backward compatibility and CLI automation. |

### End of Life (Deprecated/Removed)

| Installer | Status | Technology | Description |
|-----------|--------|------------|-------------|
| **Python Curses** | 💀 **EOL** | Python + curses | `scripts/install_ml_stack_curses.py`. No longer recommended. |
| **Go Installer** | ❌ **REMOVED** | Go + Bubble Tea | Completely removed from repository. Use Rusty-Stack TUI instead. |

## Installation Methods

### Recommended: Rusty-Stack TUI

```bash
# One-line install
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash

# Or manual build
./scripts/run_rusty_stack.sh
```

**Features:**
- Modern TUI with keyboard navigation
- Hardware detection (GPU, ROCm, disk space)
- Component selection across 5 categories
- Configuration options (batch mode, auto-confirm, themes)
- Live installation progress with log capture
- Completion summary with install/failed/skipped breakdown

### For Compatibility: PyPI Package

```bash
pip install stans-ml-stack
ml-stack-install  # Launches curses UI
```

**Use this if:**
- You prefer installing via pip
- You're integrating into a Python-based workflow
- You need the legacy CLI commands

## Migration Guide

### From Python Curses to Rusty-Stack

If you're currently using the Python curses installer (`install_ml_stack_curses.py`):

1. **No migration needed** - Rusty-Stack uses the same backend shell scripts
2. **Similar experience** - The TUI provides equivalent functionality with improved UX
3. **Same configuration** - Environment variables and settings are compatible

### From Go Installer

The Go installer has been completely removed from the repository.

1. **Switch to Rusty-Stack** - The Go installer is no longer available
2. **Your installation is still valid** - Components installed via the Go installer remain functional
3. **Use Rusty-Stack for updates** - Run Rusty-Stack to update or add components

## Backend Architecture

All installers (Rusty-Stack, curses, PyPI) ultimately execute the same backend shell scripts:

```
rusty-stack/        →  TUI frontend
scripts/*.sh        →  Installation scripts
stans_ml_stack/     →  Python package (CLI wrappers)
```

The shell scripts handle:
- ROCm installation (multi-channel support)
- PyTorch with ROCm support
- vLLM and vLLM Studio
- Flash Attention (CK and Triton variants)
- DeepSpeed, Megatron-LM
- Triton, BITSANDBYTES
- Environment setup and verification

## Troubleshooting

### Rusty-Stack Won't Build

**Problem:** `cargo build` fails

**Solutions:**
1. Install Rust via rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Check for system dependencies: `sudo apt install build-essential pkg-config`
3. See [docs/rusty_stack_guide.md](rusty_stack_guide.md) for details

### Legacy Installer Issues

**Problem:** Python curses or Go installer fails

**Solution:** Use Rusty-Stack TUI instead. The legacy installers are deprecated and may have compatibility issues with newer Python versions.

### Component Installation Fails

**Problem:** A specific component fails to install

**Solution:** All installers use the same backend scripts. Check the logs:
- Rusty-Stack: `~/.mlstack/logs/rusty-stack.log`
- Curses: `logs/ml_stack_install_*.log`

## Future Plans

- [ ] Additional TUI themes
- [ ] Web UI option
- [ ] Container-based installation
- [ ] Automated testing across all ROCm channels

## Related Documentation

- [Rusty-Stack Guide](rusty_stack_guide.md) - Detailed Rusty-Stack usage
- [Multi-Channel ROCm Guide](MULTI_CHANNEL_GUIDE.md) - ROCm channel selection
- [Migration Guide](MIGRATION.md) - Project rebranding details
