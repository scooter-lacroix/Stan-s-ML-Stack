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

### End of Life (Deprecated)

| Installer | Status | Technology | Description |
|-----------|--------|------------|-------------|
| **Python Curses** | 💀 **EOL** | Python + curses | `scripts/install_ml_stack_curses.py`. No longer recommended. |
| **Go Installer** | ❌ **DEPRECATED** | Go + Bubble Tea | `mlstack-installer/`. No longer maintained. |

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

If you're using the deprecated Go installer:

1. **Switch to Rusty-Stack** - The Go installer is no longer maintained
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

## 2026-02 Stabilization Status

This release cycle focused on ROCm reinstall reliability, multi-distro behavior, and benchmark/runtime correctness.

### Implemented

- Cross-distro ROCm force reinstall with:
  - hard purge,
  - reboot,
  - resume install,
  - mandatory second reboot.
- Arch/CachyOS ROCm install hardening:
  - AUR availability checks,
  - `pacman` for repo packages,
  - non-root `yay/paru`,
  - sudo ticket keepalive.
- Shared benchmark runtime preflight (`scripts/lib/benchmark_common.sh`) used by benchmark scripts.
- vLLM dependency reconciliation for commonly missing modules in runtime and benchmark paths.
- Megatron benchmark command integration (`scripts/run_megatron_benchmarks.sh`).
- DeepSpeed benchmark stabilization and improved output capture.
- Persistent environment hardening:
  - iGPU filtering for visible-device exports,
  - bash/zsh/fish support,
  - managed Triton cache environment.
- Benchmark HTML report export enhancements:
  - chart axes/labels,
  - data points and animations,
  - summary/detail table views,
  - export path notifications in TUI.

### Verification Snapshot

- Rust crate build check passes: `cargo check --manifest-path rusty-stack/Cargo.toml -q`
- Shell syntax checks pass for updated install/benchmark/env scripts via `bash -n`
- vLLM benchmark run recorded with successful JSON payload including throughput and discrete-only visible devices (`0,1`) in latest validated log set.

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
