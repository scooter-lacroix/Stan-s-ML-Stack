# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Rusty Stack** (formerly Stan's ML Stack) is a comprehensive machine learning environment optimized for AMD GPUs with ROCm support. It provides installation tools, core ML components, and utilities for training and deploying ML models on AMD hardware.

## Key Commands

### Installation

```bash
# Primary installer (Rust TUI)
./scripts/run_rusty_stack.sh

# Or build and run directly
cd rusty-stack && cargo build --release && ./target/release/rusty-stack

# Quick install (one-line)
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash

# PyPI installation
pip install stans-ml-stack
```

### Environment Setup

```bash
./scripts/enhanced_setup_environment.sh
source ~/.mlstack_env
```

### Verification

```bash
./scripts/enhanced_verify_installation.sh
./scripts/custom_verify_installation.sh
```

### Testing

```bash
# Run all tests
./tests/run_all_tests.sh

# Run specific test categories
./tests/run_integration_tests.sh
./tests/run_performance_tests.sh

# Run validation tests
cd tests/validation && ./run_all_tests.sh
```

### Benchmarks

```bash
./scripts/run_benchmarks.sh
./scripts/run_all_benchmarks_suite.sh
```

### Docker

```bash
docker build -t stans-ml-stack .
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it stans-ml-stack
```

## Architecture

### Directory Structure

```
├── rusty-stack/           # Rust TUI installer (primary)
│   └── src/
│       ├── main.rs        # Entry point (ratatui + crossterm)
│       ├── app.rs         # Application state machine
│       ├── installer.rs   # Script execution engine
│       ├── hardware.rs    # GPU/ROCm detection
│       └── widgets/       # UI components
├── scripts/               # Shell installation scripts
│   ├── install_*.sh       # Component installers
│   ├── install_ml_stack_ui.py    # Textual UI (deprecated)
│   └── install_ml_stack_curses.py # Curses UI (deprecated)
├── stans_ml_stack/        # Python package
│   ├── cli/               # CLI tools (install, verify, repair)
│   ├── installers/        # Python installers
│   ├── core/              # Core ML utilities
│   └── utils/             # Benchmarks and utilities
├── core/                  # Core Python modules (mirrored)
├── extensions/            # Extension modules (mirrored)
├── benchmarks/            # Performance benchmarks
├── tests/                 # Test suites
│   ├── integration/       # Integration tests
│   ├── verification/      # Installation verification
│   ├── validation/        # Script validation
│   └── performance/       # Performance tests
└── docs/                  # Documentation
```

### Key Components

**Core:** ROCm, PyTorch, ONNX Runtime, MIGraphX, Flash Attention, RCCL, MPI, Megatron-LM

**Extensions:** Triton, bitsandbytes, vLLM, ComfyUI, DeepSpeed, Weights & Biases

### Code Mirroring

The `core/`, `extensions/`, and `benchmarks/` directories are mirrored under `stans_ml_stack/core/`, `stans_ml_stack/core/extensions/`, and `stans_ml_stack/utils/benchmarks/` for package distribution.

### Installer Architecture

1. **Rusty-Stack (Primary):** Rust TUI using ratatui + crossterm. Executes shell scripts via `bash` with `sudo -S` when needed.
2. **Python UIs (Deprecated):** Textual and curses-based installers maintained for backward compatibility.
3. **Shell Scripts:** Individual `install_*.sh` scripts in `scripts/` for each component.

### Configuration

- Config: `~/.mlstack/config/config.json`
- Logs: `~/.mlstack/logs/rusty-stack.log`
- Environment: `~/.mlstack_env`

## ROCm Channels

Three channels available (select via TUI or `INSTALL_ROCM_PRESEEDED_CHOICE` env var):
1. **Legacy (ROCm 6.4.3)** - Production-proven stability
2. **Stable (ROCm 7.1)** - Production-ready for RDNA 3
3. **Latest (ROCm 7.2)** - Default, expanded RDNA 4 support

See `docs/MULTI_CHANNEL_GUIDE.md` for channel-specific component versions.

## Development Notes

- **Package Manager:** Uses `uv` for Python packages (falls back to pip)
- **Python Version:** Supports 3.10-3.13
- **Target Hardware:** AMD RDNA 2/3/4 GPUs (7900 XTX, 7800 XT, 9070 XT, etc.)
- **Shell Scripts:** All scripts in `scripts/` should be executable and handle errors gracefully
- **Rust Build:** Use `cargo build --release` in `rusty-stack/`

## LeIndex Usage

LeIndex is **MANDATORY** for code exploration in this codebase:

```bash
# Set project path
leindex -p ~/Documents/Product/Stan-s-ML-Stack search "query"

# Force reindex
leindex index ~/Documents/Product/Stan-s-ML-Stack --force

# Analyze files
leindex -p ~/Documents/Product/Stan-s-ML-Stack analyze <file>
```
