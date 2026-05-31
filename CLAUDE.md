# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Rusty Stack** (formerly Stan's ML Stack) is a comprehensive machine learning environment optimized for AMD GPUs with ROCm support. It provides installation tools, core ML components, and utilities for training and deploying ML models on AMD hardware.

## Key Commands

### Installation

```bash
# Primary installer — unified rusty CLI (TUI mode)
cd rusty-stack && cargo build --release
./target/release/rusty              # Launch interactive TUI installer
./target/release/rusty-stack        # Same as above (backward compat alias)

# Quick install (one-line)
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash

# PyPI installation (backward compat)
pip install Rusty-Stack
```

### Component Updates

```bash
rusty update                  # Scan and update components
rusty update --scan-only      # Scan without applying changes
rusty update --all-safe       # Apply all safe updates
rusty update --json           # Machine-readable output
```

### Verification

```bash
rusty verify --full           # Full component verification
rusty verify --enhanced       # Enhanced verification (all components)
rusty verify --build          # Verify and rebuild failed components
```

### Benchmarks

```bash
rusty bench --all             # Run full benchmark suite
rusty bench --rocm            # ROCm benchmarks
rusty bench --json <name>     # JSON output for a specific benchmark
```

### Environment Setup

```bash
# Environment setup is handled by the rusty CLI bootstrap module
# Or source the env file directly:
source ~/.mlstack_env
```

### Testing

```bash
# Rust test suite
cd rusty-stack && cargo test

# Run with no TUI features
cargo test --no-default-features
```

### Docker

```bash
docker build -t stans-ml-stack .
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it stans-ml-stack
```

## Architecture

### Directory Structure

```
├── rusty-stack/           # Rust TUI installer + platform tools (primary)
│   └── src/
│       ├── main.rs        # TUI entry point (3-line wrapper)
│       ├── lib.rs         # Library crate with run_tui() and all modules
│       ├── app.rs         # Application state machine (tui feature-gated)
│       ├── installer.rs   # Native Rust installer dispatch + legacy script fallback
│       ├── hardware.rs    # GPU/ROCm detection (delegates to platform/)
│       ├── core/          # Shared types, manifest schema, validation state machine
│       ├── platform/      # Hardware/distro detection, component registry, environment
│       ├── orchestrator/  # Update planner, apply engine, verify runner, upgrade
│       ├── adapter/       # Adapter registry with Rust and legacy script executors
│       ├── telemetry/     # Stability benchmark, anonymous payload, HTTPS submission
│       ├── installers/    # Native Rust installer modules
│       │   ├── common/    # Shared infra (package_manager, distro, guard, utils, etc.)
│       │   └── components/# Per-component installers (rocm, pytorch, triton, etc.)
│       ├── verification/  # Native Rust verification (verify --full, --enhanced, --build)
│       ├── benchmark_runners/ # Native Rust benchmark runners
│       ├── bootstrap/     # install.sh, run_rusty_stack.sh, enhanced_setup_environment.sh equivalents
│       ├── bin/
│       │   └── rusty.rs   # Unified CLI (update, upgrade, bench, verify subcommands)
│       └── widgets/       # UI components (tui feature-gated)
├── scripts/               # Active shell scripts (verification + benchmarks only)
│   ├── verify_installation.sh
│   ├── enhanced_verify_installation.sh
│   ├── verify_and_build.sh
│   ├── run_*_benchmarks.sh
│   ├── lib/               # Shared shell libraries (sourced by active scripts)
│   ├── install_ml_stack_curses.py  # DEPRECATED — use rusty CLI
│   └── install_ml_stack_ui.py      # DEPRECATED — use rusty CLI
├── archive/               # Archived (non-active) scripts
│   ├── scripts/           # ~60 archived shell scripts + lib files
│   └── python/            # Archived Python scripts
├── stans_ml_stack/        # Python package (backward compat)
│   ├── cli/               # CLI tools (install.py DEPRECATED, verify, repair)
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

1. **Unified `rusty` CLI (Primary):** Single binary with clap subcommands. `rusty` (no args) launches TUI. Subcommands: `update` (scan/plan/apply/verify), `upgrade` (binary/runtime upgrade), `bench` (benchmark runner with --json), `verify` (installation verification with --full/--enhanced/--build).
2. **Rusty-Stack TUI:** Interactive terminal UI using ratatui + crossterm. Accessible via `rusty` (no args) or `rusty-stack` binary. All 24 installer components use native Rust — no bash subprocess for installation.
3. **Native Rust Installers:** 24 installer components ported from shell scripts to Rust modules under `src/installers/components/`. Shared infrastructure in `src/installers/common/` (package manager, distro detection, guard, utils, benchmark common).
4. **Verification Module:** `src/verification/` provides full, enhanced, and build verification using `component_status.rs` detection — no shell subprocess.
5. **Benchmark Runners:** `src/benchmark_runners/` dispatches to existing `benchmarks/` module functions. Supports `--json` output.
6. **Bootstrap Module:** `src/bootstrap/` provides install.sh, run_rusty_stack.sh, and enhanced_setup_environment.sh equivalents in native Rust.
7. **Python UIs (Deprecated):** Textual and curses-based installers marked deprecated. Use `rusty` CLI instead.
8. **Shell Scripts (Legacy):** Active scripts in `scripts/` are limited to verification and benchmark runners (still referenced by `state.rs`). All installer scripts archived to `archive/scripts/`.

### Configuration

- Config: `~/.mlstack/config/config.json`
- Logs: `~/.mlstack/logs/rusty-stack.log`
- Environment: `~/.mlstack_env`

## ROCm Channels

Three channels available (select via TUI or `INSTALL_ROCM_PRESEEDED_CHOICE` env var):
1. **Legacy (ROCm 6.4.3)** - Production-proven stability
2. **Stable (ROCm 7.2.3)** - Production-ready for RDNA 3
3. **Latest (ROCm 7.2.4)** - Default, expanded RDNA 4 support

See `docs/MULTI_CHANNEL_GUIDE.md` for channel-specific component versions.

## Development Notes

- **Package Manager:** Uses `uv` for Python packages (falls back to pip)
- **Python Version:** Supports 3.10-3.13
- **Target Hardware:** AMD RDNA 2/3/4 GPUs (7900 XTX, 7800 XT, 9070 XT, etc.)
- **Shell Scripts:** All scripts in `scripts/` should be executable and handle errors gracefully
- **Rust Build:** Use `cargo build --release` in `rusty-stack/`
- **Migration Guide:** See [MIGRATION.md](MIGRATION.md) for the complete migration guide from the legacy shell/Python architecture to Rusty Stack

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
<!-- NEXUS:START -->
## Nexus Memory Substrate
- Identity: [Soul](/home/scooter/.config/nexus/soul.md)
- Project Context: [Project Context](/home/scooter/Documents/Product/Stan-s-ML-Stack/.nexus/context.md)
<!-- NEXUS:END -->
