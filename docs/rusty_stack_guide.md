# Rusty-Stack Installer Guide

> **PRIMARY INSTALLER** - Rusty-Stack is the recommended installer for Rusty Stack (formerly Stan's ML Stack).

Rusty-Stack is the Rust-based TUI installer for Rusty Stack.

## Quick Start

```bash
# From repository root
./scripts/run_rusty_stack.sh

# Or via console entry point (pip install -e .)
ml-stack-install

# Or manual build
cd rusty-stack
cargo build --release
./target/release/Rusty-Stack
```

## Features

- Hardware detection with AMD GPU and ROCm awareness
- Preflight checks for disk, memory, GPU presence, and ROCm availability
- Component selection across foundation, core, extensions, environment, and verification categories
- **Performance Category**: Dedicated tasks for vLLM, ROCm, and PyTorch hardware benchmarking
- Configuration screen with batch mode, auto-confirm, theme, and performance profile toggles
- Live installation progress with captured logs
- **Comparative Benchmarking**: Integrated dashboard for "Before vs After" performance analysis
- Completion summary with install/failed/skipped breakdown

## Performance & Benchmarking Dashboard

Rusty-Stack includes a comprehensive performance validation suite that can be accessed by pressing `B` after a successful installation or by running tasks in the **Performance** category.

### Comparative Analysis (Before vs. After)
The TUI automatically manages a performance baseline for your system:
- **Baseline**: The first successful benchmark run recorded in `~/.rusty-stack/logs`.
- **Latest**: Your most recent run.
- **Deltas**: The TUI displays a percentage change (e.g., `+12.5%` speedup in Green or `-5.0%` regression in Red) to help validate the impact of ROCm updates, kernel optimizations, or driver changes.

### Captured Metrics
- **GPU Summary**: Aggregate peak TFLOPS (FP16), GB/s, and hardware telemetry (Clock/Temp) across all detected cards.
- **Memory Bandwidth**: Real HBM throughput and System-to-Device (PCIe) bandwidth tests.
- **PyTorch Performance**: Standardized GEMM and Convolution GFLOPS measurements.
- **vLLM Throughput**: Real-world tokens/second performance using standard models (e.g., `opt-125m`).
- **Flash Attention**: Quantitative speedup and memory savings analysis compared to standard attention.

## Architecture

Rusty-Stack uses a **frontend + backend** architecture:

```
┌────────────────────────────────────────┐
│     Rusty-Stack TUI (Frontend)        │
│     Rust + Ratatui                     │
└──────────────┬─────────────────────────┘
               │
               │ Executes shell scripts
               ▼
┌────────────────────────────────────────┐
│     Shell Scripts (Backend)            │
│     scripts/install_*.sh               │
└────────────────────────────────────────┘
```

The TUI provides a user-friendly interface, while the actual installation is performed by the same battle-tested shell scripts used by all other installers.

## Sudo Handling

Rusty-Stack prompts for your sudo password inside the TUI (only when not running as root) and pipes it to `sudo -S` for script execution.

## Logs

Runtime logs are written to:

```
~/.mlstack/logs/rusty-stack.log
```

## Troubleshooting

- Run with an interactive terminal (TTY) and a valid `$TERM`.
- Set `MLSTACK_NO_ALT_SCREEN=1` if your terminal struggles with alternate screen buffers.
- If screen artifacts appear, press `Ctrl+C` to exit safely.
- Ensure `rustc` + `cargo` are installed if building manually.
