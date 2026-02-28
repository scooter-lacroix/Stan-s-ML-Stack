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
./target/release/rusty-stack
```

## Features

- Hardware detection with AMD GPU and ROCm awareness
- Preflight checks for disk, memory, GPU presence, and ROCm availability
- Component selection across foundation, core, extensions, **UI/UX**, environment, and verification categories
- **Performance Category**: Dedicated tasks for vLLM, ROCm, and PyTorch hardware benchmarking
- Configuration screen with batch mode, auto-confirm, theme, and performance profile toggles
- Live installation progress with captured logs
- **Comparative Benchmarking**: Integrated dashboard for "Before vs After" performance analysis
- Completion summary with install/failed/skipped breakdown

## UI/UX Components

Rusty-Stack includes user-facing applications that install to your home directory and **don't require sudo**:

### ComfyUI (ROCm Edition)
A powerful node-based UI for AI image generation with full AMD GPU acceleration.

**Installation:** Select from the UI/UX category in the installer

**Run commands:**
```bash
comfy                    # Start ComfyUI with manager
comfy --listen 0.0.0.0   # Allow network access
comfy --port 8188        # Custom port
```

**Web Interface:** http://localhost:8188

**Features:**
- Full ROCm GPU acceleration
- Automatic torch dependency filtering (uses your existing ROCm PyTorch)
- Model preservation during updates
- ComfyUI Manager integration

### vLLM Studio
Web UI for vLLM model management and deployment.

**Installation:** Select from the UI/UX category in the installer

**Run command:**
```bash
vllm-studio              # Start the controller
```

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

### HTML Export

From the benchmark screen, press `E` to export a full HTML report.

- Export notifications now show success/failure and output path.
- Default output directory is `~/.mlstack/reports/`.
- Report contains animated charts, labeled axes, plotted points, and summary tables.

### Benchmark Log Parsing

Rusty-Stack now parses benchmark JSON from mixed log streams using a dedicated parser module (`src/benchmark_logs.rs`). This improves recovery from noisy script output and ensures summary/error views are populated from real run data.

### Multi-GPU and iGPU Filtering

On mixed iGPU+dGPU systems, benchmark runtime prep filters integrated GPUs and exports only discrete devices in `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`. This behavior is shared between persistent environment setup and benchmark runner preflight.

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

Benchmark logs and benchmark JSON files are also written under the MLStack log directory (typically `~/.mlstack/logs/`) with timestamped filenames.

## Troubleshooting

- Run with an interactive terminal (TTY) and a valid `$TERM`.
- Set `MLSTACK_NO_ALT_SCREEN=1` if your terminal struggles with alternate screen buffers.
- If screen artifacts appear, press `Ctrl+C` to exit safely.
- Ensure `rustc` + `cargo` are installed if building manually.
