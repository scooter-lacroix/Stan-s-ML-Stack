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
- Configuration screen with batch mode, auto-confirm, theme, and performance profile toggles
- Live installation progress with captured logs
- Completion summary with install/failed/skipped breakdown

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
