# Rusty-Stack Installer Guide

Rusty-Stack is the primary TUI installer for Stan's ML Stack.

## Quick Start

```bash
./scripts/run_rusty_stack.sh
```

## Manual Build

```bash
cd rusty-stack
cargo build --release
./target/release/Rusty-Stack
```

## Notes

- Rusty-Stack stores configuration in `~/.mlstack/config/config.json`.
- Logs are stored in `~/.mlstack/logs/rusty-stack.log`.
- Use `MLSTACK_NO_ALT_SCREEN=1` if your terminal cannot render the alternate screen buffer.
