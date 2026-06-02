# Rusty-Stack UI Documentation

Rusty-Stack is the Rust-based TUI installer for Stan's ML Stack. It provides hardware detection, preflight checks, component selection, configuration, installation progress, and completion summaries.

## Entry Points

```bash
./scripts/run_rusty_stack.sh
```

## Configuration

- Stored at `~/.mlstack/config/config.json`
- Includes install path, batch mode, auto-confirm, theme, and performance profile

## Logs

- `~/.mlstack/logs/rusty-stack.log`

## Preflight Checks

Rusty-Stack performs critical checks for:
- Root privileges
- Disk space (50GB minimum, 100GB recommended)
- Network connectivity
- GPU detection
- ROCm driver presence
- CPU and memory compatibility
- Package manager availability
- Python version >= 3.9
- System dependency tools
- Distribution compatibility
