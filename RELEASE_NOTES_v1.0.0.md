# Release Notes v1.0.0

## Rusty-Stack TUI Installer

This release delivers the Rusty-Stack terminal installer as the primary TUI for Stan's ML Stack.

### Highlights

- Rust-based installer using ratatui + crossterm
- Hardware detection with ROCm awareness
- Preflight checks with critical/warning classifications
- Component selection across foundation/core/extensions
- Configuration persistence in `~/.mlstack/config/config.json`
- Installation progress with live log capture

### Installation

```bash
./scripts/run_rusty_stack.sh
```
