# Architectural Decisions and Technical Debt

## Overview

This document tracks architectural decisions and technical debt for the Rusty-Stack installer.

## Key Decisions

1. **Language**: Rust for memory safety and performance.
2. **UI**: ratatui + crossterm for terminal rendering.
3. **Execution**: Installer scripts executed via `bash` with `sudo -S` when needed.
4. **Configuration**: JSON config stored under `~/.mlstack/config/config.json`.
5. **Logging**: Persistent log file in `~/.mlstack/logs/rusty-stack.log`.

## Technical Debt

- Extend hardware detection with deeper ROCm driver parsing.
- Expand preflight diagnostics for GPU topology and storage performance.
- Add structured benchmark/test summary parsing.
