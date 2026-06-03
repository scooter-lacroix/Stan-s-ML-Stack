# Migration Guide: Stan's ML Stack → Rusty Stack

This document explains the gradual rebranding from "Stan's ML Stack" to "Rusty Stack" and what you need to know as a user.

## Overview

Rusty Stack (formerly "Stan's ML Stack") is the same AMD GPU-focused machine learning environment you know and love. The project is undergoing a gradual rebranding to reflect its modern Rust-based TUI installer.

### What's Changing?

| Aspect | Before | After |
|--------|--------|-------|
| Project Name | Stan's ML Stack | Rusty Stack |
| Primary Installer | Python curses UI | `cargo install rusty-stack --locked` + Rusty Stack TUI/CLI |
| Repository | `scooter-lacroix/Stan-s-ML-Stack` | **Unchanged** (same repository) |
| Python Package | `stans-ml-stack` | `Rusty-Stack` compatibility wrapper |
| Rust Package | N/A | `rusty-stack` on crates.io |

### What's Staying the Same?

- **Repository URL**: `https://github.com/scooter-lacroix/Stan-s-ML-Stack`
- **Primary Install**: `cargo install rusty-stack --locked`
- **PyPI Package**: `pip install Rusty-Stack` remains as a compatibility wrapper
- **Backend Scripts**: archived and no longer part of the active install path
- **Components**: ROCm, PyTorch, vLLM, etc. - same components
- **Functionality**: Your ML stack works exactly the same

## Installation Migration

### If You Used the Python Curses Installer

The Python curses installer (`install_ml_stack_curses.py`) is now deprecated.

**Old way:**
```bash
./scripts/install_ml_stack_curses.py
```

**New way:**
```bash
cargo install rusty-stack --locked
rusty-stack
```

**Good news:** Existing installations remain compatible; new installs should use the Rust CLI/TUI.

### If You Used the PyPI Package

The PyPI package is maintained for backward compatibility.

```bash
pip install Rusty-Stack  # Renamed from stans-ml-stack
```

The package now installs the matching crates.io `rusty` binary and exposes compatibility entrypoints.

### If You Used the Go Installer

The Go installer (`mlstack-installer/`) is deprecated and no longer maintained.

**Migration:** Switch to Rusty-Stack TUI:
```bash
cargo install rusty-stack --locked
rusty-stack
```

## Component Installation

Your existing component installations remain functional. The shell scripts in `scripts/` haven't changed:

```bash
./scripts/install_rocm.sh          # Same as before
./scripts/install_pytorch.sh       # Same as before
./scripts/install_vllm.sh          # Same as before
```

## Environment Configuration

Environment variables and configuration remain unchanged:

```bash
# These still work exactly the same
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
```

## Documentation References

Documentation has been updated to reflect the new branding:

- Old: `Stan's ML Stack Documentation`
- New: `Rusty Stack Documentation`

The `Rusty-Stack` Python package (formerly `stans-ml-stack`) provides backward-compatible CLI tools.

## Timeline

- **Phase 1** (Current): Documentation and visible branding updates
- **Phase 2** (Future): Optional new PyPI package (`rusty-stack`)
- **Phase 3** (Future): Evaluate deprecation of legacy installers

## FAQ

### Do I need to reinstall anything?

**No.** Your existing Rusty Stack (Stan's ML Stack) installation continues to work. The rebranding is primarily about the project name and the recommended installer.

### Will the Rusty-Stack PyPI package work?

**Yes.** The package has been renamed from `stans-ml-stack` to `Rusty-Stack` on PyPI. Install with `pip install Rusty-Stack`.

### Should I switch to Rusty-Stack TUI?

**Yes.** Rusty-Stack TUI is the primary installer and receives active development. The Python curses and Go installers are deprecated.

### What about my existing scripts?

**They continue to work.** The backend shell scripts haven't changed, and the PyPI package is maintained for compatibility.

### Is the repository URL changing?

**No.** The repository remains at `https://github.com/scooter-lacroix/Stan-s-ML-Stack`.

## Getting Help

If you encounter any issues during the transition:

1. **Check the documentation**: See [docs/index.md](index.md) and [docs/INSTALLER_STATUS.md](INSTALLER_STATUS.md)
2. **Run verification**: `./scripts/enhanced_verify_installation.sh`
3. **Report issues**: https://github.com/scooter-lacroix/Stan-s-ML-Stack/issues

## Summary

| Question | Answer |
|----------|--------|
| Do I need to reinstall? | No |
| Is my installation broken? | No |
| Should I use Rusty-Stack TUI? | Yes, it's the recommended installer |
| Will Rusty-Stack PyPI package work? | Yes, renamed from stans-ml-stack |
| Is the repository URL changing? | No |

**Bottom line:** This is a branding change with minimal impact on existing installations. The Rusty-Stack TUI provides an improved installation experience while maintaining full compatibility with the existing codebase.
