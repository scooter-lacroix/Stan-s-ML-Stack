# Migration Guide: Stan's ML Stack → Rusty Stack

This document explains the gradual rebranding from "Stan's ML Stack" to "Rusty Stack" and what you need to know as a user.

## Overview

Rusty Stack (formerly "Stan's ML Stack") is the same AMD GPU-focused machine learning environment you know and love. The project is undergoing a gradual rebranding to reflect its modern Rust-based TUI installer.

### What's Changing?

| Aspect | Before | After |
|--------|--------|-------|
| Project Name | Stan's ML Stack | Rusty Stack |
| Primary Installer | Python curses UI | Rusty-Stack TUI (Rust + Ratatui) |
| Repository | `scooter-lacroix/Stan-s-ML-Stack` | **Unchanged** (same repository) |
| Python Package | `stans-ml-stack` | **Unchanged** (maintained for compatibility) |

### What's Staying the Same?

- **Repository URL**: `https://github.com/scooter-lacroix/Stan-s-ML-Stack`
- **PyPI Package**: `pip install stans-ml-stack` still works
- **Backend Scripts**: All `scripts/install_*.sh` scripts remain unchanged
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
# One-line install
curl -fsSL https://raw.githubusercontent.com/scooter-lacroix/Stan-s-ML-Stack/main/scripts/install.sh | bash

# Or manual build
./scripts/run_rusty_stack.sh
```

**Good news:** Both installers use the same backend shell scripts, so your existing installation is compatible.

### If You Used the PyPI Package

The PyPI package is maintained for backward compatibility.

```bash
pip install stans-ml-stack  # Still works!
```

The package now points to Rusty-Stack TUI as the recommended installer.

### If You Used the Go Installer

The Go installer (`mlstack-installer/`) has been completely removed from the repository.

**Migration:** Switch to Rusty-Stack TUI:
```bash
./scripts/run_rusty_stack.sh
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

The `stans-ml-stack` Python package is still referenced for backward compatibility.

## Timeline

- **Phase 1** (Current): Documentation and visible branding updates
- **Phase 2** (Future): Optional new PyPI package (`rusty-stack`)
- **Phase 3** (Future): Evaluate deprecation of legacy installers

## FAQ

### Do I need to reinstall anything?

**No.** Your existing Rusty Stack (Stan's ML Stack) installation continues to work. The rebranding is primarily about the project name and the recommended installer.

### Will the stans-ml-stack PyPI package continue to work?

**Yes.** The `stans-ml-stack` package will continue to be published and maintained for backward compatibility.

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
| Will stans-ml-stack package still work? | Yes, maintained for compatibility |
| Is the repository URL changing? | No |

**Bottom line:** This is a branding change with minimal impact on existing installations. The Rusty-Stack TUI provides an improved installation experience while maintaining full compatibility with the existing codebase.
