# Stan's ML Stack: Rust Porting Guide

This guide provides a comprehensive overview of the new Rust-based architecture of Stan's ML Stack. The project has undergone a complete migration from legacy Python and Bash scripts to a native Rust implementation to improve reliability, performance, and maintainability.

## 1. Introduction

The objective of the Rust port was to:
- **Native Implementation**: Replace fragile shell delegation with pure Rust logic.
- **ROCm-Only Enforcement**: Hard-gate installations to prevent accidental CUDA PyTorch setups.
- **Unified Architecture**: Consolidate hardware detection, environment management, and component installation into a coherent crate workspace.
- **Modern UI**: Provide a responsive, animated TUI using `ratatui`.

## 2. Crate Structure

The project is organized into a set of specialized crates under the `crates/` directory:

| Crate | Purpose |
|-------|---------|
| `mlstack-hardware` | Multi-method hardware discovery (rocminfo, lspci, sysfs) with iGPU filtering. |
| `mlstack-env` | Persistent environment variable management with user/system strategies. |
| `mlstack-installers` | Native installers for ROCm, PyTorch, Triton, vLLM, etc., with dependency resolution. |
| `mlstack-tui` | The terminal user interface with smooth animations and real-time logging. |
| `mlstack-integration-tests` | Comprehensive integration tests ported from legacy Python suites. |

## 3. Key Features

### Native Hardware Discovery
Unlike the previous implementation which relied on `grep` and `awk` over shell output, `mlstack-hardware` uses a trait-based detector system. It prioritizes `rocminfo` and falls back to `lspci` and `sysfs`. It automatically filters out integrated GPUs (like Ryzen "Raphael" iGPUs) to ensure ROCm applications target the correct discrete hardware.

### Persistent Environment Management
The `mlstack-env` crate manages "sticky" environment variables. It can persist settings to `~/.mlstack_env` for shell sourcing or integrate with `systemd` to ensure variables like `HSA_OVERRIDE_GFX_VERSION` are available immediately upon boot.

### Unified Package Manager
The `UnifiedPackageManager` in `mlstack-installers` handles the complex dependency graph of the ML stack (e.g., ROCm -> PyTorch -> Triton -> Flash Attention). It performs topological sorting to ensure components are installed in the correct order and supports "force" reinstallations to repair broken environments.

### Automated Verification & Repair
The `VerificationModule` and `RepairModule` provide a one-click health check and fix system. They detect CUDA version mismatches, missing execution providers in ONNX Runtime, and incorrect visibility settings, offering automated repair actions.

## 4. Getting Started

To build and run the new Rust TUI:

```bash
# Build the entire workspace
cargo build --release

# Run the TUI
./target/release/mlstack-tui
```

## 5. API Overview

Developers can leverage the underlying libraries directly:

```rust
use mlstack_hardware::HardwareDiscovery;
use mlstack_installers::verification::VerificationModule;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Discover hardware
    let discovery = HardwareDiscovery::new();
    let gpus = discovery.detect_gpus()?;
    
    // Run health check
    let results = VerificationModule::run_all().await;
    
    Ok(())
}
```

## 6. Migration from Legacy Scripts

Most legacy `.sh` scripts in the `scripts/` directory are now deprecated. Below is a mapping of old scripts to their new Rust crate equivalents:

| Legacy Script | New Rust Equivalent |
|---------------|----------------------|
| `install_rocm.sh` | `mlstack-installers` (RocmInstaller) |
| `install_pytorch_rocm.sh` | `mlstack-installers` (PyTorchInstaller) |
| `enhanced_setup_environment.sh` | `mlstack-env` |
| `gpu_detection_utils.sh` | `mlstack-hardware` |
| `enhanced_verify_installation.sh` | `mlstack-installers` (VerificationModule) |
| `repair_ml_stack.sh` | `mlstack-installers` (RepairModule) |

---
*Documentation updated: January 31, 2026*
