# Migration Guide: Stan's ML Stack → Rusty Stack

> **Audience:** Existing users and contributors of Stan's ML Stack who are migrating to Rusty Stack.
>
> This document covers everything you need to know about the migration from the shell/Python-based Stan's ML Stack to the Rust-native Rusty Stack. It includes architecture changes, feature mappings, breaking changes, CLI command equivalents, and rollback instructions.

---

## Table of Contents

1. [Project Overview and Migration Rationale](#1-project-overview-and-migration-rationale)
2. [Architecture Changes](#2-architecture-changes)
3. [Feature Mapping Table](#3-feature-mapping-table)
4. [Breaking Changes and Deprecations](#4-breaking-changes-and-deprecations)
5. [Installation Differences](#5-installation-differences)
6. [Environment Setup Changes](#6-environment-setup-changes)
7. [CLI Command Mapping](#7-cli-command-mapping)
8. [Development Workflow Changes](#8-development-workflow-changes)
9. [Testing Approach](#9-testing-approach)
10. [Rollback Instructions](#10-rollback-instructions)

---

## 1. Project Overview and Migration Rationale

### Why Rust?

The original Stan's ML Stack was built on ~47 shell scripts and Python-based UIs. While functional, this architecture had several limitations:

- **Error handling**: Shell scripts lack structured error types, making it difficult to distinguish between "package not found" vs. "network timeout" vs. "permission denied."
- **Testability**: Shell scripts are hard to unit test. Validating command construction required running actual commands or fragile string matching.
- **Performance**: Spawning a bash subprocess for every installation step adds overhead — process creation, script parsing, environment setup.
- **Maintainability**: ~35,000 lines of shell code with duplicated patterns (package manager calls, distro detection, ROCm env setup) across scripts.
- **Cross-platform**: Shell scripts are inherently Unix-only. Windows users needed WSL2 as a compatibility layer.

Rust provides:

- **Type-safe error handling** via `anyhow::Result<T>` — every failure mode is a structured enum variant
- **First-class testing** via `cargo test` — unit tests, integration tests, and CLI tests with `assert_cmd`
- **Zero subprocess overhead** for ported components — native Rust functions replace bash spawns
- **Feature gates** — TUI code is optional; the CLI works without any terminal UI dependencies
- **Single binary distribution** — one `rusty` executable replaces four separate binaries

### Why a Unified CLI?

Previously, the project shipped four separate binaries:

| Old Binary | Purpose |
|---|---|
| `rusty-stack` | TUI installer |
| `rusty-stack-update` | Component updates |
| `rusty-stack-upgrade` | Binary self-upgrade |
| `rusty-stack-bench` | Benchmark runner |

These shared no CLI interface consistency. The unified `rusty` CLI uses `clap` with subcommands:

```
rusty              # Launch TUI installer
rusty update       # Component updates
rusty upgrade      # Binary self-upgrade
rusty bench        # Benchmark runner
rusty verify       # Installation verification
```

One binary, consistent flags, `--json` output for all subcommands, and `--help` everywhere.

---

## 2. Architecture Changes

### Before: Shell Scripts + Python UIs

```
┌─────────────────────────────────────────────────────────┐
│  User Interface Layer                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ curses UI    │  │ Textual UI   │  │ rusty-stack   │  │
│  │ (Python)     │  │ (Python)     │  │ TUI (Rust)    │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                   │          │
├─────────┼─────────────────┼───────────────────┼──────────┤
│  Script Layer             │                   │          │
│  ┌────────────────────────┴───────────────────┘          │
│  │  47 shell scripts (install_*.sh, verify_*.sh, etc.)   │
│  │  ┌─────────────────────────────────────────────────┐  │
│  │  │  lib/ (shared shell libraries)                   │  │
│  │  │  • installer_guard.sh    • package_manager.sh    │  │
│  │  │  • distro_detection.sh   • rocm_env.sh          │  │
│  │  │  • benchmark_common.sh   • ui_installer_helper   │  │
│  │  └─────────────────────────────────────────────────┘  │
│  └───────────────────────────────────────────────────────┘
│         │                                                │
├─────────┼────────────────────────────────────────────────┤
│  System Layer                                           │
│  ┌──────┴───────────────────────────────────────────┐    │
│  │  bash → apt/dnf/pacman, pip, git, cmake, make    │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Key issues:** Every installer spawned a bash subprocess, parsed output with grep/awk, and had its own copy of error handling and logging patterns.

### After: Native Rust Modules

```
┌─────────────────────────────────────────────────────────┐
│  User Interface Layer                                    │
│  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │ rusty-stack  │  │ rusty (unified CLI)              │  │
│  │ TUI (Rust)   │  │ ┌────────┐ ┌────────┐ ┌──────┐  │  │
│  │              │  │ │ update │ │ upgrade│ │ bench│  │  │
│  └──────┬───────┘  │ │ verify │ │  (TUI) │ │      │  │  │
│         │          │ └────────┘ └────────┘ └──────┘  │  │
│         │          └──────────────────────────────────┘  │
├─────────┼────────────────────────────────────────────────┤
│  Rust Installer Layer                                    │
│  ┌──────┴───────────────────────────────────────────┐    │
│  │  src/installers/                                  │    │
│  │  ┌──────────────────┐  ┌───────────────────────┐  │    │
│  │  │ common/          │  │ components/ (24)       │  │    │
│  │  │ • package_manager│  │ • rocm, pytorch,      │  │    │
│  │  │ • distro         │  │   triton, mpi4py,     │  │    │
│  │  │ • guard          │  │   deepspeed, vllm,    │  │    │
│  │  │ • rocm_env       │  │   megatron, onnx,     │  │    │
│  │  │ • utils          │  │   comfyui, wandb, ...  │  │    │
│  │  │ • env_validation │  │                        │  │    │
│  │  │ • benchmark_comm │  └───────────────────────┘  │    │
│  │  └──────────────────┘                              │    │
│  ├───────────────────────────────────────────────────┤    │
│  │  src/verification/  src/benchmark_runners/         │    │
│  │  src/bootstrap/     src/platform/                  │    │
│  └───────────────────────────────────────────────────┘    │
│         │                                                │
├─────────┼────────────────────────────────────────────────┤
│  System Layer                                           │
│  ┌──────┴───────────────────────────────────────────┐    │
│  │  std::process::Command → apt/dnf/pacman, pip,    │    │
│  │  git, cmake, make (only when needed)              │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Key improvements:**

- **No bash subprocess for ported components** — installer dispatch calls Rust functions directly
- **Shared infrastructure** — package manager, distro detection, ROCm env are unified modules, not duplicated per script
- **Feature-gated TUI** — `cargo check --no-default-features` compiles the CLI without any TUI dependencies
- **Dependency-aware ordering** — components declare dependencies; installer resolves topological order

### Module Organization

| Rust Module | Purpose | Replaces |
|---|---|---|
| `rusty-stack/src/installers/common/package_manager.rs` | Package manager abstraction (apt/dnf/pacman) | `lib/package_manager.sh` |
| `rusty-stack/src/installers/common/distro.rs` | Distro detection (delegates to `platform::detection`) | `lib/distro_detection.sh` |
| `rusty-stack/src/installers/common/guard.rs` | Structured error handling and logging | `lib/installer_guard.sh` |
| `rusty-stack/src/installers/common/rocm_env.rs` | ROCm environment configuration | `lib/rocm_env.sh` |
| `rusty-stack/src/installers/common/utils.rs` | Shared utilities (color output, command checks) | `common_utils.sh` |
| `rusty-stack/src/installers/common/env_validation.rs` | Environment file validation | `env_validation_utils.sh` |
| `rusty-stack/src/installers/common/benchmark_common.rs` | Benchmark result parsing and logging | `lib/benchmark_common.sh` |
| `rusty-stack/src/installers/common/ui_helper.rs` | CLI argument parsing and validation | `lib/ui_installer_helper.sh` |
| `rusty-stack/src/installers/common/package_mappings.rs` | Package name mapping per distro | `lib/package_mappings.sh` |
| `rusty-stack/src/installers/components/*.rs` | 24 native component installers | `scripts/install_*.sh` |
| `rusty-stack/src/verification/mod.rs` | Installation verification | `scripts/verify_*.sh` |
| `rusty-stack/src/benchmark_runners/mod.rs` | Benchmark execution | `scripts/run_*_benchmarks.sh` |
| `rusty-stack/src/bootstrap/` | Install, launch, environment setup | `scripts/install.sh`, `run_rusty_stack.sh`, `enhanced_setup_environment.sh` |

---

## 3. Feature Mapping Table

### Installer Components (24 ported to native Rust)

| Component ID | Old Shell Script | New Rust Module | Category |
|---|---|---|---|
| `rocm` | `install_rocm.sh` | `installers/components/rocm.rs` | Foundation |
| `pytorch` | `install_pytorch_rocm.sh` | `installers/components/pytorch.rs` | Foundation |
| `triton` | `install_triton.sh` | `installers/components/triton.rs` | Foundation |
| `mpi4py` | `install_mpi4py.sh` | `installers/components/mpi4py.rs` | Foundation |
| `deepspeed` | `install_deepspeed.sh` | `installers/components/deepspeed.rs` | Foundation |
| `ml-stack-core` | `install_ml_stack.sh` | `installers/components/ml_stack.rs` | Core |
| `flash-attn` | `install_flash_attention_ck.sh` | `installers/components/flash_attention_ck.rs` | Core |
| `repair-stack` | `repair_ml_stack.sh` | `installers/components/repair.rs` | Core |
| `permanent-env` | `setup_permanent_rocm_env.sh` | `installers/components/permanent_env.rs` | Environment |
| `megatron` | `install_megatron.sh` | `installers/components/megatron.rs` | Extension |
| `vllm` | `install_vllm_multi.sh` | `installers/components/vllm_multi.rs` | Extension |
| `aiter` | `install_aiter.sh` | `installers/components/aiter.rs` | Extension |
| `onnx` | `build_onnxruntime_multi.sh` | `installers/components/onnxruntime.rs` | Extension |
| `bitsandbytes` | `install_bitsandbytes_multi.sh` | `installers/components/bitsandbytes_multi.rs` | Extension |
| `rocm-smi` | `install_rocm_smi.sh` | `installers/components/rocm_smi.rs` | Extension |
| `migraphx` | `install_migraphx_multi.sh` | `installers/components/migraphx_multi.rs` | Extension |
| `pytorch-profiler` | `install_pytorch_profiler.sh` | `installers/components/pytorch_profiler.rs` | Extension |
| `wandb` | `install_wandb.sh` | `installers/components/wandb.rs` | Extension |
| `vllm-studio` | `install_vllm_studio.sh` | `installers/components/vllm_studio.rs` | UI/UX |
| `comfyui` | `install_comfyui.sh` | `installers/components/comfyui.rs` | UI/UX |
| `textgen` | `install_textgen.sh` | `installers/components/textgen.rs` | UI/UX |
| `amdgpu-drivers` | `install_amdgpu_drivers.sh` | `installers/components/amdgpu_drivers.rs` | Foundation |
| `migraphx-python` | `install_migraphx_python.sh` | `installers/components/migraphx_python.rs` | Extension |

### Verification Components (native Rust via CLI)

| Component ID | Old Shell Script | New Rust Entry Point |
|---|---|---|
| `verify-basic` | `verify_installation.sh` | `rusty verify --full` |
| `verify-enhanced` | `enhanced_verify_installation.sh` | `rusty verify --enhanced` |
| `verify-build` | `verify_and_build.sh` | `rusty verify --build` |

### Benchmark Components (native Rust via CLI)

| Component ID | Old Shell Script | New Rust Entry Point |
|---|---|---|
| `gpu-capability` | (new) | `rusty bench gpu-capability` |
| `memory-bandwidth` | `test_gpu_memory_bandwidth.sh` | `rusty bench memory-bandwidth` |
| `tensor-core` | (new) | `rusty bench tensor-core` |
| `gemm` | (new) | `rusty bench gemm` |
| `pytorch` | `run_pytorch_benchmarks.sh` | `rusty bench pytorch` |
| `flash-attention` | (new) | `rusty bench flash-attention` |
| `vllm` | `run_vllm_benchmarks.sh` | `rusty bench vllm` |
| `deepspeed` | `run_deepspeed_benchmarks.sh` | `rusty bench deepspeed` |
| `megatron` | `run_megatron_benchmarks.sh` | `rusty bench megatron` |
| `rocm` | `run_rocm_benchmarks.sh` | `rusty bench --rocm` |
| `rocm-smi` | `run_rocm_smi_benchmarks.sh` | `rusty bench rocm-smi` |
| `mlperf-inference` | `run_mlperf_inference.sh` | `rusty bench mlperf-inference` |
| `all` | `run_all_benchmarks_suite.sh` | `rusty bench all` |
| `all-pre` | (new) | `rusty bench all-pre` |

### Shared Infrastructure Mapping

| Old Shell Library | New Rust Module | Lines (Shell → Rust) |
|---|---|---|
| `lib/installer_guard.sh` (952 lines) | `installers/common/guard.rs` | 952 → ~700 |
| `lib/distro_detection.sh` (747 lines) | `installers/common/distro.rs` + `platform/detection.rs` | 747 → ~300 (delegated) |
| `lib/package_manager.sh` (1,243 lines) | `installers/common/package_manager.rs` | 1,243 → ~900 |
| `lib/rocm_env.sh` (508 lines) | `installers/common/rocm_env.rs` + `platform/environment.rs` | 508 → ~400 |
| `lib/benchmark_common.sh` (2,168 lines) | `installers/common/benchmark_common.rs` | 2,168 → ~1,000 |
| `lib/ui_installer_helper.sh` (228 lines) | `installers/common/ui_helper.rs` | 228 → ~500 |
| `common_utils.sh` (94 lines) | `installers/common/utils.rs` | 94 → ~350 |
| `env_validation_utils.sh` (93 lines) | `installers/common/env_validation.rs` | 93 → ~450 |
| `package_manager_utils.sh` (318 lines) | `installers/common/package_mappings.rs` | 318 → ~600 |

### Archived Scripts

The following scripts have been archived to `archive/scripts/` and are no longer active:

| Archived Script | Reason |
|---|---|
| `install_ml_stack_part{1,2,3,4}.sh` | Absorbed into unified `ml_stack.rs` |
| `install_bitsandbytes.sh` | Replaced by `bitsandbytes_multi.rs` |
| `install_pytorch_multi.sh` | Absorbed into `pytorch.rs` |
| `install_triton_multi.sh` | Absorbed into `triton.rs` |
| `install_rccl.sh`, `install_rccl_multi.sh` | Absorbed into `ml_stack.rs` |
| `install_ml_stack_extensions.sh` | Absorbed into individual component installers |
| `install_components_simple.sh` | Replaced by native Rust dispatch |
| `install_vllm.sh` | Replaced by `vllm_multi.rs` |
| `create_persistent_env.sh` | Replaced by `permanent_env.rs` |
| `setup_environment.sh` | Replaced by `bootstrap/env_setup.rs` |
| `update_stack.sh`, `update_helper.sh` | Replaced by `rusty update` |
| `cleanup_for_git.sh`, `prepare_for_git.sh` | Utility scripts, archived |
| `check_components.sh`, `ml_stack_component_detector.sh` | Replaced by `component_status.rs` |
| `build_flash_attn_amd.sh` | Absorbed into `flash_attention_ck.rs` |
| `run_ml_stack_ui.sh`, `run_rusty_stack.py` | Replaced by `rusty` CLI |
| `run_tests.sh` | Replaced by `cargo test` |
| `run_vllm.sh` | Utility script, archived |
| `comprehensive_rebuild.sh` | Replaced by `rusty verify --build` |
| `scorched_earth_cleanup.sh`, `system_python_cleanup.sh` | Utility scripts, archived |
| `fix_python_symlinks.sh` | Utility script, archived |
| `gpu_detection_utils.sh` | Replaced by `hardware.rs` + `platform/` |
| `analyze_nvidia_dependency_risk.sh` | Utility script, archived |
| `build_onnxruntime_outline.sh` | Absorbed into `onnxruntime.rs` |
| `portability_patch`, `portability_patch.rs` | Legacy patches, archived |

---

## 4. Breaking Changes and Deprecations

### Binary Consolidation

| Before | After | Migration Path |
|---|---|---|
| `rusty-stack` (TUI) | `rusty-stack` (unchanged) | No change needed |
| `rusty-stack-update` | `rusty update` | Update scripts/aliases |
| `rusty-stack-upgrade` | `rusty upgrade` | Update scripts/aliases |
| `rusty-stack-bench` | `rusty bench` | Update scripts/aliases |
| *(none)* | `rusty verify` | New subcommand |

**The old separate binaries no longer exist.** If you have scripts or aliases referencing `rusty-stack-update`, `rusty-stack-upgrade`, or `rusty-stack-bench`, update them to use the unified `rusty` CLI.

### Deprecated Python UIs

The following Python-based installers are deprecated and will not receive further updates:

| File | Status | Replacement |
|---|---|---|
| `scripts/install_ml_stack_curses.py` | **Deprecated** | `rusty` (TUI) or `rusty-stack` |
| `scripts/install_ml_stack_ui.py` | **Deprecated** | `rusty` (TUI) or `rusty-stack` |
| `stans_ml_stack/cli/install.py` | **Deprecated** | `rusty` CLI |

These files remain in the repository with deprecation headers but should not be used for new installations.

### Deprecated Shell Scripts

All installer shell scripts have been archived to `archive/scripts/`. They are no longer invoked by the Rusty Stack system. If you were directly running any `install_*.sh` script, use the corresponding `rusty` subcommand instead.

### Configuration File Location

| Item | Before | After |
|---|---|---|
| Config directory | `~/.mlstack/config/` | `~/.mlstack/config/` (unchanged) |
| Config file | `config.json` | `config.json` (unchanged) |
| Log file | `~/.mlstack/logs/rusty-stack.log` | `~/.mlstack/logs/rusty-stack.log` (unchanged) |
| Environment file | `~/.mlstack_env` | `~/.mlstack_env` (now managed by bootstrap module) |

### Python Package

| Before | After |
|---|---|
| `pip install stans-ml-stack` | Still works for backward compatibility |
| `stans_ml_stack` Python package | Archived to `archive/legacy-ml-stack/stans_ml_stack/` |

The Python package is archived but remains installable for backward compatibility. New installations should use the Rust CLI.

---

## 5. Installation Differences

### Before: Shell Script Installation

```bash
# Clone the repository
git clone https://github.com/scooter-lacroix/Stan-s-ML-Stack.git
cd Stan-s-ML-Stack

# Run the installer script
chmod +x scripts/install.sh
./scripts/install.sh

# Or use the curses UI
python3 scripts/install_ml_stack_curses.py

# Or use the Textual UI
python3 scripts/install_ml_stack_ui.py

# Verify
./scripts/verify_installation.sh
```

### After: Rusty Stack Installation

```bash
# Install from crates.io
cargo install rusty-stack --locked

# Run the TUI installer
rusty-stack

# Or use individual subcommands
rusty-stack update --scan-only     # Check for updates
rusty-stack update --all-safe      # Apply safe updates
rusty-stack verify --full          # Verify installation
rusty-stack bench --list           # List benchmarks
rusty-stack bench --all            # Run all benchmarks
```

### Source Build

Build from source only when developing Rusty Stack itself:

```bash
git clone https://github.com/scooter-lacroix/Stan-s-ML-Stack.git
cd Stan-s-ML-Stack/rusty-stack
cargo build --release
./target/release/rusty
```

### Build Requirements

| Requirement | Before | After |
|---|---|---|
| Python | 3.10-3.13 | 3.10-3.13 (unchanged) |
| Rust | Not required | Current stable Rust, via rustup, for `cargo install rusty-stack` or source builds |
| Shell | bash | bash (for remaining active scripts) |
| Package manager | apt/dnf/pacman | apt/dnf/pacman (unchanged) |

---

## 6. Environment Setup Changes

### Before: Shell-Managed Environment

```bash
# Environment was set up by shell scripts
source ~/.mlstack_env

# Setup was done by:
./scripts/enhanced_setup_environment.sh
# or
./scripts/setup_environment.sh
# or
./scripts/create_persistent_env.sh
```

The `~/.mlstack_env` file was generated by shell scripts with inline environment variable exports.

### After: Bootstrap-Managed Environment

```bash
# Environment is still sourced the same way
source ~/.mlstack_env

# But now managed by the Rust bootstrap module:
# rusty-stack/src/bootstrap/env_setup.rs — equivalent to enhanced_setup_environment.sh
# rusty-stack/src/bootstrap/install.rs   — equivalent to install.sh
# rusty-stack/src/bootstrap/launcher.rs  — equivalent to run_rusty_stack.sh
```

The `~/.mlstack_env` file format is unchanged — it remains a shell-sourceable file with environment variable exports. The difference is that it's now generated by Rust code instead of shell scripts.

### Key Environment Variables

| Variable | Purpose | Changed? |
|---|---|---|
| `MLSTACK_REPO_ROOT` | Root of the ML Stack repository | No |
| `MLSTACK_SCRIPTS_DIR` | Location of active scripts | No |
| `HIP_VISIBLE_DEVICES` | GPU filtering | No |
| `HSA_OVERRIDE_GFX_VERSION` | GPU architecture override | No |
| `ROCM_HOME` | ROCm installation path | No |
| `MLSTACK_NO_ALT_SCREEN` | Disable TUI alternate screen | No (new variable) |
| `INSTALL_ROCM_PRESEEDED_CHOICE` | ROCm channel selection | No |
| `NO_COLOR` | Disable colored output | No |

---

## 7. CLI Command Mapping

### Update Commands

| Before (Shell) | After (Rust CLI) | Notes |
|---|---|---|
| `./scripts/update_stack.sh` | `rusty update` | Full interactive update |
| *(no equivalent)* | `rusty update --scan-only` | Preview without applying |
| *(no equivalent)* | `rusty update --all-safe` | Apply only safe updates |
| *(no equivalent)* | `rusty update --include-experimental` | Include experimental components |
| *(no equivalent)* | `rusty update --json` | Machine-readable JSON output |
| *(no equivalent)* | `rusty update pytorch triton` | Update specific components |

### Upgrade Commands

| Before | After | Notes |
|---|---|---|
| `rusty-stack-upgrade` | `rusty upgrade` | Upgrade the Rusty Stack binary |
| *(no equivalent)* | `rusty upgrade --dry-run` | Check without applying |
| *(no equivalent)* | `rusty upgrade --yes` | Non-interactive mode |
| *(no equivalent)* | `rusty upgrade --binary-path /path` | Custom binary path |

### Verification Commands

| Before (Shell) | After (Rust CLI) | Notes |
|---|---|---|
| `./scripts/verify_installation.sh` | `rusty verify --full` | Core component verification |
| `./scripts/enhanced_verify_installation.sh` | `rusty verify --enhanced` | All-component verification |
| `./scripts/verify_and_build.sh` | `rusty verify --build` | Verify and identify rebuild targets |
| *(no equivalent)* | `rusty verify --json` | JSON output |

### Benchmark Commands

| Before (Shell) | After (Rust CLI) | Notes |
|---|---|---|
| `./scripts/run_all_benchmarks_suite.sh` | `rusty bench all` | Full benchmark suite |
| `./scripts/run_rocm_benchmarks.sh` | `rusty bench --rocm` | ROCm benchmarks |
| `./scripts/run_pytorch_benchmarks.sh` | `rusty bench pytorch` | PyTorch benchmarks |
| `./scripts/run_vllm_benchmarks.sh` | `rusty bench vllm` | vLLM benchmarks |
| `./scripts/run_deepspeed_benchmarks.sh` | `rusty bench deepspeed` | DeepSpeed benchmarks |
| `./scripts/run_megatron_benchmarks.sh` | `rusty bench megatron` | Megatron benchmarks |
| `./scripts/run_rocm_smi_benchmarks.sh` | `rusty bench rocm-smi` | ROCm SMI benchmarks |
| `./scripts/test_gpu_memory_bandwidth.sh` | `rusty bench memory-bandwidth` | Memory bandwidth test |
| `./scripts/run_mlperf_inference.sh` | `rusty bench mlperf-inference` | MLPerf inference |
| *(no equivalent)* | `rusty bench --list` | List available benchmarks |
| *(no equivalent)* | `rusty bench --json <name>` | JSON output |

### TUI Commands

| Before | After | Notes |
|---|---|---|
| `rusty-stack` | `rusty-stack` (unchanged) | TUI installer |
| `rusty` (no args) | `rusty` (no args) | Also launches TUI |

---

## 8. Development Workflow Changes

### Before: Shell Script Contributions

```bash
# Create a new installer script
cat > scripts/install_new_component.sh << 'EOF'
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/lib/installer_guard.sh"
source "$(dirname "$0")/lib/package_manager.sh"
source "$(dirname "$0")/lib/distro_detection.sh"

# ... installation logic ...
install_packages cmake rocm-libs
pip install some-package
EOF
chmod +x scripts/install_new_component.sh

# Test by running the script
./scripts/install_new_component.sh

# Register in state.rs (Rust) by adding script path
```

### After: Rust Module Contributions

```bash
# Create a new installer module
cat > rusty-stack/src/installers/components/new_component.rs << 'RUST'
//! Native Rust installer for new_component.

use crate::installers::common::package_manager::PackageManager;
use crate::installers::common::guard::{InstallerResult, InstallerError};
use crate::installers::common::distro::DistroFamily;

/// Configuration for the new_component installer.
pub struct NewComponentConfig {
    pub install_prefix: String,
    pub python_version: String,
}

/// Installer for new_component.
pub struct NewComponentInstaller;

impl NewComponentInstaller {
    /// Install new_component.
    pub fn install(config: &NewComponentConfig) -> InstallerResult<()> {
        // Use shared infrastructure
        let pm = PackageManager::detect();
        pm.install(&["cmake", "build-essential"])?;

        // Construct pip command
        let output = std::process::Command::new("pip")
            .args(["install", "some-package"])
            .output()
            .map_err(|e| InstallerError::CommandFailed {
                command: "pip install some-package".into(),
                reason: e.to_string(),
            })?;

        if !output.status.success() {
            return Err(InstallerError::CommandFailed {
                command: "pip install some-package".into(),
                reason: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_command_construction() {
        // Verify the installer constructs correct commands
        let config = NewComponentConfig {
            install_prefix: "/opt/ml-stack".into(),
            python_version: "3.12".into(),
        };
        // Test command construction without executing
        assert!(true); // Real tests verify command args
    }
}
RUST
```

### Development Workflow Summary

| Task | Before | After |
|---|---|---|
| Add installer | Create `.sh` file in `scripts/` | Create `.rs` file in `src/installers/components/` |
| Add shared utility | Add to `scripts/lib/` | Add to `src/installers/common/` |
| Run tests | `./scripts/run_tests.sh` | `cargo test` |
| Check formatting | N/A | `cargo fmt --check` |
| Lint | `shellcheck` | `cargo clippy -- -D warnings` |
| Build | N/A | `cargo build --release` |
| Run without TUI | N/A | `cargo check --no-default-features` |
| Integration tests | Shell scripts in `tests/` | Rust tests in `tests/*.rs` with `assert_cmd` |

### Code Organization Conventions

| Convention | Rule |
|---|---|
| Error handling | Use `anyhow::Result<T>` for fallible operations |
| Structured errors | Use `InstallerError` enum from `common::guard` |
| Package operations | Use `PackageManager` from `common::package_manager` |
| Distro detection | Use `platform::detection` (never re-implement) |
| ROCm detection | Use `platform::linux` (never re-implement) |
| Feature gates | TUI-only code behind `#[cfg(feature = "tui")]` |
| Testing | Every module has `#[cfg(test)] mod tests` |
| Documentation | All public types have doc comments |

---

## 9. Testing Approach

### Before: Shell Script Testing

```bash
# Tests were shell scripts
./scripts/run_tests.sh

# Verification was a shell script
./scripts/verify_installation.sh
./scripts/enhanced_verify_installation.sh

# No unit testing framework
# Tests relied on exit codes and output parsing
```

### After: Rust Testing

```bash
# Run all tests
cargo test

# Run specific test module
cargo test --lib installer
cargo test --test rusty_cli

# Run without TUI features
cargo test --no-default-features

# Run with verbose output
cargo test -- --nocapture

# Run specific test by name
cargo test test_rocm_installer

# Skip known-failing platform tests
cargo test -- --skip test_package_manager_rhel_family --skip test_package_manager_suse_family
```

### Test Categories

| Category | Location | Count | Framework |
|---|---|---|---|
| Unit tests | `src/**/*.rs` (`#[cfg(test)]` blocks) | ~100+ | Built-in `#[test]` |
| Integration tests | `tests/*.rs` | ~12 files | `assert_cmd` + `predicates` |
| CLI tests | `tests/rusty_cli.rs` | 5+ | `assert_cmd` |
| Installer dispatch | `tests/installer_dispatch.rs` | 24+ | `assert_cmd` |
| Update flow | `tests/update_cli.rs`, `tests/update_apply.rs` | 10+ | `assert_cmd` |
| Verify/bench CLI | `tests/verify_bench_cli.rs` | 5+ | `assert_cmd` |
| Adapter parity | `tests/adapter_parity.rs` | 10+ | Built-in |
| Migration wave 1 | `tests/migration_wave1.rs` | 15+ | Built-in |
| Windows/WSL | `tests/windows_wsl.rs` | 5+ | Built-in |

### Before/After Test Comparison

```bash
# BEFORE: Shell script testing
./scripts/run_tests.sh
# Output: Mixed pass/fail with unclear error messages
# No structured test results

# AFTER: Rust testing
cargo test 2>&1
# Output:
# running 49 tests
# test installers::common::guard::tests::test_error_display ... ok
# test installers::common::package_manager::tests::test_apt_install ... ok
# test installers::components::rocm::tests::test_channel_selection ... ok
# ...
# test result: ok. 49 passed; 0 failed; 0 ignored; 0 measured
```

### Writing New Tests

```rust
// Unit test for command construction
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_installer_correct_wheel_url() {
        let config = PytorchConfig {
            rocm_version: "6.4.3".into(),
            python_version: "3.12".into(),
        };
        let url = PytorchInstaller::build_index_url(&config);
        assert!(url.contains("rocm6.2"));
    }

    #[test]
    fn test_rocm_channel_version_pins() {
        let legacy = RocmChannel::Legacy;
        assert_eq!(legacy.version(), "6.4.3");

        let stable = RocmChannel::Stable;
        assert_eq!(stable.version(), "7.2.3");
    }
}
```

```rust
// Integration test for CLI
use assert_cmd::Command;

#[test]
fn test_rusty_update_scan_only() {
    let mut cmd = Command::cargo_bin("rusty").unwrap();
    cmd.arg("update").arg("--scan-only").arg("--json");
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["summary"]["scan_only"], true);
}
```

---

## 10. Rollback Instructions

If you need to revert to the legacy shell-based installation:

### Option 1: Use the Archive

All legacy scripts are preserved in the `archive/` directory:

```bash
# Archived installer scripts
ls archive/scripts/install_*.sh

# Archived shared libraries
ls archive/scripts/lib/

# Archived Python UIs
ls archive/scripts/install_ml_stack_curses.py
ls archive/scripts/install_ml_stack_ui.py

# Archived Python package
ls archive/legacy-ml-stack/stans_ml_stack/
```

To use archived scripts:

```bash
# Run an archived installer directly
bash archive/scripts/install_rocm.sh

# Run the archived verification
bash archive/scripts/verify_installation.sh
```

### Option 2: Use an Older Git Commit

```bash
# View commit history
git log --oneline -20

# Find the last commit before the migration (pre-Milestone 1)
# Commits with "feat(m1)" or later are part of the migration
# Commits before "refactor(m1)" are pre-migration

# Checkout a pre-migration commit
git checkout <pre-migration-commit>

# Use the old shell scripts
cd scripts
./install.sh
```

### Option 3: Keep Both Systems

Since Rusty Stack and the shell scripts can coexist:

```bash
# Build Rusty Stack
cd rusty-stack && cargo build --release

# Use Rusty Stack for new installations
./target/release/rusty

# Fall back to archived scripts for specific components
bash archive/scripts/install_vllm.sh
```

### What Cannot Be Rolled Back

- **Configuration files** (`~/.mlstack/`) are format-compatible between old and new systems
- **Environment file** (`~/.mlstack_env`) is format-compatible
- **Installed components** (ROCm, PyTorch, etc.) are identical regardless of installer
- **The old separate binaries** (`rusty-stack-update`, `rusty-stack-upgrade`, `rusty-stack-bench`) are removed from the build — you would need to build from a pre-migration commit

### Rolling Forward

If you encounter issues with the new system:

1. **Report the issue** at [GitHub Issues](https://github.com/scooter-lacroix/Stan-s-ML-Stack/issues)
2. **Use `--json` output** for structured error reporting: `rusty update --json 2>&1 | tee report.json`
3. **Run verification** to identify what's broken: `rusty verify --enhanced`
4. **Use `--scan-only`** to preview changes without applying: `rusty update --scan-only`

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│  OLD COMMAND                              NEW COMMAND         │
├──────────────────────────────────────────────────────────────┤
│  rusty-stack-update                       rusty update        │
│  rusty-stack-upgrade                      rusty upgrade       │
│  rusty-stack-bench                        rusty bench         │
│  ./scripts/verify_installation.sh         rusty verify --full │
│  ./scripts/enhanced_verify_installation   rusty verify --enhanced │
│  ./scripts/verify_and_build.sh            rusty verify --build│
│  ./scripts/run_all_benchmarks_suite.sh    rusty bench all     │
│  ./scripts/install_ml_stack_curses.py     rusty (TUI)         │
│  ./scripts/install_ml_stack_ui.py         rusty (TUI)         │
│  ./scripts/install.sh                     rusty                │
│  cargo test (4 binaries)                  cargo test (2 bins)  │
└──────────────────────────────────────────────────────────────┘
```
