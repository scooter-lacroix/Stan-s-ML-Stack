# Code Context: Force Reinstall & Benchmark UI Logic

## Files Retrieved

1. `src/config.rs` (lines 1-150) — Config loading, `force_reinstall` field definition, persistence
2. `src/app.rs` (lines 2120-2140) — Env var propagation on install start; (lines 1880-1915) — TUI config display; (lines 2020-2026) — toggle force_reinstall; (lines 41-115) — `TaskStatus` enum for install progress UI; (lines 2240-2280) — status mapping logic
3. `src/installer.rs` (lines 51-195) — `run_installation()` top-level dispatch; (lines 2445-2570) — `run_native_installer()` ROCm dispatch with force_reinstall check; (lines 3650-3860) — `execute_script_installation()` env var injection for force_reinstall; (lines 3853-3868) — `--force` flag append for script-based components; (lines 2292-2392) — `git_clone_or_pull()` idempotent clone logic; (lines 2759-2800) — deepspeed install dispatch; (lines 2657-2760) — triton install; (lines 2854-2944) — flash-attn install; (lines 2944-3010) — megatron install; (lines 3008-3060) — vllm install; (lines 3300-3370) — onnxruntime install with uninstall step
4. `src/installers/components/rocm.rs` (lines 140-165) — `RocmConfig` struct with `force_reinstall: bool`
5. `src/installers/components/onnxruntime.rs` (lines 360-395) — `build_uninstall_command()` pip uninstall
6. `src/installers/components/bitsandbytes_multi.rs` (lines 240-265) — `build_uninstall_command()` pip uninstall
7. `src/installers/components/deepspeed.rs` (lines 170-240) — `build_install_command()` with `--force-reinstall`; `build_force_reinstall_command()`
8. `src/installers/components/vllm_multi.rs` (lines 340-380) — `build_force_reinstall_command()` with `--force-reinstall`
9. `src/installers/components/mpi4py.rs` (lines 310-330) — `force_reinstall` adds `--force-reinstall` pip flag
10. `src/installers/components/mod.rs` (lines 160-230) — `is_native_component()`, dependency declarations
11. `src/orchestrator/planner.rs` (lines 380-440, 580-640) — planner treats same-version as Safe "reinstall"
12. `src/core/types.rs` (lines 80-180) — `Stage` enum including `Installing`, `Benchmarks`
13. `src/widgets/benchmarks_page.rs` (lines 1-240) — benchmark results loading, stale detection, status fields
14. `src/benchmarks/mod.rs` (lines 1-80) — benchmark runner infrastructure
15. `src/benchmark_runners/mod.rs` (lines 1-60) — benchmark name registry and dispatch
16. `src/bin/rusty.rs` (lines 735-765) — CLI force_reinstall env var propagation

---

## 1. Force Reinstall Flow

### Config Read & Persistence

**`src/config.rs:16`** — `force_reinstall: bool` is a field on `InstallerConfig`.

**Load path** (`src/config.rs:98-109`): Reads `~/.mlstack/config/config.json` → parses JSON → `user_preferences.force_reinstall` → applies to struct.

**Save path** (`src/config.rs:70`): Writes back to `user_preferences.force_reinstall` in JSON.

**Default** (`src/config.rs:131`): `false`.

### Propagation to Installer

There are **two independent propagation paths**:

#### TUI Path (`src/app.rs:2128-2136`)
```rust
fn start_installation(&mut self) {
    if self.config.force_reinstall {
        std::env::set_var("FORCE", "true");
        std::env::set_var("PYTORCH_REINSTALL", "true");
        std::env::set_var("MLSTACK_FORCE_REINSTALL", "true");
    } else {
        std::env::remove_var("FORCE");
        std::env::remove_var("PYTORCH_REINSTALL");
        std::env::remove_var("MLSTACK_FORCE_REINSTALL");
    }
}
```
Three env vars are set: `FORCE`, `PYTORCH_REINSTALL`, `MLSTACK_FORCE_REINSTALL`.

#### CLI Path (`src/bin/rusty.rs:742-743`)
```rust
if config.force_reinstall {
    std::env::set_var("MLSTACK_FORCE_REINSTALL", "1");
}
```
Only sets `MLSTACK_FORCE_REINSTALL` (not `FORCE` or `PYTORCH_REINSTALL`). **Potential inconsistency** — TUI sets 3 vars, CLI sets only 1.

### TUI Toggle (`src/app.rs:2025`)
```rust
self.config.force_reinstall = !self.config.force_reinstall;
```
Shown in Configuration stage as "Force Reinstall All: on/off" with help text: "FORCE REINSTALL ALL COMPONENTS. Forces purging and re-downloading of everything." (line 1911).

### Orchestrator/Planner

**`src/orchestrator/planner.rs:391`**: Same-version installs are classified as `UpdateClassification::Safe` with reason `"reinstall v{} (validated)"`. This is for the update planner, not the initial install flow. The planner does **not** read `force_reinstall` from config — it classifies same-version reinstalls as safe updates.

### Script-Based Installer Path (`src/installer.rs:3686-3695, 3834-3835, 3853-3868`)

When a component uses the script-based installer (non-native), `force_reinstall` is propagated via env vars:

```rust
let force_reinstall = std::env::var("MLSTACK_FORCE_REINSTALL")
    .or_else(|_| std::env::var("FORCE"))
    .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
    .unwrap_or(false);

// Sets on the spawned bash command:
.env("FORCE", force_env_value)
.env("PYTORCH_REINSTALL", force_env_value)
.env("MLSTACK_FORCE_REINSTALL", force_env_value)
```

Additionally, if `force_reinstall` is true, the script path is scanned for `--force` argument support and appends `--force` to the command if found (lines 3853-3868).

---

## 2. ROCm Driver Reinstall

### No Uninstall Step
ROCm has **no uninstall/remove step** before reinstalling. The flow in `src/installer.rs:2506-2570` for Arch/pacman:

1. Lists all ROCm packages via `inst.pacman_rocm_packages()`
2. Filters already-installed via `filter_already_installed_pacman()`
3. If all installed and `force_reinstall` is false → skips entirely (`return Ok(())`)
4. If all installed and `force_reinstall` is true → logs "force-reinstalling" and proceeds with `all_pkgs`
5. Runs `yay -S --needed --noconfirm <packages>` (note: `--needed` flag actually **prevents** reinstalling already-installed packages!)

**Critical issue**: Even with `force_reinstall=true`, the pacman command includes `--needed`, which tells pacman/yay to skip packages that are already installed. This means `force_reinstall` for ROCm on Arch is **effectively a no-op** — the flag says "reinstall all" but `--needed` says "skip installed ones."

For apt/dnf/zypper distros, ROCm uses standard package install commands without any uninstall step either.

### No Reboot Mechanism
There is **no reboot mechanism** anywhere in the codebase. Searching for "reboot", "restart", "systemd" in the Rust source yields no hits related to installation reboots. The ROCm installer proceeds directly from package install to completion without reboot markers or systemd service management.

---

## 3. Component Uninstall Before Reinstall

### Summary Table

| Component | Uninstall Before Reinstall? | Mechanism |
|-----------|----------------------------|-----------|
| **ROCm** | ❌ No | No uninstall; `yay -S --needed` (contradicts force_reinstall) |
| **PyTorch** | ❌ No | No uninstall step in native installer; `PYTORCH_REINSTALL` env var set but script handles it |
| **ONNX Runtime** | ✅ Yes | `build_uninstall_command()` runs `pip uninstall -y onnxruntime onnxruntime-rocm onnxruntime-gpu` **always** (line 3356), not conditional on force_reinstall |
| **bitsandbytes** | ❌ No | Has `build_uninstall_command()` but it's **never called** in the install dispatch |
| **DeepSpeed** | Partial | `DeepSpeedConfig.force_reinstall` adds `--force-reinstall` pip flag; also has `build_force_reinstall_command()` for retry |
| **MPI4Py** | ✅ Conditional | `force_reinstall` adds `--force-reinstall` to pip install args |
| **vLLM** | ✅ Conditional | `build_force_reinstall_command()` adds `--force-reinstall` (used in retry, not main install) |
| **Triton** | ❌ No | No uninstall; `git_clone_or_pull` is idempotent |
| **Flash Attention** | ❌ No | No uninstall; `git_clone_or_pull` + cmake rebuild |
| **Megatron** | ❌ No | No uninstall; removes `.egg-info` dirs only |
| **AITER** | ❌ No | No uninstall; `git_clone_or_pull` is idempotent |

### Git-Based Components

All git-based components (triton, flash-attn, megatron, aiter, onnx, bitsandbytes, rocm-smi) use `git_clone_or_pull()` (`src/installer.rs:2295`):

1. If `.git` exists in target dir → `git pull --ff-only`, fallback to `git fetch --all` + `git reset --hard origin/HEAD`
2. If directory exists without `.git` → backs up to `.bak`, then clones fresh
3. If no directory → clones fresh

**No explicit `rm -rf` before clone.** The `git reset --hard` effectively replaces working tree content but preserves `.git/config` and local branches.

### Pip-Based Components

For most pip components, the native installer dispatch does **not** call any uninstall before install. The exceptions:
- **ONNX Runtime**: Always runs `pip uninstall -y onnxruntime onnxruntime-rocm onnxruntime-gpu` before installing (line 3356). This is unconditional, not tied to `force_reinstall`.
- **DeepSpeed**: If `DeepSpeedConfig.force_reinstall` is true, adds `--force-reinstall` flag to pip install (line 183). However, the native installer creates `DeepSpeedConfig::default()` which has `force_reinstall: false` — **it never reads the env var**.
- **MPI4Py**: Similarly, `force_reinstall` in config adds `--force-reinstall` pip flag (line 324-325), but the native installer uses `Mpi4PyConfig::default()` with `force_reinstall: false`.

### Package-Based Components (ROCm)

Arch: Uses `yay -S --needed --noconfirm` — the `--needed` flag contradicts force_reinstall intent.
apt/dnf/zypper: Standard install commands, no uninstall-before-reinstall.

---

## 4. Benchmark UI Status

### Benchmark Results Loading

**`src/widgets/benchmarks_page.rs`**: The `load_benchmark_results()` function (line 110) loads benchmark data from JSON log files in `~/.mlstack/logs/` and related directories. It:
1. Searches for log files matching patterns: `rocm_benchmarks`, `gpu_memory_bandwidth`, `pytorch_performance`, `vllm_benchmarks`, `deepspeed_benchmarks`, `megatron_benchmarks`, `full_benchmarks`
2. Parses JSON from log contents
3. Checks staleness: `STALE_FULL_BENCHMARK_WINDOW = 15 minutes`, `STALE_ERROR_WINDOW = 15 minutes`

### Status Display

The benchmark page uses **status strings** in its JSON/HTML export (lines 2011-2067):
- `"ok"` — GPU detected, benchmark has valid results
- `"missing"` — no GPU detected
- `"degraded"` — benchmark has zero or invalid throughput values (vllm, deepspeed, megatron)

**There is no "Running" or "Installing" status for benchmarks in the TUI benchmark page.** These status strings are only used in the HTML/JSON export output. The TUI benchmark page shows static results from log files.

### Benchmark Execution

Benchmarks are executed via `src/benchmarks/mod.rs` and dispatched through `src/benchmark_runners/mod.rs`. The runner names are: `gpu-capability`, `memory-bandwidth`, `tensor-core`, `gemm`, `pytorch`, `flash-attention`, `vllm`, `deepspeed`, `megatron`, `all-pre`, `all`.

Benchmarks are **not** run during the install flow. They are separate components (`mlperf-inference`, `rocm-benchmarks`, `all-benchmarks`) that map to script-based installers.

### The "Installing" String in Benchmark Context

The string "Installing" appears in `src/installer.rs:2448`:
```rust
format!("[native] Installing {} via Rust module...", component.name)
```
This is a log message, not a benchmark status. The word "Running" appears only in `src/platform/wsl.rs:596` as a test fixture string, not related to benchmarks.

---

## 5. Component Status Tracking

### TaskStatus Enum (`src/app.rs:41-83`)

```rust
enum TaskStatus {
    Pending,   // progress == 0.0
    Running,   // 0.0 < progress < 1.0
    Done,      // progress == 1.0 && installed == true
    Failed,    // progress == 1.0 && installed == false
    Skipped,   // verification only: progress >= 1.0 && !installed (non-verification)
}
```

### Status Transitions

1. **Initial**: All components start with `progress: 0.0`, `installed: false` → `TaskStatus::Pending`
2. **ComponentStart** event (line 2177): Sets `progress: 0.05` → `TaskStatus::Running`
3. **Progress** events: Update `component.progress` → stays `Running` until 1.0
4. **ComponentComplete** event (line 2189): Sets `progress: 1.0`, `installed: success` → `Done` or `Failed`

### Stage Enum (`src/core/types.rs:92-107`)

```rust
pub enum Stage {
    Welcome, HardwareDetect, Preflight, ComponentSelect,
    Configuration, Confirm, Installing, Complete, Benchmarks, Recovery,
}
```

The `Stage::Installing` stage is entered when `start_installation()` is called (line 2139).

### InstallStatus Struct (`src/core/types.rs:405`)
```rust
pub struct InstallStatus {
    pub progress: f32,
    pub message: String,
}
```

### Where "Installing" Status Is Set

- `src/app.rs:2139`: `self.stage = Stage::Installing` — when user confirms install
- `src/app.rs:2181`: `self.install_status.message = format!("Installing {}", name)` — per-component start
- `src/installer.rs:2448`: Log message `"[native] Installing {} via Rust module..."`
- `src/installers/common/guard.rs:795`: `tracker.advance("Installing")` — guard/probe tracking

---

## Architecture

```
Config (config.json)
  └─ force_reinstall: bool
       │
       ├─ TUI path (app.rs:2128)
       │   ├─ FORCE=true
       │   ├─ PYTORCH_REINSTALL=true
       │   └─ MLSTACK_FORCE_REINSTALL=true
       │        │
       │        ├─ Native installer (installer.rs:2540, 2566)
       │   │    │   └─ Arch/ROCm: checks MLSTACK_FORCE_REINSTALL to decide all_pkgs vs need_install
       │   │    │       BUT: --needed flag in yay command contradicts reinstall intent
       │   │    │
       │   │    └─ Other components: config.force_reinstall in component configs
       │   │        BUT: Native dispatch uses ::default() → always false
       │   │
       │   └─ Script installer (installer.rs:3686)
       │       ├─ Sets env vars on spawned bash process
       │       └─ Appends --force flag if script supports it
       │
       └─ CLI path (bin/rusty.rs:742)
           └─ MLSTACK_FORCE_REINSTALL=1 only (missing FORCE, PYTORCH_REINSTALL)
```

### Key Gap

The native installer dispatch creates component configs with `::default()` which sets `force_reinstall: false`. The env vars (`MLSTACK_FORCE_REINSTALL`) are read **only** in the ROCm Arch path and the script-based installer path. Most pip-based native components (deepspeed, mpi4py, pytorch) never check the env var — their `force_reinstall` config field stays `false`.

---

## Start Here

Open **`src/installer.rs`** at line 2445 (`run_native_installer`) — this is the central dispatch where all 24 native components are installed. The ROCm force_reinstall logic is at lines 2539-2570. Then look at **`src/app.rs`** line 2128 for how env vars are set from the TUI toggle.

For the benchmark UI, start with **`src/widgets/benchmarks_page.rs`** at `load_benchmark_results()` (line 110) to understand the data flow, then the status rendering around lines 2011-2067.

---

## Open Questions / Risks

1. **ROCm force_reinstall is broken on Arch**: `yay -S --needed` skips already-installed packages, making force_reinstall a no-op.
2. **Native component configs ignore force_reinstall**: DeepSpeed, MPI4Py, PyTorch native installers use `Config::default()` which always has `force_reinstall: false`. The env var is never read by these components.
3. **bitsandbytes has unused uninstall**: `build_uninstall_command()` exists but is never called.
4. **CLI vs TUI env var inconsistency**: CLI only sets `MLSTACK_FORCE_REINSTALL`, TUI sets 3 vars.
5. **No reboot mechanism** for ROCm driver reinstalls, which may be required for kernel module changes.
6. **No "Running"/"Installing" benchmark status in TUI**: Benchmark page shows results from log files only; there's no live benchmark execution status in the UI.
