# Code Context: ROCm Script Execution Pipeline & Silent Failure Analysis

## Files Retrieved
1. `src/installer.rs` (lines 1-200, 1680-2130) — Core installation engine: `run_installation()`, `run_native_installer()`, `execute_native_command()`, exit code handling
2. `src/installers/components/rocm.rs` (full file) — ROCm native installer: channel versions, package commands, `is_up_to_date_output()`
3. `src/installers/components/mod.rs` (full file) — `NATIVE_COMPONENT_IDS`, `is_native_component()`, `get_dependencies()`, topological sort
4. `src/orchestrator/apply.rs` (full file) — `ApplyEngine`, `ApplyExecutor` trait, `NoOpExecutor`, dependency-safe execution
5. `src/bin/rusty.rs` (lines 505-610) — `RustyExecutor`: the real update executor that spawns the TUI in batch mode
6. `src/adapter/mod.rs` (lines 345-360) — `RegistryExecutor`: adapter-based ApplyExecutor
7. `src/adapter/legacy_adapter.rs` (lines 27-130) — Legacy script execution path
8. `src/state.rs` (lines 167-197) — ROCm component definition (`id: "rocm"`, `script: ""`, `needs_sudo: true`)
9. `src/platform/registry.rs` (lines 74-80) — ROCm registry entry (`installer_script: "install_rocm.sh"`)
10. `src/app.rs` (lines 126-172, 2111-2150, 2419-2425) — TUI App init, `start_installation()`, `selected_components()`
11. `src/lib.rs` (lines 62-120) — `run_tui()` entry point
12. `src/config.rs` (line 125) — `batch_mode` reads from `MLSTACK_BATCH_MODE` env var

## Key Code

### Two Distinct Execution Paths for ROCm

**Path A: TUI Installer** (`rusty` with no args → `run_tui()` → `App::new()`)
```
main.rs → lib.rs:run_tui() → App::new(scripts_dir) → App::start_installation()
  → installer::run_installation(components, config, sudo_password, sender, input_rx)
    → is_native_component("rocm") == true
    → run_native_installer(component, ctx)
      → match "rocm" → RocmInstaller::new(RocmConfig::default())
        → builds apt/dnf/pacman commands based on distro
        → execute_native_command() for each command
```

**Path B: Update CLI** (`rusty update` → `RustyExecutor`)
```
bin/rusty.rs:run_update() → build_plan() → apply_plan()
  → RustyExecutor::apply_component("rocm", proposed_version)
    → spawns: rusty_bin with MLSTACK_BATCH_MODE=1 + MLSTACK_TARGET_COMPONENT=rocm
    → child process runs full TUI → exit code determines success
```

### Critical: RustyExecutor Spawns a Full TUI Process

```rust
// bin/rusty.rs:519-548
impl ApplyExecutor for RustyExecutor {
    fn apply_component(&self, component_id: &str, _proposed_version: &str) -> Result<(), String> {
        let rusty_bin = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("rusty"));
        let result = std::process::Command::new(&rusty_bin)
            .env("MLSTACK_BATCH_MODE", "1")
            .env("MLSTACK_TARGET_COMPONENT", component_id)
            .output();
        match result {
            Ok(output) => {
                if output.status.success() { Ok(()) }
                else { Err(format!("{} installation failed (exit {}): {}", ...)) }
            }
            Err(e) => {
                // CRITICAL: Returns Ok(()) on spawn failure!
                eprintln!("[WARN] Could not execute {:?}: {}", rusty_bin, e);
                Ok(())  // ← Silent success on binary execution failure!
            }
        }
    }
}
```

### ROCm Channel Version Mapping (hardcoded in Rust)
```rust
// installers/components/rocm.rs
pub enum RocmChannel {
    Legacy,  // 6.4.3, pkg: "6.4.60403-1"
    Stable,  // 7.1,    pkg: "7.1.70100-1"
    Latest,  // 7.2.1,  pkg: "7.2.1.70201-1"  ← HARDCODED, not 7.2.2!
}
```

### Exit Code Handling in Native Installer
```rust
// installer.rs:1967-2024 (execute_native_command)
if !status.success() {
    // UP-TO-DATE PATTERN MATCHING — potential silent success
    if is_up_to_date_output(&combined_output) {
        return Ok(());  // Non-zero exit → OK if output matches patterns
    }
    bail!("{} failed: {} (exit code {})", component_name, error_detail, code);
}
```

### Up-to-Date Output Detection (possible false positive)
```rust
// installer.rs:1781-1789
fn is_up_to_date_output(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("up to date -- skipping")
        || lower.contains("up to date, skipping")
        || lower.contains("there is nothing to do")
        || lower.contains("already installed")
        || (lower.contains("warning:") && lower.contains("up to date"))
}
```

### Pacman "All Already Installed" Early Return
```rust
// installer.rs:2357-2370 (inside run_native_installer "rocm" match arm)
// For Arch/pacman:
let need_install = RocmInstaller::filter_already_installed_pacman(&all_pkgs);
if need_install.is_empty() {
    // Returns Ok(()) without doing anything!
    return Ok(());
}
```

### Sudo Handling in Native Path
```rust
// installer.rs:1845-1853
let component_needs_sudo = component.needs_sudo && needs_sudo();
let sudo_pw: Option<&str> = if component_needs_sudo {
    ctx.sudo_password.as_deref()
} else {
    None
};
if component_needs_sudo && ctx.sudo_password.is_none() {
    bail!("{} requires sudo but no password provided", component.name);
}
```

## Architecture

### Update Flow (rocm reinstall)
1. `rusty update` → `run_scan()` detects installed components
2. `build_plan()` creates `PlanOutput` with items (rocm: current=7.2.0, proposed=7.2.2)
3. `apply_plan()` → `ApplyEngine::new(RustyExecutor)`
4. For each selected item: `RustyExecutor::apply_component("rocm", "7.2.2")`
5. Spawns child: `<rusty_bin>` with env `MLSTACK_BATCH_MODE=1` `MLSTACK_TARGET_COMPONENT=rocm`
6. **Child runs full TUI** → `App::new()` → loads ALL components → `selected_components()` returns only those with `.selected = true`
7. **PROBLEM**: `MLSTACK_TARGET_COMPONENT` is SET by the parent but the TUI App::new() never reads it. The env var is only set, never consumed to filter/autoselect components.
8. Child TUI starts at `Stage::Welcome` → waits for Enter key → never proceeds in batch mode (no auto-start)

### TUI Install Flow (native ROCm installer)
1. User selects rocm in TUI → `start_installation()`
2. `run_installation()` checks `is_native_component("rocm")` → true
3. `run_native_installer()` → match "rocm" arm
4. Creates `RocmInstaller::new(RocmConfig::default())` → `channel: Latest` (7.2.1)
5. Detects distro → builds apt/dnf/pacman commands
6. For each command → `execute_native_command()`
7. Non-zero exit: checks `is_up_to_date_output()` → if match, returns Ok(()) silently
8. Verification runs after installation → both must pass for final success

## Failure Mode Hypotheses

### H1: `MLSTACK_TARGET_COMPONENT` is Never Consumed (MOST LIKELY)
The `RustyExecutor` sets `MLSTACK_TARGET_COMPONENT=rocm` and `MLSTACK_BATCH_MODE=1` as env vars when spawning the child TUI process. However:
- `App::new()` calls `default_components()` with ALL components
- There is **no code** that reads `MLSTACK_TARGET_COMPONENT` to filter/autoselect
- The batch_mode flag is only read for config (`config.rs:125`), not for auto-starting installation
- The child TUI starts at `Stage::Welcome` and waits for keyboard input (Enter keypress)
- When run via `.output()` (captured stdout/stderr, no stdin), the TUI can't receive keyboard events
- The process likely exits cleanly (exit 0) because no TTY events are available

**Impact**: The update reports "Succeeded" because the child process exits 0, but rocm was never actually installed.

### H2: Binary Spawn Failure Returns Ok(())
In `RustyExecutor::apply_component()`, line 541-545:
```rust
Err(e) => {
    eprintln!("[WARN] Could not execute {:?}: {}", rusty_bin, e);
    Ok(())  // Silent success!
}
```
If the binary can't be executed (e.g., wrong path, permissions), the update reports success.

### H3: Version Mismatch — Hardcoded 7.2.1 vs Target 7.2.2
`RocmChannel::Latest` is hardcoded to `7.2.1` with `pkg_version: "7.2.1.70201-1"`. If the update system proposes `7.2.2` but the installer always installs `7.2.1`, the installation "succeeds" but the version doesn't change.

### H4: is_up_to_date_output() False Positive
If `apt-get install` outputs "already installed" or "there is nothing to do" (because 7.2.1 is already present), the native installer returns `Ok(())` despite the target being 7.2.2. This is pacman-specific logic but the pattern matching is broad.

### H5: Pacman Early Return for Already-Installed
On Arch systems, `filter_already_installed_pacman()` checks if all ROCm packages are installed and returns `Ok(())` immediately without reinstalling — even if the versions are wrong.

### H6: TUI Child Exits Cleanly on Missing TTY
When `RustyExecutor` calls `.output()`, the child's stdin is piped (no TTY). The TUI event loop uses `crossterm::event::poll()` which may fail or return immediately with no events. The `run_app()` loop may exit cleanly because `app.should_exit` could be set by error handling, or the TUI may panic and the panic hook sets `disable_raw_mode` + prints to stderr (which is captured and not surfaced).

## Start Here
Open `src/bin/rusty.rs` lines 505-610 — the `RustyExecutor` and `apply_plan()` function. This is the critical code path where the update invokes installation. Verify whether the spawned TUI child process actually executes the rocm installer or silently exits.

## Supervisor coordination
No supervisor coordination needed. The findings are self-contained. The most likely root cause is **H1**: `MLSTACK_TARGET_COMPONENT` is set as an env var but never consumed by the TUI App, causing the update to spawn a TUI that sits at the welcome screen and exits cleanly without installing anything.

## Open Questions
1. Does the child TUI process actually auto-start when `MLSTACK_BATCH_MODE=1`? The batch mode flag is read into config but there's no visible auto-start logic.
2. Is there additional code (perhaps in `app.rs` `on_tick` or `handle_key`) that detects batch mode and auto-advances through stages?
3. What does the TUI do when stdin has no TTY? Does crossterm's event poll return errors that cause an exit?
4. Where is the manifest version 7.2.2 for rocm defined? The `RocmChannel::Latest` hardcodes 7.2.1.
