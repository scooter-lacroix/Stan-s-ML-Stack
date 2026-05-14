# Logging & Error Diagnostics Investigation

## Files Retrieved

1. `Cargo.toml` — dependencies (no logging crate)
2. `src/main.rs` — TUI entry point (trivial)
3. `src/lib.rs` (lines 77-105) — panic hook, TUI lifecycle
4. `src/config.rs` (full file) — config paths, log_dir definition
5. `src/installer.rs` (lines 1-3727, 3327-3827) — script execution, error handling, success determination
6. `src/installer.rs` (lines 1784-2022) — `is_up_to_date_output` false-positive path
7. `src/installer.rs` (lines 2093-2150) — `build_common_env`
8. `src/app.rs` (lines 108-119, 2490-2530) — log persistence to disk
9. `src/bin/rusty.rs` (lines 509-545) — `RustyExecutor`, the "Succeeded" reporter
10. `src/orchestrator/apply.rs` (lines 155-400) — `ApplySummary`, `ApplyEngine`, `ApplyExecutor` trait
11. `src/benchmark_logs.rs` (full file) — benchmark log directory resolution

---

## 1. Logging Framework

**Finding: No structured logging framework exists.**

- `Cargo.toml` has zero logging dependencies: no `log`, `tracing`, `env_logger`, `slog`, `fern`, or `log4rs`.
- The `tracing` crate only appears in `onnxruntime/` (a vendored submodule), not in the rusty-stack codebase itself.
- All "logging" is done via:
  - `InstallerEvent::Log(String, bool)` — an in-memory event channel (mpsc)
  - `println!` / `eprintln!` in `bin/rusty.rs` (CLI output only)
  - `eprintln!` in `lib.rs` (panic hook only)

**The only persistent log mechanism is in `app.rs` (lines 2494-2505):**
```rust
// Only write non-transient logs to disk
if !is_transient {
    let log_dir = std::path::Path::new(&self.config.log_dir);
    let log_path = log_dir.join("rusty-stack.log");
    if std::fs::create_dir_all(log_dir).is_ok() {
        let _ = std::fs::OpenOptions::new()
            .create(true).append(true).open(log_path)
            .and_then(|mut file| writeln!(file, "{}", entry));
    }
}
```

**Critical problem:** This disk logging only works in the **TUI mode** (via `app.rs` `on_tick`). When the CLI binary (`rusty`) runs batch mode for updates, it spawns the TUI as a subprocess with `.output()` — the subprocess runs headless and **never processes `InstallerEvent::Log` through the app event loop**. The events go into the channel but the TUI's `on_tick()` is never called because the terminal is not being rendered.

**Log file location:** `~/.mlstack/logs/rusty-stack.log` (derived from `config.rs` lines 26-30: `config_path.parent().parent().join("logs")` → `~/.mlstack/config/config.json` → `~/.mlstack/logs/`).

**Bottom line:** During `rusty update` → `RustyExecutor::apply_component()` → spawns TUI batch → log events are emitted to the channel but never written to disk. The parent `rusty` process only captures the child's stdout/stderr via `.output()`, and the TUI writes **nothing** to stdout/stderr during batch mode (it's all in-memory events). **No logs survive after the process exits.**

---

## 2. Error Capture in `installer.rs`

### Bash Script Path (`run_script`, line 3327)

**Stdout and stderr ARE both captured and streamed** in real-time:
- `child.stdout.take()` → spawned reader thread (line ~3670)
- `child.stderr.take()` → spawned reader thread (line ~3700)
- Both stream to `InstallerEvent::Log` via the sender
- Lines are buffered byte-by-byte up to 8KB per line (line ~3690)

**Exit code IS checked** (line 3735):
```rust
if !status.success() {
    let code = status.code().unwrap_or(-1);
    // Detailed error messages per exit code...
    bail!("{} failed: {} (exit code {})", ...);
}
```

### Native Rust Path (`execute_native_command`, line ~1850)

**Same pattern:**
- Stdout/stderr captured in separate threads (lines ~1920-1970)
- Exit code checked (line ~1993)
- Combined output captured for `is_up_to_date_output` pattern matching (line ~2012)

**Both paths properly capture errors and return `Err` on non-zero exit codes.**

### Verification Path (`run_verification_command`, line 1467)

- Uses `.output()` (blocking, captures stdout+stderr)
- Returns `Ok((output.status.success(), captured_output))` — **exit code determines success**

---

## 3. Success/Failure Determination — "Succeeded (1)" Root Cause

### The Update Flow

1. `rusty update` → `update_impl::apply_plan()` (rusty.rs:509)
2. `ApplyEngine::apply()` iterates items, calls `RustyExecutor::apply_component()` (apply.rs:302)
3. `RustyExecutor` spawns `rusty` binary with `MLSTACK_BATCH_MODE=1` and `MLSTACK_TARGET_COMPONENT=<id>` (rusty.rs:526-528)

### The Critical Bug: `MLSTACK_TARGET_COMPONENT` is Never Consumed

`MLSTACK_TARGET_COMPONENT` is set in `rusty.rs:528` but **never read anywhere** in the codebase. A `grep` for `TARGET_COMPONENT` only returns the line that sets it. The TUI has no code to:
- Filter components based on `MLSTACK_TARGET_COMPONENT`
- Auto-select a specific component
- Skip the component selection screen

**What actually happens:** The TUI starts, displays the component selection screen, and waits for user input. In batch mode with `.output()`, the child process's stdin is connected but no input is sent. The TUI **hangs waiting for keyboard input** and may eventually time out or exit.

### The False Positive Paths

**Path A — `is_up_to_date_output()` bypasses exit code check (installer.rs:1789-2017):**

```rust
fn is_up_to_date_output(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("up to date -- skipping")
        || lower.contains("up to date, skipping")
        || lower.contains("there is nothing to do")
        || lower.contains("already installed")
}
```

If yay/pacman prints "already installed" in its output (even with a non-zero exit code), `execute_native_command` returns `Ok(())` — **treated as success**. For ROCm specifically on Arch Linux, the native installer (installer.rs:2210) runs `filter_already_installed_pacman()` first, then if all packages are installed, returns early with `Ok(())` — so it never even executes the package manager. **A "reinstall" that does nothing is reported as success.**

**Path B — ROCm early-return for all-installed packages (installer.rs:2226-2237):**
```rust
if need_install.is_empty() {
    let _ = sender.send(InstallerEvent::Log(
        format!("[native] {} — all ROCm packages already installed, skipping", component.name),
        false,
    ));
    return Ok(());
}
```
This means if ROCm packages exist but are broken/misconfigured, the installer says "already installed, skipping" and reports success.

**Path C — `RustyExecutor` error swallowing (rusty.rs:540-546):**
```rust
Err(e) => {
    // If we can't run the binary, log a warning and succeed
    // so the apply continues for other components.
    eprintln!("  [WARN] Could not execute {:?}: {}. Component '{}' may need manual reinstallation.", rusty_bin, e, component_id);
    Ok(())
}
```
If the binary can't be executed at all, **it returns `Ok(())` — treated as success**. The "Succeeded (1)" output in `print_apply_summary` would show the component as succeeded even though nothing happened.

**Path D — TUI exits with 0 by default (lib.rs):** The TUI event loop exits via `app.should_exit = true` which returns `Ok(())`. Even if the installation never ran, the process exits with code 0.

### Most Likely Root Cause for "Succeeded (1)" with Failed ROCm

The most likely scenario is a combination of:
1. `MLSTACK_TARGET_COMPONENT` is not consumed → TUI doesn't know what to install
2. ROCm packages are detected as "already installed" by `filter_already_installed_pacman`
3. The native installer returns early with `Ok(())`
4. Verification may pass superficially (binaries exist on disk)
5. But the actual reinstall/repair never happened

---

## 4. Log File Configuration

### Config Paths (config.rs)

- Config file: `~/.mlstack/config/config.json`
- Log directory: `~/.mlstack/logs/` (derived as `config_path.parent().parent().join("logs")`)
- Log file: `~/.mlstack/logs/rusty-stack.log`
- Benchmark logs: `$HOME/.rusty-stack/logs/` or `${TMPDIR:-/tmp}/rusty-stack/logs/`

### Problems

1. **Log dir may not exist:** `create_dir_all` is only called on write, but if it fails, the error is silently discarded (`let _ = ...`)
2. **Benchmark vs installer log separation:** Benchmarks write to `$HOME/.rusty-stack/logs/`, installer writes to `~/.mlstack/logs/` — two different directories
3. **No log rotation:** The installer log is append-only with a 2000-line in-memory buffer that drains 500 lines at a time (app.rs:2490), but the on-disk file grows unbounded
4. **Batch mode produces no disk logs:** See section 1 above

---

## Root Cause Hypotheses (Ranked)

### H1: `MLSTACK_TARGET_COMPONENT` is dead code — TUI runs full install or hangs (HIGH)
The variable is set but never consumed. When `rusty update` tries to reinstall rocm, the TUI subprocess either:
- Runs all components (not just rocm), or
- Hangs waiting for user input in the component selection screen

The `RustyExecutor` only checks the child's exit code, not what actually happened.

### H2: "Already installed" short-circuit reports success without reinstalling (HIGH)
`filter_already_installed_pacman` + early `return Ok(())` means broken packages that are partially present are never reinstalled.

### H3: Binary execution failure is swallowed as success (MEDIUM)
The `Err(e)` arm in `RustyExecutor::apply_component()` returns `Ok(())` when the binary can't be executed.

### H4: No log persistence in batch/update mode (HIGH)
`InstallerEvent::Log` events are only persisted to disk by the TUI's `on_tick()` handler. When run as a subprocess with `.output()`, the TUI's event loop never processes these events → no disk logs → impossible to diagnose post-failure.

---

## Remaining Clarification Questions

1. **Was the update run via `rusty update` (CLI) or through the TUI?** The logging behavior differs significantly.
2. **What distro is the system?** The Arch/pacman `is_up_to_date_output` bypass is distro-specific.
3. **What does the actual "Succeeded (1)" output look like in full?** Is there also a "Failed" section that's empty?
4. **Is the `rusty` binary available on the system when updates run?** If not, Path C (error swallowing) is the exact cause.
5. **What was the ROCm reinstall supposed to fix?** If it was a version upgrade, the "already installed" check would still detect the old version as installed.

---

## Start Here

Open `src/bin/rusty.rs` lines 509-546 — this is where `RustyExecutor` is defined and where the "Succeeded" vs "Failed" determination is made. The `apply_component` method is the gatekeeper for success reporting, and it has three paths to false-positive success.

Then check `src/installer.rs` lines 2210-2240 for the ROCm native installer's `filter_already_installed_pacman` early return.
