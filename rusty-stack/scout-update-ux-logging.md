# Scout Findings: Update CLI Flow, Progress Reporting & Logging

## Files Retrieved

1. `src/bin/rusty.rs` (full file, ~1120 lines) — CLI entry point, update/apply flow, `RustyExecutor`
2. `src/orchestrator/apply.rs` (full file, ~580 lines) — `ApplyEngine`, `ApplyExecutor` trait, dependency sort
3. `src/orchestrator/planner.rs` (lines 1–100) — classification, `PlannerItem`, selection defaults
4. `src/installer.rs` (lines 1–1466, 3327–3757) — TUI's `run_installation()`, `run_script()` with streaming stdout/stderr
5. `src/config.rs` (full file, ~140 lines) — `InstallerConfig`, `log_dir` field
6. `src/benchmark_logs.rs` (full file) — existing benchmark log directory infrastructure
7. `src/app.rs` (lines 2130–2530) — TUI log-to-file writing, `push_log_ext()`
8. `Cargo.toml` — dependencies (no logging crate)
9. `src/lib.rs` (lines 55–115) — `run_tui()` entry point
10. `src/orchestrator/mod.rs` (full file) — module structure

---

## Issue 1: No Confirmation Prompt with `--all-safe`

### Root Cause: Confirmed — by design, but problematic

**File:** `src/bin/rusty.rs`, lines 260–265

```rust
} else if all_safe {
    // --all-safe: apply immediately without prompting
    println!("\nApplying {} safe updates...", plan.summary.selected);
    let apply_result = apply_plan(&plan);
    print_apply_summary(&apply_result);
```

When `--all-safe` is passed, the code explicitly **skips the confirmation prompt** and applies immediately. The comment says "apply immediately without prompting."

Compare with the interactive path (lines 270–287):
```rust
} else {
    println!("\nReady to apply {} updates.", plan.summary.selected);
    print!("Apply now? [y/N] ");
    let _ = io::stdout().flush();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() { ... }
}
```

The confirmation only happens when `all_safe == false` AND `scan_only == false` AND `plan.summary.selected > 0`.

**There is no middle ground.** `--all-safe --include-experimental` could result in a plan that includes non-safe items (if experimental items get selected somehow), but the flow still bypasses confirmation.

**Hypothesis:** The `--all-safe` flag was intended to mean "auto-apply safe items" but the code path doesn't show a plan-then-apply pattern — it just applies everything selected.

---

## Issue 2: No Live Progress During Apply

### Root Cause: `.output()` blocks until process completes, capturing all stdout/stderr

**File:** `src/bin/rusty.rs`, lines 523–534 (inside `RustyExecutor::apply_component`)

```rust
let result = std::process::Command::new(&rusty_bin)
    .env("MLSTACK_BATCH_MODE", "1")
    .env("MLSTACK_TARGET_COMPONENT", component_id)
    .output();  // <-- BLOCKING: waits for entire process to finish
```

**`.output()`** is a blocking call that:
1. Spawns the child process
2. Captures ALL stdout and stderr into memory buffers
3. Waits for the process to exit
4. Returns only after completion

This means the user sees "Applying 1 safe updates..." and then **nothing** until the entire child process finishes. For a ROCm reinstall that could take 10+ minutes, the terminal appears frozen.

**Contrast with TUI's approach:** The TUI's `run_script()` in `src/installer.rs` (lines 3327–3757) does it properly:
- Uses `.spawn()` instead of `.output()`
- Takes `child.stdout.take()` and `child.stderr.take()` 
- Spawns **dedicated reader threads** that stream lines via `BufReader`
- Sends `InstallerEvent::Log` and `InstallerEvent::Progress` per line
- The TUI app renders these in real-time

**Additional problem:** `MLSTACK_TARGET_COMPONENT` is **never read** anywhere in the codebase. A `grep -rn "TARGET_COMPONENT"` across all source files shows it's only set at line 528 of `rusty.rs` — no code reads this env var. This means the `rusty` binary is launched, enters TUI mode (or no-args mode), and may install **everything**, not just the targeted component.

---

## Issue 3: No Detailed Logs / Silent Failures

### Root Cause: Multiple compounding issues

#### 3a. No logging framework
**File:** `Cargo.toml`

The project has **no logging dependency** — no `log`, `tracing`, `env_logger`, `log4rs`, or `fern`. All output is via raw `println!`/`eprintln!` calls.

#### 3b. The CLI path never writes log files
**File:** `src/bin/rusty.rs`

The CLI `apply_plan()` function:
- Calls `RustyExecutor::apply_component()` which uses `.output()` (captures to memory)
- Only prints the final `ApplySummary` via `print_apply_summary()`
- **Never writes to any log file**
- The captured stdout from the child process is **discarded** entirely

Compare with the TUI path:
**File:** `src/app.rs`, lines 2494–2508
```rust
let log_dir = std::path::Path::new(&self.config.log_dir);
let log_path = log_dir.join("rusty-stack.log");
if std::fs::create_dir_all(log_dir).is_ok() {
    let _ = std::fs::OpenOptions::new()
        .create(true).append(true).open(log_path)
        .and_then(|mut file| writeln!(file, "{}", entry));
}
```

The TUI writes to `~/.mlstack/logs/rusty-stack.log`. The CLI does not.

#### 3c. Silent false-success on process spawn failure
**File:** `src/bin/rusty.rs`, lines 536–542

```rust
Err(e) => {
    // If we can't run the binary, log a warning and succeed
    // so the apply continues for other components.
    eprintln!("  [WARN] Could not execute {:?}: {}. Component '{}' may need manual reinstallation.", rusty_bin, e, component_id);
    Ok(())  // <-- RETURNS SUCCESS even though nothing was installed!
}
```

If the binary can't be executed (e.g., path issue, permissions), the executor returns `Ok(())`, making the apply engine think it succeeded.

#### 3d. False success from child process exit code
**File:** `src/bin/rusty.rs`, lines 530–534

The child `rusty` process launched in "batch mode" sets `MLSTACK_BATCH_MODE=1` and `MLSTACK_TARGET_COMPONENT`, but:
- `MLSTACK_TARGET_COMPONENT` is never consumed — the child doesn't know which component to install
- The child launches the TUI (since no subcommand is given), which in batch mode installs **all selected components**, not just the one requested
- If the child exits 0 (because it launched and completed something), the parent reports success for that specific component, even though it may have installed something entirely different

#### 3e. `ApplyEngine` doesn't log internally
**File:** `src/orchestrator/apply.rs`

The apply engine has no logging. It only produces an `ApplySummary` struct with success/failed/blocked buckets. There's no intermediate logging during execution.

---

## Architecture

```
CLI Flow (rusty update --all-safe):
  main() → update_impl::run()
    → run_scan()                    [detect installed components]
    → build_plan()                  [classify updates]
    → apply_plan()                  [if --all-safe, no confirmation]
      → RustyExecutor::apply_component()  [for each selected item]
        → Command::new("rusty").output()  [BLOCKING, captures all output]
        → Returns Ok/Err based on exit code
      → ApplyEngine sorts deps, runs each, collects summary
    → print_apply_summary()         [prints final tally]

TUI Flow (for contrast):
  main() → run_tui() → App::event_loop
    → App::start_installation()
      → thread::spawn(run_installation(...))  [background thread]
        → run_script() or run_native_installer()
          → Command::new("bash").spawn()     [non-blocking]
          → stdout/stderr reader threads      [streaming]
          → InstallerEvent::Log/Progress      [sent via mpsc channel]
      → App::poll_installer()                [renders events in TUI]
    → push_log_ext() writes to rusty-stack.log
```

**Key gap:** The CLI path has no streaming, no log files, and the `RustyExecutor` spawns a new `rusty` binary as a subprocess rather than calling the installer functions directly.

---

## Summary of Root Causes

| Issue | Root Cause | Severity |
|-------|-----------|----------|
| No confirmation | `--all-safe` intentionally skips prompt (line 261) | Medium — design decision |
| No progress | `.output()` blocks (line 529); no streaming stdout | High — UX broken for long ops |
| Silent failures | Spawn error returns `Ok(())` (line 542); `TARGET_COMPONENT` never consumed; no log file writes | High — data integrity |

---

## Start Here

Open `src/bin/rusty.rs` — specifically the `apply_plan()` function (line ~505) and the `RustyExecutor` struct (line ~517). This is where all three issues converge:
1. The executor uses `.output()` instead of `.spawn()` + streaming
2. The executor returns `Ok(())` on spawn failure
3. The captured child stdout is never displayed or logged
4. `MLSTACK_TARGET_COMPONENT` is never consumed by the child

The fix likely requires:
- Replacing `.output()` with `.spawn()` + BufReader streaming (pattern exists in `installer.rs:run_script()`)
- Adding a `--yes` or confirmation step for `--all-safe`
- Writing apply logs to `~/.mlstack/logs/update-{timestamp}.log`
- Either consuming `MLSTACK_TARGET_COMPONENT` in the TUI flow OR calling installer functions directly instead of spawning a subprocess

---

## Remaining Questions

1. **Should `--all-safe` get a confirmation?** The current design treats it as "I know what I want, just do it." A `--yes` flag could be added for true non-interactive use, while `--all-safe` alone gets a prompt.

2. **Should `RustyExecutor` call installer functions directly?** Instead of spawning a child `rusty` process, the CLI could import and call `run_installation()` with the right `Component` list, reusing the TUI's streaming infrastructure. This would also fix the `TARGET_COMPONENT` issue.

3. **Is there a planned logging framework?** Adding `tracing` or at minimum `log` + `env_logger` would help across all three issues. The existing `benchmark_logs.rs` infrastructure could be extended for update logs.

4. **What should the child process actually do?** Currently `rusty` with no args launches the TUI. The `MLSTACK_TARGET_COMPONENT` env var was clearly intended to scope the TUI's batch mode to a single component, but this was never implemented.
