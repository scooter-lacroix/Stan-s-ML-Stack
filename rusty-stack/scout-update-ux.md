# Update Flow UX & Progress Reporting Investigation

## Files Retrieved

1. `src/bin/rusty.rs` (lines 1-790) — CLI entry point with all subcommand handlers, including `update_impl::run()`, `apply_plan()`, and `RustyExecutor`
2. `src/orchestrator/apply.rs` (lines 1-400+) — `ApplyEngine` that executes planner items in dependency order; defines `ApplyExecutor` trait
3. `src/orchestrator/planner.rs` (lines 160-200) — `PlannerOptions` with `all_safe`, `include_experimental` flags
4. `src/lib.rs` (lines 1-158) — `run_tui()` entry point, no batch-mode fast path
5. `src/config.rs` (line 125) — `batch_mode` reads `MLSTACK_BATCH_MODE` env var

## Key Code

### 1. Confirmation Bypass (`--all-safe`)

**File:** `src/bin/rusty.rs`, lines ~355-370 (inside `update_impl::run()`)

```rust
} else if all_safe {
    // --all-safe: apply immediately without prompting
    println!("\nApplying {} safe updates...", plan.summary.selected);
    let apply_result = apply_plan(&plan);
    print_apply_summary(&apply_result);
```

**Root cause for no confirmation:** When `--all-safe` is passed, the code skips the confirmation prompt entirely and goes straight to `apply_plan()`. There's no `Apply now? [y/N]` prompt. This is by design ("apply immediately without prompting"), but the only feedback is a single `println!` before the blocking call.

### 2. Blocking Behavior — `.output()` Captures All stdout/stderr

**File:** `src/bin/rusty.rs`, lines ~526-529

```rust
let result = std::process::Command::new(&rusty_bin)
    .env("MLSTACK_BATCH_MODE", "1")
    .env("MLSTACK_TARGET_COMPONENT", component_id)
    .output();
```

**Root cause for the hang:** `Command::output()` is a **blocking call** that:
- Waits for the child process to complete
- Captures ALL stdout and stderr into memory buffers
- Returns only after the child exits
- Produces **zero output** to the parent's terminal during execution

The child process being launched is **the same `rusty` binary** — which calls `run_tui()` (see `src/lib.rs`). Even though `MLSTACK_BATCH_MODE=1` is set, `run_tui()` does NOT check this env var. It unconditionally:
1. Calls `enable_raw_mode()` — puts the terminal in raw mode
2. Enters the alternate screen (`EnterAlternateScreen`)
3. Starts the full TUI event loop with ratatui

**The child TUI runs in raw mode with alternate screen, but its output is piped/captured by `.output()`.** The parent sees nothing until the child exits. The user's terminal appears frozen because:
- The parent is blocked on `.output()`
- The child's output (TUI rendering) goes to the pipe, not the terminal
- There is no progress indicator or timeout

### 3. No Progress Reporting Mechanism

**File:** `src/orchestrator/apply.rs`, `ApplyEngine::apply()` method

The apply engine iterates over items sequentially and calls `self.executor.apply_component()` for each one. There is **no progress callback, no progress bar, no spinner, no logging** during execution. It's a bare synchronous loop:

```rust
for item in &sorted {
    // ... dependency check ...
    let status = self.executor.apply_component(...);
    match status { ... }
}
```

The `ApplyExecutor` trait has no progress hook:
```rust
pub trait ApplyExecutor: Send + Sync {
    fn apply_component(&self, component_id: &str, proposed_version: &str) -> Result<(), String>;
}
```

### 4. TUI Does Not Handle Batch Mode

**File:** `src/lib.rs`, `run_tui()` function (lines ~98-158)

`run_tui()` does NOT check `MLSTACK_BATCH_MODE` or `MLSTACK_TARGET_COMPONENT`. It always launches the full interactive TUI. The `batch_mode` flag in `config.rs` is read but used only for installer behavior, not for CLI output mode.

## Architecture

```
CLI: rusty update --all-safe --include-experimental
  │
  ├─ Phase 1: run_scan()        → detect installed components
  ├─ Phase 2: build_plan()      → classify updates, filter to safe-only
  ├─ Phase 3: apply_plan()      → ApplyEngine.apply()
  │   └─ RustyExecutor.apply_component()
  │       └─ Command::new(rusty_bin)
  │           .env(MLSTACK_BATCH_MODE=1)
  │           .output()          ← BLOCKS HERE, captures all output
  │           └─ child runs run_tui()  ← full TUI launched!
  └─ Phase 4: print_apply_summary()
```

## Root Cause Hypotheses

### H1: `.output()` swallows all child output (PRIMARY)
`Command::output()` captures stdout/stderr into memory. The child process (full TUI) writes its ratatui frames to the captured pipe. The parent terminal shows nothing until `.output()` returns. This is the main reason for the "hang" — the user sees the `println!("Applying N safe updates...")` message and then nothing until completion or ctrl+c.

### H2: Child process launches full TUI instead of headless install
The `RustyExecutor` re-invokes the same binary, which falls through to `run_tui()`. The TUI sets up raw mode and alternate screen, but since output is piped, these terminal escapes go to the pipe buffer. The TUI likely runs to completion (hence "Succeeded"), but:
- It may wait for user input events that never come (crossterm key events on a piped stdin)
- Or it may detect non-TTY and exit immediately (depending on crossterm behavior)
- The ctrl+c from the user likely kills the parent, which kills the child, and the child's buffered output is discarded

### H3: No progress reporting infrastructure
There is no progress bar, spinner, status line, or periodic output between the `println!("Applying...")` and the final `print_apply_summary()`. Even if the child output were streamed, there's no mechanism to show per-component progress.

### H4: `--all-safe` intentionally skips confirmation
This is documented behavior (`// --all-safe: apply immediately without prompting`), but combined with H1, it means the user has zero interactivity: no confirmation prompt AND no progress output.

## Suggested Fixes

1. **Replace `.output()` with `.spawn()` + streaming**: Use `Command::spawn()` and stream stdout/stderr in real-time with `BufReader`. Or use `.status()` if child output isn't needed.
2. **Add a headless/batch execution path**: `run_tui()` should detect `MLSTACK_BATCH_MODE` and `MLSTACK_TARGET_COMPONENT` and run a non-TUI installation path instead.
3. **Add progress output in `ApplyEngine`**: Print a line before and after each component: `println!("  Installing {}...", component_id)` / `println!("  ✓ {} done", component_id)`.
4. **Consider a confirmation prompt even with `--all-safe`**: Or at minimum print a clear "This will apply N updates without further confirmation" message before proceeding.

## Remaining Questions

1. Does the child TUI actually complete, or does it hang waiting for terminal events? (Need to test on real hardware — the ctrl+c behavior suggests it completes only because the user interrupted it.)
2. Is there an existing batch/headless install path for individual components, or does the TUI always need to be involved?
3. What do the installer scripts (`../scripts/`) look like — could the executor invoke them directly instead of re-running the full TUI binary?

## Start Here

Open `src/bin/rusty.rs` — specifically the `apply_plan()` function (line ~500) and `RustyExecutor` (line ~515). This is where the blocking occurs and where fixes should be applied. The `run_tui()` function in `src/lib.rs` (line ~98) is the second file to modify to add a batch-mode fast path.
