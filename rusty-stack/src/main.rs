//! rusty — Unified CLI for Rusty Stack.
//!
//! Provides all Rusty Stack CLI subcommands in a single binary:
//!
//! - `rusty` (no args) — Launch the TUI installer
//! - `rusty update` — Scan, plan, and apply component updates
//! - `rusty upgrade` — Upgrade the Rusty Stack binary itself
//! - `rusty bench` — Run benchmarks
//! - `rusty verify` — Verify ML Stack installation
//! - `rusty uninstall` — Uninstall the Rusty-managed ML stack (cross-shell + cross-distro)
//! - `rusty reinstall` — Force reinstall: uninstall then relaunch the installer
//!
//! # Usage
//!
//! ```text
//! rusty [COMMAND]
//!
//! Commands:
//!   update   Update Rusty Stack ML components
//!   upgrade  Upgrade Rusty Stack to the latest version
//!   bench    Run benchmarks
//!   verify   Verify ML Stack installation
//!
//! Options:
//!   -h, --help     Show help
//!   -V, --version  Show version
//! ```

use clap::{Parser, Subcommand};
use serde::Serialize;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process;

use rusty_stack::orchestrator::planner::PlanOutput;

/// Inline, animated apply-phase UI (ratatui `Viewport::Inline`). Pure state
/// + renderer are tested; the live panel is only constructed on a TTY.
#[cfg(feature = "tui")]
mod apply_ui;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "rusty-stack",
    version = VERSION,
    about = "Rusty Stack — AMD GPU ML environment manager",
    long_about = "Unified CLI for Rusty Stack. Provides component updates, binary upgrades, \
                  and benchmark execution. Run without arguments to launch the TUI installer."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Subcommands>,
}

#[derive(Subcommand, Debug)]
enum Subcommands {
    /// Update Rusty Stack ML components.
    ///
    /// Scans installed components, resolves the manifest, builds an update plan,
    /// classifies updates (safe/guarded/blocked/candidate/experimental), and
    /// applies selected updates.
    ///
    /// FLAGS:
    ///   --scan-only          List available updates without applying
    ///   --all-safe           Auto-select and apply safe updates (with countdown)
    ///   --include-experimental Include experimental-tier updates with --all-safe
    ///   --yes                Skip all prompts (auto-apply immediately)
    ///   --json               Machine-readable JSON output
    ///
    /// INTERACTIVE MODE (no flags):
    ///   Displays a numbered list of available updates.
    ///   Use spacebar to toggle selection, Enter to confirm.
    ///   Press 'a' to select all, 'n' to select none, Esc to cancel.
    ///
    /// EXAMPLES:
    ///   rusty-stack update --scan-only              # List updates only
    ///   rusty-stack update --all-safe              # Auto-apply safe updates
    ///   rusty-stack update --all-safe --yes        # Auto-apply safe updates (no countdown)
    ///   rusty-stack update --include-experimental   # Include experimental in --all-safe
    ///   rusty-stack update pytorch onnx             # Update specific components
    Update {
        /// Produce plan without applying any changes.
        #[arg(long)]
        scan_only: bool,

        /// Apply only safe-classified updates.
        /// Shows plan with countdown confirmation unless --yes is also set.
        #[arg(long)]
        all_safe: bool,

        /// Include experimental components in the plan.
        #[arg(long)]
        include_experimental: bool,

        /// Skip confirmation prompts (auto-apply). Useful for scripting/CI.
        #[arg(long, short = 'y')]
        yes: bool,

        /// Force JSON output mode.
        #[arg(long)]
        json: bool,

        /// Specific components to update.
        /// Each targeted component still runs a full compatibility check.
        #[arg(value_name = "COMPONENT")]
        components: Vec<String>,
    },

    /// Upgrade Rusty Stack to the latest version.
    ///
    /// Downloads and replaces the Rusty Stack binary with the latest released
    /// version. Checks manifest version compatibility, verifies binary
    /// integrity, and supports rollback on failure.
    Upgrade {
        /// Skip interactive confirmation prompts.
        /// Output is structured JSON in this mode.
        #[arg(long, short = 'y')]
        yes: bool,

        /// Path to the binary to upgrade.
        /// Defaults to the current executable.
        #[arg(long)]
        binary_path: Option<PathBuf>,

        /// Directory for backup files.
        /// Defaults to ~/.mlstack/backups.
        #[arg(long)]
        backup_dir: Option<PathBuf>,

        /// Path to the cached remote manifest.
        /// Defaults to ~/.mlstack/cache/remote_manifest.json.
        #[arg(long)]
        cached_manifest: Option<PathBuf>,

        /// Dry run: check for available upgrades without applying.
        #[arg(long)]
        dry_run: bool,
    },

    /// Run benchmarks.
    ///
    /// Executes the specified benchmark and prints results.
    /// Use --json for machine-readable output.
    Bench {
        /// Benchmark to run.
        ///
        /// Pre-installation: gpu-capability, memory-bandwidth, tensor-core
        /// GPU Performance: gemm
        /// Component: pytorch, flash-attention, vllm, deepspeed, megatron
        /// Combined: all-pre, all
        #[arg(value_name = "BENCHMARK")]
        benchmark: Option<String>,

        /// Output results in JSON format.
        #[arg(long)]
        json: bool,

        /// List available benchmarks.
        #[arg(long)]
        list: bool,
    },

    /// Check for Rust dependency updates from crates.io.
    ///
    /// Scans Cargo.toml direct dependencies, queries crates.io for the latest
    /// stable versions, and reports which dependencies have updates available.
    /// Respects a configurable lag period to avoid freshly-published versions.
    ///
    /// Exit codes: 0 = all up to date, 1 = updates available, 2 = error.
    Deps {
        /// Set lag period in days (default: 7).
        /// Only reports updates published more than this many days ago.
        #[arg(long, default_value = "7")]
        lag: u64,

        /// Show full API responses for debugging.
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Path to the crate directory containing Cargo.toml.
        /// Defaults to the rusty-stack crate adjacent to this binary.
        #[arg(long)]
        dir: Option<PathBuf>,

        /// Output results in JSON format.
        #[arg(long)]
        json: bool,
    },

    /// Verify ML Stack installation.
    ///
    /// Checks component installation status using native Rust detection
    /// (no shell subprocesses). Supports three modes:
    ///
    /// - --full: Core component verification (equivalent to verify_installation.sh)
    /// - --enhanced: All-component verification (equivalent to enhanced_verify_installation.sh)
    /// - --build: Verify and identify components needing rebuild (equivalent to verify_and_build.sh)
    Verify {
        /// Run full verification (core components: ROCm, PyTorch, Triton, MPI4Py, DeepSpeed, ML Stack Core).
        #[arg(long)]
        full: bool,

        /// Run enhanced verification (all components including ROCm tools, Python imports, environment).
        #[arg(long)]
        enhanced: bool,

        /// Run verify-and-build mode (core + build-critical components; identifies rebuild targets).
        #[arg(long)]
        build: bool,

        /// Output results in JSON format.
        #[arg(long)]
        json: bool,
    },

    /// Uninstall the Rusty-managed ML stack.
    ///
    /// Removes the Python ML packages, ROCm/amdgpu system packages (unless
    /// --keep-rocm), /opt/rocm (unless --keep-rocm), the env files Rusty wrote,
    /// the shell sourcing lines, and the installed-component registry.
    /// Cross-shell (fish/bash/zsh) and cross-distro (apt/dnf/pacman/zypper).
    Uninstall {
        /// Keep ROCm/amdgpu system packages and /opt/rocm (only remove Python +
        /// env/registry).
        #[arg(long)]
        keep_rocm: bool,

        /// Also remove ~/.mlstack/ (logs, cache, global venv).
        #[arg(long)]
        purge_dir: bool,

        /// Skip the confirmation notice.
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Force reinstall: uninstall the stack, then relaunch the TUI installer.
    Reinstall {
        /// Keep ROCm/amdgpu system packages during the uninstall phase.
        #[arg(long)]
        keep_rocm: bool,

        /// Also remove ~/.mlstack/ during the uninstall phase.
        #[arg(long)]
        purge_dir: bool,
    },

    /// Install the ML stack.
    ///
    /// Ensures the install target env exists and launches the installer. Use
    /// `--env <name>` to isolate ALL components into `~/.mlstack/envs/<name>/`
    /// (Tenet 1: install-to-env), or `--global` for the single managed global
    /// env `~/.mlstack/global/`. The sourcing command + path are printed.
    Install {
        /// Isolate all components into ~/.mlstack/envs/<name>/ (named env).
        #[arg(long, value_name = "NAME")]
        env: Option<String>,

        /// Install into the single managed global env ~/.mlstack/global/ (default).
        #[arg(long)]
        global: bool,
    },
}

// ===========================================================================
// Update subcommand implementation
// ===========================================================================

// ===========================================================================
// Shared sudo-credential helpers (install + uninstall/reinstall)
// ===========================================================================

/// Sudo credential resolution shared by the install and uninstall/reinstall
/// paths. `sudo -n` fails whenever a password is required; these helpers read a
/// password from a TTY (no echo) so privileged steps can run via an askpass
/// helper unattended.
mod sudo_creds {
    use std::io::{IsTerminal, Write};

    /// Read a password from TTY without echo. Returns `None` if not a TTY or
    /// reading fails. Uses raw stdin reads (avoids `read_line` buffering).
    pub fn read_password_from_tty() -> Option<String> {
        if !std::io::stdin().is_terminal() {
            // Not a TTY — do not consume piped stdin as sudo password.
            return None;
        }
        // rpassword reads from /dev/tty with echo disabled and restores the
        // terminal state itself — robust across terminals. This replaces a
        // hand-rolled termios raw byte-at-a-time read that interacted poorly
        // with stdin buffering and delivered a corrupted password to sudo's
        // askpass helper (3 "incorrect password" attempts even though the
        // user typed it correctly).
        match rpassword::read_password() {
            Ok(pw) => {
                if pw.is_empty() {
                    None
                } else {
                    Some(pw)
                }
            }
            Err(_) => None,
        }
    }

    /// True when sudo can run non-interactively for the current user (cached
    /// credential or NOPASSWD policy).
    pub fn can_sudo_non_interactive() -> bool {
        std::process::Command::new("sudo")
            .arg("-n")
            .arg("true")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Resolve a sudo password for the uninstall/reinstall privileged steps.
    ///
    /// Precedence: already-root ⇒ `None` (no sudo needed);
    /// `MLSTACK_SUDO_PASSWORD` env; non-interactive sudo already works ⇒
    /// `None`; otherwise prompt on the TTY.
    pub fn resolve_for_uninstall() -> Option<String> {
        #[cfg(unix)]
        if unsafe { libc::geteuid() } == 0 {
            return None;
        }
        if let Ok(pw) = std::env::var("MLSTACK_SUDO_PASSWORD") {
            if !pw.is_empty() {
                return Some(pw);
            }
        }
        if can_sudo_non_interactive() {
            return None;
        }
        if std::io::stdin().is_terminal() {
            eprint!("    sudo password for uninstall: ");
            let _ = std::io::stderr().flush();
            read_password_from_tty()
        } else {
            None
        }
    }
}

mod update_impl {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread::JoinHandle;

    /// TTY-only activity indicator for blocking update phases.
    ///
    /// Progress never goes to stdout, so JSON mode remains machine-readable.
    struct ActivityIndicator {
        active: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl ActivityIndicator {
        fn start(message: &str) -> Self {
            let active = Arc::new(AtomicBool::new(true));
            let thread_active = Arc::clone(&active);
            let message = message.to_string();
            let handle = std::thread::spawn(move || {
                const FRAMES: &[char] = &[
                    '\u{280b}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283c}', '\u{2834}',
                    '\u{2826}', '\u{2827}', '\u{2807}', '\u{280f}',
                ];
                let mut frame = 0usize;
                while thread_active.load(Ordering::Relaxed) {
                    eprint!("\r  {} {}", FRAMES[frame % FRAMES.len()], message);
                    let _ = io::stderr().flush();
                    frame = frame.wrapping_add(1);
                    std::thread::sleep(std::time::Duration::from_millis(90));
                }
            });
            Self {
                active,
                handle: Some(handle),
            }
        }

        fn finish(mut self, message: &str) {
            self.active.store(false, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            // Clear the entire line to remove any spinner artifacts
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
            eprintln!("  ✓ {}", message);
            let _ = io::stderr().flush();
        }
    }

    impl Drop for ActivityIndicator {
        fn drop(&mut self) {
            self.active.store(false, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        }
    }

    /// Pure renderer for the interactive selection body.
    ///
    /// Returns the full multi-line text (terminated with CRLF) that should be
    /// written to the terminal for a given selection/cursor state. Extracted so
    /// the layout can be unit-tested without a live TTY.
    ///
    /// Column widths are derived from the longest value in each column so rows
    /// stay aligned regardless of version-string length.
    pub(super) fn render_selection_body(
        plan: &PlanOutput,
        selected_indices: &[usize],
        cursor_idx: usize,
    ) -> String {
        let total = plan.plan.len();

        let rows: Vec<(String, String, String, String)> = plan
            .plan
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let current = if item.current_version.is_empty() {
                    "not installed".to_string()
                } else {
                    item.current_version.clone()
                };
                (
                    format!("{}.", idx + 1),
                    item.component_id.clone(),
                    current,
                    item.proposed_version.clone(),
                )
            })
            .collect();

        let width_num = rows.iter().map(|r| r.0.chars().count()).max().unwrap_or(1);
        let width_name = rows
            .iter()
            .map(|r| r.1.chars().count())
            .max()
            .unwrap_or(0)
            .max(4);
        let width_cur = rows
            .iter()
            .map(|r| r.2.chars().count())
            .max()
            .unwrap_or(0)
            .max(5);
        let width_new = rows
            .iter()
            .map(|r| r.3.chars().count())
            .max()
            .unwrap_or(0)
            .max(3);

        let mut out = String::new();
        out.push_str("Interactive Update Selection\r\n");
        out.push_str("===========================\r\n\r\n");
        out.push_str(
            "Controls: \u{2191}\u{2193} Navigate | Space Toggle | Enter Confirm | a:Select All | n:Select None | Esc:Cancel\r\n\r\n",
        );

        for (idx, (num, name, current, proposed)) in rows.iter().enumerate() {
            let is_selected = selected_indices.contains(&idx);
            let is_cursor = idx == cursor_idx;
            let cursor_marker = if is_cursor { '>' } else { ' ' };
            let marker = if is_selected { "[x]" } else { "[ ]" };
            let class = &plan.plan[idx].classification;
            out.push_str(&format!(
                "{} {} {:<width_num$} {:<width_name$} {:>width_cur$} \u{2192} {:>width_new$} ({})\r\n",
                cursor_marker,
                marker,
                num,
                name,
                current,
                proposed,
                class,
                width_num = width_num,
                width_name = width_name,
                width_cur = width_cur,
                width_new = width_new,
            ));
        }

        out.push_str(&format!(
            "\r\nSelected: {} of {}\r\n",
            selected_indices.len(),
            total
        ));
        out
    }

    /// Interactive selection UI for update plan items.
    ///
    /// Inline redraw model — does NOT take over the terminal:
    /// - Raw mode is enabled only for single-key input.
    /// - On each redraw the cursor is moved up N lines and every line is
    ///   cleared in place (\x1b[2K), so prior output scrolls naturally
    ///   instead of being wiped by a full-screen clear.
    /// - The cursor is hidden during the loop and restored on every exit path.
    /// - A panic hook guarantees raw mode is disabled and the cursor is shown
    ///   even if a panic (or Ctrl+C abort) interrupts the loop.
    /// - Column widths are derived from the actual data so version strings of
    ///   different lengths remain aligned.
    fn interactive_selection(plan: &PlanOutput) -> Result<PlanOutput, String> {
        use crossterm::{
            event::{self, Event, KeyCode, KeyEventKind},
            execute,
            terminal::{disable_raw_mode, enable_raw_mode},
        };
        use std::io::{self, Write};

        let total = plan.plan.len();

        let mut selected_indices: Vec<usize> = plan
            .plan
            .iter()
            .enumerate()
            .filter(|(_, item)| item.selected)
            .map(|(i, _)| i)
            .collect();

        // ---- Non-TTY fallback: one prompt, one read, no raw mode ----
        if !io::stdin().is_terminal() {
            return interactive_selection_nontty(plan);
        }

        // ---- TTY path ----
        let mut cursor_idx: usize = 0;
        let mut stdout = io::stdout();

        // Install a panic hook so an unexpected panic restores the terminal.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new({
            let prev = std::sync::Arc::new(prev_hook);
            move |info| {
                let _ = disable_raw_mode();
                let mut out = io::stdout();
                let _ = execute!(out, crossterm::cursor::Show);
                let _ = out.flush();
                if let Ok(p) = std::sync::Arc::try_unwrap(std::sync::Arc::clone(&prev)) {
                    p(info);
                } else {
                    prev(info);
                }
            }
        }));

        let restore_terminal = || {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), crossterm::cursor::Show);
            let _ = io::stdout().flush();
        };

        if let Err(e) = enable_raw_mode() {
            // Restore the previous panic hook before bailing.
            let _ = std::panic::take_hook();
            return Err(format!("Failed to enable raw mode: {e}"));
        }
        let _ = execute!(stdout, crossterm::cursor::Hide);
        let _ = stdout.flush();

        // Render once and remember how many lines we drew, so subsequent
        // redraws can rewind by exactly that many lines.
        let mut drawn_lines: usize = 0;
        let mut first_draw = true;

        // Body text is built by the pure `render_selection_body` helper so it
        // can be unit-tested without a TTY.

        loop {
            // Move cursor up `drawn_lines` lines and clear each one, then render.
            if !first_draw && drawn_lines > 0 {
                // \x1b[<n>A  = up n lines; \x1b[2K clears the current line.
                let _ = write!(stdout, "\x1b[{}A", drawn_lines);
                for _ in 0..drawn_lines {
                    let _ = write!(stdout, "\r\x1b[2K");
                }
                // After clearing, cursor is at the top of our block, col 0.
                // Move down one line per cleared row is unnecessary because we
                // write each new line with \r\n which advances as it goes.
            }

            let body = render_selection_body(plan, &selected_indices, cursor_idx);
            drawn_lines = body.matches('\n').count();
            let _ = write!(stdout, "{body}");
            let _ = stdout.flush();
            first_draw = false;

            // Block on a single key event.
            let ev = match event::read() {
                Ok(ev) => ev,
                Err(e) => {
                    restore_terminal();
                    return Err(format!("Failed to read event: {e}"));
                }
            };

            if let Event::Key(key) = ev {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    // Ctrl+C in raw mode arrives as Char('c') + CONTROL (ISIG is
                    // cleared by enable_raw_mode, so it is NOT delivered as
                    // SIGINT). Treat it as an explicit cancel, matching Esc.
                    KeyCode::Char('c')
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL) =>
                    {
                        restore_terminal();
                        return Err("Cancelled by user".to_string());
                    }
                    KeyCode::Up => {
                        cursor_idx = cursor_idx.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        if cursor_idx + 1 < total {
                            cursor_idx += 1;
                        }
                    }
                    KeyCode::Char(' ') => {
                        if selected_indices.contains(&cursor_idx) {
                            selected_indices.retain(|&i| i != cursor_idx);
                        } else {
                            selected_indices.push(cursor_idx);
                        }
                    }
                    KeyCode::Char('a') => {
                        selected_indices = (0..total).collect();
                    }
                    KeyCode::Char('n') => {
                        selected_indices.clear();
                    }
                    KeyCode::Enter => {
                        // The cursor already sits one line below the redrawn
                        // body (the last written line was "Selected: N of M\r\n"),
                        // so the apply panel's inline viewport can attach here
                        // directly. A single newline adds one separator line —
                        // NOT a full `drawn_lines` cursor-down, which left a
                        // ~14-line gap between "Selected" and the panel.
                        let _ = write!(stdout, "\r\n");
                        let _ = stdout.flush();
                        break;
                    }
                    KeyCode::Esc => {
                        restore_terminal();
                        return Err("Cancelled by user".to_string());
                    }
                    _ => {}
                }
            }
        }

        restore_terminal();

        // Build new plan with selections
        let mut new_plan = plan.clone();
        for (idx, item) in new_plan.plan.iter_mut().enumerate() {
            item.selected = selected_indices.contains(&idx);
        }
        new_plan.summary.selected = selected_indices.len();

        Ok(new_plan)
    }

    /// Non-interactive fallback used when stdin is not a TTY (piped input).
    ///
    /// Prints a one-shot numbered list and reads a single line of input to
    /// determine the selection. No raw mode, no redraw.
    fn interactive_selection_nontty(plan: &PlanOutput) -> Result<PlanOutput, String> {
        use std::io::{self, Write};

        let mut selected_indices: Vec<usize> = Vec::new();

        println!("Interactive Update Selection");
        println!("===========================\n");
        println!("Controls: enter numbers (comma-separated), or 'all' / 'none'\n");

        for (idx, item) in plan.plan.iter().enumerate() {
            let current = if item.current_version.is_empty() {
                "not installed"
            } else {
                &item.current_version
            };
            println!(
                "  {}. {} {} \u{2192} {} ({})",
                idx + 1,
                item.component_id,
                current,
                item.proposed_version,
                item.classification
            );
        }

        print!("\nEnter numbers to select (comma-separated, or 'all'/'none'): ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            match input.trim().to_lowercase().as_str() {
                "all" => selected_indices = (0..plan.plan.len()).collect(),
                "none" => selected_indices.clear(),
                _ => {
                    for part in input.split(',') {
                        if let Ok(num) = part.trim().parse::<usize>() {
                            if num > 0 && num <= plan.plan.len() {
                                selected_indices.push(num - 1);
                            }
                        }
                    }
                }
            }
        }

        let mut new_plan = plan.clone();
        // Dedup selected indices so `1,1,2` reports 2 selected (not 3) and each
        // item is flagged at most once. Sort first so dedup is stable.
        selected_indices.sort_unstable();
        selected_indices.dedup();
        for (idx, item) in new_plan.plan.iter_mut().enumerate() {
            item.selected = selected_indices.contains(&idx);
        }
        new_plan.summary.selected = selected_indices.len();

        Ok(new_plan)
    }

    pub fn run(
        scan_only: bool,
        all_safe: bool,
        include_experimental: bool,
        yes: bool,
        json: bool,
        components: Vec<String>,
    ) {
        // Determine output mode: JSON if explicitly requested or not a TTY
        let json_mode = json || !io::stdout().is_terminal();

        // Initialize logging for update operations.
        //
        // Use the interactive (stdout-free) logger even for human-readable
        // mode so that any `tracing::info!` fired before/during the inline
        // selection UI cannot corrupt stdout mid-render. Stderr keeps a
        // compact INFO+ stream for live diagnostics.
        let _log_guard = if json_mode {
            rusty_stack::logging::init_batch_logging("update")
        } else {
            rusty_stack::logging::init_interactive_logging("update")
        };
        tracing::info!(
            scan_only = scan_only,
            all_safe = all_safe,
            include_experimental = include_experimental,
            yes = yes,
            json_mode = json_mode,
            components = ?components,
            "Update command started"
        );

        // Build planner options from CLI args
        let options = rusty_stack::orchestrator::planner::PlannerOptions {
            scan_only,
            all_safe,
            include_experimental,
            target_components: components.clone(),
            json_output: json_mode,
        };

        // Phase 1: Scan. Keep animated progress off stdout so JSON stays valid.
        let scan = if json_mode {
            run_scan()
        } else {
            let activity = ActivityIndicator::start("Scanning hardware and installed components");
            let scan = run_scan();
            activity.finish("Scan complete");
            scan
        };

        if json_mode {
            // JSON output mode
            match build_plan(&scan, &options) {
                Ok(plan) => {
                    let can_apply_non_interactive = yes || all_safe;
                    if scan_only || plan.summary.selected == 0 || !can_apply_non_interactive {
                        let status = if scan_only {
                            "scan_only"
                        } else if plan.summary.selected == 0 {
                            "no_updates"
                        } else {
                            "confirmation_required"
                        };
                        let output = JsonOutput {
                            scan,
                            plan: Some(plan),
                            apply: None,
                            summary: JsonSummary {
                                status: status.to_string(),
                                scan_only,
                                error: None,
                            },
                        };
                        match serde_json::to_string(&output) {
                            Ok(payload) => println!("{payload}"),
                            Err(err) => {
                                eprintln!(
                                    "{{\"status\":\"error\",\"error\":\"failed to serialize update output: {}\"}}",
                                    err.to_string().replace('"', "\\\"")
                                );
                                process::exit(1);
                            }
                        }
                    } else {
                        // Apply the plan
                        let apply_result = apply_plan(&plan, json_mode);
                        let output = JsonOutput {
                            scan,
                            plan: Some(plan),
                            apply: Some(serde_json::to_value(&apply_result).unwrap_or_default()),
                            summary: JsonSummary {
                                status: if apply_result.has_failures() {
                                    "partial"
                                } else {
                                    "applied"
                                }
                                .to_string(),
                                scan_only: false,
                                error: None,
                            },
                        };
                        match serde_json::to_string(&output) {
                            Ok(payload) => println!("{payload}"),
                            Err(err) => {
                                eprintln!(
                                    "{{\"status\":\"error\",\"error\":\"failed to serialize update output: {}\"}}",
                                    err.to_string().replace('"', "\\\"")
                                );
                                process::exit(1);
                            }
                        }
                        if apply_result.has_failures() {
                            process::exit(1);
                        }
                    }
                }
                Err(error) => {
                    let output = JsonOutput {
                        scan,
                        plan: None,
                        apply: None,
                        summary: JsonSummary {
                            status: "error".to_string(),
                            scan_only,
                            error: Some(error.to_string()),
                        },
                    };
                    match serde_json::to_string(&output) {
                        Ok(payload) => println!("{payload}"),
                        Err(err) => {
                            eprintln!(
                                "{{\"status\":\"error\",\"error\":\"failed to serialize update error output: {}\"}}",
                                err.to_string().replace('"', "\\\"")
                            );
                        }
                    }
                    process::exit(1);
                }
            }
        } else {
            // Interactive/human-readable output mode
            println!("Rusty Stack Update v{VERSION}");
            println!("========================\n");

            print!("{}", format_scan_human(&scan));

            // Phase 2: Plan
            let plan_result = {
                let activity =
                    ActivityIndicator::start("Resolving manifest and building update plan");
                let result = build_plan(&scan, &options);
                activity.finish(if result.is_ok() {
                    "Update plan ready"
                } else {
                    "Update plan failed"
                });
                result
            };

            match plan_result {
                Ok(plan) => {
                    // The interactive selection screen renders its own copy of
                    // the plan; printing the human-readable plan block first
                    // would leave a stale duplicate above the inline UI. Only
                    // emit it for the non-interactive branches below.
                    let going_interactive = !(scan_only || plan.plan.is_empty() || yes || all_safe);
                    if !going_interactive {
                        print!("{}", format_plan_output_human(&plan));
                    }

                    if scan_only {
                        println!("\n(scan-only mode: no changes will be applied)");
                    } else if plan.plan.is_empty() {
                        println!("\nNo updates available.");
                        tracing::info!("No updates available");
                    } else if yes || all_safe {
                        // --all-safe or --yes: show countdown unless --yes skips it
                        if yes {
                            tracing::info!(
                                count = plan.summary.selected,
                                "Auto-applying (--yes flag)"
                            );
                            println!(
                                "\nApplying {} updates (auto-confirmed)...",
                                plan.summary.selected
                            );
                        } else {
                            // --all-safe: 10-second countdown with cancel
                            if !countdown_confirm(plan.summary.selected) {
                                tracing::info!("Update cancelled during countdown");
                                println!("\nCancelled.");
                                return;
                            }
                        }
                        let log_path = rusty_stack::logging::log_dir();
                        println!("  Logging to: {}", log_path.display());
                        println!();
                        let apply_result = apply_plan(&plan, json_mode);
                        print_apply_summary(&apply_result);
                        if apply_result.has_failures() {
                            process::exit(1);
                        }
                    } else {
                        // Interactive selection: allow user to toggle items with spacebar
                        // Always show interactive UI when there are available updates
                        tracing::info!(
                            "Starting interactive selection for {} items",
                            plan.plan.len()
                        );
                        match interactive_selection(&plan) {
                            Ok(selected_plan) => {
                                if selected_plan.summary.selected == 0 {
                                    println!("\nNo updates selected.");
                                    tracing::info!(
                                        "No updates selected after interactive selection"
                                    );
                                    return;
                                }
                                let apply_result = apply_plan(&selected_plan, json_mode);
                                print_apply_summary(&apply_result);
                                if apply_result.has_failures() {
                                    process::exit(1);
                                }
                            }
                            Err(e) => {
                                println!("\nCancelled: {}", e);
                                tracing::info!("Update cancelled: {}", e);
                            }
                        }
                    }
                }
                Err(error) => {
                    eprintln!("Error: {error}");
                    process::exit(1);
                }
            }
        }
    }

    /// Run the scan phase: detect hardware, installed components, versions.
    fn run_scan() -> rusty_stack::orchestrator::planner::ScanOutput {
        use rusty_stack::orchestrator::planner::{InstalledComponent, ScanOutput};
        use rusty_stack::platform::registry::{
            detect_all_installed, get_version, known_components,
        };

        let installed_ids = detect_all_installed();
        let mut installed = Vec::new();

        for id in &installed_ids {
            let version = get_version(id);
            installed.push(InstalledComponent {
                id: id.clone(),
                version,
                status: "installed".to_string(),
            });
        }

        // Also include known components that are NOT installed
        for comp in known_components() {
            if !installed_ids.contains(&comp.id) {
                installed.push(InstalledComponent {
                    id: comp.id.clone(),
                    version: "not installed".to_string(),
                    status: "not-installed".to_string(),
                });
            }
        }

        // Detect ROCm info
        let (rocm_version, gpu_architecture, rocm_channel) = detect_rocm_info();

        // Hardware detection is authoritative for ROCm. Some installations
        // have a working ROCm runtime but fail the broader component probe.
        if !rocm_version.is_empty() {
            if let Some(rocm) = installed.iter_mut().find(|comp| comp.id == "rocm") {
                rocm.version = rocm_version.clone();
                rocm.status = "installed".to_string();
            } else {
                installed.push(InstalledComponent {
                    id: "rocm".to_string(),
                    version: rocm_version.clone(),
                    status: "installed".to_string(),
                });
            }
        }

        ScanOutput {
            installed,
            manifest_source: "baseline".to_string(),
            rocm_channel,
            rocm_version,
            gpu_architecture,
        }
    }

    /// Detect ROCm version, GPU architecture, and channel.
    fn detect_rocm_info() -> (String, String, String) {
        #[cfg(unix)]
        {
            use rusty_stack::platform::linux::detect_gpu;

            let gpu_info = detect_gpu();
            let rocm_version = gpu_info.rocm_version.clone();

            let gpu_architecture = if gpu_info.architecture.is_empty() {
                "unknown".to_string()
            } else {
                gpu_info.architecture.clone()
            };

            // Determine channel from version
            let rocm_channel = if rocm_version.is_empty() {
                String::new()
            } else if rocm_version.starts_with("6.4") {
                "legacy".to_string()
            } else if rocm_version.starts_with("7.2") {
                // 7.2.0-7.2.3 -> stable; 7.2.4+ -> latest
                let patch: u32 = rocm_version
                    .split('.')
                    .nth(2)
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(0);
                if patch <= 3 {
                    "stable".to_string()
                } else {
                    "latest".to_string()
                }
            } else if rocm_version.starts_with("7.1") {
                "stable".to_string()
            } else {
                "latest".to_string()
            };

            (rocm_version, gpu_architecture, rocm_channel)
        }

        #[cfg(not(unix))]
        {
            (String::new(), "unknown".to_string(), String::new())
        }
    }

    /// Build the update plan from scan results and manifest.
    fn build_plan(
        scan: &rusty_stack::orchestrator::planner::ScanOutput,
        options: &rusty_stack::orchestrator::planner::PlannerOptions,
    ) -> Result<
        rusty_stack::orchestrator::planner::PlanOutput,
        rusty_stack::orchestrator::planner::PlannerError,
    > {
        use rusty_stack::core::manifest::{self, Manifest, ManifestFetcher};
        use rusty_stack::core::types::ExecutorKind;
        use rusty_stack::orchestrator::planner::{
            CompatibilityContext, PlanOutput, PlanSummary, PlannerItemOutput, UpdatePlanner,
        };
        use std::collections::HashSet;

        struct UpdateManifestFetcher;
        impl UpdateManifestFetcher {
            fn cached_manifest_path() -> std::path::PathBuf {
                std::env::var("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("."))
                    .join(".mlstack")
                    .join("cache")
                    .join("remote_manifest.json")
            }
        }
        impl ManifestFetcher for UpdateManifestFetcher {
            fn fetch_remote(&self) -> Option<Manifest> {
                let url = std::env::var("MLSTACK_REMOTE_MANIFEST_URL").ok()?;
                let body = ureq::Agent::new_with_defaults()
                    .get(&url)
                    .header("Accept", "application/json")
                    .header("User-Agent", "rusty-stack-update")
                    .call()
                    .ok()?
                    .into_body()
                    .read_to_string()
                    .ok()?;
                serde_json::from_str::<Manifest>(&body).ok()
            }

            fn load_cached(&self) -> Option<Manifest> {
                let path = Self::cached_manifest_path();
                let json = std::fs::read_to_string(path).ok()?;
                serde_json::from_str::<Manifest>(&json).ok()
            }
        }

        // Resolve update manifest with trust-checked fallback chain:
        // fresh remote (when configured) -> cached -> bundled baseline.
        let manifest = manifest::resolve_manifest(&UpdateManifestFetcher).manifest;

        // Build compatibility context from scan results
        let mut context = CompatibilityContext::new();
        context.rocm_version = scan.rocm_version.clone();
        context.rocm_channel = scan.rocm_channel.clone();
        context.gpu_architecture = scan.gpu_architecture.clone();
        context.available_executors = HashSet::from([
            ExecutorKind::LegacyScript,
            ExecutorKind::Rust,
            ExecutorKind::ExternalPackageManager,
        ]);

        for comp in &scan.installed {
            if comp.status == "installed" {
                context.installed_components.insert(comp.id.clone());
                context
                    .installed_versions
                    .insert(comp.id.clone(), comp.version.clone());
            }
        }

        context.runtime_version = VERSION.to_string();

        // Build the plan
        let planner = UpdatePlanner::new();
        let items = planner.build_plan(&manifest, &context, options)?;

        let plan_output: Vec<PlannerItemOutput> =
            items.iter().map(PlannerItemOutput::from).collect();
        let summary = PlanSummary::from_items(&items);

        Ok(PlanOutput {
            plan: plan_output,
            summary,
        })
    }

    /// Format the scan as human-readable text.
    fn format_scan_human(scan: &rusty_stack::orchestrator::planner::ScanOutput) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "ROCm: {} ({})\n",
            scan.rocm_version, scan.rocm_channel
        ));
        output.push_str(&format!("GPU:  {}\n\n", scan.gpu_architecture));

        output.push_str("Installed components:\n");
        for comp in &scan.installed {
            if comp.status == "installed" {
                output.push_str(&format!("  {} v{}\n", comp.id, comp.version));
            }
        }

        output
    }

    /// Format the plan output as human-readable text.
    fn format_plan_output_human(plan: &rusty_stack::orchestrator::planner::PlanOutput) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "\nUpdate Plan ({} items: {} safe, {} guarded, {} candidate, {} experimental, {} blocked)\n\n",
            plan.summary.total,
            plan.summary.safe,
            plan.summary.guarded,
            plan.summary.candidate,
            plan.summary.experimental,
            plan.summary.blocked,
        ));

        for item in &plan.plan {
            let sel = if item.selected { "✓" } else { " " };
            let vis = if item.visible { "" } else { " [hidden]" };
            let version_str = if item.current_version.is_empty() {
                format!("(new) → {}", item.proposed_version)
            } else if item.current_version == item.proposed_version {
                format!("{} (reinstall)", item.current_version)
            } else {
                format!("{} → {}", item.current_version, item.proposed_version)
            };
            output.push_str(&format!(
                "  {} {:15} {}  ({}){}\n",
                sel, item.component_id, version_str, item.classification, vis,
            ));
        }

        output.push_str(&format!(
            "\nSelected: {} of {} items\n",
            plan.summary.selected, plan.summary.total
        ));

        output
    }

    /// Show a 10-second countdown confirmation before applying updates.
    ///
    /// Press Enter to apply immediately, 'n' or Ctrl+C to cancel.
    /// Returns `true` if confirmed, `false` if cancelled.
    fn countdown_confirm(count: usize) -> bool {
        use std::io::stdin;
        use std::thread;
        use std::time::{Duration, Instant};

        println!("\n  ┌───────────────────────────────────────────────────┐");
        println!(
            "  │ {} update{} will be applied in 10 seconds.        │",
            count,
            if count > 1 { "s" } else { "" }
        );
        println!("  │ Press Enter to apply now, 'n' to cancel.         │");
        println!("  └───────────────────────────────────────────────────┘");

        // Set stdin to non-blocking raw mode for the countdown
        #[cfg(unix)]
        {
            use std::io::Read;
            let mut termios: libc::termios = unsafe { std::mem::zeroed() };
            if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut termios) } == 0 {
                let original = termios;
                // Set to raw mode (no echo, no canonical, non-blocking reads)
                termios.c_lflag &= !(libc::ICANON | libc::ECHO);
                termios.c_cc[libc::VMIN] = 0;
                termios.c_cc[libc::VTIME] = 0;
                let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) };

                // Ensure terminal is restored even on panic/ctrl+c
                let restore_termios = || {
                    let _ =
                        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original) };
                };

                let mut confirmed = false;
                let mut cancelled = false;

                for remaining in (1..=10).rev() {
                    eprint!("\r  ⏳ Applying in {:2}s... ", remaining);
                    let _ = std::io::stderr().flush();

                    // Poll stdin for one full second so the countdown is truly 10 seconds.
                    let tick_start = Instant::now();
                    while tick_start.elapsed() < Duration::from_secs(1) {
                        thread::sleep(Duration::from_millis(100));
                        let mut buf = [0u8; 1];
                        if let Ok(1) = stdin().read(&mut buf) {
                            match buf[0] {
                                b'\n' | b'\r' => {
                                    confirmed = true;
                                    break;
                                }
                                b'n' | b'N' | b'q' | b'Q' => {
                                    cancelled = true;
                                    break;
                                }
                                3 => {
                                    // Ctrl+C
                                    cancelled = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    if confirmed || cancelled {
                        break;
                    }
                }

                // Restore original terminal settings (always executed)
                restore_termios();

                eprint!("\r{}\r", " ".repeat(40));

                if cancelled {
                    return false;
                }
                if confirmed {
                    println!("  ✓ Confirmed.");
                    return true;
                }
                // Countdown expired — auto-confirm
                println!("  ✓ Countdown elapsed — auto-applying.");
                return true;
            }
        }

        // Fallback for non-unix: simple prompt
        print!("  Apply {} updates? [Y/n] ", count);
        let _ = std::io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let answer = input.trim().to_lowercase();
            answer != "n" && answer != "no"
        } else {
            false
        }
    }
    /// Apply the selected items using the native installer (direct function calls).
    ///
    /// Uses `DirectInstallerExecutor` which calls `installer::run_installation()`
    /// directly - no subprocess spawning.
    fn apply_plan(
        plan: &rusty_stack::orchestrator::planner::PlanOutput,
        json_mode: bool,
    ) -> rusty_stack::orchestrator::apply::ApplySummary {
        use rusty_stack::orchestrator::apply::{ApplyEngine, ApplyExecutor, ApplyOptions};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // Sudo-password helpers live in the top-level `sudo_creds` module so the
        // uninstall/reinstall paths can share them.

        /// Direct installer executor - calls Rust installer functions in-process.
        struct DirectInstallerExecutor {
            cancelled: Arc<AtomicBool>,
            json_mode: bool,
            /// Inline animated apply panel. `None` (non-TTY / JSON / no-tui) keeps
            /// the legacy per-line stdout/stderr output unchanged.
            #[cfg(feature = "tui")]
            panel: std::cell::RefCell<Option<apply_ui::ApplyPanel>>,
        }

        impl DirectInstallerExecutor {
            fn new(cancelled: Arc<AtomicBool>, json_mode: bool) -> Self {
                Self {
                    cancelled,
                    json_mode,
                    #[cfg(feature = "tui")]
                    panel: std::cell::RefCell::new(None),
                }
            }

            /// Attach the inline apply panel (TTY, non-JSON only) and render its
            /// initial pending-state frame.
            #[cfg(feature = "tui")]
            fn set_panel(&self, panel: apply_ui::ApplyPanel) {
                let mut guard = self.panel.borrow_mut();
                *guard = Some(panel);
                if let Some(p) = guard.as_mut() {
                    p.draw();
                }
            }

            /// Whether the animated panel owns the terminal right now.
            #[cfg(feature = "tui")]
            fn panel_active(&self) -> bool {
                self.panel.borrow().is_some()
            }

            #[cfg(not(feature = "tui"))]
            fn panel_active(&self) -> bool {
                false
            }

            /// Fold one installer event into the animated panel and redraw it.
            /// No-op when no panel is attached (non-TTY / JSON / no-tui).
            #[cfg(feature = "tui")]
            fn fold_panel_event(
                &self,
                ev: &rusty_stack::installer::InstallerEvent,
                running_id: &str,
            ) {
                if let Some(p) = self.panel.borrow_mut().as_mut() {
                    p.state_mut().apply_event(ev, running_id);
                    p.draw();
                }
            }

            /// Advance the panel's animation frame and redraw (called on each
            /// `recv_timeout` wake so the spinner/border animate even while the
            /// installer is silent). No-op when no panel is attached.
            #[cfg(feature = "tui")]
            fn tick_panel(&self) {
                if let Some(p) = self.panel.borrow_mut().as_mut() {
                    p.tick();
                }
            }
            fn component_for_id(id: &str) -> Option<rusty_stack::state::Component> {
                // The bundled manifest/registry still emit the legacy `flash-attn`
                // id, while default_components() now exposes the split
                // `flash-attn-triton` / `flash-attn-ck` components. The native
                // installer treats a generic `flash-attn` as the recommended
                // Triton backend (installer.rs flash-attn dispatch), so normalize
                // the legacy id here — otherwise an available `flash-attn` update
                // fails immediately with "Unknown component ID" before reaching
                // the installer that supports the alias.
                let normalized = match id {
                    "flash-attn" => "flash-attn-triton",
                    other => other,
                };
                rusty_stack::state::default_components()
                    .into_iter()
                    .find(|c| c.id == normalized)
                    .map(|mut c| {
                        c.selected = true;
                        c
                    })
            }
        }

        impl ApplyExecutor for DirectInstallerExecutor {
            fn apply_component(
                &self,
                component_id: &str,
                proposed_version: &str,
            ) -> Result<(), String> {
                if self.cancelled.load(Ordering::Relaxed) {
                    return Err(format!("{} cancelled by user", component_id));
                }
                tracing::info!(
                    component = component_id,
                    proposed_version,
                    "Starting component installation"
                );
                let Some(component) = Self::component_for_id(component_id) else {
                    return Err(format!("Unknown component ID: {}", component_id));
                };

                // Resolve sudo password for components that need it
                #[cfg(unix)]
                let sudo_password = if component.needs_sudo {
                    if unsafe { libc::geteuid() } == 0 {
                        None
                    } else if let Ok(pw) = std::env::var("MLSTACK_SUDO_PASSWORD") {
                        if pw.is_empty() {
                            None
                        } else {
                            Some(pw)
                        }
                    } else if sudo_creds::can_sudo_non_interactive() {
                        tracing::info!(
                            component = component_id,
                            "Using non-interactive sudo path (-n)"
                        );
                        None
                    } else {
                        eprint!("    sudo password for {}: ", component.name);
                        let _ = std::io::stderr().flush();
                        let password = sudo_creds::read_password_from_tty();
                        if let Some(ref _pw) = password {
                            tracing::info!(component = component_id, "Sudo password provided");
                        } else {
                            tracing::warn!(component = component_id, "No sudo password provided");
                        }
                        password
                    }
                } else {
                    None
                };
                #[cfg(not(unix))]
                let sudo_password: Option<String> = None;

                let (tx, rx) = std::sync::mpsc::channel();
                let (_, input_rx) = std::sync::mpsc::channel();
                let scripts_dir = rusty_stack::detect_scripts_dir();
                let config = rusty_stack::config::InstallerConfig::load_or_default(&scripts_dir)
                    .unwrap_or_else(|_| {
                        rusty_stack::config::InstallerConfig::default_with_paths(
                            &scripts_dir,
                            format!(
                                "{}/logs",
                                std::env::var("HOME").unwrap_or_else(|_| ".".into())
                            ),
                            rusty_stack::config::config_file_path().unwrap_or_else(|_| {
                                std::path::PathBuf::from("/tmp/mlstack/config/config.json")
                            }),
                        )
                    });

                // Propagate config flags to env vars that run_installation reads
                if config.force_reinstall {
                    std::env::set_var("MLSTACK_FORCE_REINSTALL", "1");
                    std::env::set_var("FORCE", "true");
                    std::env::set_var("PYTORCH_REINSTALL", "true");
                }
                if config.install_method != "auto" {
                    std::env::set_var("MLSTACK_INSTALL_METHOD", &config.install_method);
                }
                let component_name = component.name.clone();
                let cid = component_id.to_string();
                let target_version = proposed_version.to_string();
                let handle = std::thread::spawn(move || {
                    rusty_stack::installer::run_installation_with_version(
                        vec![component],
                        config,
                        sudo_password,
                        tx,
                        input_rx,
                        Some(target_version),
                    );
                });
                let mut success = true;
                let mut error_msg = String::new();
                let spinner: &[char] = &[
                    '\u{280b}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283c}', '\u{2834}',
                    '\u{2826}', '\u{2827}', '\u{2807}', '\u{280f}',
                ];
                let mut si = 0usize;
                let panel_active = self.panel_active();
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(ev) => {
                            // Drive the animated panel (no-op when none attached).
                            #[cfg(feature = "tui")]
                            self.fold_panel_event(&ev, &cid);
                            match ev {
                                rusty_stack::installer::InstallerEvent::Log(line, _) => {
                                    // Raw pip/git chatter is suppressed on the TTY
                                    // (the panel shows the milestone; full log → file).
                                    if !panel_active {
                                        if self.json_mode {
                                            eprintln!("    | {}", line);
                                        } else {
                                            println!("    | {}", line);
                                        }
                                    }
                                    tracing::info!(component = %cid, log = %line);
                                }
                                rusty_stack::installer::InstallerEvent::Progress {
                                    progress,
                                    message,
                                    ..
                                } => {
                                    if !panel_active {
                                        let pct = (progress * 100.0) as u8;
                                        let s = spinner[si % spinner.len()];
                                        si += 1;
                                        eprint!(
                                            "\r    {} {} [{:>3}%] {}    ",
                                            s, cid, pct, message
                                        );
                                        let _ = std::io::stderr().flush();
                                    }
                                }
                                rusty_stack::installer::InstallerEvent::ComponentStart {
                                    name,
                                    ..
                                } => {
                                    if !panel_active {
                                        eprint!("\r    ");
                                        if self.json_mode {
                                            eprintln!("    > Installing {}...", name);
                                        } else {
                                            println!("    > Installing {}...", name);
                                        }
                                    }
                                    tracing::info!(component = %cid, name = %name, "Component started");
                                }
                                rusty_stack::installer::InstallerEvent::ComponentComplete {
                                    success: s,
                                    message,
                                    ..
                                } => {
                                    if !panel_active {
                                        eprint!("\r{}\r", " ".repeat(80));
                                        if s {
                                            if self.json_mode {
                                                eprintln!(
                                                    "    ok {} - {}",
                                                    component_name, message
                                                );
                                            } else {
                                                println!("    ok {} - {}", component_name, message);
                                            }
                                        } else {
                                            if self.json_mode {
                                                eprintln!(
                                                    "    FAIL {} - {}",
                                                    component_name, message
                                                );
                                            } else {
                                                println!(
                                                    "    FAIL {} - {}",
                                                    component_name, message
                                                );
                                            }
                                        }
                                    }
                                    if s {
                                        tracing::info!(component = %cid, "Completed successfully");
                                    } else {
                                        tracing::error!(component = %cid, error = %message, "Failed");
                                        success = false;
                                        error_msg = message;
                                    }
                                }
                                rusty_stack::installer::InstallerEvent::VerificationReport {
                                    lines,
                                    ..
                                } => {
                                    if !panel_active {
                                        for line in &lines {
                                            if self.json_mode {
                                                eprintln!("    | {}", line);
                                            } else {
                                                println!("    | {}", line);
                                            }
                                        }
                                    }
                                }
                                rusty_stack::installer::InstallerEvent::Finished { success: s } => {
                                    if !s {
                                        success = false;
                                        if error_msg.is_empty() {
                                            error_msg = "Finished with errors".into();
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if self.cancelled.load(Ordering::Relaxed) {
                                return Err(format!("{} cancelled", cid));
                            }
                            // Idle wake: advance the spinner/border animation so
                            // the panel stays live while the installer is silent.
                            #[cfg(feature = "tui")]
                            if panel_active {
                                self.tick_panel();
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            break;
                        }
                    }
                }
                // Treat join errors as failures
                match handle.join() {
                    Ok(_) => {}
                    Err(_) => {
                        success = false;
                        error_msg = "Installer thread panicked or disconnected".into();
                    }
                }
                eprint!("\r{}\r", " ".repeat(80));
                if success {
                    tracing::info!(component=%cid,"Succeeded");
                    Ok(())
                } else {
                    tracing::error!(component=%cid,error=%error_msg,"Failed");
                    Err(error_msg)
                }
            }
        }

        // Convert plan output back to PlannerItems for the apply engine.
        // We reconstruct minimal PlannerItems from the plan output data.
        let items: Vec<rusty_stack::orchestrator::planner::PlannerItem> = plan
            .plan
            .iter()
            .map(|item| {
                use rusty_stack::core::plan::{PlanItem, PlanItemInput};
                use rusty_stack::core::types::ValidationTier;
                use rusty_stack::orchestrator::planner::UpdateClassification;

                let classification = match item.classification.as_str() {
                    "safe" => UpdateClassification::Safe,
                    "guarded" => UpdateClassification::Guarded,
                    "blocked" => UpdateClassification::Blocked,
                    "candidate" => UpdateClassification::Candidate,
                    "experimental" => UpdateClassification::Experimental,
                    _ => UpdateClassification::Guarded,
                };

                let tier = match item.risk_tier.as_str() {
                    "validated" => ValidationTier::Validated,
                    "candidate" => ValidationTier::Candidate,
                    "experimental" => ValidationTier::Experimental,
                    "blocked" => ValidationTier::Blocked,
                    _ => ValidationTier::Candidate,
                };

                rusty_stack::orchestrator::planner::PlannerItem {
                    plan_item: PlanItem::new(PlanItemInput {
                        component_id: item.component_id.clone(),
                        current_version: item.current_version.clone(),
                        proposed_version: item.proposed_version.clone(),
                        validation_tier: tier,
                        selected: item.selected,
                        rationale: item.rationale.clone(),
                        dependencies: item.dependencies.clone(),
                        isolation_safe: matches!(classification, UpdateClassification::Safe),
                    }),
                    classification,
                    visible: item.visible,
                    selected: item.selected,
                    classification_reason: item.rationale.clone(),
                    requires_hardware_check: false,
                    min_rocm_version: String::new(),
                }
            })
            .collect();

        let cancelled = Arc::new(AtomicBool::new(false));
        let executor = DirectInstallerExecutor::new(cancelled.clone(), json_mode);
        // On a real TTY (non-JSON), attach the inline animated panel. Non-TTY
        // (piped) and JSON callers keep the legacy per-line output unchanged.
        #[cfg(feature = "tui")]
        if !json_mode {
            let rows: Vec<(&str, &str, &str, &str)> = plan
                .plan
                .iter()
                .filter(|i| i.selected)
                .map(|i| {
                    (
                        i.component_id.as_str(),
                        i.component_id.as_str(),
                        i.current_version.as_str(),
                        i.proposed_version.as_str(),
                    )
                })
                .collect();
            if let Some(panel) = apply_ui::ApplyPanel::try_new("Rusty Stack Update", rows) {
                executor.set_panel(panel);
            }
        }
        let engine = ApplyEngine::new(executor);
        // The real apply opts INTO the post-install honesty guard (verify the
        // installed version actually advanced to the target). Tests default this
        // off (mock executors don't install).
        let apply_opts = ApplyOptions {
            verify_post_install_version: true,
            ..ApplyOptions::default()
        };
        engine.apply(&items, &apply_opts)
    }

    /// Print a human-readable summary of the apply results.
    fn print_apply_summary(summary: &rusty_stack::orchestrator::apply::ApplySummary) {
        println!("\nApply Results:");
        println!("---------------");

        if !summary.success.is_empty() {
            println!("\n  Succeeded ({}):", summary.success.len());
            for item in &summary.success {
                println!(
                    "    ✓ {} {} → {}",
                    item.component_id, item.current_version, item.proposed_version
                );
            }
        }

        if !summary.failed.is_empty() {
            println!("\n  Failed ({}):", summary.failed.len());
            for item in &summary.failed {
                println!("    ✗ {} — {}", item.component_id, item.error_message);
            }
        }

        if !summary.blocked.is_empty() {
            println!("\n  Blocked by dependency ({}):", summary.blocked.len());
            for item in &summary.blocked {
                println!("    ! {} — {}", item.component_id, item.error_message);
            }
        }

        if !summary.held_back.is_empty() {
            println!("\n  Held back ({}):", summary.held_back.len());
            for item in &summary.held_back {
                println!("    - {} (not selected)", item.component_id);
            }
        }

        let total = summary.total();
        let succeeded = summary.success.len();
        if summary.has_failures() {
            println!(
                "\n  {}/{} components updated successfully.",
                succeeded, total
            );
        } else {
            println!("\n  All {} components updated successfully.", succeeded);
        }
    }

    // JSON output types for update
    #[derive(Debug, Serialize)]
    struct JsonOutput {
        scan: rusty_stack::orchestrator::planner::ScanOutput,
        plan: Option<rusty_stack::orchestrator::planner::PlanOutput>,
        apply: Option<serde_json::Value>,
        summary: JsonSummary,
    }

    #[derive(Debug, Serialize)]
    struct JsonSummary {
        status: String,
        scan_only: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    }
}

#[cfg(test)]
mod update_selection_tests {
    use super::update_impl;
    use rusty_stack::orchestrator::planner::{PlanOutput, PlanSummary, PlannerItemOutput};

    fn sample_plan() -> PlanOutput {
        let items = vec![
            ("pytorch", "2.12.1+rocm7.2", "2.13.0", "guarded"),
            ("onnx", "1.23.2", "1.27.1", "guarded"),
            ("migraphx", "", "2.12.0", "candidate"),
            ("megatron", "core_v0.15.0rc7", "25.1", "guarded"),
            ("aiter", "0.0.0", "0.1.0", "guarded"),
            ("comfyui", "v0.20.1", "latest", "guarded"),
            ("vllm-studio", "v2.1.0", "latest", "guarded"),
            ("textgen", "v4.7.3", "latest", "guarded"),
            ("permanent-env", "installed", "1.0.0", "guarded"),
        ];
        let plan: Vec<PlannerItemOutput> = items
            .iter()
            .map(|(id, cur, prop, class)| PlannerItemOutput {
                component_id: id.to_string(),
                current_version: cur.to_string(),
                proposed_version: prop.to_string(),
                classification: class.to_string(),
                risk_tier: "medium".to_string(),
                selected: false,
                visible: true,
                rationale: String::new(),
                dependencies: vec![],
                exclusive_group: String::new(),
            })
            .collect();
        PlanOutput {
            plan,
            summary: PlanSummary {
                total: 9,
                safe: 0,
                guarded: 8,
                candidate: 1,
                experimental: 0,
                blocked: 0,
                selected: 0,
            },
        }
    }

    #[test]
    fn body_has_header_and_controls() {
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[], 0);
        assert!(body.starts_with("Interactive Update Selection\r\n"));
        assert!(body.contains("===========================\r\n\r\n"));
        assert!(body.contains("Controls:"));
        assert!(body.contains("\u{2191}\u{2193}".chars().next().unwrap()));
    }

    #[test]
    fn body_renders_all_items() {
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[], 0);
        for (id, cur, prop, class) in [
            ("pytorch", "2.12.1+rocm7.2", "2.13.0", "guarded"),
            ("migraphx", "not installed", "2.12.0", "candidate"),
            ("permanent-env", "installed", "1.0.0", "guarded"),
        ] {
            assert!(body.contains(id), "missing {id}");
            assert!(body.contains(cur), "missing current '{cur}' for {id}");
            assert!(body.contains(prop), "missing proposed '{prop}' for {id}");
            assert!(
                body.contains(&format!("({})", class)),
                "missing classification '{class}' for {id}"
            );
        }
    }

    #[test]
    fn empty_current_version_shows_not_installed() {
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[], 0);
        assert!(body.contains("not installed"));
        // Ensure the literal empty string isn't what gets rendered.
        let migraphx_line = body
            .lines()
            .find(|l| l.contains("migraphx"))
            .expect("migraphx line");
        assert!(migraphx_line.contains("not installed"));
        assert!(migraphx_line.contains("2.12.0"));
        assert!(migraphx_line.contains("(candidate)"));
    }

    #[test]
    fn selected_and_cursor_markers_are_correct() {
        let plan = sample_plan();
        // Cursor on row 2 (index 1), item 3 (index 2) selected.
        let body = update_impl::render_selection_body(&plan, &[2], 1);
        let lines: Vec<&str> = body.lines().collect();
        // Find the row lines (they start with a cursor marker: '>' or ' ').
        let rows: Vec<&&str> = lines
            .iter()
            .filter(|l| l.contains("[ ]") || l.contains("[x]"))
            .collect();
        assert_eq!(rows.len(), 9);
        // Row index 1 should have the cursor marker.
        assert!(rows[1].starts_with(">"), "cursor marker wrong: {}", rows[1]);
        // Row index 2 should be checked.
        assert!(
            rows[2].contains("[x]"),
            "selected marker wrong: {}",
            rows[2]
        );
        // Other rows should be unchecked and not cursor.
        for (i, r) in rows.iter().enumerate() {
            if i != 2 {
                assert!(r.contains("[ ]"), "unexpected checked row {i}: {}", r);
            }
            if i != 1 {
                assert!(r.starts_with(" "), "unexpected cursor row {i}: {}", r);
            }
        }
    }

    #[test]
    fn selected_count_reflects_input() {
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[0, 1, 2], 0);
        assert!(body.contains("Selected: 3 of 9"));
        let body_none = update_impl::render_selection_body(&plan, &[], 0);
        assert!(body_none.contains("Selected: 0 of 9"));
    }

    #[test]
    fn columns_are_aligned_despite_uneven_versions() {
        // The longest current-version string sets the column width; shorter
        // values must be right-padded so the arrow column starts at the same
        // column on every row.
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[], 0);
        let arrow_cols: Vec<usize> = body
            .lines()
            .filter(|l| l.contains("\u{2192}".chars().next().unwrap()))
            .map(|l| l.find("\u{2192}".chars().next().unwrap()).unwrap())
            .collect();
        assert!(arrow_cols.len() >= 2, "expected multiple item rows");
        let first = arrow_cols[0];
        for col in &arrow_cols {
            assert_eq!(*col, first, "misaligned arrow column: {} vs {}", col, first);
        }
    }

    #[test]
    fn body_uses_crlf_line_endings() {
        // Raw mode only moves the cursor down with LF; CR is required to reset
        // the column. Every line must end with \r\n, never bare \n.
        let plan = sample_plan();
        let body = update_impl::render_selection_body(&plan, &[], 0);
        assert!(!body.contains("\n") || body.contains("\r\n"));
        // No bare-LF sequences: every \n must be preceded by \r.
        let mut prev = ' ';
        for (i, ch) in body.char_indices() {
            if ch == '\n' && prev != '\r' {
                panic!("bare LF at byte {i} in body");
            }
            prev = ch;
        }
    }
}

// ===========================================================================
// Upgrade subcommand implementation
// ===========================================================================

mod upgrade_impl {
    use super::*;
    use rusty_stack::orchestrator::upgrade::{
        self, BinaryDownloader, DownloadResult, ReleaseInfo, ReleaseProvider, SmokeTester,
        UpgradeError, UpgradeOptions, UpgradeResult, UpgradeStatus, UserInteractor, VersionInfo,
    };

    pub fn run(
        yes: bool,
        binary_path: Option<PathBuf>,
        backup_dir: Option<PathBuf>,
        cached_manifest: Option<PathBuf>,
        dry_run: bool,
    ) {
        let schema_version = rusty_stack::core::manifest::CURRENT_SCHEMA_VERSION;

        let current_version = VersionInfo {
            version: VERSION.to_string(),
            schema_version,
        };

        let options = UpgradeOptions {
            non_interactive: yes,
            binary_path,
            backup_dir,
            cached_manifest_path: cached_manifest,
        };

        // Dry run mode
        if dry_run {
            match RealReleaseProvider.fetch_latest_release() {
                Ok(release) => {
                    let upgrade_available = is_release_newer(VERSION, &release.version);
                    if yes {
                        let output = serde_json::json!({
                            "current_version": VERSION,
                            "schema_version": schema_version,
                            "latest_version": release.version,
                            "latest_schema_version": release.schema_version,
                            "upgrade_available": upgrade_available,
                            "download_url": release.download_url,
                        });
                        match serde_json::to_string(&output) {
                            Ok(payload) => println!("{payload}"),
                            Err(err) => {
                                eprintln!(
                                    "{{\"status\":\"error\",\"error\":\"failed to serialize dry-run output: {}\"}}",
                                    err.to_string().replace('"', "\\\"")
                                );
                                process::exit(1);
                            }
                        }
                    } else {
                        println!("Current version: v{VERSION}");
                        println!("Schema version: {schema_version}");
                        println!("Latest release: v{}", release.version);
                        println!(
                            "Upgrade available: {}",
                            if upgrade_available { "yes" } else { "no" }
                        );
                    }
                }
                Err(error) => {
                    if yes {
                        let output = serde_json::json!({
                            "current_version": VERSION,
                            "schema_version": schema_version,
                            "upgrade_available": false,
                            "error": error.to_string(),
                        });
                        match serde_json::to_string(&output) {
                            Ok(payload) => eprintln!("{payload}"),
                            Err(err) => {
                                eprintln!(
                                    "{{\"status\":\"error\",\"error\":\"failed to serialize dry-run error output: {}\"}}",
                                    err.to_string().replace('"', "\\\"")
                                );
                            }
                        }
                        process::exit(1);
                    } else {
                        println!("Current version: v{VERSION}");
                        println!("Schema version: {schema_version}");
                        eprintln!("Unable to check latest release: {error}");
                        process::exit(1);
                    }
                }
            }
            return;
        }

        let result = upgrade::run_upgrade(
            &current_version,
            &options,
            &RealReleaseProvider,
            &RealDownloader,
            &RealSmokeTester,
            &RealInteractor,
        );

        match result {
            Ok(upgrade_result) => {
                if yes {
                    println!("{}", format_json_output(&upgrade_result));
                } else {
                    print_interactive_result(&upgrade_result);
                }
                process::exit(0);
            }
            Err(error) => {
                if matches!(error, UpgradeError::NoUpgradeAvailable { .. }) {
                    if yes {
                        eprintln!("{}", format_error_json(&error));
                    } else {
                        print_interactive_error(&error);
                    }
                    process::exit(0);
                }
                if yes {
                    eprintln!("{}", format_error_json(&error));
                } else {
                    print_interactive_error(&error);
                }
                process::exit(1);
            }
        }
    }

    fn is_release_newer(current: &str, latest: &str) -> bool {
        let current = current.trim().trim_start_matches('v');
        let latest = latest.trim().trim_start_matches('v');
        match (
            semver::Version::parse(current),
            semver::Version::parse(latest),
        ) {
            (Ok(cur), Ok(lat)) => lat > cur,
            _ => latest != current,
        }
    }

    // Real implementations for CLI usage

    const RELEASES_API_URL: &str =
        "https://api.github.com/repos/scooter-lacroix/Stan-s-ML-Stack/releases/latest";
    const UPGRADE_USER_AGENT: &str = "rusty-stack-upgrade";

    /// Real release provider that fetches from GitHub releases API.
    struct RealReleaseProvider;

    impl ReleaseProvider for RealReleaseProvider {
        fn fetch_latest_release(&self) -> std::result::Result<ReleaseInfo, UpgradeError> {
            let release: serde_json::Value = ureq::Agent::new_with_defaults()
                .get(RELEASES_API_URL)
                .header("Accept", "application/vnd.github+json")
                .header("User-Agent", UPGRADE_USER_AGENT)
                .call()
                .map_err(|e| UpgradeError::DownloadFailed {
                    reason: format!("failed to fetch latest release: {e}"),
                })?
                .into_body()
                .read_json()
                .map_err(|e| UpgradeError::DownloadFailed {
                    reason: format!("failed to parse latest release JSON: {e}"),
                })?;

            let tag = release
                .get("tag_name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: "latest release is missing tag_name".to_string(),
                })?;
            let version = tag.strip_prefix('v').unwrap_or(tag).to_string();

            let assets = release
                .get("assets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: "latest release is missing assets".to_string(),
                })?;

            let wanted_suffix = target_release_asset_suffix();
            let selected_asset = assets
                .iter()
                .find(|asset| {
                    asset
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|name| name.ends_with(wanted_suffix))
                        .unwrap_or(false)
                })
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: format!(
                        "no release asset found for this platform (expected suffix: {wanted_suffix})"
                    ),
                })?;

            let asset_name = selected_asset
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: "selected asset is missing name".to_string(),
                })?;
            let download_url = selected_asset
                .get("browser_download_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: "selected asset is missing browser_download_url".to_string(),
                })?
                .to_string();

            let checksum = selected_asset
                .get("digest")
                .and_then(|v| v.as_str())
                .and_then(|digest| digest.strip_prefix("sha256:"))
                .map(|s| s.to_string())
                .or_else(|| checksum_from_sums_asset(assets, asset_name))
                .ok_or_else(|| UpgradeError::DownloadFailed {
                    reason: format!("no SHA256 checksum found for release asset '{asset_name}'"),
                })?;

            Ok(ReleaseInfo {
                version,
                download_url,
                checksum,
                min_runtime_version: "0.0.0".to_string(),
                schema_version: rusty_stack::core::manifest::CURRENT_SCHEMA_VERSION,
            })
        }
    }

    /// Real binary downloader using HTTP.
    struct RealDownloader;

    impl BinaryDownloader for RealDownloader {
        fn download(
            &self,
            release: &ReleaseInfo,
        ) -> std::result::Result<DownloadResult, UpgradeError> {
            let url = release.download_url.as_str();
            let archive_bytes = ureq::Agent::new_with_defaults()
                .get(url)
                .header("User-Agent", UPGRADE_USER_AGENT)
                .call()
                .map_err(|e| UpgradeError::DownloadFailed {
                    reason: format!("failed to download release asset: {e}"),
                })?
                .into_body()
                .read_to_vec()
                .map_err(|e| UpgradeError::DownloadFailed {
                    reason: format!("failed to read downloaded asset bytes: {e}"),
                })?;

            if url.ends_with(".tar.gz") || url.ends_with(".tgz") {
                upgrade::verify_integrity(&archive_bytes, &release.checksum)?;
                return Ok(DownloadResult {
                    binary_data: extract_binary_from_tar_gz(&archive_bytes)?,
                    integrity_verified: true,
                });
            }

            if url.ends_with(".zip") {
                upgrade::verify_integrity(&archive_bytes, &release.checksum)?;
                return Ok(DownloadResult {
                    binary_data: extract_binary_from_zip(&archive_bytes)?,
                    integrity_verified: true,
                });
            }

            Ok(DownloadResult {
                binary_data: archive_bytes,
                integrity_verified: false,
            })
        }
    }

    fn target_release_asset_suffix() -> &'static str {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            "linux-x86_64.tar.gz"
        }
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            "windows-x86_64.zip"
        }
        #[cfg(not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "x86_64")
        )))]
        {
            "unsupported-platform"
        }
    }

    fn expected_binary_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "rusty-stack.exe"
        }
        #[cfg(not(target_os = "windows"))]
        {
            "rusty-stack"
        }
    }

    fn checksum_from_sums_asset(
        assets: &[serde_json::Value],
        target_asset_name: &str,
    ) -> Option<String> {
        let sums_url = assets.iter().find_map(|asset| {
            let name = asset.get("name")?.as_str()?;
            if name == "SHA256SUMS" {
                asset
                    .get("browser_download_url")?
                    .as_str()
                    .map(|s| s.to_string())
            } else {
                None
            }
        })?;

        let sums_text = ureq::Agent::new_with_defaults()
            .get(&sums_url)
            .header("User-Agent", UPGRADE_USER_AGENT)
            .call()
            .ok()?
            .into_body()
            .read_to_string()
            .ok()?;

        for line in sums_text.lines() {
            let mut parts = line.split_whitespace();
            let checksum = parts.next()?;
            let filename = parts.next()?.trim_start_matches('*');
            if filename == target_asset_name {
                return Some(checksum.to_string());
            }
        }
        None
    }

    fn find_file_recursively(root: &Path, filename: &str) -> Option<PathBuf> {
        let entries = std::fs::read_dir(root).ok()?;
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_file_recursively(&path, filename) {
                    return Some(found);
                }
                continue;
            }
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n == filename)
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
        None
    }

    fn extract_binary_from_tar_gz(data: &[u8]) -> std::result::Result<Vec<u8>, UpgradeError> {
        let temp_dir = tempfile::tempdir().map_err(|e| UpgradeError::DownloadFailed {
            reason: format!("failed to create temp dir for archive extraction: {e}"),
        })?;
        let archive_path = temp_dir.path().join("release.tar.gz");
        std::fs::write(&archive_path, data).map_err(|e| UpgradeError::DownloadFailed {
            reason: format!("failed to write temporary archive: {e}"),
        })?;

        let status = std::process::Command::new("tar")
            .arg("-xzf")
            .arg(&archive_path)
            .arg("-C")
            .arg(temp_dir.path())
            .status()
            .map_err(|e| UpgradeError::DownloadFailed {
                reason: format!("failed to execute tar for archive extraction: {e}"),
            })?;

        if !status.success() {
            return Err(UpgradeError::DownloadFailed {
                reason: format!(
                    "archive extraction failed with exit code {:?}",
                    status.code()
                ),
            });
        }

        let binary_path = find_file_recursively(temp_dir.path(), expected_binary_name())
            .ok_or_else(|| UpgradeError::DownloadFailed {
                reason: format!(
                    "archive did not contain expected binary '{}'",
                    expected_binary_name()
                ),
            })?;

        std::fs::read(&binary_path).map_err(|e| UpgradeError::DownloadFailed {
            reason: format!(
                "failed to read extracted binary from {}: {e}",
                binary_path.display()
            ),
        })
    }

    #[cfg(target_os = "windows")]
    fn extract_binary_from_zip(data: &[u8]) -> std::result::Result<Vec<u8>, UpgradeError> {
        let temp_dir = tempfile::tempdir().map_err(|e| UpgradeError::DownloadFailed {
            reason: format!("failed to create temp dir for zip extraction: {e}"),
        })?;
        let archive_path = temp_dir.path().join("release.zip");
        std::fs::write(&archive_path, data).map_err(|e| UpgradeError::DownloadFailed {
            reason: format!("failed to write temporary zip archive: {e}"),
        })?;

        let script = format!(
            "Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force",
            archive_path.display(),
            temp_dir.path().display()
        );
        let status = std::process::Command::new("powershell")
            .args(["-NoLogo", "-NoProfile", "-Command", &script])
            .status()
            .map_err(|e| UpgradeError::DownloadFailed {
                reason: format!("failed to execute PowerShell zip extraction: {e}"),
            })?;

        if !status.success() {
            return Err(UpgradeError::DownloadFailed {
                reason: format!("zip extraction failed with exit code {:?}", status.code()),
            });
        }

        let binary_path = find_file_recursively(temp_dir.path(), expected_binary_name())
            .ok_or_else(|| UpgradeError::DownloadFailed {
                reason: format!(
                    "zip archive did not contain expected binary '{}'",
                    expected_binary_name()
                ),
            })?;

        std::fs::read(&binary_path).map_err(|e| UpgradeError::DownloadFailed {
            reason: format!(
                "failed to read extracted binary from {}: {e}",
                binary_path.display()
            ),
        })
    }

    #[cfg(not(target_os = "windows"))]
    fn extract_binary_from_zip(_data: &[u8]) -> std::result::Result<Vec<u8>, UpgradeError> {
        Err(UpgradeError::DownloadFailed {
            reason: "zip upgrades are not supported on this platform".to_string(),
        })
    }

    /// Real smoke tester that runs the binary with --version.
    struct RealSmokeTester;

    impl SmokeTester for RealSmokeTester {
        fn test(&self, binary_path: &Path) -> std::result::Result<(), UpgradeError> {
            let output = std::process::Command::new(binary_path)
                .arg("--version")
                .output()
                .map_err(|e| UpgradeError::SmokeTestFailed {
                    reason: format!("failed to execute smoke test: {e}"),
                })?;

            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(UpgradeError::SmokeTestFailed {
                    reason: format!(
                        "smoke test failed with exit code {:?}: {stderr}",
                        output.status.code()
                    ),
                })
            }
        }
    }

    /// Real user interactor using stdin/stdout.
    struct RealInteractor;

    impl UserInteractor for RealInteractor {
        fn confirm_upgrade(&self, current: &str, target: &str) -> bool {
            print!("Upgrade Rusty Stack from v{current} to v{target}? [y/N] ");
            io::stdout().flush().ok();

            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
        }
    }

    fn format_json_output(upgrade_result: &UpgradeResult) -> String {
        serde_json::to_string_pretty(upgrade_result)
            .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize result: {e}\"}}"))
    }

    fn format_error_json(error: &UpgradeError) -> String {
        let status = match error {
            UpgradeError::Declined => "declined",
            UpgradeError::NoUpgradeAvailable { .. } => "no_upgrade",
            UpgradeError::IncompatibleRuntime { .. } => "refused",
            UpgradeError::RuntimeTooOld { .. } => "refused",
            _ => "error",
        };
        format!(
            r#"{{"status": "{status}", "error": "{}"}}"#,
            error.to_string().replace('"', "\\\"")
        )
    }

    fn print_interactive_result(result: &UpgradeResult) {
        match result.status {
            UpgradeStatus::Success => {
                println!(
                    "✓ Successfully upgraded Rusty Stack from v{} to v{}",
                    result.previous_version, result.new_version
                );
                println!("  Backup saved to: {}", result.backup_path.display());
            }
            UpgradeStatus::RolledBack => {
                println!(
                    "✗ Upgrade from v{} to v{} failed — rolled back to previous version",
                    result.previous_version, result.new_version
                );
            }
            UpgradeStatus::Refused => {
                println!("Upgrade refused due to compatibility issues.");
            }
        }
    }

    fn print_interactive_error(error: &UpgradeError) {
        match error {
            UpgradeError::IncompatibleRuntime { current, required } => {
                eprintln!(
                    "✗ Upgrade refused: current version v{current} does not meet required v{required}"
                );
                eprintln!("  A newer runtime is required before upgrading.");
            }
            UpgradeError::RuntimeTooOld {
                current_schema,
                manifest_schema,
            } => {
                eprintln!(
                    "✗ Runtime too old: schema v{current_schema} cannot parse manifest schema v{manifest_schema}"
                );
                eprintln!(
                    "  A manual upgrade is required. Please download the latest binary from:"
                );
                eprintln!("  https://github.com/scooter-lacroix/Stan-s-ML-Stack/releases");
            }
            UpgradeError::IntegrityCheckFailed { expected, actual } => {
                eprintln!("✗ Binary integrity check failed!");
                eprintln!("  Expected: {expected}");
                eprintln!("  Actual:   {actual}");
                eprintln!("  The downloaded binary may be corrupted or tampered with.");
            }
            UpgradeError::SmokeTestFailed { reason } => {
                eprintln!("✗ Post-upgrade smoke test failed: {reason}");
                eprintln!("  Rolled back to previous version.");
            }
            UpgradeError::DownloadFailed { reason } => {
                eprintln!("✗ Download failed: {reason}");
            }
            UpgradeError::Declined => {
                println!("Upgrade cancelled.");
            }
            UpgradeError::NoUpgradeAvailable { current, latest } => {
                println!("Already up to date (current: v{current}, latest: v{latest}).");
            }
            UpgradeError::IoError { path, reason } => {
                eprintln!("✗ I/O error at {path}: {reason}");
            }
        }
    }
}

// ===========================================================================
// Bench subcommand implementation
// ===========================================================================

mod bench_impl {
    use super::*;

    #[derive(Debug, Serialize)]
    struct BenchmarkOutput {
        name: String,
        success: bool,
        execution_time_ms: u128,
        results: serde_json::Value,
        errors: Vec<String>,
    }

    pub fn run(benchmark: Option<&str>, json: bool, list: bool) {
        if list {
            print_benchmark_list();
            return;
        }

        let name = match benchmark {
            Some(name) => name,
            None => {
                eprintln!("error: no benchmark specified. Use --list to see available benchmarks.");
                process::exit(1);
            }
        };

        // Delegate to benchmark_runners module
        let result = match rusty_stack::benchmark_runners::run_benchmark(name) {
            Ok(output) => {
                // Convert from benchmark_runners::BenchmarkOutput to our local type
                BenchmarkOutput {
                    name: output.name,
                    success: output.success,
                    execution_time_ms: output.execution_time_ms,
                    results: output.results,
                    errors: output.errors,
                }
            }
            Err(err_msg) => {
                let output = BenchmarkOutput {
                    name: name.to_string(),
                    success: false,
                    execution_time_ms: 0,
                    results: serde_json::Value::Object(serde_json::Map::new()),
                    errors: vec![err_msg],
                };
                if json {
                    let json_str = serde_json::to_string_pretty(&output).unwrap_or_default();
                    println!("{}", json_str);
                } else {
                    eprintln!("Error: {}", output.errors.join(", "));
                }
                process::exit(1);
            }
        };

        if json {
            let json_str = serde_json::to_string_pretty(&result).unwrap_or_default();
            println!("{}", json_str);
        } else {
            println!("Benchmark: {}", result.name);
            println!(
                "Status: {}",
                if result.success { "SUCCESS" } else { "FAILED" }
            );
            println!("Time: {} ms", result.execution_time_ms);

            if !result.errors.is_empty() {
                println!("Errors:");
                for e in &result.errors {
                    println!("  - {}", e);
                }
            }

            if result.success {
                if let Some(map) = result.results.as_object() {
                    println!("\nResults:");
                    for (key, value) in map {
                        if let Some(obj) = value.as_object() {
                            let status = obj
                                .get("success")
                                .and_then(|v| v.as_bool())
                                .map(|ok| if ok { "SUCCESS" } else { "FAILED" })
                                .unwrap_or("UNKNOWN");
                            let time_ms = obj
                                .get("execution_time_ms")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            println!("  {} [{} | {} ms]", key, status, time_ms);
                            if let Some(metrics) = obj.get("metrics").and_then(|m| m.as_object()) {
                                for (metric_key, metric_val) in metrics.iter().take(4) {
                                    println!("    - {}: {}", metric_key, metric_val);
                                }
                                if metrics.len() > 4 {
                                    println!("    - ... {} more metrics", metrics.len() - 4);
                                }
                            }
                        } else {
                            println!("  {}: {}", key, value);
                        }
                    }
                }
            }
        }

        if !result.success {
            process::exit(1);
        }
    }

    fn print_benchmark_list() {
        println!("Available benchmarks:");
        println!();
        println!("Pre-installation:");
        println!("  gpu-capability       - GPU capability detection");
        println!("  memory-bandwidth     - HBM memory bandwidth test");
        println!("  tensor-core          - Tensor Core performance");
        println!();
        println!("GPU Performance:");
        println!("  gemm                 - Matrix multiplication benchmark");
        println!();
        println!("Component Benchmarks:");
        println!("  pytorch              - PyTorch performance tests");
        println!("  llama-cpp            - LLaMA.cpp benchmark suite");
        println!("  flash-attention      - Flash Attention vs standard");
        println!("  vllm                 - vLLM throughput benchmark");
        println!("  deepspeed            - DeepSpeed ZeRO performance");
        println!("  megatron             - Megatron-LM throughput/import benchmark");
        println!();
        println!("Combined:");
        println!("  all-pre              - All pre-installation benchmarks");
        println!("  all                  - All benchmarks (optional failures non-fatal)");
    }
}

// ===========================================================================
// Deps subcommand implementation
// ===========================================================================

mod deps_impl {
    use super::*;

    pub fn run(lag_days: u64, verbose: bool, dir: Option<PathBuf>, json: bool) {
        let crate_dir = dir.unwrap_or_else(|| {
            // Default: look for rusty-stack/ relative to the repo root
            let exe = std::env::current_exe().unwrap_or_default();
            let repo_root = exe
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .unwrap_or(Path::new("."));
            repo_root.join("rusty-stack")
        });

        let cargo_toml_path = crate_dir.join("Cargo.toml");
        if !cargo_toml_path.exists() {
            eprintln!(
                "error: Cargo.toml not found at {}",
                cargo_toml_path.display()
            );
            process::exit(2);
        }

        let cargo_lock_path = crate_dir
            .join("Cargo.lock")
            .exists()
            .then(|| crate_dir.join("Cargo.lock"))
            .or_else(|| {
                let project_lock = crate_dir.parent()?.join("Cargo.lock");
                project_lock.exists().then_some(project_lock)
            });

        let deps = parse_direct_deps(&cargo_toml_path);
        if deps.is_empty() {
            println!("No dependencies found in Cargo.toml");
            return;
        }

        if !json {
            println!("═══════════════════════════════════════════════════════════════");
            println!("  Dependency Update Check");
            println!("  Crate:     {}", crate_dir.display());
            println!("  Lag:       {} days", lag_days);
            println!("═══════════════════════════════════════════════════════════════");
            println!();
            println!(
                "Found {} direct dependencies. Checking crates.io...",
                deps.len()
            );
            println!();
        }

        let mut up_to_date = 0u32;
        let mut updates_available = 0u32;
        let mut lag_blocked = 0u32;
        let mut check_failed = 0u32;
        let mut results = Vec::new();

        for dep in &deps {
            let locked_ver = cargo_lock_path
                .as_ref()
                .and_then(|p| find_locked_version(p, &dep.name))
                .unwrap_or_else(|| "0.0.0".to_string());

            match query_crates_io(&dep.name) {
                Ok(info) => {
                    if verbose && !json {
                        println!(
                            "  API response for {}: latest={}, updated={}",
                            dep.name,
                            info.latest,
                            info.updated.as_deref().unwrap_or("N/A")
                        );
                    }

                    let locked = semver::Version::parse(&locked_ver).ok();
                    let latest = semver::Version::parse(&info.latest).ok();

                    let is_newer = match (&locked, &latest) {
                        (Some(l), Some(r)) => r > l,
                        _ => info.latest != locked_ver,
                    };

                    if !is_newer {
                        if !json {
                            println!(
                                "  \u{2705} {} \u{2014} {} (up to date)",
                                dep.name, locked_ver
                            );
                        }
                        results.push(DepResult {
                            name: dep.name.clone(),
                            locked: locked_ver.clone(),
                            latest: info.latest.clone(),
                            status: "up_to_date".into(),
                        });
                        up_to_date += 1;
                    } else if let Some(days_since) = info.days_since_publish() {
                        if days_since >= lag_days as i64 {
                            if !json {
                                println!("  \u{1F4E6} {} \u{2014} {} \u{2192} {} (published {} days ago)", dep.name, locked_ver, info.latest, days_since);
                            }
                            results.push(DepResult {
                                name: dep.name.clone(),
                                locked: locked_ver.clone(),
                                latest: info.latest.clone(),
                                status: "update_available".into(),
                            });
                            updates_available += 1;
                        } else {
                            if !json {
                                println!("  \u{23F3} {} \u{2014} {} \u{2192} {} (published {} days ago, lag: {}d)", dep.name, locked_ver, info.latest, days_since, lag_days);
                            }
                            results.push(DepResult {
                                name: dep.name.clone(),
                                locked: locked_ver.clone(),
                                latest: info.latest.clone(),
                                status: "lag_blocked".into(),
                            });
                            lag_blocked += 1;
                        }
                    } else {
                        if !json {
                            println!("  \u{1F4E6} {} \u{2014} {} \u{2192} {} (no publish date, assuming eligible)", dep.name, locked_ver, info.latest);
                        }
                        results.push(DepResult {
                            name: dep.name.clone(),
                            locked: locked_ver.clone(),
                            latest: info.latest.clone(),
                            status: "update_available".into(),
                        });
                        updates_available += 1;
                    }
                }
                Err(e) => {
                    if !json {
                        println!(
                            "  \u{26A0}\u{FE0F}  {} \u{2014} failed to query crates.io: {}",
                            dep.name, e
                        );
                    }
                    results.push(DepResult {
                        name: dep.name.clone(),
                        locked: locked_ver.clone(),
                        latest: "unknown".into(),
                        status: "check_failed".into(),
                    });
                    check_failed += 1;
                }
            }
        }

        if json {
            let output = serde_json::json!({
                "crate": crate_dir.display().to_string(),
                "lag_days": lag_days,
                "results": results,
                "summary": {
                    "up_to_date": up_to_date,
                    "updates_available": updates_available,
                    "lag_blocked": lag_blocked,
                    "check_failed": check_failed,
                }
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        } else {
            println!();
            println!("═══════════════════════════════════════════════════════════════");
            println!("  Summary");
            println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
            println!("  Up to date:        {}", up_to_date);
            println!("  Updates available: {}", updates_available);
            println!("  Lag-blocked:       {}", lag_blocked);
            println!("  Check failed:      {}", check_failed);
            println!("═══════════════════════════════════════════════════════════════");
        }

        if updates_available > 0 {
            process::exit(1);
        } else if check_failed > 0 {
            process::exit(2);
        }
    }

    // -- Helpers --

    struct DirectDep {
        name: String,
    }

    #[derive(Serialize)]
    struct DepResult {
        name: String,
        locked: String,
        latest: String,
        status: String,
    }

    struct CrateInfo {
        latest: String,
        updated: Option<String>,
    }

    impl CrateInfo {
        fn days_since_publish(&self) -> Option<i64> {
            let updated = self.updated.as_ref()?;
            let published =
                chrono::DateTime::<chrono::FixedOffset>::parse_from_rfc3339(updated).ok()?;
            let now = chrono::Utc::now();
            let duration = now.signed_duration_since(published.with_timezone(&chrono::Utc));
            Some(duration.num_days())
        }
    }

    fn parse_direct_deps(cargo_toml: &Path) -> Vec<DirectDep> {
        let content = std::fs::read_to_string(cargo_toml).unwrap_or_default();
        let mut deps = Vec::new();
        let mut in_deps = false;
        let mut in_dep_subtable = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == "[dependencies]" {
                in_deps = true;
                in_dep_subtable = false;
                continue;
            }
            if in_deps && trimmed.starts_with('[') {
                if let Some(name) = trimmed
                    .strip_prefix("[dependencies.")
                    .and_then(|s| s.strip_suffix(']'))
                    .map(|s| s.trim_matches('"').to_string())
                    .filter(|s| !s.is_empty())
                {
                    deps.push(DirectDep { name });
                    in_dep_subtable = true;
                    continue;
                }
                break;
            }
            if in_deps && !in_dep_subtable {
                // Match: name = "version" or name = { version = "...", ... }
                if let Some(eq_pos) = trimmed.find('=') {
                    let name = trimmed[..eq_pos].trim().to_string();
                    if name.is_empty() || name.starts_with('#') {
                        continue;
                    }
                    deps.push(DirectDep { name });
                }
            }
        }
        deps
    }

    fn find_locked_version(lock_path: &Path, dep_name: &str) -> Option<String> {
        let content = std::fs::read_to_string(lock_path).ok()?;
        let needle = format!("name = \"{}\"", dep_name);
        let mut found_name = false;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == needle {
                found_name = true;
                continue;
            }
            if found_name && trimmed.starts_with("version =") {
                return trimmed.split('"').nth(1).map(|s| s.to_string());
            }
            if found_name && trimmed.starts_with("name =") {
                found_name = false;
            }
        }
        None
    }

    fn query_crates_io(name: &str) -> anyhow::Result<CrateInfo> {
        let url = format!("https://crates.io/api/v1/crates/{}", name);
        let agent = ureq::Agent::new_with_defaults();

        let response = agent
            .get(&url)
            .header(
                "User-Agent",
                "rusty-stack-dep-checker (github.com/scooter-lacroix)",
            )
            .call()
            .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;

        let body: serde_json::Value = response
            .into_body()
            .read_json()
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

        let latest = body["crate"]["max_stable_version"]
            .as_str()
            .or_else(|| body["crate"]["max_version"].as_str())
            .unwrap_or("0.0.0")
            .to_string();

        let updated = body["crate"]["updated_at"]
            .as_str()
            .map(|s: &str| s.to_string());

        Ok(CrateInfo { latest, updated })
    }
}

// ===========================================================================
// Verify subcommand implementation
// ===========================================================================

mod verify_impl {
    use super::*;

    pub fn run(_full: bool, enhanced: bool, build: bool, json: bool) {
        // Determine mode — default to full if none specified
        let mode = if enhanced {
            "enhanced"
        } else if build {
            "build"
        } else {
            "full" // default or --full
        };

        let result = match mode {
            "enhanced" => rusty_stack::verification::enhanced_verify(),
            "build" => rusty_stack::verification::verify_and_build(),
            _ => rusty_stack::verification::full_verify(),
        };

        if json {
            let output = rusty_stack::verification::format_result_json(&result);
            println!("{}", output);
        } else {
            let output = rusty_stack::verification::format_result_human(&result);
            print!("{}", output);
        }

        // For --build mode, list failed components that need rebuild
        if build && !result.all_passed {
            let failed = rusty_stack::verification::failed_components(&result);
            if !json {
                eprintln!("\nComponents needing rebuild:");
                for id in &failed {
                    eprintln!("  - {}", id);
                }
            }
        }

        if !result.all_passed {
            process::exit(1);
        }
    }
}

// ===========================================================================
// Main
// ===========================================================================

// ===========================================================================
// Uninstall / Reinstall subcommand implementation
// ===========================================================================

mod uninstall_impl {
    use super::*;

    pub fn run(keep_rocm: bool, purge_dir: bool, yes: bool, _sudo_password: Option<String>) {
        let _log_guard = rusty_stack::logging::init_logging("uninstall");
        println!("Rusty Stack — uninstall (keep_rocm={keep_rocm}, purge_dir={purge_dir})");
        // Resolve a sudo password (env > TTY prompt) so privileged steps
        // (system-package purge, /opt/rocm removal) can run unattended via
        // askpass — `sudo -n` alone fails when a password is required.
        let sudo_password = sudo_creds::resolve_for_uninstall();
        let opts = rusty_stack::uninstall::UninstallOptions {
            keep_rocm,
            purge_mlstack_dir: purge_dir,
            yes,
            sudo_password,
        };
        match rusty_stack::uninstall::uninstall_stack(&opts) {
            Ok(report) => {
                println!("\n=== Uninstall report ===");
                println!(
                    "pip packages targeted: {}",
                    if report.pip_uninstall_attempted.is_empty() {
                        "(none)".into()
                    } else {
                        report.pip_uninstall_attempted.join(", ")
                    }
                );
                println!(
                    "ROCm system packages purged: {}",
                    report.system_packages_purged
                );
                println!("/opt/rocm removed: {}", report.opt_rocm_removed);
                if !report.env_files_removed.is_empty() {
                    println!("removed: {}", report.env_files_removed.join(", "));
                }
                if !report.sourcing_lines_stripped.is_empty() {
                    println!(
                        "shell sourcing stripped from: {}",
                        report.sourcing_lines_stripped.join(", ")
                    );
                }
                println!("registry cleared: {}", report.registry_cleared);
                if !report.warnings.is_empty() {
                    eprintln!("\nWarnings:");
                    for w in &report.warnings {
                        eprintln!("  - {w}");
                    }
                }
                println!("\nUninstall complete. Reboot recommended before reinstalling.");
            }
            Err(e) => {
                eprintln!("Uninstall failed: {e:#}");
                process::exit(1);
            }
        }
    }
}

mod reinstall_impl {
    use super::*;

    /// Force-reinstall = uninstall the stack, then relaunch the TUI installer.
    /// (Stage 6: previously broken — `rusty` had no reinstall path at all.)
    pub fn run(keep_rocm: bool, purge_dir: bool, _sudo_password: Option<String>) {
        let _log_guard = rusty_stack::logging::init_logging("reinstall");
        println!("Rusty Stack — reinstall: uninstalling, then relaunching installer…");
        let sudo_password = sudo_creds::resolve_for_uninstall();
        let opts = rusty_stack::uninstall::UninstallOptions {
            keep_rocm,
            purge_mlstack_dir: purge_dir,
            yes: true,
            sudo_password,
        };
        if let Err(e) = rusty_stack::uninstall::uninstall_stack(&opts) {
            eprintln!("Uninstall phase failed: {e:#}");
            process::exit(1);
        }
        println!("\nUninstall phase complete. Launching TUI installer…");
        #[cfg(feature = "tui")]
        {
            if let Err(e) = rusty_stack::run_tui() {
                eprintln!("TUI error: {e}");
                process::exit(1);
            }
        }
        #[cfg(not(feature = "tui"))]
        {
            eprintln!("TUI not available in this build; run a TUI-enabled `rusty` to install.");
        }
    }
}

mod install_impl {
    use super::*;

    /// Install command (Tenet 1): ensures the target env (named or global),
    /// pins it for the installer, prints the sourcing command + path, then
    /// launches the installer. `--env <name>` isolates all components into
    /// `~/.mlstack/envs/<name>/`; `--global` (default) uses `~/.mlstack/global/`.
    pub fn run(env: Option<String>, global: bool) {
        let _log_guard = rusty_stack::logging::init_logging("install");
        use rusty_stack::platform::environment::{
            ensure_global_venv, ensure_named_venv, resolve_canonical_python_bin,
        };
        let bootstrap = resolve_canonical_python_bin();

        match env.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(name) => match ensure_named_venv(name, &bootstrap) {
                Ok(py) => {
                    std::env::set_var("MLSTACK_ENV_NAME", name);
                    println!("[install] Named env '{name}' ready: {}", py.display());
                    println!("[install] ALL components will install into ~/.mlstack/envs/{name}/");
                    println!("\n=== Sourcing command (run in your shell to activate the env) ===");
                    println!("  bash/zsh:  source ~/.mlstack/envs/{name}/bin/activate");
                    println!("  fish:      source ~/.mlstack/envs/{name}/bin/activate.fish");
                    println!("  env path:  ~/.mlstack/envs/{name}/");
                }
                Err(e) => {
                    eprintln!("[install] failed to create named env '{name}': {e:#}");
                    process::exit(1);
                }
            },
            None => {
                let _ = global;
                match ensure_global_venv(&bootstrap) {
                    Ok(py) => {
                        std::env::set_var("MLSTACK_PYTHON_BIN", py.to_string_lossy().to_string());
                        println!("[install] Global env ready: {}", py.display());
                        println!(
                            "[install] Components will install into the managed global env ~/.mlstack/global/"
                        );
                        println!("\n=== Sourcing command ===");
                        println!("  bash/zsh:  source ~/.mlstack_env");
                        println!(
                            "  fish:      ~/.config/fish/conf.d/mlstack_env.fish (auto-loaded)"
                        );
                    }
                    Err(e) => {
                        eprintln!("[install] failed to create global env: {e:#}");
                        process::exit(1);
                    }
                }
            }
        }

        println!("\nLaunching installer…");
        #[cfg(feature = "tui")]
        {
            if let Err(e) = rusty_stack::run_tui() {
                eprintln!("TUI error: {e}");
                process::exit(1);
            }
        }
        #[cfg(not(feature = "tui"))]
        {
            eprintln!("TUI not available in this build.");
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Subcommands::Update {
            scan_only,
            all_safe,
            include_experimental,
            yes,
            json,
            components,
        }) => {
            update_impl::run(
                scan_only,
                all_safe,
                include_experimental,
                yes,
                json,
                components,
            );
        }
        Some(Subcommands::Upgrade {
            yes,
            binary_path,
            backup_dir,
            cached_manifest,
            dry_run,
        }) => {
            upgrade_impl::run(yes, binary_path, backup_dir, cached_manifest, dry_run);
        }
        Some(Subcommands::Bench {
            benchmark,
            json,
            list,
        }) => {
            bench_impl::run(benchmark.as_deref(), json, list);
        }
        Some(Subcommands::Deps {
            lag,
            verbose,
            dir,
            json,
        }) => {
            deps_impl::run(lag, verbose, dir, json);
        }
        Some(Subcommands::Verify {
            full,
            enhanced,
            build,
            json,
        }) => {
            verify_impl::run(full, enhanced, build, json);
        }
        Some(Subcommands::Uninstall {
            keep_rocm,
            purge_dir,
            yes,
        }) => {
            uninstall_impl::run(keep_rocm, purge_dir, yes, None);
        }
        Some(Subcommands::Reinstall {
            keep_rocm,
            purge_dir,
        }) => {
            reinstall_impl::run(keep_rocm, purge_dir, None);
        }
        Some(Subcommands::Install { env, global }) => {
            install_impl::run(env, global);
        }
        None => {
            // No subcommand — launch TUI
            #[cfg(feature = "tui")]
            {
                if let Err(e) = rusty_stack::run_tui() {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            }

            #[cfg(not(feature = "tui"))]
            {
                eprintln!(
                    "error: TUI is not available in this build. \
                     Install with TUI support or use a subcommand (update, upgrade, bench)."
                );
                process::exit(1);
            }
        }
    }
}
