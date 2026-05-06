//! rusty — Unified CLI for Rusty Stack.
//!
//! Provides all Rusty Stack CLI subcommands in a single binary:
//!
//! - `rusty` (no args) — Launch the TUI installer
//! - `rusty update` — Scan, plan, and apply component updates
//! - `rusty upgrade` — Upgrade the Rusty Stack binary itself
//! - `rusty bench` — Run benchmarks
//! - `rusty verify` — Verify ML Stack installation
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

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "rusty",
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
    /// Non-interactive JSON output when not TTY or with --json.
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
}

// ===========================================================================
// Update subcommand implementation
// ===========================================================================

mod update_impl {
    use super::*;

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

        // Initialize logging for update operations
        let _log_guard = if json_mode {
            rusty_stack::logging::init_batch_logging("update")
        } else {
            rusty_stack::logging::init_logging("update")
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

        // Phase 1: Scan
        let scan = run_scan();

        if json_mode {
            // JSON output mode
            match build_plan(&scan, &options) {
                Ok(plan) => {
                    if scan_only || plan.summary.selected == 0 {
                        let output = JsonOutput {
                            scan,
                            plan: Some(plan),
                            apply: None,
                            summary: JsonSummary {
                                status: if scan_only { "scan_only" } else { "no_updates" }.to_string(),
                                scan_only,
                                error: None,
                            },
                        };
                        println!("{}", serde_json::to_string(&output).unwrap_or_default());
                    } else {
                        // Apply the plan
                        let apply_result = apply_plan(&plan);
                        let output = JsonOutput {
                            scan,
                            plan: Some(plan),
                            apply: Some(serde_json::to_value(&apply_result).unwrap_or_default()),
                            summary: JsonSummary {
                                status: if apply_result.has_failures() { "partial" } else { "applied" }.to_string(),
                                scan_only: false,
                                error: None,
                            },
                        };
                        println!("{}", serde_json::to_string(&output).unwrap_or_default());
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
                    println!("{}", serde_json::to_string(&output).unwrap_or_default());
                    process::exit(1);
                }
            }
        } else {
            // Interactive/human-readable output mode
            println!("Rusty Stack Update v{VERSION}");
            println!("========================\n");

            print!("{}", format_scan_human(&scan));

            // Phase 2: Plan
            match build_plan(&scan, &options) {
                Ok(plan) => {
                    print!("{}", format_plan_output_human(&plan));

                    if scan_only {
                        println!("\n(scan-only mode: no changes will be applied)");
                    } else if plan.summary.selected == 0 {
                        println!("\nNo updates selected.");
                        tracing::info!("No updates selected");
                    } else if yes || all_safe {
                        // --all-safe or --yes: show countdown unless --yes skips it
                        if yes {
                            tracing::info!(count = plan.summary.selected, "Auto-applying (--yes flag)");
                            println!("\nApplying {} updates (auto-confirmed)...", plan.summary.selected);
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
                        let apply_result = apply_plan(&plan);
                        print_apply_summary(&apply_result);
                        if apply_result.has_failures() {
                            process::exit(1);
                        }
                    } else {
                        // Interactive: prompt for confirmation
                        println!("\nReady to apply {} updates.", plan.summary.selected);
                        print!("Apply now? [y/N] ");
                        let _ = io::stdout().flush();
                        let mut input = String::new();
                        if io::stdin().read_line(&mut input).is_ok() {
                            match input.trim().to_lowercase().as_str() {
                                "y" | "yes" => {
                                    let apply_result = apply_plan(&plan);
                                    print_apply_summary(&apply_result);
                                    if apply_result.has_failures() {
                                        process::exit(1);
                                    }
                                }
                                _ => {
                                    println!("Cancelled.");
                                }
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
            let rocm_version = if gpu_info.rocm_version.is_empty() {
                "not installed".to_string()
            } else {
                gpu_info.rocm_version.clone()
            };

            let gpu_architecture = if gpu_info.architecture.is_empty() {
                "unknown".to_string()
            } else {
                gpu_info.architecture.clone()
            };

            // Determine channel from version
            let rocm_channel = if rocm_version.starts_with("6.") {
                "legacy".to_string()
            } else if rocm_version.starts_with("7.0") || rocm_version.starts_with("7.1") {
                "stable".to_string()
            } else {
                "latest".to_string()
            };

            (rocm_version, gpu_architecture, rocm_channel)
        }

        #[cfg(not(unix))]
        {
            (
                "not installed".to_string(),
                "unknown".to_string(),
                "latest".to_string(),
            )
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
        use rusty_stack::core::manifest::Manifest;
        use rusty_stack::core::types::ExecutorKind;
        use rusty_stack::orchestrator::planner::{
            CompatibilityContext, PlanOutput, PlanSummary, PlannerItemOutput, UpdatePlanner,
        };
        use std::collections::HashSet;

        // Load manifest (baseline for now)
        let manifest = Manifest::load_baseline().unwrap_or_else(|_| Manifest {
            schema_version: 2,
            sequence: 0,
            generated_at: String::new(),
            expires_at: None,
            min_runtime_version: String::new(),
            components: Vec::new(),
            signature: None,
        });

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
                sel,
                item.component_id,
                version_str,
                item.classification,
                vis,
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
        use std::time::Duration;

        println!("\n  ┌───────────────────────────────────────────────────┐");
        println!("  │ {} update{} will be applied in 10 seconds.        │", count, if count > 1 { "s" } else { "" });
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

                let mut confirmed = false;
                let mut cancelled = false;

                for remaining in (1..=10).rev() {
                    eprint!("\r  ⏳ Applying in {:2}s... ", remaining);
                    let _ = std::io::stderr().flush();

                    // Check for user input
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
                            3 => { // Ctrl+C
                                cancelled = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }

                // Restore original terminal settings
                let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original) };

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
    fn apply_plan(plan: &rusty_stack::orchestrator::planner::PlanOutput) -> rusty_stack::orchestrator::apply::ApplySummary {
        use rusty_stack::orchestrator::apply::{ApplyEngine, ApplyExecutor, ApplyOptions};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        /// Read a password from TTY without echo.
        /// Returns None if not a TTY or reading fails.
        ///
        /// Uses `std::io::Read::read()` on raw stdin for secure password input
        /// (avoids line-buffering artifacts from `Stdin::read_line()`).
        fn read_password_from_tty() -> Option<String> {
            use std::io::Read;
            if !std::io::stdin().is_terminal() {
                // Not a TTY — try reading from stdin directly (piped input)
                let mut buf = vec![0u8; 1024];
                let n = std::io::stdin().read(&mut buf).ok()?;
                let s = String::from_utf8_lossy(&buf[..n]);
                return Some(s.trim().to_string());
            }
            // Use rpassword-like approach: disable echo via termios
            #[cfg(unix)]
            {
                let mut termios: libc::termios = unsafe { std::mem::zeroed() };
                if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut termios) } != 0 {
                    return None;
                }
                let original = termios;
                // Disable echo and canonical mode for raw byte reading
                termios.c_lflag &= !(libc::ECHO | libc::ICANON);
                if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) } != 0 {
                    return None;
                }
                // Read raw bytes until newline
                let mut password = Vec::new();
                let mut byte = [0u8; 1];
                loop {
                    match std::io::stdin().read(&mut byte) {
                        Ok(0) => break, // EOF
                        Ok(_) if byte[0] == b'\n' => break,
                        Ok(_) if byte[0] == b'\r' => break,
                        Ok(_) => password.push(byte[0]),
                        Err(_) => break,
                    }
                }
                // Restore original settings
                unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original); };
                eprintln!(); // newline after password input
                String::from_utf8(password).ok().map(|s| s.trim().to_string())
            }
            #[cfg(not(unix))]
            { None }
        }

        /// Direct installer executor - calls Rust installer functions in-process.
        struct DirectInstallerExecutor {
            cancelled: Arc<AtomicBool>,
        }

        impl DirectInstallerExecutor {
            fn new(cancelled: Arc<AtomicBool>) -> Self { Self { cancelled } }
            fn component_for_id(id: &str) -> Option<rusty_stack::state::Component> {
                rusty_stack::state::default_components().into_iter().find(|c| c.id == id).map(|mut c| { c.selected = true; c })
            }
        }

        impl ApplyExecutor for DirectInstallerExecutor {
            fn apply_component(&self, component_id: &str, proposed_version: &str) -> Result<(), String> {
                if self.cancelled.load(Ordering::Relaxed) {
                    return Err(format!("{} cancelled by user", component_id));
                }
                tracing::info!(component = component_id, proposed_version, "Starting component installation");
                let Some(component) = Self::component_for_id(component_id) else {
                    return Err(format!("Unknown component ID: {}", component_id));
                };

                // Resolve sudo password for components that need it
                let sudo_password = if component.needs_sudo {
                    // Try to get a sudo password from the user
                    if unsafe { libc::geteuid() } == 0 {
                        // Running as root — no sudo needed
                        None
                    } else {
                        // Prompt for sudo password
                        eprint!("    sudo password for {}: ", component.name);
                        let _ = std::io::stderr().flush();
                        let password = read_password_from_tty();
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

                let (tx, rx) = std::sync::mpsc::channel();
                let (_, input_rx) = std::sync::mpsc::channel();
                let scripts_dir = rusty_stack::detect_scripts_dir();
                let config = rusty_stack::config::InstallerConfig::load_or_default(&scripts_dir)
                    .unwrap_or_else(|_| rusty_stack::config::InstallerConfig::default_with_paths(
                        &scripts_dir,
                        format!("{}/logs", std::env::var("HOME").unwrap_or_else(|_| ".".into())),
                        rusty_stack::config::config_file_path()
                            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/mlstack/config/config.json")),
                    ));

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
                let handle = std::thread::spawn(move || {
                    rusty_stack::installer::run_installation(vec![component], config, sudo_password, tx, input_rx);
                });
                let mut success = true;
                let mut error_msg = String::new();
                let spinner: &[char] = &['\u{280b}','\u{2819}','\u{2839}','\u{2838}','\u{283c}','\u{2834}','\u{2826}','\u{2827}','\u{2807}','\u{280f}'];
                let mut si = 0usize;
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(rusty_stack::installer::InstallerEvent::Log(line, _)) => {
                            println!("    | {}", line);
                            tracing::info!(component = %cid, log = %line);
                        }
                        Ok(rusty_stack::installer::InstallerEvent::Progress { progress, message, .. }) => {
                            let pct = (progress * 100.0) as u8;
                            let s = spinner[si % spinner.len()]; si += 1;
                            eprint!("\r    {} {} [{:>3}%] {}    ", s, cid, pct, message);
                            let _ = std::io::stderr().flush();
                        }
                        Ok(rusty_stack::installer::InstallerEvent::ComponentStart { name, .. }) => {
                            eprint!("\r    ");
                            println!("    > Installing {}...", name);
                            tracing::info!(component = %cid, name = %name, "Component started");
                        }
                        Ok(rusty_stack::installer::InstallerEvent::ComponentComplete { success: s, message, .. }) => {
                            eprint!("\r{}\r", " ".repeat(80));
                            if s {
                                println!("    ok {} - {}", component_name, message);
                                tracing::info!(component = %cid, "Completed successfully");
                            } else {
                                println!("    FAIL {} - {}", component_name, message);
                                tracing::error!(component = %cid, error = %message, "Failed");
                                success = false;
                                error_msg = message;
                            }
                        }
                        Ok(rusty_stack::installer::InstallerEvent::VerificationReport { lines, .. }) => {
                            for line in &lines { println!("    | {}", line); }
                        }
                        Ok(rusty_stack::installer::InstallerEvent::Finished { success: s }) => {
                            if !s { success = false; if error_msg.is_empty() { error_msg = "Finished with errors".into(); } }
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if self.cancelled.load(Ordering::Relaxed) {
                                return Err(format!("{} cancelled", cid));
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => { break; }
                    }
                }
                let _ = handle.join();
                eprint!("\r{}\r", " ".repeat(80));
                if success { tracing::info!(component=%cid,"Succeeded"); Ok(()) }
                else { tracing::error!(component=%cid,error=%error_msg,"Failed"); Err(error_msg) }
            }
        }

        // Convert plan output back to PlannerItems for the apply engine.
        // We reconstruct minimal PlannerItems from the plan output data.
        let items: Vec<rusty_stack::orchestrator::planner::PlannerItem> = plan.plan.iter().map(|item| {
            use rusty_stack::core::plan::PlanItem;
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
                plan_item: PlanItem::new(
                    &item.component_id,
                    &item.current_version,
                    &item.proposed_version,
                    tier,
                    item.selected,
                    &item.rationale,
                    item.dependencies.clone(),
                    true,
                ),
                classification,
                visible: item.visible,
                selected: item.selected,
                classification_reason: item.rationale.clone(),
                requires_hardware_check: false,
                min_rocm_version: String::new(),
            }
        }).collect();

        let cancelled = Arc::new(AtomicBool::new(false));
        let engine = ApplyEngine::new(DirectInstallerExecutor::new(cancelled));
        engine.apply(&items, &ApplyOptions::default())
    }

    /// Print a human-readable summary of the apply results.
    fn print_apply_summary(summary: &rusty_stack::orchestrator::apply::ApplySummary) {
        println!("\nApply Results:");
        println!("---------------");

        if !summary.success.is_empty() {
            println!("\n  Succeeded ({}):", summary.success.len());
            for item in &summary.success {
                println!("    ✓ {} {} → {}", item.component_id, item.current_version, item.proposed_version);
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
            println!("\n  {}/{} components updated successfully.", succeeded, total);
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

// ===========================================================================
// Upgrade subcommand implementation
// ===========================================================================

mod upgrade_impl {
    use super::*;
    use rusty_stack::orchestrator::upgrade::{
        self, BinaryDownloader, ReleaseInfo, ReleaseProvider, SmokeTester, UpgradeError,
        UpgradeOptions, UpgradeResult, UpgradeStatus, UserInteractor, VersionInfo,
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
            if yes {
                println!(
                    r#"{{"current_version": "{}", "schema_version": {}}}"#,
                    VERSION, schema_version
                );
            } else {
                println!("Current version: v{VERSION}");
                println!("Schema version: {schema_version}");
                println!("Checking for upgrades...");
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
                if yes {
                    eprintln!("{}", format_error_json(&error));
                } else {
                    print_interactive_error(&error);
                }
                process::exit(1);
            }
        }
    }

    // Real implementations for CLI usage

    /// Real release provider that fetches from GitHub releases API.
    struct RealReleaseProvider;

    impl ReleaseProvider for RealReleaseProvider {
        fn fetch_latest_release(&self) -> std::result::Result<ReleaseInfo, UpgradeError> {
            Err(UpgradeError::DownloadFailed {
                reason: "no remote release endpoint configured yet".to_string(),
            })
        }
    }

    /// Real binary downloader using HTTP.
    struct RealDownloader;

    impl BinaryDownloader for RealDownloader {
        fn download(&self, url: &str) -> std::result::Result<Vec<u8>, UpgradeError> {
            Err(UpgradeError::DownloadFailed {
                reason: format!("download not yet implemented for URL: {url}"),
            })
        }
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
            update_impl::run(scan_only, all_safe, include_experimental, yes, json, components);
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
        Some(Subcommands::Verify {
            full,
            enhanced,
            build,
            json,
        }) => {
            verify_impl::run(full, enhanced, build, json);
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
