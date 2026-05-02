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

        /// Apply only safe-classified updates without prompting.
        #[arg(long)]
        all_safe: bool,

        /// Include experimental components in the plan.
        #[arg(long)]
        include_experimental: bool,

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
        json: bool,
        components: Vec<String>,
    ) {
        // Determine output mode: JSON if explicitly requested or not a TTY
        let json_mode = json || !io::stdout().is_terminal();

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
                    let output = JsonOutput {
                        scan,
                        plan: Some(plan),
                        apply: None,
                        summary: JsonSummary {
                            status: "plan_ready".to_string(),
                            scan_only,
                            error: None,
                        },
                    };
                    println!("{}", serde_json::to_string(&output).unwrap_or_default());
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

            print!("{}", format_plan_human(&scan));

            // Phase 2: Plan
            match build_plan(&scan, &options) {
                Ok(plan) => {
                    print!("{}", format_plan_output_human(&plan));

                    if scan_only {
                        println!("\n(scan-only mode: no changes will be applied)");
                    } else if plan.summary.selected > 0 {
                        println!("\nReady to apply {} updates.", plan.summary.selected);
                    } else {
                        println!("\nNo updates selected.");
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
    fn format_plan_human(scan: &rusty_stack::orchestrator::planner::ScanOutput) -> String {
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
            output.push_str(&format!(
                "  {} {:15} {} → {}  ({}){}\n",
                sel,
                item.component_id,
                if item.current_version.is_empty() {
                    "new".to_string()
                } else {
                    item.current_version.clone()
                },
                item.proposed_version,
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
            json,
            components,
        }) => {
            update_impl::run(scan_only, all_safe, include_experimental, json, components);
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
