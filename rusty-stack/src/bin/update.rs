//! rusty-stack-update — Update command for Rusty Stack components.
//!
//! Scans installed components, resolves the manifest, builds an update plan,
//! classifies updates (safe/guarded/blocked/candidate/experimental), and
//! applies selected updates.
//!
//! # Usage
//!
//! ```text
//! rusty-stack-update [OPTIONS] [COMPONENT...]
//!
//! Options:
//!   --scan-only            Produce plan without applying
//!   --all-safe             Apply only safe-classified updates
//!   --include-experimental Include experimental components
//!   --json                 Force JSON output (also auto-detected from TTY)
//!   -h, --help             Show help
//!   -V, --version          Show version
//! ```
//!
//! # Non-interactive Mode
//!
//! When stdout is not a TTY or `--json` is set, output is structured JSON
//! with `scan`, `plan`, `apply`, and `summary` keys.

use clap::Parser;
use serde::Serialize;
use std::io::{self, IsTerminal};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "rusty-stack-update",
    version = VERSION,
    about = "Update Rusty Stack ML components",
    long_about = "Scans installed components, resolves the manifest, builds an update plan, \
                  classifies updates (safe/guarded/blocked/candidate/experimental), and \
                  applies selected updates. Non-interactive JSON output when not TTY."
)]
struct Cli {
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
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Run the scan phase: detect hardware, installed components, versions.
fn run_scan() -> rusty_stack::orchestrator::planner::ScanOutput {
    use rusty_stack::orchestrator::planner::{InstalledComponent, ScanOutput};
    use rusty_stack::platform::registry::{detect_all_installed, get_version, known_components};

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

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

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

    let plan_output: Vec<PlannerItemOutput> = items.iter().map(PlannerItemOutput::from).collect();
    let summary = PlanSummary::from_items(&items);

    Ok(PlanOutput {
        plan: plan_output,
        summary,
    })
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Format the plan as human-readable text.
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

/// Format an error as human-readable text.
fn format_error_human(error: &rusty_stack::orchestrator::planner::PlannerError) -> String {
    format!("Error: {error}")
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

/// Top-level JSON output structure for non-TTY mode.
#[derive(Debug, Serialize)]
struct JsonOutput {
    scan: rusty_stack::orchestrator::planner::ScanOutput,
    plan: Option<rusty_stack::orchestrator::planner::PlanOutput>,
    apply: Option<serde_json::Value>,
    summary: JsonSummary,
}

/// Summary section of JSON output.
#[derive(Debug, Serialize)]
struct JsonSummary {
    status: String,
    scan_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // Determine output mode: JSON if explicitly requested or not a TTY
    let json_mode = cli.json || !io::stdout().is_terminal();

    // Build planner options from CLI args
    let options = rusty_stack::orchestrator::planner::PlannerOptions {
        scan_only: cli.scan_only,
        all_safe: cli.all_safe,
        include_experimental: cli.include_experimental,
        target_components: cli.components.clone(),
        json_output: json_mode,
    };

    // Phase 1: Scan
    let scan = run_scan();

    if json_mode {
        // JSON output mode — produce structured JSON with scan, plan, apply, summary
        match build_plan(&scan, &options) {
            Ok(plan) => {
                let output = JsonOutput {
                    scan,
                    plan: Some(plan),
                    apply: None,
                    summary: JsonSummary {
                        status: "plan_ready".to_string(),
                        scan_only: cli.scan_only,
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
                        scan_only: cli.scan_only,
                        error: Some(error.to_string()),
                    },
                };
                println!("{}", serde_json::to_string(&output).unwrap_or_default());
                std::process::exit(1);
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

                if cli.scan_only {
                    println!("\n(scan-only mode: no changes will be applied)");
                } else if plan.summary.selected > 0 {
                    println!("\nReady to apply {} updates.", plan.summary.selected);
                } else {
                    println!("\nNo updates selected.");
                }
            }
            Err(error) => {
                eprintln!("{}", format_error_human(&error));
                std::process::exit(1);
            }
        }
    }
}
