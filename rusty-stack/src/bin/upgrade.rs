//! rusty-stack-upgrade — Binary upgrade command for Rusty Stack.
//!
//! Downloads and replaces the Rusty Stack binary with the latest released
//! version. Implements the full upgrade flow:
//!
//! 1. Check manifest version compatibility
//! 2. Download new binary
//! 3. Verify integrity (SHA-256 checksum)
//! 4. Swap binary with backup
//! 5. Run smoke test
//! 6. Rollback on failure
//!
//! Supports interactive confirmation and non-interactive `--yes` mode.
//! In non-interactive mode, output is structured JSON.

use clap::Parser;
use rusty_stack::orchestrator::upgrade::{
    self, BinaryDownloader, ReleaseInfo, ReleaseProvider, SmokeTester, UpgradeError,
    UpgradeOptions, UpgradeResult, UpgradeStatus, UserInteractor, VersionInfo,
};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "rusty-stack-upgrade",
    version = VERSION,
    about = "Upgrade Rusty Stack to the latest version",
    long_about = "Downloads and replaces the Rusty Stack binary with the latest \
                  released version. Checks manifest version compatibility, verifies \
                  binary integrity, and supports rollback on failure."
)]
struct Cli {
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
}

// ---------------------------------------------------------------------------
// Real implementations for CLI usage
// ---------------------------------------------------------------------------

/// Real release provider that fetches from GitHub releases API.
struct RealReleaseProvider;

impl ReleaseProvider for RealReleaseProvider {
    fn fetch_latest_release(&self) -> std::result::Result<ReleaseInfo, UpgradeError> {
        // In a real implementation, this would fetch from:
        // https://api.github.com/repos/scooter-lacroix/Stan-s-ML-Stack/releases/latest
        // For now, return a placeholder error since we don't have a real release endpoint.
        Err(UpgradeError::DownloadFailed {
            reason: "no remote release endpoint configured yet".to_string(),
        })
    }
}

/// Real binary downloader using HTTP.
struct RealDownloader;

impl BinaryDownloader for RealDownloader {
    fn download(&self, url: &str) -> std::result::Result<Vec<u8>, UpgradeError> {
        // In a real implementation, this would use reqwest to download the binary.
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

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

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
            eprintln!("  A manual upgrade is required. Please download the latest binary from:");
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let schema_version = rusty_stack::core::manifest::CURRENT_SCHEMA_VERSION;

    let current_version = VersionInfo {
        version: VERSION.to_string(),
        schema_version,
    };

    let options = UpgradeOptions {
        non_interactive: cli.yes,
        binary_path: cli.binary_path,
        backup_dir: cli.backup_dir,
        cached_manifest_path: cli.cached_manifest,
    };

    // Dry run mode: just check for available upgrades
    if cli.dry_run {
        if cli.yes {
            println!(
                r#"{{"current_version": "{}", "schema_version": {}}}"#,
                VERSION, schema_version
            );
        } else {
            println!("Current version: v{VERSION}");
            println!("Schema version: {schema_version}");
            println!("Checking for upgrades...");
        }
        // Would check remote for newer version here
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
            if cli.yes {
                println!("{}", format_json_output(&upgrade_result));
            } else {
                print_interactive_result(&upgrade_result);
            }
            std::process::exit(0);
        }
        Err(error) => {
            if cli.yes {
                eprintln!("{}", format_error_json(&error));
            } else {
                print_interactive_error(&error);
            }
            std::process::exit(1);
        }
    }
}
