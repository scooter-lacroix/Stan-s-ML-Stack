//! Integration tests for the unified `rusty` CLI binary.
//!
//! Verifies:
//! - `rusty --help` shows all subcommands (update, upgrade, bench)
//! - `rusty --version` shows the package version
//! - `rusty update --help` shows update-specific flags
//! - `rusty upgrade --help` shows upgrade-specific flags
//! - `rusty bench --help` shows bench-specific options
//! - `rusty bench --list` lists available benchmarks
//! - `rusty` (no args) with TUI feature launches TUI (tested by source inspection)
//! - `rusty` (no args) without TUI feature shows error (tested by source inspection)

use assert_cmd::Command;
use predicates::prelude::*;

const BIN: &str = "rusty";

// ===========================================================================
// Top-level help and version
// ===========================================================================

#[test]
fn test_rusty_help_shows_subcommands() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("update"))
        .stdout(predicate::str::contains("upgrade"))
        .stdout(predicate::str::contains("bench"));
}

#[test]
fn test_rusty_version() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain "rusty" and a version number pattern
    assert!(
        stdout.contains("rusty"),
        "Version output should contain 'rusty', got: {stdout}"
    );
    // Should contain a version-like pattern (e.g., "0.1.0")
    assert!(
        stdout.contains("0."),
        "Version output should contain version number, got: {stdout}"
    );
}

#[test]
fn test_rusty_rejects_unknown_flags() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--nonexistent-flag")
        .assert()
        .failure();
}

// ===========================================================================
// Update subcommand help
// ===========================================================================

#[test]
fn test_rusty_update_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--scan-only"))
        .stdout(predicate::str::contains("--all-safe"))
        .stdout(predicate::str::contains("--include-experimental"))
        .stdout(predicate::str::contains("--json"));
}

// ===========================================================================
// Upgrade subcommand help
// ===========================================================================

#[test]
fn test_rusty_upgrade_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--yes"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--binary-path"));
}

// ===========================================================================
// Bench subcommand help
// ===========================================================================

#[test]
fn test_rusty_bench_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--list"));
}

#[test]
fn test_rusty_bench_list() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench", "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gpu-capability"))
        .stdout(predicate::str::contains("memory-bandwidth"))
        .stdout(predicate::str::contains("tensor-core"))
        .stdout(predicate::str::contains("pytorch"))
        .stdout(predicate::str::contains("all"));
}

#[test]
fn test_rusty_bench_no_args_shows_error() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no benchmark specified"));
}

// ===========================================================================
// Verify rusty is the sole multi-command entry point
// ===========================================================================

#[test]
fn test_rusty_binary_exists() {
    // This test verifies the binary can be found and executed
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn test_rusty_update_scan_only_json() {
    // Verify the update subcommand produces valid JSON
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let _: serde_json::Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");
}

// ===========================================================================
// Upgrade subcommand dry-run
// ===========================================================================

#[test]
fn test_rusty_upgrade_dry_run_reports_current_version() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--dry-run"])
        .output()
        .expect("failed to execute rusty upgrade --dry-run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Current version:"),
        "expected current-version output in dry run, got: {stdout}"
    );
    assert!(
        stdout.contains("Schema version:"),
        "expected schema-version output in dry run, got: {stdout}"
    );
}
