//! CLI integration tests for `rusty upgrade` subcommand.
//!
//! Tests the CLI surface using `assert_cmd` to verify:
//! - `rusty upgrade --help` works
//! - `rusty --version` works
//! - `rusty upgrade --yes` non-interactive mode produces JSON output
//! - `rusty upgrade --dry-run` mode reports current version
//! - Error handling produces proper exit codes

use assert_cmd::Command;
use predicates::prelude::*;

/// The unified binary name.
const BIN: &str = "rusty";

// ---- Help and version tests ----

#[test]
fn test_upgrade_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Rusty Stack"))
        .stdout(predicate::str::contains("--yes"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--binary-path"));
}

#[test]
fn test_upgrade_version() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("rusty"));
}

// ---- Dry run tests ----

#[test]
fn test_upgrade_dry_run_interactive() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Current version:"))
        .stdout(predicate::str::contains("Schema version:"));
}

#[test]
fn test_upgrade_dry_run_non_interactive() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--dry-run", "--yes"])
        .assert()
        .success()
        .stdout(predicate::str::contains("current_version"));
}

// ---- Non-interactive mode error produces JSON ----

#[test]
fn test_upgrade_non_interactive_produces_json_on_error() {
    // Without a real release endpoint, the upgrade will fail with a download error.
    // In --yes mode, the error should be JSON-formatted.
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--yes"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("\"status\":"));
}

// ---- Interactive mode error is human-readable ----

#[test]
fn test_upgrade_interactive_error_is_human_readable() {
    // Without a real release endpoint, the upgrade will fail.
    // In interactive mode (no --yes), the error should be human-readable.
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Download failed"));
}

// ---- Invalid arguments are rejected ----

#[test]
fn test_upgrade_rejects_unknown_flags() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--nonexistent-flag"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unexpected argument").or(predicate::str::contains("error")),
        );
}
