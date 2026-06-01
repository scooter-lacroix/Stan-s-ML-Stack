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
use serde_json::Value;

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
    let assert = Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--dry-run"])
        .assert();
    let output = assert.get_output();

    // Dry-run always reports local runtime info.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Current version:"));
    assert!(stdout.contains("Schema version:"));

    // Remote lookup may succeed (upgrade metadata) or fail (error details),
    // depending on platform asset availability and network state.
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        assert!(stdout.contains("Latest release:"));
        assert!(stdout.contains("Upgrade available:"));
    } else {
        assert!(stderr.contains("Unable to check latest release:"));
    }
}

#[test]
fn test_upgrade_dry_run_non_interactive() {
    let assert = Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--dry-run", "--yes"])
        .assert();
    let output = assert.get_output();

    // In --yes mode, dry-run emits JSON payload on either success (stdout)
    // or failure (stderr).
    let payload = if output.status.success() {
        &output.stdout
    } else {
        &output.stderr
    };
    let payload = String::from_utf8_lossy(payload);
    let parsed: Value = serde_json::from_str(&payload).expect("dry-run JSON payload");
    assert!(parsed.get("current_version").is_some());
    assert!(parsed.get("schema_version").is_some());
    assert!(parsed.get("upgrade_available").is_some());

    if output.status.success() {
        assert!(parsed.get("latest_version").is_some());
    } else {
        assert!(parsed.get("error").is_some());
    }
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
