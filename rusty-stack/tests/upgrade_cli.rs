//! CLI integration tests for `rusty-stack upgrade` subcommand.
//!
//! Tests the CLI surface using `assert_cmd` to verify:
//! - `rusty-stack upgrade --help` works
//! - `rusty-stack --version` works
//! - `rusty-stack upgrade --yes` non-interactive mode produces JSON output
//! - `rusty-stack upgrade --dry-run` mode reports current version
//! - Error handling produces proper exit codes

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

/// The unified binary name.
const BIN: &str = "rusty-stack";

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

// ---- Non-interactive mode always emits machine-readable JSON ----

#[test]
fn test_upgrade_non_interactive_produces_json_on_error() {
    // In --yes (non-interactive) mode the outcome is ALWAYS machine-readable
    // JSON on stderr — a `"status"` object — whether an upgrade was applied,
    // none was needed (`no_upgrade`), or it failed. The exact status/exit code
    // depends on the resolved latest version at run time (e.g. when the local
    // version already leads the published one, `no_upgrade` is a *success*),
    // so we assert the shape, not a specific outcome or exit code.
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade", "--yes"])
        .assert()
        .stderr(predicate::str::contains("\"status\":"));
}

// ---- Interactive mode output is human-readable ----

#[test]
fn test_upgrade_interactive_error_is_human_readable() {
    // Interactive mode (no --yes) must ALWAYS render human-readable output —
    // success, "already up to date", a declined prompt, OR a real error — and
    // never the raw JSON payload that --yes mode emits. Release/version state
    // is environment-dependent (current vs. latest, endpoint reachability), so
    // we assert the human-readable contract, not one specific error string.
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["upgrade"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    // Interactive mode never emits the --yes JSON payload.
    assert!(
        !combined.contains("\"status\":"),
        "interactive upgrade must stay human-readable, got JSON: {combined}"
    );
    // ...and always prints a human-readable line (prompt / result / cancel /
    // already-up-to-date / error marker).
    let human_markers = [
        "Upgrade Rusty Stack",
        "Upgrade cancelled",
        "Already up to date",
        "Successfully upgraded",
        "✗",
    ];
    assert!(
        human_markers.iter().any(|m| combined.contains(m)),
        "interactive upgrade must emit a human-readable message, got: {combined}"
    );
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
