//! CLI integration tests for `rusty update` subcommand.
//!
//! Tests the CLI surface using `assert_cmd` to verify:
//! - `rusty update --help` works
//! - `rusty --version` works
//! - `rusty update --scan-only` produces JSON output
//! - `rusty update --all-safe` applies only safe updates
//! - `rusty update --include-experimental` includes experimental components
//! - `rusty update --json` forces JSON output
//! - Targeted component selection works
//! - Unknown component produces error
//! - Non-interactive output is machine-readable JSON

use assert_cmd::Command;
use predicates::prelude::*;

/// The unified binary name.
const BIN: &str = "rusty";

// ===========================================================================
// Help and version tests
// ===========================================================================

#[test]
fn test_update_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Scans installed components"))
        .stdout(predicate::str::contains("--scan-only"))
        .stdout(predicate::str::contains("--all-safe"))
        .stdout(predicate::str::contains("--include-experimental"))
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn test_update_version() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("rusty"));
}

// ===========================================================================
// JSON output tests (non-interactive / non-TTY mode)
// ===========================================================================

#[test]
fn test_update_scan_only_json_output() {
    // --scan-only with --json should produce valid JSON
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Must have scan, plan, apply, summary keys
    assert!(parsed.get("scan").is_some(), "JSON must have 'scan' key");
    assert!(parsed.get("plan").is_some(), "JSON must have 'plan' key");
    assert!(
        parsed.get("summary").is_some(),
        "JSON must have 'summary' key"
    );

    // scan_only should be true
    let summary = parsed.get("summary").unwrap();
    assert_eq!(summary.get("scan_only").unwrap().as_bool(), Some(true));
    assert_eq!(summary.get("status").unwrap().as_str(), Some("plan_ready"));
}

#[test]
fn test_update_json_output_has_scan_info() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let scan = parsed.get("scan").unwrap();
    // Must have rocm_version, rocm_channel, gpu_architecture, installed
    assert!(scan.get("rocm_version").is_some());
    assert!(scan.get("rocm_channel").is_some());
    assert!(scan.get("gpu_architecture").is_some());
    assert!(scan.get("installed").is_some());
}

#[test]
fn test_update_json_output_has_plan() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let plan = parsed.get("plan").unwrap();
    // Plan should have plan array and summary
    assert!(plan.get("plan").is_some());
    assert!(plan.get("summary").is_some());

    let summary = plan.get("summary").unwrap();
    assert!(summary.get("total").is_some());
    assert!(summary.get("safe").is_some());
    assert!(summary.get("guarded").is_some());
    assert!(summary.get("candidate").is_some());
    assert!(summary.get("blocked").is_some());
    assert!(summary.get("selected").is_some());
}

#[test]
fn test_update_all_safe_json() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--all-safe", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let plan = parsed.get("plan").unwrap();
    let plan_items = plan.get("plan").unwrap().as_array().unwrap();

    // All items should be classified as "safe"
    for item in plan_items {
        let classification = item.get("classification").unwrap().as_str().unwrap();
        assert_eq!(
            classification, "safe",
            "With --all-safe, only safe items should appear"
        );
    }
}

#[test]
fn test_update_include_experimental_adds_items() {
    // Without --include-experimental
    let output_without = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout_without = String::from_utf8_lossy(&output_without.stdout);
    let parsed_without: serde_json::Value = serde_json::from_str(&stdout_without).unwrap();
    let plan_without = parsed_without
        .get("plan")
        .unwrap()
        .get("plan")
        .unwrap()
        .as_array()
        .unwrap()
        .len();

    // With --include-experimental
    let output_with = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--include-experimental", "--json"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout_with = String::from_utf8_lossy(&output_with.stdout);
    let parsed_with: serde_json::Value = serde_json::from_str(&stdout_with).unwrap();
    let plan_with = parsed_with
        .get("plan")
        .unwrap()
        .get("plan")
        .unwrap()
        .as_array()
        .unwrap()
        .len();

    // With experimental should have >= items
    assert!(
        plan_with >= plan_without,
        "Including experimental should have >= items"
    );
}

// ===========================================================================
// Targeted component tests
// ===========================================================================

#[test]
fn test_update_target_unknown_component_fails() {
    // Targeting a nonexistent component should fail
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json", "nonexistent-component-xyz"])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn test_update_target_known_component_succeeds() {
    // Targeting a known component (rocm) should succeed
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only", "--json", "rocm"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let plan_items = parsed
        .get("plan")
        .unwrap()
        .get("plan")
        .unwrap()
        .as_array()
        .unwrap();

    // Should only have items for the targeted component
    for item in plan_items {
        let id = item.get("component_id").unwrap().as_str().unwrap();
        assert_eq!(id, "rocm", "Only rocm should be in the plan");
    }
}

// ===========================================================================
// Error handling tests
// ===========================================================================

#[test]
fn test_update_rejects_unknown_flags() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--nonexistent-flag"])
        .assert()
        .failure();
}

#[test]
fn test_update_non_tty_produces_json() {
    // When piped (non-TTY), output should be JSON by default
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["update", "--scan-only"])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should parse as valid JSON
    let _: serde_json::Value =
        serde_json::from_str(&stdout).expect("Non-TTY output should be valid JSON");
}
