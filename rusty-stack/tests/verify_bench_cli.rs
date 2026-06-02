//! Integration tests for `rusty verify` and `rusty bench` subcommands.
//!
//! Tests the CLI surface using assert_cmd:
//! - `rusty verify --help` shows verify subcommand options
//! - `rusty verify --full` runs full verification
//! - `rusty verify --enhanced` runs enhanced verification
//! - `rusty verify --build` runs verify-and-build mode
//! - `rusty bench <name>` dispatches to correct benchmark
//! - `rusty bench --all` runs full benchmark suite
//! - `rusty bench --json` produces parseable JSON
//! - Unknown benchmark names return error exit code 1

use assert_cmd::Command;
use predicates::prelude::*;

const BIN: &str = "rusty";

// ===========================================================================
// Verify subcommand — help and flags
// ===========================================================================

#[test]
fn test_rusty_verify_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["verify", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--full"))
        .stdout(predicate::str::contains("--enhanced"))
        .stdout(predicate::str::contains("--build"));
}

#[test]
fn test_rusty_help_shows_verify() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("verify"));
}

#[test]
fn test_rusty_verify_no_flags_shows_error() {
    // Running verify without --full, --enhanced, or --build should show usage error
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["verify"])
        .assert()
        .failure();
}

#[test]
fn test_rusty_verify_full_runs() {
    // --full should run without crashing; exit code depends on what's installed
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["verify", "--full"])
        .assert()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should produce output mentioning component names
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("ROCm")
            || combined.contains("PyTorch")
            || combined.contains("Verification")
            || combined.contains("verification")
            || combined.contains("rocm")
            || combined.contains("pytorch")
            || combined.contains("component"),
        "verify --full should produce verification output, got: {combined}"
    );
}

#[test]
fn test_rusty_verify_enhanced_runs() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["verify", "--enhanced"])
        .assert()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("ROCm")
            || combined.contains("PyTorch")
            || combined.contains("Verification")
            || combined.contains("verification")
            || combined.contains("rocm")
            || combined.contains("pytorch")
            || combined.contains("component"),
        "verify --enhanced should produce verification output, got: {combined}"
    );
}

#[test]
fn test_rusty_verify_build_runs() {
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["verify", "--build"])
        .assert()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("ROCm")
            || combined.contains("PyTorch")
            || combined.contains("Verification")
            || combined.contains("verification")
            || combined.contains("rocm")
            || combined.contains("pytorch")
            || combined.contains("component")
            || combined.contains("build"),
        "verify --build should produce verification output, got: {combined}"
    );
}

// ===========================================================================
// Bench subcommand — existing tests extended for new --all flag
// ===========================================================================

#[test]
fn test_rusty_bench_unknown_name_returns_error() {
    // Unknown benchmark names should return exit code 1
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench", "nonexistent-benchmark-xyz"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("Unknown benchmark"));
}

#[test]
fn test_rusty_bench_json_gpu_capability() {
    // gpu-capability benchmark with --json should produce parseable JSON
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench", "--json", "gpu-capability"])
        .assert()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("JSON parse error: {e}\nOutput: {stdout}"));
    assert!(
        parsed.get("name").is_some() || parsed.get("success").is_some(),
        "JSON should contain 'name' or 'success' field, got: {stdout}"
    );
}

#[test]
fn test_rusty_bench_all_runs() {
    // --all should run the full benchmark suite
    let output = Command::cargo_bin(BIN)
        .unwrap()
        .args(["bench", "--json", "all"])
        .timeout(std::time::Duration::from_secs(120))
        .assert()
        .get_output()
        .clone();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("JSON parse error: {e}\nOutput: {stdout}"));
    // The "all" benchmark should produce a result with multiple sub-results
    assert!(
        parsed.get("success").is_some() || parsed.get("name").is_some(),
        "JSON should contain benchmark result fields, got: {stdout}"
    );
}
