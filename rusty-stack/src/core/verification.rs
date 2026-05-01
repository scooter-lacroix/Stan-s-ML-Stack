//! Post-apply verification result types.
//!
//! This module implements:
//! - [`VerificationCheck`] — a single named check with status and message.
//! - [`VerificationResult`] — aggregate result for a component, containing
//!   multiple checks and a computed `success` field.
//!
//! # Schema Completeness (VAL-CORE-016)
//!
//! A `VerificationResult` must contain:
//! - `component_id` — identifies the verified component
//! - `success` — `true` if all checks passed or only warned
//! - `checks` — list of individual check results
//!
//! Success rules:
//! - All checks pass → `success: true`
//! - Any check fails → `success: false`
//! - Warnings only → `success: true`

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// CheckStatus
// ---------------------------------------------------------------------------

/// Status of a single verification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Passed,
    Warning,
    Failed,
}

impl CheckStatus {
    pub fn label(self) -> &'static str {
        match self {
            CheckStatus::Passed => "passed",
            CheckStatus::Warning => "warning",
            CheckStatus::Failed => "failed",
        }
    }
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// VerificationCheck
// ---------------------------------------------------------------------------

/// A single verification check within a component's verification result.
///
/// Each check has a name (e.g., "binary_exists", "version_matches"),
/// a status (passed/warning/failed), and an optional message with details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationCheck {
    /// Name of the check (e.g., "binary_exists").
    pub name: String,
    /// Result status of this check.
    pub status: CheckStatus,
    /// Human-readable details about the check outcome.
    pub message: String,
}

impl VerificationCheck {
    /// Create a passing check.
    pub fn passed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Passed,
            message: message.into(),
        }
    }

    /// Create a warning check.
    pub fn warning(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Warning,
            message: message.into(),
        }
    }

    /// Create a failing check.
    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Failed,
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Aggregate verification result for a single component.
///
/// Contains the component ID, a list of individual checks, and a computed
/// `success` field derived from the check statuses:
/// - `success: true` if all checks passed or only warned
/// - `success: false` if any check failed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationResult {
    /// Component that was verified.
    pub component_id: String,
    /// Whether verification succeeded overall.
    pub success: bool,
    /// Individual check results.
    pub checks: Vec<VerificationCheck>,
}

impl VerificationResult {
    /// Create a new verification result from individual checks.
    ///
    /// The `success` field is automatically computed:
    /// - `true` if no check has `CheckStatus::Failed`
    /// - `false` if any check failed
    pub fn new(component_id: impl Into<String>, checks: Vec<VerificationCheck>) -> Self {
        let success = !checks.iter().any(|c| c.status == CheckStatus::Failed);
        Self {
            component_id: component_id.into(),
            success,
            checks,
        }
    }

    /// Create a successful result with no checks.
    pub fn ok(component_id: impl Into<String>) -> Self {
        Self {
            component_id: component_id.into(),
            success: true,
            checks: vec![],
        }
    }

    /// Create a failed result with a single check.
    pub fn fail(component_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            component_id: component_id.into(),
            success: false,
            checks: vec![VerificationCheck::failed("verification", reason)],
        }
    }

    /// Whether all checks passed (no warnings, no failures).
    pub fn is_all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.status == CheckStatus::Passed)
    }

    /// Whether there are any warnings.
    pub fn has_warnings(&self) -> bool {
        self.checks.iter().any(|c| c.status == CheckStatus::Warning)
    }

    /// Number of checks.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-CORE-016: VerificationResult schema completeness
    // =======================================================================

    #[test]
    fn test_verification_result_all_pass_success() {
        let result = VerificationResult::new(
            "pytorch",
            vec![
                VerificationCheck::passed("binary_exists", "Found at /usr/bin/python"),
                VerificationCheck::passed("version_matches", "2.2.0 installed"),
                VerificationCheck::passed("import_test", "import torch succeeded"),
            ],
        );
        assert!(result.success, "All-pass should have success=true");
        assert!(result.is_all_passed());
        assert!(!result.has_warnings());
        assert_eq!(result.check_count(), 3);
    }

    #[test]
    fn test_verification_result_one_failure_not_success() {
        let result = VerificationResult::new(
            "triton",
            vec![
                VerificationCheck::passed("binary_exists", "Found"),
                VerificationCheck::failed("import_test", "import triton failed with ImportError"),
            ],
        );
        assert!(!result.success, "Any failure should have success=false");
        assert!(!result.is_all_passed());
    }

    #[test]
    fn test_verification_result_warnings_only_success() {
        let result = VerificationResult::new(
            "rocm",
            vec![
                VerificationCheck::passed("rocm_detect", "ROCm found at /opt/rocm"),
                VerificationCheck::warning("version_old", "ROCm 6.4 is not the latest"),
            ],
        );
        assert!(result.success, "Warnings-only should have success=true");
        assert!(result.has_warnings());
        assert!(!result.is_all_passed());
    }

    #[test]
    fn test_verification_result_ok_empty() {
        let result = VerificationResult::ok("empty-comp");
        assert!(result.success);
        assert!(result.checks.is_empty());
        assert_eq!(result.check_count(), 0);
    }

    #[test]
    fn test_verification_result_fail_single() {
        let result = VerificationResult::fail("broken-comp", "installation failed");
        assert!(!result.success);
        assert_eq!(result.checks.len(), 1);
        assert_eq!(result.checks[0].status, CheckStatus::Failed);
        assert_eq!(result.checks[0].name, "verification");
    }

    // =======================================================================
    // CheckStatus
    // =======================================================================

    #[test]
    fn test_check_status_serde_roundtrip() {
        let statuses = [
            CheckStatus::Passed,
            CheckStatus::Warning,
            CheckStatus::Failed,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: CheckStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
    }

    #[test]
    fn test_check_status_labels() {
        assert_eq!(CheckStatus::Passed.label(), "passed");
        assert_eq!(CheckStatus::Warning.label(), "warning");
        assert_eq!(CheckStatus::Failed.label(), "failed");
    }

    #[test]
    fn test_check_status_display() {
        assert_eq!(format!("{}", CheckStatus::Passed), "passed");
        assert_eq!(format!("{}", CheckStatus::Warning), "warning");
        assert_eq!(format!("{}", CheckStatus::Failed), "failed");
    }

    // =======================================================================
    // VerificationCheck
    // =======================================================================

    #[test]
    fn test_verification_check_constructors() {
        let passed = VerificationCheck::passed("test", "all good");
        assert_eq!(passed.status, CheckStatus::Passed);

        let warning = VerificationCheck::warning("test", "be careful");
        assert_eq!(warning.status, CheckStatus::Warning);

        let failed = VerificationCheck::failed("test", "broke");
        assert_eq!(failed.status, CheckStatus::Failed);
    }

    // =======================================================================
    // Serde roundtrip
    // =======================================================================

    #[test]
    fn test_verification_check_serde_roundtrip() {
        let check = VerificationCheck::failed("import_test", "ModuleNotFoundError");
        let json = serde_json::to_string(&check).unwrap();
        let back: VerificationCheck = serde_json::from_str(&json).unwrap();
        assert_eq!(check, back);
    }

    #[test]
    fn test_verification_result_serde_roundtrip() {
        let result = VerificationResult::new(
            "pytorch",
            vec![
                VerificationCheck::passed("binary", "found"),
                VerificationCheck::warning("version", "not latest"),
            ],
        );
        let json = serde_json::to_string(&result).unwrap();
        let back: VerificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn test_verification_result_serializes_to_json() {
        let result = VerificationResult::new(
            "rocm",
            vec![
                VerificationCheck::passed("rocm_path", "/opt/rocm exists"),
                VerificationCheck::passed("version", "7.2.1"),
            ],
        );
        let json = serde_json::to_string(&result).unwrap();
        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["component_id"], "rocm");
        assert_eq!(parsed["success"], true);
        assert!(parsed["checks"].is_array());
        assert_eq!(parsed["checks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_verification_result_failure_serializes_correctly() {
        let result = VerificationResult::new(
            "triton",
            vec![VerificationCheck::failed("import", "failed")],
        );
        let json = serde_json::to_string(&result).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["success"], false);
    }
}
