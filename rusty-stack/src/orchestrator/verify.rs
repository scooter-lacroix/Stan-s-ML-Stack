//! Post-apply verification orchestration.
//!
//! This module implements:
//! - [`VerifyRunner`] — runs verification checks for each successfully-applied component.
//! - [`VerifySummary`] — partitions results into verified / failed / not-verified.
//!
//! # Verification Flow (VAL-UPD-017)
//!
//! After the apply phase completes, verification runs for each component
//! that was applied successfully. Each verification produces a
//! [`VerificationResult`](crate::core::verification::VerificationResult).
//!
//! # Summary Partitioning (VAL-UPD-018)
//!
//! The final summary partitions every component into exactly one bucket:
//! - `verified` — applied and verified successfully
//! - `failed` — applied but verification failed
//! - `not_verified` — not applied (held back, blocked, or skipped)

use crate::core::verification::{VerificationCheck, VerificationResult};
use crate::orchestrator::apply::{ApplyStatus, ApplySummary};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Verify Executor Trait
// ---------------------------------------------------------------------------

/// Trait for verifying a single component after apply.
///
/// Implementations can check binary existence, version matches,
/// import tests, etc.
pub trait VerifyExecutor: Send + Sync {
    /// Verify a single component.
    ///
    /// Returns a [`VerificationResult`] with individual checks.
    fn verify_component(&self, component_id: &str, expected_version: &str) -> VerificationResult;
}

// ---------------------------------------------------------------------------
// Default Verify Executor
// ---------------------------------------------------------------------------

/// A no-op verifier that always returns success.
/// Used as a default when no real verifier is configured.
pub struct NoOpVerifyExecutor;

impl VerifyExecutor for NoOpVerifyExecutor {
    fn verify_component(&self, component_id: &str, _expected_version: &str) -> VerificationResult {
        VerificationResult::new(
            component_id,
            vec![VerificationCheck::passed(
                "default_check",
                "no-op verification passed",
            )],
        )
    }
}

// ---------------------------------------------------------------------------
// Verify Item
// ---------------------------------------------------------------------------

/// Verification result for a single component, with apply context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerifyItem {
    /// Component ID.
    pub component_id: String,
    /// The apply status that preceded verification.
    pub apply_status: ApplyStatus,
    /// The verification result (None if not verified).
    pub verification_result: Option<VerificationResult>,
    /// Overall verification status.
    pub verified: VerifyStatus,
}

/// Overall verification status for a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifyStatus {
    /// Successfully verified after apply.
    Verified,
    /// Verification failed after apply.
    VerifyFailed,
    /// Not verified (was not applied).
    NotVerified,
}

impl VerifyStatus {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            VerifyStatus::Verified => "verified",
            VerifyStatus::VerifyFailed => "verify_failed",
            VerifyStatus::NotVerified => "not_verified",
        }
    }
}

impl fmt::Display for VerifyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Verify Summary
// ---------------------------------------------------------------------------

/// Summary of the verification phase, partitioning all components.
///
/// Every component from the apply phase appears in exactly one bucket:
/// - `verified` — applied successfully and verification passed
/// - `failed` — applied but verification failed
/// - `not_verified` — not applied (held back, blocked, skipped, or apply-failed)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerifySummary {
    /// Components that were verified successfully.
    pub verified: Vec<VerifyItem>,
    /// Components that failed verification.
    pub failed: Vec<VerifyItem>,
    /// Components that were not verified (not applied).
    pub not_verified: Vec<VerifyItem>,
}

impl VerifySummary {
    /// Create an empty summary.
    pub fn new() -> Self {
        Self {
            verified: Vec::new(),
            failed: Vec::new(),
            not_verified: Vec::new(),
        }
    }

    /// Total items across all buckets.
    pub fn total(&self) -> usize {
        self.verified.len() + self.failed.len() + self.not_verified.len()
    }

    /// Check that every component is in exactly one bucket.
    pub fn is_complete_partition(&self) -> bool {
        let all_ids: HashSet<&str> = self
            .verified
            .iter()
            .chain(self.failed.iter())
            .chain(self.not_verified.iter())
            .map(|i| i.component_id.as_str())
            .collect();

        let total_count = self.verified.len() + self.failed.len() + self.not_verified.len();
        all_ids.len() == total_count
    }
}

impl Default for VerifySummary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Verify Runner
// ---------------------------------------------------------------------------

/// Runs post-apply verification for each successfully-applied component.
///
/// # Algorithm
///
/// 1. For each item in the apply summary:
///    - If `Success`: run verification, classify as verified or verify_failed
///    - If `Failed` / `BlockedByFailure` / `HeldBack` / `Skipped`: mark as not_verified
/// 2. Produce a `VerifySummary` with complete partitioning.
pub struct VerifyRunner {
    /// The verifier to use for each component.
    executor: Box<dyn VerifyExecutor>,
}

impl VerifyRunner {
    /// Create a new verify runner with the given executor.
    pub fn new(executor: impl VerifyExecutor + 'static) -> Self {
        Self {
            executor: Box::new(executor),
        }
    }

    /// Create a new verify runner with the default no-op executor.
    pub fn new_noop() -> Self {
        Self::new(NoOpVerifyExecutor)
    }

    /// Run verification for all items in the apply summary.
    ///
    /// Returns a `VerifySummary` with complete partitioning.
    pub fn verify(&self, apply_summary: &ApplySummary) -> VerifySummary {
        let mut summary = VerifySummary::new();

        // Process successful applies — run verification
        for item in &apply_summary.success {
            let result = self
                .executor
                .verify_component(&item.component_id, &item.proposed_version);

            let verified = result.success;

            let verify_item = VerifyItem {
                component_id: item.component_id.clone(),
                apply_status: ApplyStatus::Success,
                verification_result: Some(result),
                verified: if verified {
                    VerifyStatus::Verified
                } else {
                    VerifyStatus::VerifyFailed
                },
            };

            if verified {
                summary.verified.push(verify_item);
            } else {
                summary.failed.push(verify_item);
            }
        }

        // Process failed, blocked, held_back, skipped — no verification
        for item in apply_summary
            .failed
            .iter()
            .chain(apply_summary.blocked.iter())
            .chain(apply_summary.held_back.iter())
            .chain(apply_summary.skipped.iter())
        {
            summary.not_verified.push(VerifyItem {
                component_id: item.component_id.clone(),
                apply_status: item.status,
                verification_result: None,
                verified: VerifyStatus::NotVerified,
            });
        }

        summary
    }
}

// Need HashSet for is_complete_partition
use std::collections::HashSet;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::verification::{CheckStatus, VerificationCheck};
    use crate::orchestrator::apply::ApplyStatus;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_apply_summary(
        success_ids: &[&str],
        failed_ids: &[&str],
        held_back_ids: &[&str],
        blocked_ids: &[&str],
    ) -> ApplySummary {
        use crate::orchestrator::apply::ApplyItem;
        let mut summary = ApplySummary::new();
        for &id in success_ids {
            summary
                .success
                .push(ApplyItem::new(id, "1.0.0", "1.1.0", ApplyStatus::Success));
        }
        for &id in failed_ids {
            let mut item = ApplyItem::new(id, "1.0.0", "1.1.0", ApplyStatus::Failed);
            item.error_message = "test failure".to_string();
            summary.failed.push(item);
        }
        for &id in held_back_ids {
            let mut item = ApplyItem::new(id, "1.0.0", "1.1.0", ApplyStatus::HeldBack);
            item.was_selected = false;
            summary.held_back.push(item);
        }
        for &id in blocked_ids {
            let mut item = ApplyItem::new(id, "1.0.0", "1.1.0", ApplyStatus::BlockedByFailure);
            item.error_message = "blocked by dependency".to_string();
            summary.blocked.push(item);
        }
        summary
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-017: Verification runs after apply
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_runs_for_successful_applies() {
        let apply_summary = make_apply_summary(&["a", "b", "c"], &[], &[], &[]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        // All successful applies should be verified
        assert_eq!(verify_summary.verified.len(), 3);
        assert_eq!(verify_summary.failed.len(), 0);
        assert_eq!(verify_summary.not_verified.len(), 0);

        // Each verified item should have a verification result
        for item in &verify_summary.verified {
            assert!(item.verification_result.is_some());
            assert!(item.verification_result.as_ref().unwrap().success);
        }
    }

    #[test]
    fn test_verify_not_run_for_failed_applies() {
        let apply_summary = make_apply_summary(&[], &["a"], &[], &[]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        assert_eq!(verify_summary.verified.len(), 0);
        assert_eq!(verify_summary.not_verified.len(), 1);
        assert_eq!(verify_summary.not_verified[0].component_id, "a");
        assert_eq!(
            verify_summary.not_verified[0].verified,
            VerifyStatus::NotVerified
        );
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-018: Summary shows success/failure/holdbacks
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_summary_complete_partitioning() {
        let apply_summary = make_apply_summary(&["a", "b"], &["c"], &["d"], &["e"]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        assert!(verify_summary.is_complete_partition());
        assert_eq!(verify_summary.total(), 5);
        assert_eq!(verify_summary.verified.len(), 2);
        assert_eq!(verify_summary.failed.len(), 0);
        assert_eq!(verify_summary.not_verified.len(), 3);
    }

    #[test]
    fn test_verify_summary_partition_no_overlaps() {
        let apply_summary = make_apply_summary(&["a"], &["b"], &["c"], &["d"]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        let verified_ids: HashSet<&str> = verify_summary
            .verified
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        let failed_ids: HashSet<&str> = verify_summary
            .failed
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        let not_verified_ids: HashSet<&str> = verify_summary
            .not_verified
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();

        // No overlaps
        assert!(verified_ids.is_disjoint(&failed_ids));
        assert!(verified_ids.is_disjoint(&not_verified_ids));
        assert!(failed_ids.is_disjoint(&not_verified_ids));

        // Union equals all items
        let all: HashSet<&str> = verified_ids
            .union(&failed_ids)
            .cloned()
            .collect::<HashSet<&str>>()
            .union(&not_verified_ids)
            .cloned()
            .collect();
        assert_eq!(all.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Verification failure case
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_failure_after_successful_apply() {
        // Custom verifier that fails for component "b"
        struct FailForB;
        impl VerifyExecutor for FailForB {
            fn verify_component(
                &self,
                component_id: &str,
                _expected_version: &str,
            ) -> VerificationResult {
                if component_id == "b" {
                    VerificationResult::new(
                        component_id,
                        vec![VerificationCheck::failed("import_test", "import failed")],
                    )
                } else {
                    VerificationResult::new(
                        component_id,
                        vec![VerificationCheck::passed("ok", "ok")],
                    )
                }
            }
        }

        let apply_summary = make_apply_summary(&["a", "b", "c"], &[], &[], &[]);
        let runner = VerifyRunner::new(FailForB);
        let verify_summary = runner.verify(&apply_summary);

        assert_eq!(verify_summary.verified.len(), 2);
        assert_eq!(verify_summary.failed.len(), 1);
        assert_eq!(verify_summary.failed[0].component_id, "b");
        assert_eq!(
            verify_summary.failed[0].verified,
            VerifyStatus::VerifyFailed
        );
        assert!(verify_summary.failed[0]
            .verification_result
            .as_ref()
            .unwrap()
            .checks
            .iter()
            .any(|c| c.status == CheckStatus::Failed));
    }

    // -----------------------------------------------------------------------
    // Empty apply summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_empty_apply_summary() {
        let apply_summary = ApplySummary::new();
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        assert_eq!(verify_summary.total(), 0);
        assert!(verify_summary.is_complete_partition());
    }

    // -----------------------------------------------------------------------
    // VerifyStatus serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_status_serde_roundtrip() {
        let statuses = [
            VerifyStatus::Verified,
            VerifyStatus::VerifyFailed,
            VerifyStatus::NotVerified,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: VerifyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
    }

    // -----------------------------------------------------------------------
    // VerifySummary serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_summary_serde_roundtrip() {
        let apply_summary = make_apply_summary(&["a"], &["b"], &["c"], &[]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        let json = serde_json::to_string(&verify_summary).unwrap();
        let back: VerifySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(verify_summary, back);
    }

    // -----------------------------------------------------------------------
    // VerifyItem serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_item_serde_roundtrip() {
        let item = VerifyItem {
            component_id: "pytorch".to_string(),
            apply_status: ApplyStatus::Success,
            verification_result: Some(VerificationResult::new(
                "pytorch",
                vec![VerificationCheck::passed("import", "ok")],
            )),
            verified: VerifyStatus::Verified,
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: VerifyItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }

    // -----------------------------------------------------------------------
    // VerifyStatus display
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_status_display() {
        assert_eq!(format!("{}", VerifyStatus::Verified), "verified");
        assert_eq!(format!("{}", VerifyStatus::VerifyFailed), "verify_failed");
        assert_eq!(format!("{}", VerifyStatus::NotVerified), "not_verified");
    }

    // -----------------------------------------------------------------------
    // Full flow: apply → verify → summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_apply_verify_flow() {
        // Simulate: a succeeds, b fails, c held back
        let apply_summary = make_apply_summary(&["a"], &["b"], &["c"], &[]);
        let runner = VerifyRunner::new_noop();
        let verify_summary = runner.verify(&apply_summary);

        // a verified
        assert_eq!(verify_summary.verified.len(), 1);
        assert_eq!(verify_summary.verified[0].component_id, "a");

        // b and c not verified
        assert_eq!(verify_summary.not_verified.len(), 2);
        let not_verified_ids: HashSet<&str> = verify_summary
            .not_verified
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        assert!(not_verified_ids.contains("b"));
        assert!(not_verified_ids.contains("c"));

        // Complete partition
        assert!(verify_summary.is_complete_partition());
        assert_eq!(verify_summary.total(), 3);
    }
}
