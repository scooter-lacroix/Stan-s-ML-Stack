//! Integration tests for the update apply and verify flow.
//!
//! These tests exercise the full apply → verify pipeline with mock executors,
//! validating dependency ordering, failure isolation, and summary partitioning.

use rusty_stack::core::plan::PlanItem;
use rusty_stack::core::types::ValidationTier;
use rusty_stack::core::verification::{VerificationCheck, VerificationResult};
use rusty_stack::orchestrator::apply::{ApplyEngine, ApplyExecutor, ApplyOptions, ApplyStatus};
use rusty_stack::orchestrator::planner::{PlannerItem, UpdateClassification};
use rusty_stack::orchestrator::verify::{VerifyExecutor, VerifyRunner, VerifyStatus};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Mock Apply Executor
// ---------------------------------------------------------------------------

/// A mock executor that records calls and can be configured to fail for
/// specific components.
#[derive(Debug)]
struct MockApplyExecutor {
    /// Components that should fail.
    fail_for: HashSet<String>,
    /// Record of all apply calls, in order.
    calls: Arc<Mutex<Vec<(String, String)>>>,
}

impl MockApplyExecutor {
    fn new() -> Self {
        Self {
            fail_for: HashSet::new(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_failures(fail_ids: &[&str]) -> Self {
        Self {
            fail_for: fail_ids.iter().map(|s| s.to_string()).collect(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ApplyExecutor for MockApplyExecutor {
    fn apply_component(&self, component_id: &str, proposed_version: &str) -> Result<(), String> {
        self.calls
            .lock()
            .unwrap()
            .push((component_id.to_string(), proposed_version.to_string()));

        if self.fail_for.contains(component_id) {
            Err(format!("mock failure for '{component_id}'"))
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Mock Verify Executor
// ---------------------------------------------------------------------------

/// A mock verifier that can be configured to fail for specific components.
struct MockVerifyExecutor {
    /// Components that should fail verification.
    fail_for: HashSet<String>,
}

impl MockVerifyExecutor {
    fn new() -> Self {
        Self {
            fail_for: HashSet::new(),
        }
    }

    fn with_failures(fail_ids: &[&str]) -> Self {
        Self {
            fail_for: fail_ids.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl VerifyExecutor for MockVerifyExecutor {
    fn verify_component(&self, component_id: &str, expected_version: &str) -> VerificationResult {
        if self.fail_for.contains(component_id) {
            VerificationResult::new(
                component_id,
                vec![
                    VerificationCheck::passed("binary_exists", "found"),
                    VerificationCheck::failed(
                        "version_check",
                        format!("expected {expected_version}, got 0.0.0"),
                    ),
                ],
            )
        } else {
            VerificationResult::new(
                component_id,
                vec![
                    VerificationCheck::passed("binary_exists", "found"),
                    VerificationCheck::passed("version_check", expected_version),
                ],
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_planner_item(
    id: &str,
    current: &str,
    proposed: &str,
    deps: Vec<&str>,
    selected: bool,
    classification: UpdateClassification,
) -> PlannerItem {
    PlannerItem {
        plan_item: PlanItem::new(
            id,
            current,
            proposed,
            ValidationTier::Validated,
            selected,
            "test",
            deps.into_iter().map(|s| s.to_string()).collect(),
            true,
        ),
        classification,
        visible: true,
        selected,
        classification_reason: "test".to_string(),
        requires_hardware_check: false,
        min_rocm_version: String::new(),
    }
}

fn simple_item(id: &str, selected: bool) -> PlannerItem {
    make_planner_item(
        id,
        "1.0.0",
        "1.1.0",
        vec![],
        selected,
        UpdateClassification::Safe,
    )
}

fn item_with_deps(id: &str, deps: Vec<&str>, selected: bool) -> PlannerItem {
    make_planner_item(
        id,
        "1.0.0",
        "1.1.0",
        deps,
        selected,
        UpdateClassification::Safe,
    )
}

// ===========================================================================
// Integration Tests
// ===========================================================================

// -----------------------------------------------------------------------
// VAL-UPD-014: Apply executes in dependency-safe order
// -----------------------------------------------------------------------

#[test]
fn test_integration_dependency_order_rocm_before_pytorch() {
    let executor = MockApplyExecutor::new();
    let calls = executor.calls.clone();

    let items = vec![
        item_with_deps("pytorch", vec!["rocm"], true),
        simple_item("rocm", true),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    assert_eq!(summary.success.len(), 2);
    assert_eq!(summary.failed.len(), 0);

    // Verify execution order: rocm before pytorch
    let guard = calls.lock().unwrap();
    let call_order: Vec<&str> = guard.iter().map(|(id, _)| id.as_str()).collect();
    let rocm_pos = call_order.iter().position(|&id| id == "rocm").unwrap();
    let pytorch_pos = call_order.iter().position(|&id| id == "pytorch").unwrap();
    assert!(
        rocm_pos < pytorch_pos,
        "rocm must be applied before pytorch (rocm at {rocm_pos}, pytorch at {pytorch_pos})"
    );
}

#[test]
fn test_integration_dependency_chain_ordering() {
    let executor = MockApplyExecutor::new();
    let calls = executor.calls.clone();

    // rocm → pytorch → triton (each depends on the previous)
    let items = vec![
        item_with_deps("triton", vec!["pytorch"], true),
        item_with_deps("pytorch", vec!["rocm"], true),
        simple_item("rocm", true),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    assert_eq!(summary.success.len(), 3);

    let guard = calls.lock().unwrap();
    let call_order: Vec<&str> = guard.iter().map(|(id, _)| id.as_str()).collect();
    let rocm_pos = call_order.iter().position(|&id| id == "rocm").unwrap();
    let pytorch_pos = call_order.iter().position(|&id| id == "pytorch").unwrap();
    let triton_pos = call_order.iter().position(|&id| id == "triton").unwrap();

    assert!(rocm_pos < pytorch_pos, "rocm before pytorch");
    assert!(pytorch_pos < triton_pos, "pytorch before triton");
}

// -----------------------------------------------------------------------
// VAL-UPD-015: Failed guarded update stops dependent work
// -----------------------------------------------------------------------

#[test]
fn test_integration_failed_guarded_stops_dependents() {
    let executor = MockApplyExecutor::with_failures(&["pytorch"]);

    let items = vec![
        make_planner_item(
            "flash-attn",
            "1.0.0",
            "2.0.0",
            vec!["pytorch"],
            true,
            UpdateClassification::Guarded,
        ),
        make_planner_item(
            "pytorch",
            "2.4.0",
            "2.5.0",
            vec!["rocm"],
            true,
            UpdateClassification::Guarded,
        ),
        make_planner_item(
            "rocm",
            "7.2.0",
            "7.2.1",
            vec![],
            true,
            UpdateClassification::Safe,
        ),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    // rocm succeeded
    assert_eq!(summary.success.len(), 1);
    assert_eq!(summary.success[0].component_id, "rocm");

    // pytorch failed
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.failed[0].component_id, "pytorch");
    assert_eq!(summary.failed[0].status, ApplyStatus::Failed);

    // flash-attn blocked by pytorch's failure
    assert_eq!(summary.blocked.len(), 1);
    assert_eq!(summary.blocked[0].component_id, "flash-attn");
    assert_eq!(summary.blocked[0].status, ApplyStatus::BlockedByFailure);
}

// -----------------------------------------------------------------------
// VAL-UPD-016: Isolation-safe — unrelated work continues after failure
// -----------------------------------------------------------------------

#[test]
fn test_integration_unrelated_work_continues_after_failure() {
    let executor = MockApplyExecutor::with_failures(&["pytorch"]);

    let items = vec![
        item_with_deps("triton", vec!["pytorch"], true),
        simple_item("pytorch", true),
        simple_item("wandb", true),
        simple_item("rocm-smi", true),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    // pytorch failed
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.failed[0].component_id, "pytorch");

    // triton blocked (depends on pytorch)
    assert_eq!(summary.blocked.len(), 1);
    assert_eq!(summary.blocked[0].component_id, "triton");

    // wandb and rocm-smi succeeded (unrelated)
    assert_eq!(summary.success.len(), 2);
    let success_ids: HashSet<&str> = summary
        .success
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(success_ids.contains("wandb"));
    assert!(success_ids.contains("rocm-smi"));
}

// -----------------------------------------------------------------------
// VAL-UPD-017: Verification runs after apply for each successful component
// -----------------------------------------------------------------------

#[test]
fn test_integration_verify_after_apply_success() {
    let apply_summary = {
        let executor = MockApplyExecutor::new();
        let items = vec![
            simple_item("rocm", true),
            item_with_deps("pytorch", vec!["rocm"], true),
        ];
        let engine = ApplyEngine::new(executor);
        engine.apply(&items, &ApplyOptions::default())
    };

    assert_eq!(apply_summary.success.len(), 2);

    // Verify each successful component
    let verifier = MockVerifyExecutor::new();
    let runner = VerifyRunner::new(verifier);
    let verify_summary = runner.verify(&apply_summary);

    // Both should be verified
    assert_eq!(verify_summary.verified.len(), 2);
    assert_eq!(verify_summary.failed.len(), 0);

    for item in &verify_summary.verified {
        assert!(item.verification_result.is_some());
        assert!(item.verification_result.as_ref().unwrap().success);
    }
}

#[test]
fn test_integration_verify_detects_failure() {
    let apply_summary = {
        let executor = MockApplyExecutor::new();
        let items = vec![simple_item("rocm", true), simple_item("pytorch", true)];
        let engine = ApplyEngine::new(executor);
        engine.apply(&items, &ApplyOptions::default())
    };

    // Verify with a verifier that fails for pytorch
    let verifier = MockVerifyExecutor::with_failures(&["pytorch"]);
    let runner = VerifyRunner::new(verifier);
    let verify_summary = runner.verify(&apply_summary);

    assert_eq!(verify_summary.verified.len(), 1);
    assert_eq!(verify_summary.verified[0].component_id, "rocm");

    assert_eq!(verify_summary.failed.len(), 1);
    assert_eq!(verify_summary.failed[0].component_id, "pytorch");
    assert_eq!(
        verify_summary.failed[0].verified,
        VerifyStatus::VerifyFailed
    );
}

// -----------------------------------------------------------------------
// VAL-UPD-018: Summary shows success/failure/holdbacks with complete partitioning
// -----------------------------------------------------------------------

#[test]
fn test_integration_full_summary_partitioning() {
    let executor = MockApplyExecutor::with_failures(&["pytorch"]);

    let items = vec![
        item_with_deps("triton", vec!["pytorch"], true),
        simple_item("pytorch", true),
        simple_item("wandb", true),
        simple_item("mpi4py", false), // held back
    ];

    let engine = ApplyEngine::new(executor);
    let apply_summary = engine.apply(&items, &ApplyOptions::default());

    // Verify complete partitioning
    assert!(apply_summary.is_complete_partition());
    assert_eq!(apply_summary.total(), 4);

    // Verify buckets
    assert_eq!(apply_summary.success.len(), 1); // wandb
    assert_eq!(apply_summary.failed.len(), 1); // pytorch
    assert_eq!(apply_summary.blocked.len(), 1); // triton
    assert_eq!(apply_summary.held_back.len(), 1); // mpi4py

    // Now verify
    let verifier = MockVerifyExecutor::new();
    let runner = VerifyRunner::new(verifier);
    let verify_summary = runner.verify(&apply_summary);

    assert!(verify_summary.is_complete_partition());
    assert_eq!(verify_summary.total(), 4);
    assert_eq!(verify_summary.verified.len(), 1); // wandb verified
    assert_eq!(verify_summary.not_verified.len(), 3); // pytorch, triton, mpi4py
}

// -----------------------------------------------------------------------
// VAL-UPD-012: User can untick any preselected component
// -----------------------------------------------------------------------

#[test]
fn test_integration_user_untick_preselected() {
    let executor = MockApplyExecutor::new();
    let calls = executor.calls.clone();

    let items = vec![
        simple_item("rocm", true),
        simple_item("pytorch", false), // user unticked
        simple_item("triton", false),  // user unticked
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    // Only rocm was applied
    assert_eq!(summary.success.len(), 1);
    assert_eq!(summary.success[0].component_id, "rocm");

    // pytorch and triton held back
    assert_eq!(summary.held_back.len(), 2);
    let held_ids: HashSet<&str> = summary
        .held_back
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(held_ids.contains("pytorch"));
    assert!(held_ids.contains("triton"));

    // Only rocm was actually executed
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "rocm");
}

// -----------------------------------------------------------------------
// End-to-end: apply + verify with mixed outcomes
// -----------------------------------------------------------------------

#[test]
fn test_integration_end_to_end_apply_verify() {
    let apply_summary = {
        let executor = MockApplyExecutor::with_failures(&["deepspeed"]);
        let items = vec![
            simple_item("rocm", true),
            item_with_deps("pytorch", vec!["rocm"], true),
            item_with_deps("deepspeed", vec!["pytorch"], true),
            simple_item("wandb", true),
            simple_item("comfyui", false), // held back
        ];
        let engine = ApplyEngine::new(executor);
        engine.apply(&items, &ApplyOptions::default())
    };

    // Apply results
    assert!(apply_summary.is_complete_partition());
    assert_eq!(apply_summary.total(), 5);
    assert_eq!(apply_summary.success.len(), 3); // rocm, pytorch, wandb
    assert_eq!(apply_summary.failed.len(), 1); // deepspeed
    assert_eq!(apply_summary.held_back.len(), 1); // comfyui

    // Verify
    let verifier = MockVerifyExecutor::with_failures(&["wandb"]);
    let runner = VerifyRunner::new(verifier);
    let verify_summary = runner.verify(&apply_summary);

    assert!(verify_summary.is_complete_partition());
    assert_eq!(verify_summary.total(), 5);

    // rocm and pytorch verified
    assert_eq!(verify_summary.verified.len(), 2);
    let verified_ids: HashSet<&str> = verify_summary
        .verified
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(verified_ids.contains("rocm"));
    assert!(verified_ids.contains("pytorch"));

    // wandb failed verification
    assert_eq!(verify_summary.failed.len(), 1);
    assert_eq!(verify_summary.failed[0].component_id, "wandb");

    // deepspeed and comfyui not verified
    assert_eq!(verify_summary.not_verified.len(), 2);
    let not_verified_ids: HashSet<&str> = verify_summary
        .not_verified
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(not_verified_ids.contains("deepspeed"));
    assert!(not_verified_ids.contains("comfyui"));
}

// -----------------------------------------------------------------------
// Transitive failure propagation
// -----------------------------------------------------------------------

#[test]
fn test_integration_transitive_failure_propagation() {
    // A → B → C chain. C fails → B blocked → A blocked
    let executor = MockApplyExecutor::with_failures(&["c"]);

    let items = vec![
        item_with_deps("a", vec!["b"], true),
        item_with_deps("b", vec!["c"], true),
        simple_item("c", true),
        simple_item("unrelated", true),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.failed[0].component_id, "c");

    assert_eq!(summary.blocked.len(), 2);
    let blocked_ids: HashSet<&str> = summary
        .blocked
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(blocked_ids.contains("a"));
    assert!(blocked_ids.contains("b"));

    assert_eq!(summary.success.len(), 1);
    assert_eq!(summary.success[0].component_id, "unrelated");
}

// -----------------------------------------------------------------------
// Diamond dependency with failure at root
// -----------------------------------------------------------------------

#[test]
fn test_integration_diamond_dependency_failure_at_root() {
    let executor = MockApplyExecutor::with_failures(&["rocm"]);

    let items = vec![
        item_with_deps("flash-attn", vec!["pytorch"], true),
        item_with_deps("pytorch", vec!["rocm"], true),
        item_with_deps("triton", vec!["pytorch"], true),
        simple_item("rocm", true),
        simple_item("wandb", true),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    // rocm failed
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.failed[0].component_id, "rocm");

    // pytorch, flash-attn, triton all blocked
    assert_eq!(summary.blocked.len(), 3);
    let blocked_ids: HashSet<&str> = summary
        .blocked
        .iter()
        .map(|i| i.component_id.as_str())
        .collect();
    assert!(blocked_ids.contains("pytorch"));
    assert!(blocked_ids.contains("flash-attn"));
    assert!(blocked_ids.contains("triton"));

    // wandb succeeded (unrelated)
    assert_eq!(summary.success.len(), 1);
    assert_eq!(summary.success[0].component_id, "wandb");
}

// -----------------------------------------------------------------------
// All items deselected — no-op
// -----------------------------------------------------------------------

#[test]
fn test_integration_all_deselected_is_noop() {
    let executor = MockApplyExecutor::new();
    let calls = executor.calls.clone();

    let items = vec![
        simple_item("rocm", false),
        simple_item("pytorch", false),
        simple_item("triton", false),
    ];

    let engine = ApplyEngine::new(executor);
    let summary = engine.apply(&items, &ApplyOptions::default());

    assert_eq!(summary.success.len(), 0);
    assert_eq!(summary.failed.len(), 0);
    assert_eq!(summary.held_back.len(), 3);

    // No actual apply calls
    assert!(calls.lock().unwrap().is_empty());
}
