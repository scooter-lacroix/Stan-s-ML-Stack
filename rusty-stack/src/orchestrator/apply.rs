//! Apply engine — dependency-safe execution of selected update items.
//!
//! This module implements:
//! - [`ApplyStatus`] — per-item outcome (success, failed, blocked-by-failure, held-back, skipped)
//! - [`ApplyItem`] — an item in the apply queue with execution status
//! - [`ApplyEngine`] — executes selected items in dependency order, batches safe items,
//!   isolates failures so unrelated work continues
//!
//! # Execution Model
//!
//! 1. Items are sorted in dependency-safe order (topological sort).
//! 2. Items with `isolation_safe: true` and no failed dependencies may be batched.
//! 3. When a guarded item fails, all items that depend on it are marked
//!    `BlockedByFailure` — but unrelated items continue.
//! 4. Items that were not selected by the user are marked `HeldBack`.
//!
//! # Failure Isolation (VAL-UPD-015, VAL-UPD-016)
//!
//! If component B fails and A depends on B:
//! - A is marked `BlockedByFailure` with a reference to B's failure.
//! - Unrelated component C (no dependency on B) continues and succeeds/fails
//!   independently.

use crate::core::plan::PlanItem;
use crate::orchestrator::planner::PlannerItem;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// ApplyStatus
// ---------------------------------------------------------------------------

/// Outcome status for a single item after the apply phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApplyStatus {
    /// Item was applied successfully.
    Success,
    /// Item failed during apply.
    Failed,
    /// Item was blocked because a dependency failed.
    BlockedByFailure,
    /// Item was held back (user deselected or not in scope).
    HeldBack,
    /// Item was skipped (not selected, not eligible).
    Skipped,
}

impl ApplyStatus {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            ApplyStatus::Success => "success",
            ApplyStatus::Failed => "failed",
            ApplyStatus::BlockedByFailure => "blocked-by-failure",
            ApplyStatus::HeldBack => "held-back",
            ApplyStatus::Skipped => "skipped",
        }
    }
}

impl fmt::Display for ApplyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// ApplyItem
// ---------------------------------------------------------------------------

/// An item that has been processed by the apply engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApplyItem {
    /// Component ID.
    pub component_id: String,
    /// Current version before apply.
    pub current_version: String,
    /// Version that was applied (or attempted).
    pub proposed_version: String,
    /// Outcome of the apply operation.
    pub status: ApplyStatus,
    /// Error message if failed or blocked.
    pub error_message: String,
    /// Component IDs that this item depends on.
    pub dependencies: Vec<String>,
    /// Whether this item was selected for apply.
    pub was_selected: bool,
}

impl ApplyItem {
    /// Create a new apply item with the given status.
    pub fn new(
        component_id: impl Into<String>,
        current_version: impl Into<String>,
        proposed_version: impl Into<String>,
        status: ApplyStatus,
    ) -> Self {
        Self {
            component_id: component_id.into(),
            current_version: current_version.into(),
            proposed_version: proposed_version.into(),
            status,
            error_message: String::new(),
            dependencies: Vec::new(),
            was_selected: true,
        }
    }

    /// Create a held-back item (user deselected).
    pub fn held_back(item: &PlannerItem) -> Self {
        Self {
            component_id: item.plan_item.component_id.clone(),
            current_version: item.plan_item.current_version.clone(),
            proposed_version: item.plan_item.proposed_version.clone(),
            status: ApplyStatus::HeldBack,
            error_message: String::new(),
            dependencies: item.plan_item.dependencies.clone(),
            was_selected: false,
        }
    }

    /// Create a blocked-by-failure item.
    pub fn blocked_by_failure(item: &PlannerItem, blocker_id: &str, reason: &str) -> Self {
        Self {
            component_id: item.plan_item.component_id.clone(),
            current_version: item.plan_item.current_version.clone(),
            proposed_version: item.plan_item.proposed_version.clone(),
            status: ApplyStatus::BlockedByFailure,
            error_message: format!("blocked by failed dependency '{blocker_id}': {reason}"),
            dependencies: item.plan_item.dependencies.clone(),
            was_selected: true,
        }
    }

    /// Create a skipped item.
    pub fn skipped(item: &PlannerItem) -> Self {
        Self {
            component_id: item.plan_item.component_id.clone(),
            current_version: item.plan_item.current_version.clone(),
            proposed_version: item.plan_item.proposed_version.clone(),
            status: ApplyStatus::Skipped,
            error_message: String::new(),
            dependencies: item.plan_item.dependencies.clone(),
            was_selected: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Apply Executor Trait
// ---------------------------------------------------------------------------

/// Trait for executing a single component update.
///
/// Implementations can invoke shell scripts, Rust-native installers,
/// or be mocked for testing.
pub trait ApplyExecutor: Send + Sync {
    /// Apply a single component update.
    ///
    /// Returns `Ok(())` on success, `Err` with a descriptive message on failure.
    fn apply_component(&self, component_id: &str, proposed_version: &str) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// Default Executor (no-op for testing, real impl would call scripts)
// ---------------------------------------------------------------------------

/// A no-op executor that always succeeds.
/// Used as a default when no real executor is configured.
pub struct NoOpExecutor;

impl ApplyExecutor for NoOpExecutor {
    fn apply_component(&self, _component_id: &str, _proposed_version: &str) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Apply Options
// ---------------------------------------------------------------------------

/// Options controlling apply behavior.
#[derive(Debug, Clone, Default)]
pub struct ApplyOptions {
    /// Component IDs that should be forced to fail (for testing).
    pub force_fail: HashSet<String>,
}

// ---------------------------------------------------------------------------
// Apply Summary
// ---------------------------------------------------------------------------

/// Summary of the apply phase, partitioning all components into
/// success / failed / held_back buckets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApplySummary {
    /// Items that were applied successfully.
    pub success: Vec<ApplyItem>,
    /// Items that failed during apply.
    pub failed: Vec<ApplyItem>,
    /// Items held back (not selected by user).
    pub held_back: Vec<ApplyItem>,
    /// Items blocked by a failed dependency.
    pub blocked: Vec<ApplyItem>,
    /// Items skipped (not eligible).
    pub skipped: Vec<ApplyItem>,
}

impl ApplySummary {
    /// Create an empty summary.
    pub fn new() -> Self {
        Self {
            success: Vec::new(),
            failed: Vec::new(),
            held_back: Vec::new(),
            blocked: Vec::new(),
            skipped: Vec::new(),
        }
    }

    /// Total number of items across all buckets.
    pub fn total(&self) -> usize {
        self.success.len()
            + self.failed.len()
            + self.held_back.len()
            + self.blocked.len()
            + self.skipped.len()
    }

    /// Whether any item failed.
    pub fn has_failures(&self) -> bool {
        !self.failed.is_empty() || !self.blocked.is_empty()
    }

    /// Check that every component is in exactly one bucket (complete partitioning).
    pub fn is_complete_partition(&self) -> bool {
        let all_ids: HashSet<&str> = self
            .success
            .iter()
            .chain(self.failed.iter())
            .chain(self.held_back.iter())
            .chain(self.blocked.iter())
            .chain(self.skipped.iter())
            .map(|i| i.component_id.as_str())
            .collect();

        let total_count = self.success.len()
            + self.failed.len()
            + self.held_back.len()
            + self.blocked.len()
            + self.skipped.len();

        all_ids.len() == total_count
    }
}

impl Default for ApplySummary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Apply Engine
// ---------------------------------------------------------------------------

/// Executes selected update items in dependency-safe order with failure isolation.
///
/// # Algorithm
///
/// 1. Separate selected items from deselected (held back).
/// 2. Sort selected items in dependency order (topological sort).
/// 3. Execute each item:
///    a. Check if any dependency has failed → mark `BlockedByFailure`.
///    b. Execute via the configured [`ApplyExecutor`].
///    c. Record success or failure.
/// 4. Unrelated items continue even if some fail.
pub struct ApplyEngine {
    /// The executor that applies individual components.
    executor: Box<dyn ApplyExecutor>,
}

impl ApplyEngine {
    /// Create a new apply engine with the given executor.
    pub fn new(executor: impl ApplyExecutor + 'static) -> Self {
        Self {
            executor: Box::new(executor),
        }
    }

    /// Create a new apply engine with the default no-op executor.
    pub fn new_noop() -> Self {
        Self::new(NoOpExecutor)
    }

    /// Apply the given planner items.
    ///
    /// Items that are not selected are held back. Selected items are executed
    /// in dependency-safe order with failure isolation.
    pub fn apply(&self, items: &[PlannerItem], options: &ApplyOptions) -> ApplySummary {
        let mut summary = ApplySummary::new();

        // Separate selected from deselected
        let (selected, deselected): (Vec<&PlannerItem>, Vec<&PlannerItem>) =
            items.iter().partition(|i| i.selected);

        // Mark deselected items as held back
        for item in &deselected {
            summary.held_back.push(ApplyItem::held_back(item));
        }

        // Sort selected items in dependency order
        let sorted = match dependency_sort(selected.clone()) {
            Ok(sorted) => sorted,
            Err(_) => {
                // If we can't sort due to cycles, fail all selected items
                for item in &selected {
                    let mut apply_item = ApplyItem::new(
                        &item.plan_item.component_id,
                        &item.plan_item.current_version,
                        &item.plan_item.proposed_version,
                        ApplyStatus::Failed,
                    );
                    apply_item.error_message = "dependency cycle detected".to_string();
                    apply_item.dependencies = item.plan_item.dependencies.clone();
                    summary.failed.push(apply_item);
                }
                return summary;
            }
        };

        // Track failed component IDs for dependency blocking
        let mut failed_ids: HashSet<String> = HashSet::new();
        let mut failure_reasons: HashMap<String, String> = HashMap::new();

        // Execute in order
        for item in &sorted {
            // Check if any dependency has failed
            let blocked_by = self.find_failed_dependency(&item.plan_item, &failed_ids);

            if let Some(blocker_id) = blocked_by {
                let reason = failure_reasons
                    .get(&blocker_id)
                    .cloned()
                    .unwrap_or_default();
                summary
                    .blocked
                    .push(ApplyItem::blocked_by_failure(item, &blocker_id, &reason));
                // Also mark as failed for downstream dependents
                failed_ids.insert(item.plan_item.component_id.clone());
                failure_reasons.insert(
                    item.plan_item.component_id.clone(),
                    format!("dependency '{blocker_id}' failed"),
                );
                continue;
            }

            // Apply the component
            let status = if options.force_fail.contains(&item.plan_item.component_id) {
                Err(format!(
                    "forced failure for component '{}'",
                    item.plan_item.component_id
                ))
            } else {
                self.executor.apply_component(
                    &item.plan_item.component_id,
                    &item.plan_item.proposed_version,
                )
            };

            match status {
                Ok(()) => {
                    let mut apply_item = ApplyItem::new(
                        &item.plan_item.component_id,
                        &item.plan_item.current_version,
                        &item.plan_item.proposed_version,
                        ApplyStatus::Success,
                    );
                    apply_item.dependencies = item.plan_item.dependencies.clone();
                    summary.success.push(apply_item);
                }
                Err(err) => {
                    let mut apply_item = ApplyItem::new(
                        &item.plan_item.component_id,
                        &item.plan_item.current_version,
                        &item.plan_item.proposed_version,
                        ApplyStatus::Failed,
                    );
                    apply_item.error_message = err.clone();
                    apply_item.dependencies = item.plan_item.dependencies.clone();
                    summary.failed.push(apply_item);

                    failed_ids.insert(item.plan_item.component_id.clone());
                    failure_reasons.insert(item.plan_item.component_id.clone(), err);
                }
            }
        }

        summary
    }

    /// Find a failed dependency for the given item.
    ///
    /// Returns the first failed dependency ID, if any.
    fn find_failed_dependency(
        &self,
        item: &PlanItem,
        failed_ids: &HashSet<String>,
    ) -> Option<String> {
        item.dependencies
            .iter()
            .find(|dep| failed_ids.contains(*dep))
            .cloned()
    }
}

// ---------------------------------------------------------------------------
// Dependency Sort
// ---------------------------------------------------------------------------

/// Sort planner items in dependency-safe execution order.
///
/// Uses topological sort (Kahn's algorithm). Returns an error if a cycle
/// is detected.
fn dependency_sort(items: Vec<&PlannerItem>) -> Result<Vec<&PlannerItem>, String> {
    let n = items.len();
    if n == 0 {
        return Ok(items);
    }

    let id_to_idx: HashMap<&str, usize> = items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.plan_item.component_id.as_str(), i))
        .collect();

    // Build adjacency list and in-degree counts
    let mut in_degree: Vec<usize> = vec![0; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (idx, item) in items.iter().enumerate() {
        for dep_id in &item.plan_item.dependencies {
            if let Some(&dep_idx) = id_to_idx.get(dep_id.as_str()) {
                adj[dep_idx].push(idx);
                in_degree[idx] += 1;
            }
        }
    }

    // BFS
    let mut queue = std::collections::VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut sorted_indices = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        sorted_indices.push(idx);
        for &neighbor in &adj[idx] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if sorted_indices.len() != n {
        return Err("circular dependency detected".to_string());
    }

    // Reorder
    let mut sorted: Vec<&PlannerItem> = Vec::with_capacity(n);
    for idx in sorted_indices {
        sorted.push(items[idx]);
    }

    Ok(sorted)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::plan::PlanItem;
    use crate::core::types::ValidationTier;
    use crate::orchestrator::planner::{PlannerItem, UpdateClassification};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_plan_item(id: &str, current: &str, proposed: &str, deps: Vec<&str>) -> PlanItem {
        PlanItem::new(
            id,
            current,
            proposed,
            ValidationTier::Validated,
            true,
            "test",
            deps.into_iter().map(|s| s.to_string()).collect(),
            true,
        )
    }

    fn make_planner_item(
        id: &str,
        current: &str,
        proposed: &str,
        deps: Vec<&str>,
        selected: bool,
    ) -> PlannerItem {
        PlannerItem {
            plan_item: make_plan_item(id, current, proposed, deps),
            classification: UpdateClassification::Safe,
            visible: true,
            selected,
            classification_reason: "test".to_string(),
            requires_hardware_check: false,
            min_rocm_version: String::new(),
        }
    }

    fn make_planner_item_with_classification(
        id: &str,
        current: &str,
        proposed: &str,
        deps: Vec<&str>,
        selected: bool,
        classification: UpdateClassification,
    ) -> PlannerItem {
        PlannerItem {
            plan_item: make_plan_item(id, current, proposed, deps),
            classification,
            visible: true,
            selected,
            classification_reason: "test".to_string(),
            requires_hardware_check: false,
            min_rocm_version: String::new(),
        }
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-014: Apply executes in dependency-safe order
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_dependency_safe_order_b_before_a() {
        // A depends on B → B must complete before A starts
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec!["b"], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], true),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions::default();
        let summary = engine.apply(&items, &options);

        // Both should succeed
        assert_eq!(summary.success.len(), 2);
        assert_eq!(summary.failed.len(), 0);

        // Verify B appears before A in the success list
        let order: Vec<&str> = summary
            .success
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        let b_pos = order.iter().position(|&id| id == "b").unwrap();
        let a_pos = order.iter().position(|&id| id == "a").unwrap();
        assert!(
            b_pos < a_pos,
            "B should be applied before A (B at {b_pos}, A at {a_pos})"
        );
    }

    #[test]
    fn test_apply_dependency_chain_c_before_b_before_a() {
        // A depends on B, B depends on C → C, B, A order
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec!["b"], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec!["c"], true),
            make_planner_item("c", "1.0.0", "1.1.0", vec![], true),
        ];

        let engine = ApplyEngine::new_noop();
        let summary = engine.apply(&items, &ApplyOptions::default());

        assert_eq!(summary.success.len(), 3);
        let order: Vec<&str> = summary
            .success
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        let c_pos = order.iter().position(|&id| id == "c").unwrap();
        let b_pos = order.iter().position(|&id| id == "b").unwrap();
        let a_pos = order.iter().position(|&id| id == "a").unwrap();
        assert!(c_pos < b_pos, "C before B");
        assert!(b_pos < a_pos, "B before A");
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-015: Failed guarded update stops dependent work
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_failed_guarded_stops_dependents() {
        // B fails → A (depends on B) should be blocked-by-failure
        let items = vec![
            make_planner_item_with_classification(
                "a",
                "1.0.0",
                "1.1.0",
                vec!["b"],
                true,
                UpdateClassification::Guarded,
            ),
            make_planner_item_with_classification(
                "b",
                "1.0.0",
                "1.1.0",
                vec![],
                true,
                UpdateClassification::Guarded,
            ),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions {
            force_fail: HashSet::from(["b".to_string()]),
        };
        let summary = engine.apply(&items, &options);

        // B should be failed
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].component_id, "b");

        // A should be blocked by failure
        assert_eq!(summary.blocked.len(), 1);
        assert_eq!(summary.blocked[0].component_id, "a");
        assert_eq!(summary.blocked[0].status, ApplyStatus::BlockedByFailure);
        assert!(summary.blocked[0].error_message.contains("b"));
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-016: Isolation-safe — unrelated work continues after failure
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_unrelated_continues_after_failure() {
        // B fails. A depends on B (blocked). C and D are unrelated (should succeed).
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec!["b"], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("c", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("d", "1.0.0", "1.1.0", vec![], true),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions {
            force_fail: HashSet::from(["b".to_string()]),
        };
        let summary = engine.apply(&items, &options);

        // B failed
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].component_id, "b");

        // A blocked by B's failure
        assert_eq!(summary.blocked.len(), 1);
        assert_eq!(summary.blocked[0].component_id, "a");

        // C and D succeed (unrelated)
        assert_eq!(summary.success.len(), 2);
        let success_ids: HashSet<&str> = summary
            .success
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        assert!(success_ids.contains("c"));
        assert!(success_ids.contains("d"));
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-012: User can untick any preselected component
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_user_can_untick_preselected_component() {
        // A is preselected (selected=true), B is deselected (selected=false)
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], false),
        ];

        let engine = ApplyEngine::new_noop();
        let summary = engine.apply(&items, &ApplyOptions::default());

        // A succeeds
        assert_eq!(summary.success.len(), 1);
        assert_eq!(summary.success[0].component_id, "a");

        // B is held back
        assert_eq!(summary.held_back.len(), 1);
        assert_eq!(summary.held_back[0].component_id, "b");
        assert_eq!(summary.held_back[0].status, ApplyStatus::HeldBack);
        assert!(!summary.held_back[0].was_selected);
    }

    #[test]
    fn test_apply_deselect_all_results_in_noop() {
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], false),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], false),
        ];

        let engine = ApplyEngine::new_noop();
        let summary = engine.apply(&items, &ApplyOptions::default());

        assert_eq!(summary.success.len(), 0);
        assert_eq!(summary.failed.len(), 0);
        assert_eq!(summary.held_back.len(), 2);
    }

    // -----------------------------------------------------------------------
    // ApplyStatus serde roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_status_serde_roundtrip() {
        let statuses = [
            ApplyStatus::Success,
            ApplyStatus::Failed,
            ApplyStatus::BlockedByFailure,
            ApplyStatus::HeldBack,
            ApplyStatus::Skipped,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: ApplyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
    }

    #[test]
    fn test_apply_status_serializes_to_kebab_case() {
        assert_eq!(
            serde_json::to_string(&ApplyStatus::Success).unwrap(),
            "\"success\""
        );
        assert_eq!(
            serde_json::to_string(&ApplyStatus::BlockedByFailure).unwrap(),
            "\"blocked-by-failure\""
        );
        assert_eq!(
            serde_json::to_string(&ApplyStatus::HeldBack).unwrap(),
            "\"held-back\""
        );
    }

    // -----------------------------------------------------------------------
    // ApplySummary complete partitioning (VAL-UPD-018)
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_summary_complete_partitioning() {
        // 4 items: 2 succeed, 1 fails, 1 held back
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("c", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("d", "1.0.0", "1.1.0", vec![], false),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions {
            force_fail: HashSet::from(["c".to_string()]),
        };
        let summary = engine.apply(&items, &options);

        // Verify complete partitioning
        assert!(summary.is_complete_partition());
        assert_eq!(summary.total(), 4);
        assert_eq!(summary.success.len(), 2);
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.held_back.len(), 1);
    }

    // -----------------------------------------------------------------------
    // ApplySummary has_failures
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_summary_has_failures() {
        let mut summary = ApplySummary::new();
        assert!(!summary.has_failures());

        summary
            .success
            .push(ApplyItem::new("a", "1.0.0", "1.1.0", ApplyStatus::Success));
        assert!(!summary.has_failures());

        summary
            .failed
            .push(ApplyItem::new("b", "1.0.0", "1.1.0", ApplyStatus::Failed));
        assert!(summary.has_failures());
    }

    // -----------------------------------------------------------------------
    // Dependency sort
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_sort_empty() {
        let items: Vec<&PlannerItem> = vec![];
        let result = dependency_sort(items);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dependency_sort_no_deps() {
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec![], true),
        ];
        let result = dependency_sort(items.iter().collect());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_dependency_sort_cycle_detected() {
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec!["b"], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec!["a"], true),
        ];
        let result = dependency_sort(items.iter().collect());
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ApplyItem constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_item_held_back() {
        let item = make_planner_item("test", "1.0.0", "1.1.0", vec![], false);
        let apply_item = ApplyItem::held_back(&item);
        assert_eq!(apply_item.component_id, "test");
        assert_eq!(apply_item.status, ApplyStatus::HeldBack);
        assert!(!apply_item.was_selected);
    }

    #[test]
    fn test_apply_item_blocked_by_failure() {
        let item = make_planner_item("a", "1.0.0", "1.1.0", vec!["b"], true);
        let apply_item = ApplyItem::blocked_by_failure(&item, "b", "install failed");
        assert_eq!(apply_item.component_id, "a");
        assert_eq!(apply_item.status, ApplyStatus::BlockedByFailure);
        assert!(apply_item.error_message.contains("b"));
        assert!(apply_item.error_message.contains("install failed"));
        assert!(apply_item.was_selected);
    }

    // -----------------------------------------------------------------------
    // Multi-level dependency failure propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_transitive_dependency_block() {
        // C depends on B, B depends on A. A fails → B blocked → C blocked.
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec!["a"], true),
            make_planner_item("c", "1.0.0", "1.1.0", vec!["b"], true),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions {
            force_fail: HashSet::from(["a".to_string()]),
        };
        let summary = engine.apply(&items, &options);

        // A failed
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].component_id, "a");

        // B and C blocked
        assert_eq!(summary.blocked.len(), 2);
        let blocked_ids: HashSet<&str> = summary
            .blocked
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        assert!(blocked_ids.contains("b"));
        assert!(blocked_ids.contains("c"));
    }

    // -----------------------------------------------------------------------
    // Diamond dependency
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_diamond_dependency() {
        // D depends on B and C, B depends on A, C depends on A
        // If A fails, B and C are blocked (depend on A), D is blocked (depends on B/C)
        let items = vec![
            make_planner_item("a", "1.0.0", "1.1.0", vec![], true),
            make_planner_item("b", "1.0.0", "1.1.0", vec!["a"], true),
            make_planner_item("c", "1.0.0", "1.1.0", vec!["a"], true),
            make_planner_item("d", "1.0.0", "1.1.0", vec!["b", "c"], true),
        ];

        let engine = ApplyEngine::new_noop();
        let options = ApplyOptions {
            force_fail: HashSet::from(["a".to_string()]),
        };
        let summary = engine.apply(&items, &options);

        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.blocked.len(), 3);
        assert_eq!(summary.success.len(), 0);
    }

    // -----------------------------------------------------------------------
    // All items succeed
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_all_succeed() {
        let items = vec![
            make_planner_item("rocm", "7.2.0", "7.2.1", vec![], true),
            make_planner_item("pytorch", "2.4.0", "2.4.1", vec!["rocm"], true),
            make_planner_item("triton", "3.0.0", "3.0.1", vec!["pytorch"], true),
        ];

        let engine = ApplyEngine::new_noop();
        let summary = engine.apply(&items, &ApplyOptions::default());

        assert_eq!(summary.success.len(), 3);
        assert_eq!(summary.failed.len(), 0);
        assert_eq!(summary.blocked.len(), 0);
        assert_eq!(summary.held_back.len(), 0);
    }

    // -----------------------------------------------------------------------
    // ApplySummary serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_summary_serde_roundtrip() {
        let mut summary = ApplySummary::new();
        summary
            .success
            .push(ApplyItem::new("a", "1.0.0", "1.1.0", ApplyStatus::Success));
        summary
            .failed
            .push(ApplyItem::new("b", "1.0.0", "1.1.0", ApplyStatus::Failed));
        summary
            .held_back
            .push(ApplyItem::new("c", "1.0.0", "1.1.0", ApplyStatus::HeldBack));

        let json = serde_json::to_string(&summary).unwrap();
        let back: ApplySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(summary, back);
    }

    // -----------------------------------------------------------------------
    // ApplyItem serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_item_serde_roundtrip() {
        let item = ApplyItem {
            component_id: "pytorch".to_string(),
            current_version: "2.4.0".to_string(),
            proposed_version: "2.5.0".to_string(),
            status: ApplyStatus::Failed,
            error_message: "install script returned non-zero exit code".to_string(),
            dependencies: vec!["rocm".to_string()],
            was_selected: true,
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: ApplyItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }
}
