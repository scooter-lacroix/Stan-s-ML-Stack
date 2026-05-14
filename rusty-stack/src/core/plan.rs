//! Plan types and topological sorting.
//!
//! This module implements:
//! - [`PlanItem`] — a component update plan item with risk tier, dependencies, etc.
//! - [`topological_sort`] — sorts components by dependency order
//! - [`Plan`] — a full update plan with items and summary

use crate::core::types::{RiskTier, ValidationTier};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Plan Item
// ---------------------------------------------------------------------------

/// A single item in an update plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanItem {
    /// Component ID.
    pub component_id: String,
    /// Currently installed version (empty if not installed).
    pub current_version: String,
    /// Proposed version to install.
    pub proposed_version: String,
    /// Validation tier of this component.
    pub validation_tier: ValidationTier,
    /// Whether this item is selected for apply.
    pub selected: bool,
    /// Human-readable rationale for this update.
    pub rationale: String,
    /// Component IDs this item depends on.
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Whether this component can be installed in isolation.
    pub isolation_safe: bool,
    /// Risk tier (low/medium/high).
    pub risk_tier: RiskTier,
}

impl PlanItem {
    /// Create a new plan item.
    pub fn new(
        component_id: &str,
        current_version: &str,
        proposed_version: &str,
        validation_tier: ValidationTier,
        selected: bool,
        rationale: &str,
        dependencies: Vec<String>,
        isolation_safe: bool,
    ) -> Self {
        Self {
            component_id: component_id.to_string(),
            current_version: current_version.to_string(),
            proposed_version: proposed_version.to_string(),
            validation_tier,
            selected,
            rationale: rationale.to_string(),
            dependencies,
            isolation_safe,
            risk_tier: RiskTier::Medium, // default
        }
    }
}

// ---------------------------------------------------------------------------
// Topological Sort
// ---------------------------------------------------------------------------

/// Sort components topologically by their dependencies.
///
/// Returns a new vector with components ordered such that all dependencies
/// of a component appear before the component itself.
///
/// # Arguments
/// * `items` - Vector of plan items
///
/// # Returns
/// A new vector with items in topological order.
///
/// # Panics
/// Panics if a dependency cycle is detected.
pub fn topological_sort(items: Vec<PlanItem>) -> Vec<PlanItem> {
    // Track remaining dependencies for each node
    let mut remaining_deps: HashMap<String, Vec<String>> = HashMap::new();
    // Track which nodes depend on each node (for cleanup)
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
    let mut all_nodes: HashSet<String> = HashSet::new();

    // Initialize tracking structures
    for item in &items {
        all_nodes.insert(item.component_id.clone());
        // Each node starts with all its declared dependencies
        remaining_deps.insert(item.component_id.clone(), item.dependencies.clone());
        // Track dependents for cleanup
        for dep in &item.dependencies {
            dependents
                .entry(dep.clone())
                .or_default()
                .push(item.component_id.clone());
        }
    }

    // Kahn's algorithm
    let mut sorted_nodes = Vec::new();

    // Start with nodes that have no dependencies
    let mut queue: Vec<String> = all_nodes
        .iter()
        .filter(|id| remaining_deps.get(*id).map_or(true, |d| d.is_empty()))
        .cloned()
        .collect();

    while let Some(node) = queue.pop() {
        sorted_nodes.push(node.clone());

        // Find all nodes that depend on this node
        if let Some(dep_on_node) = dependents.get(&node) {
            for dependent in dep_on_node.clone() {
                // Remove this node from dependent's remaining dependencies
                if let Some(deps) = remaining_deps.get_mut(&dependent) {
                    deps.retain(|d| d != &node);
                    // If all dependencies are now satisfied, add to queue
                    if deps.is_empty() {
                        queue.push(dependent);
                    }
                }
            }
        }
    }

    if sorted_nodes.len() != all_nodes.len() {
        panic!("Dependency cycle detected in plan items");
    }

    // Reconstruct plan items in topological order
    let mut item_map: HashMap<String, PlanItem> = items
        .into_iter()
        .map(|item| (item.component_id.clone(), item))
        .collect();

    sorted_nodes
        .into_iter()
        .filter_map(|id| item_map.remove(&id))
        .collect()
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A full update plan with items and summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    /// Plan items in execution order.
    pub items: Vec<PlanItem>,
    /// Summary statistics.
    pub summary: PlanSummary,
}

/// Summary statistics for a plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanSummary {
    pub total: usize,
    pub selected: usize,
    pub safe: usize,
    pub guarded: usize,
    pub candidate: usize,
    pub experimental: usize,
    pub blocked: usize,
}

impl Plan {
    /// Create a new plan from items.
    pub fn new(items: Vec<PlanItem>) -> Self {
        let items = topological_sort(items);
        let summary = PlanSummary::from_items(&items);
        Self { items, summary }
    }
}

impl PlanSummary {
    /// Compute summary from a list of plan items.
    pub fn from_items(items: &[PlanItem]) -> Self {
        Self {
            total: items.len(),
            selected: items.iter().filter(|i| i.selected).count(),
            safe: items
                .iter()
                .filter(|i| i.risk_tier == RiskTier::Low)
                .count(),
            guarded: items
                .iter()
                .filter(|i| i.risk_tier == RiskTier::Medium)
                .count(),
            candidate: items
                .iter()
                .filter(|i| i.risk_tier == RiskTier::High)
                .count(),
            experimental: items
                .iter()
                .filter(|i| i.validation_tier == ValidationTier::Experimental)
                .count(),
            blocked: 0, // blocked items are filtered out
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{RiskTier, ValidationTier};

    // -----------------------------------------------------------------------
    // VAL-PLAN-001: Topological sort orders dependencies before dependents
    // -----------------------------------------------------------------------

    #[test]
    fn test_topological_sort_simple_dependency() {
        let items = vec![
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec!["rocm".to_string()],
                true,
            ),
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
        ];

        let sorted = topological_sort(items);
        assert_eq!(sorted[0].component_id, "rocm");
        assert_eq!(sorted[1].component_id, "pytorch");
    }

    #[test]
    fn test_topological_sort_complex_dependency_chain() {
        let items = vec![
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec!["rocm".to_string()],
                true,
            ),
            PlanItem::new(
                "triton",
                "3.0.0",
                "3.1.0",
                ValidationTier::Validated,
                true,
                "update triton",
                vec!["pytorch".to_string()],
                true,
            ),
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
        ];

        let sorted = topological_sort(items);
        assert_eq!(sorted[0].component_id, "rocm");
        assert_eq!(sorted[1].component_id, "pytorch");
        assert_eq!(sorted[2].component_id, "triton");
    }

    #[test]
    fn test_topological_sort_empty_dependencies() {
        let items = vec![
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec![],
                true,
            ),
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
        ];

        let sorted = topological_sort(items);
        // Order of independent items is not guaranteed, but all should be present
        let ids: HashSet<_> = sorted.iter().map(|i| i.component_id.clone()).collect();
        assert!(ids.contains("pytorch"));
        assert!(ids.contains("rocm"));
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-002: Topological sort detects dependency cycles
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Dependency cycle detected")]
    fn test_topological_sort_detects_cycle() {
        let items = vec![
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec!["triton".to_string()],
                true,
            ),
            PlanItem::new(
                "triton",
                "3.0.0",
                "3.1.0",
                ValidationTier::Validated,
                true,
                "update triton",
                vec!["pytorch".to_string()],
                true,
            ),
        ];

        topological_sort(items);
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-003: Plan creation applies topological sort
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_creation_applies_topological_sort() {
        let items = vec![
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec!["rocm".to_string()],
                true,
            ),
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
        ];

        let plan = Plan::new(items);
        assert_eq!(plan.items[0].component_id, "rocm");
        assert_eq!(plan.items[1].component_id, "pytorch");
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-004: Plan summary counts are accurate
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_summary_counts() {
        let items = vec![
            PlanItem {
                component_id: "rocm".to_string(),
                current_version: "7.2.0".to_string(),
                proposed_version: "7.3.0".to_string(),
                validation_tier: ValidationTier::Validated,
                selected: true,
                rationale: "update rocm".to_string(),
                dependencies: vec![],
                isolation_safe: true,
                risk_tier: RiskTier::Low,
            },
            PlanItem {
                component_id: "pytorch".to_string(),
                current_version: "2.4.0".to_string(),
                proposed_version: "2.5.0".to_string(),
                validation_tier: ValidationTier::Validated,
                selected: true,
                rationale: "update pytorch".to_string(),
                dependencies: vec!["rocm".to_string()],
                isolation_safe: true,
                risk_tier: RiskTier::Medium,
            },
            PlanItem {
                component_id: "deepspeed".to_string(),
                current_version: "".to_string(),
                proposed_version: "0.15.0".to_string(),
                validation_tier: ValidationTier::Experimental,
                selected: false,
                rationale: "new experimental".to_string(),
                dependencies: vec!["pytorch".to_string()],
                isolation_safe: true,
                risk_tier: RiskTier::High,
            },
        ];

        let plan = Plan::new(items);
        assert_eq!(plan.summary.total, 3);
        assert_eq!(plan.summary.selected, 2);
        assert_eq!(plan.summary.safe, 1);
        assert_eq!(plan.summary.guarded, 1);
        assert_eq!(plan.summary.candidate, 1);
        assert_eq!(plan.summary.experimental, 1);
        assert_eq!(plan.summary.blocked, 0);
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-005: PlanItem serde roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_item_serde_roundtrip() {
        let item = PlanItem {
            component_id: "pytorch".to_string(),
            current_version: "2.4.0".to_string(),
            proposed_version: "2.5.0".to_string(),
            validation_tier: ValidationTier::Validated,
            selected: true,
            rationale: "update pytorch".to_string(),
            dependencies: vec!["rocm".to_string()],
            isolation_safe: true,
            risk_tier: RiskTier::Medium,
        };

        let json = serde_json::to_string(&item).unwrap();
        let back: PlanItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-006: Plan serde roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_serde_roundtrip() {
        let items = vec![
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
            PlanItem::new(
                "pytorch",
                "2.4.0",
                "2.5.0",
                ValidationTier::Validated,
                true,
                "update pytorch",
                vec!["rocm".to_string()],
                true,
            ),
        ];

        let plan = Plan::new(items);
        let json = serde_json::to_string(&plan).unwrap();
        let back: Plan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan.items, back.items);
        assert_eq!(plan.summary, back.summary);
    }

    // -----------------------------------------------------------------------
    // VAL-PLAN-007: Topological sort with dependencies from baseline manifest
    // -----------------------------------------------------------------------

    #[test]
    fn test_topological_sort_with_llama_cpp_dependencies() {
        let items = vec![
            PlanItem::new(
                "llama-cpp",
                "",
                "latest",
                ValidationTier::Experimental,
                true,
                "install llama-cpp",
                vec!["rocm".to_string()],
                true,
            ),
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.3.0",
                ValidationTier::Validated,
                true,
                "update rocm",
                vec![],
                true,
            ),
        ];

        let sorted = topological_sort(items);
        assert_eq!(sorted[0].component_id, "rocm");
        assert_eq!(sorted[1].component_id, "llama-cpp");
    }
}
