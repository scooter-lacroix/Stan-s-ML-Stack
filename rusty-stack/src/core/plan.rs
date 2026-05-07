//! Update plan construction with risk tier assignment and dependency ordering.
//!
//! This module implements:
//! - [`PlanItem`] — a single component update entry with risk tier, validation tier,
//!   dependency list, and selection state.
//! - [`UpdatePlan`] — an ordered collection of plan items with topological sorting
//!   and circular dependency detection.
//!
//! # Risk Tier Assignment (VAL-CORE-014)
//!
//! | Risk Tier | Criteria |
//! |-----------|----------|
//! | `Low`     | Same version already installed (reinstall) |
//! | `Medium`  | New install or minor version upgrade |
//! | `High`    | Major version upgrade or Candidate/Experimental tier |
//!
//! # Dependency Ordering (VAL-CORE-015)
//!
//! If component B declares a dependency on component A, then A must appear
//! before B in execution order. Circular dependencies are detected and
//! reported as [`DependencyCycleError`].

use crate::core::types::{RiskTier, ValidationTier};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Dependency Cycle Error
// ---------------------------------------------------------------------------

/// Error returned when a circular dependency is detected in the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyCycleError {
    pub cycle: Vec<String>,
}

impl fmt::Display for DependencyCycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "circular dependency detected: {}",
            self.cycle.join(" → ")
        )
    }
}

impl std::error::Error for DependencyCycleError {}

// ---------------------------------------------------------------------------
// PlanItem
// ---------------------------------------------------------------------------

/// A single component entry in an update plan.
///
/// Each item captures the current and proposed versions, validation and risk
/// tiers, default selection state, a human-readable rationale, dependency
/// list, and whether the item can be applied in isolation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanItem {
    /// Unique component identifier (e.g., `"pytorch"`).
    pub component_id: String,
    /// Currently installed version (may be empty if not installed).
    pub current_version: String,
    /// Version proposed by the update.
    pub proposed_version: String,
    /// Validation maturity tier of the component.
    pub validation_tier: ValidationTier,
    /// Risk tier assigned based on version delta and validation tier.
    pub risk_tier: RiskTier,
    /// Whether this item is pre-selected for apply.
    pub selection_default: bool,
    /// Human-readable explanation of why this update is proposed.
    pub rationale: String,
    /// Component IDs that must be applied before this one.
    pub dependencies: Vec<String>,
    /// Whether this item can be applied independently of other failures.
    pub isolation_safe: bool,
}

impl PlanItem {
    /// Create a new plan item with automatic risk tier assignment.
    ///
    /// Risk tier is computed from the version delta and validation tier:
    /// - `Low`: same version already installed
    /// - `Medium`: new install or minor version bump with Validated tier
    /// - `High`: major version bump or Candidate/Experimental tier
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        component_id: impl Into<String>,
        current_version: impl Into<String>,
        proposed_version: impl Into<String>,
        validation_tier: ValidationTier,
        selection_default: bool,
        rationale: impl Into<String>,
        dependencies: Vec<String>,
        isolation_safe: bool,
    ) -> Self {
        let current = current_version.into();
        let proposed = proposed_version.into();
        let risk_tier = Self::compute_risk_tier(&current, &proposed, validation_tier);

        Self {
            component_id: component_id.into(),
            current_version: current,
            proposed_version: proposed,
            validation_tier,
            risk_tier,
            selection_default,
            rationale: rationale.into(),
            dependencies,
            isolation_safe,
        }
    }

    /// Compute risk tier based on version delta and validation tier.
    ///
    /// Rules:
    /// - Same version (reinstall) → `Low`
    /// - Empty current version (new install) → `Medium`
    /// - Minor/patch version bump with `Validated` tier → `Medium`
    /// - Major version bump → `High`
    /// - `Candidate` or `Experimental` tier → `High`
    fn compute_risk_tier(
        current: &str,
        proposed: &str,
        validation_tier: ValidationTier,
    ) -> RiskTier {
        // Candidate/Experimental always high risk
        if matches!(
            validation_tier,
            ValidationTier::Candidate | ValidationTier::Experimental
        ) {
            return RiskTier::High;
        }

        // Same version → low risk (reinstall)
        if !current.is_empty() && current == proposed {
            return RiskTier::Low;
        }

        // New install → medium risk
        if current.is_empty() {
            return RiskTier::Medium;
        }

        // Parse versions to determine bump level
        if let (Some(cur_parts), Some(pro_parts)) =
            (parse_version_parts(current), parse_version_parts(proposed))
        {
            if pro_parts[0] > cur_parts[0] {
                // Major version bump → high risk
                return RiskTier::High;
            }
            // Minor or patch bump → medium risk
            return RiskTier::Medium;
        }

        // Fallback: if we can't parse, treat as medium
        RiskTier::Medium
    }
}

/// Parse a version string into its major/minor/patch components.
fn parse_version_parts(version: &str) -> Option<Vec<u32>> {
    let base = version.split('-').next()?;
    base.split('.')
        .map(|s| s.parse::<u32>().ok())
        .collect::<Option<Vec<_>>>()
}

// ---------------------------------------------------------------------------
// UpdatePlan
// ---------------------------------------------------------------------------

/// An ordered collection of update plan items.
///
/// Supports topological sorting by dependency and circular dependency
/// detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdatePlan {
    /// Plan items, potentially in dependency order after sorting.
    pub items: Vec<PlanItem>,
}

impl UpdatePlan {
    /// Create a new empty plan.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Create a plan from a list of items.
    pub fn from_items(items: Vec<PlanItem>) -> Self {
        Self { items }
    }

    /// Add an item to the plan.
    pub fn add(&mut self, item: PlanItem) {
        self.items.push(item);
    }

    /// Sort items in dependency-safe execution order using topological sort.
    ///
    /// If a circular dependency is detected, returns [`DependencyCycleError`]
    /// with the cycle path.
    ///
    /// # Algorithm
    ///
    /// Uses Kahn's algorithm (BFS-based topological sort):
    /// 1. Compute in-degree for each node
    /// 2. Start with nodes that have no dependencies
    /// 3. Process nodes in order, decrementing in-degrees
    /// 4. If not all nodes processed → cycle detected
    pub fn sort_by_dependencies(&mut self) -> Result<(), DependencyCycleError> {
        if self.items.is_empty() {
            return Ok(());
        }

        let n = self.items.len();
        let id_to_idx: HashMap<&str, usize> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| (item.component_id.as_str(), i))
            .collect();

        // Build adjacency list and in-degree counts
        let mut in_degree: Vec<usize> = vec![0; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (idx, item) in self.items.iter().enumerate() {
            for dep_id in &item.dependencies {
                if let Some(&dep_idx) = id_to_idx.get(dep_id.as_str()) {
                    adj[dep_idx].push(idx);
                    in_degree[idx] += 1;
                }
                // Dependencies on items not in the plan are ignored
            }
        }

        // BFS: start with zero in-degree nodes
        let mut queue: VecDeque<usize> = VecDeque::new();
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
            // Cycle detected — find the cycle for error reporting
            let cycle = Self::find_cycle(&self.items, &id_to_idx);
            return Err(DependencyCycleError { cycle });
        }

        // Reorder items
        let mut new_items: Vec<PlanItem> = Vec::with_capacity(n);
        for idx in sorted_indices {
            new_items.push(self.items[idx].clone());
        }
        self.items = new_items;

        Ok(())
    }

    /// Find a cycle in the dependency graph for error reporting.
    ///
    /// Uses DFS with recursion stack to find the first cycle.
    fn find_cycle(items: &[PlanItem], id_to_idx: &HashMap<&str, usize>) -> Vec<String> {
        let n = items.len();
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];
        let mut path: Vec<String> = Vec::new();

        for start in 0..n {
            if !visited[start] {
                if let Some(cycle) = Self::dfs_cycle(
                    start,
                    items,
                    id_to_idx,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                ) {
                    return cycle;
                }
            }
        }

        // Fallback: return all items with remaining in-degree > 0
        items.iter().map(|i| i.component_id.clone()).collect()
    }

    #[allow(clippy::ptr_arg)]
    fn dfs_cycle(
        node: usize,
        items: &[PlanItem],
        id_to_idx: &HashMap<&str, usize>,
        visited: &mut [bool],
        rec_stack: &mut [bool],
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        visited[node] = true;
        rec_stack[node] = true;
        path.push(items[node].component_id.clone());

        for dep_id in &items[node].dependencies {
            if let Some(&dep_idx) = id_to_idx.get(dep_id.as_str()) {
                if !visited[dep_idx] {
                    if let Some(cycle) =
                        Self::dfs_cycle(dep_idx, items, id_to_idx, visited, rec_stack, path)
                    {
                        return Some(cycle);
                    }
                } else if rec_stack[dep_idx] {
                    // Found cycle — extract it from path
                    let cycle_start = path
                        .iter()
                        .position(|id| id == &items[dep_idx].component_id)
                        .unwrap();
                    let mut cycle: Vec<String> = path[cycle_start..].to_vec();
                    cycle.push(items[dep_idx].component_id.clone());
                    return Some(cycle);
                }
            }
        }

        path.pop();
        rec_stack[node] = false;
        None
    }

    /// Sort items by risk tier (ascending: Low, Medium, High).
    pub fn sort_by_risk(&mut self) {
        self.items.sort_by_key(|a| a.risk_tier.order());
    }

    /// Get items filtered by risk tier.
    pub fn items_by_risk(&self, tier: RiskTier) -> Vec<&PlanItem> {
        self.items.iter().filter(|i| i.risk_tier == tier).collect()
    }

    /// Get a plan item by component ID.
    pub fn get(&self, component_id: &str) -> Option<&PlanItem> {
        self.items.iter().find(|i| i.component_id == component_id)
    }

    /// Number of items in the plan.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for UpdatePlan {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-CORE-014: Risk tier assignment
    // =======================================================================

    #[test]
    fn test_plan_risk_tier_low_for_same_version() {
        let item = PlanItem::new(
            "pytorch",
            "2.1.0",
            "2.1.0",
            ValidationTier::Validated,
            true,
            "Reinstall",
            vec![],
            true,
        );
        assert_eq!(item.risk_tier, RiskTier::Low);
    }

    #[test]
    fn test_plan_risk_tier_medium_for_new_install() {
        let item = PlanItem::new(
            "triton",
            "", // not installed
            "3.0.0",
            ValidationTier::Validated,
            false,
            "New install",
            vec![],
            true,
        );
        assert_eq!(item.risk_tier, RiskTier::Medium);
    }

    #[test]
    fn test_plan_risk_tier_medium_for_minor_bump() {
        let item = PlanItem::new(
            "pytorch",
            "2.1.0",
            "2.2.0",
            ValidationTier::Validated,
            true,
            "Minor upgrade",
            vec![],
            true,
        );
        assert_eq!(item.risk_tier, RiskTier::Medium);
    }

    #[test]
    fn test_plan_risk_tier_medium_for_patch_bump() {
        let item = PlanItem::new(
            "rocm",
            "7.2.0",
            "7.2.1",
            ValidationTier::Validated,
            true,
            "Patch update",
            vec![],
            true,
        );
        assert_eq!(item.risk_tier, RiskTier::Medium);
    }

    #[test]
    fn test_plan_risk_tier_high_for_major_bump() {
        let item = PlanItem::new(
            "pytorch",
            "2.1.0",
            "3.0.0",
            ValidationTier::Validated,
            false,
            "Major upgrade",
            vec![],
            false,
        );
        assert_eq!(item.risk_tier, RiskTier::High);
    }

    #[test]
    fn test_plan_risk_tier_high_for_candidate_tier() {
        let item = PlanItem::new(
            "triton",
            "3.0.0",
            "3.1.0",
            ValidationTier::Candidate,
            false,
            "Candidate component",
            vec![],
            false,
        );
        assert_eq!(item.risk_tier, RiskTier::High);
    }

    #[test]
    fn test_plan_risk_tier_high_for_experimental_tier() {
        let item = PlanItem::new(
            "deepspeed",
            "",
            "0.14.0",
            ValidationTier::Experimental,
            false,
            "Experimental",
            vec![],
            false,
        );
        assert_eq!(item.risk_tier, RiskTier::High);
    }

    #[test]
    fn test_plan_risk_tier_sort_order() {
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "high",
                "1.0.0",
                "2.0.0",
                ValidationTier::Validated,
                false,
                "",
                vec![],
                true,
            ),
            PlanItem::new(
                "low",
                "1.0.0",
                "1.0.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
            PlanItem::new(
                "medium",
                "",
                "1.0.0",
                ValidationTier::Validated,
                false,
                "",
                vec![],
                true,
            ),
        ]);
        plan.sort_by_risk();
        assert_eq!(plan.items[0].component_id, "low");
        assert_eq!(plan.items[1].component_id, "medium");
        assert_eq!(plan.items[2].component_id, "high");
    }

    // =======================================================================
    // VAL-CORE-015: Dependency ordering (topological sort)
    // =======================================================================

    #[test]
    fn test_plan_dependency_ordering_linear() {
        // B depends on A → A must come first
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "b",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "B update",
                vec!["a".to_string()],
                true,
            ),
            PlanItem::new(
                "a",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "A update",
                vec![],
                true,
            ),
        ]);
        plan.sort_by_dependencies().unwrap();

        let order: Vec<&str> = plan.items.iter().map(|i| i.component_id.as_str()).collect();
        let a_pos = order.iter().position(|&id| id == "a").unwrap();
        let b_pos = order.iter().position(|&id| id == "b").unwrap();
        assert!(a_pos < b_pos, "A should appear before B in execution order");
    }

    #[test]
    fn test_plan_dependency_ordering_diamond() {
        // D depends on B and C, B depends on A, C depends on A
        // → A must come before B and C, B and C must come before D
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "d",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["b".to_string(), "c".to_string()],
                true,
            ),
            PlanItem::new(
                "c",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["a".to_string()],
                true,
            ),
            PlanItem::new(
                "b",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["a".to_string()],
                true,
            ),
            PlanItem::new(
                "a",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
        ]);
        plan.sort_by_dependencies().unwrap();

        let order: Vec<&str> = plan.items.iter().map(|i| i.component_id.as_str()).collect();
        let a_pos = order.iter().position(|&id| id == "a").unwrap();
        let b_pos = order.iter().position(|&id| id == "b").unwrap();
        let c_pos = order.iter().position(|&id| id == "c").unwrap();
        let d_pos = order.iter().position(|&id| id == "d").unwrap();

        assert!(a_pos < b_pos, "A before B");
        assert!(a_pos < c_pos, "A before C");
        assert!(b_pos < d_pos, "B before D");
        assert!(c_pos < d_pos, "C before D");
    }

    #[test]
    fn test_plan_circular_dependency_detected() {
        // A → B → A cycle
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "a",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["b".to_string()],
                true,
            ),
            PlanItem::new(
                "b",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["a".to_string()],
                true,
            ),
        ]);
        let result = plan.sort_by_dependencies();
        assert!(result.is_err(), "Circular dependency should be detected");
        let err = result.unwrap_err();
        assert!(!err.cycle.is_empty(), "Cycle path should be non-empty");
    }

    #[test]
    fn test_plan_circular_dependency_three_way() {
        // A → B → C → A cycle
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "a",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["c".to_string()],
                true,
            ),
            PlanItem::new(
                "b",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["a".to_string()],
                true,
            ),
            PlanItem::new(
                "c",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec!["b".to_string()],
                true,
            ),
        ]);
        let result = plan.sort_by_dependencies();
        assert!(
            result.is_err(),
            "Three-way circular dependency should be detected"
        );
    }

    #[test]
    fn test_plan_no_dependencies_preserves_order() {
        let mut plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "c",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
            PlanItem::new(
                "a",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
            PlanItem::new(
                "b",
                "1.0.0",
                "1.1.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
        ]);
        plan.sort_by_dependencies().unwrap();
        // With zero dependencies, all items have in-degree 0, so BFS order
        // preserves insertion order
        assert_eq!(plan.items[0].component_id, "c");
        assert_eq!(plan.items[1].component_id, "a");
        assert_eq!(plan.items[2].component_id, "b");
    }

    #[test]
    fn test_plan_empty_plan_sort_succeeds() {
        let mut plan = UpdatePlan::new();
        assert!(plan.sort_by_dependencies().is_ok());
    }

    // =======================================================================
    // Serde roundtrip
    // =======================================================================

    #[test]
    fn test_plan_item_serde_roundtrip() {
        let item = PlanItem::new(
            "pytorch",
            "2.1.0",
            "2.2.0",
            ValidationTier::Validated,
            true,
            "Minor version upgrade",
            vec!["rocm".to_string()],
            true,
        );
        let json = serde_json::to_string(&item).unwrap();
        let back: PlanItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }

    #[test]
    fn test_update_plan_serde_roundtrip() {
        let plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "rocm",
                "7.2.0",
                "7.2.1",
                ValidationTier::Validated,
                true,
                "Patch",
                vec![],
                true,
            ),
            PlanItem::new(
                "pytorch",
                "2.1.0",
                "2.2.0",
                ValidationTier::Validated,
                true,
                "Minor",
                vec!["rocm".to_string()],
                true,
            ),
        ]);
        let json = serde_json::to_string(&plan).unwrap();
        let back: UpdatePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn test_dependency_cycle_error_display() {
        let err = DependencyCycleError {
            cycle: vec!["a".to_string(), "b".to_string(), "a".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("a"));
        assert!(msg.contains("b"));
        assert!(msg.contains("circular dependency"));
    }

    #[test]
    fn test_dependency_cycle_error_is_std_error() {
        let err = DependencyCycleError {
            cycle: vec!["a".to_string()],
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_plan_items_by_risk() {
        let plan = UpdatePlan::from_items(vec![
            PlanItem::new(
                "low",
                "1.0.0",
                "1.0.0",
                ValidationTier::Validated,
                true,
                "",
                vec![],
                true,
            ),
            PlanItem::new(
                "high",
                "1.0.0",
                "2.0.0",
                ValidationTier::Validated,
                false,
                "",
                vec![],
                false,
            ),
            PlanItem::new(
                "medium",
                "",
                "1.0.0",
                ValidationTier::Validated,
                false,
                "",
                vec![],
                true,
            ),
        ]);
        assert_eq!(plan.items_by_risk(RiskTier::Low).len(), 1);
        assert_eq!(plan.items_by_risk(RiskTier::Medium).len(), 1);
        assert_eq!(plan.items_by_risk(RiskTier::High).len(), 1);
    }

    #[test]
    fn test_plan_get_by_id() {
        let plan = UpdatePlan::from_items(vec![PlanItem::new(
            "pytorch",
            "2.1.0",
            "2.2.0",
            ValidationTier::Validated,
            true,
            "",
            vec![],
            true,
        )]);
        assert!(plan.get("pytorch").is_some());
        assert!(plan.get("nonexistent").is_none());
    }

    #[test]
    fn test_plan_len_and_is_empty() {
        let mut plan = UpdatePlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        plan.add(PlanItem::new(
            "a",
            "1.0.0",
            "1.1.0",
            ValidationTier::Validated,
            true,
            "",
            vec![],
            true,
        ));
        assert!(!plan.is_empty());
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn test_plan_default() {
        let plan = UpdatePlan::default();
        assert!(plan.is_empty());
    }
}
