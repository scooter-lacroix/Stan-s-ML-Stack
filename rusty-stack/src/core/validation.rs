//! Validation state machine governing tier transitions and UI filtering.
//!
//! This module implements:
//! - [`ValidationStateMachine`] — enforces legal transitions between validation
//!   tiers (Validated ↔ Candidate, Candidate ↔ Experimental, any → Blocked).
//! - [`ValidationEntry`] — pairs a component ID with its current tier, defaulting
//!   to `Candidate` for new components.
//! - [`TierFilter`] — determines visibility and preselection for UI rendering.
//!
//! # Legal Transitions
//!
//! ```text
//! Validated  ←→  Candidate  ←→  Experimental
//!     ↓             ↓               ↓
//!           Blocked (terminal)
//! ```
//!
//! Illegal transitions:
//! - `Experimental → Validated` (must go through Candidate first)
//! - `Blocked → anything` (Blocked is terminal)
//!
//! # UI Filtering Rules
//!
//! | Tier         | Visible | Preselected |
//! |--------------|---------|-------------|
//! | Validated    | Yes     | Yes         |
//! | Candidate    | Yes     | No          |
//! | Experimental | No*     | No          |
//! | Blocked      | No      | No          |
//!
//! \* Experimental components are visible only with an explicit flag.

use crate::core::types::ValidationTier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Transition Error
// ---------------------------------------------------------------------------

/// Error returned when an illegal tier transition is attempted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionError {
    pub from: ValidationTier,
    pub to: ValidationTier,
    pub reason: String,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "illegal transition from {:?} to {:?}: {}",
            self.from, self.to, self.reason
        )
    }
}

impl std::error::Error for TransitionError {}

// ---------------------------------------------------------------------------
// ValidationStateMachine
// ---------------------------------------------------------------------------

/// State machine governing transitions between validation tiers.
///
/// The machine enforces the following transition rules:
/// - `Validated ↔ Candidate` — bidirectional
/// - `Candidate ↔ Experimental` — bidirectional
/// - `Any → Blocked` — one-way (terminal)
/// - `Validated ↔ Validated` — self-transition (no-op, succeeds)
/// - `Candidate ↔ Candidate` — self-transition (no-op, succeeds)
/// - `Experimental ↔ Experimental` — self-transition (no-op, succeeds)
/// - `Blocked ↔ Blocked` — self-transition (no-op, succeeds)
///
/// Illegal transitions return a [`TransitionError`].
#[derive(Debug, Clone, Default)]
pub struct ValidationStateMachine {
    /// Map from component ID to its current validation tier.
    entries: HashMap<String, ValidationTier>,
}

impl ValidationStateMachine {
    /// Create a new empty state machine.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a new component. Defaults to [`ValidationTier::Candidate`].
    ///
    /// If the component already exists, this is a no-op (preserves existing tier).
    pub fn register(&mut self, component_id: impl Into<String>) {
        let id = component_id.into();
        self.entries.entry(id).or_insert(ValidationTier::Candidate);
    }

    /// Register a component with an explicit initial tier.
    ///
    /// If the component already exists, this is a no-op.
    pub fn register_with_tier(&mut self, component_id: impl Into<String>, tier: ValidationTier) {
        let id = component_id.into();
        self.entries.entry(id).or_insert(tier);
    }

    /// Attempt to transition a component to a new tier.
    ///
    /// Returns `Ok(())` if the transition is legal, `Err(TransitionError)` otherwise.
    pub fn transition(
        &mut self,
        component_id: &str,
        target: ValidationTier,
    ) -> Result<(), TransitionError> {
        let current = self
            .entries
            .get(component_id)
            .copied()
            .unwrap_or(ValidationTier::Candidate);

        if Self::is_legal_transition(current, target) {
            self.entries.insert(component_id.to_string(), target);
            Ok(())
        } else {
            Err(TransitionError {
                from: current,
                to: target,
                reason: Self::transition_reason(current, target),
            })
        }
    }

    /// Query the current tier of a component.
    ///
    /// Returns `ValidationTier::Candidate` for unregistered components.
    pub fn tier(&self, component_id: &str) -> ValidationTier {
        self.entries
            .get(component_id)
            .copied()
            .unwrap_or(ValidationTier::Candidate)
    }

    /// Check if a transition is legal without performing it.
    pub fn can_transition(&self, component_id: &str, target: ValidationTier) -> bool {
        let current = self
            .entries
            .get(component_id)
            .copied()
            .unwrap_or(ValidationTier::Candidate);
        Self::is_legal_transition(current, target)
    }

    /// Determine if a transition from `from` to `to` is legal.
    ///
    /// Legal transitions:
    /// - Self-transitions (any → same): always legal
    /// - Validated ↔ Candidate: bidirectional
    /// - Candidate ↔ Experimental: bidirectional
    /// - Any → Blocked: one-way
    ///
    /// Illegal:
    /// - Experimental → Validated (must go through Candidate)
    /// - Blocked → anything except Blocked (terminal state)
    fn is_legal_transition(from: ValidationTier, to: ValidationTier) -> bool {
        if from == to {
            return true; // self-transitions
        }
        match from {
            ValidationTier::Validated => {
                matches!(to, ValidationTier::Candidate | ValidationTier::Blocked)
            }
            ValidationTier::Candidate => matches!(
                to,
                ValidationTier::Validated | ValidationTier::Experimental | ValidationTier::Blocked
            ),
            ValidationTier::Experimental => {
                matches!(to, ValidationTier::Candidate | ValidationTier::Blocked)
            }
            ValidationTier::Blocked => false, // terminal: no outgoing transitions
        }
    }

    /// Human-readable reason for why a transition is illegal.
    fn transition_reason(from: ValidationTier, to: ValidationTier) -> String {
        match from {
            ValidationTier::Blocked => {
                "Blocked is a terminal state; no transitions out allowed".to_string()
            }
            ValidationTier::Experimental if to == ValidationTier::Validated => {
                "Experimental components must pass through Candidate before reaching Validated"
                    .to_string()
            }
            _ => format!("transition from {:?} to {:?} is not allowed", from, to),
        }
    }

    /// Get all registered component IDs and their tiers.
    pub fn entries(&self) -> &HashMap<String, ValidationTier> {
        &self.entries
    }

    /// Force-set a component's tier without transition checks.
    ///
    /// Used internally for loading persisted state or administrative overrides.
    pub fn force_set(&mut self, component_id: impl Into<String>, tier: ValidationTier) {
        self.entries.insert(component_id.into(), tier);
    }
}

// ---------------------------------------------------------------------------
// ValidationEntry — serializable component+tier pair
// ---------------------------------------------------------------------------

/// A component paired with its current validation tier.
///
/// New entries default to [`ValidationTier::Candidate`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationEntry {
    pub component_id: String,
    pub tier: ValidationTier,
}

impl ValidationEntry {
    /// Create a new entry defaulting to Candidate tier.
    pub fn new(component_id: impl Into<String>) -> Self {
        Self {
            component_id: component_id.into(),
            tier: ValidationTier::Candidate,
        }
    }

    /// Create a new entry with an explicit tier.
    pub fn with_tier(component_id: impl Into<String>, tier: ValidationTier) -> Self {
        Self {
            component_id: component_id.into(),
            tier,
        }
    }

    /// Whether this component is visible in the default UI view.
    pub fn is_visible(&self) -> bool {
        self.tier.is_visible()
    }

    /// Whether this component is visible when `include_experimental` is set.
    pub fn is_visible_with_experimental(&self) -> bool {
        match self.tier {
            ValidationTier::Validated
            | ValidationTier::Candidate
            | ValidationTier::Experimental => true,
            ValidationTier::Blocked => false,
        }
    }

    /// Whether this component should be preselected in the UI.
    pub fn is_preselected(&self) -> bool {
        self.tier.is_preselected()
    }
}

// ---------------------------------------------------------------------------
// TierFilter — batch filtering for UI rendering
// ---------------------------------------------------------------------------

/// Result of applying tier filtering to a list of entries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilteredEntries {
    /// Entries visible in the default view, with preselection flags.
    pub visible: Vec<VisibleEntry>,
    /// Entries hidden from the default view (experimental).
    pub hidden: Vec<ValidationEntry>,
}

/// A visible entry with its preselection state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VisibleEntry {
    pub component_id: String,
    pub tier: ValidationTier,
    pub preselected: bool,
}

/// Apply tier filtering to a list of validation entries.
///
/// Returns a [`FilteredEntries`] struct partitioning entries into:
/// - `visible`: entries that should be shown in the default UI view, with
///   preselection flags set per the spec.
/// - `hidden`: entries hidden from the default view (experimental).
///
/// Blocked entries are excluded entirely (never shown).
pub fn filter_entries(entries: &[ValidationEntry]) -> FilteredEntries {
    let mut visible = Vec::new();
    let mut hidden = Vec::new();

    for entry in entries {
        match entry.tier {
            ValidationTier::Validated => {
                visible.push(VisibleEntry {
                    component_id: entry.component_id.clone(),
                    tier: entry.tier,
                    preselected: true,
                });
            }
            ValidationTier::Candidate => {
                visible.push(VisibleEntry {
                    component_id: entry.component_id.clone(),
                    tier: entry.tier,
                    preselected: false,
                });
            }
            ValidationTier::Experimental => {
                hidden.push(entry.clone());
            }
            ValidationTier::Blocked => {
                // Never shown — silently excluded
            }
        }
    }

    FilteredEntries { visible, hidden }
}

/// Apply tier filtering with `include_experimental` flag.
///
/// When `include_experimental` is true, experimental entries are promoted
/// to the visible list (but never preselected).
pub fn filter_entries_with_experimental(
    entries: &[ValidationEntry],
    include_experimental: bool,
) -> FilteredEntries {
    if include_experimental {
        let mut visible = Vec::new();

        for entry in entries {
            match entry.tier {
                ValidationTier::Validated => {
                    visible.push(VisibleEntry {
                        component_id: entry.component_id.clone(),
                        tier: entry.tier,
                        preselected: true,
                    });
                }
                ValidationTier::Candidate => {
                    visible.push(VisibleEntry {
                        component_id: entry.component_id.clone(),
                        tier: entry.tier,
                        preselected: false,
                    });
                }
                ValidationTier::Experimental => {
                    visible.push(VisibleEntry {
                        component_id: entry.component_id.clone(),
                        tier: entry.tier,
                        preselected: false,
                    });
                }
                ValidationTier::Blocked => {
                    // Never shown
                }
            }
        }

        FilteredEntries {
            visible,
            hidden: vec![],
        }
    } else {
        filter_entries(entries)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // VAL-CORE-011: Legal transitions
    // =======================================================================

    #[test]
    fn test_validation_legal_transition_validated_to_candidate() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("pytorch", ValidationTier::Validated);
        assert!(sm.transition("pytorch", ValidationTier::Candidate).is_ok());
        assert_eq!(sm.tier("pytorch"), ValidationTier::Candidate);
    }

    #[test]
    fn test_validation_legal_transition_candidate_to_validated() {
        let mut sm = ValidationStateMachine::new();
        sm.register("pytorch"); // defaults to Candidate
        assert!(sm.transition("pytorch", ValidationTier::Validated).is_ok());
        assert_eq!(sm.tier("pytorch"), ValidationTier::Validated);
    }

    #[test]
    fn test_validation_legal_transition_candidate_to_experimental() {
        let mut sm = ValidationStateMachine::new();
        sm.register("triton"); // defaults to Candidate
        assert!(sm
            .transition("triton", ValidationTier::Experimental)
            .is_ok());
        assert_eq!(sm.tier("triton"), ValidationTier::Experimental);
    }

    #[test]
    fn test_validation_legal_transition_experimental_to_candidate() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("triton", ValidationTier::Experimental);
        assert!(sm.transition("triton", ValidationTier::Candidate).is_ok());
        assert_eq!(sm.tier("triton"), ValidationTier::Candidate);
    }

    #[test]
    fn test_validation_legal_transition_any_to_blocked() {
        for from_tier in ValidationTier::all() {
            let mut sm = ValidationStateMachine::new();
            sm.register_with_tier("comp", *from_tier);
            let result = sm.transition("comp", ValidationTier::Blocked);
            assert!(
                result.is_ok(),
                "transition from {:?} to Blocked should be legal",
                from_tier
            );
            assert_eq!(sm.tier("comp"), ValidationTier::Blocked);
        }
    }

    #[test]
    fn test_validation_self_transitions_are_legal() {
        for tier in ValidationTier::all() {
            let mut sm = ValidationStateMachine::new();
            sm.register_with_tier("comp", *tier);
            let result = sm.transition("comp", *tier);
            assert!(
                result.is_ok(),
                "self-transition for {:?} should be legal",
                tier
            );
        }
    }

    // =======================================================================
    // VAL-CORE-011: Illegal transitions
    // =======================================================================

    #[test]
    fn test_validation_illegal_transition_experimental_to_validated() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("triton", ValidationTier::Experimental);
        let result = sm.transition("triton", ValidationTier::Validated);
        assert!(
            result.is_err(),
            "Experimental → Validated should be illegal"
        );
        let err = result.unwrap_err();
        assert_eq!(err.from, ValidationTier::Experimental);
        assert_eq!(err.to, ValidationTier::Validated);
        // Verify state was not changed
        assert_eq!(sm.tier("triton"), ValidationTier::Experimental);
    }

    #[test]
    fn test_validation_illegal_transition_blocked_to_validated() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Blocked);
        let result = sm.transition("comp", ValidationTier::Validated);
        assert!(result.is_err(), "Blocked → Validated should be illegal");
        assert_eq!(result.unwrap_err().from, ValidationTier::Blocked);
        assert_eq!(sm.tier("comp"), ValidationTier::Blocked);
    }

    #[test]
    fn test_validation_illegal_transition_blocked_to_candidate() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Blocked);
        let result = sm.transition("comp", ValidationTier::Candidate);
        assert!(result.is_err(), "Blocked → Candidate should be illegal");
    }

    #[test]
    fn test_validation_illegal_transition_blocked_to_experimental() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Blocked);
        let result = sm.transition("comp", ValidationTier::Experimental);
        assert!(result.is_err(), "Blocked → Experimental should be illegal");
    }

    #[test]
    fn test_validation_blocked_is_terminal() {
        // Once blocked, no transition out should work
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Blocked);
        for target in ValidationTier::all() {
            if *target == ValidationTier::Blocked {
                continue; // self-transition is ok
            }
            assert!(
                !sm.can_transition("comp", *target),
                "Blocked → {:?} should be illegal",
                target
            );
        }
    }

    #[test]
    fn test_validation_error_display() {
        let err = TransitionError {
            from: ValidationTier::Blocked,
            to: ValidationTier::Validated,
            reason: "Blocked is a terminal state".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Blocked"));
        assert!(msg.contains("Validated"));
    }

    // =======================================================================
    // VAL-CORE-012: Initial state is Candidate
    // =======================================================================

    #[test]
    fn test_validation_new_component_defaults_to_candidate() {
        let mut sm = ValidationStateMachine::new();
        sm.register("new-component");
        assert_eq!(sm.tier("new-component"), ValidationTier::Candidate);
    }

    #[test]
    fn test_validation_unregistered_component_returns_candidate() {
        let sm = ValidationStateMachine::new();
        assert_eq!(sm.tier("nonexistent"), ValidationTier::Candidate);
    }

    #[test]
    fn test_validation_entry_new_defaults_to_candidate() {
        let entry = ValidationEntry::new("test-comp");
        assert_eq!(entry.tier, ValidationTier::Candidate);
        assert_eq!(entry.component_id, "test-comp");
    }

    #[test]
    fn test_validation_register_existing_is_noop() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Validated);
        sm.register("comp"); // should not overwrite
        assert_eq!(sm.tier("comp"), ValidationTier::Validated);
    }

    // =======================================================================
    // VAL-CORE-013: Tier filtering — UI visibility and preselection
    // =======================================================================

    #[test]
    fn test_validation_filter_validated_visible_and_preselected() {
        let entries = vec![ValidationEntry::with_tier(
            "pytorch",
            ValidationTier::Validated,
        )];
        let filtered = filter_entries(&entries);
        assert_eq!(filtered.visible.len(), 1);
        assert_eq!(filtered.hidden.len(), 0);
        assert_eq!(filtered.visible[0].component_id, "pytorch");
        assert!(filtered.visible[0].preselected);
        assert_eq!(filtered.visible[0].tier, ValidationTier::Validated);
    }

    #[test]
    fn test_validation_filter_candidate_visible_not_preselected() {
        let entries = vec![ValidationEntry::with_tier(
            "triton",
            ValidationTier::Candidate,
        )];
        let filtered = filter_entries(&entries);
        assert_eq!(filtered.visible.len(), 1);
        assert_eq!(filtered.hidden.len(), 0);
        assert!(!filtered.visible[0].preselected);
    }

    #[test]
    fn test_validation_filter_experimental_hidden_from_default_view() {
        let entries = vec![ValidationEntry::with_tier(
            "deepspeed",
            ValidationTier::Experimental,
        )];
        let filtered = filter_entries(&entries);
        assert_eq!(filtered.visible.len(), 0);
        assert_eq!(filtered.hidden.len(), 1);
        assert_eq!(filtered.hidden[0].component_id, "deepspeed");
    }

    #[test]
    fn test_validation_filter_blocked_never_shown() {
        let entries = vec![ValidationEntry::with_tier(
            "bad-comp",
            ValidationTier::Blocked,
        )];
        let filtered = filter_entries(&entries);
        assert_eq!(filtered.visible.len(), 0);
        assert_eq!(filtered.hidden.len(), 0);
    }

    #[test]
    fn test_validation_filter_all_tiers_combined() {
        let entries = vec![
            ValidationEntry::with_tier("rocm", ValidationTier::Validated),
            ValidationEntry::with_tier("triton", ValidationTier::Candidate),
            ValidationEntry::with_tier("deepspeed", ValidationTier::Experimental),
            ValidationEntry::with_tier("bad-comp", ValidationTier::Blocked),
        ];
        let filtered = filter_entries(&entries);

        // 2 visible (Validated + Candidate), 1 hidden (Experimental), 1 excluded (Blocked)
        assert_eq!(filtered.visible.len(), 2);
        assert_eq!(filtered.hidden.len(), 1);

        // Check preselection
        let rocm = filtered
            .visible
            .iter()
            .find(|e| e.component_id == "rocm")
            .unwrap();
        assert!(rocm.preselected);
        let triton = filtered
            .visible
            .iter()
            .find(|e| e.component_id == "triton")
            .unwrap();
        assert!(!triton.preselected);
    }

    #[test]
    fn test_validation_filter_include_experimental_flag() {
        let entries = vec![
            ValidationEntry::with_tier("rocm", ValidationTier::Validated),
            ValidationEntry::with_tier("deepspeed", ValidationTier::Experimental),
        ];

        // Without flag: experimental is hidden
        let filtered = filter_entries_with_experimental(&entries, false);
        assert_eq!(filtered.visible.len(), 1);
        assert_eq!(filtered.hidden.len(), 1);

        // With flag: experimental is visible but not preselected
        let filtered = filter_entries_with_experimental(&entries, true);
        assert_eq!(filtered.visible.len(), 2);
        assert_eq!(filtered.hidden.len(), 0);
        let deepspeed = filtered
            .visible
            .iter()
            .find(|e| e.component_id == "deepspeed")
            .unwrap();
        assert!(!deepspeed.preselected);
    }

    // =======================================================================
    // Additional coverage
    // =======================================================================

    #[test]
    fn test_validation_entry_visibility() {
        let validated = ValidationEntry::with_tier("a", ValidationTier::Validated);
        let candidate = ValidationEntry::with_tier("b", ValidationTier::Candidate);
        let experimental = ValidationEntry::with_tier("c", ValidationTier::Experimental);
        let blocked = ValidationEntry::with_tier("d", ValidationTier::Blocked);

        assert!(validated.is_visible());
        assert!(candidate.is_visible());
        assert!(!experimental.is_visible());
        assert!(!blocked.is_visible());

        // With experimental flag
        assert!(validated.is_visible_with_experimental());
        assert!(candidate.is_visible_with_experimental());
        assert!(experimental.is_visible_with_experimental());
        assert!(!blocked.is_visible_with_experimental());
    }

    #[test]
    fn test_validation_entry_preselection() {
        let validated = ValidationEntry::with_tier("a", ValidationTier::Validated);
        let candidate = ValidationEntry::with_tier("b", ValidationTier::Candidate);
        let experimental = ValidationEntry::with_tier("c", ValidationTier::Experimental);
        let blocked = ValidationEntry::with_tier("d", ValidationTier::Blocked);

        assert!(validated.is_preselected());
        assert!(!candidate.is_preselected());
        assert!(!experimental.is_preselected());
        assert!(!blocked.is_preselected());
    }

    #[test]
    fn test_validation_entry_serde_roundtrip() {
        let entry = ValidationEntry::with_tier("pytorch", ValidationTier::Validated);
        let json = serde_json::to_string(&entry).unwrap();
        let back: ValidationEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn test_validation_can_transition_query() {
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Validated);
        assert!(sm.can_transition("comp", ValidationTier::Candidate));
        assert!(sm.can_transition("comp", ValidationTier::Blocked));
        assert!(sm.can_transition("comp", ValidationTier::Validated)); // self
        assert!(!sm.can_transition("comp", ValidationTier::Experimental));
    }

    #[test]
    fn test_validation_force_set() {
        let mut sm = ValidationStateMachine::new();
        sm.register("comp");
        // Force-set to Blocked (normally legal, but this bypasses checks)
        sm.force_set("comp", ValidationTier::Blocked);
        assert_eq!(sm.tier("comp"), ValidationTier::Blocked);
    }

    #[test]
    fn test_validation_entries_iterator() {
        let mut sm = ValidationStateMachine::new();
        sm.register("a");
        sm.register_with_tier("b", ValidationTier::Validated);
        assert_eq!(sm.entries().len(), 2);
        assert_eq!(sm.entries().get("a"), Some(&ValidationTier::Candidate));
        assert_eq!(sm.entries().get("b"), Some(&ValidationTier::Validated));
    }

    #[test]
    fn test_validation_multi_step_transition() {
        // Experimental → Candidate → Validated (legal path)
        let mut sm = ValidationStateMachine::new();
        sm.register_with_tier("comp", ValidationTier::Experimental);

        // Step 1: Experimental → Candidate
        assert!(sm.transition("comp", ValidationTier::Candidate).is_ok());
        assert_eq!(sm.tier("comp"), ValidationTier::Candidate);

        // Step 2: Candidate → Validated
        assert!(sm.transition("comp", ValidationTier::Validated).is_ok());
        assert_eq!(sm.tier("comp"), ValidationTier::Validated);
    }

    #[test]
    fn test_validation_filtered_entries_serde_roundtrip() {
        let filtered = FilteredEntries {
            visible: vec![VisibleEntry {
                component_id: "rocm".to_string(),
                tier: ValidationTier::Validated,
                preselected: true,
            }],
            hidden: vec![ValidationEntry::with_tier(
                "exp",
                ValidationTier::Experimental,
            )],
        };
        let json = serde_json::to_string(&filtered).unwrap();
        let back: FilteredEntries = serde_json::from_str(&json).unwrap();
        assert_eq!(filtered, back);
    }

    #[test]
    fn test_validation_transition_error_is_std_error() {
        let err = TransitionError {
            from: ValidationTier::Blocked,
            to: ValidationTier::Validated,
            reason: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err; // verify trait object compat
    }
}
