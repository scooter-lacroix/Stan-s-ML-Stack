//! Update planner — computes eligible updates and classifies by risk.
//!
//! This module implements:
//! - [`UpdateClassification`] — safe/guarded/blocked/candidate/experimental
//! - [`PlannerItem`] — an update candidate with classification, selection state, rationale
//! - [`UpdatePlanner`] — builds an update plan from scan results and manifest data
//!
//! # Classification Rules
//!
//! | Classification | Criteria |
//! |----------------|----------|
//! | `Safe`         | Patch-only bump, Validated tier, no new dependencies, compatible hardware |
//! | `Guarded`      | Minor bump or new dependencies or config changes |
//! | `Blocked`      | Incompatible hardware, missing executor, manifest-declared incompatibility |
//! | `Candidate`    | Major bump that passes all compatibility checks |
//! | `Experimental` | Component marked `Experimental` validation tier |
//!
//! # Selection Defaults
//!
//! | Classification | Visible | Preselected |
//! |----------------|---------|-------------|
//! | Safe           | Yes     | Yes         |
//! | Guarded        | Yes     | No          |
//! | Candidate      | Yes     | No          |
//! | Experimental   | No*     | No          |
//! | Blocked        | No      | No          |
//!
//! \* Visible only with `--include-experimental` flag.

use crate::core::manifest::{Manifest, ManifestComponent};
use crate::core::plan::PlanItem;
use crate::core::types::{ExecutorKind, ValidationTier};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Update Classification
// ---------------------------------------------------------------------------

/// Classification of an update's risk and suitability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdateClassification {
    /// Patch-only bump, validated, no new deps, compatible hardware.
    Safe,
    /// Minor bump, new dependencies, or config changes.
    Guarded,
    /// Incompatible hardware, missing executor, or manifest-declared incompatibility.
    Blocked,
    /// Major version bump that passes all compatibility checks.
    Candidate,
    /// Experimental validation tier component.
    Experimental,
}

impl UpdateClassification {
    /// Human-readable label for this classification.
    pub fn label(self) -> &'static str {
        match self {
            UpdateClassification::Safe => "safe",
            UpdateClassification::Guarded => "guarded",
            UpdateClassification::Blocked => "blocked",
            UpdateClassification::Candidate => "candidate",
            UpdateClassification::Experimental => "experimental",
        }
    }

    /// Whether updates with this classification are visible by default.
    pub fn is_visible(self) -> bool {
        !matches!(
            self,
            UpdateClassification::Blocked | UpdateClassification::Experimental
        )
    }

    /// Whether updates with this classification are preselected by default.
    pub fn is_preselected(self) -> bool {
        matches!(self, UpdateClassification::Safe)
    }
}

impl fmt::Display for UpdateClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Planner Item (extends PlanItem with classification)
// ---------------------------------------------------------------------------

/// An update candidate produced by the planner.
///
/// Extends the base [`PlanItem`] with classification, visibility, and
/// compatibility information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannerItem {
    /// Base plan item with risk tier, dependencies, etc.
    #[serde(flatten)]
    pub plan_item: PlanItem,
    /// Update classification (safe/guarded/blocked/candidate/experimental).
    pub classification: UpdateClassification,
    /// Whether this item is visible in the default plan view.
    pub visible: bool,
    /// Whether this item is preselected for apply.
    pub selected: bool,
    /// Reason for the classification (human-readable).
    pub classification_reason: String,
    /// Whether this component requires hardware compatibility check.
    pub requires_hardware_check: bool,
    /// Minimum ROCm version required for this update (empty = any).
    pub min_rocm_version: String,
}

// ---------------------------------------------------------------------------
// Compatibility Context
// ---------------------------------------------------------------------------

/// Context about the system's capabilities used for compatibility checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompatibilityContext {
    /// Current ROCm version (empty if not installed).
    pub rocm_version: String,
    /// Active ROCm channel (Legacy/Stable/Latest).
    pub rocm_channel: String,
    /// Available executor kinds on this platform.
    pub available_executors: HashSet<ExecutorKind>,
    /// Component IDs that are currently installed.
    pub installed_components: HashSet<String>,
    /// Version map: component_id → installed version string.
    pub installed_versions: HashMap<String, String>,
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_architecture: String,
    /// Current runtime version of the binary.
    pub runtime_version: String,
}

impl CompatibilityContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a component is installed.
    pub fn is_installed(&self, component_id: &str) -> bool {
        self.installed_components.contains(component_id)
    }

    /// Get the installed version of a component.
    pub fn installed_version(&self, component_id: &str) -> Option<&str> {
        self.installed_versions
            .get(component_id)
            .map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Planner Options
// ---------------------------------------------------------------------------

/// Options controlling planner behavior.
#[derive(Debug, Clone, Default)]
pub struct PlannerOptions {
    /// Only scan and produce a plan, don't apply.
    pub scan_only: bool,
    /// Apply all safe-classified updates without prompting.
    pub all_safe: bool,
    /// Include experimental components in the plan.
    pub include_experimental: bool,
    /// Specific component IDs to update (empty = all eligible).
    pub target_components: Vec<String>,
    /// Force JSON output mode.
    pub json_output: bool,
}

// ---------------------------------------------------------------------------
// Planner Error
// ---------------------------------------------------------------------------

/// Errors that can occur during planning.
#[derive(Debug, Clone, PartialEq)]
pub enum PlannerError {
    /// A targeted component is blocked.
    ComponentBlocked {
        component_id: String,
        reason: String,
    },
    /// A targeted component is unknown.
    ComponentUnknown { component_id: String },
    /// The runtime version is too old for the manifest.
    RuntimeTooOld { current: String, required: String },
    /// Dependency conflict detected.
    DependencyConflict {
        component_id: String,
        dependency_id: String,
        reason: String,
    },
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlannerError::ComponentBlocked {
                component_id,
                reason,
            } => {
                write!(f, "component '{component_id}' is blocked: {reason}")
            }
            PlannerError::ComponentUnknown { component_id } => {
                write!(f, "unknown component: '{component_id}'")
            }
            PlannerError::RuntimeTooOld { current, required } => {
                write!(
                    f,
                    "runtime version {current} is too old; requires {required}. Run `rusty upgrade` first."
                )
            }
            PlannerError::DependencyConflict {
                component_id,
                dependency_id,
                reason,
            } => {
                write!(
                    f,
                    "dependency conflict for '{component_id}' on '{dependency_id}': {reason}"
                )
            }
        }
    }
}

impl std::error::Error for PlannerError {}

// ---------------------------------------------------------------------------
// Update Planner
// ---------------------------------------------------------------------------

/// Builds an update plan from manifest data and system context.
///
/// The planner:
/// 1. Resolves the manifest (baseline + overlay)
/// 2. For each component, runs compatibility checks
/// 3. Classifies each update (safe/guarded/blocked/candidate/experimental)
/// 4. Applies selection defaults
/// 5. Enforces dependency rules
/// 6. Returns a filtered, ordered plan
pub struct UpdatePlanner;

impl UpdatePlanner {
    /// Create a new planner instance.
    pub fn new() -> Self {
        Self
    }

    /// Build an update plan from the given manifest and compatibility context.
    ///
    /// Applies classification rules, selection defaults, and dependency ordering.
    /// Returns a plan with [`PlannerItem`] entries.
    pub fn build_plan(
        &self,
        manifest: &Manifest,
        context: &CompatibilityContext,
        options: &PlannerOptions,
    ) -> Result<Vec<PlannerItem>, PlannerError> {
        // Check runtime version compatibility
        self.check_runtime_version(manifest, context)?;

        let mut items = Vec::new();

        for component in &manifest.components {
            // If specific targets were given, only include those
            if !options.target_components.is_empty() {
                if !options.target_components.contains(&component.id) {
                    continue;
                }
                // Verify the targeted component is known
                if !options
                    .target_components
                    .iter()
                    .all(|t| manifest.components.iter().any(|c| c.id == *t))
                {
                    // Check if any target is not in the manifest
                    for target in &options.target_components {
                        if !manifest.components.iter().any(|c| c.id == *target) {
                            return Err(PlannerError::ComponentUnknown {
                                component_id: target.clone(),
                            });
                        }
                    }
                }
            }

            // Classify the update
            let classification = self.classify_update(component, context);

            // If blocked, handle differently
            if classification == UpdateClassification::Blocked {
                // If explicitly targeted, return error
                if options.target_components.contains(&component.id) {
                    let reason = self.block_reason(component, context);
                    return Err(PlannerError::ComponentBlocked {
                        component_id: component.id.clone(),
                        reason,
                    });
                }
                // Otherwise, include in plan but mark as blocked
                items.push(self.build_planner_item(component, classification, context));
                continue;
            }

            // Filter experimental unless flag is set
            if classification == UpdateClassification::Experimental && !options.include_experimental
            {
                continue;
            }

            items.push(self.build_planner_item(component, classification, context));
        }

        // If targeting specific components, check all were found
        if !options.target_components.is_empty() {
            let found_ids: HashSet<&str> = items
                .iter()
                .map(|i| i.plan_item.component_id.as_str())
                .collect();
            for target in &options.target_components {
                if !found_ids.contains(target.as_str()) {
                    return Err(PlannerError::ComponentUnknown {
                        component_id: target.clone(),
                    });
                }
            }
        }

        // Apply --all-safe: only keep safe items if flag is set
        if options.all_safe {
            items.retain(|i| i.classification == UpdateClassification::Safe);
        }

        // Enforce dependency rules
        self.enforce_dependency_rules(&mut items)?;

        Ok(items)
    }

    /// Classify a single component update.
    fn classify_update(
        &self,
        component: &ManifestComponent,
        context: &CompatibilityContext,
    ) -> UpdateClassification {
        // Experimental tier always classifies as experimental
        if component.validation_tier == ValidationTier::Experimental {
            return UpdateClassification::Experimental;
        }

        // Blocked tier always classifies as blocked
        if component.validation_tier == ValidationTier::Blocked {
            return UpdateClassification::Blocked;
        }

        // Check hardware compatibility
        if !self.is_hardware_compatible(component, context) {
            return UpdateClassification::Blocked;
        }

        // Check executor availability
        if !self.is_executor_available(component, context) {
            return UpdateClassification::Blocked;
        }

        // Check ROCm version compatibility
        if !self.is_rocm_compatible(component, context) {
            return UpdateClassification::Blocked;
        }

        // Classify based on version delta
        let current_version = context.installed_version(&component.id).unwrap_or("");

        // Not installed → Candidate (not safe to auto-install)
        if current_version.is_empty() {
            return UpdateClassification::Candidate;
        }

        // Same version → Safe (reinstall)
        if current_version == component.version {
            return UpdateClassification::Safe;
        }

        // Parse version delta
        match self.version_bump_level(current_version, &component.version) {
            BumpLevel::Patch => {
                // Patch bump: safe if validated tier, guarded otherwise
                if component.validation_tier == ValidationTier::Validated {
                    UpdateClassification::Safe
                } else {
                    UpdateClassification::Guarded
                }
            }
            BumpLevel::Minor => UpdateClassification::Guarded,
            BumpLevel::Major => UpdateClassification::Candidate,
            BumpLevel::Unknown => {
                // Can't determine version delta → guarded
                UpdateClassification::Guarded
            }
        }
    }

    /// Build a planner item from a manifest component and its classification.
    fn build_planner_item(
        &self,
        component: &ManifestComponent,
        classification: UpdateClassification,
        context: &CompatibilityContext,
    ) -> PlannerItem {
        let current_version = context
            .installed_version(&component.id)
            .unwrap_or("")
            .to_string();

        let visible = classification.is_visible();
        let selected = classification.is_preselected();

        let classification_reason = self.classification_reason(component, classification, context);

        // Dependencies: for now, derive from known patterns
        let dependencies = self.derive_dependencies(&component.id);

        PlannerItem {
            plan_item: PlanItem::new(
                &component.id,
                &current_version,
                &component.version,
                component.validation_tier,
                selected,
                &classification_reason,
                dependencies,
                true, // isolation_safe by default
            ),
            classification,
            visible,
            selected,
            classification_reason,
            requires_hardware_check: self.requires_hardware_check(&component.id),
            min_rocm_version: component.min_rocm_version.clone(),
        }
    }

    /// Check hardware compatibility for a component.
    fn is_hardware_compatible(
        &self,
        component: &ManifestComponent,
        context: &CompatibilityContext,
    ) -> bool {
        // Check ROCm channel compatibility at the hardware level
        if !component.compatible_channels.is_empty()
            && self.requires_rocm(&component.id)
            && !component
                .compatible_channels
                .iter()
                .any(|ch| ch.eq_ignore_ascii_case(&context.rocm_channel))
        {
            return false;
        }
        true
    }

    /// Check if an executor is available for a component.
    fn is_executor_available(
        &self,
        _component: &ManifestComponent,
        context: &CompatibilityContext,
    ) -> bool {
        // At minimum, we need LegacyScript or Rust executor
        context
            .available_executors
            .contains(&ExecutorKind::LegacyScript)
            || context.available_executors.contains(&ExecutorKind::Rust)
    }

    /// Check ROCm version compatibility.
    fn is_rocm_compatible(
        &self,
        component: &ManifestComponent,
        context: &CompatibilityContext,
    ) -> bool {
        // Components that don't need ROCm are always compatible
        if !self.requires_rocm(&component.id) {
            return true;
        }

        // If ROCm is not installed, ROCm-requiring components are blocked
        if context.rocm_version.is_empty() {
            return false;
        }

        // Check min ROCm version if specified in manifest
        if !component.min_rocm_version.is_empty()
            && !version_gte(&context.rocm_version, &component.min_rocm_version)
        {
            return false;
        }

        // Check ROCm channel compatibility
        if !component.compatible_channels.is_empty()
            && !component
                .compatible_channels
                .iter()
                .any(|ch| ch.eq_ignore_ascii_case(&context.rocm_channel))
        {
            return false;
        }

        true
    }

    /// Check if a component requires ROCm.
    fn requires_rocm(&self, component_id: &str) -> bool {
        matches!(
            component_id,
            "rocm"
                | "rocm-smi"
                | "pytorch"
                | "triton"
                | "onnx"
                | "migraphx"
                | "flash-attn"
                | "rccl"
                | "vllm"
                | "aiter"
                | "deepspeed"
                | "bitsandbytes"
        )
    }

    /// Check if a component requires hardware compatibility check.
    fn requires_hardware_check(&self, component_id: &str) -> bool {
        self.requires_rocm(component_id)
    }

    /// Get the reason a component is blocked.
    fn block_reason(
        &self,
        component: &ManifestComponent,
        context: &CompatibilityContext,
    ) -> String {
        if component.validation_tier == ValidationTier::Blocked {
            return "component is marked as blocked in manifest".to_string();
        }

        if !self.is_hardware_compatible(component, context) {
            return format!(
                "incompatible with ROCm channel '{}' (requires: {})",
                context.rocm_channel,
                component.compatible_channels.join(", ")
            );
        }

        if !self.is_executor_available(component, context) {
            return "no available executor for this platform".to_string();
        }

        if !self.is_rocm_compatible(component, context) {
            if context.rocm_version.is_empty() {
                return "requires ROCm but ROCm is not installed".to_string();
            }
            if !component.min_rocm_version.is_empty() {
                return format!(
                    "requires ROCm >= {} but installed version is {}",
                    component.min_rocm_version, context.rocm_version
                );
            }
            return format!(
                "incompatible with installed ROCm version ({})",
                context.rocm_version
            );
        }

        "unknown reason".to_string()
    }

    /// Get the classification reason string.
    fn classification_reason(
        &self,
        component: &ManifestComponent,
        classification: UpdateClassification,
        context: &CompatibilityContext,
    ) -> String {
        let current = context.installed_version(&component.id).unwrap_or("");

        match classification {
            UpdateClassification::Safe => {
                if current == component.version {
                    format!("reinstall v{} (validated)", component.version)
                } else {
                    format!(
                        "patch update {} → {} (validated, no new deps)",
                        current, component.version
                    )
                }
            }
            UpdateClassification::Guarded => {
                format!(
                    "minor update {} → {} (new dependencies or config changes)",
                    current, component.version
                )
            }
            UpdateClassification::Blocked => self.block_reason(component, context),
            UpdateClassification::Candidate => {
                if current.is_empty() {
                    format!("new install v{}", component.version)
                } else {
                    format!(
                        "major update {} → {} (compatible)",
                        current, component.version
                    )
                }
            }
            UpdateClassification::Experimental => {
                format!("experimental component v{}", component.version)
            }
        }
    }

    /// Derive known dependencies for a component.
    fn derive_dependencies(&self, component_id: &str) -> Vec<String> {
        match component_id {
            "pytorch" => vec!["rocm".to_string()],
            "triton" => vec!["pytorch".to_string()],
            "flash-attn" => vec!["pytorch".to_string()],
            "deepspeed" => vec!["pytorch".to_string()],
            "vllm" => vec!["pytorch".to_string()],
            "onnx" => vec!["rocm".to_string()],
            "migraphx" => vec!["rocm".to_string()],
            "bitsandbytes" => vec!["pytorch".to_string()],
            "aiter" => vec!["pytorch".to_string()],
            "mpi4py" => vec![],
            "wandb" => vec![],
            "comfyui" => vec!["pytorch".to_string()],
            "vllm-studio" => vec!["vllm".to_string()],
            "textgen" => vec!["pytorch".to_string()],
            "rocm" => vec![],
            "rocm-smi" => vec!["rocm".to_string()],
            "permanent-env" => vec![],
            _ => vec![],
        }
    }

    /// Derive known dependencies for a component.
    fn version_bump_level(&self, current: &str, proposed: &str) -> BumpLevel {
        let cur_parts = parse_version_parts(current);
        let pro_parts = parse_version_parts(proposed);

        match (cur_parts, pro_parts) {
            (Some(cur), Some(pro)) if !cur.is_empty() && !pro.is_empty() => {
                if pro[0] > cur[0] {
                    BumpLevel::Major
                } else if cur.len() >= 2 && pro.len() >= 2 && pro[1] > cur[1] {
                    BumpLevel::Minor
                } else {
                    BumpLevel::Patch
                }
            }
            _ => BumpLevel::Unknown,
        }
    }

    /// Check runtime version against manifest requirements.
    fn check_runtime_version(
        &self,
        manifest: &Manifest,
        context: &CompatibilityContext,
    ) -> Result<(), PlannerError> {
        if manifest.min_runtime_version.is_empty() {
            return Ok(());
        }

        if context.runtime_version.is_empty() {
            return Ok(());
        }

        if !version_gte(&context.runtime_version, &manifest.min_runtime_version) {
            return Err(PlannerError::RuntimeTooOld {
                current: context.runtime_version.clone(),
                required: manifest.min_runtime_version.clone(),
            });
        }

        Ok(())
    }

    /// Enforce dependency rules on the plan items.
    ///
    /// If component A depends on B and both are selected:
    /// - If B is deselected, A must also be deselected (or B re-selected)
    fn enforce_dependency_rules(&self, items: &mut [PlannerItem]) -> Result<(), PlannerError> {
        // Auto-select dependencies of selected items
        // We need to collect the deps that need auto-selection first, then apply
        let mut changed = true;
        while changed {
            changed = false;

            // Collect (index, needs_select) pairs for deps
            let deps_to_select: Vec<String> = items
                .iter()
                .filter(|i| i.selected)
                .flat_map(|i| i.plan_item.dependencies.clone())
                .collect();

            for dep_id in &deps_to_select {
                if let Some(dep_item) = items
                    .iter_mut()
                    .find(|i| i.plan_item.component_id == *dep_id)
                {
                    if !dep_item.selected
                        && dep_item.classification != UpdateClassification::Blocked
                    {
                        dep_item.selected = true;
                        changed = true;
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for UpdatePlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Version bump level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BumpLevel {
    Patch,
    Minor,
    Major,
    Unknown,
}

/// Parse a version string into numeric parts.
fn parse_version_parts(version: &str) -> Option<Vec<u32>> {
    let base = version.split('-').next()?;
    base.split('.')
        .map(|s| s.parse::<u32>().ok())
        .collect::<Option<Vec<_>>>()
}

/// Compare two version strings: returns true if `actual >= required`.
///
/// Handles versions like "7.2.1", "7.2", "7". Missing parts are treated as 0.
fn version_gte(actual: &str, required: &str) -> bool {
    let act = parse_version_parts(actual).unwrap_or_default();
    let req = parse_version_parts(required).unwrap_or_default();

    let max_len = act.len().max(req.len());
    for i in 0..max_len {
        let a = act.get(i).copied().unwrap_or(0);
        let r = req.get(i).copied().unwrap_or(0);
        if a > r {
            return true;
        }
        if a < r {
            return false;
        }
    }
    true // equal
}

// ---------------------------------------------------------------------------
// Scan Result JSON output
// ---------------------------------------------------------------------------

/// Machine-readable scan output for non-TTY / JSON mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanOutput {
    /// Components detected as installed.
    pub installed: Vec<InstalledComponent>,
    /// Manifest source used for planning.
    pub manifest_source: String,
    /// ROCm channel in use.
    pub rocm_channel: String,
    /// ROCm version detected.
    pub rocm_version: String,
    /// GPU architecture detected.
    pub gpu_architecture: String,
}

/// An installed component in scan output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstalledComponent {
    pub id: String,
    pub version: String,
    pub status: String,
}

// ---------------------------------------------------------------------------
// Plan Output JSON
// ---------------------------------------------------------------------------

/// Machine-readable plan output for non-TTY / JSON mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanOutput {
    /// The plan items.
    pub plan: Vec<PlannerItemOutput>,
    /// Summary statistics.
    pub summary: PlanSummary,
}

/// A single item in the plan output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannerItemOutput {
    pub component_id: String,
    pub current_version: String,
    pub proposed_version: String,
    pub classification: String,
    pub risk_tier: String,
    pub selected: bool,
    pub visible: bool,
    pub rationale: String,
    pub dependencies: Vec<String>,
}

impl From<&PlannerItem> for PlannerItemOutput {
    fn from(item: &PlannerItem) -> Self {
        Self {
            component_id: item.plan_item.component_id.clone(),
            current_version: item.plan_item.current_version.clone(),
            proposed_version: item.plan_item.proposed_version.clone(),
            classification: item.classification.label().to_string(),
            risk_tier: item.plan_item.risk_tier.label().to_string(),
            selected: item.selected,
            visible: item.visible,
            rationale: item.plan_item.rationale.clone(),
            dependencies: item.plan_item.dependencies.clone(),
        }
    }
}

/// Summary statistics for the plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanSummary {
    pub total: usize,
    pub safe: usize,
    pub guarded: usize,
    pub candidate: usize,
    pub experimental: usize,
    pub blocked: usize,
    pub selected: usize,
}

impl PlanSummary {
    /// Compute summary from a list of planner items.
    pub fn from_items(items: &[PlannerItem]) -> Self {
        Self {
            total: items.len(),
            safe: items
                .iter()
                .filter(|i| i.classification == UpdateClassification::Safe)
                .count(),
            guarded: items
                .iter()
                .filter(|i| i.classification == UpdateClassification::Guarded)
                .count(),
            candidate: items
                .iter()
                .filter(|i| i.classification == UpdateClassification::Candidate)
                .count(),
            experimental: items
                .iter()
                .filter(|i| i.classification == UpdateClassification::Experimental)
                .count(),
            blocked: items
                .iter()
                .filter(|i| i.classification == UpdateClassification::Blocked)
                .count(),
            selected: items.iter().filter(|i| i.selected).count(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::{Manifest, ManifestComponent};
    use crate::core::types::{Category, ValidationTier};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_component(id: &str, version: &str, tier: ValidationTier) -> ManifestComponent {
        ManifestComponent {
            id: id.to_string(),
            version: version.to_string(),
            script: format!("install_{id}.sh"),
            category: Category::Core,
            validation_tier: tier,
            min_rocm_version: String::new(),
            compatible_channels: vec![],
        }
    }

    fn make_manifest(components: Vec<ManifestComponent>) -> Manifest {
        Manifest {
            schema_version: 2,
            sequence: 1,
            generated_at: String::new(),
            expires_at: None,
            min_runtime_version: String::new(),
            components,
            signature: None,
        }
    }

    fn make_context() -> CompatibilityContext {
        let mut ctx = CompatibilityContext::new();
        ctx.rocm_version = "7.2.1".to_string();
        ctx.rocm_channel = "latest".to_string();
        ctx.gpu_architecture = "gfx1100".to_string();
        ctx.available_executors = HashSet::from([
            ExecutorKind::LegacyScript,
            ExecutorKind::Rust,
            ExecutorKind::ExternalPackageManager,
        ]);
        ctx
    }

    fn planner() -> UpdatePlanner {
        UpdatePlanner::new()
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-002: Safe updates (patch-only, validated) classified correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_safe_classification_patch_bump_validated() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let component = make_component("pytorch", "2.4.1", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Safe);
    }

    #[test]
    fn test_safe_classification_same_version_reinstall() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let component = make_component("pytorch", "2.4.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Safe);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-003: Guarded updates (minor bump, new deps) classified correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_guarded_classification_minor_bump() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let component = make_component("pytorch", "2.5.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Guarded);
    }

    #[test]
    fn test_guarded_classification_patch_bump_candidate_tier() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("triton".to_string(), "3.0.0".to_string());
        context.installed_components.insert("triton".to_string());

        // Patch bump but Candidate tier → guarded
        let component = make_component("triton", "3.0.1", ValidationTier::Candidate);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Guarded);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-004: Blocked updates (incompatible hardware, missing executor)
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_classification_missing_executor() {
        let mut context = CompatibilityContext::new();
        // No executors available
        context.rocm_version = "7.2.1".to_string();

        let component = make_component("pytorch", "2.4.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Blocked);
    }

    #[test]
    fn test_blocked_classification_blocked_tier() {
        let context = make_context();
        let component = make_component("bad-pkg", "1.0.0", ValidationTier::Blocked);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Blocked);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-005: Candidate updates (major bump, compatible)
    // -----------------------------------------------------------------------

    #[test]
    fn test_candidate_classification_major_bump() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let component = make_component("pytorch", "3.0.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Candidate);
    }

    #[test]
    fn test_candidate_classification_new_install() {
        let context = make_context();
        // Not installed → Candidate
        let component = make_component("triton", "3.1.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Candidate);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-006: Experimental updates hidden without --include-experimental
    // -----------------------------------------------------------------------

    #[test]
    fn test_experimental_classification() {
        let context = make_context();
        let component = make_component("deepspeed", "0.15.0", ValidationTier::Experimental);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Experimental);
    }

    #[test]
    fn test_experimental_hidden_without_flag() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "deepspeed",
            "0.15.0",
            ValidationTier::Experimental,
        )]);

        let options = PlannerOptions {
            include_experimental: false,
            ..Default::default()
        };

        let items = planner().build_plan(&manifest, &context, &options).unwrap();
        assert!(
            items.is_empty(),
            "Experimental items should be hidden without flag"
        );
    }

    #[test]
    fn test_experimental_visible_with_flag() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "deepspeed",
            "0.15.0",
            ValidationTier::Experimental,
        )]);

        let options = PlannerOptions {
            include_experimental: true,
            ..Default::default()
        };

        let items = planner().build_plan(&manifest, &context, &options).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].classification, UpdateClassification::Experimental);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-007: Safe updates preselected by default
    // -----------------------------------------------------------------------

    #[test]
    fn test_safe_updates_preselected() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.4.1",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        assert_eq!(items.len(), 1);
        assert!(items[0].selected, "Safe updates should be preselected");
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-008: Guarded updates suggested but not auto-applied
    // -----------------------------------------------------------------------

    #[test]
    fn test_guarded_updates_not_preselected() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.5.0",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        assert_eq!(items.len(), 1);
        assert!(
            !items[0].selected,
            "Guarded updates should NOT be preselected"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-009: Candidate updates visible but never preselected
    // -----------------------------------------------------------------------

    #[test]
    fn test_candidate_updates_visible_not_preselected() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "3.0.0",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        assert_eq!(items.len(), 1);
        assert!(items[0].visible, "Candidate updates should be visible");
        assert!(
            !items[0].selected,
            "Candidate updates should NOT be preselected"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-010: Experimental updates hidden unless --include-experimental
    // (covered above in test_experimental_hidden_without_flag)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // VAL-UPD-011: Blocked updates never offered, explicit targeting fails
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_not_in_default_plan() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "bad-pkg",
            "1.0.0",
            ValidationTier::Blocked,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        // Blocked items should be in the plan but not visible or selected
        assert_eq!(items.len(), 1);
        assert!(!items[0].visible);
        assert!(!items[0].selected);
    }

    #[test]
    fn test_blocked_explicit_target_fails() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "bad-pkg",
            "1.0.0",
            ValidationTier::Blocked,
        )]);

        let options = PlannerOptions {
            target_components: vec!["bad-pkg".to_string()],
            ..Default::default()
        };

        let result = planner().build_plan(&manifest, &context, &options);
        assert!(result.is_err());
        match result.unwrap_err() {
            PlannerError::ComponentBlocked {
                component_id,
                reason,
            } => {
                assert_eq!(component_id, "bad-pkg");
                assert!(!reason.is_empty());
            }
            other => panic!("Expected ComponentBlocked, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-013: Dependency rules enforced in plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_auto_selected() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("rocm".to_string(), "7.2.0".to_string());
        context.installed_components.insert("rocm".to_string());
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![
            make_component("rocm", "7.2.1", ValidationTier::Validated), // safe (patch)
            make_component("pytorch", "2.4.1", ValidationTier::Validated), // safe (patch)
        ]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        // Both should be selected (pytorch depends on rocm, both safe)
        let rocm_item = items
            .iter()
            .find(|i| i.plan_item.component_id == "rocm")
            .unwrap();
        let pytorch_item = items
            .iter()
            .find(|i| i.plan_item.component_id == "pytorch")
            .unwrap();

        assert!(rocm_item.selected, "ROCm should be selected (safe)");
        assert!(pytorch_item.selected, "PyTorch should be selected (safe)");
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-019: --scan-only produces plan without applying
    // -----------------------------------------------------------------------

    #[test]
    fn test_scan_only_produces_plan() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.4.1",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions {
            scan_only: true,
            ..Default::default()
        };

        let items = planner().build_plan(&manifest, &context, &options).unwrap();
        assert_eq!(items.len(), 1);
        // scan_only doesn't change the plan content, just prevents apply
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-020: --all-safe applies only validated safe updates
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_safe_filters_to_safe_only() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());
        context
            .installed_versions
            .insert("triton".to_string(), "3.0.0".to_string());
        context.installed_components.insert("triton".to_string());

        let manifest = make_manifest(vec![
            make_component("pytorch", "2.4.1", ValidationTier::Validated), // safe (patch)
            make_component("triton", "3.1.0", ValidationTier::Validated),  // guarded (minor)
        ]);

        let options = PlannerOptions {
            all_safe: true,
            ..Default::default()
        };

        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        assert_eq!(items.len(), 1, "Only safe items should remain");
        assert_eq!(items[0].classification, UpdateClassification::Safe);
        assert_eq!(items[0].plan_item.component_id, "pytorch");
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-021: Targeted component still runs full compatibility check
    // -----------------------------------------------------------------------

    #[test]
    fn test_targeted_blocked_component_fails() {
        let context = make_context();
        let manifest = make_manifest(vec![
            make_component("pytorch", "2.5.0", ValidationTier::Validated),
            make_component("bad-pkg", "1.0.0", ValidationTier::Blocked),
        ]);

        let options = PlannerOptions {
            target_components: vec!["bad-pkg".to_string()],
            ..Default::default()
        };

        let result = planner().build_plan(&manifest, &context, &options);
        assert!(result.is_err(), "Targeting a blocked component should fail");
    }

    #[test]
    fn test_targeted_unknown_component_fails() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.5.0",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions {
            target_components: vec!["nonexistent".to_string()],
            ..Default::default()
        };

        let result = planner().build_plan(&manifest, &context, &options);
        assert!(
            result.is_err(),
            "Targeting a nonexistent component should fail"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-023: Manifest fetch failure falls back to cached/baseline
    // (This is tested in core::manifest tests; planner accepts any Manifest)
    // -----------------------------------------------------------------------

    #[test]
    fn test_planner_works_with_baseline_manifest() {
        let context = make_context();
        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.4.0",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();
        assert!(!items.is_empty());
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-024: Runtime too old produces upgrade instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_runtime_too_old_error() {
        let error = PlannerError::RuntimeTooOld {
            current: "0.1.0".to_string(),
            required: "0.2.0".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("0.1.0"));
        assert!(msg.contains("0.2.0"));
        assert!(msg.contains("rusty upgrade"));
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-026: Hardware scan respects ROCm channel selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_rocm_channel_in_context() {
        let mut context = make_context();
        context.rocm_channel = "legacy".to_string();

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.4.0",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();
        assert!(!items.is_empty());
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-027: Signed remote manifest validated
    // (Manifest trust is tested in core::manifest; planner accepts any Manifest)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Classification serde roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_classification_serde_roundtrip() {
        let classifications = [
            UpdateClassification::Safe,
            UpdateClassification::Guarded,
            UpdateClassification::Blocked,
            UpdateClassification::Candidate,
            UpdateClassification::Experimental,
        ];
        for c in &classifications {
            let json = serde_json::to_string(c).unwrap();
            let back: UpdateClassification = serde_json::from_str(&json).unwrap();
            assert_eq!(*c, back);
        }
    }

    // -----------------------------------------------------------------------
    // PlanSummary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_summary_from_items() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());
        context
            .installed_versions
            .insert("triton".to_string(), "3.0.0".to_string());
        context.installed_components.insert("triton".to_string());

        let manifest = make_manifest(vec![
            make_component("pytorch", "2.4.1", ValidationTier::Validated), // safe
            make_component("triton", "3.1.0", ValidationTier::Validated),  // guarded
            make_component("new-pkg", "1.0.0", ValidationTier::Validated), // candidate (new)
            make_component("bad-pkg", "1.0.0", ValidationTier::Blocked),   // blocked
        ]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        let summary = PlanSummary::from_items(&items);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.safe, 1);
        assert_eq!(summary.guarded, 1);
        assert_eq!(summary.candidate, 1);
        assert_eq!(summary.blocked, 1);
        assert_eq!(summary.selected, 1); // only safe is selected
    }

    // -----------------------------------------------------------------------
    // PlannerItemOutput conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_planner_item_output_conversion() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());

        let manifest = make_manifest(vec![make_component(
            "pytorch",
            "2.4.1",
            ValidationTier::Validated,
        )]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        let output: Vec<PlannerItemOutput> = items.iter().map(PlannerItemOutput::from).collect();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].component_id, "pytorch");
        assert_eq!(output[0].classification, "safe");
        assert!(output[0].selected);
    }

    // -----------------------------------------------------------------------
    // Full plan build with multiple classifications
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_plan_build_mixed_classifications() {
        let mut context = make_context();
        context
            .installed_versions
            .insert("rocm".to_string(), "7.2.0".to_string());
        context.installed_components.insert("rocm".to_string());
        context
            .installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        context.installed_components.insert("pytorch".to_string());
        context
            .installed_versions
            .insert("triton".to_string(), "3.0.0".to_string());
        context.installed_components.insert("triton".to_string());

        let manifest = make_manifest(vec![
            make_component("rocm", "7.2.1", ValidationTier::Validated), // safe (patch)
            make_component("pytorch", "2.5.0", ValidationTier::Validated), // guarded (minor)
            make_component("triton", "4.0.0", ValidationTier::Validated), // candidate (major)
            make_component("wandb", "0.17.0", ValidationTier::Validated), // candidate (new install)
            make_component("deepspeed", "0.15.0", ValidationTier::Experimental), // experimental
        ]);

        let options = PlannerOptions {
            include_experimental: true,
            ..Default::default()
        };

        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        assert_eq!(items.len(), 5);

        let rocm = items
            .iter()
            .find(|i| i.plan_item.component_id == "rocm")
            .unwrap();
        assert_eq!(rocm.classification, UpdateClassification::Safe);
        assert!(rocm.selected);

        let pytorch = items
            .iter()
            .find(|i| i.plan_item.component_id == "pytorch")
            .unwrap();
        assert_eq!(pytorch.classification, UpdateClassification::Guarded);
        assert!(!pytorch.selected);

        let triton = items
            .iter()
            .find(|i| i.plan_item.component_id == "triton")
            .unwrap();
        assert_eq!(triton.classification, UpdateClassification::Candidate);
        assert!(!triton.selected);

        let wandb = items
            .iter()
            .find(|i| i.plan_item.component_id == "wandb")
            .unwrap();
        assert_eq!(wandb.classification, UpdateClassification::Candidate);
        assert!(!wandb.selected);

        let deepspeed = items
            .iter()
            .find(|i| i.plan_item.component_id == "deepspeed")
            .unwrap();
        assert_eq!(deepspeed.classification, UpdateClassification::Experimental);
        assert!(!deepspeed.selected);
    }

    // -----------------------------------------------------------------------
    // CompatibilityContext tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compatibility_context_default() {
        let ctx = CompatibilityContext::default();
        assert!(ctx.rocm_version.is_empty());
        assert!(ctx.rocm_channel.is_empty());
        assert!(ctx.available_executors.is_empty());
        assert!(ctx.installed_components.is_empty());
        assert!(ctx.installed_versions.is_empty());
        assert!(ctx.gpu_architecture.is_empty());
        assert!(ctx.runtime_version.is_empty());
    }

    #[test]
    fn test_compatibility_context_is_installed() {
        let mut ctx = CompatibilityContext::new();
        assert!(!ctx.is_installed("pytorch"));
        ctx.installed_components.insert("pytorch".to_string());
        assert!(ctx.is_installed("pytorch"));
    }

    #[test]
    fn test_compatibility_context_installed_version() {
        let mut ctx = CompatibilityContext::new();
        assert!(ctx.installed_version("pytorch").is_none());
        ctx.installed_versions
            .insert("pytorch".to_string(), "2.4.0".to_string());
        assert_eq!(ctx.installed_version("pytorch"), Some("2.4.0"));
    }

    // -----------------------------------------------------------------------
    // PlannerError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_planner_error_display() {
        let err = PlannerError::ComponentBlocked {
            component_id: "bad-pkg".to_string(),
            reason: "incompatible hardware".to_string(),
        };
        assert!(err.to_string().contains("bad-pkg"));
        assert!(err.to_string().contains("incompatible hardware"));

        let err = PlannerError::ComponentUnknown {
            component_id: "nonexistent".to_string(),
        };
        assert!(err.to_string().contains("nonexistent"));

        let err = PlannerError::RuntimeTooOld {
            current: "0.1.0".to_string(),
            required: "0.2.0".to_string(),
        };
        assert!(err.to_string().contains("0.1.0"));
        assert!(err.to_string().contains("rusty upgrade"));
    }

    #[test]
    fn test_planner_error_is_std_error() {
        let err = PlannerError::ComponentUnknown {
            component_id: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err;
    }

    // -----------------------------------------------------------------------
    // ScanOutput / PlanOutput serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_scan_output_serde_roundtrip() {
        let output = ScanOutput {
            installed: vec![InstalledComponent {
                id: "pytorch".to_string(),
                version: "2.4.0".to_string(),
                status: "installed".to_string(),
            }],
            manifest_source: "baseline".to_string(),
            rocm_channel: "latest".to_string(),
            rocm_version: "7.2.1".to_string(),
            gpu_architecture: "gfx1100".to_string(),
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: ScanOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output, back);
    }

    #[test]
    fn test_plan_output_serde_roundtrip() {
        let output = PlanOutput {
            plan: vec![PlannerItemOutput {
                component_id: "pytorch".to_string(),
                current_version: "2.4.0".to_string(),
                proposed_version: "2.4.1".to_string(),
                classification: "safe".to_string(),
                risk_tier: "low".to_string(),
                selected: true,
                visible: true,
                rationale: "patch update".to_string(),
                dependencies: vec!["rocm".to_string()],
            }],
            summary: PlanSummary {
                total: 1,
                safe: 1,
                guarded: 0,
                candidate: 0,
                experimental: 0,
                blocked: 0,
                selected: 1,
            },
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: PlanOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output, back);
    }

    // -----------------------------------------------------------------------
    // UpdateClassification properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_classification_visibility() {
        assert!(UpdateClassification::Safe.is_visible());
        assert!(UpdateClassification::Guarded.is_visible());
        assert!(!UpdateClassification::Blocked.is_visible());
        assert!(UpdateClassification::Candidate.is_visible());
        assert!(!UpdateClassification::Experimental.is_visible());
    }

    #[test]
    fn test_classification_preselection() {
        assert!(UpdateClassification::Safe.is_preselected());
        assert!(!UpdateClassification::Guarded.is_preselected());
        assert!(!UpdateClassification::Blocked.is_preselected());
        assert!(!UpdateClassification::Candidate.is_preselected());
        assert!(!UpdateClassification::Experimental.is_preselected());
    }

    #[test]
    fn test_classification_labels() {
        assert_eq!(UpdateClassification::Safe.label(), "safe");
        assert_eq!(UpdateClassification::Guarded.label(), "guarded");
        assert_eq!(UpdateClassification::Blocked.label(), "blocked");
        assert_eq!(UpdateClassification::Candidate.label(), "candidate");
        assert_eq!(UpdateClassification::Experimental.label(), "experimental");
    }

    // -----------------------------------------------------------------------
    // Dependency derivation
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_dependencies_pytorch() {
        let deps = planner().derive_dependencies("pytorch");
        assert!(deps.contains(&"rocm".to_string()));
    }

    #[test]
    fn test_derive_dependencies_triton() {
        let deps = planner().derive_dependencies("triton");
        assert!(deps.contains(&"pytorch".to_string()));
    }

    #[test]
    fn test_derive_dependencies_rocm() {
        let deps = planner().derive_dependencies("rocm");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_derive_dependencies_unknown() {
        let deps = planner().derive_dependencies("unknown-component");
        assert!(deps.is_empty());
    }

    // -----------------------------------------------------------------------
    // Version bump level detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_version_bump_patch() {
        assert_eq!(
            planner().version_bump_level("1.0.0", "1.0.1"),
            BumpLevel::Patch
        );
    }

    #[test]
    fn test_version_bump_minor() {
        assert_eq!(
            planner().version_bump_level("1.0.0", "1.1.0"),
            BumpLevel::Minor
        );
    }

    #[test]
    fn test_version_bump_major() {
        assert_eq!(
            planner().version_bump_level("1.0.0", "2.0.0"),
            BumpLevel::Major
        );
    }

    #[test]
    fn test_version_bump_unknown() {
        assert_eq!(
            planner().version_bump_level("abc", "def"),
            BumpLevel::Unknown
        );
    }

    // -----------------------------------------------------------------------
    // version_gte helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_version_gte_equal() {
        assert!(version_gte("7.2.1", "7.2.1"));
    }

    #[test]
    fn test_version_gte_greater() {
        assert!(version_gte("7.3.0", "7.2.1"));
        assert!(version_gte("8.0.0", "7.2.1"));
        assert!(version_gte("7.2.2", "7.2.1"));
    }

    #[test]
    fn test_version_gte_less() {
        assert!(!version_gte("7.2.0", "7.2.1"));
        assert!(!version_gte("6.0.0", "7.0.0"));
    }

    #[test]
    fn test_version_gte_different_lengths() {
        assert!(version_gte("7.2", "7.2.0"));
        assert!(version_gte("7.2.1", "7.2"));
        assert!(!version_gte("7.1", "7.2.0"));
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-004: Blocked by ROCm version requirement
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_by_rocm_version_requirement() {
        let context = make_context(); // rocm_version = "7.2.1"

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: "99.0.0".to_string(), // impossibly high
            compatible_channels: vec![],
        };

        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Blocked);
    }

    #[test]
    fn test_not_blocked_by_satisfied_rocm_requirement() {
        let context = make_context(); // rocm_version = "7.2.1"

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: "7.0.0".to_string(),
            compatible_channels: vec![],
        };

        let classification = planner().classify_update(&component, &context);
        assert_ne!(classification, UpdateClassification::Blocked);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-026: Hardware scan respects ROCm channel selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_by_incompatible_channel() {
        let mut context = make_context();
        context.rocm_channel = "legacy".to_string();

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: String::new(),
            compatible_channels: vec!["latest".to_string(), "stable".to_string()],
        };

        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Blocked);
    }

    #[test]
    fn test_allowed_by_compatible_channel() {
        let mut context = make_context();
        context.rocm_channel = "latest".to_string();

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: String::new(),
            compatible_channels: vec!["latest".to_string(), "stable".to_string()],
        };

        let classification = planner().classify_update(&component, &context);
        assert_ne!(classification, UpdateClassification::Blocked);
    }

    #[test]
    fn test_channel_filter_legacy_excludes_latest_only() {
        let mut context = make_context();
        context.rocm_channel = "legacy".to_string();

        let manifest = make_manifest(vec![ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: String::new(),
            compatible_channels: vec!["latest".to_string()],
        }]);

        let options = PlannerOptions::default();
        let items = planner().build_plan(&manifest, &context, &options).unwrap();

        // The item should be blocked (not visible)
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].classification, UpdateClassification::Blocked);
        assert!(!items[0].visible);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-024: Runtime too old produces upgrade instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_runtime_too_old_blocks_plan() {
        let context = make_context(); // runtime_version is empty by default

        let mut manifest = make_manifest(vec![make_component(
            "pytorch",
            "3.0.0",
            ValidationTier::Validated,
        )]);
        manifest.min_runtime_version = "99.0.0".to_string();

        let mut ctx = context;
        ctx.runtime_version = "0.1.0".to_string();

        let options = PlannerOptions::default();
        let result = planner().build_plan(&manifest, &ctx, &options);
        assert!(result.is_err());
        match result.unwrap_err() {
            PlannerError::RuntimeTooOld { current, required } => {
                assert_eq!(current, "0.1.0");
                assert_eq!(required, "99.0.0");
            }
            other => panic!("Expected RuntimeTooOld, got: {other:?}"),
        }
    }

    #[test]
    fn test_runtime_version_satisfied() {
        let mut context = make_context();
        context.runtime_version = "1.0.0".to_string();

        let mut manifest = make_manifest(vec![make_component(
            "pytorch",
            "3.0.0",
            ValidationTier::Validated,
        )]);
        manifest.min_runtime_version = "0.5.0".to_string();

        let options = PlannerOptions::default();
        let result = planner().build_plan(&manifest, &context, &options);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-004: Blocked when ROCm not installed but required
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocked_when_rocm_not_installed_but_required() {
        let mut context = CompatibilityContext::new();
        context.rocm_version = String::new(); // no ROCm
        context.available_executors =
            HashSet::from([ExecutorKind::LegacyScript, ExecutorKind::Rust]);

        let component = make_component("pytorch", "3.0.0", ValidationTier::Validated);
        let classification = planner().classify_update(&component, &context);
        assert_eq!(classification, UpdateClassification::Blocked);
    }

    // -----------------------------------------------------------------------
    // VAL-UPD-027: Block reason includes ROCm version info
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_reason_includes_rocm_version() {
        let context = make_context(); // rocm_version = "7.2.1"

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: "99.0.0".to_string(),
            compatible_channels: vec![],
        };

        let reason = planner().block_reason(&component, &context);
        assert!(reason.contains("99.0.0"));
        assert!(reason.contains("7.2.1"));
    }

    #[test]
    fn test_block_reason_channel_incompatibility() {
        let mut context = make_context();
        context.rocm_channel = "legacy".to_string();

        let component = ManifestComponent {
            id: "pytorch".to_string(),
            version: "3.0.0".to_string(),
            script: "install_pytorch.sh".to_string(),
            category: crate::core::types::Category::Core,
            validation_tier: ValidationTier::Validated,
            min_rocm_version: String::new(),
            compatible_channels: vec!["latest".to_string()],
        };

        let reason = planner().block_reason(&component, &context);
        assert!(reason.contains("legacy"));
    }
}
