//! Core foundational types for Rusty Stack.
//!
//! This module defines the canonical types used across all modules:
//! component identity, categories, stages, hardware info, platform
//! capabilities, validation tiers, and risk classification.

use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

// ---------------------------------------------------------------------------
// Category
// ---------------------------------------------------------------------------

/// Component category classification.
///
/// Each component belongs to exactly one of seven categories.
/// Serializes to/from title-case string representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Foundation,
    Core,
    Extension,
    UiUx,
    Environment,
    Verification,
    Performance,
}

impl Category {
    /// All category variants in canonical order.
    pub fn all() -> &'static [Category] {
        &[
            Category::Foundation,
            Category::Core,
            Category::Extension,
            Category::UiUx,
            Category::Environment,
            Category::Verification,
            Category::Performance,
        ]
    }

    /// Returns the string label used for serialization.
    pub fn label(self) -> &'static str {
        match self {
            Category::Foundation => "Foundation",
            Category::Core => "Core",
            Category::Extension => "Extension",
            Category::UiUx => "UiUx",
            Category::Environment => "Environment",
            Category::Verification => "Verification",
            Category::Performance => "Performance",
        }
    }

    /// Parse a category from its string label.
    pub fn from_label(s: &str) -> Option<Category> {
        match s {
            "Foundation" => Some(Category::Foundation),
            "Core" => Some(Category::Core),
            "Extension" => Some(Category::Extension),
            "UiUx" => Some(Category::UiUx),
            "Environment" => Some(Category::Environment),
            "Verification" => Some(Category::Verification),
            "Performance" => Some(Category::Performance),
            _ => None,
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl Serialize for Category {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.label())
    }
}

impl<'de> Deserialize<'de> for Category {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Category::from_label(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown Category variant: {s}")))
    }
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// TUI navigation / installation lifecycle stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Welcome,
    HardwareDetect,
    Preflight,
    ComponentSelect,
    Configuration,
    Confirm,
    Installing,
    Complete,
    Benchmarks,
    Recovery,
}

impl Stage {
    /// All stage variants in lifecycle order.
    pub fn all() -> &'static [Stage] {
        &[
            Stage::Welcome,
            Stage::HardwareDetect,
            Stage::Preflight,
            Stage::ComponentSelect,
            Stage::Configuration,
            Stage::Confirm,
            Stage::Installing,
            Stage::Complete,
            Stage::Benchmarks,
            Stage::Recovery,
        ]
    }

    /// Returns the string label used for serialization.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Welcome => "Welcome",
            Stage::HardwareDetect => "HardwareDetect",
            Stage::Preflight => "Preflight",
            Stage::ComponentSelect => "ComponentSelect",
            Stage::Configuration => "Configuration",
            Stage::Confirm => "Confirm",
            Stage::Installing => "Installing",
            Stage::Complete => "Complete",
            Stage::Benchmarks => "Benchmarks",
            Stage::Recovery => "Recovery",
        }
    }

    /// Parse a stage from its string label.
    pub fn from_label(s: &str) -> Option<Stage> {
        match s {
            "Welcome" => Some(Stage::Welcome),
            "HardwareDetect" => Some(Stage::HardwareDetect),
            "Preflight" => Some(Stage::Preflight),
            "ComponentSelect" => Some(Stage::ComponentSelect),
            "Configuration" => Some(Stage::Configuration),
            "Confirm" => Some(Stage::Confirm),
            "Installing" => Some(Stage::Installing),
            "Complete" => Some(Stage::Complete),
            "Benchmarks" => Some(Stage::Benchmarks),
            "Recovery" => Some(Stage::Recovery),
            _ => None,
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl Serialize for Stage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.label())
    }
}

impl<'de> Deserialize<'de> for Stage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Stage::from_label(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown Stage variant: {s}")))
    }
}

// ---------------------------------------------------------------------------
// GPUInfo
// ---------------------------------------------------------------------------

/// GPU hardware information detected at runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GPUInfo {
    pub model: String,
    pub driver: String,
    pub architecture: String,
    pub rocm_version: String,
    pub gpu_count: usize,
    pub memory_gb: f32,
    #[serde(default)]
    pub temperature_c: Option<f32>,
    #[serde(default)]
    pub power_watts: Option<f32>,
}

// ---------------------------------------------------------------------------
// SystemInfo
// ---------------------------------------------------------------------------

/// System-level information detected at runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemInfo {
    pub os: String,
    pub distribution: String,
    pub kernel: String,
    pub cpu_model: String,
    pub memory_gb: f32,
    pub storage_gb: f32,
    pub storage_available_gb: f32,
}

// ---------------------------------------------------------------------------
// PreflightStatus / PreflightType
// ---------------------------------------------------------------------------

/// Result status of a single preflight check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreflightStatus {
    Passed,
    Warning,
    Failed,
}

impl PreflightStatus {
    /// Human-readable label for this status.
    pub fn label(self) -> &'static str {
        match self {
            PreflightStatus::Passed => "passed",
            PreflightStatus::Warning => "warning",
            PreflightStatus::Failed => "failed",
        }
    }
}

/// Severity classification of a preflight check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreflightType {
    Critical,
    Warning,
    Info,
}

impl PreflightType {
    /// Human-readable label for this type.
    pub fn label(self) -> &'static str {
        match self {
            PreflightType::Critical => "critical",
            PreflightType::Warning => "warning",
            PreflightType::Info => "info",
        }
    }
}

// ---------------------------------------------------------------------------
// BackendMode
// ---------------------------------------------------------------------------

/// Execution backend classification based on OS and WSL2 state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendMode {
    LinuxNative,
    WslBackedLinux,
    WindowsNative,
}

impl BackendMode {
    pub fn label(self) -> &'static str {
        match self {
            BackendMode::LinuxNative => "LinuxNative",
            BackendMode::WslBackedLinux => "WslBackedLinux",
            BackendMode::WindowsNative => "WindowsNative",
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutorKind
// ---------------------------------------------------------------------------

/// How a component's installation is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutorKind {
    Rust,
    LegacyScript,
    ExternalPackageManager,
    Unsupported,
}

impl ExecutorKind {
    pub fn label(self) -> &'static str {
        match self {
            ExecutorKind::Rust => "Rust",
            ExecutorKind::LegacyScript => "LegacyScript",
            ExecutorKind::ExternalPackageManager => "ExternalPackageManager",
            ExecutorKind::Unsupported => "Unsupported",
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationTier
// ---------------------------------------------------------------------------

/// Validation maturity tier for a component.
///
/// Governs visibility, preselection, and update eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationTier {
    Validated,
    Candidate,
    Experimental,
    Blocked,
}

impl ValidationTier {
    /// All tiers in order of decreasing trust.
    pub fn all() -> &'static [ValidationTier] {
        &[
            ValidationTier::Validated,
            ValidationTier::Candidate,
            ValidationTier::Experimental,
            ValidationTier::Blocked,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            ValidationTier::Validated => "Validated",
            ValidationTier::Candidate => "Candidate",
            ValidationTier::Experimental => "Experimental",
            ValidationTier::Blocked => "Blocked",
        }
    }

    /// Whether components with this tier are visible by default.
    pub fn is_visible(self) -> bool {
        match self {
            ValidationTier::Validated | ValidationTier::Candidate => true,
            ValidationTier::Experimental | ValidationTier::Blocked => false,
        }
    }

    /// Whether components with this tier are pre-selected by default.
    pub fn is_preselected(self) -> bool {
        matches!(self, ValidationTier::Validated)
    }
}

// ---------------------------------------------------------------------------
// RiskTier
// ---------------------------------------------------------------------------

/// Risk classification for an update plan item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskTier {
    Low,
    Medium,
    High,
}

impl RiskTier {
    pub fn label(self) -> &'static str {
        match self {
            RiskTier::Low => "low",
            RiskTier::Medium => "medium",
            RiskTier::High => "high",
        }
    }

    /// Numeric ordering for sorting (higher = riskier).
    pub fn order(self) -> u8 {
        match self {
            RiskTier::Low => 0,
            RiskTier::Medium => 1,
            RiskTier::High => 2,
        }
    }
}

impl PartialOrd for RiskTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RiskTier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order().cmp(&other.order())
    }
}

// ---------------------------------------------------------------------------
// InstallStatus
// ---------------------------------------------------------------------------

/// Tracks the progress of an ongoing installation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InstallStatus {
    pub progress: f32,
    pub message: String,
    pub completed: bool,
}

// ---------------------------------------------------------------------------
// InstallerConfig
// ---------------------------------------------------------------------------

/// Configuration for the installer subsystem.
///
/// The `config_path` field is skipped during serialization because it
/// is a local filesystem path that should not be persisted or transmitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstallerConfig {
    pub scripts_dir: String,
    pub python_bin: String,
    pub rocm_path: String,
    pub verbose: bool,
    pub dry_run: bool,
    #[serde(skip)]
    pub config_path: std::path::PathBuf,
}

impl Default for InstallerConfig {
    fn default() -> Self {
        Self {
            scripts_dir: String::new(),
            python_bin: String::new(),
            rocm_path: String::new(),
            verbose: false,
            dry_run: false,
            config_path: std::path::PathBuf::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentIdentity
// ---------------------------------------------------------------------------

/// Validation errors for [`ComponentIdentity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentIdentityError {
    InvalidId { id: String, reason: String },
    InvalidVersion { version: String, reason: String },
    InvalidScriptPath { path: String, reason: String },
}

impl fmt::Display for ComponentIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentIdentityError::InvalidId { id, reason } => {
                write!(f, "invalid component id '{id}': {reason}")
            }
            ComponentIdentityError::InvalidVersion { version, reason } => {
                write!(f, "invalid version '{version}': {reason}")
            }
            ComponentIdentityError::InvalidScriptPath { path, reason } => {
                write!(f, "invalid script path '{path}': {reason}")
            }
        }
    }
}

impl std::error::Error for ComponentIdentityError {}

/// Canonical identity for an installable component.
///
/// All fields are validated on construction:
/// - `id` must match `^[a-z0-9][a-z0-9-]{1,63}$`
/// - `version` must be valid semver (`MAJOR.MINOR.PATCH[-pre]`)
/// - `script` must be a relative path with no `..` traversal
/// - `category` must be a valid [`Category`] variant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentIdentity {
    pub id: String,
    pub version: String,
    pub script: String,
    pub category: Category,
}

impl ComponentIdentity {
    /// Validate and construct a new [`ComponentIdentity`].
    ///
    /// Returns an error if any field fails validation.
    pub fn new(
        id: impl Into<String>,
        version: impl Into<String>,
        script: impl Into<String>,
        category: Category,
    ) -> Result<Self, ComponentIdentityError> {
        let id = id.into();
        let version = version.into();
        let script = script.into();

        validate_id(&id)?;
        validate_version(&version)?;
        validate_script_path(&script)?;

        Ok(Self {
            id,
            version,
            script,
            category,
        })
    }
}

/// Validates that `id` matches `^[a-z0-9][a-z0-9-]{1,63}$`.
fn validate_id(id: &str) -> Result<(), ComponentIdentityError> {
    let re = Regex::new(r"^[a-z0-9][a-z0-9-]{1,63}$").unwrap();
    if re.is_match(id) {
        Ok(())
    } else {
        Err(ComponentIdentityError::InvalidId {
            id: id.to_string(),
            reason: "must match ^[a-z0-9][a-z0-9-]{1,63}$".to_string(),
        })
    }
}

/// Validates that `version` is valid semver (MAJOR.MINOR.PATCH with optional pre-release).
fn validate_version(version: &str) -> Result<(), ComponentIdentityError> {
    let re = Regex::new(r"^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$").unwrap();
    if re.is_match(version) {
        Ok(())
    } else {
        Err(ComponentIdentityError::InvalidVersion {
            version: version.to_string(),
            reason: "must be valid semver (MAJOR.MINOR.PATCH[-pre])".to_string(),
        })
    }
}

/// Validates that `path` is a relative path with no `..` traversal.
fn validate_script_path(path: &str) -> Result<(), ComponentIdentityError> {
    if path.is_empty() {
        return Err(ComponentIdentityError::InvalidScriptPath {
            path: path.to_string(),
            reason: "path cannot be empty".to_string(),
        });
    }
    if path.contains("..") {
        return Err(ComponentIdentityError::InvalidScriptPath {
            path: path.to_string(),
            reason: "path must not contain '..' traversal".to_string(),
        });
    }
    if std::path::Path::new(path).is_absolute() {
        return Err(ComponentIdentityError::InvalidScriptPath {
            path: path.to_string(),
            reason: "path must be relative, not absolute".to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Component (legacy compatibility)
// ---------------------------------------------------------------------------

/// Full component representation for the TUI installer.
///
/// This type extends [`ComponentIdentity`] with display and state
/// information used by the existing TUI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Component {
    pub id: String,
    pub name: String,
    pub description: String,
    pub script: String,
    pub category: Category,
    pub required: bool,
    pub selected: bool,
    pub installed: bool,
    pub progress: f32,
    pub estimate: String,
    pub needs_sudo: bool,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Category tests (VAL-CORE-001, VAL-CORE-003) ----

    #[test]
    fn test_category_has_seven_variants() {
        assert_eq!(Category::all().len(), 7);
    }

    #[test]
    fn test_category_serde_roundtrip_all_variants() {
        for cat in Category::all() {
            let json = serde_json::to_string(cat).unwrap();
            let back: Category = serde_json::from_str(&json).unwrap();
            assert_eq!(*cat, back, "Category roundtrip failed for {:?}", cat);
        }
    }

    #[test]
    fn test_category_serializes_to_title_case() {
        assert_eq!(
            serde_json::to_string(&Category::Foundation).unwrap(),
            "\"Foundation\""
        );
        assert_eq!(serde_json::to_string(&Category::UiUx).unwrap(), "\"UiUx\"");
    }

    #[test]
    fn test_category_rejects_unknown_string() {
        let result = serde_json::from_str::<Category>("\"Unknown\"");
        assert!(result.is_err(), "Should reject unknown Category string");
    }

    #[test]
    fn test_category_from_label_roundtrip() {
        for cat in Category::all() {
            let label = cat.label();
            assert_eq!(Category::from_label(label), Some(*cat));
        }
        assert_eq!(Category::from_label("NonExistent"), None);
    }

    // ---- Stage tests (VAL-CORE-004) ----

    #[test]
    fn test_stage_has_ten_variants() {
        assert_eq!(Stage::all().len(), 10);
    }

    #[test]
    fn test_stage_serde_roundtrip_all_variants() {
        for stage in Stage::all() {
            let json = serde_json::to_string(stage).unwrap();
            let back: Stage = serde_json::from_str(&json).unwrap();
            assert_eq!(*stage, back, "Stage roundtrip failed for {:?}", stage);
        }
    }

    #[test]
    fn test_stage_rejects_unknown_string() {
        let result = serde_json::from_str::<Stage>("\"NonExistent\"");
        assert!(result.is_err(), "Should reject unknown Stage string");
    }

    #[test]
    fn test_stage_from_label_roundtrip() {
        for stage in Stage::all() {
            let label = stage.label();
            assert_eq!(Stage::from_label(label), Some(*stage));
        }
        assert_eq!(Stage::from_label("NonExistent"), None);
    }

    // ---- GPUInfo tests (VAL-CORE-002) ----

    #[test]
    fn test_gpu_info_serde_roundtrip_fully_populated() {
        let info = GPUInfo {
            model: "AMD Radeon RX 7900 XTX".to_string(),
            driver: "6.7.0".to_string(),
            architecture: "gfx1100".to_string(),
            rocm_version: "7.2.1".to_string(),
            gpu_count: 2,
            memory_gb: 24.0,
            temperature_c: Some(65.0),
            power_watts: Some(280.0),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: GPUInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn test_gpu_info_serde_roundtrip_default() {
        let info = GPUInfo::default();
        let json = serde_json::to_string(&info).unwrap();
        let back: GPUInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn test_gpu_info_optional_fields_none() {
        let info = GPUInfo {
            model: "AMD Radeon RX 7900 XTX".to_string(),
            driver: "6.7.0".to_string(),
            architecture: "gfx1100".to_string(),
            rocm_version: "7.2.1".to_string(),
            gpu_count: 1,
            memory_gb: 24.0,
            temperature_c: None,
            power_watts: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: GPUInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
        assert!(back.temperature_c.is_none());
        assert!(back.power_watts.is_none());
    }

    #[test]
    fn test_gpu_info_optional_fields_some() {
        let info = GPUInfo {
            temperature_c: Some(42.0),
            power_watts: Some(150.0),
            ..Default::default()
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: GPUInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.temperature_c, Some(42.0));
        assert_eq!(back.power_watts, Some(150.0));
    }

    // ---- SystemInfo tests (VAL-CORE-002) ----

    #[test]
    fn test_system_info_serde_roundtrip_fully_populated() {
        let info = SystemInfo {
            os: "Linux".to_string(),
            distribution: "Arch Linux".to_string(),
            kernel: "6.7.0-arch1".to_string(),
            cpu_model: "AMD Ryzen 9 7950X".to_string(),
            memory_gb: 54.0,
            storage_gb: 500.0,
            storage_available_gb: 153.0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: SystemInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn test_system_info_serde_roundtrip_default() {
        let info = SystemInfo::default();
        let json = serde_json::to_string(&info).unwrap();
        let back: SystemInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    // ---- PreflightStatus tests (VAL-CORE-021) ----

    #[test]
    fn test_preflight_status_serde_roundtrip() {
        let statuses = [
            PreflightStatus::Passed,
            PreflightStatus::Warning,
            PreflightStatus::Failed,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: PreflightStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
    }

    #[test]
    fn test_preflight_status_label() {
        assert_eq!(PreflightStatus::Passed.label(), "passed");
        assert_eq!(PreflightStatus::Warning.label(), "warning");
        assert_eq!(PreflightStatus::Failed.label(), "failed");
    }

    // ---- PreflightType tests (VAL-CORE-021) ----

    #[test]
    fn test_preflight_type_serde_roundtrip() {
        let types = [
            PreflightType::Critical,
            PreflightType::Warning,
            PreflightType::Info,
        ];
        for pt in &types {
            let json = serde_json::to_string(pt).unwrap();
            let back: PreflightType = serde_json::from_str(&json).unwrap();
            assert_eq!(*pt, back);
        }
    }

    #[test]
    fn test_preflight_type_label() {
        assert_eq!(PreflightType::Critical.label(), "critical");
        assert_eq!(PreflightType::Warning.label(), "warning");
        assert_eq!(PreflightType::Info.label(), "info");
    }

    // ---- BackendMode tests (VAL-CORE-018) ----

    #[test]
    fn test_backend_mode_serde_roundtrip() {
        let modes = [
            BackendMode::LinuxNative,
            BackendMode::WslBackedLinux,
            BackendMode::WindowsNative,
        ];
        for mode in &modes {
            let json = serde_json::to_string(mode).unwrap();
            let back: BackendMode = serde_json::from_str(&json).unwrap();
            assert_eq!(*mode, back);
        }
    }

    #[test]
    fn test_backend_mode_classifies_correctly() {
        assert_eq!(BackendMode::LinuxNative.label(), "LinuxNative");
        assert_eq!(BackendMode::WslBackedLinux.label(), "WslBackedLinux");
        assert_eq!(BackendMode::WindowsNative.label(), "WindowsNative");
    }

    // ---- ExecutorKind tests (VAL-CORE-019) ----

    #[test]
    fn test_executor_kind_serde_roundtrip() {
        let kinds = [
            ExecutorKind::Rust,
            ExecutorKind::LegacyScript,
            ExecutorKind::ExternalPackageManager,
            ExecutorKind::Unsupported,
        ];
        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let back: ExecutorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    fn test_executor_kind_classifies_correctly() {
        assert_eq!(ExecutorKind::Rust.label(), "Rust");
        assert_eq!(ExecutorKind::LegacyScript.label(), "LegacyScript");
        assert_eq!(
            ExecutorKind::ExternalPackageManager.label(),
            "ExternalPackageManager"
        );
        assert_eq!(ExecutorKind::Unsupported.label(), "Unsupported");
    }

    // ---- ValidationTier tests ----

    #[test]
    fn test_validation_tier_serde_roundtrip() {
        for tier in ValidationTier::all() {
            let json = serde_json::to_string(tier).unwrap();
            let back: ValidationTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, back);
        }
    }

    #[test]
    fn test_validation_tier_visibility() {
        assert!(ValidationTier::Validated.is_visible());
        assert!(ValidationTier::Candidate.is_visible());
        assert!(!ValidationTier::Experimental.is_visible());
        assert!(!ValidationTier::Blocked.is_visible());
    }

    #[test]
    fn test_validation_tier_preselection() {
        assert!(ValidationTier::Validated.is_preselected());
        assert!(!ValidationTier::Candidate.is_preselected());
        assert!(!ValidationTier::Experimental.is_preselected());
        assert!(!ValidationTier::Blocked.is_preselected());
    }

    // ---- RiskTier tests ----

    #[test]
    fn test_risk_tier_serde_roundtrip() {
        let tiers = [RiskTier::Low, RiskTier::Medium, RiskTier::High];
        for tier in &tiers {
            let json = serde_json::to_string(tier).unwrap();
            let back: RiskTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, back);
        }
    }

    #[test]
    fn test_risk_tier_ordering() {
        assert!(RiskTier::Low < RiskTier::Medium);
        assert!(RiskTier::Medium < RiskTier::High);
        assert!(RiskTier::Low < RiskTier::High);
    }

    // ---- InstallStatus tests ----

    #[test]
    fn test_install_status_serde_roundtrip() {
        let status = InstallStatus {
            progress: 0.75,
            message: "Installing PyTorch...".to_string(),
            completed: false,
        };
        let json = serde_json::to_string(&status).unwrap();
        let back: InstallStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, back);
    }

    #[test]
    fn test_install_status_default() {
        let status = InstallStatus::default();
        assert_eq!(status.progress, 0.0);
        assert!(status.message.is_empty());
        assert!(!status.completed);
    }

    // ---- InstallerConfig tests (VAL-CORE-022) ----

    #[test]
    fn test_installer_config_serde_skip_field() {
        let config = InstallerConfig {
            scripts_dir: "/opt/scripts".to_string(),
            python_bin: "/usr/bin/python3".to_string(),
            rocm_path: "/opt/rocm".to_string(),
            verbose: true,
            dry_run: false,
            config_path: std::path::PathBuf::from("/custom/path/config.json"),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: InstallerConfig = serde_json::from_str(&json).unwrap();

        // config_path should be default (empty) after roundtrip
        assert_eq!(back.config_path, std::path::PathBuf::new());
        // all other fields should match
        assert_eq!(back.scripts_dir, "/opt/scripts");
        assert_eq!(back.python_bin, "/usr/bin/python3");
        assert_eq!(back.rocm_path, "/opt/rocm");
        assert!(back.verbose);
        assert!(!back.dry_run);
    }

    #[test]
    fn test_installer_config_default() {
        let config = InstallerConfig::default();
        assert!(config.scripts_dir.is_empty());
        assert!(config.python_bin.is_empty());
        assert!(config.rocm_path.is_empty());
        assert!(!config.verbose);
        assert!(!config.dry_run);
        assert_eq!(config.config_path, std::path::PathBuf::new());
    }

    // ---- ComponentIdentity tests (VAL-CORE-020) ----

    #[test]
    fn test_component_identity_valid_construction() {
        let ci = ComponentIdentity::new(
            "pytorch",
            "2.1.0",
            "scripts/install_pytorch.sh",
            Category::Foundation,
        )
        .unwrap();
        assert_eq!(ci.id, "pytorch");
        assert_eq!(ci.version, "2.1.0");
        assert_eq!(ci.script, "scripts/install_pytorch.sh");
        assert_eq!(ci.category, Category::Foundation);
    }

    #[test]
    fn test_component_identity_id_valid() {
        // Valid: lowercase, digits, hyphens, 2-64 chars (regex: first char + 1-63 more)
        assert!(ComponentIdentity::new("ab", "1.0.0", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("a-b", "1.0.0", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("0a", "1.0.0", "s.sh", Category::Core).is_ok());
        // Exactly 64 chars total (first char + 63 subsequent) — max allowed
        assert!(ComponentIdentity::new(
            "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "1.0.0",
            "s.sh",
            Category::Core
        )
        .is_ok());
    }

    #[test]
    fn test_component_identity_id_invalid() {
        // Too short (1 char)
        assert!(ComponentIdentity::new("a", "1.0.0", "s.sh", Category::Core).is_err());
        // Uppercase
        assert!(ComponentIdentity::new("PyTorch", "1.0.0", "s.sh", Category::Core).is_err());
        // Underscores
        assert!(ComponentIdentity::new("py_torch", "1.0.0", "s.sh", Category::Core).is_err());
        // Spaces
        assert!(ComponentIdentity::new("py torch", "1.0.0", "s.sh", Category::Core).is_err());
        // Starts with hyphen
        assert!(ComponentIdentity::new("-pytorch", "1.0.0", "s.sh", Category::Core).is_err());
        // Too long (65 chars: first char + 64 subsequent exceeds {1,63})
        assert!(ComponentIdentity::new(
            "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "1.0.0",
            "s.sh",
            Category::Core
        )
        .is_err());
        // Empty
        assert!(ComponentIdentity::new("", "1.0.0", "s.sh", Category::Core).is_err());
    }

    #[test]
    fn test_component_identity_version_valid() {
        assert!(ComponentIdentity::new("ab", "1.0.0", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("ab", "0.0.0", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("ab", "99.99.99", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("ab", "1.0.0-rc1", "s.sh", Category::Core).is_ok());
        assert!(ComponentIdentity::new("ab", "1.0.0-alpha.1", "s.sh", Category::Core).is_ok());
    }

    #[test]
    fn test_component_identity_version_invalid() {
        // Missing patch
        assert!(ComponentIdentity::new("ab", "1.0", "s.sh", Category::Core).is_err());
        // Text in version
        assert!(ComponentIdentity::new("ab", "v1.0.0", "s.sh", Category::Core).is_err());
        // Empty
        assert!(ComponentIdentity::new("ab", "", "s.sh", Category::Core).is_err());
        // Random string
        assert!(ComponentIdentity::new("ab", "latest", "s.sh", Category::Core).is_err());
    }

    #[test]
    fn test_component_identity_script_path_valid() {
        assert!(ComponentIdentity::new("ab", "1.0.0", "install.sh", Category::Core).is_ok());
        assert!(
            ComponentIdentity::new("ab", "1.0.0", "scripts/install.sh", Category::Core).is_ok()
        );
        assert!(
            ComponentIdentity::new("ab", "1.0.0", "dir/sub/install.sh", Category::Core).is_ok()
        );
    }

    #[test]
    fn test_component_identity_script_path_invalid() {
        // Traversal
        assert!(ComponentIdentity::new("ab", "1.0.0", "../install.sh", Category::Core).is_err());
        assert!(
            ComponentIdentity::new("ab", "1.0.0", "dir/../../install.sh", Category::Core).is_err()
        );
        // Absolute path
        assert!(
            ComponentIdentity::new("ab", "1.0.0", "/usr/bin/install.sh", Category::Core).is_err()
        );
        // Empty
        assert!(ComponentIdentity::new("ab", "1.0.0", "", Category::Core).is_err());
    }

    #[test]
    fn test_component_identity_serde_roundtrip() {
        let ci = ComponentIdentity::new(
            "pytorch",
            "2.1.0",
            "scripts/install_pytorch.sh",
            Category::Foundation,
        )
        .unwrap();
        let json = serde_json::to_string(&ci).unwrap();
        let back: ComponentIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(ci, back);
    }

    // ---- Component (legacy) tests (VAL-CORE-001) ----

    #[test]
    fn test_component_serde_roundtrip_fully_populated() {
        let comp = Component {
            id: "pytorch".to_string(),
            name: "PyTorch with ROCm".to_string(),
            description: "PyTorch optimized for AMD GPUs".to_string(),
            script: "install_pytorch_rocm.sh".to_string(),
            category: Category::Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".to_string(),
            needs_sudo: true,
        };
        let json = serde_json::to_string(&comp).unwrap();
        let back: Component = serde_json::from_str(&json).unwrap();
        assert_eq!(comp, back);
    }

    #[test]
    fn test_component_with_each_category_roundtrip() {
        for cat in Category::all() {
            let comp = Component {
                id: format!("test-{cat:?}"),
                name: format!("Test {cat:?}"),
                description: "test".to_string(),
                script: "test.sh".to_string(),
                category: *cat,
                required: false,
                selected: false,
                installed: false,
                progress: 0.0,
                estimate: "1 min".to_string(),
                needs_sudo: false,
            };
            let json = serde_json::to_string(&comp).unwrap();
            let back: Component = serde_json::from_str(&json).unwrap();
            assert_eq!(comp.category, back.category);
        }
    }
}
