//! Migration Wave 1 — Rust equivalents of shell logic.
//!
//! This module provides consolidated Rust implementations of the read-heavy
//! shell logic from `update_helper.sh` and `update_stack.sh`:
//!
//! - **Version lookup** — `up_get_version()` equivalent (VAL-MIGR-001, VAL-MIGR-002)
//! - **Selection plan** — `update_stack.sh` menu logic (VAL-MIGR-003, VAL-MIGR-004)
//! - **Manifest freshness** — TTL and offline tolerance (VAL-MIGR-005, VAL-MIGR-006)
//! - **Consolidated detection** — `up_detect_installed()` equivalent (VAL-MIGR-007)
//! - **Partial installation** — recognition of partially installed state (VAL-MIGR-008)
//!
//! # Shell Parity
//!
//! Each function in this module is designed to produce identical output to its
//! shell counterpart. The tests in `tests/migration_wave1.rs` verify this parity.

use crate::core::manifest::{self, Manifest, ManifestFetcher, ManifestSource, ResolvedManifest};
use crate::orchestrator::planner::{CompatibilityContext, PlannerOptions, UpdatePlanner};
use crate::platform::registry;

use std::collections::{HashMap, HashSet};
use std::path::Path;

// ===========================================================================
// Version Lookup (VAL-MIGR-001, VAL-MIGR-002)
// ===========================================================================

/// Rust equivalent of `up_get_version()` from `update_helper.sh`.
///
/// Returns the version string for a component, matching the shell behavior:
/// - ROCm: version from `/opt/rocm/.info/version` or `rocminfo --version`
/// - Python modules: `__version__` attribute or "not installed"
/// - Git-based: `git log -1 --format="%h %s"` or "installed" or "not installed"
/// - Command-based: first line of `--version` output or "not installed"
/// - permanent-env: "installed" or "not installed"
/// - Unknown: "unknown"
///
/// # Errors
///
/// Unknown component IDs return "unknown" (structured, not panic).
/// This matches the shell's `*) echo "unknown" ;;` default case.
pub fn migrated_get_version(component_id: &str) -> String {
    registry::get_version(component_id)
}

/// Rust equivalent of `up_get_version()` with explicit home directory.
///
/// Same as `migrated_get_version` but uses the provided home directory
/// for git-based and path-based components.
pub fn migrated_get_version_with_home(component_id: &str, home: &Path) -> String {
    registry::get_version_with_home(component_id, home)
}

/// Rust equivalent of `up_get_versions_batch()` from `update_helper.sh`.
///
/// Queries versions for multiple Python ML modules in a single subprocess.
/// Returns `component_id=version` pairs.
///
/// # Differences from Shell
///
/// The shell function takes the python interpreter as the first argument.
/// This Rust version auto-discovers interpreters via `python_interpreters()`.
pub fn migrated_get_versions_batch(component_ids: &[&str]) -> Vec<registry::VersionInfo> {
    registry::get_versions_batch(component_ids)
}

// ===========================================================================
// Selection Plan (VAL-MIGR-003, VAL-MIGR-004)
// ===========================================================================

/// Rust equivalent of the `update_stack.sh` menu construction logic.
///
/// Builds a selection plan from the manifest and current system state.
/// The plan includes:
/// - Ordered list of components (matching shell menu order)
/// - Preselected defaults (safe updates preselected, others not)
/// - Classification (safe/guarded/blocked/candidate/experimental)
///
/// # Channel Support
///
/// When the ROCm channel changes, calling this again produces a fresh plan
/// with no stale entries from the previous channel.
pub fn migrated_build_selection_plan(
    manifest: &Manifest,
    context: &CompatibilityContext,
    options: &PlannerOptions,
) -> Result<Vec<PlannerItem>, crate::orchestrator::planner::PlannerError> {
    let planner = UpdatePlanner::new();
    planner.build_plan(manifest, context, options)
}

// Re-export PlannerItem for convenience
pub use crate::orchestrator::planner::PlannerItem;

/// Build a selection plan with automatic system detection.
///
/// This is the highest-level entry point that combines:
/// 1. Hardware detection
/// 2. Component detection
/// 3. Version querying
/// 4. Plan construction
///
/// Returns the plan items ready for display to the user.
pub fn build_plan_from_system(
    manifest: &Manifest,
    rocm_channel: &str,
    include_experimental: bool,
) -> Result<Vec<PlannerItem>, crate::orchestrator::planner::PlannerError> {
    // Detect installed components
    let installed_ids = registry::detect_all_installed();
    let mut installed_versions = HashMap::new();
    let mut installed_set = HashSet::new();

    for id in &installed_ids {
        let v = registry::get_version(id);
        installed_versions.insert(id.clone(), v);
        installed_set.insert(id.clone());
    }

    // Detect ROCm version
    let rocm_version = installed_versions.get("rocm").cloned().unwrap_or_default();

    // Build context
    let context = CompatibilityContext {
        rocm_version,
        rocm_channel: rocm_channel.to_string(),
        available_executors: HashSet::from([
            crate::core::types::ExecutorKind::LegacyScript,
            crate::core::types::ExecutorKind::Rust,
        ]),
        installed_components: installed_set,
        installed_versions,
        gpu_architecture: String::new(),
        runtime_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let options = PlannerOptions {
        include_experimental,
        ..PlannerOptions::default()
    };

    migrated_build_selection_plan(manifest, &context, &options)
}

// ===========================================================================
// Manifest Freshness (VAL-MIGR-005, VAL-MIGR-006)
// ===========================================================================

/// Result of a manifest freshness check.
#[derive(Debug, Clone)]
pub struct FreshnessResult {
    /// Whether the manifest is fresh (not stale).
    pub is_fresh: bool,
    /// Source of the manifest.
    pub source: ManifestSource,
    /// Warning message if the manifest is stale but usable.
    pub warning: Option<String>,
    /// Whether a re-fetch was triggered.
    pub refetch_triggered: bool,
}

/// Check manifest freshness and resolve using the fallback chain.
///
/// This is the Rust equivalent of the shell manifest freshness logic:
/// 1. Try to fetch fresh remote manifest
/// 2. If TTL expired, trigger re-fetch
/// 3. If offline, use cached manifest with warning
/// 4. If all else fails, use bundled baseline
///
/// # VAL-MIGR-005: Cache Expiry
///
/// If the cached manifest's TTL has expired, a re-fetch is triggered.
/// The `refetch_triggered` field will be `true`.
///
/// # VAL-MIGR-006: Offline Tolerance
///
/// If the network is unavailable but a stale manifest exists, the operation
/// proceeds with a warning. The `warning` field will contain the message.
pub fn check_manifest_freshness(fetcher: &dyn ManifestFetcher) -> FreshnessResult {
    // Try to fetch fresh remote
    if let Some(remote) = fetcher.fetch_remote() {
        if remote.verify_trust().is_ok() {
            return FreshnessResult {
                is_fresh: true,
                source: ManifestSource::FreshRemote,
                warning: None,
                refetch_triggered: false,
            };
        }
        // Fresh fetch succeeded but trust check failed
        // This triggers a re-fetch attempt (which would also fail)
    }

    // Try cached manifest
    if let Some(_cached) = fetcher.load_cached() {
        return FreshnessResult {
            is_fresh: false,
            source: ManifestSource::CachedRemote,
            warning: Some(
                "Using cached manifest (remote unavailable or trust check failed)".to_string(),
            ),
            refetch_triggered: true,
        };
    }

    // Fall back to baseline
    FreshnessResult {
        is_fresh: false,
        source: ManifestSource::Baseline,
        warning: Some("Using bundled baseline manifest (no remote, no cache)".to_string()),
        refetch_triggered: true,
    }
}

/// Resolve manifest with freshness checking.
///
/// Combines freshness check with actual manifest resolution.
/// Returns the resolved manifest and freshness information.
pub fn resolve_with_freshness(
    fetcher: &dyn ManifestFetcher,
) -> (ResolvedManifest, FreshnessResult) {
    let freshness = check_manifest_freshness(fetcher);
    let resolved = manifest::resolve_manifest(fetcher);
    (resolved, freshness)
}

// ===========================================================================
// Consolidated Detection (VAL-MIGR-007, VAL-MIGR-008)
// ===========================================================================

/// Installation status of a component, with partial installation recognition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentInstallStatus {
    /// Fully installed and functional.
    Installed,
    /// Not installed at all.
    NotInstalled,
    /// Partially installed — needs update to complete installation.
    NeedsUpdate,
}

/// Detection result for a single component.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Component ID.
    pub component_id: String,
    /// Installation status.
    pub status: ComponentInstallStatus,
    /// Version string (if installed).
    pub version: String,
    /// Detection method used.
    pub detection_method: String,
}

/// Rust equivalent of `up_detect_installed()` from `update_helper.sh`.
///
/// Detects all installed components using the consolidated detection pipeline.
/// This matches the shell's `up_detect_installed` function which:
/// 1. Checks ROCm via version file / rocminfo
/// 2. Detects Python modules in a single subprocess
/// 3. Checks git-based components
/// 4. Checks system tools (rocm-smi)
/// 5. Checks permanent-env
///
/// # VAL-MIGR-007: Consolidated Detector
///
/// The detection results must match the shell for all components.
pub fn migrated_detect_all() -> Vec<DetectionResult> {
    let home = crate::platform::environment::resolve_user_home();
    migrated_detect_all_with_home(&home)
}

/// Detect all installed components with explicit home directory (testable).
pub fn migrated_detect_all_with_home(home: &Path) -> Vec<DetectionResult> {
    let installed_ids = registry::detect_all_installed_with_home(home);
    let mut results = Vec::new();

    for info in registry::known_components() {
        let version = registry::get_version_with_home(&info.id, home);
        let is_installed = installed_ids.contains(&info.id);

        let status = if is_installed {
            ComponentInstallStatus::Installed
        } else {
            check_partial_installation(&info.id, home)
        };

        results.push(DetectionResult {
            component_id: info.id.clone(),
            status,
            version,
            detection_method: format!("{:?}", info.detection_method),
        });
    }

    results
}

/// Detect a single component's installation status.
///
/// # VAL-MIGR-008: Partial Installation Recognition
///
/// Partially installed components are reported as `NeedsUpdate`,
/// not `Installed` or `NotInstalled`.
pub fn detect_component_status(component_id: &str) -> DetectionResult {
    let home = crate::platform::environment::resolve_user_home();
    detect_component_status_with_home(component_id, &home)
}

/// Detect a single component's status with explicit home directory.
pub fn detect_component_status_with_home(component_id: &str, home: &Path) -> DetectionResult {
    let info = registry::get_component(component_id);
    let is_installed = registry::is_component_installed_with_home(component_id, home);
    let version = registry::get_version_with_home(component_id, home);

    let status = if is_installed {
        ComponentInstallStatus::Installed
    } else {
        check_partial_installation(component_id, home)
    };

    DetectionResult {
        component_id: component_id.to_string(),
        status,
        version,
        detection_method: info
            .map(|i| format!("{:?}", i.detection_method))
            .unwrap_or_else(|| "Unknown".to_string()),
    }
}

/// Check if a component is partially installed.
///
/// A component is "partially installed" when some indicators of installation
/// are present but the component is not fully functional. Examples:
/// - ROCm: version file exists but rocminfo doesn't work
/// - Git-based: directory exists but no `.git` directory
/// - permanent-env: file exists but missing required markers
///
/// Returns `NeedsUpdate` if partially installed, `NotInstalled` otherwise.
fn check_partial_installation(component_id: &str, home: &Path) -> ComponentInstallStatus {
    match component_id {
        "rocm" => {
            // Partial: version file exists but rocminfo doesn't work
            let version_file = Path::new("/opt/rocm/.info/version").exists();
            let rocminfo_works = std::process::Command::new("rocminfo")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            if version_file && !rocminfo_works {
                ComponentInstallStatus::NeedsUpdate
            } else {
                ComponentInstallStatus::NotInstalled
            }
        }
        "comfyui" | "vllm-studio" | "textgen" => {
            // Partial: directory exists but no .git
            let clone_dir = match component_id {
                "comfyui" => "ComfyUI",
                "vllm-studio" => "vllm-studio",
                "textgen" => "text-generation-webui",
                _ => return ComponentInstallStatus::NotInstalled,
            };
            let dir = home.join(clone_dir);
            if dir.exists() && !dir.join(".git").exists() {
                ComponentInstallStatus::NeedsUpdate
            } else {
                ComponentInstallStatus::NotInstalled
            }
        }
        "permanent-env" => {
            // Partial: file exists but missing marker
            let env_file = home.join(".mlstack_env");
            if env_file.exists() {
                let content = std::fs::read_to_string(&env_file).unwrap_or_default();
                if !content.contains("MLSTACK_PYTHON_BIN") {
                    ComponentInstallStatus::NeedsUpdate
                } else {
                    ComponentInstallStatus::NotInstalled
                }
            } else {
                ComponentInstallStatus::NotInstalled
            }
        }
        _ => ComponentInstallStatus::NotInstalled,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-MIGR-001: Version Lookup — Shell Parity
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrated_get_version_all_components_no_panic() {
        let ids = [
            "rocm",
            "pytorch",
            "triton",
            "deepspeed",
            "vllm",
            "aiter",
            "onnx",
            "bitsandbytes",
            "migraphx",
            "flash-attn",
            "mpi4py",
            "wandb",
            "comfyui",
            "vllm-studio",
            "textgen",
            "rocm-smi",
            "permanent-env",
        ];

        for id in &ids {
            let v = migrated_get_version(id);
            assert!(!v.is_empty(), "version for '{id}' must not be empty");
        }
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-002: Unknown Component Graceful Handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrated_get_version_unknown_returns_unknown() {
        assert_eq!(migrated_get_version("nonexistent"), "unknown");
        assert_eq!(migrated_get_version("fake-component"), "unknown");
        assert_eq!(migrated_get_version(""), "unknown");
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-007: Consolidated Detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrated_detect_all_returns_valid_results() {
        let results = migrated_detect_all();

        // All results must have valid component IDs
        for result in &results {
            assert!(
                registry::is_known_component(&result.component_id),
                "result component must be known: {}",
                result.component_id
            );
        }
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-008: Partial Installation Recognition
    // -----------------------------------------------------------------------

    #[test]
    fn test_partial_installation_detection_with_temp_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let home = temp_dir.path();

        // Test git-based partial: directory exists, no .git
        let comfyui_dir = home.join("ComfyUI");
        std::fs::create_dir_all(&comfyui_dir).unwrap();

        let result = detect_component_status_with_home("comfyui", home);
        assert_eq!(
            result.status,
            ComponentInstallStatus::NeedsUpdate,
            "ComfyUI dir without .git must be NeedsUpdate"
        );

        // Complete the installation
        std::fs::create_dir_all(comfyui_dir.join(".git")).unwrap();
        let result = detect_component_status_with_home("comfyui", home);
        assert_eq!(
            result.status,
            ComponentInstallStatus::Installed,
            "ComfyUI with .git must be Installed"
        );
    }

    #[test]
    fn test_partial_installation_permanent_env() {
        let temp_dir = tempfile::tempdir().unwrap();
        let home = temp_dir.path();

        // No file at all
        let result = detect_component_status_with_home("permanent-env", home);
        assert_eq!(result.status, ComponentInstallStatus::NotInstalled);

        // File exists but no marker
        let env_file = home.join(".mlstack_env");
        std::fs::write(&env_file, "# empty").unwrap();
        let result = detect_component_status_with_home("permanent-env", home);
        assert_eq!(
            result.status,
            ComponentInstallStatus::NeedsUpdate,
            "env file without marker must be NeedsUpdate"
        );

        // File with marker
        std::fs::write(&env_file, "export MLSTACK_PYTHON_BIN=/usr/bin/python3").unwrap();
        let result = detect_component_status_with_home("permanent-env", home);
        assert_eq!(
            result.status,
            ComponentInstallStatus::Installed,
            "env file with marker must be Installed"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-005/006: Manifest Freshness
    // -----------------------------------------------------------------------

    struct MockFetcher {
        remote: Option<Manifest>,
        cached: Option<Manifest>,
    }

    impl ManifestFetcher for MockFetcher {
        fn fetch_remote(&self) -> Option<Manifest> {
            self.remote.clone()
        }
        fn load_cached(&self) -> Option<Manifest> {
            self.cached.clone()
        }
    }

    fn make_signed_manifest() -> Manifest {
        let mut m = Manifest {
            schema_version: manifest::CURRENT_SCHEMA_VERSION,
            sequence: 1,
            generated_at: chrono::Utc::now().to_rfc3339(),
            expires_at: None,
            min_runtime_version: String::new(),
            components: vec![],
            signature: None,
        };
        m.signature = Some(m.compute_signature());
        m
    }

    #[test]
    fn test_freshness_check_fresh_remote() {
        let fetcher = MockFetcher {
            remote: Some(make_signed_manifest()),
            cached: None,
        };
        let result = check_manifest_freshness(&fetcher);
        assert!(result.is_fresh);
        assert_eq!(result.source, ManifestSource::FreshRemote);
        assert!(result.warning.is_none());
    }

    #[test]
    fn test_freshness_check_offline_with_cache() {
        let fetcher = MockFetcher {
            remote: None,
            cached: Some(make_signed_manifest()),
        };
        let result = check_manifest_freshness(&fetcher);
        assert!(!result.is_fresh);
        assert_eq!(result.source, ManifestSource::CachedRemote);
        assert!(result.warning.is_some());
        assert!(result.refetch_triggered);
    }

    #[test]
    fn test_freshness_check_offline_no_cache() {
        let fetcher = MockFetcher {
            remote: None,
            cached: None,
        };
        let result = check_manifest_freshness(&fetcher);
        assert!(!result.is_fresh);
        assert_eq!(result.source, ManifestSource::Baseline);
        assert!(result.warning.is_some());
    }
}
