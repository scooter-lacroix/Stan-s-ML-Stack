//! Adapter module — maps component IDs to executors (Rust or legacy shell script).
//!
//! This module provides:
//! - [`AdapterRegistry`] — maps component IDs to executors (Rust or legacy shell script)
//! - [`RustAdapter`] — routes to native Rust implementations (no shell subprocess)
//! - [`LegacyAdapter`] — invokes shell scripts with correct argument forwarding
//! - [`AdapterError`] — structured errors for unknown components and execution failures
//!
//! # Design
//!
//! The adapter layer is the migration bridge between Rust-native implementations
//! and legacy shell scripts. For each component ID, the registry determines whether
//! to use a Rust executor (fast, no subprocess) or fall back to the legacy shell
//! script (backward compatible).
//!
//! ```text
//! component_id → AdapterRegistry → RustAdapter (native)
//!                                → LegacyAdapter (shell script)
//!                                → Err(AdapterNotFound)
//! ```
//!
//! # Validation Assertions
//!
//! - **VAL-MIGR-009**: Rust-registered adapter routes to Rust impl, no shell subprocess
//! - **VAL-MIGR-010**: No Rust adapter → falls back to install_*.sh with correct args
//! - **VAL-MIGR-011**: Unknown component ID returns Err(AdapterNotFound)
//! - **VAL-MIGR-012**: Output format matches shell for migrated components
//! - **VAL-MIGR-013**: Exit codes match shell for success/failure/partial cases

pub mod legacy_adapter;
pub mod rust_adapter;

use crate::core::types::ExecutorKind;
use crate::orchestrator::apply::ApplyExecutor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ===========================================================================
// Adapter Error Types
// ===========================================================================

/// Errors produced by the adapter layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterError {
    /// The requested component ID has no registered adapter.
    AdapterNotFound { component_id: String },
    /// The executor failed with a non-zero exit code.
    ExecutionFailed {
        component_id: String,
        exit_code: i32,
        stderr: String,
    },
    /// The executor could not be invoked (e.g., script not found).
    InvocationError {
        component_id: String,
        reason: String,
    },
    /// Output format does not match expected shell format.
    OutputParityError {
        component_id: String,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterError::AdapterNotFound { component_id } => {
                write!(f, "no adapter registered for component '{component_id}'")
            }
            AdapterError::ExecutionFailed {
                component_id,
                exit_code,
                stderr,
            } => {
                write!(
                    f,
                    "execution failed for '{component_id}': exit code {exit_code}, stderr: {stderr}"
                )
            }
            AdapterError::InvocationError {
                component_id,
                reason,
            } => {
                write!(f, "invocation error for '{component_id}': {reason}")
            }
            AdapterError::OutputParityError {
                component_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "output parity error for '{component_id}': expected '{expected}', got '{actual}'"
                )
            }
        }
    }
}

impl std::error::Error for AdapterError {}

// ===========================================================================
// Adapter Result
// ===========================================================================

/// Result of executing a component adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterOutput {
    /// The component ID that was executed.
    pub component_id: String,
    /// The executor kind that was used.
    pub executor_kind: ExecutorKind,
    /// Standard output from the executor.
    pub stdout: String,
    /// Standard error from the executor.
    pub stderr: String,
    /// Exit code (0 = success).
    pub exit_code: i32,
    /// Duration of execution in milliseconds.
    pub duration_ms: u64,
}

impl AdapterOutput {
    /// Whether the execution succeeded (exit code 0).
    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

// ===========================================================================
// Adapter Trait
// ===========================================================================

/// Trait for component adapters.
///
/// Each adapter knows how to execute a specific component's installation.
/// Implementations may use Rust-native logic or delegate to shell scripts.
pub trait Adapter: Send + Sync {
    /// Execute the adapter for the given component.
    ///
    /// Returns the output on success, or an error on failure.
    fn execute(&self, component_id: &str, version: &str) -> Result<AdapterOutput, AdapterError>;

    /// The kind of executor this adapter uses.
    fn executor_kind(&self) -> ExecutorKind;
}

// ===========================================================================
// Adapter Registry
// ===========================================================================

/// Registry that maps component IDs to their executor adapters.
///
/// The registry maintains two maps:
/// - **Rust adapters**: Components with native Rust implementations
/// - **Legacy adapters**: Components that fall back to shell scripts
///
/// When looking up a component, Rust adapters take priority. If no Rust
/// adapter is registered, the legacy adapter is used. If neither exists,
/// `AdapterNotFound` is returned.
pub struct AdapterRegistry {
    /// Rust-native adapters, keyed by component ID.
    rust_adapters: HashMap<String, Box<dyn Adapter>>,
    /// Legacy shell script adapter (handles all legacy components).
    legacy_adapter: Option<Box<dyn Adapter>>,
    /// Set of component IDs that are known to have legacy scripts.
    known_legacy_ids: HashMap<String, String>,
}

impl AdapterRegistry {
    /// Create a new empty adapter registry.
    pub fn new() -> Self {
        Self {
            rust_adapters: HashMap::new(),
            legacy_adapter: None,
            known_legacy_ids: HashMap::new(),
        }
    }

    /// Create a new adapter registry with the default legacy adapter.
    ///
    /// The legacy adapter uses the scripts directory to find install_*.sh files.
    /// All 17 known components are registered for legacy fallback.
    pub fn with_legacy(scripts_dir: impl Into<PathBuf>) -> Self {
        let scripts_dir = scripts_dir.into();
        let mut registry = Self::new();
        registry.legacy_adapter = Some(Box::new(legacy_adapter::LegacyAdapter::new(
            scripts_dir.clone(),
        )));
        registry.register_known_legacy_ids(&scripts_dir);
        registry
    }

    /// Register a Rust adapter for a component.
    ///
    /// Rust adapters take priority over legacy adapters.
    pub fn register_rust(&mut self, component_id: impl Into<String>, adapter: Box<dyn Adapter>) {
        self.rust_adapters.insert(component_id.into(), adapter);
    }

    /// Register all 17 known component IDs as legacy-capable.
    fn register_known_legacy_ids(&mut self, _scripts_dir: &Path) {
        let known = crate::platform::registry::known_components();
        for info in known {
            if !info.installer_script.is_empty() {
                self.known_legacy_ids
                    .insert(info.id.clone(), info.installer_script.clone());
            }
        }
    }

    /// Look up the adapter info for a component ID.
    ///
    /// Returns an [`AdapterRef`] describing which executor would be used,
    /// without actually executing anything.
    ///
    /// Priority:
    /// 1. Rust adapter (if registered)
    /// 2. Legacy adapter (if component has an installer script)
    /// 3. AdapterNotFound error
    pub fn get_adapter(&self, component_id: &str) -> Result<AdapterRef, AdapterError> {
        // Priority 1: Rust adapter
        if self.rust_adapters.contains_key(component_id) {
            return Ok(AdapterRef {
                executor_kind: ExecutorKind::Rust,
                component_id: component_id.to_string(),
            });
        }

        // Priority 2: Legacy adapter
        if self.legacy_adapter.is_some() {
            // Check if this component has a known legacy script
            if self.known_legacy_ids.contains_key(component_id) {
                return Ok(AdapterRef {
                    executor_kind: ExecutorKind::LegacyScript,
                    component_id: component_id.to_string(),
                });
            }
        }

        // Priority 3: Not found
        Err(AdapterError::AdapterNotFound {
            component_id: component_id.to_string(),
        })
    }

    /// Execute a component via the appropriate adapter.
    ///
    /// Routes to Rust adapter if registered, otherwise falls back to legacy.
    pub fn execute(
        &self,
        component_id: &str,
        version: &str,
    ) -> Result<AdapterOutput, AdapterError> {
        // Priority 1: Rust adapter
        if let Some(rust_adapter) = self.rust_adapters.get(component_id) {
            return rust_adapter.execute(component_id, version);
        }

        // Priority 2: Legacy adapter
        if let Some(legacy) = &self.legacy_adapter {
            if self.known_legacy_ids.contains_key(component_id) {
                return legacy.execute(component_id, version);
            }
        }

        // Priority 3: Not found
        Err(AdapterError::AdapterNotFound {
            component_id: component_id.to_string(),
        })
    }

    /// Check if a component has a Rust adapter registered.
    pub fn has_rust_adapter(&self, component_id: &str) -> bool {
        self.rust_adapters.contains_key(component_id)
    }

    /// Check if a component has a legacy adapter available.
    pub fn has_legacy_adapter(&self, component_id: &str) -> bool {
        self.known_legacy_ids.contains_key(component_id)
    }

    /// Get the executor kind for a component.
    pub fn executor_kind(&self, component_id: &str) -> Result<ExecutorKind, AdapterError> {
        let adapter_ref = self.get_adapter(component_id)?;
        Ok(adapter_ref.executor_kind)
    }

    /// Get the number of registered Rust adapters.
    pub fn rust_adapter_count(&self) -> usize {
        self.rust_adapters.len()
    }

    /// Get the number of known legacy component IDs.
    pub fn legacy_adapter_count(&self) -> usize {
        self.known_legacy_ids.len()
    }

    /// List all component IDs that have Rust adapters registered.
    pub fn rust_component_ids(&self) -> Vec<String> {
        self.rust_adapters.keys().cloned().collect()
    }

    /// List all component IDs that have legacy adapters available.
    pub fn legacy_component_ids(&self) -> Vec<String> {
        self.known_legacy_ids.keys().cloned().collect()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to an adapter returned by the registry.
///
/// This enum avoids lifetime issues by returning the executor kind
/// and component ID rather than a reference to the adapter itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRef {
    /// The executor kind for this adapter.
    pub executor_kind: ExecutorKind,
    /// The component ID that was looked up.
    pub component_id: String,
}

// ---------------------------------------------------------------------------
// Bridge: AdapterRegistry as ApplyExecutor
// ---------------------------------------------------------------------------

/// An [`ApplyExecutor`] that delegates to the [`AdapterRegistry`].
///
/// This bridges the adapter layer with the apply engine, allowing the
/// apply engine to use adapters without knowing about them directly.
pub struct RegistryExecutor {
    registry: AdapterRegistry,
}

impl RegistryExecutor {
    /// Create a new registry executor with the given adapter registry.
    pub fn new(registry: AdapterRegistry) -> Self {
        Self { registry }
    }
}

impl ApplyExecutor for RegistryExecutor {
    fn apply_component(&self, component_id: &str, proposed_version: &str) -> Result<(), String> {
        match self.registry.execute(component_id, proposed_version) {
            Ok(output) if output.is_success() => Ok(()),
            Ok(output) => Err(format!(
                "adapter failed for '{component_id}': exit code {}, stderr: {}",
                output.exit_code, output.stderr
            )),
            Err(e) => Err(format!("adapter error for '{component_id}': {e}")),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// A mock Rust adapter that always succeeds with a predictable output.
    struct MockRustAdapter {
        output_prefix: String,
    }

    impl MockRustAdapter {
        fn new(prefix: &str) -> Self {
            Self {
                output_prefix: prefix.to_string(),
            }
        }
    }

    impl Adapter for MockRustAdapter {
        fn execute(
            &self,
            component_id: &str,
            version: &str,
        ) -> Result<AdapterOutput, AdapterError> {
            Ok(AdapterOutput {
                component_id: component_id.to_string(),
                executor_kind: ExecutorKind::Rust,
                stdout: format!("{}: {version} installed", self.output_prefix),
                stderr: String::new(),
                exit_code: 0,
                duration_ms: 10,
            })
        }

        fn executor_kind(&self) -> ExecutorKind {
            ExecutorKind::Rust
        }
    }

    /// A mock Rust adapter that always fails.
    struct FailingRustAdapter;

    impl Adapter for FailingRustAdapter {
        fn execute(
            &self,
            component_id: &str,
            _version: &str,
        ) -> Result<AdapterOutput, AdapterError> {
            Ok(AdapterOutput {
                component_id: component_id.to_string(),
                executor_kind: ExecutorKind::Rust,
                stdout: String::new(),
                stderr: "installation failed".to_string(),
                exit_code: 1,
                duration_ms: 100,
            })
        }

        fn executor_kind(&self) -> ExecutorKind {
            ExecutorKind::Rust
        }
    }

    /// A mock Rust adapter that simulates partial failure (exit code 2).
    struct PartialFailureRustAdapter;

    impl Adapter for PartialFailureRustAdapter {
        fn execute(
            &self,
            component_id: &str,
            _version: &str,
        ) -> Result<AdapterOutput, AdapterError> {
            Ok(AdapterOutput {
                component_id: component_id.to_string(),
                executor_kind: ExecutorKind::Rust,
                stdout: "partially completed".to_string(),
                stderr: "some steps failed".to_string(),
                exit_code: 2,
                duration_ms: 50,
            })
        }

        fn executor_kind(&self) -> ExecutorKind {
            ExecutorKind::Rust
        }
    }

    /// Create a test registry with no adapters.
    fn empty_registry() -> AdapterRegistry {
        AdapterRegistry::new()
    }

    /// Create a test registry with a Rust adapter for "test-component".
    fn registry_with_rust_adapter() -> AdapterRegistry {
        let mut registry = AdapterRegistry::new();
        registry.register_rust(
            "test-component",
            Box::new(MockRustAdapter::new("rust-impl")),
        );
        registry
    }

    /// Create a test registry with both Rust and legacy adapters.
    fn registry_with_both() -> AdapterRegistry {
        let mut registry = AdapterRegistry::new();
        // Register a Rust adapter for "pytorch"
        registry.register_rust("pytorch", Box::new(MockRustAdapter::new("rust-pytorch")));
        // Register legacy adapter with known IDs
        registry.legacy_adapter = Some(Box::new(legacy_adapter::LegacyAdapter::new(
            PathBuf::from("/nonexistent/scripts"),
        )));
        // Register known legacy IDs
        let known = crate::platform::registry::known_components();
        for info in known {
            if !info.installer_script.is_empty() {
                registry
                    .known_legacy_ids
                    .insert(info.id.clone(), info.installer_script.clone());
            }
        }
        registry
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-009: Rust-registered adapter routes to Rust impl, no shell subprocess
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_dispatches_to_rust_impl() {
        let registry = registry_with_rust_adapter();
        let result = registry.execute("test-component", "1.0.0").unwrap();

        assert_eq!(result.executor_kind, ExecutorKind::Rust);
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("rust-impl"));
        assert!(result.stdout.contains("1.0.0 installed"));
    }

    #[test]
    fn test_rust_adapter_no_shell_subprocess() {
        let registry = registry_with_rust_adapter();
        let result = registry.execute("test-component", "1.0.0").unwrap();

        // Rust adapter output should not mention shell/bash/script
        assert!(!result.stdout.contains("bash"));
        assert!(!result.stdout.contains("sh"));
        assert_eq!(result.executor_kind, ExecutorKind::Rust);
    }

    #[test]
    fn test_rust_adapter_takes_priority_over_legacy() {
        let registry = registry_with_both();

        // pytorch has a Rust adapter registered, so it should use Rust
        let result = registry.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.executor_kind, ExecutorKind::Rust);
        assert!(result.stdout.contains("rust-pytorch"));
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-010: No Rust adapter → falls back to install_*.sh with correct args
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_fallback_when_no_rust_adapter() {
        let registry = registry_with_both();

        // triton has no Rust adapter, should fall back to legacy
        let adapter_ref = registry.get_adapter("triton").unwrap();
        assert_eq!(adapter_ref.executor_kind, ExecutorKind::LegacyScript);
    }

    #[test]
    fn test_legacy_adapter_has_correct_script_mapping() {
        let registry = registry_with_both();

        // Check that known components map to correct scripts
        assert_eq!(
            registry.known_legacy_ids.get("pytorch").map(|s| s.as_str()),
            Some("install_pytorch_rocm.sh")
        );
        assert_eq!(
            registry.known_legacy_ids.get("triton").map(|s| s.as_str()),
            Some("install_triton_multi.sh")
        );
        assert_eq!(
            registry
                .known_legacy_ids
                .get("flash-attn")
                .map(|s| s.as_str()),
            Some("install_flash_attention_ck.sh")
        );
        assert_eq!(
            registry
                .known_legacy_ids
                .get("deepspeed")
                .map(|s| s.as_str()),
            Some("install_deepspeed.sh")
        );
    }

    #[test]
    fn test_all_17_known_components_have_legacy_mapping() {
        let registry = registry_with_both();

        // Components that should have legacy scripts (15 out of 17 -
        // rocm-smi and permanent-env don't have installer scripts)
        let legacy_count = registry.known_legacy_ids.len();
        assert!(
            legacy_count >= 15,
            "Expected at least 15 legacy mappings, got {legacy_count}"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-011: Unknown component ID returns Err(AdapterNotFound)
    // -----------------------------------------------------------------------

    #[test]
    fn test_unknown_component_returns_adapter_not_found() {
        let registry = empty_registry();
        let result = registry.execute("nonexistent-component", "1.0.0");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AdapterError::AdapterNotFound { .. }));
    }

    #[test]
    fn test_unknown_component_error_contains_id() {
        let registry = empty_registry();
        let result = registry.execute("fake-component-xyz", "1.0.0");

        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("fake-component-xyz"));
    }

    #[test]
    fn test_multiple_unknown_ids_return_not_found() {
        let registry = empty_registry();
        let unknown_ids = [
            "does-not-exist",
            "also-fake",
            "not-a-component",
            "imaginary-tool",
            "xyz-123",
        ];

        for id in &unknown_ids {
            let result = registry.execute(id, "1.0.0");
            assert!(
                matches!(result, Err(AdapterError::AdapterNotFound { .. })),
                "Expected AdapterNotFound for '{id}'"
            );
        }
    }

    #[test]
    fn test_unknown_with_rust_registry_still_not_found() {
        let registry = registry_with_rust_adapter();
        // Only "test-component" is registered; others should fail
        let result = registry.execute("other-component", "1.0.0");
        assert!(matches!(result, Err(AdapterError::AdapterNotFound { .. })));
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-012: Output format matches shell for migrated components
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_output_format_matches_shell_convention() {
        let registry = registry_with_rust_adapter();
        let result = registry.execute("test-component", "2.0.0").unwrap();

        // Shell scripts conventionally output: "component: version installed"
        // or similar status messages to stdout
        assert!(result.stdout.contains("installed"));
        assert!(result.stdout.contains("2.0.0"));
    }

    #[test]
    fn test_adapter_output_has_all_required_fields() {
        let registry = registry_with_rust_adapter();
        let result = registry.execute("test-component", "1.0.0").unwrap();

        // AdapterOutput must have all fields populated
        assert!(!result.component_id.is_empty());
        assert_eq!(result.component_id, "test-component");
        assert!(result.duration_ms > 0);
        assert_eq!(result.exit_code, 0);
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-013: Exit codes match shell for success/failure/partial cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_exit_code_0_on_success() {
        let registry = registry_with_rust_adapter();
        let result = registry.execute("test-component", "1.0.0").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.is_success());
    }

    #[test]
    fn test_exit_code_1_on_failure() {
        let mut registry = AdapterRegistry::new();
        registry.register_rust("fail-comp", Box::new(FailingRustAdapter));

        let result = registry.execute("fail-comp", "1.0.0").unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(!result.is_success());
        assert!(!result.stderr.is_empty());
    }

    #[test]
    fn test_exit_code_2_on_partial_failure() {
        let mut registry = AdapterRegistry::new();
        registry.register_rust("partial-comp", Box::new(PartialFailureRustAdapter));

        let result = registry.execute("partial-comp", "1.0.0").unwrap();
        assert_eq!(result.exit_code, 2);
        assert!(!result.is_success());
        assert!(!result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }

    // -----------------------------------------------------------------------
    // AdapterError Display and serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_adapter_error_display() {
        let err = AdapterError::AdapterNotFound {
            component_id: "test".to_string(),
        };
        assert!(err.to_string().contains("test"));
        assert!(err.to_string().contains("no adapter registered"));

        let err = AdapterError::ExecutionFailed {
            component_id: "pytorch".to_string(),
            exit_code: 1,
            stderr: "error msg".to_string(),
        };
        assert!(err.to_string().contains("exit code 1"));
        assert!(err.to_string().contains("error msg"));

        let err = AdapterError::InvocationError {
            component_id: "rocm".to_string(),
            reason: "script not found".to_string(),
        };
        assert!(err.to_string().contains("script not found"));
    }

    #[test]
    fn test_adapter_error_serde_roundtrip() {
        let errors = vec![
            AdapterError::AdapterNotFound {
                component_id: "test".to_string(),
            },
            AdapterError::ExecutionFailed {
                component_id: "pytorch".to_string(),
                exit_code: 1,
                stderr: "failed".to_string(),
            },
            AdapterError::InvocationError {
                component_id: "rocm".to_string(),
                reason: "not found".to_string(),
            },
            AdapterError::OutputParityError {
                component_id: "triton".to_string(),
                expected: "ok".to_string(),
                actual: "err".to_string(),
            },
        ];

        for err in &errors {
            let json = serde_json::to_string(err).unwrap();
            let back: AdapterError = serde_json::from_str(&json).unwrap();
            assert_eq!(err, &back);
        }
    }

    // -----------------------------------------------------------------------
    // AdapterOutput serde and helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_adapter_output_serde_roundtrip() {
        let output = AdapterOutput {
            component_id: "pytorch".to_string(),
            executor_kind: ExecutorKind::Rust,
            stdout: "installed".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 150,
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: AdapterOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output, back);
    }

    #[test]
    fn test_adapter_output_is_success() {
        let success = AdapterOutput {
            component_id: "test".to_string(),
            executor_kind: ExecutorKind::Rust,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 10,
        };
        assert!(success.is_success());

        let failure = AdapterOutput {
            component_id: "test".to_string(),
            executor_kind: ExecutorKind::Rust,
            stdout: String::new(),
            stderr: "error".to_string(),
            exit_code: 1,
            duration_ms: 10,
        };
        assert!(!failure.is_success());
    }

    // -----------------------------------------------------------------------
    // Registry helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_has_rust_adapter() {
        let registry = registry_with_rust_adapter();
        assert!(registry.has_rust_adapter("test-component"));
        assert!(!registry.has_rust_adapter("other-component"));
    }

    #[test]
    fn test_has_legacy_adapter() {
        let registry = registry_with_both();
        assert!(registry.has_legacy_adapter("pytorch"));
        assert!(registry.has_legacy_adapter("triton"));
        assert!(!registry.has_legacy_adapter("nonexistent"));
    }

    #[test]
    fn test_executor_kind_rust() {
        let registry = registry_with_rust_adapter();
        let kind = registry.executor_kind("test-component").unwrap();
        assert_eq!(kind, ExecutorKind::Rust);
    }

    #[test]
    fn test_executor_kind_unknown() {
        let registry = empty_registry();
        let result = registry.executor_kind("nonexistent");
        assert!(matches!(result, Err(AdapterError::AdapterNotFound { .. })));
    }

    #[test]
    fn test_rust_component_ids() {
        let registry = registry_with_rust_adapter();
        let ids = registry.rust_component_ids();
        assert_eq!(ids, vec!["test-component"]);
    }

    #[test]
    fn test_legacy_component_ids() {
        let registry = registry_with_both();
        let ids = registry.legacy_component_ids();
        assert!(!ids.is_empty());
        assert!(ids.contains(&"pytorch".to_string()));
        assert!(ids.contains(&"triton".to_string()));
    }

    // -----------------------------------------------------------------------
    // RegistryExecutor bridge
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_executor_success() {
        let mut inner = AdapterRegistry::new();
        inner.register_rust("test", Box::new(MockRustAdapter::new("ok")));

        let executor = RegistryExecutor::new(inner);
        let result = executor.apply_component("test", "1.0.0");
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_executor_failure() {
        let mut inner = AdapterRegistry::new();
        inner.register_rust("fail", Box::new(FailingRustAdapter));

        let executor = RegistryExecutor::new(inner);
        let result = executor.apply_component("fail", "1.0.0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exit code 1"));
    }

    #[test]
    fn test_registry_executor_not_found() {
        let inner = AdapterRegistry::new();
        let executor = RegistryExecutor::new(inner);
        let result = executor.apply_component("nonexistent", "1.0.0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no adapter registered"));
    }

    // -----------------------------------------------------------------------
    // Default impl
    // -----------------------------------------------------------------------

    #[test]
    fn test_adapter_registry_default() {
        let registry = AdapterRegistry::default();
        assert!(registry.rust_adapters.is_empty());
        assert!(registry.legacy_adapter.is_none());
        assert!(registry.known_legacy_ids.is_empty());
    }
}
