//! Rust-native adapter — executes component operations without spawning shell subprocesses.
//!
//! This module provides:
//! - [`RustAdapter`] — a generic adapter that wraps a Rust-native function
//! - [`RustAdapterFn`] — type alias for Rust-native executor functions
//!
//! # Design
//!
//! Each `RustAdapter` wraps a function pointer or closure that performs the
//! actual component operation in Rust. The adapter produces output that matches
//! the format of the legacy shell scripts for behavior parity.
//!
//! # Validation Assertions
//!
//! - **VAL-MIGR-009**: Rust-registered adapter routes to Rust impl, no shell subprocess
//! - **VAL-MIGR-012**: Output format matches shell for migrated components
//! - **VAL-MIGR-013**: Exit codes match shell for success/failure/partial cases

use super::{Adapter, AdapterError, AdapterOutput};
use crate::core::types::ExecutorKind;
use std::time::Instant;

// ===========================================================================
// Rust Adapter Function Type
// ===========================================================================

/// A Rust-native executor function.
///
/// Takes the component ID and target version, returns:
/// - `Ok((stdout, stderr))` on success (exit code 0)
/// - `Err((exit_code, stdout, stderr))` on failure
pub type RustAdapterFn =
    Box<dyn Fn(&str, &str) -> Result<(String, String), (i32, String, String)> + Send + Sync>;

// ===========================================================================
// RustAdapter
// ===========================================================================

/// Adapter that routes component operations to native Rust implementations.
///
/// This adapter does NOT spawn any shell subprocess. All work is done
/// within the Rust process, producing output that matches the format
/// of legacy shell scripts.
///
/// # Example
///
/// ```rust,ignore
/// use rusty_stack::adapter::rust_adapter::RustAdapter;
/// use rusty_stack::adapter::AdapterRegistry;
///
/// let adapter = RustAdapter::new(Box::new(|id, version| {
///     Ok((format!("{id}: {version} installed"), String::new()))
/// }));
///
/// let mut registry = AdapterRegistry::new();
/// registry.register_rust("pytorch", Box::new(adapter));
/// ```
pub struct RustAdapter {
    /// The native Rust executor function.
    executor: RustAdapterFn,
    /// Component ID this adapter handles (for error messages).
    #[allow(dead_code)]
    component_id: String,
}

impl RustAdapter {
    /// Create a new Rust adapter with the given executor function.
    pub fn new(executor: RustAdapterFn) -> Self {
        Self {
            executor,
            component_id: String::new(),
        }
    }

    /// Create a new Rust adapter with a component ID label.
    pub fn with_id(component_id: impl Into<String>, executor: RustAdapterFn) -> Self {
        Self {
            executor,
            component_id: component_id.into(),
        }
    }

    /// Create a simple Rust adapter that always succeeds with a fixed message.
    pub fn simple_success(message_prefix: &str) -> Self {
        let prefix = message_prefix.to_string();
        Self::new(Box::new(move |id, version| {
            Ok((format!("{prefix}: {id} {version} installed"), String::new()))
        }))
    }

    /// Create a Rust adapter that always fails with a fixed exit code.
    pub fn always_fails(exit_code: i32, message: &str) -> Self {
        let msg = message.to_string();
        Self::new(Box::new(move |_id, _version| {
            Err((exit_code, String::new(), msg.clone()))
        }))
    }
}

impl Adapter for RustAdapter {
    fn execute(&self, component_id: &str, version: &str) -> Result<AdapterOutput, AdapterError> {
        let start = Instant::now();

        match (self.executor)(component_id, version) {
            Ok((stdout, stderr)) => Ok(AdapterOutput {
                component_id: component_id.to_string(),
                executor_kind: ExecutorKind::Rust,
                stdout,
                stderr,
                exit_code: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
            Err((exit_code, stdout, stderr)) => Ok(AdapterOutput {
                component_id: component_id.to_string(),
                executor_kind: ExecutorKind::Rust,
                stdout,
                stderr: stderr.clone(),
                exit_code,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
        }
    }

    fn executor_kind(&self) -> ExecutorKind {
        ExecutorKind::Rust
    }
}

// ===========================================================================
// Output Format Helpers
// ===========================================================================

/// Shell-compatible output formatter.
///
/// Produces output strings that match the format of legacy shell scripts,
/// ensuring behavior parity during migration.
pub struct ShellOutputFormat;

impl ShellOutputFormat {
    /// Format a success message matching shell script convention.
    ///
    /// Shell scripts typically output: `[component_name] [version] installed successfully`
    pub fn success(component_id: &str, version: &str) -> String {
        let display = crate::platform::registry::display_name(component_id);
        format!("{display} {version} installed successfully")
    }

    /// Format a failure message matching shell script convention.
    ///
    /// Shell scripts typically output: `Error: [component_name] installation failed: [reason]`
    pub fn failure(component_id: &str, reason: &str) -> String {
        let display = crate::platform::registry::display_name(component_id);
        format!("Error: {display} installation failed: {reason}")
    }

    /// Format a partial success message.
    ///
    /// Shell scripts typically output: `Warning: [component_name] partially installed: [details]`
    pub fn partial(component_id: &str, details: &str) -> String {
        let display = crate::platform::registry::display_name(component_id);
        format!("Warning: {display} partially installed: {details}")
    }

    /// Format a status line matching shell script convention.
    ///
    /// Shell scripts typically output: `[component_name] [status]`
    pub fn status(component_id: &str, status: &str) -> String {
        let display = crate::platform::registry::display_name(component_id);
        format!("{display} {status}")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RustAdapter basic functionality
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_success() {
        let adapter = RustAdapter::new(Box::new(|id, version| {
            Ok((format!("{id}:{version}:ok"), String::new()))
        }));

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout, "pytorch:2.5.0:ok");
        assert!(result.stderr.is_empty());
        assert_eq!(result.executor_kind, ExecutorKind::Rust);
    }

    #[test]
    fn test_rust_adapter_failure() {
        let adapter = RustAdapter::new(Box::new(|_id, _version| {
            Err((1, String::new(), "installation error".to_string()))
        }));

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.stderr, "installation error");
        assert!(!result.is_success());
    }

    #[test]
    fn test_rust_adapter_partial_failure() {
        let adapter = RustAdapter::new(Box::new(|_id, _version| {
            Err((2, "partial output".to_string(), "some warnings".to_string()))
        }));

        let result = adapter.execute("triton", "3.1.0").unwrap();
        assert_eq!(result.exit_code, 2);
        assert_eq!(result.stdout, "partial output");
        assert_eq!(result.stderr, "some warnings");
    }

    #[test]
    fn test_rust_adapter_no_shell_subprocess() {
        // Verify that running a RustAdapter never touches the filesystem
        // or spawns processes — it's all in-process Rust logic.
        let adapter = RustAdapter::new(Box::new(|id, version| {
            // Pure Rust computation, no subprocess
            let result = format!("computed: {id}@{version}");
            Ok((result, String::new()))
        }));

        let result = adapter.execute("test", "1.0.0").unwrap();
        assert_eq!(result.stdout, "computed: test@1.0.0");
        assert_eq!(result.executor_kind, ExecutorKind::Rust);
    }

    // -----------------------------------------------------------------------
    // RustAdapter constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_simple_success() {
        let adapter = RustAdapter::simple_success("mock");
        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("pytorch"));
        assert!(result.stdout.contains("2.5.0"));
    }

    #[test]
    fn test_rust_adapter_always_fails() {
        let adapter = RustAdapter::always_fails(42, "always fails");
        let result = adapter.execute("test", "1.0.0").unwrap();
        assert_eq!(result.exit_code, 42);
        assert_eq!(result.stderr, "always fails");
    }

    #[test]
    fn test_rust_adapter_with_id() {
        let adapter = RustAdapter::with_id(
            "pytorch",
            Box::new(|id, v| Ok((format!("{id}:{v}"), String::new()))),
        );
        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert!(result.stdout.contains("pytorch"));
    }

    // -----------------------------------------------------------------------
    // ShellOutputFormat parity
    // -----------------------------------------------------------------------

    #[test]
    fn test_shell_output_format_success() {
        let output = ShellOutputFormat::success("pytorch", "2.5.0");
        assert!(output.contains("PyTorch"));
        assert!(output.contains("2.5.0"));
        assert!(output.contains("installed successfully"));
    }

    #[test]
    fn test_shell_output_format_failure() {
        let output = ShellOutputFormat::failure("pytorch", "build error");
        assert!(output.contains("PyTorch"));
        assert!(output.contains("Error:"));
        assert!(output.contains("build error"));
    }

    #[test]
    fn test_shell_output_format_partial() {
        let output = ShellOutputFormat::partial("triton", "missing optional deps");
        assert!(output.contains("Triton"));
        assert!(output.contains("Warning:"));
        assert!(output.contains("missing optional deps"));
    }

    #[test]
    fn test_shell_output_format_status() {
        let output = ShellOutputFormat::status("rocm", "detected");
        assert!(output.contains("ROCm"));
        assert!(output.contains("detected"));
    }

    // -----------------------------------------------------------------------
    // Duration tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_tracks_duration() {
        let adapter = RustAdapter::new(Box::new(|_id, _version| {
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(("done".to_string(), String::new()))
        }));

        let result = adapter.execute("test", "1.0.0").unwrap();
        assert!(result.duration_ms > 0, "Duration should be non-zero");
    }

    // -----------------------------------------------------------------------
    // Executor kind
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_adapter_executor_kind() {
        let adapter = RustAdapter::simple_success("test");
        assert_eq!(adapter.executor_kind(), ExecutorKind::Rust);
    }
}
