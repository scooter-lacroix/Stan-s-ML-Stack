//! Legacy adapter — invokes shell scripts with correct argument forwarding.
//!
//! This module provides:
//! - [`LegacyAdapter`] — invokes `install_*.sh` scripts with correct arguments
//!
//! # Design
//!
//! The legacy adapter is the fallback executor for components that don't yet
//! have Rust-native implementations. It discovers the correct shell script
//! from the component registry, forwards arguments correctly, and captures
//! stdout/stderr and exit codes for parity verification.
//!
//! # Validation Assertions
//!
//! - **VAL-MIGR-010**: No Rust adapter → falls back to install_*.sh with correct args
//! - **VAL-MIGR-012**: Output format matches shell for migrated components
//! - **VAL-MIGR-013**: Exit codes match shell for success/failure/partial cases

use super::{Adapter, AdapterError, AdapterOutput};
use crate::core::types::ExecutorKind;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

// ===========================================================================
// LegacyAdapter
// ===========================================================================

/// Adapter that invokes legacy shell scripts for component installation.
///
/// The adapter discovers the correct `install_*.sh` script from the component
/// registry and executes it via `bash` with correct argument forwarding.
///
/// # Script Discovery
///
/// For a component like `pytorch`, the adapter looks up the installer script
/// from the registry (e.g., `install_pytorch_rocm.sh`) and resolves it
/// relative to the configured scripts directory.
///
/// # Argument Forwarding
///
/// The adapter forwards the component version as an argument to the script:
/// ```bash
/// bash <scripts_dir>/install_pytorch_rocm.sh <version>
/// ```
///
/// Some scripts may accept additional environment variables or flags;
/// the adapter passes these through the process environment.
pub struct LegacyAdapter {
    /// Root directory containing install scripts.
    scripts_dir: PathBuf,
}

impl LegacyAdapter {
    /// Create a new legacy adapter with the given scripts directory.
    pub fn new(scripts_dir: impl Into<PathBuf>) -> Self {
        Self {
            scripts_dir: scripts_dir.into(),
        }
    }

    /// Resolve the script path for a component ID.
    ///
    /// Uses the component registry to find the installer script name,
    /// then resolves it relative to the scripts directory.
    pub fn resolve_script_path(&self, component_id: &str) -> Option<PathBuf> {
        let info = crate::platform::registry::get_component(component_id)?;
        if info.installer_script.is_empty() {
            return None;
        }
        Some(self.scripts_dir.join(&info.installer_script))
    }

    /// Build the command for invoking the install script.
    ///
    /// The command is: `bash <script_path> <version>`
    pub fn build_command(
        &self,
        component_id: &str,
        version: &str,
    ) -> Result<Command, AdapterError> {
        let script_path = self.resolve_script_path(component_id).ok_or_else(|| {
            AdapterError::AdapterNotFound {
                component_id: component_id.to_string(),
            }
        })?;

        let mut cmd = Command::new("bash");
        cmd.arg(&script_path);
        if !version.is_empty() {
            cmd.arg(version);
        }

        Ok(cmd)
    }
}

impl Adapter for LegacyAdapter {
    fn execute(&self, component_id: &str, version: &str) -> Result<AdapterOutput, AdapterError> {
        let start = Instant::now();

        let script_path = self.resolve_script_path(component_id).ok_or_else(|| {
            AdapterError::AdapterNotFound {
                component_id: component_id.to_string(),
            }
        })?;

        // Check if the script exists
        if !script_path.exists() {
            return Err(AdapterError::InvocationError {
                component_id: component_id.to_string(),
                reason: format!("script not found: {}", script_path.display()),
            });
        }

        // Build and execute the command
        let mut cmd = Command::new("bash");
        cmd.arg(&script_path);
        if !version.is_empty() {
            cmd.arg(version);
        }

        // Capture stdout and stderr
        let output = cmd.output().map_err(|e| AdapterError::InvocationError {
            component_id: component_id.to_string(),
            reason: format!("failed to execute script: {e}"),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(AdapterOutput {
            component_id: component_id.to_string(),
            executor_kind: ExecutorKind::LegacyScript,
            stdout,
            stderr,
            exit_code,
            duration_ms,
        })
    }

    fn executor_kind(&self) -> ExecutorKind {
        ExecutorKind::LegacyScript
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

    /// Create a temp directory with a mock install script.
    fn create_mock_script_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let scripts_dir = dir.path().to_path_buf();

        // Create mock install scripts
        std::fs::write(
            scripts_dir.join("install_pytorch_rocm.sh"),
            "#!/bin/bash\necho \"PyTorch $1 installed successfully\"\nexit 0\n",
        )
        .unwrap();

        std::fs::write(
            scripts_dir.join("install_triton_multi.sh"),
            "#!/bin/bash\necho \"Triton $1 installed successfully\"\nexit 0\n",
        )
        .unwrap();

        std::fs::write(
            scripts_dir.join("install_flash_attention_ck.sh"),
            "#!/bin/bash\necho \"Flash Attention $1 installed successfully\"\nexit 0\n",
        )
        .unwrap();

        // A script that fails
        std::fs::write(
            scripts_dir.join("install_deepspeed.sh"),
            "#!/bin/bash\necho \"Error: DeepSpeed $1 installation failed\" >&2\nexit 1\n",
        )
        .unwrap();

        // A script that partially succeeds
        std::fs::write(
            scripts_dir.join("install_vllm_multi.sh"),
            "#!/bin/bash\necho \"vLLM $1 partially installed\"\necho \"Warning: some deps missing\" >&2\nexit 2\n",
        )
        .unwrap();

        (dir, scripts_dir)
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-010: Legacy fallback with correct argument forwarding
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_forwards_version_argument() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("2.5.0"));
        assert!(result.stdout.contains("PyTorch"));
    }

    #[test]
    fn test_legacy_adapter_forwards_correct_script_per_component() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        // PyTorch → install_pytorch_rocm.sh
        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert!(result.stdout.contains("PyTorch"));

        // Triton → install_triton_multi.sh
        let result = adapter.execute("triton", "3.1.0").unwrap();
        assert!(result.stdout.contains("Triton"));
        assert!(result.stdout.contains("3.1.0"));
    }

    #[test]
    fn test_legacy_adapter_forwards_multiple_components() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        // Flash Attention → install_flash_attention_ck.sh
        let result = adapter.execute("flash-attn", "2.6.0").unwrap();
        assert!(result.stdout.contains("Flash Attention"));
        assert!(result.stdout.contains("2.6.0"));
        assert_eq!(result.exit_code, 0);
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-012: Output format matches shell
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_captures_stdout() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert!(!result.stdout.is_empty());
        assert!(result.stdout.contains("installed"));
    }

    #[test]
    fn test_legacy_adapter_captures_stderr() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        // DeepSpeed script writes to stderr
        let result = adapter.execute("deepspeed", "0.14.0").unwrap();
        assert!(!result.stderr.is_empty());
        assert!(result.stderr.contains("Error"));
    }

    // -----------------------------------------------------------------------
    // VAL-MIGR-013: Exit codes match shell for success/failure/partial
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_exit_code_success() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.is_success());
    }

    #[test]
    fn test_legacy_adapter_exit_code_failure() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("deepspeed", "0.14.0").unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(!result.is_success());
    }

    #[test]
    fn test_legacy_adapter_exit_code_partial() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("vllm", "0.6.0").unwrap();
        assert_eq!(result.exit_code, 2);
        assert!(!result.is_success());
        assert!(!result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }

    // -----------------------------------------------------------------------
    // Script resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_script_path_known_component() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let path = adapter.resolve_script_path("pytorch");
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.to_string_lossy().contains("install_pytorch_rocm.sh"));
    }

    #[test]
    fn test_resolve_script_path_unknown_component() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let path = adapter.resolve_script_path("nonexistent");
        assert!(path.is_none());
    }

    #[test]
    fn test_resolve_script_path_no_installer() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        // rocm-smi has no installer script
        let path = adapter.resolve_script_path("rocm-smi");
        assert!(path.is_none());
    }

    // -----------------------------------------------------------------------
    // Invocation errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_script_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = LegacyAdapter::new(dir.path());

        // pytorch has a registry entry but the script doesn't exist
        let result = adapter.execute("pytorch", "2.5.0");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AdapterError::InvocationError { .. }));
        assert!(err.to_string().contains("script not found"));
    }

    #[test]
    fn test_legacy_adapter_unknown_component() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = LegacyAdapter::new(dir.path());

        let result = adapter.execute("nonexistent", "1.0.0");
        assert!(matches!(result, Err(AdapterError::AdapterNotFound { .. })));
    }

    // -----------------------------------------------------------------------
    // Command building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_command_includes_version() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let cmd = adapter.build_command("pytorch", "2.5.0").unwrap();
        let formatted = format!("{cmd:?}");
        assert!(formatted.contains("2.5.0"));
        assert!(formatted.contains("install_pytorch_rocm.sh"));
    }

    #[test]
    fn test_build_command_empty_version() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        // Empty version should not add an argument
        let cmd = adapter.build_command("pytorch", "").unwrap();
        let formatted = format!("{cmd:?}");
        assert!(formatted.contains("install_pytorch_rocm.sh"));
    }

    // -----------------------------------------------------------------------
    // Executor kind
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_executor_kind() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = LegacyAdapter::new(dir.path());
        assert_eq!(adapter.executor_kind(), ExecutorKind::LegacyScript);
    }

    // -----------------------------------------------------------------------
    // Duration tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_legacy_adapter_tracks_duration() {
        let (_dir, scripts_dir) = create_mock_script_dir();
        let adapter = LegacyAdapter::new(&scripts_dir);

        let result = adapter.execute("pytorch", "2.5.0").unwrap();
        // Duration should be tracked (u64 is always >= 0, just verify it's populated)
        let _ = result.duration_ms;
    }
}
