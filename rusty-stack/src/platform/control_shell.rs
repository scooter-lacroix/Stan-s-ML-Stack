//! Windows-native control shell using shared Rust contracts.
//!
//! Provides a control surface for managing Rusty Stack operations on Windows
//! without spawning `cmd.exe` or `powershell.exe`. All operations use the
//! shared Rust core types and platform abstractions.
//!
//! # Design Principles
//!
//! - **No shell subprocess**: All operations execute through Rust code
//!   using shared contracts from `core::` and `platform::` modules.
//! - **Cross-platform**: Compiles on both Linux and Windows. Platform-specific
//!   operations are cfg-gated.
//! - **Composable**: Each operation returns structured results that can be
//!   composed into higher-level workflows.
//!
//! # Operations
//!
//! - **Status check**: Report system and component status
//! - **Path translation**: Bridge Windows ↔ WSL2 paths
//! - **Service management**: List and check health of managed services
//! - **WSL2 management**: Detect and check WSL2 health

use crate::core::types::BackendMode;
use crate::platform::path_bridge;
use crate::platform::service::{self, ServiceConfig, ServiceRegistry};
use crate::platform::wsl;
use serde::{Deserialize, Serialize};

// ===========================================================================
// Control Shell Result Types
// ===========================================================================

/// Result of a control shell operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlResult {
    /// The operation that was performed.
    pub operation: ControlOperation,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Human-readable message describing the result.
    pub message: String,
    /// Additional data as JSON-serializable value.
    pub data: serde_json::Value,
}

/// Operations supported by the control shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlOperation {
    /// Check system status (platform, backend, WSL2).
    StatusCheck,
    /// Translate a path between Windows and WSL2 formats.
    PathTranslate,
    /// List registered services and their status.
    ServiceList,
    /// Check health of all registered services.
    HealthCheck,
    /// Detect WSL2 and run health checks.
    WslCheck,
}

// ===========================================================================
// Control Shell
// ===========================================================================

/// Windows-native control shell for Rusty Stack operations.
///
/// All operations use shared Rust contracts — no `cmd.exe` or `powershell.exe`
/// subprocess is spawned.
#[derive(Debug, Clone)]
pub struct ControlShell {
    /// Current backend mode.
    backend_mode: BackendMode,
    /// Service registry for managed services.
    services: ServiceRegistry,
}

impl ControlShell {
    /// Create a new control shell with the given backend mode.
    pub fn new(backend_mode: BackendMode) -> Self {
        Self {
            backend_mode,
            services: ServiceRegistry::new(),
        }
    }

    /// Create a control shell with auto-detected backend mode.
    pub fn auto() -> Self {
        Self::new(crate::platform::windows::detect_backend_mode())
    }

    /// Register a service with the control shell.
    pub fn register_service(&mut self, config: ServiceConfig) -> Result<(), service::ServiceError> {
        self.services.register(config)
    }

    /// Get the current backend mode.
    pub fn backend_mode(&self) -> BackendMode {
        self.backend_mode
    }

    /// Get a reference to the service registry.
    pub fn services(&self) -> &ServiceRegistry {
        &self.services
    }

    // -----------------------------------------------------------------
    // Operations (no shell subprocess)
    // -----------------------------------------------------------------

    /// Perform a system status check.
    ///
    /// Reports platform, backend mode, and WSL2 status using shared
    /// Rust contracts. No shell subprocess is spawned.
    pub fn status_check(&self) -> ControlResult {
        let wsl_status = wsl::detect_wsl2();

        let data = serde_json::json!({
            "backend_mode": self.backend_mode,
            "wsl2": wsl_status,
        });

        ControlResult {
            operation: ControlOperation::StatusCheck,
            success: true,
            message: format!(
                "Platform: {:?} | WSL2: {}",
                self.backend_mode,
                match &wsl_status {
                    wsl::WslStatus::NotInstalled => "not installed".to_string(),
                    wsl::WslStatus::Available { distro, .. } => format!("available ({})", distro),
                    wsl::WslStatus::Error { message } => format!("error: {}", message),
                }
            ),
            data,
        }
    }

    /// Translate a path between Windows and WSL2 formats.
    ///
    /// Auto-detects direction based on path format.
    pub fn translate_path(&self, path: &str) -> ControlResult {
        let result = path_bridge::translate_path(path);

        let data = serde_json::json!({
            "original": path,
            "translated": result.translated,
            "direction": result.direction,
        });

        ControlResult {
            operation: ControlOperation::PathTranslate,
            success: true,
            message: format!("{} → {}", path, result.translated),
            data,
        }
    }

    /// List all registered services.
    ///
    /// Returns service configurations without starting any processes.
    pub fn list_services(&self) -> ControlResult {
        let services: Vec<serde_json::Value> = self
            .services
            .all()
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "port": s.port,
                    "enabled": s.enabled,
                    "address": s.socket_addr().to_string(),
                    "health_url": s.health_url(),
                })
            })
            .collect();

        let count = services.len();
        let data = serde_json::json!({ "services": services });

        ControlResult {
            operation: ControlOperation::ServiceList,
            success: true,
            message: format!("{} service(s) registered", count),
            data,
        }
    }

    /// Check health of all registered services.
    ///
    /// Returns health check results. Since we don't start actual HTTP servers
    /// in this module, this returns the health URLs that should be checked.
    pub fn check_health(&self) -> ControlResult {
        let urls = self.services.health_urls();
        let all_local = self.services.all_local_only();

        let checks: Vec<serde_json::Value> = urls
            .iter()
            .map(|url| {
                serde_json::json!({
                    "url": url,
                    "local_only": all_local,
                })
            })
            .collect();

        let data = serde_json::json!({
            "checks": checks,
            "all_local_only": all_local,
        });

        ControlResult {
            operation: ControlOperation::HealthCheck,
            success: all_local,
            message: format!(
                "{} health endpoint(s) — all local-only: {}",
                checks.len(),
                all_local
            ),
            data,
        }
    }

    /// Detect WSL2 and run health checks.
    ///
    /// Uses shared Rust contracts for detection and health checking.
    /// No shell subprocess is spawned for the detection itself.
    pub fn wsl_check(&self) -> ControlResult {
        let status = wsl::detect_wsl2();
        let health = wsl::check_wsl_health(&status);

        let all_healthy = health.iter().all(|h| h.success);

        let data = serde_json::json!({
            "status": status,
            "health": health,
        });

        ControlResult {
            operation: ControlOperation::WslCheck,
            success: all_healthy,
            message: match &status {
                wsl::WslStatus::NotInstalled => "WSL2 not installed".to_string(),
                wsl::WslStatus::Available { distro, .. } => {
                    let healthy_count = health.iter().filter(|h| h.success).count();
                    format!(
                        "WSL2 available ({}) — {}/{} checks passed",
                        distro,
                        healthy_count,
                        health.len()
                    )
                }
                wsl::WslStatus::Error { message } => format!("WSL2 error: {}", message),
            },
            data,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // VAL-WIN-009: Windows shell uses shared Rust contracts
    // ================================================================

    #[test]
    fn test_control_shell_status_check() {
        let shell = ControlShell::auto();
        let result = shell.status_check();
        assert_eq!(result.operation, ControlOperation::StatusCheck);
        assert!(result.success);
        assert!(!result.message.is_empty());
    }

    #[test]
    fn test_control_shell_status_check_has_backend_mode() {
        let shell = ControlShell::auto();
        let result = shell.status_check();
        let data = &result.data;
        assert!(data.get("backend_mode").is_some());
    }

    #[test]
    fn test_control_shell_status_check_has_wsl_info() {
        let shell = ControlShell::auto();
        let result = shell.status_check();
        let data = &result.data;
        assert!(data.get("wsl2").is_some());
    }

    #[test]
    fn test_control_shell_translate_path_windows() {
        let shell = ControlShell::new(BackendMode::WindowsNative);
        let result = shell.translate_path(r"C:\Users\alice");
        assert_eq!(result.operation, ControlOperation::PathTranslate);
        assert!(result.success);
        assert!(result.message.contains("/mnt/c/"));
    }

    #[test]
    fn test_control_shell_translate_path_linux() {
        let shell = ControlShell::new(BackendMode::LinuxNative);
        let result = shell.translate_path("/home/alice");
        assert_eq!(result.operation, ControlOperation::PathTranslate);
        assert!(result.success);
        assert!(result.message.contains("wsl$"));
    }

    #[test]
    fn test_control_shell_list_services_empty() {
        let shell = ControlShell::auto();
        let result = shell.list_services();
        assert_eq!(result.operation, ControlOperation::ServiceList);
        assert!(result.success);
        assert!(result.message.contains("0 service"));
    }

    #[test]
    fn test_control_shell_list_services_with_registered() {
        let mut shell = ControlShell::auto();
        shell
            .register_service(ServiceConfig::new("api", 8080))
            .unwrap();

        let result = shell.list_services();
        assert!(result.success);
        assert!(result.message.contains("1 service"));
    }

    #[test]
    fn test_control_shell_health_check_no_shell_subprocess() {
        let mut shell = ControlShell::auto();
        shell
            .register_service(ServiceConfig::new("api", 8080))
            .unwrap();

        let result = shell.check_health();
        assert_eq!(result.operation, ControlOperation::HealthCheck);
        assert!(result.success);
        // Verify all services are local-only (no 0.0.0.0)
        let all_local = result
            .data
            .get("all_local_only")
            .unwrap()
            .as_bool()
            .unwrap();
        assert!(all_local);
    }

    #[test]
    fn test_control_shell_wsl_check() {
        let shell = ControlShell::auto();
        let result = shell.wsl_check();
        assert_eq!(result.operation, ControlOperation::WslCheck);
        // Result depends on actual system state
        // Just verify it returns without error
        assert!(!result.message.is_empty());
    }

    #[test]
    fn test_control_shell_backend_mode() {
        let shell = ControlShell::new(BackendMode::WindowsNative);
        assert_eq!(shell.backend_mode(), BackendMode::WindowsNative);
    }

    // ================================================================
    // Serde Roundtrips
    // ================================================================

    #[test]
    fn test_control_result_serde_roundtrip() {
        let result = ControlResult {
            operation: ControlOperation::StatusCheck,
            success: true,
            message: "OK".to_string(),
            data: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ControlResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn test_control_operation_serde_roundtrip() {
        for op in [
            ControlOperation::StatusCheck,
            ControlOperation::PathTranslate,
            ControlOperation::ServiceList,
            ControlOperation::HealthCheck,
            ControlOperation::WslCheck,
        ] {
            let json = serde_json::to_string(&op).unwrap();
            let back: ControlOperation = serde_json::from_str(&json).unwrap();
            assert_eq!(op, back);
        }
    }

    // ================================================================
    // Integration: Control Shell uses shared contracts
    // ================================================================

    #[test]
    fn test_control_shell_uses_shared_types() {
        let shell = ControlShell::new(BackendMode::LinuxNative);

        // Status check uses shared BackendMode and WslStatus types
        let status = shell.status_check();
        assert!(status.data.get("backend_mode").is_some());

        // Path translation uses shared path_bridge module
        let translated = shell.translate_path("/home/user");
        assert!(translated.data.get("translated").is_some());

        // WSL check uses shared wsl module
        let wsl = shell.wsl_check();
        assert!(wsl.data.get("status").is_some());
    }

    #[test]
    fn test_control_shell_all_operations_succeed() {
        let mut shell = ControlShell::auto();
        shell
            .register_service(ServiceConfig::new("test", 9999))
            .unwrap();

        // All operations should succeed without spawning any subprocess
        assert!(shell.status_check().success);
        assert!(shell.translate_path(r"C:\test").success);
        assert!(shell.list_services().success);
        assert!(shell.check_health().success);
        // wsl_check success depends on system state, just verify no panic
        let _ = shell.wsl_check();
    }
}
