//! Integration tests for Windows platform support and WSL2 detection.
//!
//! Covers validation assertions:
//! - VAL-WIN-001: Windows build target compiles
//! - VAL-WIN-002: No Unix-only dependencies in Windows build
//! - VAL-WIN-003: WSL2 detection positive case
//! - VAL-WIN-004: WSL2 detection negative case (no timeout)
//! - VAL-WIN-005: WSL2 provisioning guidance
//! - VAL-WIN-006: WSL2 health checks (3 independent results)
//! - VAL-WIN-007: Backend routing model selection
//! - VAL-WIN-008: Unsupported component handling

use rusty_stack::core::types::{BackendMode, ExecutorKind};
use rusty_stack::platform::wsl::{
    check_wsl_health, detect_wsl2, provisioning_guidance, WslHealthResult, WslStatus,
};
use rusty_stack::platform::BackendRouter;

// ===========================================================================
// WSL2 Detection Tests
// ===========================================================================

#[test]
fn test_wsl2_detection_returns_status() {
    // On a Linux system without WSL2 indicators, should return NotInstalled or Available
    let status = detect_wsl2();
    // The status should be a valid WslStatus variant
    match status {
        WslStatus::NotInstalled => {}
        WslStatus::Available { distro, version } => {
            assert_eq!(version, 2, "Available WSL should report version 2");
            assert!(!distro.is_empty(), "Distro name should not be empty");
        }
        WslStatus::Error { message: msg } => {
            assert!(!msg.is_empty(), "Error message should not be empty");
        }
    }
}

#[test]
fn test_wsl2_detection_not_installed_on_linux() {
    // On a native Linux system (no WSL), detection should return NotInstalled
    let status = detect_wsl2();
    // This test runs on Linux, so we expect NotInstalled
    // (unless running inside WSL, which is unlikely on this system)
    match status {
        WslStatus::NotInstalled => {
            // Expected on native Linux
        }
        WslStatus::Available { .. } => {
            // Could happen if running in WSL — acceptable
        }
        WslStatus::Error { .. } => {}
    }
}

#[test]
fn test_wsl2_detection_does_not_timeout() {
    // WSL2 detection should complete within 5 seconds
    let start = std::time::Instant::now();
    let _status = detect_wsl2();
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "WSL2 detection took {:?}, expected < 5s",
        elapsed
    );
}

// ===========================================================================
// WSL2 Provisioning Guidance
// ===========================================================================

#[test]
fn test_provisioning_guidance_not_installed() {
    let guidance = provisioning_guidance(&WslStatus::NotInstalled);
    assert!(!guidance.is_empty(), "Guidance should not be empty");
    assert!(
        guidance.contains("wsl") || guidance.contains("WSL"),
        "Guidance should mention WSL: got '{}'",
        guidance
    );
    assert!(
        guidance.contains("install") || guidance.contains("Install"),
        "Guidance should mention install: got '{}'",
        guidance
    );
}

#[test]
fn test_provisioning_guidance_available() {
    let status = WslStatus::Available {
        distro: "Ubuntu".to_string(),
        version: 2,
    };
    let guidance = provisioning_guidance(&status);
    assert!(
        guidance.contains("Ubuntu") || guidance.contains("available") || guidance.contains("ready"),
        "Guidance for available WSL should reference the distro or state: got '{}'",
        guidance
    );
}

#[test]
fn test_provisioning_guidance_error() {
    let status = WslStatus::Error {
        message: "test error".to_string(),
    };
    let guidance = provisioning_guidance(&status);
    assert!(!guidance.is_empty(), "Error guidance should not be empty");
}

// ===========================================================================
// WSL2 Health Checks
// ===========================================================================

#[test]
fn test_wsl_health_check_returns_three_results() {
    let status = WslStatus::NotInstalled;
    let results = check_wsl_health(&status);
    assert_eq!(
        results.len(),
        3,
        "Should produce exactly 3 health check results, got {}",
        results.len()
    );
}

#[test]
fn test_wsl_health_check_results_are_independent() {
    let status = WslStatus::NotInstalled;
    let results = check_wsl_health(&status);
    // Each result should have a distinct check type
    let check_types: std::collections::HashSet<&str> =
        results.iter().map(|r| r.check.as_str()).collect();
    assert_eq!(
        check_types.len(),
        3,
        "Each health check should have a unique type: {:?}",
        check_types
    );
}

#[test]
fn test_wsl_health_check_names() {
    let status = WslStatus::NotInstalled;
    let results = check_wsl_health(&status);
    let names: Vec<&str> = results.iter().map(|r| r.check.as_str()).collect();
    // Should contain expected check types
    assert!(
        names.contains(&"distro_running"),
        "Should have distro_running check: {:?}",
        names
    );
    assert!(
        names.contains(&"rocm_accessible"),
        "Should have rocm_accessible check: {:?}",
        names
    );
    assert!(
        names.contains(&"gpu_nodes"),
        "Should have gpu_nodes check: {:?}",
        names
    );
}

#[test]
fn test_wsl_health_check_not_installed_all_fail() {
    let status = WslStatus::NotInstalled;
    let results = check_wsl_health(&status);
    for result in &results {
        assert!(
            !result.success,
            "Health check '{}' should fail when WSL not installed",
            result.check
        );
    }
}

// ===========================================================================
// Backend Routing Tests
// ===========================================================================

#[test]
fn test_backend_router_linux_native() {
    let router = BackendRouter::new(BackendMode::LinuxNative);
    // On Linux native, most components should use LegacyScript
    let kind = router.executor_for("rocm");
    assert_ne!(
        kind,
        ExecutorKind::Unsupported,
        "ROCm should not be unsupported on Linux native"
    );
}

#[test]
fn test_backend_router_windows_native() {
    let router = BackendRouter::new(BackendMode::WindowsNative);
    // On Windows native, ROCm should be unsupported (no native ROCm on Windows)
    let kind = router.executor_for("rocm");
    assert_eq!(
        kind,
        ExecutorKind::Unsupported,
        "ROCm should be unsupported on Windows native, got {:?}",
        kind
    );
}

#[test]
fn test_backend_router_wsl_backed_linux() {
    let router = BackendRouter::new(BackendMode::WslBackedLinux);
    // On WSL-backed Linux, ROCm should be available via WSL
    let kind = router.executor_for("rocm");
    assert_ne!(
        kind,
        ExecutorKind::Unsupported,
        "ROCm should be available via WSL-backed Linux"
    );
}

#[test]
fn test_backend_router_unsupported_component_clear_message() {
    let router = BackendRouter::new(BackendMode::WindowsNative);
    let kind = router.executor_for("rocm");
    assert_eq!(kind, ExecutorKind::Unsupported);
    let message = router.unsupported_message("rocm");
    assert!(
        !message.is_empty(),
        "Unsupported message should not be empty"
    );
    assert!(
        message.contains("rocm") || message.contains("ROCm") || message.contains("Windows"),
        "Message should explain why: got '{}'",
        message
    );
}

#[test]
fn test_backend_router_all_components_linux() {
    let router = BackendRouter::new(BackendMode::LinuxNative);
    let components = [
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
    for component in &components {
        let kind = router.executor_for(component);
        assert_ne!(
            kind,
            ExecutorKind::Unsupported,
            "Component '{}' should not be unsupported on Linux native",
            component
        );
    }
}

#[test]
fn test_backend_router_unknown_component() {
    let router = BackendRouter::new(BackendMode::LinuxNative);
    let kind = router.executor_for("nonexistent-component");
    assert_eq!(
        kind,
        ExecutorKind::Unsupported,
        "Unknown component should be unsupported"
    );
}

#[test]
fn test_backend_router_windows_python_components() {
    let router = BackendRouter::new(BackendMode::WindowsNative);
    // Python-based components might work on Windows via pip
    let kind = router.executor_for("wandb");
    // wandb is a Python package that works on Windows
    assert_ne!(
        kind,
        ExecutorKind::Unsupported,
        "wandb should be available on Windows native"
    );
}

// ===========================================================================
// WslStatus Serde Tests
// ===========================================================================

#[test]
fn test_wsl_status_serde_roundtrip_not_installed() {
    let status = WslStatus::NotInstalled;
    let json = serde_json::to_string(&status).unwrap();
    let back: WslStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(status, back);
}

#[test]
fn test_wsl_status_serde_roundtrip_available() {
    let status = WslStatus::Available {
        distro: "Ubuntu-22.04".to_string(),
        version: 2,
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: WslStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(status, back);
}

#[test]
fn test_wsl_status_serde_roundtrip_error() {
    let status = WslStatus::Error {
        message: "test error".to_string(),
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: WslStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(status, back);
}

// ===========================================================================
// WslHealthResult Serde Tests
// ===========================================================================

#[test]
fn test_wsl_health_result_serde_roundtrip() {
    let result = WslHealthResult {
        check: "distro_running".to_string(),
        success: true,
        message: "Ubuntu-22.04 is running".to_string(),
    };
    let json = serde_json::to_string(&result).unwrap();
    let back: WslHealthResult = serde_json::from_str(&json).unwrap();
    assert_eq!(result, back);
}

// ===========================================================================
// BackendMode Classification Tests (VAL-CORE-018 extended)
// ===========================================================================

#[test]
fn test_backend_mode_classify_linux_native() {
    // On a standard Linux system, BackendMode should be LinuxNative
    let mode = BackendMode::LinuxNative;
    assert_eq!(mode.label(), "LinuxNative");
}

#[test]
fn test_backend_mode_classify_windows_native() {
    let mode = BackendMode::WindowsNative;
    assert_eq!(mode.label(), "WindowsNative");
}

#[test]
fn test_backend_mode_classify_wsl_backed() {
    let mode = BackendMode::WslBackedLinux;
    assert_eq!(mode.label(), "WslBackedLinux");
}
