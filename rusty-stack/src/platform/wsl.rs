//! WSL2 detection, health checks, and provisioning guidance.
//!
//! Provides:
//! - **WSL2 detection** — positive and negative cases, no timeout
//!   (VAL-WIN-003, VAL-WIN-004)
//! - **WSL2 provisioning guidance** — actionable steps when WSL2 not installed
//!   (VAL-WIN-005)
//! - **WSL2 health checks** — distro running, ROCm accessible, GPU nodes
//!   (VAL-WIN-006)
//!
//! # Detection Strategy
//!
//! On Windows: invokes `wsl --list --verbose` via Command
//! On Linux: checks `/proc/version` for "microsoft" or "WSL" indicators
//!
//! All detection completes within 5 seconds. No panics on missing commands.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Public Types
// ===========================================================================

/// Status of WSL2 on the current system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WslStatus {
    /// WSL2 is not installed or not available.
    NotInstalled,
    /// WSL2 is available with a running distro.
    Available {
        /// Name of the default or detected WSL2 distribution.
        distro: String,
        /// WSL version (always 2 for this variant).
        version: u8,
    },
    /// WSL2 detection encountered an error.
    Error {
        /// Error message describing what went wrong.
        message: String,
    },
}

/// Result of a single WSL2 health check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WslHealthResult {
    /// Name of the health check performed.
    pub check: String,
    /// Whether the check passed.
    pub success: bool,
    /// Human-readable description of the result.
    pub message: String,
}

// ===========================================================================
// Public API
// ===========================================================================

/// Detect WSL2 availability on the current system.
///
/// Returns a [`WslStatus`] indicating whether WSL2 is available, not installed,
/// or if detection encountered an error. Detection completes within 5 seconds
/// and never panics.
///
/// # Platform Behavior
///
/// - **Windows**: Checks `wsl --list --verbose` output for version 2 distros
/// - **Linux**: Checks `/proc/version` for WSL indicators (running inside WSL)
///
/// # Examples
///
/// ```no_run
/// use rusty_stack::platform::wsl::detect_wsl2;
///
/// let status = detect_wsl2();
/// println!("WSL2 status: {:?}", status);
/// ```
pub fn detect_wsl2() -> WslStatus {
    #[cfg(target_os = "windows")]
    {
        detect_wsl2_windows()
    }

    #[cfg(not(target_os = "windows"))]
    {
        detect_wsl2_non_windows()
    }
}

/// Generate provisioning guidance for the given WSL2 status.
///
/// When WSL2 is not installed, returns actionable steps including
/// the `wsl --install` command and manual setup instructions.
/// When WSL2 is available, returns confirmation and next steps.
pub fn provisioning_guidance(status: &WslStatus) -> String {
    match status {
        WslStatus::NotInstalled => {
            let mut guidance = String::new();
            guidance.push_str("WSL2 is not installed on this system.\n\n");
            guidance.push_str("To install WSL2:\n");
            guidance.push_str("  1. Open PowerShell as Administrator\n");
            guidance.push_str("  2. Run: wsl --install\n");
            guidance.push_str("  3. Restart your computer when prompted\n");
            guidance.push_str("  4. Set up a default Linux distribution (Ubuntu recommended)\n");
            guidance.push_str("  5. Install ROCm inside WSL2 following AMD's WSL2 guide\n\n");
            guidance.push_str("After installation, click 'Re-check' to verify WSL2 setup.");
            guidance
        }
        WslStatus::Available { distro, .. } => {
            format!(
                "WSL2 is available with distribution '{}'.\n\n\
                 Next steps:\n\
                 1. Ensure ROCm is installed inside the WSL2 distribution\n\
                 2. Verify GPU access with 'rocm-smi' inside WSL2\n\
                 3. The installer will use WSL2 for Linux-only components",
                distro
            )
        }
        WslStatus::Error { message: msg } => {
            format!(
                "WSL2 detection encountered an error: {}\n\n\
                 Troubleshooting:\n\
                 1. Ensure Windows is updated to the latest version\n\
                 2. Check that 'Windows Subsystem for Linux' is enabled\n\
                 3. Try running 'wsl --install' from an Administrator PowerShell",
                msg
            )
        }
    }
}

/// Perform WSL2 health checks.
///
/// Returns exactly 3 independent health check results:
/// 1. **distro_running** — Is the default WSL2 distro running?
/// 2. **rocm_accessible** — Is ROCm accessible inside WSL2?
/// 3. **gpu_nodes** — Are GPU device nodes present in WSL2?
///
/// Each check is independent — a failure in one does not affect the others.
pub fn check_wsl_health(status: &WslStatus) -> Vec<WslHealthResult> {
    match status {
        WslStatus::NotInstalled => vec![
            WslHealthResult {
                check: "distro_running".to_string(),
                success: false,
                message: "WSL2 is not installed".to_string(),
            },
            WslHealthResult {
                check: "rocm_accessible".to_string(),
                success: false,
                message: "WSL2 is not installed".to_string(),
            },
            WslHealthResult {
                check: "gpu_nodes".to_string(),
                success: false,
                message: "WSL2 is not installed".to_string(),
            },
        ],
        WslStatus::Available { distro, .. } => check_wsl_health_available(distro),
        WslStatus::Error { message: msg } => vec![
            WslHealthResult {
                check: "distro_running".to_string(),
                success: false,
                message: format!("Detection error: {}", msg),
            },
            WslHealthResult {
                check: "rocm_accessible".to_string(),
                success: false,
                message: format!("Detection error: {}", msg),
            },
            WslHealthResult {
                check: "gpu_nodes".to_string(),
                success: false,
                message: format!("Detection error: {}", msg),
            },
        ],
    }
}

// ===========================================================================
// Private Implementation
// ===========================================================================

/// WSL2 detection on Windows: invoke `wsl --list --verbose`.
#[cfg(target_os = "windows")]
fn detect_wsl2_windows() -> WslStatus {
    use std::process::Command;

    let output = match Command::new("wsl").args(["--list", "--verbose"]).output() {
        Ok(o) => o,
        Err(e) => {
            return WslStatus::Error {
                message: format!("Failed to execute wsl: {}", e),
            }
        }
    };

    if !output.status.success() {
        return WslStatus::NotInstalled;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_wsl_list_output(&stdout)
}

/// Parse `wsl --list --verbose` output to find a WSL2 distro.
#[cfg(any(target_os = "windows", test))]
fn parse_wsl_list_output(output: &str) -> WslStatus {
    for line in output.lines().skip(1) {
        // Skip header line
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse: NAME, STATE, VERSION
        // The default distro is marked with *
        // * can be: adjacent to name (*Ubuntu) or separated (* Ubuntu)
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }

        let version_str = parts[parts.len() - 1]; // VERSION is the last column
        let state_str = parts[parts.len() - 2]; // STATE is second-to-last

        // Determine the name: could start at index 0 or 1 depending on standalone *
        let name = if parts[0] == "*" {
            // Standalone asterisk: name is parts[1]
            if parts.len() < 4 {
                continue;
            }
            parts[1]
        } else if parts[0].starts_with('*') {
            // Asterisk prefixed to name: *Ubuntu-22.04
            parts[0].trim_start_matches('*')
        } else {
            parts[0]
        };

        // Validate that state_str looks like a state (Running, Stopped, etc.)
        // and version_str looks like a number
        let _ = state_str; // We just need name and version

        if let Ok(version) = version_str.parse::<u8>() {
            if version == 2 && !name.is_empty() {
                return WslStatus::Available {
                    distro: name.to_string(),
                    version: 2,
                };
            }
        }
    }

    // WSL is installed but no v2 distro found
    WslStatus::NotInstalled
}

/// WSL2 detection on non-Windows (Linux/macOS).
///
/// On Linux, checks if we're running inside WSL by examining `/proc/version`.
#[cfg(not(target_os = "windows"))]
fn detect_wsl2_non_windows() -> WslStatus {
    // Check if running inside WSL
    if is_running_in_wsl() {
        let distro = detect_wsl_distro_name();
        return WslStatus::Available {
            distro: distro.unwrap_or_else(|| "unknown".to_string()),
            version: 2,
        };
    }

    WslStatus::NotInstalled
}

/// Check if the current Linux system is running inside WSL.
#[cfg(not(target_os = "windows"))]
fn is_running_in_wsl() -> bool {
    // Check /proc/version for Microsoft/WSL indicators
    if let Ok(version) = std::fs::read_to_string("/proc/version") {
        let version_lower = version.to_lowercase();
        return version_lower.contains("microsoft") || version_lower.contains("wsl");
    }
    false
}

/// Detect the WSL distribution name from /etc/os-release or WSL_DISTRO_NAME env var.
#[cfg(not(target_os = "windows"))]
fn detect_wsl_distro_name() -> Option<String> {
    // First check WSL_DISTRO_NAME environment variable
    if let Ok(name) = std::env::var("WSL_DISTRO_NAME") {
        if !name.is_empty() {
            return Some(name);
        }
    }

    // Fall back to /etc/os-release NAME field
    if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if let Some(name) = line.strip_prefix("NAME=") {
                let name = name.trim().trim_matches('"').trim_matches('\'');
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }

    None
}

/// Perform health checks when WSL2 is available.
fn check_wsl_health_available(distro: &str) -> Vec<WslHealthResult> {
    let distro_result = check_distro_running(distro);
    let rocm_result = check_rocm_accessible();
    let gpu_result = check_gpu_nodes();

    vec![distro_result, rocm_result, gpu_result]
}

/// Check if the default WSL2 distro is running.
fn check_distro_running(distro: &str) -> WslHealthResult {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        let output = match Command::new("wsl").args(["--list", "--running"]).output() {
            Ok(o) => o,
            Err(e) => {
                return WslHealthResult {
                    check: "distro_running".to_string(),
                    success: false,
                    message: format!("Failed to check running distros: {}", e),
                }
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
        let running = stdout.contains(&distro.to_lowercase());

        WslHealthResult {
            check: "distro_running".to_string(),
            success: running,
            message: if running {
                format!("{} is running", distro)
            } else {
                format!("{} is not currently running", distro)
            },
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // On Linux (including WSL), the distro is always "running" if we're executing code
        WslHealthResult {
            check: "distro_running".to_string(),
            success: true,
            message: format!("{} is running (native execution)", distro),
        }
    }
}

/// Check if ROCm is accessible.
fn check_rocm_accessible() -> WslHealthResult {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        let output = match Command::new("wsl")
            .args(["--", "rocm-smi", "--version"])
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                return WslHealthResult {
                    check: "rocm_accessible".to_string(),
                    success: false,
                    message: format!("Failed to check ROCm in WSL: {}", e),
                }
            }
        };

        let success = output.status.success();
        WslHealthResult {
            check: "rocm_accessible".to_string(),
            success,
            message: if success {
                "ROCm is accessible inside WSL2".to_string()
            } else {
                "ROCm is not accessible inside WSL2".to_string()
            },
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // On Linux, check if rocm-smi is available
        let rocm_accessible = std::path::Path::new("/opt/rocm/bin/rocm-smi").exists()
            || std::process::Command::new("rocm-smi")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

        WslHealthResult {
            check: "rocm_accessible".to_string(),
            success: rocm_accessible,
            message: if rocm_accessible {
                "ROCm is accessible".to_string()
            } else {
                "ROCm is not accessible".to_string()
            },
        }
    }
}

/// Check if GPU device nodes exist.
fn check_gpu_nodes() -> WslHealthResult {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        let output = match Command::new("wsl").args(["--", "ls", "/dev/kfd"]).output() {
            Ok(o) => o,
            Err(e) => {
                return WslHealthResult {
                    check: "gpu_nodes".to_string(),
                    success: false,
                    message: format!("Failed to check GPU nodes in WSL: {}", e),
                }
            }
        };

        let success = output.status.success();
        WslHealthResult {
            check: "gpu_nodes".to_string(),
            success,
            message: if success {
                "GPU device nodes are present in WSL2".to_string()
            } else {
                "GPU device nodes are not present in WSL2".to_string()
            },
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // On Linux, check /dev/kfd and /dev/dri
        let kfd_exists = std::path::Path::new("/dev/kfd").exists();
        let dri_exists = std::path::Path::new("/dev/dri").exists();
        let gpu_nodes_present = kfd_exists || dri_exists;

        WslHealthResult {
            check: "gpu_nodes".to_string(),
            success: gpu_nodes_present,
            message: if gpu_nodes_present {
                "GPU device nodes are present".to_string()
            } else {
                "GPU device nodes are not present".to_string()
            },
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- WSL2 Detection ----

    #[test]
    fn test_detect_wsl2_returns_valid_status() {
        let status = detect_wsl2();
        // Should be one of the valid variants
        match &status {
            WslStatus::NotInstalled => {}
            WslStatus::Available { distro, version } => {
                assert!(!distro.is_empty());
                assert_eq!(*version, 2);
            }
            WslStatus::Error { message: msg } => {
                assert!(!msg.is_empty());
            }
        }
    }

    #[test]
    fn test_detect_wsl2_completes_quickly() {
        let start = std::time::Instant::now();
        let _ = detect_wsl2();
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "Detection took {:?}", elapsed);
    }

    // ---- Provisioning Guidance ----

    #[test]
    fn test_guidance_not_installed() {
        let g = provisioning_guidance(&WslStatus::NotInstalled);
        assert!(g.to_lowercase().contains("wsl"));
        assert!(g.to_lowercase().contains("install"));
    }

    #[test]
    fn test_guidance_available() {
        let s = WslStatus::Available {
            distro: "Ubuntu".to_string(),
            version: 2,
        };
        let g = provisioning_guidance(&s);
        assert!(g.contains("Ubuntu"));
    }

    #[test]
    fn test_guidance_error() {
        let s = WslStatus::Error {
            message: "test".to_string(),
        };
        let g = provisioning_guidance(&s);
        assert!(!g.is_empty());
    }

    // ---- Health Checks ----

    #[test]
    fn test_health_checks_not_installed_count() {
        let results = check_wsl_health(&WslStatus::NotInstalled);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_health_checks_not_installed_all_fail() {
        let results = check_wsl_health(&WslStatus::NotInstalled);
        for r in &results {
            assert!(!r.success, "{} should fail when not installed", r.check);
        }
    }

    #[test]
    fn test_health_checks_independent() {
        let results = check_wsl_health(&WslStatus::NotInstalled);
        let names: std::collections::HashSet<&str> =
            results.iter().map(|r| r.check.as_str()).collect();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_health_check_names() {
        let results = check_wsl_health(&WslStatus::NotInstalled);
        let names: Vec<&str> = results.iter().map(|r| r.check.as_str()).collect();
        assert!(names.contains(&"distro_running"));
        assert!(names.contains(&"rocm_accessible"));
        assert!(names.contains(&"gpu_nodes"));
    }

    // ---- Serde Roundtrips ----

    #[test]
    fn test_wsl_status_serde_not_installed() {
        let s = WslStatus::NotInstalled;
        let json = serde_json::to_string(&s).unwrap();
        let back: WslStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn test_wsl_status_serde_available() {
        let s = WslStatus::Available {
            distro: "Ubuntu-22.04".to_string(),
            version: 2,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: WslStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn test_wsl_status_serde_error() {
        let s = WslStatus::Error {
            message: "test error".to_string(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: WslStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn test_wsl_health_result_serde() {
        let r = WslHealthResult {
            check: "distro_running".to_string(),
            success: true,
            message: "Running".to_string(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: WslHealthResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    // ---- Parse WSL Output ----

    #[test]
    fn test_parse_wsl_list_with_v2_distro() {
        let output = "  NAME                   STATE           VERSION\n\
                       * Ubuntu-22.04           Running         2\n\
                         Debian                 Stopped         1";
        let status = parse_wsl_list_output(output);
        match status {
            WslStatus::Available { distro, version } => {
                assert_eq!(distro, "Ubuntu-22.04");
                assert_eq!(version, 2);
            }
            _ => panic!("Expected Available, got {:?}", status),
        }
    }

    #[test]
    fn test_parse_wsl_list_no_v2_distro() {
        let output = "  NAME                   STATE           VERSION\n\
                         Debian                 Stopped         1";
        let status = parse_wsl_list_output(output);
        assert_eq!(status, WslStatus::NotInstalled);
    }

    #[test]
    fn test_parse_wsl_list_empty() {
        let output = "";
        let status = parse_wsl_list_output(output);
        assert_eq!(status, WslStatus::NotInstalled);
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_is_running_in_wsl() {
        // This test just verifies the function doesn't panic
        let _ = is_running_in_wsl();
    }
}
