//! ROCm environment facade — thin wrapper over `platform::linux` for ROCm paths/versions.
//!
//! This module delegates all ROCm detection to `platform::linux` and does NOT
//! duplicate any detection logic. It provides a convenient struct-based API
//! for installer use.
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-007**: ROCm environment delegates to platform modules
//! - **VAL-INFRA-008**: ROCm version detection returns semantic version
//! - **VAL-INFRA-019**: No code duplication with platform modules

use crate::platform::linux;
use std::path::PathBuf;

// ===========================================================================
// RocmEnv
// ===========================================================================

/// ROCm environment information, lazily detected from the system.
///
/// All detection delegates to `platform::linux`. This struct provides
/// a convenient cached view of the ROCm environment for installer use.
///
/// # Validation
///
/// - **VAL-INFRA-007**: ROCm env uses platform::linux (no duplicate detection)
/// - **VAL-INFRA-008**: ROCm version detection returns semantic version
#[derive(Debug, Clone)]
pub struct RocmEnv {
    /// Path to the ROCm installation (e.g., `/opt/rocm`).
    path: Option<PathBuf>,
    /// ROCm version string (e.g., "6.4.3").
    version: String,
}

impl RocmEnv {
    /// Detect the ROCm environment from the system.
    ///
    /// This calls `platform::linux::detect_rocm_path()` and
    /// `platform::linux::get_rocm_version()` — no duplicate detection.
    pub fn detect() -> Self {
        let path = linux::detect_rocm_path();
        let version = linux::get_rocm_version();
        Self { path, version }
    }

    /// Create a RocmEnv with known values (useful for testing).
    pub fn from_known(path: Option<PathBuf>, version: String) -> Self {
        Self { path, version }
    }

    /// Create a RocmEnv representing no ROCm installation.
    pub fn none() -> Self {
        Self {
            path: None,
            version: String::new(),
        }
    }

    /// Get the ROCm installation path.
    pub fn path(&self) -> Option<&PathBuf> {
        self.path.as_ref()
    }

    /// Get the ROCm version string.
    ///
    /// Returns an empty string if ROCm is not installed or version
    /// cannot be determined.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Check if a ROCm installation was found.
    pub fn is_detected(&self) -> bool {
        self.path.is_some()
    }

    /// Get the ROCm bin directory path.
    ///
    /// Returns `None` if ROCm is not detected or the bin directory
    /// does not exist.
    pub fn bin_path(&self) -> Option<PathBuf> {
        self.path.as_ref().map(|p| p.join("bin")).filter(|p| p.is_dir())
    }

    /// Get the ROCm lib directory path.
    ///
    /// Checks both `lib` and `lib64` subdirectories.
    pub fn lib_path(&self) -> Option<PathBuf> {
        self.path.as_ref().and_then(|p| {
            let lib = p.join("lib");
            if lib.is_dir() {
                return Some(lib);
            }
            let lib64 = p.join("lib64");
            if lib64.is_dir() {
                return Some(lib64);
            }
            None
        })
    }

    /// Get the ROCm include directory path.
    pub fn include_path(&self) -> Option<PathBuf> {
        self.path
            .as_ref()
            .map(|p| p.join("include"))
            .filter(|p| p.is_dir())
    }

    /// Check if a specific ROCm tool exists.
    ///
    /// Checks both in the ROCm bin directory and in the system PATH.
    pub fn tool_exists(&self, tool_name: &str) -> bool {
        // Check in ROCm bin directory
        if let Some(bin) = self.bin_path() {
            let tool_path = bin.join(tool_name);
            if tool_path.exists() {
                return true;
            }
        }

        // Check in system PATH
        which_exists(tool_name)
    }

    /// Get the full path to a ROCm tool.
    ///
    /// Returns `None` if the tool is not found.
    pub fn tool_path(&self, tool_name: &str) -> Option<PathBuf> {
        // Check in ROCm bin directory first
        if let Some(bin) = self.bin_path() {
            let tool_path = bin.join(tool_name);
            if tool_path.exists() {
                return Some(tool_path);
            }
        }

        // Check in system PATH
        if let Ok(output) = std::process::Command::new("which")
            .arg(tool_name)
            .output()
        {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path_str.is_empty() {
                    return Some(PathBuf::from(path_str));
                }
            }
        }

        None
    }

    /// Get the ROCm major.minor version (e.g., "7.2" from "7.2.0").
    ///
    /// Returns an empty string if version cannot be parsed.
    pub fn version_major_minor(&self) -> String {
        extract_major_minor(&self.version)
    }

    /// Get the ROCm series candidates for wheel index probing.
    ///
    /// Returns the detected version first, then common fallback versions.
    pub fn series_candidates(&self) -> Vec<String> {
        let detected = self.version_major_minor();
        let mut candidates = Vec::new();
        if !detected.is_empty() {
            candidates.push(detected);
        }
        for fallback in &["7.2", "7.1", "7.0", "6.4", "6.3", "6.2", "6.1", "6.0", "5.7"] {
            let fb = fallback.to_string();
            if !candidates.contains(&fb) {
                candidates.push(fb);
            }
        }
        candidates
    }

    /// Get the PyTorch index URL for a given channel and ROCm series.
    pub fn torch_index_url(&self, channel: &str, series: &str) -> String {
        match channel {
            "nightly" => format!("https://download.pytorch.org/whl/nightly/rocm{series}"),
            _ => format!("https://download.pytorch.org/whl/rocm{series}"),
        }
    }

    /// Get the Radeon PyTorch index URL for a given ROCm series.
    pub fn torch_radeon_index_url(&self, series: &str) -> String {
        format!("https://repo.radeon.com/rocm/manylinux/rocm-rel-{series}/")
    }
}

impl Default for RocmEnv {
    fn default() -> Self {
        Self::detect()
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Extract major.minor from a version string.
fn extract_major_minor(version: &str) -> String {
    let version = version.trim();
    if version.is_empty() {
        return String::new();
    }

    // Find the first N.N pattern
    let chars: Vec<char> = version.chars().collect();
    let mut result = String::new();
    let mut dot_count = 0;

    for &ch in &chars {
        if ch == '.' {
            dot_count += 1;
            if dot_count > 1 {
                break;
            }
            result.push(ch);
        } else if ch.is_ascii_digit() {
            result.push(ch);
        } else {
            break;
        }
    }

    // Validate it looks like N.N
    if result.contains('.') && result.len() >= 3 {
        result
    } else {
        String::new()
    }
}

/// Check if a command exists in PATH.
fn which_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_major_minor ---

    #[test]
    fn test_extract_major_minor_standard() {
        assert_eq!(extract_major_minor("7.2.0"), "7.2");
        assert_eq!(extract_major_minor("6.4.3"), "6.4");
    }

    #[test]
    fn test_extract_major_minor_two_part() {
        assert_eq!(extract_major_minor("7.2"), "7.2");
    }

    #[test]
    fn test_extract_major_minor_empty() {
        assert_eq!(extract_major_minor(""), "");
    }

    #[test]
    fn test_extract_major_minor_invalid() {
        assert_eq!(extract_major_minor("abc"), "");
    }

    // --- RocmEnv (VAL-INFRA-007, VAL-INFRA-008) ---

    #[test]
    fn test_rocm_env_detect_no_panic() {
        // Should not panic even if ROCm is not installed
        let env = RocmEnv::detect();
        // version may be empty if no ROCm
        let _ = env.version();
    }

    #[test]
    fn test_rocm_env_from_known() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        assert!(env.is_detected());
        assert_eq!(env.version(), "7.2.0");
        assert_eq!(env.path(), Some(&PathBuf::from("/opt/rocm")));
    }

    #[test]
    fn test_rocm_env_none() {
        let env = RocmEnv::none();
        assert!(!env.is_detected());
        assert!(env.version().is_empty());
        assert!(env.path().is_none());
    }

    #[test]
    fn test_rocm_env_version_major_minor() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        assert_eq!(env.version_major_minor(), "7.2");
    }

    #[test]
    fn test_rocm_env_version_major_minor_empty() {
        let env = RocmEnv::none();
        assert!(env.version_major_minor().is_empty());
    }

    #[test]
    fn test_rocm_env_series_candidates() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let candidates = env.series_candidates();
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], "7.2");
        // Should contain fallbacks
        assert!(candidates.contains(&"7.1".to_string()));
        assert!(candidates.contains(&"6.4".to_string()));
    }

    #[test]
    fn test_rocm_env_series_candidates_no_duplicates() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let candidates = env.series_candidates();
        let mut seen = std::collections::HashSet::new();
        for c in &candidates {
            assert!(seen.insert(c.clone()), "Duplicate series candidate: {c}");
        }
    }

    #[test]
    fn test_rocm_env_torch_index_url() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        assert_eq!(
            env.torch_index_url("stable", "7.2"),
            "https://download.pytorch.org/whl/rocm7.2"
        );
        assert_eq!(
            env.torch_index_url("nightly", "7.2"),
            "https://download.pytorch.org/whl/nightly/rocm7.2"
        );
    }

    #[test]
    fn test_rocm_env_torch_radeon_index_url() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        assert_eq!(
            env.torch_radeon_index_url("7.2"),
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"
        );
    }

    #[test]
    fn test_rocm_env_default() {
        let env = RocmEnv::default();
        // Should not panic
        let _ = env.version();
    }

    #[test]
    fn test_rocm_env_bin_path_with_mock() {
        // Use /usr as a directory that has bin/
        let env = RocmEnv::from_known(Some(PathBuf::from("/usr")), "1.0.0".to_string());
        assert!(env.bin_path().is_some());
    }

    #[test]
    fn test_rocm_env_lib_path_with_mock() {
        // Use /usr as a directory that has lib/
        let env = RocmEnv::from_known(Some(PathBuf::from("/usr")), "1.0.0".to_string());
        assert!(env.lib_path().is_some());
    }

    #[test]
    fn test_rocm_env_include_path_with_mock() {
        // Use /usr as a directory that has include/
        let env = RocmEnv::from_known(Some(PathBuf::from("/usr")), "1.0.0".to_string());
        assert!(env.include_path().is_some());
    }

    #[test]
    fn test_rocm_env_bin_path_none() {
        let env = RocmEnv::none();
        assert!(env.bin_path().is_none());
    }

    #[test]
    fn test_rocm_env_tool_exists_bash() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/usr")), "1.0.0".to_string());
        // bash should exist in /usr/bin
        assert!(env.tool_exists("bash"));
    }

    #[test]
    fn test_rocm_env_tool_exists_nonexistent() {
        let env = RocmEnv::none();
        assert!(!env.tool_exists("nonexistent_tool_xyz_12345"));
    }

    #[test]
    fn test_rocm_env_tool_path_bash() {
        let env = RocmEnv::from_known(Some(PathBuf::from("/usr")), "1.0.0".to_string());
        assert!(env.tool_path("bash").is_some());
    }

    #[test]
    fn test_rocm_env_tool_path_nonexistent() {
        let env = RocmEnv::none();
        assert!(env.tool_path("nonexistent_tool_xyz_12345").is_none());
    }

    // --- VAL-INFRA-007: Delegation verification ---

    #[test]
    fn test_rocm_env_delegates_to_platform_linux() {
        // Verify that detect() returns the same result as calling platform::linux directly
        let env = RocmEnv::detect();
        let platform_path = linux::detect_rocm_path();
        let platform_version = linux::get_rocm_version();

        assert_eq!(env.path(), platform_path.as_ref());
        assert_eq!(env.version(), platform_version);
    }

    // --- VAL-INFRA-008: Version returns semantic version or empty ---

    #[test]
    fn test_rocm_version_no_panic_on_malformed() {
        let env = RocmEnv::from_known(None, "not-a-version".to_string());
        // Should not panic
        let mm = env.version_major_minor();
        assert!(mm.is_empty()); // malformed version returns empty
    }

    #[test]
    fn test_rocm_version_empty_on_not_installed() {
        let env = RocmEnv::none();
        assert!(env.version().is_empty());
        assert!(env.version_major_minor().is_empty());
    }
}
