//! ROCm Detection Module
//!
//! Provides functionality for detecting ROCm installation, version information,
//! and confidence scoring for detection results.

use serde::{Deserialize, Serialize};
use std::fs;
use std::process::Command;

/// Represents confidence levels for ROCm detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Confidence {
    /// High confidence - ROCm detected via multiple methods
    High,
    /// Medium confidence - ROCm detected via single reliable method
    Medium,
    /// Low confidence - ROCm possibly detected but unverified
    Low,
    /// No ROCm detected
    #[default]
    None,
}

impl Confidence {
    /// Returns true if confidence is at least medium.
    pub fn is_sufficient(&self) -> bool {
        matches!(self, Confidence::High | Confidence::Medium)
    }

    /// Returns a numeric score for comparison.
    pub fn score(&self) -> u8 {
        match self {
            Confidence::High => 3,
            Confidence::Medium => 2,
            Confidence::Low => 1,
            Confidence::None => 0,
        }
    }
}

/// Represents ROCm installation information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ROCmInfo {
    /// ROCm version string (e.g., "6.4.3")
    pub version: String,
    /// ROCm installation path (e.g., "/opt/rocm")
    pub install_path: String,
    /// Confidence level in the detection
    pub confidence: Confidence,
    /// Detection method used
    pub detection_method: String,
    /// Whether ROCm SMI is available
    pub rocm_smi_available: bool,
    /// Whether rocminfo is available
    pub rocminfo_available: bool,
}

impl ROCmInfo {
    /// Creates a new ROCmInfo with the given version.
    pub fn new(version: String) -> Self {
        Self {
            version,
            install_path: "/opt/rocm".to_string(),
            confidence: Confidence::Medium,
            detection_method: "unknown".to_string(),
            rocm_smi_available: false,
            rocminfo_available: false,
        }
    }

    /// Sets the installation path.
    pub fn with_path(mut self, path: String) -> Self {
        self.install_path = path;
        self
    }

    /// Sets the confidence level.
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Sets the detection method.
    pub fn with_method(mut self, method: String) -> Self {
        self.detection_method = method;
        self
    }

    /// Marks ROCm SMI as available.
    pub fn with_rocm_smi(mut self) -> Self {
        self.rocm_smi_available = true;
        self
    }

    /// Marks rocminfo as available.
    pub fn with_rocminfo(mut self) -> Self {
        self.rocminfo_available = true;
        self
    }

    /// Returns true if ROCm is properly installed and usable.
    pub fn is_usable(&self) -> bool {
        self.confidence.is_sufficient() && !self.version.is_empty() && self.version != "unknown"
    }

    /// Returns the major version number.
    pub fn major_version(&self) -> Option<u32> {
        self.version.split('.').next()?.parse().ok()
    }

    /// Returns the minor version number.
    pub fn minor_version(&self) -> Option<u32> {
        self.version.split('.').nth(1)?.parse().ok()
    }
}

impl Default for ROCmInfo {
    fn default() -> Self {
        Self {
            version: "unknown".to_string(),
            install_path: "/opt/rocm".to_string(),
            confidence: Confidence::None,
            detection_method: "none".to_string(),
            rocm_smi_available: false,
            rocminfo_available: false,
        }
    }
}

/// Handles ROCm detection using multiple methods.
pub struct ROCmDetection;

impl ROCmDetection {
    /// Creates a new ROCmDetection instance.
    pub fn new() -> Self {
        Self
    }

    /// Detects ROCm installation using all available methods.
    ///
    /// Returns ROCmInfo with the highest confidence detection.
    pub fn detect(&self) -> ROCmInfo {
        let mut best_result = ROCmInfo::default();

        // Try multiple detection methods in order of reliability
        let methods: Vec<fn() -> Option<ROCmInfo>> = vec![
            Self::detect_from_version_file,
            Self::detect_from_rocminfo,
            Self::detect_from_rocm_smi,
            Self::detect_from_path,
        ];

        for method in methods {
            if let Some(info) = method() {
                if info.confidence.score() > best_result.confidence.score() {
                    best_result = info;
                }
            }
        }

        // Check tool availability
        best_result.rocm_smi_available = Self::is_rocm_smi_available();
        best_result.rocminfo_available = Self::is_rocminfo_available();

        best_result
    }

    /// Detects ROCm from the version file at /opt/rocm/.info/version
    fn detect_from_version_file() -> Option<ROCmInfo> {
        let version_file = "/opt/rocm/.info/version";

        if let Ok(contents) = fs::read_to_string(version_file) {
            let version = contents.trim().to_string();
            if !version.is_empty() && version != "unknown" {
                return Some(
                    ROCmInfo::new(version)
                        .with_path("/opt/rocm".to_string())
                        .with_confidence(Confidence::High)
                        .with_method("version_file".to_string()),
                );
            }
        }

        None
    }

    /// Detects ROCm from rocminfo output.
    fn detect_from_rocminfo() -> Option<ROCmInfo> {
        if let Ok(output) = Command::new("rocminfo").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Look for ROCm version in output
                for line in stdout.lines() {
                    let lower = line.to_lowercase();
                    if lower.contains("rocm version") {
                        if let Some(version) = line.split(':').nth(1) {
                            let version = version.trim().to_string();
                            if !version.is_empty() {
                                return Some(
                                    ROCmInfo::new(version)
                                        .with_confidence(Confidence::Medium)
                                        .with_method("rocminfo".to_string())
                                        .with_rocminfo(),
                                );
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Detects ROCm from rocm-smi output.
    fn detect_from_rocm_smi() -> Option<ROCmInfo> {
        if let Ok(output) = Command::new("rocm-smi").arg("--showdriverversion").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse driver version which correlates to ROCm version
                for line in stdout.lines() {
                    if line.contains("Driver version") {
                        // Extract version info if present
                        return Some(
                            ROCmInfo::new("detected".to_string())
                                .with_confidence(Confidence::Low)
                                .with_method("rocm_smi".to_string())
                                .with_rocm_smi(),
                        );
                    }
                }
            }
        }

        None
    }

    /// Detects ROCm from path existence.
    fn detect_from_path() -> Option<ROCmInfo> {
        let rocm_path = std::path::Path::new("/opt/rocm");

        if rocm_path.exists() && rocm_path.is_dir() {
            // Check for key directories
            let has_bin = rocm_path.join("bin").exists();
            let has_lib = rocm_path.join("lib").exists();

            if has_bin && has_lib {
                return Some(
                    ROCmInfo::new("unknown".to_string())
                        .with_path("/opt/rocm".to_string())
                        .with_confidence(Confidence::Low)
                        .with_method("path_check".to_string()),
                );
            }
        }

        None
    }

    /// Checks if rocm-smi command is available.
    fn is_rocm_smi_available() -> bool {
        Command::new("which")
            .arg("rocm-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if rocminfo command is available.
    fn is_rocminfo_available() -> bool {
        Command::new("which")
            .arg("rocminfo")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

impl Default for ROCmDetection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_info_builder() {
        let info = ROCmInfo::new("6.4.3".to_string())
            .with_path("/opt/rocm".to_string())
            .with_confidence(Confidence::High)
            .with_method("test".to_string())
            .with_rocm_smi()
            .with_rocminfo();

        assert_eq!(info.version, "6.4.3");
        assert_eq!(info.install_path, "/opt/rocm");
        assert_eq!(info.confidence, Confidence::High);
        assert!(info.rocm_smi_available);
        assert!(info.rocminfo_available);
        assert!(info.is_usable());
    }

    #[test]
    fn test_confidence_score() {
        assert_eq!(Confidence::High.score(), 3);
        assert_eq!(Confidence::Medium.score(), 2);
        assert_eq!(Confidence::Low.score(), 1);
        assert_eq!(Confidence::None.score(), 0);
    }

    #[test]
    fn test_confidence_is_sufficient() {
        assert!(Confidence::High.is_sufficient());
        assert!(Confidence::Medium.is_sufficient());
        assert!(!Confidence::Low.is_sufficient());
        assert!(!Confidence::None.is_sufficient());
    }

    #[test]
    fn test_version_parsing() {
        let info = ROCmInfo::new("6.4.3".to_string());
        assert_eq!(info.major_version(), Some(6));
        assert_eq!(info.minor_version(), Some(4));

        let info2 = ROCmInfo::new("7.2".to_string());
        assert_eq!(info2.major_version(), Some(7));
        assert_eq!(info2.minor_version(), Some(2));
    }

    #[test]
    fn test_default_not_usable() {
        let info = ROCmInfo::default();
        assert!(!info.is_usable());
        assert_eq!(info.confidence, Confidence::None);
    }
}
