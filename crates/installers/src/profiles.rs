//! Installation Profiles
//!
//! Defines installation profiles (minimal, standard, full) for ROCm and other components.

use serde::{Deserialize, Serialize};

/// Installation profile types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InstallProfile {
    /// Minimal installation - core libraries only
    Minimal,
    /// Standard installation - recommended for most users
    #[default]
    Standard,
    /// Full installation - all components including dev tools
    Full,
}

impl InstallProfile {
    /// Returns the profile name.
    pub fn name(&self) -> &'static str {
        match self {
            InstallProfile::Minimal => "minimal",
            InstallProfile::Standard => "standard",
            InstallProfile::Full => "full",
        }
    }

    /// Returns a description of the profile.
    pub fn description(&self) -> &'static str {
        match self {
            InstallProfile::Minimal => "Core libraries only (OpenCL, clang, device libs)",
            InstallProfile::Standard => "Recommended setup with SMI, rocminfo, cmake",
            InstallProfile::Full => "Complete installation with all dev tools and samples",
        }
    }

    /// Returns estimated installation size in GB.
    pub fn estimated_size_gb(&self) -> f32 {
        match self {
            InstallProfile::Minimal => 2.0,
            InstallProfile::Standard => 8.0,
            InstallProfile::Full => 20.0,
        }
    }

    /// Returns estimated installation time in minutes.
    pub fn estimated_time_minutes(&self) -> u32 {
        match self {
            InstallProfile::Minimal => 5,
            InstallProfile::Standard => 15,
            InstallProfile::Full => 45,
        }
    }
}

/// Configuration for installation profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Selected installation profile
    pub profile: InstallProfile,
    /// Additional packages to include
    pub extra_packages: Vec<String>,
    /// Packages to exclude
    pub exclude_packages: Vec<String>,
    /// Whether to include documentation
    pub include_docs: bool,
    /// Whether to include samples
    pub include_samples: bool,
}

impl ProfileConfig {
    /// Creates a new profile configuration.
    pub fn new(profile: InstallProfile) -> Self {
        Self {
            profile,
            extra_packages: Vec::new(),
            exclude_packages: Vec::new(),
            include_docs: matches!(profile, InstallProfile::Full),
            include_samples: matches!(profile, InstallProfile::Full),
        }
    }

    /// Adds an extra package.
    pub fn with_extra_package(mut self, package: impl Into<String>) -> Self {
        self.extra_packages.push(package.into());
        self
    }

    /// Excludes a package.
    pub fn exclude_package(mut self, package: impl Into<String>) -> Self {
        self.exclude_packages.push(package.into());
        self
    }

    /// Sets documentation inclusion.
    pub fn with_docs(mut self, include: bool) -> Self {
        self.include_docs = include;
        self
    }

    /// Sets samples inclusion.
    pub fn with_samples(mut self, include: bool) -> Self {
        self.include_samples = include;
        self
    }
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self::new(InstallProfile::Standard)
    }
}

/// Returns ROCm packages for the given profile.
pub fn rocm_packages_for_profile(profile: InstallProfile) -> Vec<String> {
    let mut packages = vec![
        "rocm-opencl".to_string(),
        "rocm-clang".to_string(),
        "rocm-device-libs".to_string(),
    ];

    match profile {
        InstallProfile::Minimal => {
            // Core packages only
        }
        InstallProfile::Standard => {
            packages.extend(vec![
                "rocm-smi-lib".to_string(),
                "rocminfo".to_string(),
                "rocm-cmake".to_string(),
                "rocm-utils".to_string(),
                "hip-runtime-amd".to_string(),
                "hip-dev".to_string(),
            ]);
        }
        InstallProfile::Full => {
            packages.extend(vec![
                "rocm-smi-lib".to_string(),
                "rocminfo".to_string(),
                "rocm-cmake".to_string(),
                "rocm-utils".to_string(),
                "hip-runtime-amd".to_string(),
                "hip-dev".to_string(),
                "rocm-gdb".to_string(),
                "rocm-debug-agent".to_string(),
                "rocm-bandwidth-test".to_string(),
                "rocprofiler".to_string(),
                "roctracer".to_string(),
                "miopen-hip".to_string(),
                "miopen-hip-dev".to_string(),
                "rccl".to_string(),
                "rccl-dev".to_string(),
                "rocblas".to_string(),
                "rocblas-dev".to_string(),
                "rocfft".to_string(),
                "rocfft-dev".to_string(),
                "rocrand".to_string(),
                "rocrand-dev".to_string(),
                "rocsparse".to_string(),
                "rocsparse-dev".to_string(),
            ]);
        }
    }

    packages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_profile_name() {
        assert_eq!(InstallProfile::Minimal.name(), "minimal");
        assert_eq!(InstallProfile::Standard.name(), "standard");
        assert_eq!(InstallProfile::Full.name(), "full");
    }

    #[test]
    fn test_install_profile_estimates() {
        assert!(
            InstallProfile::Minimal.estimated_size_gb()
                < InstallProfile::Standard.estimated_size_gb()
        );
        assert!(
            InstallProfile::Standard.estimated_size_gb() < InstallProfile::Full.estimated_size_gb()
        );
    }

    #[test]
    fn test_profile_config_builder() {
        let config = ProfileConfig::new(InstallProfile::Standard)
            .with_extra_package("custom-package")
            .exclude_package("unwanted")
            .with_docs(true);

        assert_eq!(config.profile, InstallProfile::Standard);
        assert!(config
            .extra_packages
            .contains(&"custom-package".to_string()));
        assert!(config.exclude_packages.contains(&"unwanted".to_string()));
        assert!(config.include_docs);
    }

    #[test]
    fn test_rocm_packages_minimal() {
        let packages = rocm_packages_for_profile(InstallProfile::Minimal);
        assert!(packages.contains(&"rocm-opencl".to_string()));
        assert!(!packages.contains(&"rocm-smi-lib".to_string()));
    }

    #[test]
    fn test_rocm_packages_standard() {
        let packages = rocm_packages_for_profile(InstallProfile::Standard);
        assert!(packages.contains(&"rocm-opencl".to_string()));
        assert!(packages.contains(&"rocm-smi-lib".to_string()));
        assert!(!packages.contains(&"rocm-gdb".to_string()));
    }

    #[test]
    fn test_rocm_packages_full() {
        let packages = rocm_packages_for_profile(InstallProfile::Full);
        assert!(packages.contains(&"rocm-opencl".to_string()));
        assert!(packages.contains(&"rocm-smi-lib".to_string()));
        assert!(packages.contains(&"rocm-gdb".to_string()));
        assert!(packages.contains(&"miopen-hip".to_string()));
    }
}
