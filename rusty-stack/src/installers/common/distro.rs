//! Thin facade over `platform::detection` for installer use.
//!
//! This module delegates all distro detection to `platform::detection::detect_distribution()`
//! and does NOT duplicate `/etc/os-release` parsing.
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-002**: Distro detection delegates to platform::detection
//! - **VAL-INFRA-003**: Distro family classification correct

use crate::platform::detection::{detect_distribution, DistroFamily, DistroInfo, PackageManager};

/// Thin facade that wraps `platform::detection` for installer use.
///
/// Provides convenience methods for the installer layer without duplicating
/// any detection logic.
#[derive(Debug, Clone)]
pub struct DistroFacade {
    info: DistroInfo,
}

impl DistroFacade {
    /// Detect the current distribution by delegating to `platform::detection`.
    ///
    /// This is the primary constructor and performs no duplicate parsing.
    pub fn detect() -> Self {
        Self {
            info: detect_distribution(),
        }
    }

    /// Create a facade from a pre-built `DistroInfo` (useful for testing).
    pub fn from_info(info: DistroInfo) -> Self {
        Self { info }
    }

    /// Get the distribution ID (e.g., "ubuntu", "arch").
    pub fn id(&self) -> &str {
        &self.info.id
    }

    /// Get the full distribution name (e.g., "Ubuntu 24.04 LTS").
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the distribution version (e.g., "24.04", "rolling").
    pub fn version(&self) -> &str {
        &self.info.version
    }

    /// Get the distribution codename (e.g., "noble").
    pub fn codename(&self) -> &str {
        &self.info.codename
    }

    /// Get the distribution family classification.
    pub fn family(&self) -> DistroFamily {
        self.info.family
    }

    /// Get the detected package manager.
    pub fn package_manager(&self) -> PackageManager {
        self.info.pkg_manager
    }

    /// Get the ID_LIKE field from os-release.
    pub fn id_like(&self) -> &str {
        &self.info.id_like
    }

    /// Get the underlying `DistroInfo`.
    pub fn info(&self) -> &DistroInfo {
        &self.info
    }

    /// Check if this is a Debian-family distro.
    pub fn is_debian_family(&self) -> bool {
        self.info.family == DistroFamily::Debian
    }

    /// Check if this is an Arch-family distro.
    pub fn is_arch_family(&self) -> bool {
        self.info.family == DistroFamily::Arch
    }

    /// Check if this is a RHEL-family distro.
    pub fn is_rhel_family(&self) -> bool {
        self.info.family == DistroFamily::Rhel
    }

    /// Check if this is a SUSE-family distro.
    pub fn is_suse_family(&self) -> bool {
        self.info.family == DistroFamily::Suse
    }

    /// Check if the distro uses apt.
    pub fn uses_apt(&self) -> bool {
        self.info.pkg_manager == PackageManager::Apt
    }

    /// Check if the distro uses pacman.
    pub fn uses_pacman(&self) -> bool {
        self.info.pkg_manager == PackageManager::Pacman
    }

    /// Check if the distro uses dnf.
    pub fn uses_dnf(&self) -> bool {
        self.info.pkg_manager == PackageManager::Dnf
    }

    /// Check if the distro uses yum.
    pub fn uses_yum(&self) -> bool {
        self.info.pkg_manager == PackageManager::Yum
    }

    /// Check if the distro uses zypper.
    pub fn uses_zypper(&self) -> bool {
        self.info.pkg_manager == PackageManager::Zypper
    }
}

impl Default for DistroFacade {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VAL-INFRA-002: Distro detection delegates to platform::detection
    #[test]
    fn test_distro_facade_delegates_to_platform() {
        let facade = DistroFacade::detect();
        // Should return non-empty id on a real system
        assert!(
            !facade.id().is_empty(),
            "DistroFacade should delegate to platform::detection"
        );
    }

    /// VAL-INFRA-002: No duplicate /etc/os-release parsing
    #[test]
    fn test_distro_facade_from_info() {
        let info = DistroInfo {
            id: "ubuntu".to_string(),
            name: "Ubuntu 24.04".to_string(),
            version: "24.04".to_string(),
            codename: "noble".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            id_like: "debian".to_string(),
        };
        let facade = DistroFacade::from_info(info.clone());
        assert_eq!(facade.id(), "ubuntu");
        assert_eq!(facade.family(), DistroFamily::Debian);
        assert_eq!(facade.package_manager(), PackageManager::Apt);
        assert_eq!(facade.info(), &info);
    }

    /// VAL-INFRA-003: Distro family classification correct
    #[test]
    fn test_family_classification_debian() {
        let info = DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        };
        let facade = DistroFacade::from_info(info);
        assert!(facade.is_debian_family());
        assert!(!facade.is_arch_family());
        assert!(facade.uses_apt());
    }

    #[test]
    fn test_family_classification_arch() {
        let info = DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        };
        let facade = DistroFacade::from_info(info);
        assert!(facade.is_arch_family());
        assert!(!facade.is_debian_family());
        assert!(facade.uses_pacman());
    }

    #[test]
    fn test_family_classification_rhel() {
        let info = DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        };
        let facade = DistroFacade::from_info(info);
        assert!(facade.is_rhel_family());
        assert!(facade.uses_dnf());
    }

    #[test]
    fn test_family_classification_suse() {
        let info = DistroInfo {
            id: "opensuse-leap".to_string(),
            family: DistroFamily::Suse,
            pkg_manager: PackageManager::Zypper,
            ..Default::default()
        };
        let facade = DistroFacade::from_info(info);
        assert!(facade.is_suse_family());
        assert!(facade.uses_zypper());
    }

    #[test]
    fn test_family_classification_yum() {
        let info = DistroInfo {
            id: "centos".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Yum,
            ..Default::default()
        };
        let facade = DistroFacade::from_info(info);
        assert!(facade.is_rhel_family());
        assert!(facade.uses_yum());
    }

    #[test]
    fn test_default_impl() {
        let facade = DistroFacade::default();
        assert!(!facade.id().is_empty());
    }

    #[test]
    fn test_accessor_methods() {
        let info = DistroInfo {
            id: "ubuntu".to_string(),
            name: "Ubuntu 24.04 LTS".to_string(),
            version: "24.04".to_string(),
            codename: "noble".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            id_like: "debian".to_string(),
        };
        let facade = DistroFacade::from_info(info);
        assert_eq!(facade.name(), "Ubuntu 24.04 LTS");
        assert_eq!(facade.version(), "24.04");
        assert_eq!(facade.codename(), "noble");
        assert_eq!(facade.id_like(), "debian");
    }
}
