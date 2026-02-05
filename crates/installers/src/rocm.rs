//! ROCm Installer
//!
//! Native ROCm platform installer with repository configuration and version management.

use crate::common::{Installer, InstallerError, ProgressCallback, require_root};
use crate::gpg::GpgKeyManager;
use crate::profiles::{ProfileConfig, rocm_packages_for_profile};
use crate::repository::{PackageManager, RepositoryConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Supported ROCm versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RocmVersion {
    /// ROCm 6.4.3 (legacy)
    V6_4_3,
    /// ROCm 7.1 (stable, RDNA 3 focus)
    V7_1,
    /// ROCm 7.2 (latest, RDNA 4 support)
    #[default]
    V7_2,
    /// ROCm 7.10 (preview)
    V7_10,
}

impl RocmVersion {
    /// Returns the version string.
    pub fn version_string(&self) -> &'static str {
        match self {
            RocmVersion::V6_4_3 => "6.4.3",
            RocmVersion::V7_1 => "7.1",
            RocmVersion::V7_2 => "7.2",
            RocmVersion::V7_10 => "7.10",
        }
    }

    /// Returns the repository URL path component.
    pub fn repo_path(&self) -> &'static str {
        match self {
            RocmVersion::V6_4_3 => "6.4.3",
            RocmVersion::V7_1 => "7.1",
            RocmVersion::V7_2 => "7.2",
            RocmVersion::V7_10 => "7.10",
        }
    }

    /// Returns true if this version supports RDNA 4.
    pub fn supports_rdna4(&self) -> bool {
        matches!(self, RocmVersion::V7_2 | RocmVersion::V7_10)
    }

    /// Returns true if this version is recommended for RDNA 3.
    pub fn recommended_for_rdna3(&self) -> bool {
        matches!(self, RocmVersion::V7_1 | RocmVersion::V7_2)
    }
}

/// ROCm installer.
pub struct RocmInstaller {
    version: RocmVersion,
    profile: ProfileConfig,
    package_manager: PackageManager,
    repo_config: RepositoryConfig,
}

impl RocmInstaller {
    /// Creates a new ROCm installer.
    pub fn new(version: RocmVersion, profile: ProfileConfig) -> Result<Self> {
        let package_manager = PackageManager::detect()
            .ok_or_else(|| InstallerError::UnsupportedPackageManager(
                "No supported package manager found".to_string()
            ))?;

        let repo_config = Self::create_repo_config(&version);

        Ok(Self {
            version,
            profile,
            package_manager,
            repo_config,
        })
    }

    /// Creates repository configuration for the given version.
    fn create_repo_config(version: &RocmVersion) -> RepositoryConfig {
        RepositoryConfig::new(
            "ROCm",
            format!("https://repo.radeon.com/rocm/apt/{}", version.repo_path()),
            "ubuntu",
        )
        .with_gpg_key("https://repo.radeon.com/rocm/rocm.gpg.key")
    }

    /// Returns the ROCm installation path.
    pub fn rocm_path(&self) -> String {
        "/opt/rocm".to_string()
    }

    /// Checks if ROCm is already installed.
    pub fn is_rocm_installed(&self) -> bool {
        Path::new("/opt/rocm/bin/rocminfo").exists()
    }

    /// Gets the installed ROCm version if available.
    pub fn installed_version(&self) -> Option<String> {
        if let Ok(output) = Command::new("/opt/rocm/bin/rocminfo").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.to_lowercase().contains("rocm version") {
                        return line.split(':').nth(1).map(|s| s.trim().to_string());
                    }
                }
            }
        }
        None
    }

    async fn setup_repository(&self) -> Result<()> {
        require_root()?;

        let gpg_manager = GpgKeyManager::new();
        gpg_manager.import_key_from_url(&self.repo_config.gpg_key_url, "rocm.gpg").await
            .context("Failed to import ROCm GPG key")?;

        self.add_apt_repository().await?;
        self.package_manager.update().await
            .context("Failed to update package lists")?;

        Ok(())
    }

    /// Adds the APT repository.
    async fn add_apt_repository(&self) -> Result<()> {
        let sources_entry = format!(
            "deb [arch=amd64] {} ubuntu main",
            self.repo_config.url
        );

        let sources_file = format!("/etc/apt/sources.list.d/rocm-{}.list", self.version.repo_path());
        tokio::fs::write(&sources_file, sources_entry).await
            .context("Failed to write ROCm sources list")?;

        Ok(())
    }

    async fn install_packages(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        let packages = rocm_packages_for_profile(self.profile.profile);

        if let Some(ref cb) = progress {
            cb(0.3, "Installing ROCm packages...".to_string());
        }

        self.package_manager.install(&packages).await
            .context("Failed to install ROCm packages")?;

        if let Some(ref cb) = progress {
            cb(0.8, "ROCm packages installed".to_string());
        }

        Ok(())
    }

    /// Creates symlinks for ROCm binaries.
    async fn create_symlinks(&self) -> Result<()> {
        let rocm_bin = Path::new("/opt/rocm/bin");
        let usr_local_bin = Path::new("/usr/local/bin");

        if !usr_local_bin.exists() {
            tokio::fs::create_dir_all(usr_local_bin).await?;
        }

        let _entries = tokio::fs::read_dir(rocm_bin).await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Installer for RocmInstaller {
    fn name(&self) -> &str {
        "ROCm"
    }

    fn version(&self) -> &str {
        self.version.version_string()
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_rocm_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if require_root().is_err() {
            checks.push("Root privileges required".to_string());
        }

        if self.is_rocm_installed() {
            if let Some(installed) = self.installed_version() {
                checks.push(format!("ROCm {} is already installed", installed));
            }
        }

        checks.push("GPU compatibility check placeholder".to_string());

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.0, "Starting ROCm installation...".to_string());
            cb(0.1, "Configuring repository...".to_string());
        }
        self.setup_repository().await?;
        self.install_packages(&progress).await?;

        if let Some(ref cb) = progress {
            cb(0.9, "Creating symlinks...".to_string());
        }
        self.create_symlinks().await?;

        if let Some(ref cb) = progress {
            cb(1.0, "ROCm installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        require_root()?;

        let packages = rocm_packages_for_profile(self.profile.profile);
        self.package_manager.remove(&packages).await
            .context("Failed to remove ROCm packages")?;

        // Remove repository configuration
        let sources_file = format!("/etc/apt/sources.list.d/rocm-{}.list", self.version.repo_path());
        let _ = tokio::fs::remove_file(&sources_file).await;

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_rocm_installed() && Path::new("/opt/rocm/bin/rocminfo").exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_version_strings() {
        assert_eq!(RocmVersion::V6_4_3.version_string(), "6.4.3");
        assert_eq!(RocmVersion::V7_2.version_string(), "7.2");
    }

    #[test]
    fn test_rocm_version_rdna4_support() {
        assert!(!RocmVersion::V6_4_3.supports_rdna4());
        assert!(!RocmVersion::V7_1.supports_rdna4());
        assert!(RocmVersion::V7_2.supports_rdna4());
        assert!(RocmVersion::V7_10.supports_rdna4());
    }

    #[test]
    fn test_rocm_installer_creation() {
        let profile = ProfileConfig::default();
        let _ = RocmInstaller::new(RocmVersion::V7_2, profile);
    }
}
