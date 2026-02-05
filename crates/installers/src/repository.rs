//! Package Manager Repository Handling
//!
//! Supports apt, dnf, pacman, and zypper package managers.

use crate::common::{require_root, InstallerError};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;

/// Supported package managers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageManager {
    /// APT (Debian, Ubuntu)
    Apt,
    /// DNF (Fedora, RHEL, CentOS)
    Dnf,
    /// Pacman (Arch Linux)
    Pacman,
    /// Zypper (openSUSE)
    Zypper,
}

impl PackageManager {
    /// Detects the system's package manager.
    pub fn detect() -> Option<Self> {
        if Self::command_exists("apt") {
            Some(PackageManager::Apt)
        } else if Self::command_exists("dnf") {
            Some(PackageManager::Dnf)
        } else if Self::command_exists("pacman") {
            Some(PackageManager::Pacman)
        } else if Self::command_exists("zypper") {
            Some(PackageManager::Zypper)
        } else {
            None
        }
    }

    /// Returns the package manager name.
    pub fn name(&self) -> &'static str {
        match self {
            PackageManager::Apt => "apt",
            PackageManager::Dnf => "dnf",
            PackageManager::Pacman => "pacman",
            PackageManager::Zypper => "zypper",
        }
    }

    /// Updates package lists.
    pub async fn update(&self) -> Result<()> {
        require_root()?;

        let args = match self {
            PackageManager::Apt => ["update"],
            PackageManager::Dnf => ["check-update"],
            PackageManager::Pacman => ["-Sy"],
            PackageManager::Zypper => ["refresh"],
        };

        let output = Command::new(self.name())
            .args(args)
            .output()
            .context("Failed to update package lists")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::RepositoryConfigFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    /// Installs packages.
    pub async fn install(&self, packages: &[String]) -> Result<()> {
        require_root()?;

        if packages.is_empty() {
            return Ok(());
        }

        let mut cmd = Command::new(self.name());

        match self {
            PackageManager::Apt => {
                cmd.arg("install").arg("-y");
            }
            PackageManager::Dnf => {
                cmd.arg("install").arg("-y");
            }
            PackageManager::Pacman => {
                cmd.arg("-S").arg("--noconfirm");
            }
            PackageManager::Zypper => {
                cmd.arg("install").arg("-y");
            }
        }

        cmd.args(packages);

        let output = cmd.output().context("Failed to install packages")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    /// Removes packages.
    pub async fn remove(&self, packages: &[String]) -> Result<()> {
        require_root()?;

        if packages.is_empty() {
            return Ok(());
        }

        let mut cmd = Command::new(self.name());

        match self {
            PackageManager::Apt => {
                cmd.arg("remove").arg("-y");
            }
            PackageManager::Dnf => {
                cmd.arg("remove").arg("-y");
            }
            PackageManager::Pacman => {
                cmd.arg("-R").arg("--noconfirm");
            }
            PackageManager::Zypper => {
                cmd.arg("remove").arg("-y");
            }
        }

        cmd.args(packages);

        let output = cmd.output().context("Failed to remove packages")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    /// Checks if a command exists in PATH.
    fn command_exists(cmd: &str) -> bool {
        Command::new("which").arg(cmd).output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// Repository configuration for ROCm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    /// Repository name
    pub name: String,
    /// Repository URL
    pub url: String,
    /// Distribution codename
    pub codename: String,
    /// Component (e.g., "main", "universe")
    pub component: String,
    /// GPG key URL
    pub gpg_key_url: String,
    /// Whether to use HTTPS
    pub use_https: bool,
}

impl RepositoryConfig {
    /// Creates a new repository configuration.
    pub fn new(
        name: impl Into<String>,
        url: impl Into<String>,
        codename: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            codename: codename.into(),
            component: "main".to_string(),
            gpg_key_url: String::new(),
            use_https: true,
        }
    }

    /// Sets the component.
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = component.into();
        self
    }

    /// Sets the GPG key URL.
    pub fn with_gpg_key(mut self, url: impl Into<String>) -> Self {
        self.gpg_key_url = url.into();
        self
    }

    /// Sets whether to use HTTPS.
    pub fn with_https(mut self, use_https: bool) -> Self {
        self.use_https = use_https;
        self
    }

    /// Generates the apt sources list entry.
    pub fn to_sources_list_entry(&self) -> String {
        let protocol = if self.use_https { "https" } else { "http" };
        format!(
            "deb [arch=amd64] {}://{} {} {}",
            protocol,
            self.url,
            self.codename,
            self.component
        )
    }
}

impl Default for RepositoryConfig {
    fn default() -> Self {
        Self::new(
            "ROCm",
            "repo.radeon.com/rocm/apt",
            "ubuntu",
        )
        .with_gpg_key("https://repo.radeon.com/rocm/rocm.gpg.key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_manager_name() {
        assert_eq!(PackageManager::Apt.name(), "apt");
        assert_eq!(PackageManager::Dnf.name(), "dnf");
        assert_eq!(PackageManager::Pacman.name(), "pacman");
        assert_eq!(PackageManager::Zypper.name(), "zypper");
    }

    #[test]
    fn test_repository_config_builder() {
        let config = RepositoryConfig::new("Test", "https://example.com", "jammy")
            .with_component("universe")
            .with_gpg_key("https://example.com/key.gpg")
            .with_https(true);

        assert_eq!(config.name, "Test");
        assert_eq!(config.codename, "jammy");
        assert_eq!(config.component, "universe");
        assert_eq!(config.gpg_key_url, "https://example.com/key.gpg");
        assert!(config.use_https);
    }

    #[test]
    fn test_repository_config_default() {
        let config = RepositoryConfig::default();
        assert_eq!(config.name, "ROCm");
        assert!(config.use_https);
    }
}
