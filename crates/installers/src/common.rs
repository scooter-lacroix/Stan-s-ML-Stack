//! Common Installer Traits and Utilities
//!
//! Defines the core installer interface and shared utilities.

use anyhow::Result;
use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during installation.
#[derive(Error, Debug)]
pub enum InstallerError {
    #[error("Package manager not supported: {0}")]
    UnsupportedPackageManager(String),
    
    #[error("Repository configuration failed: {0}")]
    RepositoryConfigFailed(String),
    
    #[error("GPG key import failed: {0}")]
    GpgKeyImportFailed(String),
    
    #[error("Installation failed: {0}")]
    InstallationFailed(String),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Version not found: {0}")]
    VersionNotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// Callback for installation progress updates.
pub type ProgressCallback = Box<dyn Fn(f32, String) + Send + Sync>;

/// Represents the status of an installation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallationStatus {
    /// Installation not started
    NotStarted,
    /// Installation in progress with percentage
    InProgress(u8),
    /// Installation completed successfully
    Completed,
    /// Installation failed with error message
    Failed(String),
    /// Installation cancelled
    Cancelled,
}

impl fmt::Display for InstallationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstallationStatus::NotStarted => write!(f, "Not started"),
            InstallationStatus::InProgress(pct) => write!(f, "In progress ({}%)", pct),
            InstallationStatus::Completed => write!(f, "Completed"),
            InstallationStatus::Failed(msg) => write!(f, "Failed: {}", msg),
            InstallationStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Core installer trait.
#[async_trait::async_trait]
pub trait Installer {
    /// Returns the name of the installer.
    fn name(&self) -> &str;
    
    /// Returns the version being installed.
    fn version(&self) -> &str;
    
    /// Checks if the component is already installed.
    async fn is_installed(&self) -> Result<bool>;
    
    /// Performs pre-installation checks.
    async fn preflight_check(&self) -> Result<Vec<String>>;
    
    /// Installs the component.
    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()>;
    
    /// Uninstalls the component.
    async fn uninstall(&self) -> Result<()>;
    
    /// Verifies the installation.
    async fn verify(&self) -> Result<bool>;
}

/// Detects if running with root privileges.
pub fn is_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

/// Ensures root privileges, returning error if not root.
pub fn require_root() -> Result<()> {
    if !is_root() {
        Err(InstallerError::PermissionDenied(
            "This operation requires root privileges".to_string()
        ).into())
    } else {
        Ok(())
    }
}

/// Gets the system temp directory.
pub fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// Downloads a file from URL to a temporary location.
pub async fn download_file(url: &str, filename: &str) -> Result<PathBuf> {
    let temp_path = temp_dir().join(filename);
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    tokio::fs::write(&temp_path, &bytes).await?;
    Ok(temp_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_installation_status_display() {
        assert_eq!(format!("{}", InstallationStatus::NotStarted), "Not started");
        assert_eq!(format!("{}", InstallationStatus::InProgress(50)), "In progress (50%)");
        assert_eq!(format!("{}", InstallationStatus::Completed), "Completed");
        assert_eq!(format!("{}", InstallationStatus::Cancelled), "Cancelled");
    }

    #[test]
    fn test_is_root_does_not_panic() {
        // Just verify it doesn't panic
        let _ = is_root();
    }
}
