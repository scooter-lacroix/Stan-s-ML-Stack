//! GPG Key Management
//!
//! Handles GPG key import with special handling for Debian Trixie sqv fix.

use crate::common::{download_file, InstallerError};
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// Manages GPG keys for repository trust.
pub struct GpgKeyManager;

impl GpgKeyManager {
    /// Creates a new GPG key manager.
    pub fn new() -> Self {
        Self
    }

    /// Imports a GPG key from a URL.
    ///
    /// Handles Debian Trixie sqv workaround automatically.
    pub async fn import_key_from_url(&self, url: &str, key_name: &str) -> Result<()> {
        let key_path = download_file(url, key_name).await
            .context("Failed to download GPG key")?;

        self.import_key_file(&key_path).await
    }

    /// Imports a GPG key from a local file.
    pub async fn import_key_file(&self, key_path: &Path) -> Result<()> {
        if Self::is_debian_trixie_or_newer() {
            self.import_key_trixie(key_path).await
        } else {
            self.import_key_legacy(key_path).await
        }
    }

    /// Checks if running on Debian Trixie or newer.
    ///
    /// Trixie and newer use sqv instead of gpg for key verification,
    /// which requires a different key import method.
    fn is_debian_trixie_or_newer() -> bool {
        // Check for Debian version
        if let Ok(version_id) = std::fs::read_to_string("/etc/debian_version") {
            let version = version_id.trim();
            // Trixie is Debian 13, Bookworm is 12
            if version.starts_with("13.") || version.starts_with("14.") || version.starts_with("trixie") || version.starts_with("sid") {
                return true;
            }
        }

        // Check os-release for trixie/sid
        if let Ok(os_release) = std::fs::read_to_string("/etc/os-release") {
            let os_release_lower = os_release.to_lowercase();
            if os_release_lower.contains("trixie") || os_release_lower.contains("sid") {
                return true;
            }
        }

        false
    }

    /// Imports key on Debian Trixie and newer using gpg --keyring.
    async fn import_key_trixie(&self, key_path: &Path) -> Result<()> {
        // Create keyring directory if needed
        let keyring_dir = Path::new("/etc/apt/keyrings");
        if !keyring_dir.exists() {
            tokio::fs::create_dir_all(keyring_dir).await
                .context("Failed to create keyrings directory")?;
        }

        // Copy key to keyrings directory
        let dest_path = keyring_dir.join(key_path.file_name().unwrap_or("key.gpg".as_ref()));
        tokio::fs::copy(key_path, &dest_path).await
            .context("Failed to copy key to keyrings directory")?;

        // Set appropriate permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o644);
            std::fs::set_permissions(&dest_path, permissions)
                .context("Failed to set key permissions")?;
        }

        Ok(())
    }

    /// Imports key on legacy systems using apt-key.
    async fn import_key_legacy(&self, key_path: &Path) -> Result<()> {
        let output = Command::new("apt-key")
            .args(["add", key_path.to_str().unwrap_or("/dev/null")])
            .output()
            .context("Failed to run apt-key add")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::GpgKeyImportFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    /// Removes a previously imported key.
    pub async fn remove_key(&self, key_id: &str) -> Result<()> {
        let output = Command::new("apt-key")
            .args(["del", key_id])
            .output()
            .context("Failed to run apt-key del")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::GpgKeyImportFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    /// Lists imported keys.
    pub fn list_keys(&self) -> Result<Vec<String>> {
        let output = Command::new("apt-key")
            .arg("list")
            .output()
            .context("Failed to list keys")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::GpgKeyImportFailed(stderr.to_string()).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().map(|s| s.to_string()).collect())
    }
}

impl Default for GpgKeyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpg_key_manager_creation() {
        let manager = GpgKeyManager::new();
        // Just verify it creates without panic
        let _ = manager;
    }

    #[test]
    fn test_is_debian_trixie_detection() {
        // This test depends on the system, just verify it doesn't panic
        let _ = GpgKeyManager::is_debian_trixie_or_newer();
    }
}
