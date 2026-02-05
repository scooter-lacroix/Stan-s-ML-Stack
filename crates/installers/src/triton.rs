//! Triton Installer
//!
//! OpenAI Triton installer for ROCm with LLVM-AMD backend support.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Triton installation sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TritonSource {
    /// Install from PyPI
    #[default]
    Pypi,
    /// Build from source with ROCm support
    Source,
    /// Install from AMD fork
    AmdFork,
}

/// Triton version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonVersion {
    /// Version string
    pub version: String,
    /// Git commit hash (for source builds)
    pub git_commit: Option<String>,
    /// ROCm version compatibility
    pub rocm_version: String,
}

impl TritonVersion {
    /// Creates a new Triton version.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            git_commit: None,
            rocm_version: rocm_version.into(),
        }
    }

    /// Sets the git commit for source builds.
    pub fn with_commit(mut self, commit: impl Into<String>) -> Self {
        self.git_commit = Some(commit.into());
        self
    }

    /// Returns the pip install command.
    pub fn pip_install_command(&self) -> Vec<String> {
        vec![
            "pip".to_string(),
            "install".to_string(),
            format!("triton=={}", self.version),
        ]
    }
}

/// Triton installer for ROCm.
pub struct TritonInstaller {
    version: TritonVersion,
    source: TritonSource,
    rocm_path: String,
}

impl TritonInstaller {
    /// Creates a new Triton installer.
    pub fn new(version: TritonVersion, source: TritonSource) -> Self {
        Self {
            version,
            source,
            rocm_path: "/opt/rocm".to_string(),
        }
    }

    /// Sets the ROCm installation path.
    pub fn with_rocm_path(mut self, path: impl Into<String>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Checks if Triton is installed.
    pub fn is_triton_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import triton; print(triton.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if Triton has ROCm support.
    pub fn has_rocm_support(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import triton; print(triton.backends.backends)"])
            .output()
            .map(|o| {
                if o.status.success() {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    stdout.to_lowercase().contains("rocm") || stdout.to_lowercase().contains("hip")
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    /// Installs Triton from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing Triton from PyPI...".to_string());
        }

        let cmd_args = self.version.pip_install_command();

        let output = Command::new(&cmd_args[0])
            .args(&cmd_args[1..])
            .output()
            .context("Failed to run pip install for Triton")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Triton pip install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "Triton installation complete".to_string());
        }

        Ok(())
    }

    /// Builds Triton from source with ROCm support.
    async fn install_from_source(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Building Triton from source...".to_string());
        }

        // Clone repository
        let repo_url = "https://github.com/triton-lang/triton.git";
        let clone_dir = std::env::temp_dir().join("triton-build");

        if let Some(ref cb) = progress {
            cb(0.2, "Cloning Triton repository...".to_string());
        }

        let output = Command::new("git")
            .args(["clone", "--depth", "1", repo_url, clone_dir.to_str().unwrap()])
            .output()
            .context("Failed to clone Triton repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.4, "Building Triton with ROCm support...".to_string());
        }

        // Set environment variables for ROCm build
        std::env::set_var("TRITON_ROCM_PATH", &self.rocm_path);
        std::env::set_var("LLVM_ENABLE_ZSTD", "OFF");

        // Build and install
        let output = Command::new("pip")
            .args(["install", "-e", "."])
            .current_dir(clone_dir.join("python"))
            .output()
            .context("Failed to build Triton from source")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Triton build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "Triton source build complete".to_string());
        }

        Ok(())
    }

    /// Installs from AMD fork (placeholder).
    async fn install_from_amd_fork(&self, _progress: &Option<ProgressCallback>) -> Result<()> {
        Err(InstallerError::InstallationFailed(
            "AMD fork installation not yet implemented".to_string()
        ).into())
    }
}

#[async_trait::async_trait]
impl Installer for TritonInstaller {
    fn name(&self) -> &str {
        "Triton"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_triton_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        // Check Python
        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        // Check pip
        if Command::new("pip").arg("--version").output().is_err() {
            checks.push("pip is not installed".to_string());
        }

        // Check ROCm
        if !Path::new(&self.rocm_path).exists() {
            checks.push(format!("ROCm not found at {}", self.rocm_path));
        }

        // Check PyTorch
        if Command::new("python3")
            .args(["-c", "import torch"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            checks.push("PyTorch is not installed (required for Triton)".to_string());
        }

        // Check existing Triton
        if self.is_triton_installed() {
            checks.push("Triton is already installed".to_string());
            if !self.has_rocm_support() {
                checks.push("WARNING: Existing Triton may not have ROCm support".to_string());
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        match self.source {
            TritonSource::Pypi => self.install_from_pypi(&progress).await,
            TritonSource::Source => self.install_from_source(&progress).await,
            TritonSource::AmdFork => self.install_from_amd_fork(&progress).await,
        }
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "triton"])
            .output()
            .context("Failed to uninstall Triton")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_triton_installed() && self.has_rocm_support())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triton_version_creation() {
        let version = TritonVersion::new("3.0.0", "6.4");
        assert_eq!(version.version, "3.0.0");
        assert_eq!(version.rocm_version, "6.4");
    }

    #[test]
    fn test_triton_version_with_commit() {
        let version = TritonVersion::new("3.0.0", "6.4")
            .with_commit("abc123");
        assert_eq!(version.git_commit, Some("abc123".to_string()));
    }

    #[test]
    fn test_triton_pip_command() {
        let version = TritonVersion::new("3.0.0", "6.4");
        let cmd = version.pip_install_command();
        assert!(cmd.contains(&"pip".to_string()));
        assert!(cmd.contains(&"triton==3.0.0".to_string()));
    }

    #[test]
    fn test_triton_installer_creation() {
        let version = TritonVersion::new("3.0.0", "6.4");
        let installer = TritonInstaller::new(version, TritonSource::Pypi);
        assert_eq!(installer.name(), "Triton");
        assert_eq!(installer.rocm_path, "/opt/rocm");
    }
}
