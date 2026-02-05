//! Flash Attention Installer
//!
//! Flash Attention 2 installer for ROCm with custom kernel support.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Flash Attention version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionVersion {
    /// Version string
    pub version: String,
    /// Git tag or commit
    pub git_ref: String,
    /// ROCm version compatibility
    pub rocm_version: String,
}

impl FlashAttentionVersion {
    /// Creates a new Flash Attention version.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: version_str.clone(),
            version: version_str,
            rocm_version: rocm_version.into(),
        }
    }

    /// Sets the git reference.
    pub fn with_git_ref(mut self, git_ref: impl Into<String>) -> Self {
        self.git_ref = git_ref.into();
        self
    }
}

/// Flash Attention installer for ROCm.
pub struct FlashAttentionInstaller {
    version: FlashAttentionVersion,
    rocm_path: String,
    max_jobs: usize,
}

impl FlashAttentionInstaller {
    /// Creates a new Flash Attention installer.
    pub fn new(version: FlashAttentionVersion) -> Self {
        Self {
            version,
            rocm_path: "/opt/rocm".to_string(),
            max_jobs: num_cpus::get(),
        }
    }

    /// Sets the ROCm installation path.
    pub fn with_rocm_path(mut self, path: impl Into<String>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Sets the maximum parallel jobs for compilation.
    pub fn with_max_jobs(mut self, jobs: usize) -> Self {
        self.max_jobs = jobs;
        self
    }

    /// Checks if Flash Attention is installed.
    pub fn is_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import flash_attn; print(flash_attn.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if Flash Attention has ROCm support.
    pub fn has_rocm_support(&self) -> bool {
        // Check if the module loads and has ROCm-specific attributes
        Command::new("python3")
            .args(["-c", "from flash_attn import flash_attn_func; print('OK')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Builds Flash Attention from source.
    async fn build_from_source(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Building Flash Attention from source...".to_string());
        }

        // Clone repository
        let repo_url = "https://github.com/Dao-AILab/flash-attention.git";
        let clone_dir = std::env::temp_dir().join("flash-attention-build");

        if let Some(ref cb) = progress {
            cb(0.2, "Cloning Flash Attention repository...".to_string());
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch",
                &self.version.git_ref,
                "--depth",
                "1",
                repo_url,
                clone_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone Flash Attention repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.4, "Setting up ROCm build environment...".to_string());
        }

        // Set environment variables for ROCm build
        std::env::set_var("ROCM_HOME", &self.rocm_path);
        std::env::set_var("HIP_PATH", format!("{}/hip", self.rocm_path));
        std::env::set_var("MAX_JOBS", self.max_jobs.to_string());

        // For RDNA 3/4, set HSA_OVERRIDE_GFX_VERSION
        if self.version.rocm_version.starts_with("7.") {
            std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "11.0.0");
        }

        if let Some(ref cb) = progress {
            cb(0.6, "Compiling Flash Attention kernels...".to_string());
        }

        // Build and install
        let output = Command::new("pip")
            .args(["install", "-v", "."])
            .current_dir(&clone_dir)
            .output()
            .context("Failed to build Flash Attention")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Flash Attention build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "Flash Attention installation complete".to_string());
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl Installer for FlashAttentionInstaller {
    fn name(&self) -> &str {
        "Flash Attention"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_installed())
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
            checks.push("PyTorch is not installed (required for Flash Attention)".to_string());
        }

        // Check Triton (required for Flash Attention 2)
        if Command::new("python3")
            .args(["-c", "import triton"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            checks.push("Triton is not installed (required for Flash Attention 2)".to_string());
        }

        // Check git
        if Command::new("git").arg("--version").output().is_err() {
            checks.push("Git is not installed (required for source build)".to_string());
        }

        // Check existing installation
        if self.is_installed() {
            checks.push("Flash Attention is already installed".to_string());
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        self.build_from_source(&progress).await
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "flash-attn"])
            .output()
            .context("Failed to uninstall Flash Attention")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_installed() && self.has_rocm_support())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_version_creation() {
        let version = FlashAttentionVersion::new("2.5.0", "6.4");
        assert_eq!(version.version, "2.5.0");
        assert_eq!(version.git_ref, "2.5.0");
        assert_eq!(version.rocm_version, "6.4");
    }

    #[test]
    fn test_flash_attention_version_with_git_ref() {
        let version = FlashAttentionVersion::new("2.5.0", "6.4")
            .with_git_ref("v2.5.0");
        assert_eq!(version.git_ref, "v2.5.0");
    }

    #[test]
    fn test_flash_attention_installer_creation() {
        let version = FlashAttentionVersion::new("2.5.0", "6.4");
        let installer = FlashAttentionInstaller::new(version);
        assert_eq!(installer.name(), "Flash Attention");
        assert_eq!(installer.rocm_path, "/opt/rocm");
        assert!(installer.max_jobs > 0);
    }

    #[test]
    fn test_flash_attention_installer_builder() {
        let version = FlashAttentionVersion::new("2.5.0", "6.4");
        let installer = FlashAttentionInstaller::new(version)
            .with_rocm_path("/custom/rocm")
            .with_max_jobs(4);
        
        assert_eq!(installer.rocm_path, "/custom/rocm");
        assert_eq!(installer.max_jobs, 4);
    }
}
