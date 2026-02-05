//! PyTorch Installer
//!
//! ROCm-only PyTorch installer with CUDA blocking to prevent accidental CUDA installation.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use mlstack_hardware::{GPUArchitecture, HardwareDiscovery};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// PyTorch installation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PyTorchInstallMethod {
    /// Install via pip
    #[default]
    Pip,
    /// Install via conda
    Conda,
    /// Build from source
    Source,
}

/// PyTorch version with ROCm support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchVersion {
    /// PyTorch version string (e.g., "2.5.0")
    pub version: String,
    /// ROCm version compatibility
    pub rocm_version: String,
    /// Index URL for pip installation
    pub index_url: String,
}

impl PyTorchVersion {
    /// Creates a new PyTorch version specification.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        let version_str = version.into();
        let rocm_str = rocm_version.into();
        let index_url = format!(
            "https://download.pytorch.org/whl/rocm{}",
            rocm_str.replace(".", "")
        );
        
        Self {
            version: version_str,
            rocm_version: rocm_str,
            index_url,
        }
    }

    /// Returns the pip install command for this version.
    pub fn pip_install_command(&self) -> Vec<String> {
        vec![
            "pip".to_string(),
            "install".to_string(),
            "--upgrade".to_string(),
            format!("torch=={}", self.version),
            format!("--index-url={}", self.index_url),
        ]
    }
}

/// CUDA blocking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaBlockConfig {
    /// Block CUDA toolkit installation
    pub block_cuda_toolkit: bool,
    /// Block CUDA-enabled PyTorch
    pub block_cuda_pytorch: bool,
    /// Block nvidia packages
    pub block_nvidia_packages: bool,
    /// Environment variable to set
    pub cuda_visible_devices: String,
}

impl Default for CudaBlockConfig {
    fn default() -> Self {
        Self {
            block_cuda_toolkit: true,
            block_cuda_pytorch: true,
            block_nvidia_packages: true,
            cuda_visible_devices: "".to_string(),
        }
    }
}

impl CudaBlockConfig {
    /// Creates a strict CUDA blocking configuration.
    pub fn strict() -> Self {
        Self {
            block_cuda_toolkit: true,
            block_cuda_pytorch: true,
            block_nvidia_packages: true,
            cuda_visible_devices: "".to_string(),
        }
    }

    /// Returns environment variables for CUDA blocking.
    pub fn blocking_env_vars(&self) -> Vec<(String, String)> {
        let mut vars = Vec::new();
        
        if self.block_cuda_pytorch {
            vars.push(("CUDA_VISIBLE_DEVICES".to_string(), "".to_string()));
            vars.push(("FORCE_CUDA".to_string(), "0".to_string()));
        }
        
        vars
    }
}

/// PyTorch installer with ROCm-only enforcement.
pub struct PyTorchInstaller {
    version: PyTorchVersion,
    method: PyTorchInstallMethod,
    cuda_block: CudaBlockConfig,
    hardware: HardwareDiscovery,
}

impl PyTorchInstaller {
    /// Creates a new PyTorch installer.
    pub fn new(version: PyTorchVersion, method: PyTorchInstallMethod) -> Self {
        Self {
            version,
            method,
            cuda_block: CudaBlockConfig::strict(),
            hardware: HardwareDiscovery::new(),
        }
    }

    /// Sets the installation method.
    pub fn with_method(mut self, method: PyTorchInstallMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets CUDA blocking configuration.
    pub fn with_cuda_block(mut self, config: CudaBlockConfig) -> Self {
        self.cuda_block = config;
        self
    }

    /// Checks if PyTorch is already installed.
    pub fn is_pytorch_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import torch; print(torch.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if the installed PyTorch is CUDA-enabled.
    pub fn is_cuda_pytorch(&self) -> bool {
        if let Ok(output) = Command::new("python3")
            .args(["-c", "import torch; print(torch.cuda.is_available())"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout.trim() == "True";
            }
        }
        false
    }

    /// Checks if the installed PyTorch is ROCm-enabled.
    pub fn is_rocm_pytorch(&self) -> bool {
        if let Ok(output) = Command::new("python3")
            .args(["-c", "import torch; print(torch.version.hip)"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return !stdout.trim().is_empty() && stdout.trim() != "None";
            }
        }
        false
    }

    /// Detects GPU architecture and returns appropriate PyTorch version.
    pub fn detect_compatible_version(&self) -> Result<PyTorchVersion> {
        let gpus = self.hardware.detect_gpus()
            .context("Failed to detect GPUs")?;
        
        let arch = gpus.first()
            .map(|g| g.architecture)
            .unwrap_or(GPUArchitecture::Gfx1100);
        
        // RDNA 4 requires ROCm 7.2+
        let rocm_ver = if arch.is_rdna4() {
            "7.2"
        } else if arch.is_rdna3() {
            "7.1"
        } else {
            "6.4"
        };
        
        Ok(PyTorchVersion::new("2.5.0", rocm_ver))
    }

    /// Applies CUDA blocking environment variables.
    fn apply_cuda_blocking(&self) {
        for (key, value) in self.cuda_block.blocking_env_vars() {
            std::env::set_var(&key, &value);
        }
    }

    /// Installs PyTorch via pip.
    async fn install_via_pip(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing PyTorch via pip...".to_string());
        }

        // Apply CUDA blocking
        self.apply_cuda_blocking();

        let cmd_args = self.version.pip_install_command();
        
        if let Some(ref cb) = progress {
            cb(0.3, "Downloading PyTorch packages...".to_string());
        }

        let output = Command::new(&cmd_args[0])
            .args(&cmd_args[1..])
            .output()
            .context("Failed to run pip install")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("pip install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.8, "PyTorch installed, verifying...".to_string());
        }

        // Verify ROCm PyTorch was installed
        if !self.is_rocm_pytorch() {
            return Err(InstallerError::InstallationFailed(
                "CUDA PyTorch was installed instead of ROCm PyTorch. \
                This usually means the ROCm index URL was not used.".to_string()
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "PyTorch ROCm installation complete".to_string());
        }

        Ok(())
    }

    /// Installs PyTorch via conda.
    async fn install_via_conda(&self, _progress: &Option<ProgressCallback>) -> Result<()> {
        // Conda installation not yet implemented
        Err(InstallerError::UnsupportedPackageManager(
            "Conda installation not yet implemented".to_string()
        ).into())
    }

    /// Builds PyTorch from source.
    async fn install_from_source(&self, _progress: &Option<ProgressCallback>) -> Result<()> {
        // Source build not yet implemented
        Err(InstallerError::InstallationFailed(
            "Source build not yet implemented".to_string()
        ).into())
    }
}

#[async_trait::async_trait]
impl Installer for PyTorchInstaller {
    fn name(&self) -> &str {
        "PyTorch"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_pytorch_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        // Check Python is available
        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        // Check pip is available
        if Command::new("pip").arg("--version").output().is_err() {
            checks.push("pip is not installed".to_string());
        }

        // Check ROCm is installed
        if !Path::new("/opt/rocm").exists() {
            checks.push("ROCm is not installed (required for PyTorch ROCm)".to_string());
        }

        // Check for existing PyTorch installation
        if self.is_pytorch_installed() {
            if self.is_cuda_pytorch() {
                checks.push("WARNING: CUDA PyTorch is currently installed - will be replaced".to_string());
            } else if self.is_rocm_pytorch() {
                checks.push("ROCm PyTorch is already installed".to_string());
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        match self.method {
            PyTorchInstallMethod::Pip => self.install_via_pip(&progress).await,
            PyTorchInstallMethod::Conda => self.install_via_conda(&progress).await,
            PyTorchInstallMethod::Source => self.install_from_source(&progress).await,
        }
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "torch", "torchvision", "torchaudio"])
            .output()
            .context("Failed to uninstall PyTorch")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_pytorch_installed() && self.is_rocm_pytorch())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_version_creation() {
        let version = PyTorchVersion::new("2.5.0", "6.4");
        assert_eq!(version.version, "2.5.0");
        assert_eq!(version.rocm_version, "6.4");
        assert!(version.index_url.contains("rocm"));
    }

    #[test]
    fn test_pytorch_pip_command() {
        let version = PyTorchVersion::new("2.5.0", "6.4");
        let cmd = version.pip_install_command();
        assert!(cmd.contains(&"pip".to_string()));
        assert!(cmd.contains(&"install".to_string()));
        assert!(cmd.iter().any(|s| s.contains("torch")));
    }

    #[test]
    fn test_cuda_block_config() {
        let config = CudaBlockConfig::strict();
        assert!(config.block_cuda_pytorch);
        
        let vars = config.blocking_env_vars();
        assert!(vars.iter().any(|(k, _)| k == "CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn test_pytorch_installer_creation() {
        let version = PyTorchVersion::new("2.5.0", "6.4");
        let installer = PyTorchInstaller::new(version, PyTorchInstallMethod::Pip);
        assert_eq!(installer.name(), "PyTorch");
        assert!(installer.cuda_block.block_cuda_pytorch);
    }
}
