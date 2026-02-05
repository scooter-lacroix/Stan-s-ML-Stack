//! ONNX Runtime Installer
//!
//! ONNX Runtime installer for ROCm with MIGraphX and ROCm execution providers.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// ONNX Runtime installation sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OnnxSource {
    /// Install from PyPI
    #[default]
    Pypi,
    /// Build from source with ROCm support
    Source,
    /// Install from Microsoft repository
    Microsoft,
}

/// ONNX Runtime version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxVersion {
    /// Version string
    pub version: String,
    /// ROCm version compatibility
    pub rocm_version: String,
    /// Execution providers to enable
    pub execution_providers: Vec<ExecutionProvider>,
}

/// Execution providers for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// ROCm execution provider
    Rocm,
    /// MIGraphX execution provider
    MiGraphX,
    /// CPU execution provider
    Cpu,
}

impl ExecutionProvider {
    /// Returns the provider name.
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionProvider::Rocm => "ROCMExecutionProvider",
            ExecutionProvider::MiGraphX => "MIGraphXExecutionProvider",
            ExecutionProvider::Cpu => "CPUExecutionProvider",
        }
    }

    /// Returns the CMake flag for building.
    pub fn cmake_flag(&self) -> &'static str {
        match self {
            ExecutionProvider::Rocm => "-Donnxruntime_USE_ROCM=ON",
            ExecutionProvider::MiGraphX => "-Donnxruntime_USE_MIGRAPHX=ON",
            ExecutionProvider::Cpu => "-Donnxruntime_USE_CPU=ON",
        }
    }
}

impl OnnxVersion {
    /// Creates a new ONNX Runtime version.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            rocm_version: rocm_version.into(),
            execution_providers: vec![ExecutionProvider::Rocm, ExecutionProvider::Cpu],
        }
    }

    /// Creates the latest stable version.
    pub fn latest() -> Self {
        Self::new("1.17.0", "6.4")
            .with_providers(vec![ExecutionProvider::Rocm, ExecutionProvider::MiGraphX, ExecutionProvider::Cpu])
    }

    /// Sets the execution providers.
    pub fn with_providers(mut self, providers: Vec<ExecutionProvider>) -> Self {
        self.execution_providers = providers;
        self
    }

    /// Returns the pip install command.
    pub fn pip_install_command(&self) -> Vec<String> {
        vec![
            "pip".to_string(),
            "install".to_string(),
            format!("onnxruntime-rocm=={}", self.version),
        ]
    }
}

/// Detailed verification result for ONNX Runtime installation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnnxVerification {
    /// Whether ONNX Runtime is installed
    pub installed: bool,
    /// Installed version (if any)
    pub version: Option<String>,
    /// Whether ROCm execution provider is available
    pub rocm_provider_available: bool,
    /// Whether MIGraphX execution provider is available
    pub migraphx_provider_available: bool,
    /// Whether CPU execution provider is available
    pub cpu_provider_available: bool,
    /// All available execution providers
    pub available_providers: Vec<String>,
    /// Python site-packages path
    pub install_path: Option<String>,
}

/// ONNX Runtime installer for ROCm.
pub struct OnnxInstaller {
    version: OnnxVersion,
    source: OnnxSource,
    rocm_path: String,
}

impl OnnxInstaller {
    /// Creates a new ONNX Runtime installer.
    pub fn new(version: OnnxVersion, source: OnnxSource) -> Self {
        Self {
            version,
            source,
            rocm_path: "/opt/rocm".to_string(),
        }
    }

    /// Creates installer for latest PyPI version.
    pub fn latest() -> Self {
        Self::new(OnnxVersion::latest(), OnnxSource::Pypi)
    }

    /// Creates installer from PyPI.
    pub fn from_pypi() -> Self {
        Self::new(OnnxVersion::latest(), OnnxSource::Pypi)
    }

    /// Creates installer to build from source with ROCm support.
    pub fn from_source() -> Self {
        Self::new(OnnxVersion::latest(), OnnxSource::Source)
    }

    /// Sets the ROCm installation path.
    pub fn with_rocm_path(mut self, path: impl Into<String>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Gets the installed ONNX Runtime version.
    pub fn get_installed_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import onnxruntime; print(onnxruntime.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Gets all available execution providers.
    pub fn get_available_providers(&self) -> Vec<String> {
        Command::new("python3")
            .args(["-c", "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .split(',')
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Gets the installation site-packages path.
    pub fn get_install_path(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Runs a detailed verification of the ONNX Runtime installation.
    pub fn verify_installation(&self) -> Result<OnnxVerification> {
        let mut result = OnnxVerification::default();

        // Check basic installation
        result.version = self.get_installed_version();
        result.installed = result.version.is_some();

        if result.installed {
            // Get install path
            result.install_path = self.get_install_path();

            // Get available providers
            result.available_providers = self.get_available_providers();

            // Check specific providers
            result.rocm_provider_available = result.available_providers.contains(&"ROCMExecutionProvider".to_string());
            result.migraphx_provider_available = result.available_providers.contains(&"MIGraphXExecutionProvider".to_string());
            result.cpu_provider_available = result.available_providers.contains(&"CPUExecutionProvider".to_string());
        }

        Ok(result)
    }

    /// Checks if ONNX Runtime is installed.
    pub fn is_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import onnxruntime; print(onnxruntime.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if ROCm execution provider is available.
    pub fn has_rocm_provider(&self) -> bool {
        Command::new("python3")
            .args([
                "-c",
                "import onnxruntime as ort; print('ROCMExecutionProvider' in ort.get_available_providers())"
            ])
            .output()
            .map(|o| {
                if o.status.success() {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    stdout.trim() == "True"
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    /// Installs ONNX Runtime from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing ONNX Runtime from PyPI...".to_string());
        }

        let output = Command::new("pip")
            .args(["install", &format!("onnxruntime-rocm=={}", self.version.version)])
            .output()
            .context("Failed to run pip install for ONNX Runtime")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("ONNX Runtime pip install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "ONNX Runtime installation complete".to_string());
        }

        Ok(())
    }

    /// Builds ONNX Runtime from source.
    async fn install_from_source(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Building ONNX Runtime from source...".to_string());
        }

        let repo_url = "https://github.com/Microsoft/onnxruntime.git";
        let clone_dir = std::env::temp_dir().join("onnxruntime-build");

        // Clone
        if let Some(ref cb) = progress {
            cb(0.2, "Cloning ONNX Runtime repository...".to_string());
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch",
                &format!("v{}", self.version.version),
                "--depth",
                "1",
                "--recursive",
                repo_url,
                clone_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone ONNX Runtime repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.4, "Configuring build with ROCm support...".to_string());
        }

        // Build configuration
        std::env::set_var("ROCM_HOME", &self.rocm_path);

        let mut cmake_args = vec![
            "cmake".to_string(),
            "-S".to_string(),
            clone_dir.join("cmake").to_str().unwrap().to_string(),
            "-B".to_string(),
            clone_dir.join("build").to_str().unwrap().to_string(),
            "-DCMAKE_BUILD_TYPE=Release".to_string(),
            "-Donnxruntime_BUILD_SHARED_LIB=ON".to_string(),
            "-Donnxruntime_BUILD_UNIT_TESTS=OFF".to_string(),
        ];

        // Add execution provider flags
        for provider in &self.version.execution_providers {
            cmake_args.push(provider.cmake_flag().to_string());
        }

        let output = Command::new(&cmake_args[0])
            .args(&cmake_args[1..])
            .output()
            .context("Failed to configure ONNX Runtime build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("CMake configuration failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.6, "Building ONNX Runtime (this may take a while)...".to_string());
        }

        // Build
        let output = Command::new("cmake")
            .args(["--build", clone_dir.join("build").to_str().unwrap(), "--parallel"])
            .output()
            .context("Failed to build ONNX Runtime")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.9, "Installing ONNX Runtime...".to_string());
        }

        // Install Python package
        let output = Command::new("pip")
            .args(["install", "-e", "."])
            .current_dir(clone_dir.join("build"))
            .output()
            .context("Failed to install ONNX Runtime Python package")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Python package install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "ONNX Runtime source build complete".to_string());
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl Installer for OnnxInstaller {
    fn name(&self) -> &str {
        "ONNX Runtime"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 not installed".to_string());
        }

        if Command::new("pip").arg("--version").output().is_err() {
            checks.push("pip not installed".to_string());
        }

        if !Path::new(&self.rocm_path).exists() {
            checks.push(format!("ROCm not found at {}", self.rocm_path));
        }

        if self.is_installed() {
            checks.push("ONNX Runtime already installed".to_string());
            if !self.has_rocm_provider() {
                checks.push("WARNING: Existing ONNX Runtime may not have ROCm support".to_string());
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        match self.source {
            OnnxSource::Pypi | OnnxSource::Microsoft => {
                self.install_from_pypi(&progress).await
            }
            OnnxSource::Source => self.install_from_source(&progress).await,
        }
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "onnxruntime", "onnxruntime-rocm"])
            .output()
            .context("Failed to uninstall ONNX Runtime")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_installed() && self.has_rocm_provider())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_version_creation() {
        let version = OnnxVersion::new("1.17.0", "6.4");
        assert_eq!(version.version, "1.17.0");
        assert_eq!(version.rocm_version, "6.4");
        assert!(!version.execution_providers.is_empty());
    }

    #[test]
    fn test_onnx_version_latest() {
        let version = OnnxVersion::latest();
        assert!(!version.version.is_empty());
        assert!(!version.execution_providers.is_empty());
    }

    #[test]
    fn test_execution_provider_names() {
        assert_eq!(ExecutionProvider::Rocm.name(), "ROCMExecutionProvider");
        assert_eq!(ExecutionProvider::MiGraphX.name(), "MIGraphXExecutionProvider");
        assert_eq!(ExecutionProvider::Cpu.name(), "CPUExecutionProvider");
    }

    #[test]
    fn test_execution_provider_cmake_flags() {
        assert!(ExecutionProvider::Rocm.cmake_flag().contains("ROCM"));
        assert!(ExecutionProvider::MiGraphX.cmake_flag().contains("MIGRAPHX"));
    }

    #[test]
    fn test_onnx_installer_creation() {
        let version = OnnxVersion::new("1.17.0", "6.4");
        let installer = OnnxInstaller::new(version, OnnxSource::Pypi);
        assert_eq!(installer.name(), "ONNX Runtime");
        assert_eq!(installer.rocm_path, "/opt/rocm");
    }

    #[test]
    fn test_onnx_installer_latest() {
        let installer = OnnxInstaller::latest();
        assert_eq!(installer.name(), "ONNX Runtime");
    }

    #[test]
    fn test_onnx_installer_from_pypi() {
        let installer = OnnxInstaller::from_pypi();
        assert_eq!(installer.name(), "ONNX Runtime");
    }

    #[test]
    fn test_onnx_installer_from_source() {
        let installer = OnnxInstaller::from_source();
        assert_eq!(installer.name(), "ONNX Runtime");
    }

    #[test]
    fn test_onnx_installer_builder() {
        let version = OnnxVersion::new("1.17.0", "6.4");
        let installer = OnnxInstaller::new(version, OnnxSource::Source)
            .with_rocm_path("/custom/rocm");

        assert_eq!(installer.rocm_path, "/custom/rocm");
    }

    #[test]
    fn test_verification_result_default() {
        let result = OnnxVerification::default();
        assert!(!result.installed);
        assert!(result.version.is_none());
        assert!(!result.rocm_provider_available);
    }

    #[test]
    fn test_verification_installation() {
        let installer = OnnxInstaller::from_pypi();
        let result = installer.verify_installation();
        assert!(result.is_ok());
    }
}
