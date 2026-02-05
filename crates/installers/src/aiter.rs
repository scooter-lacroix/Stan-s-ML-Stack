//! AITER (AMD AI Tensor Engine for ROCm) Installer
//!
//! Pure Rust implementation for installing AITER with full RDNA 3 GPU support.
//! Leverages async operations, comprehensive error handling, and parallel builds.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Supported RDNA 3 GPU architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArchitecture {
    /// RDNA 3 Navi 31 (RX 7900 XTX, 7900 XT)
    Gfx1100,
    /// RDNA 3 Navi 32 (RX 7800 XT, 7700 XT)
    Gfx1101,
    /// RDNA 3 Navi 33 (RX 7600, 7600 XT)
    Gfx1102,
}

impl GpuArchitecture {
    /// Returns the architecture string for ROCm.
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuArchitecture::Gfx1100 => "gfx1100",
            GpuArchitecture::Gfx1101 => "gfx1101",
            GpuArchitecture::Gfx1102 => "gfx1102",
        }
    }

    /// Returns all supported architectures as a semicolon-separated string.
    pub fn all_supported() -> &'static str {
        "gfx1100;gfx1101;gfx1102"
    }

    /// Returns GPU card names for this architecture.
    pub fn card_names(&self) -> &'static [&'static str] {
        match self {
            GpuArchitecture::Gfx1100 => &[
                "RX 7900 XTX",
                "RX 7900 XT",
                "Radeon PRO W7900",
                "Radeon PRO W7800",
            ],
            GpuArchitecture::Gfx1101 => &[
                "RX 7800 XT",
                "RX 7700 XT",
                "Radeon PRO W7700",
                "Radeon PRO W7600",
            ],
            GpuArchitecture::Gfx1102 => &["RX 7600", "RX 7600 XT", "Radeon PRO W7500"],
        }
    }
}

/// AITER installation sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AiterSource {
    /// Build from official ROCm GitHub repository
    #[default]
    RocmGithub,
    /// Install from PyPI (may lack ROCm optimizations)
    Pypi,
    /// Build from local source directory
    LocalSource,
}

/// AITER version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiterVersion {
    /// Version string
    pub version: String,
    /// Git reference (branch/tag/commit)
    pub git_ref: String,
    /// Minimum ROCm version required
    pub min_rocm_version: String,
    /// Target GPU architectures
    pub target_archs: Vec<GpuArchitecture>,
}

impl AiterVersion {
    /// Creates a new AITER version configuration.
    pub fn new(version: impl Into<String>, min_rocm: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: format!("v{}", version_str),
            version: version_str,
            min_rocm_version: min_rocm.into(),
            target_archs: vec![
                GpuArchitecture::Gfx1100,
                GpuArchitecture::Gfx1101,
                GpuArchitecture::Gfx1102,
            ],
        }
    }

    /// Creates version for latest development branch.
    pub fn latest() -> Self {
        Self {
            version: "dev".to_string(),
            git_ref: "main".to_string(),
            min_rocm_version: "6.0".to_string(),
            target_archs: vec![
                GpuArchitecture::Gfx1100,
                GpuArchitecture::Gfx1101,
                GpuArchitecture::Gfx1102,
            ],
        }
    }

    /// Sets a custom git reference.
    pub fn with_git_ref(mut self, git_ref: impl Into<String>) -> Self {
        self.git_ref = git_ref.into();
        self
    }

    /// Sets target GPU architectures.
    pub fn with_archs(mut self, archs: Vec<GpuArchitecture>) -> Self {
        self.target_archs = archs;
        self
    }
}

/// Build configuration for AITER.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiterBuildConfig {
    /// Use build isolation (pip --no-build-isolation)
    pub build_isolation: bool,
    /// Install dependencies
    pub install_deps: bool,
    /// Use editable install
    pub editable: bool,
    /// Number of parallel build jobs
    pub parallel_jobs: usize,
    /// Extra pip install arguments
    pub extra_args: Vec<String>,
}

impl Default for AiterBuildConfig {
    fn default() -> Self {
        Self {
            build_isolation: false,
            install_deps: true,
            editable: false,
            parallel_jobs: num_cpus::get(),
            extra_args: Vec::new(),
        }
    }
}

/// Environment configuration for AITER.
#[derive(Debug, Clone)]
pub struct AiterEnvironment {
    /// ROCm installation path
    pub rocm_path: PathBuf,
    /// HSA override GFX version
    pub hsa_gfx_version: String,
    /// PyTorch ROCm architecture targets
    pub pytorch_rocm_arch: String,
    /// HIP visible devices
    pub hip_visible_devices: String,
    /// Disable HIP/AMD log noise
    pub amd_log_level: String,
}

impl Default for AiterEnvironment {
    fn default() -> Self {
        Self {
            rocm_path: PathBuf::from("/opt/rocm"),
            hsa_gfx_version: "11.0.0".to_string(),
            pytorch_rocm_arch: GpuArchitecture::all_supported().to_string(),
            hip_visible_devices: "0".to_string(),
            amd_log_level: "0".to_string(),
        }
    }
}

impl AiterEnvironment {
    /// Applies environment variables for the current process.
    pub fn apply(&self) {
        std::env::set_var("ROCM_PATH", &self.rocm_path);
        std::env::set_var("HSA_OVERRIDE_GFX_VERSION", &self.hsa_gfx_version);
        std::env::set_var("PYTORCH_ROCM_ARCH", &self.pytorch_rocm_arch);
        std::env::set_var("HIP_VISIBLE_DEVICES", &self.hip_visible_devices);
        std::env::set_var("CUDA_VISIBLE_DEVICES", &self.hip_visible_devices);
        std::env::set_var("AMD_LOG_LEVEL", &self.amd_log_level);

        // Add ROCm to PATH
        if let Ok(path) = std::env::var("PATH") {
            let rocm_bin = self.rocm_path.join("bin");
            std::env::set_var("PATH", format!("{}:{}", rocm_bin.display(), path));
        }

        // Add ROCm to LD_LIBRARY_PATH
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            let rocm_lib = self.rocm_path.join("lib");
            std::env::set_var(
                "LD_LIBRARY_PATH",
                format!("{}:{}", rocm_lib.display(), ld_path),
            );
        }
    }

    /// Returns environment variables as a HashMap for subprocess.
    pub fn as_env_map(&self, _build_config: &AiterBuildConfig) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert(
            "ROCM_PATH".to_string(),
            self.rocm_path.to_string_lossy().to_string(),
        );
        env.insert(
            "ROCM_HOME".to_string(),
            self.rocm_path.to_string_lossy().to_string(),
        );
        env.insert(
            "HSA_OVERRIDE_GFX_VERSION".to_string(),
            self.hsa_gfx_version.clone(),
        );
        env.insert(
            "PYTORCH_ROCM_ARCH".to_string(),
            self.pytorch_rocm_arch.clone(),
        );
        env.insert(
            "HIP_VISIBLE_DEVICES".to_string(),
            self.hip_visible_devices.clone(),
        );
        env.insert(
            "CUDA_VISIBLE_DEVICES".to_string(),
            self.hip_visible_devices.clone(),
        );
        env.insert("AMD_LOG_LEVEL".to_string(), self.amd_log_level.clone());
        env
    }
}

/// AITER installer for ROCm.
pub struct AiterInstaller {
    version: AiterVersion,
    source: AiterSource,
    build_config: AiterBuildConfig,
    environment: AiterEnvironment,
}

impl AiterInstaller {
    /// Creates a new AITER installer with default configuration.
    pub fn new(version: AiterVersion, source: AiterSource) -> Self {
        Self {
            version,
            source,
            build_config: AiterBuildConfig::default(),
            environment: AiterEnvironment::default(),
        }
    }

    /// Creates installer for latest development version.
    pub fn latest() -> Self {
        Self::new(AiterVersion::latest(), AiterSource::RocmGithub)
    }

    /// Creates installer for latest PyPI version.
    pub fn from_pypi() -> Self {
        Self::new(AiterVersion::new("0.1.0", "5.7"), AiterSource::Pypi)
    }

    /// Sets the ROCm installation path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.environment.rocm_path = path.into();
        self
    }

    /// Sets build configuration.
    pub fn with_build_config(mut self, config: AiterBuildConfig) -> Self {
        self.build_config = config;
        self
    }

    /// Sets the number of parallel build jobs.
    pub fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.build_config.parallel_jobs = jobs;
        self
    }

    /// Checks if AITER is installed.
    fn check_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import aiter; print(aiter.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if AITER has the torch module.
    fn has_torch_module(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import aiter.torch; print('ok')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets the installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import aiter; print(aiter.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Checks if PyTorch with ROCm is available.
    fn check_pytorch_rocm(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import torch; assert hasattr(torch.version, 'hip'), 'No ROCm'; print(torch.version.hip)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if ROCm is properly installed.
    fn check_rocm(&self) -> Result<String> {
        let rocminfo = Command::new("rocminfo")
            .output()
            .context("rocminfo not found - is ROCm installed?")?;

        if !rocminfo.status.success() {
            return Err(InstallerError::InstallationFailed("rocminfo failed".to_string()).into());
        }

        // Parse ROCm version from output
        let output = String::from_utf8_lossy(&rocminfo.stdout);
        for line in output.lines() {
            if line.to_lowercase().contains("rocm version") {
                if let Some(version) = line.split(':').nth(1) {
                    return Ok(version.trim().to_string());
                }
            }
        }

        // Try to get version from directory name
        for entry in std::fs::read_dir("/opt")
            .unwrap_or_else(|_| std::fs::read_dir(".").unwrap())
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("rocm-") {
                if let Some(version) = name.strip_prefix("rocm-") {
                    return Ok(version.to_string());
                }
            }
        }

        Ok("unknown".to_string())
    }

    /// Clones the AITER repository.
    async fn clone_repository(
        &self,
        target_dir: &Path,
        progress: &Option<ProgressCallback>,
    ) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.15, "Cloning AITER repository...".to_string());
        }

        let repo_url = "https://github.com/ROCm/aiter.git";

        // Remove existing directory if present
        if target_dir.exists() {
            tokio::fs::remove_dir_all(target_dir)
                .await
                .context("Failed to remove existing AITER directory")?;
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--recursive",
                "--branch",
                &self.version.git_ref,
                "--depth",
                "1",
                repo_url,
                target_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone AITER repository")?;

        if !output.status.success() {
            // Try without specific branch (use default)
            let output = Command::new("git")
                .args([
                    "clone",
                    "--recursive",
                    "--depth",
                    "1",
                    repo_url,
                    target_dir.to_str().unwrap(),
                ])
                .output()
                .context("Failed to clone AITER repository")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(format!(
                    "Git clone failed: {}",
                    stderr
                ))
                .into());
            }
        }

        if let Some(ref cb) = progress {
            cb(0.25, "Repository cloned successfully".to_string());
        }

        Ok(())
    }

    /// Installs dependencies for AITER.
    async fn install_dependencies(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if !self.build_config.install_deps {
            return Ok(());
        }

        if let Some(ref cb) = progress {
            cb(0.30, "Installing dependencies...".to_string());
        }

        let deps = [
            "pybind11>=2.10.0",
            "pandas>=1.5.0",
            "einops>=0.6.0",
            "packaging>=21.0",
            "psutil>=5.9.0",
            "numpy>=1.20.0",
            "setuptools>=42.0.0",
            "wheel>=0.37.0",
            "typing-extensions>=4.0.0",
        ];

        // Try uv first, fall back to pip
        let installer = if Command::new("uv").arg("--version").output().is_ok() {
            "uv"
        } else {
            "pip"
        };

        for dep in deps {
            let output = if installer == "uv" {
                Command::new("uv")
                    .args(["pip", "install", dep])
                    .envs(self.environment.as_env_map(&self.build_config))
                    .output()
            } else {
                Command::new("python3")
                    .args(["-m", "pip", "install", dep])
                    .envs(self.environment.as_env_map(&self.build_config))
                    .output()
            };

            if let Err(e) = output {
                eprintln!("Warning: Failed to install {}: {}", dep, e);
            }
        }

        if let Some(ref cb) = progress {
            cb(0.40, "Dependencies installed".to_string());
        }

        Ok(())
    }

    /// Builds and installs AITER from source.
    async fn build_from_source(
        &self,
        source_dir: &Path,
        progress: &Option<ProgressCallback>,
    ) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.50, "Building AITER from source...".to_string());
        }

        // Set parallel build jobs
        std::env::set_var("MAX_JOBS", self.build_config.parallel_jobs.to_string());

        // Apply environment
        self.environment.apply();

        // Build install arguments
        let mut args = vec!["-m", "pip", "install"];

        if self.build_config.editable {
            args.push("-e");
        }
        args.push(".");

        if !self.build_config.build_isolation {
            args.push("--no-build-isolation");
        }

        // Add extra args
        for arg in &self.build_config.extra_args {
            args.push(arg);
        }

        if let Some(ref cb) = progress {
            cb(0.60, "Running pip install...".to_string());
        }

        let output = Command::new("python3")
            .args(&args)
            .current_dir(source_dir)
            .envs(self.environment.as_env_map(&self.build_config))
            .output()
            .context("Failed to run pip install")?;

        if !output.status.success() {
            // Try with --no-deps as fallback
            args.push("--no-deps");
            let output = Command::new("python3")
                .args(&args)
                .current_dir(source_dir)
                .envs(self.environment.as_env_map(&self.build_config))
                .output()
                .context("Failed to run pip install with --no-deps")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(format!(
                    "AITER build failed: {}",
                    stderr
                ))
                .into());
            }
        }

        if let Some(ref cb) = progress {
            cb(0.85, "Build completed".to_string());
        }

        Ok(())
    }

    /// Installs from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.3, "Installing AITER from PyPI...".to_string());
        }

        // Apply environment
        self.environment.apply();

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "aiter"])
            .envs(self.environment.as_env_map(&self.build_config))
            .output()
            .context("Failed to install AITER from PyPI")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(format!(
                "PyPI installation failed: {}",
                stderr
            ))
            .into());
        }

        if let Some(ref cb) = progress {
            cb(0.9, "Installation complete".to_string());
        }

        Ok(())
    }

    /// Runs verification tests.
    pub fn verify_installation(&self) -> Result<AiterVerificationResult> {
        // Check basic import
        let basic_import = self.check_installed();
        let mut result = AiterVerificationResult {
            basic_import,
            ..Default::default()
        };

        if !result.basic_import {
            return Ok(result);
        }

        // Get version
        result.version = self.get_installed_version();

        // Check torch module
        result.torch_module = self.has_torch_module();

        // Check GPU detection
        let gpu_check = Command::new("python3")
            .args([
                "-c",
                r#"
import torch
if torch.cuda.is_available():
    print(f"GPU:{torch.cuda.get_device_name(0)}")
    print(f"COUNT:{torch.cuda.device_count()}")
else:
    print("NO_GPU")
"#,
            ])
            .output();

        if let Ok(output) = gpu_check {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.starts_with("GPU:") {
                        result.gpu_name = Some(line.strip_prefix("GPU:").unwrap().to_string());
                        result.gpu_available = true;
                    } else if line.starts_with("COUNT:") {
                        if let Ok(count) = line.strip_prefix("COUNT:").unwrap().parse() {
                            result.gpu_count = count;
                        }
                    }
                }
            }
        }

        // Check RDNA 3 detection
        if result.torch_module {
            let rdna3_check = Command::new("python3")
                .args([
                    "-c",
                    r#"
try:
    from aiter.torch import torch_hip
    print("RDNA3:" + str(torch_hip.is_rdna3_gpu()))
except:
    print("RDNA3:unknown")
"#,
                ])
                .output();

            if let Ok(output) = rdna3_check {
                let stdout = String::from_utf8_lossy(&output.stdout);
                result.rdna3_detected = stdout.contains("RDNA3:True");
            }
        }

        Ok(result)
    }
}

/// Verification result for AITER installation.
#[derive(Debug, Default)]
pub struct AiterVerificationResult {
    /// Basic import works
    pub basic_import: bool,
    /// Installed version
    pub version: Option<String>,
    /// Torch module available
    pub torch_module: bool,
    /// GPU available
    pub gpu_available: bool,
    /// GPU name
    pub gpu_name: Option<String>,
    /// GPU count
    pub gpu_count: usize,
    /// RDNA 3 GPU detected
    pub rdna3_detected: bool,
}

#[async_trait::async_trait]
impl Installer for AiterInstaller {
    fn name(&self) -> &str {
        "AITER"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        // Check Python
        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        // Check pip
        if Command::new("python3")
            .args(["-m", "pip", "--version"])
            .output()
            .is_err()
        {
            checks.push("pip is not installed".to_string());
        }

        // Check ROCm
        match self.check_rocm() {
            Ok(version) => {
                checks.push(format!("ROCm {} detected", version));
            }
            Err(_) => {
                checks.push("ROCm is not installed or not accessible".to_string());
            }
        }

        // Check PyTorch with ROCm
        if !self.check_pytorch_rocm() {
            checks.push("PyTorch with ROCm support is required but not found".to_string());
        }

        // Check git
        if Command::new("git").arg("--version").output().is_err() {
            checks.push("git is not installed".to_string());
        }

        // Check existing installation
        if self.check_installed() {
            if let Some(version) = self.get_installed_version() {
                checks.push(format!("AITER {} is already installed", version));
            } else {
                checks.push("AITER is already installed".to_string());
            }

            if !self.has_torch_module() {
                checks.push("WARNING: Existing AITER may lack ROCm torch module".to_string());
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.05, "Starting AITER installation...".to_string());
        }

        // Apply environment
        self.environment.apply();

        match self.source {
            AiterSource::Pypi => {
                self.install_from_pypi(&progress).await?;
            }
            AiterSource::RocmGithub => {
                // Create temp directory
                let temp_dir = std::env::temp_dir().join("aiter-build");

                // Clone repository
                self.clone_repository(&temp_dir, &progress).await?;

                // Install dependencies
                self.install_dependencies(&progress).await?;

                // Build from source
                self.build_from_source(&temp_dir, &progress).await?;

                // Cleanup
                if let Some(ref cb) = progress {
                    cb(0.95, "Cleaning up...".to_string());
                }
                let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            }
            AiterSource::LocalSource => {
                // Assume current directory contains source
                let source_dir = std::env::current_dir()?;
                self.install_dependencies(&progress).await?;
                self.build_from_source(&source_dir, &progress).await?;
            }
        }

        if let Some(ref cb) = progress {
            cb(1.0, "AITER installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "aiter"])
            .output()
            .context("Failed to uninstall AITER")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(format!(
                "Uninstall failed: {}",
                stderr
            ))
            .into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        let result = self.verify_installation()?;
        Ok(result.basic_import && (result.torch_module || !result.gpu_available))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_architecture_as_str() {
        assert_eq!(GpuArchitecture::Gfx1100.as_str(), "gfx1100");
        assert_eq!(GpuArchitecture::Gfx1101.as_str(), "gfx1101");
        assert_eq!(GpuArchitecture::Gfx1102.as_str(), "gfx1102");
    }

    #[test]
    fn test_aiter_version_creation() {
        let version = AiterVersion::new("0.1.2", "6.0");
        assert_eq!(version.version, "0.1.2");
        assert_eq!(version.git_ref, "v0.1.2");
        assert_eq!(version.min_rocm_version, "6.0");
        assert_eq!(version.target_archs.len(), 3);
    }

    #[test]
    fn test_aiter_version_latest() {
        let version = AiterVersion::latest();
        assert_eq!(version.git_ref, "main");
    }

    #[test]
    fn test_aiter_environment_default() {
        let env = AiterEnvironment::default();
        assert_eq!(env.rocm_path, PathBuf::from("/opt/rocm"));
        assert_eq!(env.hsa_gfx_version, "11.0.0");
        assert!(env.pytorch_rocm_arch.contains("gfx1100"));
    }

    #[test]
    fn test_aiter_environment_as_env_map() {
        let env = AiterEnvironment::default();
        let build = AiterBuildConfig::default();
        let map = env.as_env_map(&build);
        assert!(map.contains_key("ROCM_PATH"));
        assert!(map.contains_key("HSA_OVERRIDE_GFX_VERSION"));
        assert!(map.contains_key("PYTORCH_ROCM_ARCH"));
    }

    #[test]
    fn test_aiter_installer_creation() {
        let version = AiterVersion::new("0.1.2", "6.0");
        let installer = AiterInstaller::new(version, AiterSource::RocmGithub);
        assert_eq!(installer.name(), "AITER");
        assert_eq!(installer.version(), "0.1.2");
    }

    #[test]
    fn test_aiter_installer_builder() {
        let installer = AiterInstaller::latest()
            .with_rocm_path("/custom/rocm")
            .with_parallel_jobs(8);

        assert_eq!(
            installer.environment.rocm_path,
            PathBuf::from("/custom/rocm")
        );
        assert_eq!(installer.build_config.parallel_jobs, 8);
    }

    #[test]
    fn test_build_config_default() {
        let config = AiterBuildConfig::default();
        assert!(!config.build_isolation);
        assert!(config.install_deps);
        assert!(!config.editable);
        assert!(config.parallel_jobs > 0);
    }
}
