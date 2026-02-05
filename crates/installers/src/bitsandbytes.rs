//! BitsAndBytes Installer
//!
//! Pure Rust implementation for installing bitsandbytes with ROCm support.
//! Provides 8-bit and 4-bit quantization for LLM inference and training.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Quantization precision levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationBits {
    /// 8-bit quantization (LLM.int8())
    Int8,
    /// 4-bit quantization (QLoRA compatible)
    Int4,
    /// FP4 quantization
    Fp4,
    /// NF4 quantization (normalized float 4-bit)
    Nf4,
}

impl QuantizationBits {
    /// Returns memory reduction factor compared to FP16.
    pub fn memory_factor(&self) -> f32 {
        match self {
            QuantizationBits::Int8 => 2.0,
            QuantizationBits::Int4 => 4.0,
            QuantizationBits::Fp4 => 4.0,
            QuantizationBits::Nf4 => 4.0,
        }
    }

    /// Returns the config key for transformers library.
    pub fn transformers_key(&self) -> &'static str {
        match self {
            QuantizationBits::Int8 => "load_in_8bit",
            QuantizationBits::Int4 | QuantizationBits::Fp4 | QuantizationBits::Nf4 => "load_in_4bit",
        }
    }
}

/// BitsAndBytes installation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BitsAndBytesSource {
    /// Install from PyPI
    #[default]
    Pypi,
    /// Build from GitHub source
    Github,
    /// Install ROCm-specific fork
    RocmFork,
}

/// BitsAndBytes version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitsAndBytesVersion {
    /// Version string
    pub version: String,
    /// Git reference
    pub git_ref: String,
    /// Supports ROCm natively
    pub rocm_native: bool,
}

impl BitsAndBytesVersion {
    /// Creates a new version configuration.
    pub fn new(version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: format!("v{}", version_str),
            version: version_str,
            rocm_native: true, // Modern versions support ROCm
        }
    }

    /// Latest stable version with ROCm support.
    pub fn latest_stable() -> Self {
        Self {
            version: "0.45.5".to_string(),
            git_ref: "v0.45.5".to_string(),
            rocm_native: true,
        }
    }

    /// Development version.
    pub fn development() -> Self {
        Self {
            version: "dev".to_string(),
            git_ref: "main".to_string(),
            rocm_native: true,
        }
    }
}

/// Build configuration for BitsAndBytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitsAndBytesBuildConfig {
    /// Target GPU architectures
    pub gpu_archs: Vec<String>,
    /// Use HIP/ROCm backend
    pub use_hip: bool,
    /// Number of parallel jobs
    pub parallel_jobs: usize,
    /// Build with CUDA compatibility layer
    pub cuda_compat: bool,
    /// Extra compile flags
    pub extra_cflags: Vec<String>,
}

impl Default for BitsAndBytesBuildConfig {
    fn default() -> Self {
        Self {
            gpu_archs: vec![
                "gfx1100".to_string(),
                "gfx1101".to_string(),
                "gfx1102".to_string(),
            ],
            use_hip: true,
            parallel_jobs: num_cpus::get(),
            cuda_compat: true,
            extra_cflags: Vec::new(),
        }
    }
}

/// ROCm environment for BitsAndBytes.
#[derive(Debug, Clone)]
pub struct BitsAndBytesEnvironment {
    /// ROCm path
    pub rocm_path: PathBuf,
    /// HIP compiler path
    pub hipcc_path: PathBuf,
    /// HSA GFX version override
    pub hsa_gfx_version: String,
    /// BNB CUDA version (used even for ROCm)
    pub bnb_cuda_version: String,
}

impl Default for BitsAndBytesEnvironment {
    fn default() -> Self {
        let rocm_path = PathBuf::from("/opt/rocm");
        Self {
            hipcc_path: rocm_path.join("bin/hipcc"),
            rocm_path,
            hsa_gfx_version: "11.0.0".to_string(),
            bnb_cuda_version: "118".to_string(), // HIP compatibility
        }
    }
}

impl BitsAndBytesEnvironment {
    /// Returns environment map for subprocess.
    pub fn as_env_map(&self, build_config: &BitsAndBytesBuildConfig) -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert("ROCM_HOME".to_string(), self.rocm_path.to_string_lossy().to_string());
        env.insert("ROCM_PATH".to_string(), self.rocm_path.to_string_lossy().to_string());
        env.insert("HSA_OVERRIDE_GFX_VERSION".to_string(), self.hsa_gfx_version.clone());
        env.insert("BNB_CUDA_VERSION".to_string(), self.bnb_cuda_version.clone());

        if build_config.use_hip {
            env.insert("CC".to_string(), self.hipcc_path.to_string_lossy().to_string());
            env.insert("CXX".to_string(), self.hipcc_path.to_string_lossy().to_string());
        }

        // Set GPU architecture targets
        let arch_str = build_config.gpu_archs.join(";");
        env.insert("PYTORCH_ROCM_ARCH".to_string(), arch_str.clone());
        env.insert("HIP_ARCHITECTURES".to_string(), arch_str);

        // Add ROCm to paths
        if let Ok(path) = std::env::var("PATH") {
            let rocm_bin = self.rocm_path.join("bin");
            env.insert("PATH".to_string(), format!("{}:{}", rocm_bin.display(), path));
        }

        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            let rocm_lib = self.rocm_path.join("lib");
            env.insert("LD_LIBRARY_PATH".to_string(), format!("{}:{}", rocm_lib.display(), ld_path));
        }

        env.insert("MAX_JOBS".to_string(), build_config.parallel_jobs.to_string());

        env
    }
}

/// BitsAndBytes installer.
pub struct BitsAndBytesInstaller {
    version: BitsAndBytesVersion,
    source: BitsAndBytesSource,
    build_config: BitsAndBytesBuildConfig,
    environment: BitsAndBytesEnvironment,
}

impl BitsAndBytesInstaller {
    /// Creates a new installer.
    pub fn new(version: BitsAndBytesVersion, source: BitsAndBytesSource) -> Self {
        Self {
            version,
            source,
            build_config: BitsAndBytesBuildConfig::default(),
            environment: BitsAndBytesEnvironment::default(),
        }
    }

    /// Creates installer for latest PyPI version.
    pub fn from_pypi() -> Self {
        Self::new(BitsAndBytesVersion::latest_stable(), BitsAndBytesSource::Pypi)
    }

    /// Creates installer from GitHub.
    pub fn from_github() -> Self {
        Self::new(BitsAndBytesVersion::development(), BitsAndBytesSource::Github)
    }

    /// Creates installer from ROCm fork.
    pub fn from_rocm_fork() -> Self {
        Self::new(BitsAndBytesVersion::latest_stable(), BitsAndBytesSource::RocmFork)
    }

    /// Sets ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        let rocm_path = path.into();
        self.environment.hipcc_path = rocm_path.join("bin/hipcc");
        self.environment.rocm_path = rocm_path;
        self
    }

    /// Sets build configuration.
    pub fn with_build_config(mut self, config: BitsAndBytesBuildConfig) -> Self {
        self.build_config = config;
        self
    }

    /// Checks if bitsandbytes is installed.
    fn check_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import bitsandbytes; print(bitsandbytes.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import bitsandbytes; print(bitsandbytes.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Checks if CUDA/ROCm is available for bitsandbytes.
    #[allow(dead_code)]
    fn check_cuda_available(&self) -> bool {
        Command::new("python3")
            .args(["-c", r#"
import bitsandbytes as bnb
from bitsandbytes.cuda_setup.main import evaluate_cuda_setup
evaluate_cuda_setup()
print("OK")
"#])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Installs from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.2, "Installing bitsandbytes from PyPI...".to_string());
        }

        let env = self.environment.as_env_map(&self.build_config);

        // For ROCm, we might need the rocm wheel
        let package = format!("bitsandbytes=={}", self.version.version);

        let output = Command::new("python3")
            .args(["-m", "pip", "install", &package])
            .envs(&env)
            .output()
            .context("Failed to install bitsandbytes")?;

        if !output.status.success() {
            // Try without version constraint
            let output = Command::new("python3")
                .args(["-m", "pip", "install", "bitsandbytes"])
                .envs(&env)
                .output()
                .context("Failed to install bitsandbytes")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(
                    format!("Installation failed: {}", stderr)
                ).into());
            }
        }

        if let Some(ref cb) = progress {
            cb(0.8, "Installation complete".to_string());
        }

        Ok(())
    }

    /// Clones repository.
    async fn clone_repository(&self, target_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.15, "Cloning bitsandbytes repository...".to_string());
        }

        let repo_url = match self.source {
            BitsAndBytesSource::RocmFork => "https://github.com/ROCm/bitsandbytes.git",
            _ => "https://github.com/bitsandbytes-foundation/bitsandbytes.git",
        };

        if target_dir.exists() {
            tokio::fs::remove_dir_all(target_dir).await
                .context("Failed to remove existing directory")?;
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch", &self.version.git_ref,
                "--depth", "1",
                repo_url,
                target_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone repository")?;

        if !output.status.success() {
            // Try default branch
            let output = Command::new("git")
                .args([
                    "clone",
                    "--depth", "1",
                    repo_url,
                    target_dir.to_str().unwrap(),
                ])
                .output()
                .context("Failed to clone repository")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(
                    format!("Git clone failed: {}", stderr)
                ).into());
            }
        }

        Ok(())
    }

    /// Builds from source.
    async fn build_from_source(&self, source_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.4, "Building bitsandbytes from source...".to_string());
        }

        let env = self.environment.as_env_map(&self.build_config);

        // Build the library
        if let Some(ref cb) = progress {
            cb(0.5, "Compiling HIP kernels...".to_string());
        }

        // Check for Makefile or setup.py
        let makefile = source_dir.join("Makefile");
        if makefile.exists() && self.build_config.use_hip {
            let output = Command::new("make")
                .args(["hip"])
                .current_dir(source_dir)
                .envs(&env)
                .output()
                .context("Failed to run make")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("Make warning: {}", stderr);
            }
        }

        if let Some(ref cb) = progress {
            cb(0.7, "Installing Python package...".to_string());
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "-e", "."])
            .current_dir(source_dir)
            .envs(&env)
            .output()
            .context("Failed to install bitsandbytes")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.9, "Build complete".to_string());
        }

        Ok(())
    }

    /// Runs verification.
    pub fn verify_installation(&self) -> Result<BitsAndBytesVerification> {
        let installed = self.check_installed();
        let mut result = BitsAndBytesVerification {
            installed,
            ..Default::default()
        };

        if !result.installed {
            return Ok(result);
        }

        result.version = self.get_installed_version();

        // Check 8-bit support
        let int8_check = Command::new("python3")
            .args(["-c", r#"
import torch
import bitsandbytes as bnb
linear = bnb.nn.Linear8bitLt(128, 64)
print("INT8:OK")
"#])
            .output();

        result.int8_supported = int8_check.map(|o| o.status.success()).unwrap_or(false);

        // Check 4-bit support
        let int4_check = Command::new("python3")
            .args(["-c", r#"
import torch
import bitsandbytes as bnb
linear = bnb.nn.Linear4bit(128, 64)
print("INT4:OK")
"#])
            .output();

        result.int4_supported = int4_check.map(|o| o.status.success()).unwrap_or(false);

        // Check GPU
        let gpu_check = Command::new("python3")
            .args(["-c", r#"
import torch
print(f"GPU:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"NAME:{torch.cuda.get_device_name(0)}")
"#])
            .output();

        if let Ok(output) = gpu_check {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                result.gpu_available = stdout.contains("GPU:True");
                for line in stdout.lines() {
                    if line.starts_with("NAME:") {
                        result.gpu_name = Some(line.strip_prefix("NAME:").unwrap().to_string());
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Verification result.
#[derive(Debug, Default)]
pub struct BitsAndBytesVerification {
    /// Is installed
    pub installed: bool,
    /// Version
    pub version: Option<String>,
    /// 8-bit quantization supported
    pub int8_supported: bool,
    /// 4-bit quantization supported
    pub int4_supported: bool,
    /// GPU available
    pub gpu_available: bool,
    /// GPU name
    pub gpu_name: Option<String>,
}

#[async_trait::async_trait]
impl Installer for BitsAndBytesInstaller {
    fn name(&self) -> &str {
        "BitsAndBytes"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        if self.environment.rocm_path.exists() {
            checks.push("ROCm is installed".to_string());
        } else {
            checks.push("ROCm is not found".to_string());
        }

        if self.environment.hipcc_path.exists() {
            checks.push("HIP compiler available".to_string());
        } else {
            checks.push("HIP compiler not found".to_string());
        }

        if self.check_installed() {
            if let Some(version) = self.get_installed_version() {
                checks.push(format!("bitsandbytes {} already installed", version));
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.05, "Starting bitsandbytes installation...".to_string());
        }

        match self.source {
            BitsAndBytesSource::Pypi => {
                self.install_from_pypi(&progress).await?;
            }
            BitsAndBytesSource::Github | BitsAndBytesSource::RocmFork => {
                let temp_dir = std::env::temp_dir().join("bitsandbytes-build");
                self.clone_repository(&temp_dir, &progress).await?;
                self.build_from_source(&temp_dir, &progress).await?;
                let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            }
        }

        if let Some(ref cb) = progress {
            cb(1.0, "bitsandbytes installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "bitsandbytes"])
            .output()
            .context("Failed to uninstall")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Uninstall failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        let result = self.verify_installation()?;
        Ok(result.installed && (result.int8_supported || result.int4_supported))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_bits() {
        assert_eq!(QuantizationBits::Int8.memory_factor(), 2.0);
        assert_eq!(QuantizationBits::Int4.memory_factor(), 4.0);
        assert_eq!(QuantizationBits::Nf4.transformers_key(), "load_in_4bit");
    }

    #[test]
    fn test_version_creation() {
        let version = BitsAndBytesVersion::new("0.45.0");
        assert_eq!(version.version, "0.45.0");
        assert!(version.rocm_native);
    }

    #[test]
    fn test_build_config_default() {
        let config = BitsAndBytesBuildConfig::default();
        assert!(config.use_hip);
        assert!(config.gpu_archs.contains(&"gfx1100".to_string()));
    }

    #[test]
    fn test_installer_creation() {
        let installer = BitsAndBytesInstaller::from_pypi();
        assert_eq!(installer.name(), "BitsAndBytes");
    }
}
