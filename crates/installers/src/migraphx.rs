//! MIGraphX Installer
//!
//! Pure Rust implementation for installing AMD MIGraphX,
//! the graph optimization and inference library for AMD GPUs.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// MIGraphX optimization target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizationTarget {
    /// General inference optimization
    #[default]
    Inference,
    /// Low latency optimization
    LowLatency,
    /// High throughput optimization
    HighThroughput,
    /// Memory optimized
    MemoryOptimized,
}

/// Supported model formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX format
    Onnx,
    /// TensorFlow frozen graph
    TensorFlow,
    /// MIGraphX native format
    Migraphx,
}

/// MIGraphX installation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MigraphxSource {
    /// Install via package manager
    #[default]
    PackageManager,
    /// Build from source
    Source,
    /// Already bundled with ROCm
    RocmBundle,
}

/// MIGraphX version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigraphxVersion {
    /// Version string
    pub version: String,
    /// Git reference
    pub git_ref: String,
    /// Minimum ROCm version
    pub min_rocm_version: String,
}

impl MigraphxVersion {
    /// Creates a new version.
    pub fn new(version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: format!("rocm-{}", version_str),
            version: version_str.clone(),
            min_rocm_version: version_str,
        }
    }

    /// Latest stable version.
    pub fn latest() -> Self {
        Self {
            version: "2.12.0".to_string(),
            git_ref: "rocm-6.4".to_string(),
            min_rocm_version: "6.0".to_string(),
        }
    }
}

/// Build configuration for MIGraphX.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigraphxBuildConfig {
    /// Target GPU architectures
    pub gpu_archs: Vec<String>,
    /// Enable Python bindings
    pub python_bindings: bool,
    /// Enable ONNX support
    pub onnx_support: bool,
    /// Enable TensorFlow support
    pub tensorflow_support: bool,
    /// Number of parallel jobs
    pub parallel_jobs: usize,
    /// Build type (Release, Debug)
    pub build_type: String,
}

impl Default for MigraphxBuildConfig {
    fn default() -> Self {
        Self {
            gpu_archs: vec![
                "gfx1100".to_string(),
                "gfx1101".to_string(),
                "gfx1102".to_string(),
            ],
            python_bindings: true,
            onnx_support: true,
            tensorflow_support: false,
            parallel_jobs: num_cpus::get(),
            build_type: "Release".to_string(),
        }
    }
}

/// MIGraphX environment.
#[derive(Debug, Clone)]
pub struct MigraphxEnvironment {
    /// ROCm path
    pub rocm_path: PathBuf,
    /// MIGraphX installation path
    pub migraphx_path: PathBuf,
    /// HSA GFX version
    pub hsa_gfx_version: String,
}

impl Default for MigraphxEnvironment {
    fn default() -> Self {
        let rocm_path = PathBuf::from("/opt/rocm");
        Self {
            migraphx_path: rocm_path.clone(),
            rocm_path,
            hsa_gfx_version: "11.0.0".to_string(),
        }
    }
}

impl MigraphxEnvironment {
    /// Returns environment map.
    pub fn as_env_map(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert("ROCM_PATH".to_string(), self.rocm_path.to_string_lossy().to_string());
        env.insert("HSA_OVERRIDE_GFX_VERSION".to_string(), self.hsa_gfx_version.clone());
        env.insert("MIGRAPHX_DISABLE_FAST_GELU".to_string(), "0".to_string());

        // Add to paths
        if let Ok(path) = std::env::var("PATH") {
            let rocm_bin = self.rocm_path.join("bin");
            env.insert("PATH".to_string(), format!("{}:{}", rocm_bin.display(), path));
        }

        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            let rocm_lib = self.rocm_path.join("lib");
            let migraphx_lib = self.migraphx_path.join("lib");
            env.insert("LD_LIBRARY_PATH".to_string(),
                format!("{}:{}:{}", rocm_lib.display(), migraphx_lib.display(), ld_path));
        }

        env
    }
}

/// MIGraphX installer.
pub struct MigraphxInstaller {
    version: MigraphxVersion,
    source: MigraphxSource,
    build_config: MigraphxBuildConfig,
    environment: MigraphxEnvironment,
}

impl MigraphxInstaller {
    /// Creates a new installer.
    pub fn new(version: MigraphxVersion, source: MigraphxSource) -> Self {
        Self {
            version,
            source,
            build_config: MigraphxBuildConfig::default(),
            environment: MigraphxEnvironment::default(),
        }
    }

    /// Creates installer for package manager.
    pub fn from_package_manager() -> Self {
        Self::new(MigraphxVersion::latest(), MigraphxSource::PackageManager)
    }

    /// Creates installer from source.
    pub fn from_source() -> Self {
        Self::new(MigraphxVersion::latest(), MigraphxSource::Source)
    }

    /// Sets ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        let rocm_path = path.into();
        self.environment.migraphx_path = rocm_path.clone();
        self.environment.rocm_path = rocm_path;
        self
    }

    /// Sets build configuration.
    pub fn with_build_config(mut self, config: MigraphxBuildConfig) -> Self {
        self.build_config = config;
        self
    }

    /// Checks if MIGraphX is installed.
    fn check_installed(&self) -> bool {
        // Check for migraphx-driver binary
        let driver_path = self.environment.rocm_path.join("bin/migraphx-driver");
        if driver_path.exists() {
            return true;
        }

        // Check Python import
        Command::new("python3")
            .args(["-c", "import migraphx; print(migraphx.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        // Try Python import first
        if let Ok(output) = Command::new("python3")
            .args(["-c", "import migraphx; print(migraphx.__version__)"])
            .output()
        {
            if output.status.success() {
                return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }

        // Try migraphx-driver
        if let Ok(output) = Command::new("migraphx-driver")
            .arg("--version")
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Some(stdout.lines().next().unwrap_or("unknown").to_string());
            }
        }

        None
    }

    /// Detects package manager.
    fn detect_package_manager(&self) -> Option<&'static str> {
        if Command::new("apt-get").arg("--version").output().is_ok() {
            Some("apt")
        } else if Command::new("dnf").arg("--version").output().is_ok() {
            Some("dnf")
        } else if Command::new("yum").arg("--version").output().is_ok() {
            Some("yum")
        } else {
            None
        }
    }

    /// Installs via package manager.
    async fn install_via_package_manager(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.2, "Installing MIGraphX via package manager...".to_string());
        }

        let pm = self.detect_package_manager()
            .ok_or_else(|| InstallerError::UnsupportedPackageManager("No supported package manager found".to_string()))?;

        let packages = ["migraphx", "migraphx-dev"];

        for package in packages {
            if let Some(ref cb) = progress {
                cb(0.4, format!("Installing {}...", package));
            }

            let result = match pm {
                "apt" => {
                    Command::new("sudo")
                        .args(["apt-get", "install", "-y", package])
                        .output()
                }
                "dnf" | "yum" => {
                    Command::new("sudo")
                        .args([pm, "install", "-y", package])
                        .output()
                }
                _ => continue,
            };

            if let Ok(output) = result {
                if !output.status.success() {
                    eprintln!("Warning: Failed to install {}", package);
                }
            }
        }

        // Install Python bindings
        if self.build_config.python_bindings {
            if let Some(ref cb) = progress {
                cb(0.7, "Installing Python bindings...".to_string());
            }

            let _ = Command::new("python3")
                .args(["-m", "pip", "install", "migraphx"])
                .envs(self.environment.as_env_map())
                .output();
        }

        Ok(())
    }

    /// Clones repository.
    async fn clone_repository(&self, target_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.15, "Cloning MIGraphX repository...".to_string());
        }

        let repo_url = "https://github.com/ROCm/AMDMIGraphX.git";

        if target_dir.exists() {
            tokio::fs::remove_dir_all(target_dir).await
                .context("Failed to remove existing directory")?;
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch", &self.version.git_ref,
                "--depth", "1",
                "--recursive",
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
                    "--recursive",
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
            cb(0.3, "Configuring MIGraphX build...".to_string());
        }

        let build_dir = source_dir.join("build");
        tokio::fs::create_dir_all(&build_dir).await
            .context("Failed to create build directory")?;

        let mut env = self.environment.as_env_map();
        env.insert("CMAKE_PREFIX_PATH".to_string(), self.environment.rocm_path.to_string_lossy().to_string());

        // Configure with CMake
        let arch_str = self.build_config.gpu_archs.join(";");
        let mut cmake_args = vec![
            format!("-DCMAKE_BUILD_TYPE={}", self.build_config.build_type),
            format!("-DGPU_TARGETS={}", arch_str),
            format!("-DCMAKE_INSTALL_PREFIX={}", self.environment.migraphx_path.display()),
        ];

        if self.build_config.python_bindings {
            cmake_args.push("-DMIGRAPHX_ENABLE_PYTHON=ON".to_string());
        }

        cmake_args.push("..".to_string());

        let output = Command::new("cmake")
            .args(&cmake_args)
            .current_dir(&build_dir)
            .envs(&env)
            .output()
            .context("Failed to run cmake")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("CMake configuration failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.5, "Building MIGraphX (this may take a while)...".to_string());
        }

        // Build
        let output = Command::new("cmake")
            .args(["--build", ".", "--parallel", &self.build_config.parallel_jobs.to_string()])
            .current_dir(&build_dir)
            .envs(&env)
            .output()
            .context("Failed to build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(0.8, "Installing MIGraphX...".to_string());
        }

        // Install
        let output = Command::new("sudo")
            .args(["cmake", "--install", "."])
            .current_dir(&build_dir)
            .envs(&env)
            .output()
            .context("Failed to install")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Install failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Runs model optimization.
    pub fn optimize_model(&self, input_path: &Path, output_path: &Path, format: ModelFormat) -> Result<()> {
        let format_arg = match format {
            ModelFormat::Onnx => "--onnx",
            ModelFormat::TensorFlow => "--tf",
            ModelFormat::Migraphx => "--migraphx",
        };

        let output = Command::new("migraphx-driver")
            .args([
                "compile",
                format_arg, input_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
            ])
            .envs(self.environment.as_env_map())
            .output()
            .context("Failed to run migraphx-driver")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Model optimization failed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Runs verification tests.
    pub fn verify_installation(&self) -> Result<MigraphxVerification> {
        let installed = self.check_installed();
        let mut result = MigraphxVerification {
            installed,
            ..Default::default()
        };

        if !result.installed {
            return Ok(result);
        }

        result.version = self.get_installed_version();

        // Check Python bindings
        let python_check = Command::new("python3")
            .args(["-c", r#"
import migraphx
print("PYTHON:OK")
prog = migraphx.program()
print("PROGRAM:OK")
"#])
            .output();

        if let Ok(output) = python_check {
            let stdout = String::from_utf8_lossy(&output.stdout);
            result.python_bindings = stdout.contains("PYTHON:OK");
            result.basic_ops = stdout.contains("PROGRAM:OK");
        }

        // Check GPU detection
        let gpu_check = Command::new("migraphx-driver")
            .args(["--gpu"])
            .output();

        result.gpu_available = gpu_check.map(|o| o.status.success()).unwrap_or(false);

        Ok(result)
    }
}

/// MIGraphX verification result.
#[derive(Debug, Default)]
pub struct MigraphxVerification {
    /// Is installed
    pub installed: bool,
    /// Version
    pub version: Option<String>,
    /// Python bindings available
    pub python_bindings: bool,
    /// Basic operations work
    pub basic_ops: bool,
    /// GPU available
    pub gpu_available: bool,
}

#[async_trait::async_trait]
impl Installer for MigraphxInstaller {
    fn name(&self) -> &str {
        "MIGraphX"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if self.environment.rocm_path.exists() {
            checks.push(format!("ROCm found at {}", self.environment.rocm_path.display()));
        } else {
            checks.push("ROCm not found".to_string());
        }

        if self.check_installed() {
            if let Some(version) = self.get_installed_version() {
                checks.push(format!("MIGraphX {} already installed", version));
            }
        }

        if let Some(pm) = self.detect_package_manager() {
            checks.push(format!("Package manager: {}", pm));
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.05, "Starting MIGraphX installation...".to_string());
        }

        match self.source {
            MigraphxSource::PackageManager | MigraphxSource::RocmBundle => {
                self.install_via_package_manager(&progress).await?;
            }
            MigraphxSource::Source => {
                let temp_dir = std::env::temp_dir().join("migraphx-build");
                self.clone_repository(&temp_dir, &progress).await?;
                self.build_from_source(&temp_dir, &progress).await?;
                let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            }
        }

        if let Some(ref cb) = progress {
            cb(1.0, "MIGraphX installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        // Remove Python bindings
        let _ = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "migraphx"])
            .output();

        // Package manager uninstall
        if let Some(pm) = self.detect_package_manager() {
            match pm {
                "apt" => {
                    let _ = Command::new("sudo")
                        .args(["apt-get", "remove", "-y", "migraphx", "migraphx-dev"])
                        .output();
                }
                "dnf" | "yum" => {
                    let _ = Command::new("sudo")
                        .args([pm, "remove", "-y", "migraphx", "migraphx-devel"])
                        .output();
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        let result = self.verify_installation()?;
        Ok(result.installed && result.python_bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = MigraphxVersion::new("6.4");
        assert_eq!(version.version, "6.4");
        assert!(version.git_ref.contains("rocm"));
    }

    #[test]
    fn test_installer_creation() {
        let installer = MigraphxInstaller::from_package_manager();
        assert_eq!(installer.name(), "MIGraphX");
    }

    #[test]
    fn test_build_config_default() {
        let config = MigraphxBuildConfig::default();
        assert!(config.python_bindings);
        assert!(config.onnx_support);
        assert!(config.gpu_archs.contains(&"gfx1100".to_string()));
    }
}
