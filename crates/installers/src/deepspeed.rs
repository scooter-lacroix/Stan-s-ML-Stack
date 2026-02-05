//! DeepSpeed Installer
//!
//! Pure Rust implementation for installing DeepSpeed with ROCm support.
//! Provides ZeRO optimization stages, distributed training configuration,
//! and comprehensive AMD GPU integration.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// ZeRO optimization stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ZeroStage {
    /// No ZeRO optimization
    #[default]
    Disabled,
    /// ZeRO Stage 1: Optimizer state partitioning
    Stage1,
    /// ZeRO Stage 2: Gradient partitioning
    Stage2,
    /// ZeRO Stage 3: Parameter partitioning
    Stage3,
}

impl ZeroStage {
    /// Returns the stage number.
    pub fn as_int(&self) -> u8 {
        match self {
            ZeroStage::Disabled => 0,
            ZeroStage::Stage1 => 1,
            ZeroStage::Stage2 => 2,
            ZeroStage::Stage3 => 3,
        }
    }

    /// Returns memory savings description.
    pub fn memory_savings(&self) -> &'static str {
        match self {
            ZeroStage::Disabled => "None",
            ZeroStage::Stage1 => "~4x reduction in optimizer memory",
            ZeroStage::Stage2 => "~8x reduction in memory (optimizer + gradients)",
            ZeroStage::Stage3 => "Linear scaling with data parallelism",
        }
    }
}

/// DeepSpeed installation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DeepSpeedSource {
    /// Install from PyPI
    #[default]
    Pypi,
    /// Build from GitHub source
    Github,
    /// Build from Microsoft's ROCm fork
    MicrosoftRocm,
}

/// DeepSpeed version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSpeedVersion {
    /// Version string
    pub version: String,
    /// Git reference for source builds
    pub git_ref: String,
    /// Minimum PyTorch version required
    pub min_pytorch_version: String,
    /// Minimum ROCm version required
    pub min_rocm_version: String,
}

impl DeepSpeedVersion {
    /// Creates a new version configuration.
    pub fn new(version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: format!("v{}", version_str),
            version: version_str,
            min_pytorch_version: "1.13.0".to_string(),
            min_rocm_version: "5.4".to_string(),
        }
    }

    /// Creates latest development version.
    pub fn latest() -> Self {
        Self {
            version: "dev".to_string(),
            git_ref: "master".to_string(),
            min_pytorch_version: "2.0.0".to_string(),
            min_rocm_version: "6.0".to_string(),
        }
    }

    /// Sets custom git reference.
    pub fn with_git_ref(mut self, git_ref: impl Into<String>) -> Self {
        self.git_ref = git_ref.into();
        self
    }
}

/// DeepSpeed features/ops to build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSpeedOps {
    /// Build sparse attention ops
    pub sparse_attn: bool,
    /// Build transformer inference ops
    pub transformer_inference: bool,
    /// Build fused Adam optimizer
    pub fused_adam: bool,
    /// Build fused LAMB optimizer
    pub fused_lamb: bool,
    /// Build CPU Adam optimizer
    pub cpu_adam: bool,
    /// Build CPU AdaGrad optimizer
    pub cpu_adagrad: bool,
    /// Build async IO ops
    pub async_io: bool,
    /// Build quantization ops
    pub quantizer: bool,
    /// Build random LTD ops
    pub random_ltd: bool,
}

impl Default for DeepSpeedOps {
    fn default() -> Self {
        Self {
            sparse_attn: true,
            transformer_inference: true,
            fused_adam: true,
            fused_lamb: true,
            cpu_adam: true,
            cpu_adagrad: false,
            async_io: true,
            quantizer: true,
            random_ltd: false,
        }
    }
}

impl DeepSpeedOps {
    /// Returns environment variable string for building ops.
    pub fn as_build_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        macro_rules! set_op {
            ($name:ident, $env_var:expr) => {
                env.insert(
                    $env_var.to_string(),
                    if self.$name { "1" } else { "0" }.to_string(),
                );
            };
        }

        set_op!(sparse_attn, "DS_BUILD_SPARSE_ATTN");
        set_op!(transformer_inference, "DS_BUILD_TRANSFORMER_INFERENCE");
        set_op!(fused_adam, "DS_BUILD_FUSED_ADAM");
        set_op!(fused_lamb, "DS_BUILD_FUSED_LAMB");
        set_op!(cpu_adam, "DS_BUILD_CPU_ADAM");
        set_op!(cpu_adagrad, "DS_BUILD_CPU_ADAGRAD");
        set_op!(async_io, "DS_BUILD_AIO");
        set_op!(quantizer, "DS_BUILD_QUANTIZER");
        set_op!(random_ltd, "DS_BUILD_RANDOM_LTD");

        env
    }
}

/// DeepSpeed build configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSpeedBuildConfig {
    /// Operations to build
    pub ops: DeepSpeedOps,
    /// Number of parallel build jobs
    pub parallel_jobs: usize,
    /// Use ninja build system
    pub use_ninja: bool,
    /// Build for specific GPU architectures
    pub gpu_archs: Vec<String>,
    /// Extra CMake arguments
    pub cmake_args: Vec<String>,
}

impl Default for DeepSpeedBuildConfig {
    fn default() -> Self {
        Self {
            ops: DeepSpeedOps::default(),
            parallel_jobs: num_cpus::get(),
            use_ninja: true,
            gpu_archs: vec![
                "gfx1100".to_string(),
                "gfx1101".to_string(),
                "gfx1102".to_string(),
            ],
            cmake_args: Vec::new(),
        }
    }
}

/// ROCm environment configuration.
#[derive(Debug, Clone)]
pub struct DeepSpeedEnvironment {
    /// ROCm installation path
    pub rocm_path: PathBuf,
    /// HIP platform selection
    pub hip_platform: String,
    /// HSA GFX version override
    pub hsa_gfx_version: String,
    /// PyTorch ROCm architecture
    pub pytorch_rocm_arch: String,
    /// Use ROCm backend
    pub use_rocm: bool,
}

impl Default for DeepSpeedEnvironment {
    fn default() -> Self {
        Self {
            rocm_path: PathBuf::from("/opt/rocm"),
            hip_platform: "amd".to_string(),
            hsa_gfx_version: "11.0.0".to_string(),
            pytorch_rocm_arch: "gfx1100;gfx1101;gfx1102".to_string(),
            use_rocm: true,
        }
    }
}

impl DeepSpeedEnvironment {
    /// Returns environment variables for subprocess.
    pub fn as_env_map(&self, _build_config: &DeepSpeedBuildConfig) -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert(
            "ROCM_PATH".to_string(),
            self.rocm_path.to_string_lossy().to_string(),
        );
        env.insert(
            "ROCM_HOME".to_string(),
            self.rocm_path.to_string_lossy().to_string(),
        );
        env.insert("HIP_PLATFORM".to_string(), self.hip_platform.clone());
        env.insert(
            "HSA_OVERRIDE_GFX_VERSION".to_string(),
            self.hsa_gfx_version.clone(),
        );
        env.insert(
            "PYTORCH_ROCM_ARCH".to_string(),
            self.pytorch_rocm_arch.clone(),
        );

        if self.use_rocm {
            env.insert("DS_ACCELERATOR".to_string(), "cuda".to_string()); // DeepSpeed uses 'cuda' for ROCm too
        }

        // Add ROCm paths
        let rocm_bin = self.rocm_path.join("bin");
        if let Ok(path) = std::env::var("PATH") {
            env.insert(
                "PATH".to_string(),
                format!("{}:{}", rocm_bin.display(), path),
            );
        }

        let rocm_lib = self.rocm_path.join("lib");
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            env.insert(
                "LD_LIBRARY_PATH".to_string(),
                format!("{}:{}", rocm_lib.display(), ld_path),
            );
        }

        env
    }

    /// Applies environment to current process.
    pub fn apply(&self) {
        let build_config = DeepSpeedBuildConfig::default();
        for (key, value) in self.as_env_map(&build_config) {
            std::env::set_var(key, value);
        }
    }
}

/// DeepSpeed installer.
pub struct DeepSpeedInstaller {
    version: DeepSpeedVersion,
    source: DeepSpeedSource,
    build_config: DeepSpeedBuildConfig,
    environment: DeepSpeedEnvironment,
}

impl DeepSpeedInstaller {
    /// Creates a new installer.
    pub fn new(version: DeepSpeedVersion, source: DeepSpeedSource) -> Self {
        Self {
            version,
            source,
            build_config: DeepSpeedBuildConfig::default(),
            environment: DeepSpeedEnvironment::default(),
        }
    }

    /// Creates installer for latest PyPI version.
    pub fn from_pypi() -> Self {
        Self::new(DeepSpeedVersion::new("0.14.0"), DeepSpeedSource::Pypi)
    }

    /// Creates installer for latest GitHub version.
    pub fn from_github() -> Self {
        Self::new(DeepSpeedVersion::latest(), DeepSpeedSource::Github)
    }

    /// Sets ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.environment.rocm_path = path.into();
        self
    }

    /// Sets build configuration.
    pub fn with_build_config(mut self, config: DeepSpeedBuildConfig) -> Self {
        self.build_config = config;
        self
    }

    /// Sets parallel build jobs.
    pub fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.build_config.parallel_jobs = jobs;
        self
    }

    /// Enables/disables specific ops.
    pub fn with_ops(mut self, ops: DeepSpeedOps) -> Self {
        self.build_config.ops = ops;
        self
    }

    /// Checks if DeepSpeed is installed.
    fn check_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import deepspeed; print(deepspeed.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import deepspeed; print(deepspeed.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Checks PyTorch ROCm availability.
    fn check_pytorch_rocm(&self) -> bool {
        Command::new("python3")
            .args([
                "-c",
                "import torch; assert torch.cuda.is_available(); print(torch.version.hip)",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks ROCm installation.
    fn check_rocm(&self) -> bool {
        self.environment.rocm_path.exists()
            && Command::new("rocminfo")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }

    /// Installs dependencies.
    async fn install_dependencies(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.15, "Installing dependencies...".to_string());
        }

        let deps = ["packaging", "ninja", "pydantic", "jsonschema", "py-cpuinfo"];

        for dep in deps {
            let _ = Command::new("python3")
                .args(["-m", "pip", "install", dep])
                .envs(self.environment.as_env_map(&self.build_config))
                .output();
        }

        Ok(())
    }

    /// Installs from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.3, "Installing DeepSpeed from PyPI...".to_string());
        }

        self.environment.apply();

        let mut env = self.environment.as_env_map(&self.build_config);
        env.extend(self.build_config.ops.as_build_env());

        let output = Command::new("python3")
            .args([
                "-m",
                "pip",
                "install",
                &format!("deepspeed=={}", self.version.version),
            ])
            .envs(&env)
            .output()
            .context("Failed to run pip install")?;

        if !output.status.success() {
            // Try without version constraint
            let output = Command::new("python3")
                .args(["-m", "pip", "install", "deepspeed"])
                .envs(&env)
                .output()
                .context("Failed to install deepspeed")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(format!(
                    "DeepSpeed installation failed: {}",
                    stderr
                ))
                .into());
            }
        }

        if let Some(ref cb) = progress {
            cb(0.9, "Installation complete".to_string());
        }

        Ok(())
    }

    /// Clones GitHub repository.
    async fn clone_repository(
        &self,
        target_dir: &Path,
        progress: &Option<ProgressCallback>,
    ) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.2, "Cloning DeepSpeed repository...".to_string());
        }

        let repo_url = match self.source {
            DeepSpeedSource::MicrosoftRocm => "https://github.com/microsoft/DeepSpeed.git",
            _ => "https://github.com/microsoft/DeepSpeed.git",
        };

        if target_dir.exists() {
            tokio::fs::remove_dir_all(target_dir)
                .await
                .context("Failed to remove existing directory")?;
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch",
                &self.version.git_ref,
                "--depth",
                "1",
                repo_url,
                target_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone repository")?;

        if !output.status.success() {
            // Try without branch
            let output = Command::new("git")
                .args([
                    "clone",
                    "--depth",
                    "1",
                    repo_url,
                    target_dir.to_str().unwrap(),
                ])
                .output()
                .context("Failed to clone repository")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(InstallerError::InstallationFailed(format!(
                    "Git clone failed: {}",
                    stderr
                ))
                .into());
            }
        }

        Ok(())
    }

    /// Builds from source.
    async fn build_from_source(
        &self,
        source_dir: &Path,
        progress: &Option<ProgressCallback>,
    ) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.5, "Building DeepSpeed from source...".to_string());
        }

        let mut env = self.environment.as_env_map(&self.build_config);
        env.extend(self.build_config.ops.as_build_env());
        env.insert(
            "MAX_JOBS".to_string(),
            self.build_config.parallel_jobs.to_string(),
        );

        if self.build_config.use_ninja {
            env.insert("DS_BUILD_SYSTEM".to_string(), "ninja".to_string());
        }

        // Set GPU architectures for ROCm
        let arch_str = self.build_config.gpu_archs.join(";");
        env.insert("PYTORCH_ROCM_ARCH".to_string(), arch_str);

        if let Some(ref cb) = progress {
            cb(0.6, "Running pip install...".to_string());
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "-e", "."])
            .current_dir(source_dir)
            .envs(&env)
            .output()
            .context("Failed to build DeepSpeed")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(
                InstallerError::InstallationFailed(format!("Build failed: {}", stderr)).into(),
            );
        }

        if let Some(ref cb) = progress {
            cb(0.85, "Build complete".to_string());
        }

        Ok(())
    }

    /// Runs verification tests.
    pub fn verify_installation(&self) -> Result<DeepSpeedVerification> {
        let installed = self.check_installed();
        let mut result = DeepSpeedVerification {
            installed,
            ..Default::default()
        };

        if !result.installed {
            return Ok(result);
        }

        result.version = self.get_installed_version();

        // Check GPU availability
        let gpu_check = Command::new("python3")
            .args([
                "-c",
                r#"
import deepspeed
import torch
print(f"GPU_AVAILABLE:{torch.cuda.is_available()}")
print(f"GPU_COUNT:{torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU_NAME:{torch.cuda.get_device_name(0)}")
"#,
            ])
            .output();

        if let Ok(output) = gpu_check {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.starts_with("GPU_AVAILABLE:") {
                        result.gpu_available = line.contains("True");
                    } else if line.starts_with("GPU_COUNT:") {
                        if let Ok(count) = line.strip_prefix("GPU_COUNT:").unwrap().parse() {
                            result.gpu_count = count;
                        }
                    } else if line.starts_with("GPU_NAME:") {
                        result.gpu_name = Some(line.strip_prefix("GPU_NAME:").unwrap().to_string());
                    }
                }
            }
        }

        // Check ops availability
        let ops_check = Command::new("python3")
            .args([
                "-c",
                r#"
import deepspeed
from deepspeed.ops.op_builder import ALL_OPS
for name, builder in ALL_OPS.items():
    try:
        is_compatible = builder().is_compatible()
        print(f"OP:{name}:{is_compatible}")
    except:
        print(f"OP:{name}:error")
"#,
            ])
            .output();

        if let Ok(output) = ops_check {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.starts_with("OP:") {
                        let parts: Vec<&str> = line.split(':').collect();
                        if parts.len() >= 3 {
                            result
                                .available_ops
                                .push((parts[1].to_string(), parts[2] == "True"));
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

/// DeepSpeed verification result.
#[derive(Debug, Default)]
pub struct DeepSpeedVerification {
    /// Is installed
    pub installed: bool,
    /// Installed version
    pub version: Option<String>,
    /// GPU available
    pub gpu_available: bool,
    /// GPU count
    pub gpu_count: usize,
    /// GPU name
    pub gpu_name: Option<String>,
    /// Available ops (name, compatible)
    pub available_ops: Vec<(String, bool)>,
}

#[async_trait::async_trait]
impl Installer for DeepSpeedInstaller {
    fn name(&self) -> &str {
        "DeepSpeed"
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
        if self.check_rocm() {
            checks.push("ROCm is installed".to_string());
        } else {
            checks.push("ROCm is not installed or not accessible".to_string());
        }

        // Check PyTorch
        if self.check_pytorch_rocm() {
            checks.push("PyTorch with ROCm is available".to_string());
        } else {
            checks.push("PyTorch with ROCm is required".to_string());
        }

        // Check existing installation
        if self.check_installed() {
            if let Some(version) = self.get_installed_version() {
                checks.push(format!("DeepSpeed {} is already installed", version));
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.05, "Starting DeepSpeed installation...".to_string());
        }

        self.environment.apply();

        // Install dependencies
        self.install_dependencies(&progress).await?;

        match self.source {
            DeepSpeedSource::Pypi => {
                self.install_from_pypi(&progress).await?;
            }
            DeepSpeedSource::Github | DeepSpeedSource::MicrosoftRocm => {
                let temp_dir = std::env::temp_dir().join("deepspeed-build");
                self.clone_repository(&temp_dir, &progress).await?;
                self.build_from_source(&temp_dir, &progress).await?;
                let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            }
        }

        if let Some(ref cb) = progress {
            cb(1.0, "DeepSpeed installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "deepspeed"])
            .output()
            .context("Failed to uninstall DeepSpeed")?;

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
        Ok(result.installed && result.gpu_available)
    }
}

/// Generate a DeepSpeed configuration file.
pub fn generate_ds_config(
    zero_stage: ZeroStage,
    micro_batch_size: usize,
    gradient_accumulation_steps: usize,
    fp16: bool,
) -> serde_json::Value {
    serde_json::json!({
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {
            "enabled": fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": zero_stage.as_int(),
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "reduce_bucket_size": 50000000,
            "stage3_prefetch_bucket_size": 50000000,
            "stage3_param_persistence_threshold": 100000
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_stage() {
        assert_eq!(ZeroStage::Disabled.as_int(), 0);
        assert_eq!(ZeroStage::Stage1.as_int(), 1);
        assert_eq!(ZeroStage::Stage2.as_int(), 2);
        assert_eq!(ZeroStage::Stage3.as_int(), 3);
    }

    #[test]
    fn test_deepspeed_version() {
        let version = DeepSpeedVersion::new("0.14.0");
        assert_eq!(version.version, "0.14.0");
        assert_eq!(version.git_ref, "v0.14.0");
    }

    #[test]
    fn test_deepspeed_ops_env() {
        let ops = DeepSpeedOps::default();
        let env = ops.as_build_env();
        assert_eq!(env.get("DS_BUILD_FUSED_ADAM"), Some(&"1".to_string()));
        assert_eq!(env.get("DS_BUILD_SPARSE_ATTN"), Some(&"1".to_string()));
    }

    #[test]
    fn test_deepspeed_installer_creation() {
        let installer = DeepSpeedInstaller::from_pypi();
        assert_eq!(installer.name(), "DeepSpeed");
    }

    #[test]
    fn test_generate_ds_config() {
        let config = generate_ds_config(ZeroStage::Stage2, 4, 8, true);
        assert_eq!(config["train_micro_batch_size_per_gpu"], 4);
        assert_eq!(config["zero_optimization"]["stage"], 2);
    }
}
