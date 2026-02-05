//! PyTorch Profiler Installer
//!
//! Pure Rust implementation for installing and configuring PyTorch Profiler
//! with ROCm support for AMD GPU profiling.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Profiling activities to trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileActivity {
    /// CPU operations
    Cpu,
    /// CUDA/HIP operations
    Cuda,
    /// Both CPU and GPU
    Both,
}

impl ProfileActivity {
    /// Returns PyTorch activity enum values.
    pub fn as_pytorch_activities(&self) -> Vec<&'static str> {
        match self {
            ProfileActivity::Cpu => vec!["ProfilerActivity.CPU"],
            ProfileActivity::Cuda => vec!["ProfilerActivity.CUDA"],
            ProfileActivity::Both => vec!["ProfilerActivity.CPU", "ProfilerActivity.CUDA"],
        }
    }
}

/// Profiler output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProfilerOutput {
    /// Chrome trace format (JSON)
    #[default]
    ChromeTrace,
    /// TensorBoard format
    TensorBoard,
    /// Text table format
    TextTable,
    /// All formats
    All,
}

/// Profiler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Activities to profile
    pub activities: ProfileActivity,
    /// Record shapes
    pub record_shapes: bool,
    /// Profile memory
    pub profile_memory: bool,
    /// With stack traces
    pub with_stack: bool,
    /// With flops calculation
    pub with_flops: bool,
    /// With modules
    pub with_modules: bool,
    /// Schedule wait steps
    pub schedule_wait: usize,
    /// Schedule warmup steps
    pub schedule_warmup: usize,
    /// Schedule active steps
    pub schedule_active: usize,
    /// Schedule repeat count
    pub schedule_repeat: usize,
    /// Output directory
    pub output_dir: PathBuf,
    /// Output format
    pub output_format: ProfilerOutput,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            activities: ProfileActivity::Both,
            record_shapes: true,
            profile_memory: true,
            with_stack: false,
            with_flops: true,
            with_modules: true,
            schedule_wait: 1,
            schedule_warmup: 1,
            schedule_active: 3,
            schedule_repeat: 1,
            output_dir: PathBuf::from("./profiler_output"),
            output_format: ProfilerOutput::ChromeTrace,
        }
    }
}

impl ProfilerConfig {
    /// Generates PyTorch profiler code snippet.
    pub fn generate_profiler_code(&self) -> String {
        let activities = self.activities.as_pytorch_activities().join(", ");

        format!(r#"
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

profiler_config = {{
    "activities": [{activities}],
    "record_shapes": {record_shapes},
    "profile_memory": {profile_memory},
    "with_stack": {with_stack},
    "with_flops": {with_flops},
    "with_modules": {with_modules},
    "schedule": schedule(
        wait={wait},
        warmup={warmup},
        active={active},
        repeat={repeat}
    ),
    "on_trace_ready": tensorboard_trace_handler("{output_dir}"),
}}

# Usage:
# with profile(**profiler_config) as prof:
#     for step, data in enumerate(dataloader):
#         # training step
#         prof.step()
"#,
            activities = activities,
            record_shapes = if self.record_shapes { "True" } else { "False" },
            profile_memory = if self.profile_memory { "True" } else { "False" },
            with_stack = if self.with_stack { "True" } else { "False" },
            with_flops = if self.with_flops { "True" } else { "False" },
            with_modules = if self.with_modules { "True" } else { "False" },
            wait = self.schedule_wait,
            warmup = self.schedule_warmup,
            active = self.schedule_active,
            repeat = self.schedule_repeat,
            output_dir = self.output_dir.display(),
        )
    }
}

/// ROCm profiler environment.
#[derive(Debug, Clone)]
pub struct ProfilerEnvironment {
    /// ROCm path
    pub rocm_path: PathBuf,
    /// HSA tools library path
    pub hsa_tools_lib: Option<PathBuf>,
    /// Enable ROCm profiler
    pub enable_rocm_profiler: bool,
    /// ROCtracer path
    pub roctracer_path: Option<PathBuf>,
}

impl Default for ProfilerEnvironment {
    fn default() -> Self {
        let rocm_path = PathBuf::from("/opt/rocm");
        let hsa_tools_lib = rocm_path.join("lib/librocprofiler-sdk-tool.so");
        let roctracer_path = rocm_path.join("lib/libroctracer64.so");

        Self {
            hsa_tools_lib: if hsa_tools_lib.exists() { Some(hsa_tools_lib) } else { None },
            roctracer_path: if roctracer_path.exists() { Some(roctracer_path) } else { None },
            rocm_path,
            enable_rocm_profiler: true,
        }
    }
}

impl ProfilerEnvironment {
    /// Returns environment map.
    pub fn as_env_map(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        env.insert("ROCM_PATH".to_string(), self.rocm_path.to_string_lossy().to_string());

        if let Some(ref hsa_lib) = self.hsa_tools_lib {
            env.insert("HSA_TOOLS_LIB".to_string(), hsa_lib.to_string_lossy().to_string());
        } else if !self.enable_rocm_profiler {
            env.insert("HSA_TOOLS_LIB".to_string(), "0".to_string());
        }

        if self.enable_rocm_profiler {
            env.insert("PYTORCH_PROFILE_MEMORY".to_string(), "1".to_string());
        }

        env
    }
}

/// PyTorch Profiler installer.
pub struct PytorchProfilerInstaller {
    environment: ProfilerEnvironment,
    config: ProfilerConfig,
}

impl PytorchProfilerInstaller {
    /// Creates a new installer.
    pub fn new() -> Self {
        Self {
            environment: ProfilerEnvironment::default(),
            config: ProfilerConfig::default(),
        }
    }

    /// Sets ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.environment.rocm_path = path.into();
        self
    }

    /// Sets profiler configuration.
    pub fn with_config(mut self, config: ProfilerConfig) -> Self {
        self.config = config;
        self
    }

    /// Checks if profiler is available.
    fn check_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "from torch.profiler import profile, ProfilerActivity; print('OK')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if TensorBoard is available.
    fn check_tensorboard(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import tensorboard; print(tensorboard.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if torch-tb-profiler is available.
    fn check_tb_profiler_plugin(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import torch_tb_profiler; print('OK')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets PyTorch version.
    pub fn get_pytorch_version(&self) -> Option<String> {
        Command::new("python3")
            .args(["-c", "import torch; print(torch.__version__)"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Checks ROCm profiler support.
    fn check_rocm_profiler(&self) -> bool {
        self.environment.hsa_tools_lib.as_ref().map(|p| p.exists()).unwrap_or(false) ||
            self.environment.roctracer_path.as_ref().map(|p| p.exists()).unwrap_or(false)
    }

    /// Installs TensorBoard and profiler plugin.
    async fn install_tensorboard(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.3, "Installing TensorBoard...".to_string());
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "tensorboard"])
            .output()
            .context("Failed to install tensorboard")?;

        if !output.status.success() {
            eprintln!("Warning: TensorBoard installation may have issues");
        }

        if let Some(ref cb) = progress {
            cb(0.5, "Installing PyTorch TensorBoard profiler plugin...".to_string());
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "torch-tb-profiler"])
            .output()
            .context("Failed to install torch-tb-profiler")?;

        if !output.status.success() {
            eprintln!("Warning: torch-tb-profiler installation may have issues");
        }

        Ok(())
    }

    /// Installs ROCm profiler libraries if needed.
    async fn install_rocm_profiler(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if self.check_rocm_profiler() {
            if let Some(ref cb) = progress {
                cb(0.7, "ROCm profiler already available".to_string());
            }
            return Ok(());
        }

        if let Some(ref cb) = progress {
            cb(0.6, "Installing ROCm profiler libraries...".to_string());
        }

        // Try to install rocprofiler via package manager
        let output = Command::new("sudo")
            .args(["apt-get", "install", "-y", "rocprofiler", "roctracer"])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                // Update environment
                let hsa_lib = self.environment.rocm_path.join("lib/librocprofiler-sdk-tool.so");
                if hsa_lib.exists() {
                    std::env::set_var("HSA_TOOLS_LIB", &hsa_lib);
                }
            }
        }

        Ok(())
    }

    /// Runs a profiling session.
    pub fn run_profile(&self, script_path: &Path, output_dir: &Path) -> Result<PathBuf> {
        // Create output directory
        std::fs::create_dir_all(output_dir)
            .context("Failed to create output directory")?;

        let profile_wrapper = format!(r#"
import sys
sys.path.insert(0, '.')

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('{}'),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    exec(open('{}').read())
    prof.step()

print('Profiling complete. View with: tensorboard --logdir={}')
"#,
            output_dir.display(),
            script_path.display(),
            output_dir.display(),
        );

        let wrapper_path = output_dir.join("_profile_wrapper.py");
        std::fs::write(&wrapper_path, profile_wrapper)
            .context("Failed to write profile wrapper")?;

        let output = Command::new("python3")
            .arg(&wrapper_path)
            .envs(self.environment.as_env_map())
            .output()
            .context("Failed to run profiling")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Profiling failed: {}", stderr)
            ).into());
        }

        // Clean up wrapper
        let _ = std::fs::remove_file(&wrapper_path);

        Ok(output_dir.to_path_buf())
    }

    /// Verification result.
    pub fn verify_installation(&self) -> Result<ProfilerVerification> {
        let profiler_available = self.check_installed();
        let pytorch_version = self.get_pytorch_version();
        let tensorboard_available = self.check_tensorboard();
        let tb_profiler_plugin = self.check_tb_profiler_plugin();
        let rocm_profiler_available = self.check_rocm_profiler();

        let mut result = ProfilerVerification {
            profiler_available,
            pytorch_version,
            tensorboard_available,
            tb_profiler_plugin,
            rocm_profiler_available,
            ..Default::default()
        };

        // Check GPU profiling capability
        if result.profiler_available {
            let gpu_check = Command::new("python3")
                .args(["-c", r#"
import torch
from torch.profiler import ProfilerActivity
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)
    print("GPU_PROFILING:OK")
else:
    print("GPU_PROFILING:NO")
"#])
                .output();

            if let Ok(output) = gpu_check {
                let stdout = String::from_utf8_lossy(&output.stdout);
                result.gpu_profiling = stdout.contains("GPU_PROFILING:OK");
            }
        }

        Ok(result)
    }
}

impl Default for PytorchProfilerInstaller {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiler verification result.
#[derive(Debug, Default)]
pub struct ProfilerVerification {
    /// PyTorch profiler available
    pub profiler_available: bool,
    /// PyTorch version
    pub pytorch_version: Option<String>,
    /// TensorBoard available
    pub tensorboard_available: bool,
    /// TensorBoard profiler plugin available
    pub tb_profiler_plugin: bool,
    /// ROCm profiler libraries available
    pub rocm_profiler_available: bool,
    /// GPU profiling capability
    pub gpu_profiling: bool,
}

#[async_trait::async_trait]
impl Installer for PytorchProfilerInstaller {
    fn name(&self) -> &str {
        "PyTorch Profiler"
    }

    fn version(&self) -> &str {
        "bundled"
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if let Some(version) = self.get_pytorch_version() {
            checks.push(format!("PyTorch {} installed", version));
        } else {
            checks.push("PyTorch not found".to_string());
        }

        if self.check_installed() {
            checks.push("PyTorch profiler available".to_string());
        } else {
            checks.push("PyTorch profiler not available".to_string());
        }

        if self.check_tensorboard() {
            checks.push("TensorBoard available".to_string());
        } else {
            checks.push("TensorBoard not installed".to_string());
        }

        if self.check_tb_profiler_plugin() {
            checks.push("TensorBoard profiler plugin available".to_string());
        } else {
            checks.push("TensorBoard profiler plugin not installed".to_string());
        }

        if self.check_rocm_profiler() {
            checks.push("ROCm profiler libraries available".to_string());
        } else {
            checks.push("ROCm profiler libraries not found".to_string());
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing PyTorch Profiler dependencies...".to_string());
        }

        // PyTorch profiler is built into PyTorch, we just need the visualization tools
        self.install_tensorboard(&progress).await?;

        // Install ROCm profiler support
        self.install_rocm_profiler(&progress).await?;

        if let Some(ref cb) = progress {
            cb(1.0, "PyTorch Profiler installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        // Remove tensorboard and profiler plugin
        let _ = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "torch-tb-profiler", "tensorboard"])
            .output();

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        let result = self.verify_installation()?;
        Ok(result.profiler_available && result.tensorboard_available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_activity() {
        let both = ProfileActivity::Both;
        let activities = both.as_pytorch_activities();
        assert_eq!(activities.len(), 2);
    }

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert!(config.record_shapes);
        assert!(config.profile_memory);
        assert!(config.with_flops);
    }

    #[test]
    fn test_generate_profiler_code() {
        let config = ProfilerConfig::default();
        let code = config.generate_profiler_code();
        assert!(code.contains("profile"));
        assert!(code.contains("ProfilerActivity"));
    }

    #[test]
    fn test_installer_creation() {
        let installer = PytorchProfilerInstaller::new();
        assert_eq!(installer.name(), "PyTorch Profiler");
    }
}
