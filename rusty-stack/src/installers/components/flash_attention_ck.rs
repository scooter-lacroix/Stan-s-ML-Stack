//! Flash Attention CK installer — ports `scripts/install_flash_attention_ck.sh`.
//!
//! Constructs correct cmake commands with HIP compiler flags and ROCm path,
//! make commands with job count, and pip install from build output.
//!
//! Flash Attention CK depends on PyTorch (ROCm-enabled) and ROCm.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-008**: Flash Attention correct cmake command
//! - **VAL-INSTALL-009**: Flash Attention correct make command
//! - **VAL-INSTALL-014**: Source builds parse build output for errors
//! - **VAL-INSTALL-044**: Flash Attention declares dependency on PyTorch and ROCm

use crate::installers::common::RocmEnv;
use std::fmt;
use std::path::PathBuf;

// ===========================================================================
// Types
// ===========================================================================

/// Installation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethod {
    /// Install globally.
    Global,
    /// Install in a virtual environment.
    Venv,
    /// Try global, fallback to venv.
    Auto,
}

impl fmt::Display for InstallMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstallMethod::Global => write!(f, "global"),
            InstallMethod::Venv => write!(f, "venv"),
            InstallMethod::Auto => write!(f, "auto"),
        }
    }
}

/// GPU architecture specification for building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArch {
    /// Detected or specified GPU architecture (e.g., "gfx1100").
    Known(String),
    /// Default fallback architecture.
    Default,
}

impl GpuArch {
    /// Get the architecture string.
    pub fn as_str(&self) -> &str {
        match self {
            GpuArch::Known(s) => s,
            GpuArch::Default => "gfx1100",
        }
    }
}

/// Configuration for the Flash Attention CK installer.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Installation method.
    pub method: InstallMethod,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Installation directory (defaults to `$HOME/ml_stack/flash_attn_amd`).
    pub install_dir: Option<PathBuf>,
    /// GPU architecture override.
    pub gpu_arch: GpuArch,
    /// Number of parallel build jobs (defaults to nproc).
    pub build_jobs: Option<usize>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            method: InstallMethod::Auto,
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
            install_dir: None,
            gpu_arch: GpuArch::Default,
            build_jobs: None,
        }
    }
}

impl FlashAttentionConfig {
    /// Get the installation directory path.
    pub fn install_dir(&self) -> PathBuf {
        self.install_dir
            .clone()
            .unwrap_or_else(|| dirs_home().join("ml_stack").join("flash_attn_amd"))
    }

    /// Get the number of build jobs (defaults to num_cpus).
    pub fn build_jobs(&self) -> usize {
        self.build_jobs.unwrap_or_else(num_cpus)
    }
}

/// A constructed shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
    /// Working directory for the command.
    pub working_dir: Option<PathBuf>,
}

impl ShellCommand {
    /// Format as a shell command string.
    pub fn to_command_string(&self) -> String {
        let env_prefix = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        };
        if env_prefix.is_empty() {
            cmd
        } else {
            format!("{env_prefix} {cmd}")
        }
    }
}

/// The Flash Attention CK installer.
pub struct FlashAttentionInstaller {
    config: FlashAttentionConfig,
}

impl FlashAttentionInstaller {
    /// Create a new Flash Attention installer with the given config.
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(FlashAttentionConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependencies (VAL-INSTALL-044)
    // -----------------------------------------------------------------------

    /// Get the list of required dependencies.
    ///
    /// Flash Attention CK depends on:
    /// - PyTorch (ROCm-enabled)
    /// - ROCm
    pub fn dependencies(&self) -> &[&str] {
        &["pytorch", "rocm"]
    }

    // -----------------------------------------------------------------------
    // ROCm build environment (VAL-INSTALL-008)
    // -----------------------------------------------------------------------

    /// Get the ROCm environment variables needed for the build.
    ///
    /// The original script sets:
    /// - HSA_OVERRIDE_GFX_VERSION=11.0.0
    /// - PYTORCH_ROCM_ARCH=gfx1100 (or detected)
    /// - ROCM_PATH=/opt/rocm
    /// - PATH=/opt/rocm/bin:$PATH
    /// - LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
    /// - HSA_TOOLS_LIB (if rocprofiler library exists)
    pub fn rocm_build_env(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        let mut env = vec![
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
            (
                "PYTORCH_ROCM_ARCH".to_string(),
                self.config.gpu_arch.as_str().to_string(),
            ),
            ("ROCM_PATH".to_string(), rocm_path.clone()),
            (
                "PATH".to_string(),
                format!(
                    "{}/bin:{}",
                    rocm_path,
                    std::env::var("PATH").unwrap_or_default()
                ),
            ),
            (
                "LD_LIBRARY_PATH".to_string(),
                format!(
                    "{}/lib:{}",
                    rocm_path,
                    std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
                ),
            ),
        ];

        // HSA_TOOLS_LIB - check for rocprofiler library
        let profiler_lib = rocm_env
            .path()
            .map(|p| p.join("lib/librocprofiler-sdk-tool.so"))
            .filter(|p| p.exists());
        if let Some(lib) = profiler_lib {
            env.push((
                "HSA_TOOLS_LIB".to_string(),
                lib.to_string_lossy().to_string(),
            ));
        } else {
            env.push(("HSA_TOOLS_LIB".to_string(), "0".to_string()));
        }

        env
    }

    // -----------------------------------------------------------------------
    // Git clone command
    // -----------------------------------------------------------------------

    /// Construct the git clone command for Flash Attention.
    ///
    /// The original script clones from:
    /// `https://github.com/ROCmSoftwarePlatform/flash-attention.git`
    /// and checks out the `main` branch.
    pub fn build_git_clone_command(&self) -> ShellCommand {
        let install_dir = self.config.install_dir();
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "https://github.com/ROCmSoftwarePlatform/flash-attention.git".to_string(),
                install_dir.to_string_lossy().to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Construct the git checkout command.
    pub fn build_git_checkout_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec!["checkout".to_string(), "main".to_string()],
            env: vec![],
            working_dir: Some(self.config.install_dir()),
        }
    }

    // -----------------------------------------------------------------------
    // CMake command (VAL-INSTALL-008)
    // -----------------------------------------------------------------------

    /// Construct the cmake configure command.
    ///
    /// The original script configures cmake with:
    /// - CMAKE_PREFIX_PATH: torch cmake prefix + ROCm path
    /// - CMAKE_BUILD_TYPE=Release
    /// - GPU_TARGETS=gfx1100 (or detected)
    /// - CMAKE_CXX_FLAGS="-Wno-error"
    /// - ROCM_PATH=/opt/rocm
    ///
    /// Working directory: `<install_dir>/build`
    pub fn build_cmake_command(
        &self,
        rocm_env: &RocmEnv,
        torch_cmake_prefix: &str,
    ) -> ShellCommand {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        let prefix_path = format!("{};{};/opt/rocm", torch_cmake_prefix, rocm_path);

        let mut args = vec![
            "..".to_string(),
            format!("-DCMAKE_PREFIX_PATH={}", prefix_path),
            "-DCMAKE_BUILD_TYPE=Release".to_string(),
            format!("-DGPU_TARGETS={}", self.config.gpu_arch.as_str()),
            "-DCMAKE_CXX_FLAGS=-Wno-error".to_string(),
        ];

        args.push(format!("-DROCM_PATH={}", rocm_path));

        ShellCommand {
            program: "cmake".to_string(),
            args,
            env: vec![],
            working_dir: Some(self.config.install_dir().join("build")),
        }
    }

    // -----------------------------------------------------------------------
    // Make / build command (VAL-INSTALL-009)
    // -----------------------------------------------------------------------

    /// Construct the cmake --build command.
    ///
    /// The original script uses:
    /// `cmake --build . --config Release -j $(nproc)`
    ///
    /// Working directory: `<install_dir>/build`
    pub fn build_make_command(&self) -> ShellCommand {
        let jobs = self.config.build_jobs();
        ShellCommand {
            program: "cmake".to_string(),
            args: vec![
                "--build".to_string(),
                ".".to_string(),
                "--config".to_string(),
                "Release".to_string(),
                "-j".to_string(),
                jobs.to_string(),
            ],
            env: vec![],
            working_dir: Some(self.config.install_dir().join("build")),
        }
    }

    // -----------------------------------------------------------------------
    // Setup.py install command
    // -----------------------------------------------------------------------

    /// Construct the Python setup.py install command.
    ///
    /// The original script runs:
    /// `python3 setup_flash_attn_amd.py install`
    pub fn build_setup_install_command(&self) -> ShellCommand {
        let args = vec!["setup_flash_attn_amd.py".to_string(), "install".to_string()];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
            working_dir: Some(self.config.install_dir()),
        }
    }

    // -----------------------------------------------------------------------
    // Build error detection (VAL-INSTALL-014)
    // -----------------------------------------------------------------------

    /// Check build output for errors.
    ///
    /// Detects common cmake and make failure patterns in the output.
    /// Returns an error message if a failure is detected.
    pub fn check_build_output(
        &self,
        stdout: &str,
        stderr: &str,
        exit_code: i32,
    ) -> Result<(), String> {
        if exit_code != 0 {
            // Check for specific error patterns
            let error_patterns = [
                "CMake Error",
                "error:",
                "Error:",
                "FAILED:",
                "fatal error:",
                "undefined reference",
                "No such file or directory",
                "Permission denied",
            ];

            let combined = format!("{}\n{}", stdout, stderr);
            let mut detected_errors: Vec<&str> = Vec::new();

            for pattern in &error_patterns {
                if combined.contains(pattern) {
                    detected_errors.push(*pattern);
                }
            }

            if detected_errors.is_empty() {
                return Err(format!(
                    "Build command failed with exit code {} (no specific error pattern detected)",
                    exit_code
                ));
            }

            return Err(format!(
                "Build failed with exit code {}. Detected errors: {}",
                exit_code,
                detected_errors.join(", ")
            ));
        }

        Ok(())
    }
}

/// Get the home directory.
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
}

/// Get the number of CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-008: Flash Attention correct cmake command ---

    #[test]
    fn test_cmake_command_has_correct_flags() {
        let installer = FlashAttentionInstaller::new(FlashAttentionConfig {
            gpu_arch: GpuArch::Known("gfx1100".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_cmake_command(
            &rocm_env,
            "/usr/lib/python3/dist-packages/torch/share/cmake",
        );

        assert_eq!(cmd.program, "cmake");
        assert!(cmd.args.contains(&"..".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("CMAKE_PREFIX_PATH")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("CMAKE_BUILD_TYPE=Release")));
        assert!(cmd.args.iter().any(|a| a.contains("GPU_TARGETS=gfx1100")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("CMAKE_CXX_FLAGS=-Wno-error")));
        assert!(cmd.args.iter().any(|a| a.contains("ROCM_PATH=/opt/rocm")));
    }

    #[test]
    fn test_cmake_command_includes_torch_prefix() {
        let installer = FlashAttentionInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_cmake_command(&rocm_env, "/torch/cmake/prefix");
        let prefix_arg = cmd
            .args
            .iter()
            .find(|a| a.starts_with("-DCMAKE_PREFIX_PATH="));
        assert!(prefix_arg.is_some());
        let prefix = prefix_arg.unwrap();
        assert!(prefix.contains("/torch/cmake/prefix"));
        assert!(prefix.contains("/opt/rocm"));
    }

    #[test]
    fn test_cmake_command_working_dir() {
        let installer = FlashAttentionInstaller::new(FlashAttentionConfig {
            install_dir: Some(PathBuf::from("/home/user/ml_stack/flash_attn_amd")),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_cmake_command(&rocm_env, "/torch/cmake");
        assert_eq!(
            cmd.working_dir,
            Some(PathBuf::from("/home/user/ml_stack/flash_attn_amd/build"))
        );
    }

    #[test]
    fn test_cmake_command_custom_gpu_arch() {
        let installer = FlashAttentionInstaller::new(FlashAttentionConfig {
            gpu_arch: GpuArch::Known("gfx1030".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_cmake_command(&rocm_env, "/torch/cmake");
        assert!(cmd.args.iter().any(|a| a.contains("GPU_TARGETS=gfx1030")));
    }

    // --- VAL-INSTALL-009: Flash Attention correct make command ---

    #[test]
    fn test_make_command_correct_flags() {
        let installer = FlashAttentionInstaller::with_defaults();
        let cmd = installer.build_make_command();

        assert_eq!(cmd.program, "cmake");
        assert!(cmd.args.contains(&"--build".to_string()));
        assert!(cmd.args.contains(&".".to_string()));
        assert!(cmd.args.contains(&"--config".to_string()));
        assert!(cmd.args.contains(&"Release".to_string()));
        assert!(cmd.args.contains(&"-j".to_string()));
    }

    #[test]
    fn test_make_command_job_count() {
        let installer = FlashAttentionInstaller::new(FlashAttentionConfig {
            build_jobs: Some(8),
            ..Default::default()
        });
        let cmd = installer.build_make_command();
        assert!(cmd.args.contains(&"8".to_string()));
    }

    #[test]
    fn test_make_command_working_dir() {
        let installer = FlashAttentionInstaller::new(FlashAttentionConfig {
            install_dir: Some(PathBuf::from("/home/user/ml_stack/flash_attn_amd")),
            ..Default::default()
        });
        let cmd = installer.build_make_command();
        assert_eq!(
            cmd.working_dir,
            Some(PathBuf::from("/home/user/ml_stack/flash_attn_amd/build"))
        );
    }

    // --- Git clone command ---

    #[test]
    fn test_git_clone_command() {
        let installer = FlashAttentionInstaller::with_defaults();
        let cmd = installer.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("ROCmSoftwarePlatform/flash-attention")));
    }

    #[test]
    fn test_git_checkout_command() {
        let installer = FlashAttentionInstaller::with_defaults();
        let cmd = installer.build_git_checkout_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"checkout".to_string()));
        assert!(cmd.args.contains(&"main".to_string()));
    }

    // --- ROCm build environment ---

    #[test]
    fn test_rocm_build_env() {
        let installer = FlashAttentionInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let env = installer.rocm_build_env(&rocm_env);
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env.iter().any(|(k, _)| k == "HSA_TOOLS_LIB"));
    }

    // --- Setup.py install command ---

    #[test]
    fn test_setup_install_command() {
        let installer = FlashAttentionInstaller::with_defaults();
        let cmd = installer.build_setup_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"setup_flash_attn_amd.py".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
    }

    // --- VAL-INSTALL-044: Flash Attention declares dependency on PyTorch and ROCm ---

    #[test]
    fn test_dependencies() {
        let installer = FlashAttentionInstaller::with_defaults();
        let deps = installer.dependencies();
        assert!(deps.contains(&"pytorch"));
        assert!(deps.contains(&"rocm"));
    }

    // --- VAL-INSTALL-014: Build error detection ---

    #[test]
    fn test_build_error_detection_cmake_error() {
        let installer = FlashAttentionInstaller::with_defaults();
        let result =
            installer.check_build_output("some output", "CMake Error at CMakeLists.txt:10", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CMake Error"));
    }

    #[test]
    fn test_build_error_detection_make_error() {
        let installer = FlashAttentionInstaller::with_defaults();
        let result = installer.check_build_output(
            "compiling...",
            "error: 'undefined_var' was not declared",
            2,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("error:"));
    }

    #[test]
    fn test_build_error_detection_success() {
        let installer = FlashAttentionInstaller::with_defaults();
        let result = installer.check_build_output("Build finished successfully", "", 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_error_detection_nonzero_no_pattern() {
        let installer = FlashAttentionInstaller::with_defaults();
        let result = installer.check_build_output("clean output", "clean output", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exit code 1"));
    }

    // --- Config defaults ---

    #[test]
    fn test_config_defaults() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.method, InstallMethod::Auto);
        assert_eq!(config.python_bin, "python3");
        assert!(!config.force_reinstall);
        assert!(!config.dry_run);
        assert!(config.install_dir.is_none());
        assert_eq!(config.gpu_arch, GpuArch::Default);
        assert!(config.build_jobs.is_none());
    }

    #[test]
    fn test_gpu_arch_default() {
        assert_eq!(GpuArch::Default.as_str(), "gfx1100");
        assert_eq!(GpuArch::Known("gfx1030".to_string()).as_str(), "gfx1030");
    }

    #[test]
    fn test_command_string_format() {
        let cmd = ShellCommand {
            program: "cmake".to_string(),
            args: vec!["--build".to_string(), ".".to_string()],
            env: vec![("ROCM_PATH".to_string(), "/opt/rocm".to_string())],
            working_dir: None,
        };
        let s = cmd.to_command_string();
        assert!(s.contains("ROCM_PATH=/opt/rocm"));
        assert!(s.contains("cmake --build ."));
    }
}
