//! Megatron-LM installer — ports `scripts/install_megatron.sh`.
//!
//! Constructs correct git clone URL/branch and pip install commands.
//! Megatron depends on PyTorch (ROCm-enabled) and MPI4Py.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-012**: Megatron correct git clone
//! - **VAL-INSTALL-013**: Megatron correct pip install post-clone
//! - **VAL-INSTALL-041**: Megatron declares dependency on PyTorch and MPI4Py
//! - **VAL-INSTALL-043**: DeepSpeed declares dependency on PyTorch

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

/// Configuration for the Megatron-LM installer.
#[derive(Debug, Clone)]
pub struct MegatronConfig {
    /// Installation method.
    pub method: InstallMethod,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Installation directory (defaults to `$HOME/Megatron-LM`).
    pub install_dir: Option<PathBuf>,
}

impl Default for MegatronConfig {
    fn default() -> Self {
        Self {
            method: InstallMethod::Auto,
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
            install_dir: None,
        }
    }
}

impl MegatronConfig {
    /// Get the installation directory path.
    ///
    /// The original script uses `$HOME/Megatron-LM` as primary,
    /// with `$HOME/.mlstack/src/Megatron-LM` as fallback.
    pub fn install_dir(&self) -> PathBuf {
        self.install_dir
            .clone()
            .unwrap_or_else(|| dirs_home().join("Megatron-LM"))
    }

    /// Get the fallback installation directory.
    pub fn fallback_install_dir(&self) -> PathBuf {
        dirs_home().join(".mlstack/src/Megatron-LM")
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

/// The Megatron-LM installer.
pub struct MegatronInstaller {
    config: MegatronConfig,
}

impl MegatronInstaller {
    /// Create a new Megatron installer with the given config.
    pub fn new(config: MegatronConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MegatronConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependencies (VAL-INSTALL-041)
    // -----------------------------------------------------------------------

    /// Get the list of required dependencies.
    ///
    /// Megatron-LM depends on:
    /// - PyTorch (ROCm-enabled)
    /// - MPI4Py
    pub fn dependencies(&self) -> &[&str] {
        &["pytorch", "mpi4py"]
    }

    // -----------------------------------------------------------------------
    // ROCm environment
    // -----------------------------------------------------------------------

    /// Get the ROCm environment variables needed for Megatron.
    ///
    /// The original script sets:
    /// - AMD_LOG_LEVEL=0
    /// - HIP_VISIBLE_DEVICES
    /// - ROCR_VISIBLE_DEVICES
    /// - HSA_OVERRIDE_GFX_VERSION=11.0.0
    /// - PYTORCH_ROCM_ARCH=gfx1100
    /// - ROCM_PATH
    /// - HSA_TOOLS_LIB
    /// - PYTORCH_ALLOC_CONF=expandable_segments:True
    pub fn rocm_env(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        let mut env = vec![
            ("AMD_LOG_LEVEL".to_string(), "0".to_string()),
            (
                "HSA_OVERRIDE_GFX_VERSION".to_string(),
                "11.0.0".to_string(),
            ),
            ("PYTORCH_ROCM_ARCH".to_string(), "gfx1100".to_string()),
            ("ROCM_PATH".to_string(), rocm_path.clone()),
            (
                "PYTORCH_ALLOC_CONF".to_string(),
                "expandable_segments:True".to_string(),
            ),
        ];

        // HSA_TOOLS_LIB
        let profiler_lib = rocm_env
            .path()
            .map(|p| p.join("lib/librocprofiler-sdk-tool.so"))
            .filter(|p| p.exists());
        if let Some(lib) = profiler_lib {
            env.push(("HSA_TOOLS_LIB".to_string(), lib.to_string_lossy().to_string()));
        } else {
            env.push(("HSA_TOOLS_LIB".to_string(), "0".to_string()));
        }

        env
    }

    // -----------------------------------------------------------------------
    // Git clone command (VAL-INSTALL-012)
    // -----------------------------------------------------------------------

    /// Construct the git clone command for Megatron-LM.
    ///
    /// The original script clones:
    /// `git clone https://github.com/NVIDIA/Megatron-LM.git <dir>`
    ///
    /// It retries up to 3 times on failure.
    pub fn build_git_clone_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "https://github.com/NVIDIA/Megatron-LM.git".to_string(),
                self.config.install_dir().to_string_lossy().to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Get the maximum number of clone retries.
    pub fn max_clone_retries(&self) -> u32 {
        3
    }

    // -----------------------------------------------------------------------
    // Pip install command (VAL-INSTALL-013)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for Megatron-LM.
    ///
    /// The original script installs with:
    /// `pip install -e . --no-deps`
    ///
    /// The editable install is done from the cloned directory.
    /// Dependencies are resolved separately (safe requirements filtering).
    pub fn build_pip_install_command(&self) -> ShellCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];

        if is_global {
            args.push("--break-system-packages".to_string());
        }

        args.push("-e".to_string());
        args.push(".".to_string());
        args.push("--no-deps".to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
            working_dir: Some(self.config.install_dir()),
        }
    }

    /// Construct the pip install command for a specific dependency.
    ///
    /// Used for installing individual safe dependencies.
    pub fn build_dep_install_command(&self, package: &str) -> ShellCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];

        if is_global {
            args.push("--break-system-packages".to_string());
        }

        args.push(package.to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // Safe dependency filtering
    // -----------------------------------------------------------------------

    /// Check if a Python package is safe to install (non-CUDA/non-NVIDIA).
    ///
    /// The original script filters out packages that would conflict with
    /// the ROCm environment: nvidia-*, cuda*, torch, torchvision, etc.
    pub fn is_safe_package(&self, package_name: &str) -> bool {
        let lower = package_name.to_lowercase();
        let lower = lower.split(';').next().unwrap_or(&lower);
        let lower = lower.split('[').next().unwrap_or(lower);
        let pkg_name = lower
            .split(&['<', '>', '=', '~', '!', ' '][..])
            .next()
            .unwrap_or(lower)
            .trim();

        let blocked_prefixes = [
            "nvidia",
            "cuda",
            "cudnn",
            "cublas",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "nccl",
            "nvtx",
            "nvjitlink",
            "tensorrt",
        ];

        let blocked_exact = [
            "torch",
            "torchvision",
            "torchaudio",
            "triton",
            "xformers",
            "pytorch-cuda",
            "torch-cuda",
        ];

        if blocked_exact.contains(&pkg_name) {
            return false;
        }

        for prefix in &blocked_prefixes {
            if pkg_name.starts_with(prefix) {
                return false;
            }
        }

        // Check for patterns like cupy-cuda*
        if pkg_name.contains("cuda") {
            return false;
        }

        true
    }

    // -----------------------------------------------------------------------
    // Build error detection (VAL-INSTALL-014)
    // -----------------------------------------------------------------------

    /// Check build output for errors.
    pub fn check_build_output(&self, stdout: &str, stderr: &str, exit_code: i32) -> Result<(), String> {
        if exit_code != 0 {
            let error_patterns = [
                "error:",
                "Error:",
                "FAILED:",
                "fatal error:",
                "undefined reference",
                "No such file or directory",
                "Permission denied",
                "ImportError:",
                "ModuleNotFoundError:",
            ];

            let combined = format!("{}\n{}", stdout, stderr);
            let detected_errors: Vec<&str> = error_patterns
                .iter()
                .filter(|p| combined.contains(*p))
                .copied()
                .collect();

            if detected_errors.is_empty() {
                return Err(format!(
                    "Megatron operation failed with exit code {}",
                    exit_code
                ));
            }

            return Err(format!(
                "Megatron operation failed with exit code {}. Detected errors: {}",
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-012: Megatron correct git clone ---

    #[test]
    fn test_git_clone_command() {
        let installer = MegatronInstaller::with_defaults();
        let cmd = installer.build_git_clone_command();

        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("NVIDIA/Megatron-LM")));
    }

    #[test]
    fn test_git_clone_target_dir() {
        let installer = MegatronInstaller::new(MegatronConfig {
            install_dir: Some(PathBuf::from("/home/user/Megatron-LM")),
            ..Default::default()
        });
        let cmd = installer.build_git_clone_command();
        assert!(cmd.args.iter().any(|a| a.contains("/home/user/Megatron-LM")));
    }

    #[test]
    fn test_git_clone_retries() {
        let installer = MegatronInstaller::with_defaults();
        assert_eq!(installer.max_clone_retries(), 3);
    }

    // --- VAL-INSTALL-013: Megatron correct pip install post-clone ---

    #[test]
    fn test_pip_install_command() {
        let installer = MegatronInstaller::new(MegatronConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_pip_install_command();

        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"-e".to_string()));
        assert!(cmd.args.contains(&".".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
    }

    #[test]
    fn test_pip_install_venv_no_break_system() {
        let installer = MegatronInstaller::new(MegatronConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_pip_install_command();
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_pip_install_working_dir() {
        let installer = MegatronInstaller::new(MegatronConfig {
            install_dir: Some(PathBuf::from("/home/user/Megatron-LM")),
            ..Default::default()
        });
        let cmd = installer.build_pip_install_command();
        assert_eq!(cmd.working_dir, Some(PathBuf::from("/home/user/Megatron-LM")));
    }

    #[test]
    fn test_dep_install_command() {
        let installer = MegatronInstaller::new(MegatronConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_dep_install_command("numpy");
        assert!(cmd.args.contains(&"numpy".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    // --- VAL-INSTALL-041: Megatron declares dependency on PyTorch and MPI4Py ---

    #[test]
    fn test_dependencies() {
        let installer = MegatronInstaller::with_defaults();
        let deps = installer.dependencies();
        assert!(deps.contains(&"pytorch"));
        assert!(deps.contains(&"mpi4py"));
    }

    // --- ROCm environment ---

    #[test]
    fn test_rocm_env() {
        let installer = MegatronInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(
            Some(PathBuf::from("/opt/rocm")),
            "7.2.0".to_string(),
        );
        let env = installer.rocm_env(&rocm_env);
        assert!(env.iter().any(|(k, v)| k == "AMD_LOG_LEVEL" && v == "0"));
        assert!(env.iter().any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
        assert!(env.iter().any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
        assert!(env.iter().any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env.iter().any(|(k, v)| k == "PYTORCH_ALLOC_CONF" && v == "expandable_segments:True"));
        assert!(env.iter().any(|(k, _)| k == "HSA_TOOLS_LIB"));
    }

    // --- Safe package filtering ---

    #[test]
    fn test_safe_package_allows_megatron_deps() {
        let installer = MegatronInstaller::with_defaults();
        // These should be safe
        assert!(installer.is_safe_package("numpy"));
        assert!(installer.is_safe_package("regex"));
        assert!(installer.is_safe_package("sentencepiece"));
        assert!(installer.is_safe_package("pyyaml"));
        assert!(installer.is_safe_package("transformers"));
        assert!(installer.is_safe_package("tokenizers"));
        assert!(installer.is_safe_package("tqdm"));
        assert!(installer.is_safe_package("einops"));
        assert!(installer.is_safe_package("packaging"));
        assert!(installer.is_safe_package("omegaconf"));
    }

    #[test]
    fn test_safe_package_blocks_cuda_nvidia() {
        let installer = MegatronInstaller::with_defaults();
        // These should be blocked
        assert!(!installer.is_safe_package("torch"));
        assert!(!installer.is_safe_package("torchvision"));
        assert!(!installer.is_safe_package("torchaudio"));
        assert!(!installer.is_safe_package("triton"));
        assert!(!installer.is_safe_package("nvidia-cublas-cu12"));
        assert!(!installer.is_safe_package("nvidia-cuda-runtime-cu12"));
        assert!(!installer.is_safe_package("pytorch-cuda"));
        assert!(!installer.is_safe_package("torch-cuda"));
        assert!(!installer.is_safe_package("xformers"));
        assert!(!installer.is_safe_package("cupy-cuda12x"));
    }

    #[test]
    fn test_safe_package_with_version_spec() {
        let installer = MegatronInstaller::with_defaults();
        assert!(installer.is_safe_package("numpy>=1.21"));
        assert!(installer.is_safe_package("packaging>=20.0"));
        assert!(!installer.is_safe_package("torch>=2.0"));
    }

    // --- Build error detection ---

    #[test]
    fn test_build_error_import() {
        let installer = MegatronInstaller::with_defaults();
        let result = installer.check_build_output(
            "",
            "ImportError: No module named 'megatron'",
            1,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ImportError"));
    }

    #[test]
    fn test_build_error_success() {
        let installer = MegatronInstaller::with_defaults();
        let result = installer.check_build_output("OK", "", 0);
        assert!(result.is_ok());
    }

    // --- Config defaults ---

    #[test]
    fn test_config_defaults() {
        let config = MegatronConfig::default();
        assert_eq!(config.method, InstallMethod::Auto);
        assert_eq!(config.python_bin, "python3");
        assert!(!config.force_reinstall);
        assert!(!config.dry_run);
        assert!(config.install_dir.is_none());
    }

    #[test]
    fn test_config_install_dir_default() {
        let config = MegatronConfig::default();
        let dir = config.install_dir();
        assert!(dir.to_string_lossy().contains("Megatron-LM"));
    }

    #[test]
    fn test_config_fallback_dir() {
        let config = MegatronConfig::default();
        let fallback = config.fallback_install_dir();
        assert!(fallback.to_string_lossy().contains(".mlstack/src/Megatron-LM"));
    }

    #[test]
    fn test_command_string_format() {
        let cmd = ShellCommand {
            program: "python3".to_string(),
            args: vec!["-m".to_string(), "pip".to_string(), "install".to_string(), "numpy".to_string()],
            env: vec![("ROCM_PATH".to_string(), "/opt/rocm".to_string())],
            working_dir: None,
        };
        let s = cmd.to_command_string();
        assert!(s.contains("ROCM_PATH=/opt/rocm"));
        assert!(s.contains("python3 -m pip install numpy"));
    }
}
