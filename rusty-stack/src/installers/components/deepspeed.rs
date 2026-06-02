//! DeepSpeed installer — ports `scripts/install_deepspeed.sh`.
//!
//! Constructs correct pip install with ROCm build flags.
//! DeepSpeed depends on PyTorch (validated via preflight check).
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-006**: DeepSpeed installer correct pip command

use crate::installers::common::RocmEnv;
use std::fmt;

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

/// Configuration for the DeepSpeed installer.
#[derive(Debug, Clone)]
pub struct DeepSpeedConfig {
    /// Installation method.
    pub method: InstallMethod,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for DeepSpeedConfig {
    fn default() -> Self {
        Self {
            method: InstallMethod::Auto,
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
        }
    }
}

/// A constructed pip command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
}

impl PipCommand {
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

/// The DeepSpeed installer.
pub struct DeepSpeedInstaller {
    config: DeepSpeedConfig,
}

impl DeepSpeedInstaller {
    /// Create a new DeepSpeed installer with the given config.
    pub fn new(config: DeepSpeedConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(DeepSpeedConfig::default())
    }

    // -----------------------------------------------------------------------
    // ROCm build environment (VAL-INSTALL-006)
    // -----------------------------------------------------------------------

    /// Get the ROCm environment variables needed for DeepSpeed build.
    ///
    /// DeepSpeed auto-detects ROCm when ROCM_HOME is set.
    /// The original script sets:
    /// - ROCM_HOME=/opt/rocm
    /// - HIP_PATH=/opt/rocm
    /// - HSA_OVERRIDE_GFX_VERSION
    /// - PYTORCH_ROCM_ARCH
    pub fn rocm_build_env(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        vec![
            ("ROCM_HOME".to_string(), rocm_path.clone()),
            ("HIP_PATH".to_string(), rocm_path),
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
            ("PYTORCH_ROCM_ARCH".to_string(), "gfx1100".to_string()),
        ]
    }

    // -----------------------------------------------------------------------
    // Command construction
    // -----------------------------------------------------------------------

    /// Construct the pip install command for DeepSpeed dependencies.
    ///
    /// Installs packaging, ninja, pydantic, jsonschema, einops.
    pub fn build_deps_install_command(&self) -> PipCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if is_global {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "packaging".to_string(),
            "ninja".to_string(),
            "pydantic".to_string(),
            "jsonschema".to_string(),
            "einops".to_string(),
        ]);

        PipCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the pip install command for DeepSpeed.
    ///
    /// The original script uses `pip install deepspeed einops` with
    /// ROCM_HOME and HIP_PATH set. It does NOT set DS_ACCELERATOR=rocm
    /// as that caused ValueError in newer versions.
    pub fn build_install_command(&self) -> PipCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if is_global {
            args.push("--break-system-packages".to_string());
        }
        if self.config.force_reinstall {
            args.push("--force-reinstall".to_string());
        }
        args.push("deepspeed".to_string());
        args.push("einops".to_string());

        PipCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![], // ROCm env set separately
        }
    }

    /// Construct the retry install command with --no-deps.
    ///
    /// Used when the primary install fails.
    pub fn build_no_deps_command(&self) -> PipCommand {
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
            "--no-deps".to_string(),
        ];
        args.push("deepspeed".to_string());

        PipCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the force-reinstall retry command.
    pub fn build_force_reinstall_command(&self) -> PipCommand {
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
            "--force-reinstall".to_string(),
        ];
        args.push("deepspeed".to_string());

        PipCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Get the list of required dependencies to verify after install.
    pub fn required_dependencies(&self) -> &[&str] {
        &[
            "packaging",
            "ninja",
            "pydantic",
            "jsonschema",
            "einops",
            "deepspeed",
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-006: DeepSpeed installer correct pip command ---

    #[test]
    fn test_build_install_command_global() {
        let installer = DeepSpeedInstaller::new(DeepSpeedConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"deepspeed".to_string()));
        assert!(cmd.args.contains(&"einops".to_string()));
    }

    #[test]
    fn test_build_install_command_venv() {
        let installer = DeepSpeedInstaller::new(DeepSpeedConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        // Venv should NOT have --break-system-packages
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"deepspeed".to_string()));
    }

    #[test]
    fn test_build_install_command_force_reinstall() {
        let installer = DeepSpeedInstaller::new(DeepSpeedConfig {
            force_reinstall: true,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
    }

    #[test]
    fn test_build_deps_command() {
        let installer = DeepSpeedInstaller::with_defaults();
        let cmd = installer.build_deps_install_command();
        assert!(cmd.args.contains(&"packaging".to_string()));
        assert!(cmd.args.contains(&"ninja".to_string()));
        assert!(cmd.args.contains(&"pydantic".to_string()));
        assert!(cmd.args.contains(&"jsonschema".to_string()));
        assert!(cmd.args.contains(&"einops".to_string()));
    }

    #[test]
    fn test_build_no_deps_command() {
        let installer = DeepSpeedInstaller::with_defaults();
        let cmd = installer.build_no_deps_command();
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"deepspeed".to_string()));
    }

    #[test]
    fn test_build_force_reinstall_command() {
        let installer = DeepSpeedInstaller::with_defaults();
        let cmd = installer.build_force_reinstall_command();
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
        assert!(cmd.args.contains(&"deepspeed".to_string()));
    }

    #[test]
    fn test_rocm_build_env() {
        let installer = DeepSpeedInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let env = installer.rocm_build_env(&rocm_env);
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_HOME" && v == "/opt/rocm"));
        assert!(env.iter().any(|(k, v)| k == "HIP_PATH" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
    }

    #[test]
    fn test_rocm_build_env_no_rocm() {
        let installer = DeepSpeedInstaller::with_defaults();
        let rocm_env = RocmEnv::none();
        let env = installer.rocm_build_env(&rocm_env);
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_HOME" && v == "/opt/rocm"));
    }

    #[test]
    fn test_required_dependencies() {
        let installer = DeepSpeedInstaller::with_defaults();
        let deps = installer.required_dependencies();
        assert!(deps.contains(&"deepspeed"));
        assert!(deps.contains(&"einops"));
        assert!(deps.contains(&"ninja"));
    }

    #[test]
    fn test_command_string_format() {
        let cmd = PipCommand {
            program: "python3".to_string(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "deepspeed".to_string(),
            ],
            env: vec![("ROCM_HOME".to_string(), "/opt/rocm".to_string())],
        };
        let s = cmd.to_command_string();
        assert!(s.contains("ROCM_HOME=/opt/rocm"));
        assert!(s.contains("python3 -m pip install deepspeed"));
    }
}
