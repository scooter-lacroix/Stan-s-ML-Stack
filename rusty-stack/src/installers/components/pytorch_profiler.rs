//! PyTorch Profiler installer — ports `scripts/install_pytorch_profiler.sh`.
//!
//! Constructs correct pip install command for PyTorch Profiler (torch-tb-profiler)
//! and TensorBoard. Depends on PyTorch being installed first.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-020**: PyTorch Profiler correct pip command

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

impl std::fmt::Display for InstallMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallMethod::Global => write!(f, "global"),
            InstallMethod::Venv => write!(f, "venv"),
            InstallMethod::Auto => write!(f, "auto"),
        }
    }
}

/// Configuration for the PyTorch Profiler installer.
#[derive(Debug, Clone)]
pub struct PytorchProfilerConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether strict ROCm mode is enabled.
    pub strict_rocm: bool,
}

impl Default for PytorchProfilerConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
            strict_rocm: true,
        }
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

/// The PyTorch Profiler installer.
pub struct PytorchProfilerInstaller {
    config: PytorchProfilerConfig,
}

impl PytorchProfilerInstaller {
    /// Create a new PyTorch Profiler installer with the given config.
    pub fn new(config: PytorchProfilerConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(PytorchProfilerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependencies
    // -----------------------------------------------------------------------

    /// PyTorch Profiler depends on PyTorch.
    pub fn dependencies(&self) -> &[&str] {
        &["pytorch"]
    }

    /// Validate that all dependencies are satisfied.
    pub fn validate_dependencies(&self, installed_components: &[&str]) -> anyhow::Result<()> {
        for dep in self.dependencies() {
            if !installed_components.contains(dep) {
                anyhow::bail!(
                    "PyTorch Profiler requires '{}' to be installed first",
                    dep
                );
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-020)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for PyTorch Profiler.
    ///
    /// The original script installs:
    /// `pip install torch-tb-profiler tensorboard`
    pub fn build_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "torch-tb-profiler".to_string(),
            "tensorboard".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the ROCm environment setup for the profiler.
    pub fn build_rocm_env(&self) -> Vec<(String, String)> {
        vec![
            ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
            (
                "PATH".to_string(),
                "/opt/rocm/bin:".to_string() + &std::env::var("PATH").unwrap_or_default(),
            ),
            (
                "LD_LIBRARY_PATH".to_string(),
                "/opt/rocm/lib:".to_string()
                    + &std::env::var("LD_LIBRARY_PATH").unwrap_or_default(),
            ),
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
            ("PYTORCH_ROCM_ARCH".to_string(), "gfx1100".to_string()),
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VAL-INSTALL-020: PyTorch Profiler correct pip command ---

    #[test]
    fn test_install_command_global() {
        let installer = PytorchProfilerInstaller::new(PytorchProfilerConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"torch-tb-profiler".to_string()));
        assert!(cmd.args.contains(&"tensorboard".to_string()));
    }

    #[test]
    fn test_install_command_venv() {
        let installer = PytorchProfilerInstaller::new(PytorchProfilerConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"torch-tb-profiler".to_string()));
        assert!(cmd.args.contains(&"tensorboard".to_string()));
    }

    #[test]
    fn test_dependencies() {
        let installer = PytorchProfilerInstaller::with_defaults();
        assert!(installer.dependencies().contains(&"pytorch"));
    }

    #[test]
    fn test_validate_dependencies_success() {
        let installer = PytorchProfilerInstaller::with_defaults();
        assert!(installer
            .validate_dependencies(&["pytorch"])
            .is_ok());
    }

    #[test]
    fn test_validate_dependencies_missing() {
        let installer = PytorchProfilerInstaller::with_defaults();
        let result = installer.validate_dependencies(&[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("pytorch"));
    }

    #[test]
    fn test_rocm_env() {
        let installer = PytorchProfilerInstaller::with_defaults();
        let env = installer.build_rocm_env();
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
    }
}
