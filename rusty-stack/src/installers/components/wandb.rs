//! W&B (Weights & Biases) installer — ports `scripts/install_wandb.sh`.
//!
//! Constructs correct pip install command for the wandb package.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-021**: W&B correct pip command

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

/// Configuration for the W&B installer.
#[derive(Debug, Clone)]
pub struct WandbConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether strict ROCm mode is enabled.
    pub strict_rocm: bool,
}

impl Default for WandbConfig {
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

/// The W&B installer.
pub struct WandbInstaller {
    config: WandbConfig,
}

impl WandbInstaller {
    /// Create a new W&B installer with the given config.
    pub fn new(config: WandbConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(WandbConfig::default())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-021)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for wandb.
    ///
    /// The original script installs:
    /// `pip install wandb`
    /// With optional `--break-system-packages` for global installs.
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
        args.push("wandb".to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the ROCm environment setup for wandb.
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

    // --- VAL-INSTALL-021: W&B correct pip command ---

    #[test]
    fn test_install_command_global() {
        let installer = WandbInstaller::new(WandbConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"wandb".to_string()));
    }

    #[test]
    fn test_install_command_venv() {
        let installer = WandbInstaller::new(WandbConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"wandb".to_string()));
    }

    #[test]
    fn test_install_command_auto() {
        let installer = WandbInstaller::new(WandbConfig {
            method: InstallMethod::Auto,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"wandb".to_string()));
    }

    #[test]
    fn test_command_string_format() {
        let cmd = ShellCommand {
            program: "python3".to_string(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "wandb".to_string(),
            ],
            env: vec![],
        };
        assert_eq!(
            cmd.to_command_string(),
            "python3 -m pip install wandb"
        );
    }

    #[test]
    fn test_command_string_with_env() {
        let cmd = ShellCommand {
            program: "python3".to_string(),
            args: vec!["-m".to_string(), "pip".to_string(), "install".to_string(), "wandb".to_string()],
            env: vec![("ROCM_PATH".to_string(), "/opt/rocm".to_string())],
        };
        let s = cmd.to_command_string();
        assert!(s.contains("ROCM_PATH=/opt/rocm"));
        assert!(s.contains("python3 -m pip install wandb"));
    }

    #[test]
    fn test_rocm_env() {
        let installer = WandbInstaller::with_defaults();
        let env = installer.build_rocm_env();
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
    }
}
