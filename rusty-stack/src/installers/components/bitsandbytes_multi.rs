//! bitsandbytes installer — ports `scripts/install_bitsandbytes_multi.sh`.
//!
//! Constructs correct pip install with version constraints for ROCm.
//! Builds from ROCm fork when PyPI version lacks HIP binaries.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-017**: bitsandbytes correct pip command

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

/// Configuration for the bitsandbytes installer.
#[derive(Debug, Clone)]
pub struct BitsAndBytesConfig {
    /// ROCm version string (e.g., "7.2.0").
    pub rocm_version: String,
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_arch: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for BitsAndBytesConfig {
    fn default() -> Self {
        Self {
            rocm_version: "7.2.0".to_string(),
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
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

/// The bitsandbytes installer.
pub struct BitsAndBytesInstaller {
    config: BitsAndBytesConfig,
}

impl BitsAndBytesInstaller {
    /// Create a new bitsandbytes installer with the given config.
    pub fn new(config: BitsAndBytesConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(BitsAndBytesConfig::default())
    }

    // -----------------------------------------------------------------------
    // Git clone URLs
    // -----------------------------------------------------------------------

    /// Get the primary ROCm bitsandbytes fork URL.
    pub fn rocm_fork_url(&self) -> &'static str {
        "https://github.com/ROCm/bitsandbytes.git"
    }

    /// Get the fallback TimDettmers bitsandbytes fork URL.
    pub fn fallback_fork_url(&self) -> &'static str {
        "https://github.com/TimDettmers/bitsandbytes.git"
    }

    /// Construct the git clone command for the ROCm fork.
    pub fn build_git_clone_command(&self, target_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--recursive".to_string(),
                self.rocm_fork_url().to_string(),
                target_dir.to_string(),
            ],
            env: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-017)
    // -----------------------------------------------------------------------

    /// Construct the initial pip install command for bitsandbytes from PyPI.
    ///
    /// The original script tries:
    /// `pip install bitsandbytes --break-system-packages --no-cache-dir --no-deps`
    pub fn build_pypi_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "bitsandbytes".to_string(),
            "--no-cache-dir".to_string(),
            "--no-deps".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the build environment variables for ROCm source build.
    pub fn build_rocm_env(&self) -> Vec<(String, String)> {
        let rocm_path = "/opt/rocm".to_string();
        let version_short = self.rocm_version_short();

        vec![
            ("BNB_USE_ROCM".to_string(), "1".to_string()),
            ("BNB_ROCM_ARCH".to_string(), self.config.gpu_arch.clone()),
            ("ROCM_PATH".to_string(), rocm_path.clone()),
            ("ROCM_HOME".to_string(), rocm_path.clone()),
            ("HIP_PATH".to_string(), rocm_path),
            ("BNB_ROCM_VERSION".to_string(), version_short),
        ]
    }

    /// Extract version shortcode (e.g., "7.2.0" -> "72").
    fn rocm_version_short(&self) -> String {
        let parts: Vec<&str> = self.config.rocm_version.split('.').collect();
        if parts.len() >= 2 {
            format!("{}{}", parts[0], parts[1])
        } else {
            "72".to_string()
        }
    }

    /// Construct the build dependencies pip install command.
    pub fn build_build_deps_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "scikit-build-core".to_string(),
            "cmake".to_string(),
            "ninja".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the source build pip install command.
    ///
    /// The original script uses:
    /// `pip install . --no-build-isolation --break-system-packages
    ///  -Ccmake.define.COMPUTE_BACKEND=hip --no-deps
    ///  -Ccmake.define.ROCM_PATH=/opt/rocm`
    pub fn build_source_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            ".".to_string(),
            "--no-build-isolation".to_string(),
            "--no-deps".to_string(),
            "-Ccmake.define.COMPUTE_BACKEND=hip".to_string(),
            "-Ccmake.define.ROCM_PATH=/opt/rocm".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: self.build_rocm_env(),
        }
    }

    /// Construct the uninstall command for bitsandbytes.
    pub fn build_uninstall_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "uninstall".to_string(),
                "-y".to_string(),
                "bitsandbytes".to_string(),
            ],
            env: vec![],
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VAL-INSTALL-017: bitsandbytes correct pip command ---

    #[test]
    fn test_pypi_install_command() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_pypi_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"bitsandbytes".to_string()));
        assert!(cmd.args.contains(&"--no-cache-dir".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
    }

    #[test]
    fn test_pypi_install_command_venv() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_pypi_install_command();
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"bitsandbytes".to_string()));
    }

    #[test]
    fn test_rocm_fork_url() {
        let installer = BitsAndBytesInstaller::with_defaults();
        assert_eq!(
            installer.rocm_fork_url(),
            "https://github.com/ROCm/bitsandbytes.git"
        );
    }

    #[test]
    fn test_fallback_fork_url() {
        let installer = BitsAndBytesInstaller::with_defaults();
        assert_eq!(
            installer.fallback_fork_url(),
            "https://github.com/TimDettmers/bitsandbytes.git"
        );
    }

    #[test]
    fn test_git_clone_command() {
        let installer = BitsAndBytesInstaller::with_defaults();
        let cmd = installer.build_git_clone_command("/tmp/bnb");
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&"--recursive".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("ROCm/bitsandbytes")));
    }

    #[test]
    fn test_build_rocm_env() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            rocm_version: "7.2.0".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let env = installer.build_rocm_env();
        assert!(env.iter().any(|(k, v)| k == "BNB_USE_ROCM" && v == "1"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "BNB_ROCM_ARCH" && v == "gfx1100"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "BNB_ROCM_VERSION" && v == "72"));
    }

    #[test]
    fn test_rocm_version_short() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            rocm_version: "7.2.0".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.rocm_version_short(), "72");
    }

    #[test]
    fn test_rocm_version_short_legacy() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            rocm_version: "7.0.0".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.rocm_version_short(), "70");
    }

    #[test]
    fn test_build_deps_command() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_build_deps_command();
        assert!(cmd.args.contains(&"scikit-build-core".to_string()));
        assert!(cmd.args.contains(&"cmake".to_string()));
        assert!(cmd.args.contains(&"ninja".to_string()));
    }

    #[test]
    fn test_source_install_command() {
        let installer = BitsAndBytesInstaller::new(BitsAndBytesConfig {
            rocm_version: "7.2.0".to_string(),
            gpu_arch: "gfx1100".to_string(),
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_source_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"--no-build-isolation".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&".".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("COMPUTE_BACKEND=hip")));
        assert!(cmd.args.iter().any(|a| a.contains("ROCM_PATH=/opt/rocm")));
        // Verify ROCm env vars
        assert!(cmd.env.iter().any(|(k, v)| k == "BNB_USE_ROCM" && v == "1"));
    }

    #[test]
    fn test_uninstall_command() {
        let installer = BitsAndBytesInstaller::with_defaults();
        let cmd = installer.build_uninstall_command();
        assert!(cmd.args.contains(&"uninstall".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
        assert!(cmd.args.contains(&"bitsandbytes".to_string()));
    }
}
