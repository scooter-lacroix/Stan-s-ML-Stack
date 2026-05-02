//! AITER installer — ports `scripts/install_aiter.sh`.
//!
//! Constructs correct git clone URL/branch + pip install commands for
//! AMD Iterative Tensor Runtime (AITER). AITER depends on PyTorch and ROCm.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-016**: AITER correct git clone and pip install
//! - **VAL-INSTALL-046**: AITER declares dependency on PyTorch and ROCm

use crate::installers::common::RocmEnv;

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

/// Configuration for the AITER installer.
#[derive(Debug, Clone)]
pub struct AiterConfig {
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
    /// Whether strict ROCm mode is enabled.
    pub strict_rocm: bool,
    /// Custom AITER repo URL (overrides default).
    pub repo_url: Option<String>,
}

impl Default for AiterConfig {
    fn default() -> Self {
        Self {
            rocm_version: "7.2.0".to_string(),
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
            strict_rocm: true,
            repo_url: None,
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

/// The AITER installer.
pub struct AiterInstaller {
    config: AiterConfig,
}

impl AiterInstaller {
    /// Create a new AITER installer with the given config.
    pub fn new(config: AiterConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(AiterConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependencies (VAL-INSTALL-046)
    // -----------------------------------------------------------------------

    /// AITER depends on PyTorch and ROCm.
    pub fn dependencies(&self) -> &[&str] {
        &["pytorch", "rocm"]
    }

    /// Validate that all dependencies are satisfied.
    pub fn validate_dependencies(&self, installed_components: &[&str]) -> anyhow::Result<()> {
        for dep in self.dependencies() {
            if !installed_components.contains(dep) {
                anyhow::bail!(
                    "AITER requires '{}' to be installed first",
                    dep
                );
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Git clone URL (VAL-INSTALL-016)
    // -----------------------------------------------------------------------

    /// Get the git clone URL for AITER.
    pub fn git_clone_url(&self) -> &str {
        self.config
            .repo_url
            .as_deref()
            .unwrap_or("https://github.com/ROCm/aiter.git")
    }

    /// Construct the git clone command.
    pub fn build_git_clone_command(&self, target_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--recursive".to_string(),
                self.git_clone_url().to_string(),
                target_dir.to_string(),
            ],
            env: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-016)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for AITER build dependencies.
    ///
    /// The original script installs: packaging, pybind11, pandas, einops,
    /// psutil, numpy, setuptools, wheel, typing-extensions, cmake, ninja.
    pub fn build_deps_install_command(&self) -> ShellCommand {
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
            "--no-cache-dir".to_string(),
            "--upgrade".to_string(),
        ]);

        let deps = [
            "packaging",
            "pybind11",
            "pandas",
            "einops",
            "psutil",
            "numpy",
            "setuptools",
            "wheel",
            "typing-extensions",
            "cmake",
            "ninja",
        ];
        for dep in deps {
            args.push(dep.to_string());
        }

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the pip install command for AITER from source.
    ///
    /// The original script uses:
    /// `pip install --no-cache-dir --no-build-isolation --no-deps .`
    pub fn build_pip_install_command(&self, _src_dir: &str) -> ShellCommand {
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
            "--no-cache-dir".to_string(),
            "--no-build-isolation".to_string(),
            "--no-deps".to_string(),
            ".".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the ROCm build environment for AITER.
    pub fn build_rocm_env(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        let primary_arch = self.resolve_primary_arch();

        vec![
            ("GPU_ARCH".to_string(), primary_arch.clone()),
            ("PYTORCH_ROCM_ARCH".to_string(), primary_arch.clone()),
            ("GPU_ARCHS".to_string(), primary_arch),
            ("ROCM_PATH".to_string(), rocm_path),
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
        ]
    }

    /// Construct the AITER JIT directory environment.
    pub fn aiter_jit_env(&self) -> Vec<(String, String)> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let jit_dir = format!("{home}/.mlstack/aiter/jit");
        vec![("AITER_JIT_DIR".to_string(), jit_dir)]
    }

    /// Resolve the primary GPU architecture (first item before semicolon).
    pub fn resolve_primary_arch(&self) -> String {
        self.config
            .gpu_arch
            .split(';')
            .next()
            .unwrap_or("gfx1100")
            .trim()
            .to_string()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-016: AITER correct git clone and pip install ---

    #[test]
    fn test_git_clone_url() {
        let installer = AiterInstaller::with_defaults();
        assert_eq!(
            installer.git_clone_url(),
            "https://github.com/ROCm/aiter.git"
        );
    }

    #[test]
    fn test_git_clone_url_custom() {
        let installer = AiterInstaller::new(AiterConfig {
            repo_url: Some("https://example.com/aiter.git".to_string()),
            ..Default::default()
        });
        assert_eq!(
            installer.git_clone_url(),
            "https://example.com/aiter.git"
        );
    }

    #[test]
    fn test_git_clone_command() {
        let installer = AiterInstaller::with_defaults();
        let cmd = installer.build_git_clone_command("/tmp/aiter");
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&"--recursive".to_string()));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("ROCm/aiter")));
        assert!(cmd.args.contains(&"/tmp/aiter".to_string()));
    }

    #[test]
    fn test_deps_install_command() {
        let installer = AiterInstaller::new(AiterConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_deps_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"packaging".to_string()));
        assert!(cmd.args.contains(&"pybind11".to_string()));
        assert!(cmd.args.contains(&"pandas".to_string()));
        assert!(cmd.args.contains(&"einops".to_string()));
        assert!(cmd.args.contains(&"cmake".to_string()));
        assert!(cmd.args.contains(&"ninja".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_deps_install_command_venv() {
        let installer = AiterInstaller::new(AiterConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_deps_install_command();
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_pip_install_command() {
        let installer = AiterInstaller::new(AiterConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_pip_install_command("/tmp/aiter");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"--no-cache-dir".to_string()));
        assert!(cmd.args.contains(&"--no-build-isolation".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&".".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_build_rocm_env() {
        let installer = AiterInstaller::new(AiterConfig {
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(
            Some(PathBuf::from("/opt/rocm")),
            "7.2.0".to_string(),
        );
        let env = installer.build_rocm_env(&rocm_env);
        assert!(env.iter().any(|(k, v)| k == "GPU_ARCH" && v == "gfx1100"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
    }

    #[test]
    fn test_aiter_jit_env() {
        let installer = AiterInstaller::with_defaults();
        let env = installer.aiter_jit_env();
        assert!(env.iter().any(|(k, _)| k == "AITER_JIT_DIR"));
        assert!(env
            .iter()
            .any(|(_, v)| v.contains(".mlstack/aiter/jit")));
    }

    #[test]
    fn test_resolve_primary_arch() {
        let installer = AiterInstaller::new(AiterConfig {
            gpu_arch: "gfx1100;gfx1101;gfx1102".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.resolve_primary_arch(), "gfx1100");
    }

    #[test]
    fn test_resolve_primary_arch_single() {
        let installer = AiterInstaller::new(AiterConfig {
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.resolve_primary_arch(), "gfx1100");
    }

    // --- VAL-INSTALL-046: AITER declares dependency on PyTorch and ROCm ---

    #[test]
    fn test_dependencies() {
        let installer = AiterInstaller::with_defaults();
        assert!(installer.dependencies().contains(&"pytorch"));
        assert!(installer.dependencies().contains(&"rocm"));
    }

    #[test]
    fn test_validate_dependencies_success() {
        let installer = AiterInstaller::with_defaults();
        assert!(installer
            .validate_dependencies(&["pytorch", "rocm"])
            .is_ok());
    }

    #[test]
    fn test_validate_dependencies_missing_pytorch() {
        let installer = AiterInstaller::with_defaults();
        let result = installer.validate_dependencies(&["rocm"]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("pytorch"));
    }

    #[test]
    fn test_validate_dependencies_missing_rocm() {
        let installer = AiterInstaller::with_defaults();
        let result = installer.validate_dependencies(&["pytorch"]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("rocm"));
    }
}
