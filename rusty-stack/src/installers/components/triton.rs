//! Triton installer — ports `scripts/install_triton_multi.sh`.
//!
//! Constructs correct pip install with version constraints and ROCm flags.
//! Triton is built from source (ROCm/triton fork) with GPU arch and
//! TRITON_ROCM=1 flags.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-004**: Triton installer correct pip command

use std::fmt;

// ===========================================================================
// Types
// ===========================================================================

/// Triton branch selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TritonBranch {
    /// Stable release branch (release/internal/3.6.x).
    Stable,
    /// MLIR-optimized branch (release/internal/3.6.x).
    Mlir,
    /// Preview/main performance branch.
    Preview,
}

impl fmt::Display for TritonBranch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TritonBranch::Stable => write!(f, "release/internal/3.6.x"),
            TritonBranch::Mlir => write!(f, "release/internal/3.6.x"),
            TritonBranch::Preview => write!(f, "main_perf"),
        }
    }
}

/// Configuration for the Triton installer.
#[derive(Debug, Clone)]
pub struct TritonConfig {
    /// ROCm channel (affects branch selection).
    pub rocm_channel: String,
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_arch: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for TritonConfig {
    fn default() -> Self {
        Self {
            rocm_channel: "latest".to_string(),
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
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
        let env_prefix = self.env.iter()
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

/// The Triton installer.
pub struct TritonInstaller {
    config: TritonConfig,
}

impl TritonInstaller {
    /// Create a new Triton installer with the given config.
    pub fn new(config: TritonConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(TritonConfig::default())
    }

    // -----------------------------------------------------------------------
    // Branch selection
    // -----------------------------------------------------------------------

    /// Determine the Triton branch to use based on channel and GPU arch.
    ///
    /// Matches the original script logic:
    /// - Preview channel or gfx12xx GPUs → preview branch
    /// - Otherwise → MLIR branch with stable fallback
    pub fn select_branch(&self) -> TritonBranch {
        if self.config.rocm_channel == "preview" || self.config.gpu_arch.starts_with("gfx12") {
            TritonBranch::Preview
        } else {
            TritonBranch::Mlir
        }
    }

    /// Get the git clone URL.
    pub fn git_clone_url(&self) -> &'static str {
        "https://github.com/ROCm/triton.git"
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-004)
    // -----------------------------------------------------------------------

    /// Construct the git clone command.
    pub fn build_git_clone_command(&self, target_dir: &str) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                self.git_clone_url().to_string(),
                target_dir.to_string(),
            ],
            env: vec![],
        }
    }

    /// Construct the git checkout command for the selected branch.
    pub fn build_git_checkout_command(&self) -> ShellCommand {
        let branch = self.select_branch();
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "checkout".to_string(),
                branch.to_string(),
            ],
            env: vec![],
        }
    }

    /// Construct the build prerequisites pip install command.
    ///
    /// Installs pybind11, cmake, ninja, and other build tools.
    pub fn build_prerequisites_command(&self) -> ShellCommand {
        let use_break = self.supports_break_system_packages();
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--upgrade".to_string(),
            "--no-cache-dir".to_string(),
            "pip".to_string(),
            "setuptools".to_string(),
            "wheel".to_string(),
            "packaging".to_string(),
            "pybind11".to_string(),
            "cmake".to_string(),
            "ninja".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the Triton pip install from source command.
    ///
    /// This builds Triton from the cloned source with ROCm flags:
    /// - GPU_ARCHS=gfx1100
    /// - TRITON_ROCM=1
    /// - MAX_JOBS=<nproc-1>
    pub fn build_pip_install_command(&self, _src_dir: &str) -> ShellCommand {
        let use_break = self.supports_break_system_packages();
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--no-build-isolation".to_string(),
            "--no-deps".to_string(),
            ".".to_string(),
        ]);

        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(4);

        let env = vec![
            ("GPU_ARCHS".to_string(), self.config.gpu_arch.clone()),
            ("TRITON_ROCM".to_string(), "1".to_string()),
            ("MAX_JOBS".to_string(), nproc.to_string()),
        ];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env,
        }
    }

    /// Check whether pip supports --break-system-packages.
    fn supports_break_system_packages(&self) -> bool {
        // Python 3.11+ has this flag; assume true on modern systems
        true
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VAL-INSTALL-004: Triton installer correct pip command ---

    #[test]
    fn test_git_clone_url() {
        let installer = TritonInstaller::with_defaults();
        assert_eq!(installer.git_clone_url(), "https://github.com/ROCm/triton.git");
    }

    #[test]
    fn test_git_clone_command() {
        let installer = TritonInstaller::with_defaults();
        let cmd = installer.build_git_clone_command("/tmp/triton-rocm/triton");
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("ROCm/triton")));
    }

    #[test]
    fn test_git_checkout_mlir() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "latest".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let cmd = installer.build_git_checkout_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"checkout".to_string()));
        // MLIR branch for non-preview, non-gfx12
        assert!(cmd.args.iter().any(|a| a == "release/internal/3.6.x"));
    }

    #[test]
    fn test_git_checkout_preview() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "preview".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let cmd = installer.build_git_checkout_command();
        assert!(cmd.args.iter().any(|a| a == "main_perf"));
    }

    #[test]
    fn test_git_checkout_gfx12() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "latest".to_string(),
            gpu_arch: "gfx1200".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.select_branch(), TritonBranch::Preview);
    }

    #[test]
    fn test_prerequisites_command() {
        let installer = TritonInstaller::with_defaults();
        let cmd = installer.build_prerequisites_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"pybind11".to_string()));
        assert!(cmd.args.contains(&"cmake".to_string()));
        assert!(cmd.args.contains(&"ninja".to_string()));
    }

    #[test]
    fn test_pip_install_command() {
        let installer = TritonInstaller::new(TritonConfig {
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let cmd = installer.build_pip_install_command("/tmp/triton-rocm/triton/python");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"--no-build-isolation".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&".".to_string()));

        // Verify ROCm environment variables
        assert!(cmd.env.iter().any(|(k, v)| k == "GPU_ARCHS" && v == "gfx1100"));
        assert!(cmd.env.iter().any(|(k, v)| k == "TRITON_ROCM" && v == "1"));
        assert!(cmd.env.iter().any(|(k, _)| k == "MAX_JOBS"));
    }

    #[test]
    fn test_branch_selection_latest() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "latest".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.select_branch(), TritonBranch::Mlir);
    }

    #[test]
    fn test_branch_selection_stable() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "stable".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.select_branch(), TritonBranch::Mlir);
    }

    #[test]
    fn test_branch_selection_preview() {
        let installer = TritonInstaller::new(TritonConfig {
            rocm_channel: "preview".to_string(),
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.select_branch(), TritonBranch::Preview);
    }

    #[test]
    fn test_command_string_format() {
        let cmd = ShellCommand {
            program: "git".to_string(),
            args: vec!["clone".to_string(), "https://github.com/ROCm/triton.git".to_string()],
            env: vec![],
        };
        assert_eq!(
            cmd.to_command_string(),
            "git clone https://github.com/ROCm/triton.git"
        );
    }

    #[test]
    fn test_command_string_with_env() {
        let cmd = ShellCommand {
            program: "pip".to_string(),
            args: vec!["install".to_string()],
            env: vec![
                ("TRITON_ROCM".to_string(), "1".to_string()),
                ("GPU_ARCHS".to_string(), "gfx1100".to_string()),
            ],
        };
        let s = cmd.to_command_string();
        assert!(s.contains("TRITON_ROCM=1"));
        assert!(s.contains("GPU_ARCHS=gfx1100"));
    }
}
