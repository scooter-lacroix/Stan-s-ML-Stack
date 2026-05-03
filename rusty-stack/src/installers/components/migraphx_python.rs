//! MIGraphX Python bindings installer — ports `scripts/install_migraphx_python.sh`.
//!
//! Installs the MIGraphX Python module via pip. Requires MIGraphX system
//! package to be installed first.
//!
//! # Arch Linux / CachyOS Handling
//!
//! On Arch-family distros, the `migraphx` pip wheel is not available.
//! The installer should be skipped on these distros with a clear message
//! informing the user. Use `is_available_on_distro()` to check before
//! attempting installation.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-030**: MIGraphX Python correct pip command

use crate::installers::common::DistroFacade;
use crate::platform::detection::DistroFamily;

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

/// Configuration for the MIGraphX Python installer.
#[derive(Debug, Clone)]
pub struct MigraphxPythonConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether to force reinstall.
    pub force: bool,
}

impl Default for MigraphxPythonConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
            force: false,
        }
    }
}

/// Package name for MIGraphX Python bindings.
pub const PACKAGE_NAME: &str = "migraphx";

/// The MIGraphX Python installer.
pub struct MigraphxPythonInstaller {
    config: MigraphxPythonConfig,
}

impl MigraphxPythonInstaller {
    /// Create a new MIGraphX Python installer with the given config.
    pub fn new(config: MigraphxPythonConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MigraphxPythonConfig::default())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-030)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for MIGraphX Python.
    ///
    /// The original script runs:
    /// `uv pip install migraphx` or `pip install migraphx`
    /// With optional `--break-system-packages` for global installs.
    pub fn build_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        if self.config.force {
            args.push("--force-reinstall".to_string());
        }
        args.push(PACKAGE_NAME.to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the uv pip install command (preferred method).
    pub fn build_uv_install_command(&self) -> ShellCommand {
        let mut args = vec!["pip".to_string(), "install".to_string()];
        if self.config.force {
            args.push("--force-reinstall".to_string());
        }
        args.push(PACKAGE_NAME.to_string());

        ShellCommand {
            program: "uv".to_string(),
            args,
            env: vec![
                ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
                ("AMD_LOG_LEVEL".to_string(), "0".to_string()),
            ],
        }
    }

    /// Construct the command to check if MIGraphX system package is installed.
    pub fn build_migraphx_check_command(&self) -> ShellCommand {
        ShellCommand {
            program: "migraphx-driver".to_string(),
            args: vec!["--version".to_string()],
            env: vec![],
        }
    }

    /// Construct the command to verify MIGraphX Python module import.
    pub fn build_python_import_check(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-c".to_string(),
                "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))".to_string(),
            ],
            env: vec![],
        }
    }

    /// Construct the ROCm environment setup for the install.
    pub fn build_rocm_env(&self) -> Vec<(String, String)> {
        let gpu_visible = std::env::var("HIP_VISIBLE_DEVICES")
            .or_else(|_| std::env::var("CUDA_VISIBLE_DEVICES"))
            .unwrap_or_else(|_| "0".to_string());

        vec![
            ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
            ("AMD_LOG_LEVEL".to_string(), "0".to_string()),
            ("HIP_VISIBLE_DEVICES".to_string(), gpu_visible.clone()),
            ("ROCR_VISIBLE_DEVICES".to_string(), gpu_visible),
        ]
    }

    /// Get the package name.
    pub fn package_name(&self) -> &'static str {
        PACKAGE_NAME
    }

    // -----------------------------------------------------------------------
    // Distro availability (Arch-specific handling)
    // -----------------------------------------------------------------------

    /// Check whether MIGraphX Python bindings are available on the given distro.
    ///
    /// On Arch-family distros, the `migraphx` pip wheel does not exist in PyPI
    /// and there is no `python3-migraphx` system package. The installation
    /// should be skipped on these distros.
    pub fn is_available_on_distro(&self, distro: &DistroFacade) -> bool {
        !matches!(distro.family(), DistroFamily::Arch)
    }

    /// Build a human-readable message explaining why MIGraphX Python bindings
    /// are not available on the given distro. Returns `None` if available.
    pub fn build_unavailable_message(&self, distro: &DistroFacade) -> Option<String> {
        if self.is_available_on_distro(distro) {
            return None;
        }
        Some(format!(
            "MIGraphX Python bindings (pip install migraphx) are not available on {}. \
             The migraphx pip wheel is not published for Arch-family distros. \
             If you need Python bindings, consider using the ROCm Docker image \
             or building MIGraphX from source with Python bindings enabled.",
            distro.id()
        ))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-INSTALL-030: MIGraphX Python correct pip command
    // -----------------------------------------------------------------------

    #[test]
    fn test_package_name_matches_original_script() {
        assert_eq!(
            PACKAGE_NAME, "migraphx",
            "Package name must match install_migraphx_python.sh package"
        );
    }

    #[test]
    fn test_install_command_auto_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Auto,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_install_command_global_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_install_command_venv_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        // Venv method should NOT include --break-system-packages
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }

    #[test]
    fn test_install_command_with_force() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            force: true,
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
    }

    #[test]
    fn test_install_command_string() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_install_command();
        let s = cmd.to_command_string();
        assert!(s.contains("python3"));
        assert!(s.contains("pip install"));
        assert!(s.contains("migraphx"));
    }

    #[test]
    fn test_uv_install_command() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_uv_install_command();
        assert_eq!(cmd.program, "uv");
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
        // Should have ROCm env vars
        assert!(cmd.env.iter().any(|(k, _)| k == "ROCM_PATH"));
        assert!(cmd.env.iter().any(|(k, _)| k == "AMD_LOG_LEVEL"));
    }

    #[test]
    fn test_migraphx_check_command() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_migraphx_check_command();
        assert_eq!(cmd.program, "migraphx-driver");
        assert!(cmd.args.contains(&"--version".to_string()));
    }

    #[test]
    fn test_python_import_check() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_python_import_check();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.iter().any(|a| a.contains("import migraphx")));
    }

    #[test]
    fn test_rocm_env() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let env = installer.build_rocm_env();
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env.iter().any(|(k, v)| k == "AMD_LOG_LEVEL" && v == "0"));
    }

    // --- Arch-specific availability tests ---

    #[test]
    fn test_is_available_debian() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(
            installer.is_available_on_distro(&distro),
            "MIGraphX Python should be available on Debian"
        );
    }

    #[test]
    fn test_is_available_rhel() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        });
        assert!(
            installer.is_available_on_distro(&distro),
            "MIGraphX Python should be available on RHEL"
        );
    }

    #[test]
    fn test_not_available_on_arch() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert!(
            !installer.is_available_on_distro(&distro),
            "MIGraphX Python should NOT be available on Arch"
        );
    }

    #[test]
    fn test_not_available_on_cachyos() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "cachyos".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert!(
            !installer.is_available_on_distro(&distro),
            "MIGraphX Python should NOT be available on CachyOS"
        );
    }

    #[test]
    fn test_unavailable_message_debian_is_none() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(
            installer.build_unavailable_message(&distro).is_none(),
            "Debian should not have an unavailable message"
        );
    }

    #[test]
    fn test_unavailable_message_arch_is_informative() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let msg = installer
            .build_unavailable_message(&distro)
            .expect("Arch should have an unavailable message");
        assert!(
            msg.contains("not available"),
            "Message should say not available: {msg}"
        );
        assert!(
            msg.contains("Arch"),
            "Message should mention Arch: {msg}"
        );
        assert!(
            msg.contains("pip"),
            "Message should mention pip: {msg}"
        );
        assert!(
            msg.contains("Docker") || msg.contains("source"),
            "Message should suggest alternatives: {msg}"
        );
    }
}
