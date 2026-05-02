//! MIGraphX installer — ports `scripts/install_migraphx_multi.sh`.
//!
//! Constructs correct pip install with version for MIGraphX Python bindings.
//! Also installs system packages via package manager.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-019**: MIGraphX correct pip command

use crate::installers::common::DistroFacade;
use crate::platform::detection::DistroFamily;

// ===========================================================================
// Types
// ===========================================================================

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

/// Configuration for the MIGraphX installer.
#[derive(Debug, Clone)]
pub struct MigraphxConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for MigraphxConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            dry_run: false,
        }
    }
}

/// The MIGraphX installer.
pub struct MigraphxInstaller {
    config: MigraphxConfig,
}

impl MigraphxInstaller {
    /// Create a new MIGraphX installer with the given config.
    pub fn new(config: MigraphxConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MigraphxConfig::default())
    }

    // -----------------------------------------------------------------------
    // Package lists per distro (VAL-INSTALL-019)
    // -----------------------------------------------------------------------

    /// Get the required system packages for the given distro.
    pub fn required_packages(&self, distro: &DistroFacade) -> Vec<&str> {
        match distro.family() {
            DistroFamily::Arch => vec!["migraphx"],
            _ => vec!["migraphx", "migraphx-dev"],
        }
    }

    /// Get the optional system packages for the given distro.
    pub fn optional_packages(&self, distro: &DistroFacade) -> Vec<&str> {
        match distro.family() {
            DistroFamily::Arch => vec!["python3-migraphx", "half"],
            _ => vec!["python3-migraphx", "half"],
        }
    }

    /// Construct the package install command for the given distro.
    pub fn build_package_install_command(
        &self,
        distro: &DistroFacade,
        packages: &[&str],
    ) -> ShellCommand {
        let (program, mut base_args) = match distro.family() {
            DistroFamily::Debian => (
                "sudo".to_string(),
                vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                ],
            ),
            DistroFamily::Rhel => (
                "sudo".to_string(),
                vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                ],
            ),
            DistroFamily::Arch => (
                "sudo".to_string(),
                vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--needed".to_string(),
                    "--noconfirm".to_string(),
                ],
            ),
            DistroFamily::Suse => (
                "sudo".to_string(),
                vec![
                    "zypper".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                ],
            ),
            _ => (
                "sudo".to_string(),
                vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                ],
            ),
        };

        for pkg in packages {
            base_args.push(pkg.to_string());
        }

        ShellCommand {
            program,
            args: base_args,
            env: vec![],
        }
    }

    /// Construct the package update command for the given distro.
    pub fn build_package_update_command(&self, distro: &DistroFacade) -> ShellCommand {
        let (program, args) = match distro.family() {
            DistroFamily::Debian => (
                "sudo".to_string(),
                vec!["apt-get".to_string(), "update".to_string(), "-y".to_string()],
            ),
            DistroFamily::Rhel => (
                "sudo".to_string(),
                vec!["dnf".to_string(), "makecache".to_string()],
            ),
            DistroFamily::Arch => (
                "sudo".to_string(),
                vec!["pacman".to_string(), "-Sy".to_string()],
            ),
            DistroFamily::Suse => (
                "sudo".to_string(),
                vec!["zypper".to_string(), "refresh".to_string()],
            ),
            _ => (
                "sudo".to_string(),
                vec!["apt-get".to_string(), "update".to_string(), "-y".to_string()],
            ),
        };

        ShellCommand {
            program,
            args,
            env: vec![],
        }
    }

    /// Construct the pip install command for MIGraphX Python bindings.
    ///
    /// The original script installs system packages and verifies Python import.
    /// The pip install is from the ROCm Python path.
    pub fn build_pip_install_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--break-system-packages".to_string(),
                "migraphx".to_string(),
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
    use crate::platform::detection::{DistroInfo, PackageManager};

    // --- VAL-INSTALL-019: MIGraphX correct pip command ---

    #[test]
    fn test_required_packages_debian() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        let pkgs = installer.required_packages(&distro);
        assert!(pkgs.contains(&"migraphx"));
        assert!(pkgs.contains(&"migraphx-dev"));
    }

    #[test]
    fn test_required_packages_arch() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let pkgs = installer.required_packages(&distro);
        assert!(pkgs.contains(&"migraphx"));
        assert!(!pkgs.contains(&"migraphx-dev"));
    }

    #[test]
    fn test_optional_packages() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        let pkgs = installer.optional_packages(&distro);
        assert!(pkgs.contains(&"python3-migraphx"));
        assert!(pkgs.contains(&"half"));
    }

    #[test]
    fn test_package_install_debian() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        let cmd =
            installer.build_package_install_command(&distro, &["migraphx", "migraphx-dev"]);
        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"apt-get".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
        assert!(cmd.args.contains(&"migraphx-dev".to_string()));
    }

    #[test]
    fn test_package_install_arch() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let cmd = installer.build_package_install_command(&distro, &["migraphx"]);
        assert!(cmd.args.contains(&"pacman".to_string()));
        assert!(cmd.args.contains(&"-S".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }

    #[test]
    fn test_package_install_rhel() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        });
        let cmd =
            installer.build_package_install_command(&distro, &["migraphx", "migraphx-dev"]);
        assert!(cmd.args.contains(&"dnf".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }

    #[test]
    fn test_pip_install_command() {
        let installer = MigraphxInstaller::with_defaults();
        let cmd = installer.build_pip_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }
}
