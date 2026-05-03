//! MIGraphX installer — ports `scripts/install_migraphx_multi.sh`.
//!
//! Constructs correct pip install with version for MIGraphX Python bindings.
//! Also installs system packages via package manager.
//!
//! # Arch Linux / CachyOS Handling
//!
//! On Arch-family distros, only the `migraphx` system package is available.
//! The `migraphx-dev` and `python3-migraphx` packages are Debian/Ubuntu-specific.
//! The pip `migraphx` Python wheel is also not available for Arch.
//! The installer installs the system package only and skips pip install with
//! a clear informational message.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-019**: MIGraphX correct pip command

use crate::installers::common::DistroFacade;
use crate::platform::detection::DistroFamily;

// ===========================================================================
// Types
// ===========================================================================

/// MIGraphX support level for a given distro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigraphxSupport {
    /// Full support: system packages + Python bindings available.
    Full,
    /// Partial support: system package only, no Python bindings.
    SystemOnly,
    /// Not available on this distro.
    Unavailable,
}

impl std::fmt::Display for MigraphxSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigraphxSupport::Full => write!(f, "full"),
            MigraphxSupport::SystemOnly => write!(f, "system-only"),
            MigraphxSupport::Unavailable => write!(f, "unavailable"),
        }
    }
}

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
                vec!["dnf".to_string(), "install".to_string(), "-y".to_string()],
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
                vec![
                    "apt-get".to_string(),
                    "update".to_string(),
                    "-y".to_string(),
                ],
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
                vec![
                    "apt-get".to_string(),
                    "update".to_string(),
                    "-y".to_string(),
                ],
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
    ///
    /// **Note:** On Arch-family distros, this command should NOT be executed
    /// because the `migraphx` pip wheel is not available. Use
    /// `is_available_on_distro()` to check first.
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

    // -----------------------------------------------------------------------
    // Distro availability (Arch-specific handling)
    // -----------------------------------------------------------------------

    /// Check the level of MIGraphX support on the given distro.
    ///
    /// - **Debian/Ubuntu**: Full support (system packages + Python bindings).
    /// - **Arch/CachyOS**: System-only support (`migraphx` package available,
    ///   but no `python3-migraphx` or pip wheel).
    /// - **Others**: Assumed full support (will attempt standard install).
    pub fn is_available_on_distro(&self, distro: &DistroFacade) -> MigraphxSupport {
        match distro.family() {
            DistroFamily::Arch => MigraphxSupport::SystemOnly,
            _ => MigraphxSupport::Full,
        }
    }

    /// Build a human-readable message explaining MIGraphX limitations on the
    /// given distro. Returns `None` if there are no limitations (full support).
    pub fn build_limitation_message(&self, distro: &DistroFacade) -> Option<String> {
        match self.is_available_on_distro(distro) {
            MigraphxSupport::Full => None,
            MigraphxSupport::SystemOnly => Some(format!(
                "MIGraphX on {} has limited support: only the system package is available. \
                 Python bindings (pip install migraphx) are not available on Arch-family distros. \
                 The system package ({}) has been installed. \
                 If you need Python bindings, consider using the ROCm Docker image or building from source.",
                distro.id(),
                self.required_packages(distro).join(", ")
            )),
            MigraphxSupport::Unavailable => Some(format!(
                "MIGraphX is not available on {}. \
                 Consider using the ROCm Docker image or building from source.",
                distro.id()
            )),
        }
    }

    /// Check whether the pip install step should be skipped on this distro.
    ///
    /// On Arch-family distros, the `migraphx` pip wheel does not exist and
    /// attempting to install it will fail. This method returns `true` for
    /// those distros so the caller can skip the pip step gracefully.
    pub fn should_skip_pip_install(&self, distro: &DistroFacade) -> bool {
        matches!(self.is_available_on_distro(distro), MigraphxSupport::SystemOnly)
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
        let cmd = installer.build_package_install_command(&distro, &["migraphx", "migraphx-dev"]);
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
        let cmd = installer.build_package_install_command(&distro, &["migraphx", "migraphx-dev"]);
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

    // --- Arch-specific availability tests ---

    #[test]
    fn test_availability_debian_is_full() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert_eq!(installer.is_available_on_distro(&distro), MigraphxSupport::Full);
    }

    #[test]
    fn test_availability_arch_is_system_only() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert_eq!(
            installer.is_available_on_distro(&distro),
            MigraphxSupport::SystemOnly
        );
    }

    #[test]
    fn test_availability_cachyos_is_system_only() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "cachyos".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert_eq!(
            installer.is_available_on_distro(&distro),
            MigraphxSupport::SystemOnly
        );
    }

    #[test]
    fn test_availability_rhel_is_full() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        });
        assert_eq!(installer.is_available_on_distro(&distro), MigraphxSupport::Full);
    }

    #[test]
    fn test_limitation_message_debian_is_none() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(installer.build_limitation_message(&distro).is_none());
    }

    #[test]
    fn test_limitation_message_arch_is_some() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let msg = installer
            .build_limitation_message(&distro)
            .expect("Arch should have a limitation message");
        assert!(
            msg.contains("limited support"),
            "Message should mention limited support: {msg}"
        );
        assert!(
            msg.contains("Python bindings"),
            "Message should mention Python bindings: {msg}"
        );
        assert!(
            msg.contains("Arch"),
            "Message should mention Arch context: {msg}"
        );
    }

    #[test]
    fn test_should_skip_pip_debian_is_false() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(!installer.should_skip_pip_install(&distro));
    }

    #[test]
    fn test_should_skip_pip_arch_is_true() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert!(installer.should_skip_pip_install(&distro));
    }

    #[test]
    fn test_arch_required_packages_no_dev() {
        let installer = MigraphxInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "cachyos".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let pkgs = installer.required_packages(&distro);
        assert!(pkgs.contains(&"migraphx"), "Arch should have migraphx package");
        assert!(
            !pkgs.contains(&"migraphx-dev"),
            "Arch should NOT have migraphx-dev (Debian-specific)"
        );
        assert!(
            !pkgs.contains(&"python3-migraphx"),
            "Arch should NOT have python3-migraphx (Debian-specific)"
        );
    }

    #[test]
    fn test_migraphx_support_display() {
        assert_eq!(format!("{}", MigraphxSupport::Full), "full");
        assert_eq!(format!("{}", MigraphxSupport::SystemOnly), "system-only");
        assert_eq!(format!("{}", MigraphxSupport::Unavailable), "unavailable");
    }
}
