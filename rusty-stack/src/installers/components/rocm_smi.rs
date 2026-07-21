//! ROCm SMI installer — ports `scripts/install_rocm_smi.sh`.
//!
//! Constructs correct package manager commands per distro for installing
//! the ROCm System Management Interface. `rocm_smi_lib` has no `setup.py`
//! (it is a C++ library whose Python tools ship via the distro package),
//! so this installer is package-manager + git-clone only — no pip step.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-018**: ROCm SMI correct package command

use crate::installers::common::DistroFacade;
use crate::platform::detection::DistroFamily;
use std::path::PathBuf;

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

/// The ROCm SMI installer — a pure package-manager dispatcher (no Python,
/// no build config): `rocm_smi_lib` ships via the distro package, so this
/// installer holds no configuration state.
#[derive(Default)]
pub struct RocmSmiInstaller;

impl RocmSmiInstaller {
    /// Create a new ROCm SMI installer.
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Git clone
    // -----------------------------------------------------------------------

    /// Get the ROCm SMI library git clone URL.
    pub fn git_clone_url(&self) -> &'static str {
        "https://github.com/RadeonOpenCompute/rocm_smi_lib.git"
    }

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
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // Package manager commands (VAL-INSTALL-018)
    // -----------------------------------------------------------------------

    /// Construct the package install command for the given distro.
    ///
    /// Maps to the correct package manager commands per distro:
    /// - Debian/Ubuntu: `sudo apt install -y rocm-smi`
    /// - Fedora/RHEL: `sudo dnf install -y rocm-smi`
    /// - Arch: `sudo pacman -S --needed --noconfirm rocm-smi`
    /// - SUSE: `sudo zypper install -y rocm-smi`
    pub fn build_package_install_command(&self, distro: &DistroFacade) -> ShellCommand {
        let (program, args) = match distro.family() {
            DistroFamily::Debian => (
                "sudo".to_string(),
                vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rocm-smi".to_string(),
                ],
            ),
            DistroFamily::Rhel => (
                "sudo".to_string(),
                vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rocm-smi".to_string(),
                ],
            ),
            DistroFamily::Arch => (
                "sudo".to_string(),
                vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--needed".to_string(),
                    "--noconfirm".to_string(),
                    "rocm-smi-lib".to_string(),
                ],
            ),
            DistroFamily::Suse => (
                "sudo".to_string(),
                vec![
                    "zypper".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rocm-smi".to_string(),
                ],
            ),
            _ => (
                "sudo".to_string(),
                vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rocm-smi".to_string(),
                ],
            ),
        };

        ShellCommand {
            program,
            args,
            env: vec![],
            working_dir: None,
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
            working_dir: None,
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

    // --- VAL-INSTALL-018: ROCm SMI correct package command ---

    #[test]
    fn test_git_clone_url() {
        let installer = RocmSmiInstaller::new();
        assert_eq!(
            installer.git_clone_url(),
            "https://github.com/RadeonOpenCompute/rocm_smi_lib.git"
        );
    }

    #[test]
    fn test_git_clone_command() {
        let installer = RocmSmiInstaller::new();
        let cmd = installer.build_git_clone_command("/tmp/rocm_smi");
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("RadeonOpenCompute/rocm_smi_lib")));
    }

    #[test]
    fn test_package_install_debian() {
        let installer = RocmSmiInstaller::new();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        let cmd = installer.build_package_install_command(&distro);
        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"apt-get".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
        assert!(cmd.args.contains(&"rocm-smi".to_string()));
    }

    #[test]
    fn test_package_install_rhel() {
        let installer = RocmSmiInstaller::new();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        });
        let cmd = installer.build_package_install_command(&distro);
        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"dnf".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"rocm-smi".to_string()));
    }

    #[test]
    fn test_package_install_arch() {
        let installer = RocmSmiInstaller::new();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let cmd = installer.build_package_install_command(&distro);
        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"pacman".to_string()));
        assert!(cmd.args.contains(&"-S".to_string()));
        assert!(cmd.args.contains(&"--needed".to_string()));
        assert!(cmd.args.contains(&"--noconfirm".to_string()));
        assert!(cmd.args.contains(&"rocm-smi-lib".to_string()));
    }

    #[test]
    fn test_package_install_suse() {
        let installer = RocmSmiInstaller::new();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "opensuse".to_string(),
            family: DistroFamily::Suse,
            pkg_manager: PackageManager::Zypper,
            ..Default::default()
        });
        let cmd = installer.build_package_install_command(&distro);
        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"zypper".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"rocm-smi".to_string()));
    }

    #[test]
    fn test_package_update_debian() {
        let installer = RocmSmiInstaller::new();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        let cmd = installer.build_package_update_command(&distro);
        assert!(cmd.args.contains(&"apt-get".to_string()));
        assert!(cmd.args.contains(&"update".to_string()));
    }
}
