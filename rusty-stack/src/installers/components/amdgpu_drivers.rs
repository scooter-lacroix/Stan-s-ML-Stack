//! AMD GPU drivers installer — ports `scripts/install_amdgpu_drivers.sh`.
//!
//! Constructs correct package commands per distro family for AMDGPU driver
//! installation.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-029**: AMD GPU drivers correct package command

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

/// Configuration for the AMDGPU drivers installer.
#[derive(Debug, Clone, Default)]
pub struct AmdgpuConfig {
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether to force reinstallation.
    pub force: bool,
}

/// The AMDGPU drivers installer.
pub struct AmdgpuInstaller {
    #[allow(dead_code)]
    config: AmdgpuConfig,
}

impl AmdgpuInstaller {
    /// Create a new AMDGPU drivers installer with the given config.
    pub fn new(config: AmdgpuConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(AmdgpuConfig::default())
    }

    // -----------------------------------------------------------------------
    // Package commands per distro (VAL-INSTALL-029)
    // -----------------------------------------------------------------------

    /// Get the required system packages for the given distro.
    pub fn required_packages(&self, distro: &DistroFacade) -> Vec<&str> {
        match distro.family() {
            DistroFamily::Debian => vec![
                "amdgpu-dkms",
                "linux-headers-$(uname -r)",
                "linux-modules-extra-$(uname -r)",
            ],
            DistroFamily::Rhel => vec!["amdgpu-dkms", "kernel-devel"],
            DistroFamily::Arch => vec!["amdgpu"],
            DistroFamily::Suse => vec!["amdgpu-dkms", "kernel-default-devel"],
            _ => vec!["amdgpu-dkms"],
        }
    }

    /// Get the ROCm packages for the given distro.
    pub fn rocm_packages(&self, _distro: &DistroFacade) -> Vec<&str> {
        vec!["rocm-dev", "rocm-libs", "rocprofiler"]
    }

    /// Construct the package install command for the given distro.
    ///
    /// The original script uses:
    /// - apt: `sudo apt install -y <packages>`
    /// - dnf: `sudo dnf install -y <packages>`
    /// - pacman: `sudo pacman -S --noconfirm <packages>`
    /// - zypper: `sudo zypper install -y <packages>`
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

    /// Construct the apt-specific download and install command for amdgpu-install.
    ///
    /// The original script (Ubuntu-specific):
    /// 1. `wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/$CODENAME/amdgpu-install_6.4.60400-1_all.deb`
    /// 2. `sudo apt install -y ./amdgpu-install_6.4.60400-1_all.deb`
    /// 3. `sudo apt update`
    /// 4. `sudo apt install -y linux-headers-$(uname -r) linux-modules-extra-$(uname -r)`
    /// 5. `sudo apt install -y amdgpu-dkms`
    pub fn build_amdgpu_install_deb_commands(&self, codename: &str) -> Vec<ShellCommand> {
        let deb_url = format!(
            "https://repo.radeon.com/amdgpu-install/6.4/ubuntu/{}/amdgpu-install_6.4.60400-1_all.deb",
            codename
        );
        let deb_file = "amdgpu-install_6.4.60400-1_all.deb";

        vec![
            // Download
            ShellCommand {
                program: "wget".to_string(),
                args: vec!["-q".to_string(), deb_url],
                env: vec![],
            },
            // Install deb
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    format!("./{deb_file}"),
                ],
                env: vec![],
            },
            // Update
            ShellCommand {
                program: "sudo".to_string(),
                args: vec!["apt".to_string(), "update".to_string()],
                env: vec![],
            },
            // Linux headers
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "linux-headers-$(uname -r)".to_string(),
                    "linux-modules-extra-$(uname -r)".to_string(),
                ],
                env: vec![],
            },
            // AMDGPU DKMS
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "amdgpu-dkms".to_string(),
                ],
                env: vec![],
            },
        ]
    }

    /// Construct environment variable setup commands.
    pub fn build_env_setup_commands(&self, gpu_arch: &str) -> Vec<ShellCommand> {
        vec![
            ShellCommand {
                program: "export".to_string(),
                args: vec!["HSA_OVERRIDE_GFX_VERSION=11.0.0".to_string()],
                env: vec![],
            },
            ShellCommand {
                program: "export".to_string(),
                args: vec![format!("PYTORCH_ROCM_ARCH={gpu_arch}")],
                env: vec![],
            },
            ShellCommand {
                program: "export".to_string(),
                args: vec!["ROCM_PATH=/opt/rocm".to_string()],
                env: vec![],
            },
        ]
    }

    /// Construct the verification command.
    pub fn build_verify_command(&self) -> ShellCommand {
        ShellCommand {
            program: "lsmod".to_string(),
            args: vec![],
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
    use crate::platform::detection::{DistroFamily, DistroInfo, PackageManager};

    /// Helper to create a DistroFacade for testing.
    fn make_distro(family: DistroFamily) -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: match family {
                DistroFamily::Debian => "ubuntu",
                DistroFamily::Rhel => "fedora",
                DistroFamily::Arch => "arch",
                DistroFamily::Suse => "opensuse",
                DistroFamily::Unknown => "unknown",
            }
            .to_string(),
            name: "Test Distro".to_string(),
            version: "1.0".to_string(),
            codename: "test".to_string(),
            id_like: String::new(),
            family,
            pkg_manager: match family {
                DistroFamily::Debian => PackageManager::Apt,
                DistroFamily::Rhel => PackageManager::Dnf,
                DistroFamily::Arch => PackageManager::Pacman,
                DistroFamily::Suse => PackageManager::Zypper,
                DistroFamily::Unknown => PackageManager::Apt,
            },
        })
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-029: AMD GPU drivers correct package command
    // -----------------------------------------------------------------------

    #[test]
    fn test_package_install_command_debian() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Debian);
        let cmd = installer.build_package_install_command(&distro, &["amdgpu-dkms"]);

        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"apt-get".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
        assert!(cmd.args.contains(&"amdgpu-dkms".to_string()));
    }

    #[test]
    fn test_package_install_command_rhel() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Rhel);
        let cmd = installer.build_package_install_command(&distro, &["amdgpu-dkms"]);

        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"dnf".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
    }

    #[test]
    fn test_package_install_command_arch() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Arch);
        let cmd = installer.build_package_install_command(&distro, &["amdgpu"]);

        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"pacman".to_string()));
        assert!(cmd.args.contains(&"-S".to_string()));
        assert!(cmd.args.contains(&"--noconfirm".to_string()));
        assert!(cmd.args.contains(&"amdgpu".to_string()));
    }

    #[test]
    fn test_package_install_command_suse() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Suse);
        let cmd = installer.build_package_install_command(&distro, &["amdgpu-dkms"]);

        assert_eq!(cmd.program, "sudo");
        assert!(cmd.args.contains(&"zypper".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
    }

    #[test]
    fn test_required_packages_debian() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Debian);
        let pkgs = installer.required_packages(&distro);
        assert!(pkgs.contains(&"amdgpu-dkms"));
    }

    #[test]
    fn test_required_packages_arch() {
        let installer = AmdgpuInstaller::with_defaults();
        let distro = make_distro(DistroFamily::Arch);
        let pkgs = installer.required_packages(&distro);
        assert!(pkgs.contains(&"amdgpu"));
    }

    #[test]
    fn test_amdgpu_install_deb_commands() {
        let installer = AmdgpuInstaller::with_defaults();
        let cmds = installer.build_amdgpu_install_deb_commands("noble");
        assert_eq!(cmds.len(), 5);

        // First command should be wget
        assert_eq!(cmds[0].program, "wget");
        assert!(cmds[0].args.iter().any(|a| a.contains("repo.radeon.com")));
        assert!(cmds[0].args.iter().any(|a| a.contains("noble")));

        // Second command should install the deb
        assert!(cmds[1].args.iter().any(|a| a.contains("amdgpu-install")));

        // Third should be apt update
        assert!(cmds[2].args.contains(&"update".to_string()));

        // Fourth should install linux headers
        assert!(cmds[3].args.iter().any(|a| a.contains("linux-headers")));

        // Fifth should install amdgpu-dkms
        assert!(cmds[4].args.contains(&"amdgpu-dkms".to_string()));
    }

    #[test]
    fn test_env_setup_commands() {
        let installer = AmdgpuInstaller::with_defaults();
        let cmds = installer.build_env_setup_commands("gfx1100");
        assert_eq!(cmds.len(), 3);

        let cmd_strings: Vec<String> = cmds.iter().map(|c| c.to_command_string()).collect();
        assert!(cmd_strings
            .iter()
            .any(|s| s.contains("HSA_OVERRIDE_GFX_VERSION=11.0.0")));
        assert!(cmd_strings
            .iter()
            .any(|s| s.contains("PYTORCH_ROCM_ARCH=gfx1100")));
        assert!(cmd_strings
            .iter()
            .any(|s| s.contains("ROCM_PATH=/opt/rocm")));
    }

    #[test]
    fn test_verify_command() {
        let installer = AmdgpuInstaller::with_defaults();
        let cmd = installer.build_verify_command();
        assert_eq!(cmd.program, "lsmod");
    }
}
