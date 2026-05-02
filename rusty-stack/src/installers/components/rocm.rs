//! ROCm installer — ports `scripts/install_rocm.sh`.
//!
//! Constructs correct apt/dnf/pacman commands per distro family, with
//! channel selection (Legacy/Stable/Latest) producing correct version
//! pins and repository URLs.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-001**: ROCm installer correct package commands per distro
//! - **VAL-INSTALL-002**: ROCm channel selection correct version pins and repo URLs

use crate::installers::common::DistroFacade;
use std::fmt;

// ===========================================================================
// Types
// ===========================================================================

/// ROCm release channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmChannel {
    /// ROCm 6.4.3 — production-proven stability (Legacy).
    Legacy,
    /// ROCm 7.1 — production-ready for RDNA 3 (Stable).
    Stable,
    /// ROCm 7.2.1 — expanded RDNA 4 support (Latest, default).
    Latest,
}

impl fmt::Display for RocmChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RocmChannel::Legacy => write!(f, "legacy"),
            RocmChannel::Stable => write!(f, "stable"),
            RocmChannel::Latest => write!(f, "latest"),
        }
    }
}

impl RocmChannel {
    /// Parse from a choice number (1-3) as used in the original script.
    pub fn from_choice(choice: u8) -> Option<Self> {
        match choice {
            1 => Some(RocmChannel::Legacy),
            2 => Some(RocmChannel::Stable),
            3 => Some(RocmChannel::Latest),
            _ => None,
        }
    }

    /// Get the ROCm version string for this channel.
    pub fn version(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "6.4.3",
            RocmChannel::Stable => "7.1",
            RocmChannel::Latest => "7.2.1",
        }
    }

    /// Get the ROCm directory path component for this channel.
    pub fn dir_path(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "6.4.3",
            RocmChannel::Stable => "7.1",
            RocmChannel::Latest => "7.2.1",
        }
    }

    /// Get the package version string used in installer package names.
    pub fn pkg_version(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "6.4.60403-1",
            RocmChannel::Stable => "7.1.70100-1",
            RocmChannel::Latest => "7.2.1.70201-1",
        }
    }

    /// Get the major.minor version (e.g., "7.2") for repo URLs.
    pub fn major_minor(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "6.4.3",
            RocmChannel::Stable => "7.1",
            RocmChannel::Latest => "7.2",
        }
    }
}

/// Installation type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmInstallType {
    /// ROCm runtime + basic tools.
    Standard,
    /// ROCm runtime only.
    Minimal,
    /// ROCm + development tools + libraries.
    Full,
    /// Select specific components.
    Custom,
}

impl fmt::Display for RocmInstallType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RocmInstallType::Standard => write!(f, "standard"),
            RocmInstallType::Minimal => write!(f, "minimal"),
            RocmInstallType::Full => write!(f, "full"),
            RocmInstallType::Custom => write!(f, "custom"),
        }
    }
}

/// Configuration for the ROCm installer.
#[derive(Debug, Clone)]
pub struct RocmConfig {
    /// Selected ROCm channel.
    pub channel: RocmChannel,
    /// Installation type.
    pub install_type: RocmInstallType,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for RocmConfig {
    fn default() -> Self {
        Self {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            force_reinstall: false,
            dry_run: false,
        }
    }
}

/// Constructed command for a package manager operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageCommand {
    /// The program to run (e.g., "sudo", "apt-get").
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
}

impl PackageCommand {
    /// Format as a shell command string.
    pub fn to_command_string(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

/// Constructed repository configuration for ROCm.
#[derive(Debug, Clone)]
pub struct RepoConfig {
    /// Repository type (e.g., "ubuntu", "rhel", "sle").
    pub repo_type: String,
    /// Repository URL.
    pub url: String,
    /// GPG key URL.
    pub gpg_key_url: String,
    /// Target file path for the repo config.
    pub file_path: String,
}

/// The ROCm installer.
pub struct RocmInstaller {
    config: RocmConfig,
}

impl RocmInstaller {
    /// Create a new ROCm installer with the given config.
    pub fn new(config: RocmConfig) -> Self {
        Self { config }
    }

    /// Create with default config (Latest channel, Standard install).
    pub fn with_defaults() -> Self {
        Self::new(RocmConfig::default())
    }

    /// Get the ROCm channel.
    pub fn channel(&self) -> RocmChannel {
        self.config.channel
    }

    // -----------------------------------------------------------------------
    // Repository URL construction (VAL-INSTALL-002)
    // -----------------------------------------------------------------------

    /// Get the amdgpu/ROCm repo base URL for the given distro and channel.
    ///
    /// For apt-based distros, the URL is:
    /// `https://repo.radeon.com/amdgpu/{major_minor}/ubuntu/{codename}`
    ///
    /// For RHEL-based distros:
    /// `https://repo.radeon.com/amdgpu/{major_minor}/rhel/{rhel_version}`
    ///
    /// For SUSE:
    /// `https://repo.radeon.com/amdgpu/{major_minor}/sle/{sles_version}`
    pub fn repo_base_url(&self, distro: &DistroFacade) -> String {
        let major_minor = self.config.channel.major_minor();
        if distro.uses_apt() {
            let codename = if distro.codename().is_empty() {
                "noble"
            } else {
                distro.codename()
            };
            format!("https://repo.radeon.com/amdgpu/{major_minor}/ubuntu/{codename}")
        } else if distro.uses_dnf() || distro.uses_yum() {
            let rhel_version = Self::rhel_version(distro);
            format!("https://repo.radeon.com/amdgpu/{major_minor}/rhel/{rhel_version}")
        } else if distro.uses_zypper() {
            format!("https://repo.radeon.com/amdgpu/{major_minor}/sle/15.7")
        } else {
            // Fallback for Arch and others
            format!("https://repo.radeon.com/amdgpu/{major_minor}/ubuntu/noble")
        }
    }

    /// Get the ROCm core repo URL for apt-based distros.
    pub fn rocm_core_repo_url(&self, _distro: &DistroFacade) -> String {
        let major_minor = self.config.channel.major_minor();
        format!("https://repo.radeon.com/rocm/apt/{major_minor}")
    }

    /// Get the GPG key URL.
    pub fn gpg_key_url(&self) -> &'static str {
        "https://repo.radeon.com/rocm/rocm.gpg.key"
    }

    /// Derive the RHEL major version from distro info.
    fn rhel_version(distro: &DistroFacade) -> &'static str {
        let v = distro.version();
        if v.starts_with("10") { "10" }
        else if v.starts_with("9") { "9" }
        else if v.starts_with("8") { "8" }
        else { "9" }
    }

    // -----------------------------------------------------------------------
    // Package command construction (VAL-INSTALL-001)
    // -----------------------------------------------------------------------

    /// Construct the amdgpu-install package download URL for apt-based distros.
    pub fn amdgpu_install_deb_url(&self) -> String {
        let pkg_ver = self.config.channel.pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{pkg_ver}/ubuntu/noble/amdgpu-install_{pkg_ver}_all.deb"
        )
    }

    /// Construct the amdgpu-install RPM download URL.
    pub fn amdgpu_install_rpm_url(&self, rhel_version: &str) -> String {
        let pkg_ver = self.config.channel.pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{pkg_ver}/rhel/{rhel_version}/amdgpu-install-{pkg_ver}.el{rhel_version}.noarch.rpm"
        )
    }

    /// Construct the amdgpu-install SLES RPM download URL.
    pub fn amdgpu_install_sles_rpm_url(&self) -> String {
        let pkg_ver = self.config.channel.pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{pkg_ver}/sle/15.7/amdgpu-install-{pkg_ver}.noarch.rpm"
        )
    }

    /// Construct the apt-get install command for amdgpu-install.
    pub fn apt_install_commands(&self) -> Vec<PackageCommand> {
        let deb_url = self.amdgpu_install_deb_url();
        vec![
            // Download the .deb
            PackageCommand {
                program: "wget".to_string(),
                args: vec!["-q".to_string(), deb_url.clone()],
            },
            // Install it
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dpkg".to_string(),
                    "-i".to_string(),
                    format!("amdgpu-install_{}_all.deb", self.config.channel.pkg_version()),
                ],
            },
            // Update and install ROCm packages
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "update".to_string(),
                ],
            },
            // Install ROCm metapackage based on install type
            self.apt_rocm_metapackage_command(),
        ]
    }

    /// Get the apt metapackage install command for the selected install type.
    fn apt_rocm_metapackage_command(&self) -> PackageCommand {
        let metapkg = match self.config.install_type {
            RocmInstallType::Standard => "rocm-libs",
            RocmInstallType::Minimal => "rocm-core",
            RocmInstallType::Full => "rocm-dev",
            RocmInstallType::Custom => "rocm-libs",
        };
        PackageCommand {
            program: "sudo".to_string(),
            args: vec![
                "apt-get".to_string(),
                "install".to_string(),
                "-y".to_string(),
                metapkg.to_string(),
            ],
        }
    }

    /// Construct the dnf install commands for ROCm.
    pub fn dnf_install_commands(&self, rhel_version: &str) -> Vec<PackageCommand> {
        let rpm_url = self.amdgpu_install_rpm_url(rhel_version);
        vec![
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    rpm_url,
                ],
            },
            self.dnf_rocm_metapackage_command(),
        ]
    }

    /// Get the dnf metapackage install command.
    fn dnf_rocm_metapackage_command(&self) -> PackageCommand {
        let metapkg = match self.config.install_type {
            RocmInstallType::Standard => "rocm-libs",
            RocmInstallType::Minimal => "rocm-core",
            RocmInstallType::Full => "rocm-dev",
            RocmInstallType::Custom => "rocm-libs",
        };
        PackageCommand {
            program: "sudo".to_string(),
            args: vec![
                "dnf".to_string(),
                "install".to_string(),
                "-y".to_string(),
                metapkg.to_string(),
            ],
        }
    }

    /// Construct the pacman install commands for ROCm on Arch.
    ///
    /// Arch uses AUR packages (yay/paru) for ROCm.
    pub fn pacman_install_commands(&self, aur_helper: &str) -> Vec<PackageCommand> {
        let packages = self.pacman_rocm_packages();
        let mut args = vec!["-S".to_string(), "--noconfirm".to_string()];
        args.extend(packages);

        vec![
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![aur_helper.to_string()],
            },
            PackageCommand {
                program: aur_helper.to_string(),
                args,
            },
        ]
    }

    /// Get the list of ROCm AUR packages for pacman-based installs.
    pub fn pacman_rocm_packages(&self) -> Vec<String> {
        let mut pkgs = vec![
            "rocm-hip-sdk".to_string(),
            "rocminfo".to_string(),
            "rocm-smi-lib".to_string(),
            "hipblas".to_string(),
            "rocblas".to_string(),
            "hipsparselt".to_string(),
        ];
        // Add extra packages for Stable/Latest channels
        if self.config.channel != RocmChannel::Legacy {
            pkgs.push("rocm-opencl-sdk".to_string());
            pkgs.push("rccl".to_string());
        }
        pkgs
    }

    /// Construct the zypper install commands for ROCm on SUSE.
    pub fn zypper_install_commands(&self) -> Vec<PackageCommand> {
        let rpm_url = self.amdgpu_install_sles_rpm_url();
        vec![
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "zypper".to_string(),
                    "--non-interactive".to_string(),
                    "install".to_string(),
                    rpm_url,
                ],
            },
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "zypper".to_string(),
                    "--non-interactive".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rocm-libs".to_string(),
                ],
            },
        ]
    }

    /// Get the repository configuration for apt-based distros.
    pub fn apt_repo_config(&self, distro: &DistroFacade) -> RepoConfig {
        let codename = if distro.codename().is_empty() { "noble" } else { distro.codename() };
        let major_minor = self.config.channel.major_minor();
        RepoConfig {
            repo_type: "apt".to_string(),
            url: format!("deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/{major_minor} {codename} main"),
            gpg_key_url: "https://repo.radeon.com/rocm/rocm.gpg.key".to_string(),
            file_path: "/etc/apt/sources.list.d/rocm.list".to_string(),
        }
    }

    /// Get the amdgpu repo configuration for apt-based distros.
    pub fn amdgpu_apt_repo_config(&self, distro: &DistroFacade) -> RepoConfig {
        let codename = if distro.codename().is_empty() { "noble" } else { distro.codename() };
        let major_minor = self.config.channel.major_minor();
        RepoConfig {
            repo_type: "apt".to_string(),
            url: format!("deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/{major_minor}/ubuntu/{codename} {codename} main"),
            gpg_key_url: "https://repo.radeon.com/rocm/rocm.gpg.key".to_string(),
            file_path: "/etc/apt/sources.list.d/amdgpu.list".to_string(),
        }
    }

    /// Get the ROCm preference pin configuration for apt.
    pub fn apt_pin_config(&self) -> (String, String) {
        (
            "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n".to_string(),
            "/etc/apt/preferences.d/rocm-pin-600".to_string(),
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::detection::{DistroFamily, DistroInfo, PackageManager};

    fn ubuntu_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            name: "Ubuntu 24.04 LTS".to_string(),
            version: "24.04".to_string(),
            codename: "noble".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            id_like: "debian".to_string(),
        })
    }

    fn fedora_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            name: "Fedora Linux 41".to_string(),
            version: "41".to_string(),
            codename: String::new(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            id_like: "fedora".to_string(),
        })
    }

    fn rhel9_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "rhel".to_string(),
            name: "Red Hat Enterprise Linux 9".to_string(),
            version: "9.3".to_string(),
            codename: String::new(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            id_like: "rhel".to_string(),
        })
    }

    fn suse_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "opensuse-leap".to_string(),
            name: "openSUSE Leap 15.7".to_string(),
            version: "15.7".to_string(),
            codename: String::new(),
            family: DistroFamily::Suse,
            pkg_manager: PackageManager::Zypper,
            id_like: "suse".to_string(),
        })
    }

    // --- VAL-INSTALL-002: ROCm channel selection correct version pins ---

    #[test]
    fn test_channel_legacy_version() {
        let ch = RocmChannel::Legacy;
        assert_eq!(ch.version(), "6.4.3");
        assert_eq!(ch.pkg_version(), "6.4.60403-1");
        assert_eq!(ch.major_minor(), "6.4.3");
    }

    #[test]
    fn test_channel_stable_version() {
        let ch = RocmChannel::Stable;
        assert_eq!(ch.version(), "7.1");
        assert_eq!(ch.pkg_version(), "7.1.70100-1");
        assert_eq!(ch.major_minor(), "7.1");
    }

    #[test]
    fn test_channel_latest_version() {
        let ch = RocmChannel::Latest;
        assert_eq!(ch.version(), "7.2.1");
        assert_eq!(ch.pkg_version(), "7.2.1.70201-1");
        assert_eq!(ch.major_minor(), "7.2");
    }

    #[test]
    fn test_channel_from_choice() {
        assert_eq!(RocmChannel::from_choice(1), Some(RocmChannel::Legacy));
        assert_eq!(RocmChannel::from_choice(2), Some(RocmChannel::Stable));
        assert_eq!(RocmChannel::from_choice(3), Some(RocmChannel::Latest));
        assert_eq!(RocmChannel::from_choice(0), None);
        assert_eq!(RocmChannel::from_choice(4), None);
    }

    // --- VAL-INSTALL-001: ROCm installer correct package commands ---

    #[test]
    fn test_apt_install_commands_latest() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.apt_install_commands();
        assert_eq!(cmds.len(), 4);

        // wget download
        assert_eq!(cmds[0].program, "wget");
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.1.70201-1")));

        // dpkg install
        assert_eq!(cmds[1].program, "sudo");
        assert!(cmds[1].args.contains(&"dpkg".to_string()));

        // apt-get update
        assert_eq!(cmds[2].program, "sudo");
        assert!(cmds[2].args.contains(&"update".to_string()));

        // rocm-libs metapackage
        assert_eq!(cmds[3].program, "sudo");
        assert!(cmds[3].args.contains(&"rocm-libs".to_string()));
    }

    #[test]
    fn test_apt_install_commands_full() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Stable,
            install_type: RocmInstallType::Full,
            ..Default::default()
        });
        let cmds = installer.apt_install_commands();
        let meta_cmd = &cmds[3];
        assert!(meta_cmd.args.contains(&"rocm-dev".to_string()));
    }

    #[test]
    fn test_dnf_install_commands() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.dnf_install_commands("9");
        assert_eq!(cmds.len(), 2);

        // dnf install RPM
        assert_eq!(cmds[0].program, "sudo");
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.1.70201-1")));
        assert!(cmds[0].args.iter().any(|a| a.contains("el9")));

        // rocm-libs metapackage
        assert_eq!(cmds[1].program, "sudo");
        assert!(cmds[1].args.contains(&"rocm-libs".to_string()));
    }

    #[test]
    fn test_pacman_install_commands() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let _cmds = installer.pacman_install_commands("yay");
        // Should have packages for Latest channel
        let pkgs = installer.pacman_rocm_packages();
        assert!(pkgs.contains(&"rocm-hip-sdk".to_string()));
        assert!(pkgs.contains(&"rocminfo".to_string()));
        assert!(pkgs.contains(&"rocm-opencl-sdk".to_string())); // Latest gets extra
        assert!(pkgs.contains(&"rccl".to_string())); // Latest gets extra
    }

    #[test]
    fn test_pacman_legacy_no_extras() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let pkgs = installer.pacman_rocm_packages();
        assert!(!pkgs.contains(&"rocm-opencl-sdk".to_string()));
        assert!(!pkgs.contains(&"rccl".to_string()));
    }

    #[test]
    fn test_zypper_install_commands() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.zypper_install_commands();
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0].program, "sudo");
        assert!(cmds[0].args.iter().any(|a| a.contains("zypper")));
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.1.70201-1")));
    }

    // --- Repo URL construction ---

    #[test]
    fn test_repo_base_url_ubuntu() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let distro = ubuntu_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/ubuntu/noble");
    }

    #[test]
    fn test_repo_base_url_rhel() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let distro = rhel9_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/rhel/9");
    }

    #[test]
    fn test_repo_base_url_fedora() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Stable,
            ..Default::default()
        });
        let distro = fedora_distro();
        let url = installer.repo_base_url(&distro);
        // Fedora version "41" doesn't match 8/9/10, defaults to 9
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.1/rhel/9");
    }

    #[test]
    fn test_repo_base_url_suse() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let distro = suse_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/sle/15.7");
    }

    #[test]
    fn test_amdgpu_install_deb_url() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let url = installer.amdgpu_install_deb_url();
        assert!(url.contains("7.2.1.70201-1"));
        assert!(url.contains("ubuntu/noble"));
        assert!(url.ends_with(".deb"));
    }

    #[test]
    fn test_amdgpu_install_rpm_url() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let url = installer.amdgpu_install_rpm_url("9");
        assert!(url.contains("7.2.1.70201-1"));
        assert!(url.contains("rhel/9"));
        assert!(url.ends_with(".noarch.rpm"));
    }

    #[test]
    fn test_apt_repo_config() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let distro = ubuntu_distro();
        let repo = installer.apt_repo_config(&distro);
        assert_eq!(repo.file_path, "/etc/apt/sources.list.d/rocm.list");
        assert!(repo.url.contains("rocm/apt/7.2"));
        assert!(repo.url.contains("noble"));
    }

    #[test]
    fn test_amdgpu_apt_repo_config() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let distro = ubuntu_distro();
        let repo = installer.amdgpu_apt_repo_config(&distro);
        assert_eq!(repo.file_path, "/etc/apt/sources.list.d/amdgpu.list");
        assert!(repo.url.contains("amdgpu/7.2"));
    }

    #[test]
    fn test_apt_pin_config() {
        let installer = RocmInstaller::with_defaults();
        let (content, path) = installer.apt_pin_config();
        assert_eq!(path, "/etc/apt/preferences.d/rocm-pin-600");
        assert!(content.contains("Pin-Priority: 600"));
    }

    // --- Cross-channel parametric tests ---

    #[test]
    fn test_all_channels_version_consistency() {
        for choice in [1u8, 2, 3] {
            let ch = RocmChannel::from_choice(choice).unwrap();
            let config = RocmConfig {
                channel: ch,
                ..Default::default()
            };
            let installer = RocmInstaller::new(config);

            // Verify deb URL contains correct package version
            let deb_url = installer.amdgpu_install_deb_url();
            assert!(
                deb_url.contains(ch.pkg_version()),
                "deb URL for {:?} should contain pkg_version {}",
                ch, ch.pkg_version()
            );

            // Verify repo URL contains major_minor
            let distro = ubuntu_distro();
            let repo_url = installer.repo_base_url(&distro);
            assert!(
                repo_url.contains(ch.major_minor()),
                "repo URL for {:?} should contain major_minor {}",
                ch, ch.major_minor()
            );
        }
    }

    #[test]
    fn test_install_type_metapackages() {
        for (itype, expected_pkg) in [
            (RocmInstallType::Standard, "rocm-libs"),
            (RocmInstallType::Minimal, "rocm-core"),
            (RocmInstallType::Full, "rocm-dev"),
            (RocmInstallType::Custom, "rocm-libs"),
        ] {
            let installer = RocmInstaller::new(RocmConfig {
                install_type: itype,
                ..Default::default()
            });
            let cmds = installer.apt_install_commands();
            let meta_cmd = &cmds[3];
            assert!(
                meta_cmd.args.contains(&expected_pkg.to_string()),
                "Install type {:?} should use metapackage {}",
                itype, expected_pkg
            );
        }
    }
}
