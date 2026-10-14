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

/// Classic (amdgpu-install) ROCm fallback for the Legacy channel.
pub const LEGACY_FALLBACK_VERSION: &str = "7.2.3";
/// TheRock multi-arch apt/rpm repo root (ROCm 7.14+).
pub const THEROCK_REPO_ROOT: &str = "https://repo.amd.com/rocm/packages-multi-arch";
/// TheRock multi-arch apt GPG key.
pub const THEROCK_GPG_URL: &str = "https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg";
/// TheRock multi-arch pip index (7.14.x).
pub const THEROCK_PIP_INDEX: &str = "https://repo.amd.com/rocm/whl-multi-arch/";
/// ROCm 10.x preview pip index.
pub const PREVIEW_PIP_INDEX: &str = "https://stable.repo.amd.com/rocm/whl-next/";
/// TheRock tarball root.
pub const THEROCK_TARBALL_ROOT: &str = "https://repo.amd.com/rocm/tarball-multi-arch";
/// Runfile installer root.
pub const RUNFILE_ROOT: &str = "https://repo.radeon.com/installer/rocm-runfile-installer";

/// ROCm release channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmChannel {
    /// ROCm 7.2.4 (7.2.3 fallback) — classic amdgpu-install line (Legacy).
    Legacy,
    /// ROCm 7.14.0 — TheRock (Stable).
    Stable,
    /// ROCm 7.14.1 — TheRock (Latest, default).
    Latest,
    /// ROCm 10.1.0 — TheRock Core SDK preview (pip/tarball only).
    Preview,
}

impl fmt::Display for RocmChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RocmChannel::Legacy => write!(f, "legacy"),
            RocmChannel::Stable => write!(f, "stable"),
            RocmChannel::Latest => write!(f, "latest"),
            RocmChannel::Preview => write!(f, "preview"),
        }
    }
}

impl RocmChannel {
    /// Parse from a choice number (1-4).
    pub fn from_choice(choice: u8) -> Option<Self> {
        match choice {
            1 => Some(RocmChannel::Legacy),
            2 => Some(RocmChannel::Stable),
            3 => Some(RocmChannel::Latest),
            4 => Some(RocmChannel::Preview),
            _ => None,
        }
    }

    /// Parse a channel name (`ROCM_CHANNEL`); unknown values yield `None`.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "legacy" => Some(RocmChannel::Legacy),
            "stable" => Some(RocmChannel::Stable),
            "latest" => Some(RocmChannel::Latest),
            "preview" => Some(RocmChannel::Preview),
            _ => None,
        }
    }

    /// Channel for a detected ROCm version string.
    pub fn for_version(version: &str) -> Self {
        let mut it = version.split('.');
        let major: u32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let minor: u32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let patch: u32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        match (major, minor) {
            (10.., _) => RocmChannel::Preview,
            (7, 14..) if patch >= 1 || minor > 14 => RocmChannel::Latest,
            (7, 14..) => RocmChannel::Stable,
            _ => RocmChannel::Legacy,
        }
    }

    /// True for TheRock-based releases (7.14+), which use the multi-arch repos.
    pub fn is_therock(&self) -> bool {
        !matches!(self, RocmChannel::Legacy)
    }

    /// Get the ROCm version string for this channel.
    pub fn version(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "7.2.4",
            RocmChannel::Stable => "7.14.0",
            RocmChannel::Latest => "7.14.1",
            RocmChannel::Preview => "10.1.0",
        }
    }

    /// Get the ROCm directory path component for this channel.
    pub fn dir_path(&self) -> &'static str {
        self.version()
    }

    /// Get the classic package version string used in amdgpu-install names.
    /// Only meaningful for the Legacy channel; TheRock uses `therock_pkg_revision`.
    pub fn pkg_version(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "7.2.4.70204-1",
            RocmChannel::Stable => "7.14.0",
            RocmChannel::Latest => "7.14.1",
            RocmChannel::Preview => "10.1.0",
        }
    }

    /// Get the major.minor version (e.g., "7.14") for repo URLs and package names.
    pub fn major_minor(&self) -> &'static str {
        match self {
            RocmChannel::Legacy => "7.2",
            RocmChannel::Stable | RocmChannel::Latest => "7.14",
            RocmChannel::Preview => "10.1",
        }
    }

    /// Get the version for this channel, with optional manifest override.
    ///
    /// If a manifest is loaded, uses the manifest version for the "rocm" component.
    /// Falls back to the hardcoded version if no manifest or manifest doesn't contain rocm.
    pub fn version_from_manifest(manifest: &crate::core::manifest::Manifest) -> String {
        manifest
            .components
            .iter()
            .find(|c| c.id == "rocm")
            .map(|c| c.version.clone())
            .unwrap_or_else(|| RocmChannel::Latest.version().to_string())
    }

    /// Get the package version for this channel, with optional manifest override.
    ///
    /// Classic (<7.14) releases map `7.2.4` -> `7.2.4.70204-1`; TheRock releases
    /// use the bare version.
    pub fn pkg_version_from_manifest(manifest: &crate::core::manifest::Manifest) -> String {
        let version = Self::version_from_manifest(manifest);
        if version == "6.4.3" {
            return "6.4.60403-1".to_string();
        }
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() >= 3 {
            let major: u64 = parts[0].parse().unwrap_or(7);
            let minor: u64 = parts[1].parse().unwrap_or(2);
            let patch: u64 = parts[2].parse().unwrap_or(0);
            if major >= 10 || (major == 7 && minor >= 14) {
                return version;
            }
            let build = format!("{major}{minor:02}{patch:02}");
            format!("{}.{build}-1", version)
        } else {
            RocmChannel::Latest.pkg_version().to_string()
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
    /// Target GPU architecture (e.g. `gfx1100`) for TheRock per-arch packages.
    pub gfx_arch: String,
    /// Use the 7.2.3 classic build instead of 7.2.4 on the Legacy channel.
    pub legacy_fallback: bool,
}

impl Default for RocmConfig {
    fn default() -> Self {
        Self {
            channel: RocmChannel::Latest,
            install_type: RocmInstallType::Standard,
            force_reinstall: false,
            dry_run: false,
            gfx_arch: "gfx1100".to_string(),
            legacy_fallback: false,
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

    /// Effective ROCm version, honouring the Legacy 7.2.3 fallback.
    pub fn effective_version(&self) -> &'static str {
        if self.config.channel == RocmChannel::Legacy && self.config.legacy_fallback {
            LEGACY_FALLBACK_VERSION
        } else {
            self.config.channel.version()
        }
    }

    fn classic_release(&self) -> &'static str {
        self.effective_version()
    }

    fn classic_pkg_version(&self) -> &'static str {
        if self.config.channel == RocmChannel::Legacy && self.config.legacy_fallback {
            "7.2.3.70203-1"
        } else {
            self.config.channel.pkg_version()
        }
    }

    // -----------------------------------------------------------------------
    // TheRock (7.14+) install methods
    // -----------------------------------------------------------------------

    /// Repo directory name for a distro on the TheRock multi-arch repo
    /// (e.g. `ubuntu2404`, `debian12`, `rhel9`, `sles16`). `None` if unsupported.
    pub fn therock_distro_dir(distro: &DistroFacade) -> Option<String> {
        let v = distro.version();
        let major = v.split('.').next().unwrap_or("");
        if distro.uses_apt() {
            if distro.id() == "debian" {
                return match major {
                    "12" | "13" => Some(format!("debian{major}")),
                    _ => None,
                };
            }
            let compact = v.replace('.', "");
            match compact.as_str() {
                "2204" | "2404" | "2604" => Some(format!("ubuntu{compact}")),
                _ => match distro.codename() {
                    "jammy" => Some("ubuntu2204".into()),
                    "noble" => Some("ubuntu2404".into()),
                    _ => None,
                },
            }
        } else if distro.uses_dnf() || distro.uses_yum() {
            match major {
                "8" | "9" | "10" => Some(format!("rhel{major}")),
                _ => None,
            }
        } else if distro.uses_zypper() {
            match major {
                "15" => Some("sles15".into()),
                "16" => Some("sles16".into()),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Per-GPU metapackage name, e.g. `amdrocm7.14-gfx1100`.
    pub fn therock_package(&self) -> String {
        format!(
            "amdrocm{}-{}",
            self.config.channel.major_minor(),
            self.config.gfx_arch
        )
    }

    /// apt `Version` of the metapackage: `7.14.1` -> `7.14.1-*` (revision varies).
    pub fn therock_apt_version_glob(&self) -> String {
        format!("{}-*", self.config.channel.version())
    }

    /// Commands that configure the TheRock apt repo (key, source) and install.
    pub fn therock_apt_commands(&self, distro: &DistroFacade) -> Vec<PackageCommand> {
        let dir = Self::therock_distro_dir(distro).unwrap_or_else(|| "ubuntu2404".to_string());
        let sh = |script: String| PackageCommand {
            program: "sudo".to_string(),
            args: vec!["sh".to_string(), "-c".to_string(), script],
        };
        vec![
            sh(format!(
                "mkdir -p /etc/apt/keyrings && wget -qO /etc/apt/keyrings/rocm-multi-arch.gpg {THEROCK_GPG_URL}"
            )),
            sh(format!(
                "echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-multi-arch.gpg] {THEROCK_REPO_ROOT}/{dir} stable main' > /etc/apt/sources.list.d/rocm-multi-arch.list"
            )),
            PackageCommand {
                program: "sudo".to_string(),
                args: vec!["apt-get".to_string(), "update".to_string()],
            },
            PackageCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    format!("{}={}", self.therock_package(), self.therock_apt_version_glob()),
                ],
            },
        ]
    }

    /// Commands that configure the TheRock rpm repo and install (dnf / zypper).
    pub fn therock_rpm_commands(&self, distro: &DistroFacade) -> Vec<PackageCommand> {
        let dir = Self::therock_distro_dir(distro).unwrap_or_else(|| "rhel9".to_string());
        let url = format!("{THEROCK_REPO_ROOT}/{dir}/x86_64");
        let pkg = format!(
            "{}-{}",
            self.therock_package(),
            self.config.channel.version()
        );
        let sudo = |args: Vec<&str>| PackageCommand {
            program: "sudo".to_string(),
            args: args.into_iter().map(String::from).collect(),
        };
        if distro.uses_zypper() {
            vec![
                sudo(vec![
                    "zypper",
                    "--non-interactive",
                    "addrepo",
                    "--refresh",
                    &url,
                    "rocm-multi-arch",
                ]),
                sudo(vec![
                    "zypper",
                    "--non-interactive",
                    "--gpg-auto-import-keys",
                    "refresh",
                ]),
                sudo(vec!["zypper", "--non-interactive", "install", &pkg]),
            ]
        } else {
            vec![
                sudo(vec!["dnf", "config-manager", "--add-repo", &url]),
                sudo(vec!["dnf", "install", "-y", "--nogpgcheck", &pkg]),
            ]
        }
    }

    /// pip requirement for the ROCm SDK on this channel
    /// (`rocm[libraries,device-all]==7.14.1`).
    pub fn pip_requirement(&self) -> String {
        format!(
            "rocm[libraries,device-all]=={}",
            self.config.channel.version()
        )
    }

    /// pip index URL for this channel (`None` for classic Legacy).
    pub fn pip_index_url(&self) -> Option<&'static str> {
        match self.config.channel {
            RocmChannel::Legacy => None,
            RocmChannel::Stable | RocmChannel::Latest => Some(THEROCK_PIP_INDEX),
            RocmChannel::Preview => Some(PREVIEW_PIP_INDEX),
        }
    }

    /// TheRock GPU family for tarball names (`gfx1100` -> `gfx110X-all`).
    pub fn therock_family(gfx: &str) -> String {
        let g = gfx.trim().to_ascii_lowercase();
        let fam = match g.as_str() {
            s if s.starts_with("gfx101") => "gfx101X-dgpu",
            s if s.starts_with("gfx103") => "gfx103X-all",
            s if s.starts_with("gfx110") => "gfx110X-all",
            s if s.starts_with("gfx115") => "gfx115X-all",
            s if s.starts_with("gfx120") => "gfx120X-all",
            "gfx908" => "gfx908",
            "gfx90a" => "gfx90a",
            "gfx942" | "gfx941" | "gfx940" => "gfx94X-dcgpu",
            "gfx950" => "gfx950-dcgpu",
            _ => return g,
        };
        fam.to_string()
    }

    /// TheRock tarball URL for the configured GPU and channel.
    pub fn tarball_url(&self) -> String {
        format!(
            "{THEROCK_TARBALL_ROOT}/therock-dist-linux-{}-{}.tar.gz",
            Self::therock_family(&self.config.gfx_arch),
            self.config.channel.version()
        )
    }

    /// Runfile installer URL (`rocm-installer-7.14.0-7.run`).
    pub fn runfile_url(&self) -> String {
        let v = self.config.channel.version();
        let mm = self.config.channel.major_minor();
        format!("{RUNFILE_ROOT}/rocm-rel-{mm}/rocm-installer-{v}-7.run")
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
        if v.starts_with("10") {
            "10"
        } else if v.starts_with("9") {
            "9"
        } else if v.starts_with("8") {
            "8"
        } else {
            "9"
        }
    }

    // -----------------------------------------------------------------------
    // Package command construction (VAL-INSTALL-001)
    // -----------------------------------------------------------------------

    /// Construct the amdgpu-install package download URL for apt-based distros.
    pub fn amdgpu_install_deb_url(&self) -> String {
        let release_dir = self.classic_release();
        let pkg_ver = self.classic_pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{release_dir}/ubuntu/noble/amdgpu-install_{pkg_ver}_all.deb"
        )
    }

    /// Construct the amdgpu-install RPM download URL.
    pub fn amdgpu_install_rpm_url(&self, rhel_version: &str) -> String {
        let release_dir = self.classic_release();
        let pkg_ver = self.classic_pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{release_dir}/rhel/{rhel_version}/amdgpu-install-{pkg_ver}.el{rhel_version}.noarch.rpm"
        )
    }

    /// Construct the amdgpu-install SLES RPM download URL.
    pub fn amdgpu_install_sles_rpm_url(&self) -> String {
        let release_dir = self.classic_release();
        let pkg_ver = self.classic_pkg_version();
        format!(
            "https://repo.radeon.com/amdgpu-install/{release_dir}/sle/15.7/amdgpu-install-{pkg_ver}.noarch.rpm"
        )
    }

    /// Construct the apt-get install command for amdgpu-install.
    pub fn apt_install_commands(&self) -> Vec<PackageCommand> {
        if self.config.channel.is_therock() {
            return self.therock_apt_commands(&DistroFacade::detect());
        }
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
                    format!("amdgpu-install_{}_all.deb", self.classic_pkg_version()),
                ],
            },
            // Update and install ROCm packages
            PackageCommand {
                program: "sudo".to_string(),
                args: vec!["apt-get".to_string(), "update".to_string()],
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
        if self.config.channel.is_therock() {
            return self.therock_rpm_commands(&DistroFacade::detect());
        }
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
    /// Build the pacman/AUR (`yay`) install command for ROCm.
    ///
    /// `yay` MUST run as the user — makepkg refuses root and `yay` warns
    /// "Avoid running yay as root/sudo". This therefore returns a SINGLE
    /// user-space command, never wrapped in `sudo`. (A bare `sudo yay` with no
    /// operation defaults to `yay -Syu`, i.e. a full system upgrade, which is
    /// what previously aborted ROCm installs.) The caller feeds yay's internal
    /// `sudo pacman` non-interactively — e.g. `SUDO_ASKPASS` +
    /// `--sudoflags=-A`, as the installer's Arch path does.
    pub fn pacman_install_commands(&self, aur_helper: &str) -> Vec<PackageCommand> {
        let packages = self.pacman_rocm_packages();
        let mut args = vec![
            "-S".to_string(),
            "--needed".to_string(),
            "--noconfirm".to_string(),
        ];
        args.extend(packages);

        vec![PackageCommand {
            program: aur_helper.to_string(),
            args,
        }]
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
            // rocwmma: RDNA3+ WMMA flash-attention backend (used by Rusty Llama's
            // fattn-wmma-f16). Optional at build time (the fork gates on it), but
            // install it so the WMMA FA path is available out of the box.
            pkgs.push("rocwmma".to_string());
        }
        pkgs
    }

    /// Construct the zypper install commands for ROCm on SUSE.
    pub fn zypper_install_commands(&self) -> Vec<PackageCommand> {
        if self.config.channel.is_therock() {
            return self.therock_rpm_commands(&DistroFacade::detect());
        }
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
        let codename = if distro.codename().is_empty() {
            "noble"
        } else {
            distro.codename()
        };
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
        let codename = if distro.codename().is_empty() {
            "noble"
        } else {
            distro.codename()
        };
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

    /// Build a command to check if a single package is installed on Arch/pacman.
    ///
    /// Returns a `PackageCommand` that runs `pacman -Qi <package>`.
    /// Exit code 0 means installed, non-zero means not installed.
    pub fn build_pacman_check_installed_command(package: &str) -> PackageCommand {
        PackageCommand {
            program: "pacman".to_string(),
            args: vec!["-Qi".to_string(), package.to_string()],
        }
    }

    /// Filter out packages that are already installed on Arch/pacman systems.
    ///
    /// For each package in `packages`, runs `pacman -Qi <pkg>` to check if
    /// it's installed. Returns only the packages that are NOT yet installed.
    /// In dry-run mode, returns all packages (no checks performed).
    pub fn filter_already_installed_pacman(packages: &[String]) -> Vec<String> {
        use std::process::Command;

        packages
            .iter()
            .filter(|pkg| {
                let status = Command::new("pacman").args(["-Qi", pkg.as_str()]).output();
                match status {
                    Ok(output) => !output.status.success(),
                    Err(_) => true, // If pacman fails to run, assume not installed
                }
            })
            .cloned()
            .collect()
    }

    /// Check if command output indicates a package is already up to date.
    ///
    /// yay/pacman output patterns that should be treated as success:
    /// - `warning: <pkg> is up to date -- skipping`
    /// - `<pkg> is up to date -- skipping`
    /// - `there is nothing to do`
    /// - `already installed`
    /// - `.rc files restored` (post-install message, not an error)
    pub fn is_up_to_date_output(output: &str) -> bool {
        let lower = output.to_lowercase();
        // yay/pacman "up to date" patterns
        lower.contains("up to date -- skipping")
            || lower.contains("up to date, skipping")
            || lower.contains("up to date -- reinstalling")
            || lower.contains("there is nothing to do")
            || lower.contains("already installed")
            // pacman -S on already-installed package
            || lower.contains("warning:") && lower.contains("up to date")
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
    fn test_channel_versions() {
        assert_eq!(RocmChannel::Legacy.version(), "7.2.4");
        assert_eq!(RocmChannel::Legacy.pkg_version(), "7.2.4.70204-1");
        assert_eq!(RocmChannel::Legacy.major_minor(), "7.2");
        assert_eq!(RocmChannel::Stable.version(), "7.14.0");
        assert_eq!(RocmChannel::Latest.version(), "7.14.1");
        assert_eq!(RocmChannel::Stable.major_minor(), "7.14");
        assert_eq!(RocmChannel::Preview.version(), "10.1.0");
        assert_eq!(RocmChannel::Preview.major_minor(), "10.1");
        assert!(!RocmChannel::Legacy.is_therock());
        assert!(RocmChannel::Preview.is_therock());
    }

    #[test]
    fn test_channel_from_choice() {
        assert_eq!(RocmChannel::from_choice(1), Some(RocmChannel::Legacy));
        assert_eq!(RocmChannel::from_choice(2), Some(RocmChannel::Stable));
        assert_eq!(RocmChannel::from_choice(3), Some(RocmChannel::Latest));
        assert_eq!(RocmChannel::from_choice(4), Some(RocmChannel::Preview));
        assert_eq!(RocmChannel::from_choice(0), None);
        assert_eq!(RocmChannel::from_choice(5), None);
    }

    #[test]
    fn test_channel_parse_and_for_version() {
        assert_eq!(RocmChannel::parse("Preview"), Some(RocmChannel::Preview));
        assert_eq!(RocmChannel::parse("bogus"), None);
        assert_eq!(RocmChannel::for_version("10.1.0"), RocmChannel::Preview);
        assert_eq!(RocmChannel::for_version("7.14.1"), RocmChannel::Latest);
        assert_eq!(RocmChannel::for_version("7.14.0"), RocmChannel::Stable);
        assert_eq!(RocmChannel::for_version("7.2.4"), RocmChannel::Legacy);
        assert_eq!(RocmChannel::for_version("6.4.3"), RocmChannel::Legacy);
    }

    #[test]
    fn test_legacy_fallback_723() {
        let i = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            legacy_fallback: true,
            ..Default::default()
        });
        assert_eq!(i.effective_version(), "7.2.3");
        assert!(i
            .amdgpu_install_deb_url()
            .contains("7.2.3/ubuntu/noble/amdgpu-install_7.2.3.70203-1_all.deb"));
    }

    #[test]
    fn test_therock_urls_and_package() {
        let i = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Stable,
            gfx_arch: "gfx1100".into(),
            ..Default::default()
        });
        assert_eq!(i.therock_package(), "amdrocm7.14-gfx1100");
        assert_eq!(i.pip_requirement(), "rocm[libraries,device-all]==7.14.0");
        assert_eq!(
            i.pip_index_url(),
            Some("https://repo.amd.com/rocm/whl-multi-arch/")
        );
        assert_eq!(
            i.tarball_url(),
            "https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx110X-all-7.14.0.tar.gz"
        );
        assert_eq!(
            i.runfile_url(),
            "https://repo.radeon.com/installer/rocm-runfile-installer/rocm-rel-7.14/rocm-installer-7.14.0-7.run"
        );
    }

    #[test]
    fn test_therock_apt_commands_latest() {
        let i = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        let cmds = i.therock_apt_commands(&ubuntu_distro());
        let all: String = cmds
            .iter()
            .map(|c| c.to_command_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(all.contains("packages-multi-arch/gpg/rocm.gpg"));
        assert!(all.contains("packages-multi-arch/ubuntu2404 stable main"));
        assert!(all.contains("amdrocm7.14-gfx1100=7.14.1-*"));
    }

    #[test]
    fn test_therock_rpm_and_distro_dir() {
        let i = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Latest,
            ..Default::default()
        });
        assert_eq!(
            RocmInstaller::therock_distro_dir(&ubuntu_distro()).as_deref(),
            Some("ubuntu2404")
        );
        assert_eq!(
            RocmInstaller::therock_distro_dir(&rhel9_distro()).as_deref(),
            Some("rhel9")
        );
        let cmds = i.therock_rpm_commands(&rhel9_distro());
        assert!(cmds
            .iter()
            .any(|c| c.to_command_string().contains("amdrocm7.14-gfx1100-7.14.1")));
    }

    #[test]
    fn test_preview_pip_index() {
        let i = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Preview,
            ..Default::default()
        });
        assert_eq!(
            i.pip_index_url(),
            Some("https://stable.repo.amd.com/rocm/whl-next/")
        );
        assert_eq!(i.pip_requirement(), "rocm[libraries,device-all]==10.1.0");
    }

    #[test]
    fn test_pkg_version_from_manifest_latest() {
        let manifest = crate::core::manifest::Manifest {
            schema_version: 2,
            sequence: 1,
            generated_at: "2026-05-31T00:00:00Z".to_string(),
            expires_at: None,
            min_runtime_version: String::new(),
            components: vec![crate::core::manifest::ManifestComponent {
                id: "rocm".to_string(),
                version: "7.2.4".to_string(),
                script: "scripts/install_rocm.sh".to_string(),
                category: crate::core::Category::Foundation,
                validation_tier: crate::core::ValidationTier::Validated,
                min_rocm_version: String::new(),
                dependencies: vec![],
                compatible_channels: vec!["latest".to_string()],
                exclusive_group: String::new(),
            }],
            signature: None,
        };

        assert_eq!(
            RocmChannel::pkg_version_from_manifest(&manifest),
            "7.2.4.70204-1"
        );
    }

    #[test]
    fn test_pkg_version_from_manifest_legacy() {
        let manifest = crate::core::manifest::Manifest {
            schema_version: 2,
            sequence: 1,
            generated_at: "2026-05-31T00:00:00Z".to_string(),
            expires_at: None,
            min_runtime_version: String::new(),
            components: vec![crate::core::manifest::ManifestComponent {
                id: "rocm".to_string(),
                version: "6.4.3".to_string(),
                script: "scripts/install_rocm.sh".to_string(),
                category: crate::core::Category::Foundation,
                validation_tier: crate::core::ValidationTier::Validated,
                min_rocm_version: String::new(),
                dependencies: vec![],
                compatible_channels: vec!["legacy".to_string()],
                exclusive_group: String::new(),
            }],
            signature: None,
        };

        assert_eq!(
            RocmChannel::pkg_version_from_manifest(&manifest),
            "6.4.60403-1"
        );
    }

    // --- VAL-INSTALL-001: ROCm installer correct package commands ---

    #[test]
    fn test_apt_install_commands_latest() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.apt_install_commands();
        assert_eq!(cmds.len(), 4);

        // wget download
        assert_eq!(cmds[0].program, "wget");
        assert!(cmds[0]
            .args
            .iter()
            .any(|a| a.contains("7.2.4/ubuntu/noble")));
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.4.70204-1")));
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
            channel: RocmChannel::Legacy,
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
            channel: RocmChannel::Legacy,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.dnf_install_commands("9");
        assert_eq!(cmds.len(), 2);

        // dnf install RPM
        assert_eq!(cmds[0].program, "sudo");
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.4/rhel/9")));
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.4.70204-1")));
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
        let cmds = installer.pacman_install_commands("yay");
        // Single user-space `yay` command — never a bare `sudo yay` (which
        // defaults to `yay -Syu`, a full system upgrade as root). Regression
        // guard for the malformed two-command [sudo yay, yay …] that aborted
        // ROCm installs on Arch.
        assert_eq!(
            cmds.len(),
            1,
            "pacman install must be a single user-space yay command"
        );
        assert_eq!(cmds[0].program, "yay");
        assert!(
            !cmds.iter().any(|c| c.program == "sudo"),
            "yay must not run under sudo"
        );
        assert!(cmds[0].args.contains(&"-S".to_string()));
        assert!(cmds[0].args.contains(&"--needed".to_string()));
        assert!(cmds[0].args.contains(&"--noconfirm".to_string()));
        // Should have packages for Latest channel
        let pkgs = installer.pacman_rocm_packages();
        assert!(pkgs.contains(&"rocm-hip-sdk".to_string()));
        assert!(pkgs.contains(&"rocminfo".to_string()));
        assert!(pkgs.contains(&"rocm-opencl-sdk".to_string())); // Latest gets extra
        assert!(pkgs.contains(&"rccl".to_string())); // Latest gets extra
        assert!(pkgs.contains(&"rocwmma".to_string())); // Latest: WMMA FA backend
        for p in &pkgs {
            assert!(cmds[0].args.contains(p), "yay args missing package {p}");
        }
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
        assert!(!pkgs.contains(&"rocwmma".to_string())); // Legacy (RDNA2): no WMMA
    }

    #[test]
    fn test_zypper_install_commands() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            install_type: RocmInstallType::Standard,
            ..Default::default()
        });
        let cmds = installer.zypper_install_commands();
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0].program, "sudo");
        assert!(cmds[0].args.iter().any(|a| a.contains("zypper")));
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.4/sle/15.7")));
        assert!(cmds[0].args.iter().any(|a| a.contains("7.2.4.70204-1")));
    }

    // --- Repo URL construction ---

    #[test]
    fn test_repo_base_url_ubuntu() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let distro = ubuntu_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/ubuntu/noble");
    }

    #[test]
    fn test_repo_base_url_rhel() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let distro = rhel9_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/rhel/9");
    }

    #[test]
    fn test_repo_base_url_fedora() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let distro = fedora_distro();
        let url = installer.repo_base_url(&distro);
        // Fedora version "41" doesn't match 8/9/10, defaults to 9
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/rhel/9");
    }

    #[test]
    fn test_repo_base_url_suse() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let distro = suse_distro();
        let url = installer.repo_base_url(&distro);
        assert_eq!(url, "https://repo.radeon.com/amdgpu/7.2/sle/15.7");
    }

    #[test]
    fn test_amdgpu_install_deb_url() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let url = installer.amdgpu_install_deb_url();
        assert!(url.contains("7.2.4/ubuntu/noble"));
        assert!(url.contains("7.2.4.70204-1"));
        assert!(url.contains("ubuntu/noble"));
        assert!(url.ends_with(".deb"));
    }

    #[test]
    fn test_amdgpu_install_rpm_url() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
            ..Default::default()
        });
        let url = installer.amdgpu_install_rpm_url("9");
        assert!(url.contains("7.2.4/rhel/9"));
        assert!(url.contains("7.2.4.70204-1"));
        assert!(url.contains("rhel/9"));
        assert!(url.ends_with(".noarch.rpm"));
    }

    #[test]
    fn test_apt_repo_config() {
        let installer = RocmInstaller::new(RocmConfig {
            channel: RocmChannel::Legacy,
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
            channel: RocmChannel::Legacy,
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
                ch,
                ch.pkg_version()
            );

            // Verify repo URL contains major_minor
            let distro = ubuntu_distro();
            let repo_url = installer.repo_base_url(&distro);
            assert!(
                repo_url.contains(ch.major_minor()),
                "repo URL for {:?} should contain major_minor {}",
                ch,
                ch.major_minor()
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
                channel: RocmChannel::Legacy,
                install_type: itype,
                ..Default::default()
            });
            let cmds = installer.apt_install_commands();
            let meta_cmd = &cmds[3];
            assert!(
                meta_cmd.args.contains(&expected_pkg.to_string()),
                "Install type {:?} should use metapackage {}",
                itype,
                expected_pkg
            );
        }
    }

    // --- Up-to-date output detection ---

    #[test]
    fn test_is_up_to_date_yay_warning() {
        assert!(RocmInstaller::is_up_to_date_output(
            "warning: rocminfo is up to date -- skipping"
        ));
    }

    #[test]
    fn test_is_up_to_date_yay_warning_case_insensitive() {
        assert!(RocmInstaller::is_up_to_date_output(
            "WARNING: rocminfo is UP TO DATE -- skipping"
        ));
    }

    #[test]
    fn test_is_up_to_date_nothing_to_do() {
        assert!(RocmInstaller::is_up_to_date_output(
            " there is nothing to do\n"
        ));
    }

    #[test]
    fn test_is_up_to_date_already_installed() {
        assert!(RocmInstaller::is_up_to_date_output(
            "package already installed"
        ));
    }

    #[test]
    fn test_is_up_to_date_normal_output_is_false() {
        assert!(!RocmInstaller::is_up_to_date_output(
            "resolving dependencies...\nlooking for conflicting packages...\n"
        ));
    }

    #[test]
    fn test_is_up_to_date_error_output_is_false() {
        assert!(!RocmInstaller::is_up_to_date_output(
            "error: failed to prepare transaction\nerror: target not found: rocminfo"
        ));
    }

    #[test]
    fn test_is_up_to_date_empty_output_is_false() {
        assert!(!RocmInstaller::is_up_to_date_output(""));
    }

    #[test]
    fn test_is_up_to_date_pacman_skipping() {
        assert!(RocmInstaller::is_up_to_date_output(
            "warning: rocm-hip-sdk is up to date -- skipping"
        ));
    }

    // --- Pacman check-installed command construction ---

    #[test]
    fn test_build_pacman_check_installed_command() {
        let cmd = RocmInstaller::build_pacman_check_installed_command("rocminfo");
        assert_eq!(cmd.program, "pacman");
        assert_eq!(cmd.args, vec!["-Qi", "rocminfo"]);
    }

    #[test]
    fn test_build_pacman_check_installed_command_complex_pkg() {
        let cmd = RocmInstaller::build_pacman_check_installed_command("rocm-hip-sdk");
        assert_eq!(cmd.program, "pacman");
        assert!(cmd.args.contains(&"rocm-hip-sdk".to_string()));
    }
}
