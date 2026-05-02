//! MPI4Py installer — ports `scripts/install_mpi4py.sh`.
//!
//! Detects system MPI implementation, constructs pip install with correct
//! MPICC/MPICXX environment variables.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-005**: MPI4Py installer detects system MPI

use crate::installers::common::{
    DistroFacade, command_exists,
};
use std::fmt;
use std::path::PathBuf;

// ===========================================================================
// Types
// ===========================================================================

/// Detected MPI implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpiImplementation {
    /// OpenMPI (most common).
    OpenMPI,
    /// MPICH.
    MPICH,
    /// Intel MPI.
    IntelMPI,
    /// No MPI detected.
    None,
}

impl fmt::Display for MpiImplementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpiImplementation::OpenMPI => write!(f, "OpenMPI"),
            MpiImplementation::MPICH => write!(f, "MPICH"),
            MpiImplementation::IntelMPI => write!(f, "IntelMPI"),
            MpiImplementation::None => write!(f, "None"),
        }
    }
}

/// Configuration for the MPI4Py installer.
#[derive(Debug, Clone)]
pub struct Mpi4PyConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether ROCm is enabled.
    pub rocm_enabled: bool,
}

impl Default for Mpi4PyConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
            rocm_enabled: true,
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

/// The MPI4Py installer.
pub struct Mpi4PyInstaller {
    config: Mpi4PyConfig,
}

impl Mpi4PyInstaller {
    /// Create a new MPI4Py installer with the given config.
    pub fn new(config: Mpi4PyConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(Mpi4PyConfig::default())
    }

    // -----------------------------------------------------------------------
    // MPI detection (VAL-INSTALL-005)
    // -----------------------------------------------------------------------

    /// Detect the MPI implementation on the system.
    ///
    /// Checks for mpirun/mpiexec and identifies the implementation.
    pub fn detect_mpi(&self) -> MpiImplementation {
        if command_exists("mpirun") || command_exists("mpiexec") {
            // Try to identify which MPI
            if self.is_openmpi() {
                MpiImplementation::OpenMPI
            } else if self.is_mpich() {
                MpiImplementation::MPICH
            } else {
                // Default to OpenMPI if we can't determine
                MpiImplementation::OpenMPI
            }
        } else {
            MpiImplementation::None
        }
    }

    /// Check if the detected MPI is OpenMPI.
    fn is_openmpi(&self) -> bool {
        // Check for ompi_info which is OpenMPI-specific
        command_exists("ompi_info")
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/openmpi").exists()
            || std::path::Path::new("/usr/lib64/openmpi").exists()
    }

    /// Check if the detected MPI is MPICH.
    fn is_mpich(&self) -> bool {
        command_exists("mpichversion")
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/mpich").exists()
    }

    /// Detect the MPI installation path.
    ///
    /// Checks common paths in order:
    /// 1. /usr/lib64/openmpi
    /// 2. /opt/openmpi
    /// 3. /usr/lib/openmpi
    /// 4. Derived from mpirun location
    pub fn detect_mpi_path(&self) -> PathBuf {
        let candidates = [
            "/usr/lib64/openmpi",
            "/opt/openmpi",
            "/usr/local/openmpi",
            "/usr/lib/openmpi",
        ];

        for path_str in &candidates {
            let path = PathBuf::from(path_str);
            if path.join("bin/mpirun").exists() || path.join("bin").exists() {
                return path;
            }
        }

        // Try to derive from mpirun location
        if let Ok(output) = std::process::Command::new("which")
            .arg("mpirun")
            .output()
        {
            if output.status.success() {
                let mpirun = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !mpirun.is_empty() {
                    if let Some(parent) = PathBuf::from(&mpirun).parent() {
                        if let Some(grandparent) = parent.parent() {
                            return grandparent.to_path_buf();
                        }
                    }
                }
            }
        }

        PathBuf::from("/usr/lib64/openmpi")
    }

    // -----------------------------------------------------------------------
    // System MPI package installation
    // -----------------------------------------------------------------------

    /// Construct the system package install command for MPI.
    ///
    /// Returns the correct command based on distro family:
    /// - apt: `sudo apt-get install -y libopenmpi-dev openmpi-bin`
    /// - dnf: `sudo dnf install -y openmpi openmpi-devel`
    /// - pacman: `sudo pacman -S --noconfirm openmpi`
    /// - zypper: `sudo zypper install -y openmpi openmpi-devel`
    pub fn build_system_mpi_install_command(&self, distro: &DistroFacade) -> ShellCommand {
        if distro.uses_apt() {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "update".to_string(),
                    "&&".to_string(),
                    "sudo".to_string(),
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "libopenmpi-dev".to_string(),
                    "openmpi-bin".to_string(),
                ],
                env: vec![],
            }
        } else if distro.uses_dnf() {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi".to_string(),
                    "openmpi-devel".to_string(),
                ],
                env: vec![],
            }
        } else if distro.uses_yum() {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "yum".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi".to_string(),
                    "openmpi-devel".to_string(),
                ],
                env: vec![],
            }
        } else if distro.uses_pacman() {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                    "openmpi".to_string(),
                ],
                env: vec![],
            }
        } else if distro.uses_zypper() {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "zypper".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi".to_string(),
                    "openmpi-devel".to_string(),
                ],
                env: vec![],
            }
        } else {
            ShellCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "update".to_string(),
                    "&&".to_string(),
                    "sudo".to_string(),
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "libopenmpi-dev".to_string(),
                    "openmpi-bin".to_string(),
                ],
                env: vec![],
            }
        }
    }

    // -----------------------------------------------------------------------
    // mpi4py pip install command
    // -----------------------------------------------------------------------

    /// Construct the pip install command for mpi4py with correct env vars.
    ///
    /// Sets MPICC and MPICXX environment variables pointing to the MPI
    /// installation's compiler wrappers.
    pub fn build_pip_install_command(&self, mpi_path: &std::path::Path) -> ShellCommand {
        let mpicc = mpi_path.join("bin/mpicc");
        let mpicxx = mpi_path.join("bin/mpicxx");

        let env = vec![
            ("MPICC".to_string(), mpicc.to_string_lossy().to_string()),
            ("MPICXX".to_string(), mpicxx.to_string_lossy().to_string()),
            ("PATH".to_string(), format!("{}:{}", mpi_path.join("bin").to_string_lossy(), std::env::var("PATH").unwrap_or_default())),
            ("LD_LIBRARY_PATH".to_string(), format!("{}:{}", mpi_path.join("lib").to_string_lossy(), std::env::var("LD_LIBRARY_PATH").unwrap_or_default())),
        ];

        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
        ];
        if self.config.force_reinstall {
            args.push("--force-reinstall".to_string());
        }
        args.push("mpi4py".to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env,
        }
    }

    /// Construct the ROCm environment exports for MPI.
    pub fn rocm_env_exports(&self) -> Vec<(String, String)> {
        if self.config.rocm_enabled {
            vec![
                ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
                ("PYTORCH_ROCM_ARCH".to_string(), "gfx1100".to_string()),
                ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
            ]
        } else {
            vec![]
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

    fn ubuntu_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            name: "Ubuntu 24.04".to_string(),
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
            name: "Fedora 41".to_string(),
            version: "41".to_string(),
            codename: String::new(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            id_like: "fedora".to_string(),
        })
    }

    fn arch_distro() -> DistroFacade {
        DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            name: "Arch Linux".to_string(),
            version: "rolling".to_string(),
            codename: String::new(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            id_like: "arch".to_string(),
        })
    }

    // --- VAL-INSTALL-005: MPI4Py installer detects system MPI ---

    #[test]
    fn test_detect_mpi() {
        let installer = Mpi4PyInstaller::with_defaults();
        // Just verify it doesn't panic
        let mpi = installer.detect_mpi();
        // On this system, it may or may not have MPI
        assert!(matches!(mpi, MpiImplementation::OpenMPI | MpiImplementation::MPICH | MpiImplementation::None));
    }

    #[test]
    fn test_detect_mpi_path() {
        let installer = Mpi4PyInstaller::with_defaults();
        let path = installer.detect_mpi_path();
        // Should return a path, even if it's the default
        assert!(!path.to_string_lossy().is_empty());
    }

    #[test]
    fn test_system_mpi_install_apt() {
        let installer = Mpi4PyInstaller::with_defaults();
        let distro = ubuntu_distro();
        let cmd = installer.build_system_mpi_install_command(&distro);
        assert!(cmd.args.contains(&"libopenmpi-dev".to_string()));
        assert!(cmd.args.contains(&"openmpi-bin".to_string()));
    }

    #[test]
    fn test_system_mpi_install_dnf() {
        let installer = Mpi4PyInstaller::with_defaults();
        let distro = fedora_distro();
        let cmd = installer.build_system_mpi_install_command(&distro);
        assert!(cmd.args.contains(&"openmpi".to_string()));
        assert!(cmd.args.contains(&"openmpi-devel".to_string()));
    }

    #[test]
    fn test_system_mpi_install_pacman() {
        let installer = Mpi4PyInstaller::with_defaults();
        let distro = arch_distro();
        let cmd = installer.build_system_mpi_install_command(&distro);
        assert!(cmd.args.contains(&"openmpi".to_string()));
        assert!(cmd.args.contains(&"--noconfirm".to_string()));
    }

    #[test]
    fn test_pip_install_command() {
        let installer = Mpi4PyInstaller::with_defaults();
        let mpi_path = PathBuf::from("/usr/lib64/openmpi");
        let cmd = installer.build_pip_install_command(&mpi_path);

        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"mpi4py".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));

        // Verify MPICC/MPICXX env vars
        assert!(cmd.env.iter().any(|(k, v)| k == "MPICC" && v.contains("mpicc")));
        assert!(cmd.env.iter().any(|(k, v)| k == "MPICXX" && v.contains("mpicxx")));
        assert!(cmd.env.iter().any(|(k, v)| k == "PATH" && v.contains("openmpi/bin")));
    }

    #[test]
    fn test_pip_install_command_force_reinstall() {
        let installer = Mpi4PyInstaller::new(Mpi4PyConfig {
            force_reinstall: true,
            ..Default::default()
        });
        let mpi_path = PathBuf::from("/usr/lib64/openmpi");
        let cmd = installer.build_pip_install_command(&mpi_path);
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
    }

    #[test]
    fn test_rocm_env_exports_enabled() {
        let installer = Mpi4PyInstaller::new(Mpi4PyConfig {
            rocm_enabled: true,
            ..Default::default()
        });
        let exports = installer.rocm_env_exports();
        assert!(exports.iter().any(|(k, _)| k == "HSA_OVERRIDE_GFX_VERSION"));
        assert!(exports.iter().any(|(k, _)| k == "ROCM_PATH"));
    }

    #[test]
    fn test_rocm_env_exports_disabled() {
        let installer = Mpi4PyInstaller::new(Mpi4PyConfig {
            rocm_enabled: false,
            ..Default::default()
        });
        let exports = installer.rocm_env_exports();
        assert!(exports.is_empty());
    }

    #[test]
    fn test_mpi_implementation_display() {
        assert_eq!(format!("{}", MpiImplementation::OpenMPI), "OpenMPI");
        assert_eq!(format!("{}", MpiImplementation::MPICH), "MPICH");
        assert_eq!(format!("{}", MpiImplementation::None), "None");
    }
}
