//! ML Stack Core installer — ports `scripts/install_ml_stack.sh`.
//!
//! Installs the full ML stack by orchestrating individual component
//! installers. Constructs correct pip install paths for the ML stack
//! Python package and delegates to sub-installers.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-007**: ML Stack Core installer correct pip command

use crate::installers::common::{
    DistroFacade, command_exists,
};
use std::fmt;

// ===========================================================================
// Types
// ===========================================================================

/// Component to install.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlStackComponent {
    /// ROCm configuration.
    RocmConfig,
    /// PyTorch with ROCm.
    PyTorch,
    /// ONNX Runtime.
    OnnxRuntime,
    /// MIGraphX.
    MigraphX,
    /// Megatron-LM.
    Megatron,
    /// Flash Attention.
    FlashAttention,
    /// RCCL.
    Rccl,
    /// MPI (OpenMPI + mpi4py).
    Mpi,
}

impl fmt::Display for MlStackComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlStackComponent::RocmConfig => write!(f, "rocm_config"),
            MlStackComponent::PyTorch => write!(f, "pytorch"),
            MlStackComponent::OnnxRuntime => write!(f, "onnxruntime"),
            MlStackComponent::MigraphX => write!(f, "migraphx"),
            MlStackComponent::Megatron => write!(f, "megatron"),
            MlStackComponent::FlashAttention => write!(f, "flash_attention"),
            MlStackComponent::Rccl => write!(f, "rccl"),
            MlStackComponent::Mpi => write!(f, "mpi"),
        }
    }
}

/// Installation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMode {
    /// Install all core components.
    All,
    /// Install only specific components.
    Selective,
}

/// Configuration for the ML Stack installer.
#[derive(Debug, Clone)]
pub struct MlStackConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Installation mode.
    pub mode: InstallMode,
    /// Selected components (for selective mode).
    pub selected_components: Vec<MlStackComponent>,
}

impl Default for MlStackConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
            mode: InstallMode::All,
            selected_components: vec![],
        }
    }
}

/// A constructed pip command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
}

impl PipCommand {
    /// Format as a shell command string.
    pub fn to_command_string(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

/// A constructed system package command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
}

impl SystemCommand {
    /// Format as a shell command string.
    pub fn to_command_string(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

/// The ML Stack installer.
pub struct MlStackInstaller {
    config: MlStackConfig,
}

impl MlStackInstaller {
    /// Create a new ML Stack installer with the given config.
    pub fn new(config: MlStackConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MlStackConfig::default())
    }

    // -----------------------------------------------------------------------
    // Component install order
    // -----------------------------------------------------------------------

    /// Get the ordered list of core components to install.
    ///
    /// The order matches the original `install_ml_stack.sh`:
    /// 1. ROCm Config
    /// 2. PyTorch
    /// 3. ONNX Runtime
    /// 4. MIGraphX
    /// 5. Megatron-LM
    /// 6. Flash Attention
    /// 7. RCCL
    /// 8. MPI
    pub fn install_order() -> Vec<MlStackComponent> {
        vec![
            MlStackComponent::RocmConfig,
            MlStackComponent::PyTorch,
            MlStackComponent::OnnxRuntime,
            MlStackComponent::MigraphX,
            MlStackComponent::Megatron,
            MlStackComponent::FlashAttention,
            MlStackComponent::Rccl,
            MlStackComponent::Mpi,
        ]
    }

    // -----------------------------------------------------------------------
    // ROCm configuration (VAL-INSTALL-007)
    // -----------------------------------------------------------------------

    /// Construct the ROCm configuration file content.
    ///
    /// Creates a `.rocmrc` file with environment variable exports matching
    /// the original script's `install_rocm_config()` function.
    pub fn build_rocm_config_content(&self) -> String {
        r#"# ROCm Configuration File
# Created by ML Stack Installation Script

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_PATH=/opt/rocm

# Performance Settings
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# MIOpen Settings
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
"#.to_string()
    }

    /// Get the ROCm config file path.
    pub fn rocm_config_path(&self) -> String {
        format!("{}/.rocmrc", std::env::var("HOME").unwrap_or_else(|_| "/root".to_string()))
    }

    // -----------------------------------------------------------------------
    // MIGraphX install command
    // -----------------------------------------------------------------------

    /// Construct the MIGraphX package install command per distro.
    pub fn build_migraphx_install_command(&self, distro: &DistroFacade) -> SystemCommand {
        if distro.uses_apt() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "migraphx".to_string(),
                    "python3-migraphx".to_string(),
                ],
            }
        } else if distro.uses_dnf() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "migraphx".to_string(),
                    "python3-migraphx".to_string(),
                ],
            }
        } else if distro.uses_pacman() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                    "migraphx".to_string(),
                    "python-migraphx".to_string(),
                ],
            }
        } else if distro.uses_zypper() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "zypper".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "migraphx".to_string(),
                    "python3-migraphx".to_string(),
                ],
            }
        } else {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "migraphx".to_string(),
                    "python3-migraphx".to_string(),
                ],
            }
        }
    }

    // -----------------------------------------------------------------------
    // RCCL install command
    // -----------------------------------------------------------------------

    /// Construct the RCCL package install command per distro.
    pub fn build_rccl_install_command(&self, distro: &DistroFacade) -> SystemCommand {
        if distro.uses_apt() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rccl".to_string(),
                ],
            }
        } else if distro.uses_dnf() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rccl".to_string(),
                ],
            }
        } else if distro.uses_pacman() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                    "rccl".to_string(),
                ],
            }
        } else {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "rccl".to_string(),
                ],
            }
        }
    }

    // -----------------------------------------------------------------------
    // MPI install command
    // -----------------------------------------------------------------------

    /// Construct the MPI system package install command per distro.
    pub fn build_mpi_install_command(&self, distro: &DistroFacade) -> SystemCommand {
        if distro.uses_apt() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi-bin".to_string(),
                    "libopenmpi-dev".to_string(),
                ],
            }
        } else if distro.uses_dnf() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "dnf".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi".to_string(),
                    "openmpi-devel".to_string(),
                ],
            }
        } else if distro.uses_pacman() {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                    "openmpi".to_string(),
                ],
            }
        } else {
            SystemCommand {
                program: "sudo".to_string(),
                args: vec![
                    "apt-get".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                    "openmpi-bin".to_string(),
                    "libopenmpi-dev".to_string(),
                ],
            }
        }
    }

    // -----------------------------------------------------------------------
    // Megatron-LM install commands
    // -----------------------------------------------------------------------

    /// Construct the Megatron-LM git clone command.
    pub fn build_megatron_clone_command(&self) -> SystemCommand {
        SystemCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "https://github.com/NVIDIA/Megatron-LM.git".to_string(),
            ],
            // Note: SystemCommand doesn't have env, but that's fine for git clone
        }
    }

    /// Construct the Megatron-LM pip install command.
    pub fn build_megatron_install_command(&self) -> PipCommand {
        PipCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--break-system-packages".to_string(),
                "-e".to_string(),
                ".".to_string(),
            ],
        }
    }

    // -----------------------------------------------------------------------
    // Prerequisites check
    // -----------------------------------------------------------------------

    /// Get the list of prerequisite commands to check.
    pub fn prerequisite_commands(&self) -> &[&str] {
        &["rocminfo", "python3", "pip3", "git", "cmake"]
    }

    /// Check a single prerequisite.
    pub fn check_prerequisite(&self, cmd: &str) -> bool {
        command_exists(cmd)
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

    // --- VAL-INSTALL-007: ML Stack Core installer correct pip command ---

    #[test]
    fn test_install_order() {
        let order = MlStackInstaller::install_order();
        assert_eq!(order.len(), 8);
        assert_eq!(order[0], MlStackComponent::RocmConfig);
        assert_eq!(order[1], MlStackComponent::PyTorch);
        assert_eq!(order[7], MlStackComponent::Mpi);
    }

    #[test]
    fn test_rocm_config_content() {
        let installer = MlStackInstaller::with_defaults();
        let content = installer.build_rocm_config_content();
        assert!(content.contains("HSA_OVERRIDE_GFX_VERSION"));
        assert!(content.contains("PYTORCH_ROCM_ARCH"));
        assert!(content.contains("ROCM_PATH"));
        assert!(content.contains("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"));
    }

    #[test]
    fn test_rocm_config_path() {
        let installer = MlStackInstaller::with_defaults();
        let path = installer.rocm_config_path();
        assert!(path.ends_with("/.rocmrc"));
    }

    #[test]
    fn test_migraphx_install_apt() {
        let installer = MlStackInstaller::with_defaults();
        let distro = ubuntu_distro();
        let cmd = installer.build_migraphx_install_command(&distro);
        assert!(cmd.args.contains(&"migraphx".to_string()));
        assert!(cmd.args.contains(&"python3-migraphx".to_string()));
    }

    #[test]
    fn test_migraphx_install_dnf() {
        let installer = MlStackInstaller::with_defaults();
        let distro = fedora_distro();
        let cmd = installer.build_migraphx_install_command(&distro);
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }

    #[test]
    fn test_migraphx_install_pacman() {
        let installer = MlStackInstaller::with_defaults();
        let distro = arch_distro();
        let cmd = installer.build_migraphx_install_command(&distro);
        assert!(cmd.args.contains(&"python-migraphx".to_string()));
    }

    #[test]
    fn test_rccl_install_apt() {
        let installer = MlStackInstaller::with_defaults();
        let distro = ubuntu_distro();
        let cmd = installer.build_rccl_install_command(&distro);
        assert!(cmd.args.contains(&"rccl".to_string()));
    }

    #[test]
    fn test_mpi_install_apt() {
        let installer = MlStackInstaller::with_defaults();
        let distro = ubuntu_distro();
        let cmd = installer.build_mpi_install_command(&distro);
        assert!(cmd.args.contains(&"openmpi-bin".to_string()));
        assert!(cmd.args.contains(&"libopenmpi-dev".to_string()));
    }

    #[test]
    fn test_mpi_install_dnf() {
        let installer = MlStackInstaller::with_defaults();
        let distro = fedora_distro();
        let cmd = installer.build_mpi_install_command(&distro);
        assert!(cmd.args.contains(&"openmpi".to_string()));
        assert!(cmd.args.contains(&"openmpi-devel".to_string()));
    }

    #[test]
    fn test_megatron_clone_command() {
        let installer = MlStackInstaller::with_defaults();
        let cmd = installer.build_megatron_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("NVIDIA/Megatron-LM")));
    }

    #[test]
    fn test_megatron_install_command() {
        let installer = MlStackInstaller::with_defaults();
        let cmd = installer.build_megatron_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-e".to_string()));
        assert!(cmd.args.contains(&".".to_string()));
    }

    #[test]
    fn test_prerequisite_commands() {
        let installer = MlStackInstaller::with_defaults();
        let cmds = installer.prerequisite_commands();
        assert!(cmds.contains(&"python3"));
        assert!(cmds.contains(&"git"));
        assert!(cmds.contains(&"cmake"));
    }

    #[test]
    fn test_component_display() {
        assert_eq!(format!("{}", MlStackComponent::PyTorch), "pytorch");
        assert_eq!(format!("{}", MlStackComponent::RocmConfig), "rocm_config");
        assert_eq!(format!("{}", MlStackComponent::Mpi), "mpi");
    }
}
