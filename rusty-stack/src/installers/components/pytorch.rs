//! PyTorch installer — ports `scripts/install_pytorch_rocm.sh`.
//!
//! Constructs correct pip install commands with the right index URL for
//! ROCm version + Python version wheel selection.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-003**: PyTorch installer correct wheel selection

use crate::installers::common::RocmEnv;
use std::fmt;

// ===========================================================================
// Types
// ===========================================================================

/// PyTorch release channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TorchChannel {
    /// Stable releases only.
    Stable,
    /// Latest (stable fallback to nightly).
    Latest,
    /// Nightly builds.
    Nightly,
}

impl fmt::Display for TorchChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TorchChannel::Stable => write!(f, "stable"),
            TorchChannel::Latest => write!(f, "latest"),
            TorchChannel::Nightly => write!(f, "nightly"),
        }
    }
}

impl TorchChannel {
    /// Parse from a choice number (1-3).
    pub fn from_choice(choice: u8) -> Option<Self> {
        match choice {
            1 => Some(TorchChannel::Stable),
            2 => Some(TorchChannel::Latest),
            3 => Some(TorchChannel::Nightly),
            _ => None,
        }
    }
}

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

impl fmt::Display for InstallMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstallMethod::Global => write!(f, "global"),
            InstallMethod::Venv => write!(f, "venv"),
            InstallMethod::Auto => write!(f, "auto"),
        }
    }
}

/// Configuration for the PyTorch installer.
#[derive(Debug, Clone)]
pub struct PyTorchConfig {
    /// PyTorch release channel.
    pub channel: TorchChannel,
    /// Installation method.
    pub method: InstallMethod,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to force reinstall.
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for PyTorchConfig {
    fn default() -> Self {
        Self {
            channel: TorchChannel::Latest,
            method: InstallMethod::Auto,
            python_bin: "python3".to_string(),
            force_reinstall: false,
            dry_run: false,
        }
    }
}

/// A constructed pip install command.
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

/// The PyTorch installer.
pub struct PyTorchInstaller {
    config: PyTorchConfig,
}

impl PyTorchInstaller {
    /// Create a new PyTorch installer with the given config.
    pub fn new(config: PyTorchConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(PyTorchConfig::default())
    }

    // -----------------------------------------------------------------------
    // Index URL construction (VAL-INSTALL-003)
    // -----------------------------------------------------------------------

    /// Get the PyTorch index URL for a given ROCm major.minor version.
    ///
    /// Matches the logic in the original script:
    /// - ROCm 7.x: tries Radeon manylinux first, then PyTorch nightly
    /// - ROCm 6.4+: uses PyTorch nightly builds
    /// - ROCm 6.3: uses stable builds
    /// - ROCm 6.0-6.2: uses stable builds for 6.2
    /// - ROCm 5.x: uses stable builds for 5.7
    /// - Fallback: ROCm 6.3 stable
    pub fn index_url_for_rocm(&self, rocm_mm: &str) -> String {
        let mm: f32 = rocm_mm.parse().unwrap_or(7.2);
        if mm >= 7.0 {
            // For ROCm 7.x, use the Radeon manylinux index
            format!("https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm_mm}/")
        } else if mm >= 6.4 {
            format!("https://download.pytorch.org/whl/nightly/rocm{rocm_mm}")
        } else if mm >= 6.3 {
            format!("https://download.pytorch.org/whl/rocm{rocm_mm}")
        } else if mm >= 6.0 {
            "https://download.pytorch.org/whl/rocm6.2".to_string()
        } else if mm >= 5.0 {
            "https://download.pytorch.org/whl/rocm5.7".to_string()
        } else {
            "https://download.pytorch.org/whl/rocm6.3".to_string()
        }
    }

    /// Get the PyTorch nightly index URL for a given ROCm version.
    pub fn nightly_index_url(&self, rocm_mm: &str) -> String {
        format!("https://download.pytorch.org/whl/nightly/rocm{rocm_mm}")
    }

    /// Get the Radeon manylinux index URL.
    pub fn radeon_index_url(&self, rocm_mm: &str) -> String {
        format!("https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm_mm}/")
    }

    /// Get the fallback index URL (ROCm 7.0 stable).
    pub fn fallback_index_url(&self) -> String {
        "https://download.pytorch.org/whl/rocm7.0".to_string()
    }

    // -----------------------------------------------------------------------
    // Command construction
    // -----------------------------------------------------------------------

    /// Construct pip install command for PyTorch with ROCm support.
    ///
    /// This matches the original script's pip install logic:
    /// - Uses `uv pip install` if uv is available
    /// - Uses `python3 -m pip install` as fallback
    /// - Adds `--index-url` for the correct ROCm wheel index
    /// - Adds `--break-system-packages` for global installs
    /// - Installs torch, torchvision, torchaudio
    pub fn build_install_command(&self, rocm_mm: &str, use_uv: bool) -> PipCommand {
        let index_url = self.index_url_for_rocm(rocm_mm);
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        // For global installs, always use python3 -m pip with --break-system-packages
        // instead of `uv pip install --system`, which fails on uv-managed Python
        // installations with "externally managed" errors.
        let effective_use_uv = use_uv && !is_global;

        let mut args = Vec::new();

        if effective_use_uv {
            args.push("pip".to_string());
            args.push("install".to_string());
        } else {
            args.push("-m".to_string());
            args.push("pip".to_string());
            args.push("install".to_string());
            if is_global {
                args.push("--break-system-packages".to_string());
            }
        }

        args.push("--no-cache".to_string());
        args.push("--index-url".to_string());
        args.push(index_url);
        args.push("torch".to_string());
        args.push("torchvision".to_string());
        args.push("torchaudio".to_string());

        let program = if effective_use_uv {
            "uv".to_string()
        } else {
            self.config.python_bin.clone()
        };

        PipCommand { program, args }
    }

    /// Construct the fallback command when primary install fails.
    /// Tries the ROCm 7.0 stable index.
    pub fn build_fallback_command(&self, use_uv: bool) -> PipCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        // For global installs, always use python3 -m pip with --break-system-packages
        // instead of `uv pip install --system`, which fails on uv-managed Python.
        let effective_use_uv = use_uv && !is_global;

        let mut args = Vec::new();
        if effective_use_uv {
            args.push("pip".to_string());
            args.push("install".to_string());
        } else {
            args.push("-m".to_string());
            args.push("pip".to_string());
            args.push("install".to_string());
            if is_global {
                args.push("--break-system-packages".to_string());
            }
        }

        args.push("--no-cache".to_string());
        args.push("--index-url".to_string());
        args.push(self.fallback_index_url());
        args.push("torch".to_string());
        args.push("torchvision".to_string());
        args.push("torchaudio".to_string());

        let program = if effective_use_uv {
            "uv".to_string()
        } else {
            self.config.python_bin.clone()
        };

        PipCommand { program, args }
    }

    /// Construct the common ML dependencies install command.
    ///
    /// The original script installs `torchsde` and `sentencepiece` after PyTorch.
    pub fn build_common_deps_command(&self, use_uv: bool) -> PipCommand {
        let is_global = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        // For global installs, always use python3 -m pip with --break-system-packages
        // instead of `uv pip install --system`, which fails on uv-managed Python.
        let effective_use_uv = use_uv && !is_global;

        let mut args = Vec::new();
        if effective_use_uv {
            args.push("pip".to_string());
            args.push("install".to_string());
        } else {
            args.push("-m".to_string());
            args.push("pip".to_string());
            args.push("install".to_string());
            args.push("--break-system-packages".to_string());
        }
        args.push("--no-cache-dir".to_string());
        args.push("--upgrade".to_string());
        args.push("torchsde".to_string());
        args.push("sentencepiece".to_string());

        let program = if effective_use_uv {
            "uv".to_string()
        } else {
            self.config.python_bin.clone()
        };
        PipCommand { program, args }
    }

    /// Construct the ROCm environment variable exports.
    ///
    /// Returns a list of (VAR, VALUE) pairs matching the original script.
    pub fn rocm_env_exports(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let mut exports = vec![
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
            ("PYTORCH_ROCM_ARCH".to_string(), "gfx1100".to_string()),
            (
                "ROCM_PATH".to_string(),
                rocm_env
                    .path()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/opt/rocm".to_string()),
            ),
        ];

        // HSA_TOOLS_LIB - check for rocprofiler library
        // Try the ROCm 7.x layout first (lib/rocprofiler-sdk/), then the old layout (lib/)
        let rocm_lib = rocm_env.path().and_then(|p| {
            let new_layout = p.join("lib/rocprofiler-sdk/librocprofiler-sdk-tool.so");
            let old_layout = p.join("lib/librocprofiler-sdk-tool.so");
            if new_layout.exists() {
                Some(new_layout)
            } else if old_layout.exists() {
                Some(old_layout)
            } else {
                None
            }
        });
        if let Some(lib) = rocm_lib {
            exports.push((
                "HSA_TOOLS_LIB".to_string(),
                lib.to_string_lossy().to_string(),
            ));
        } else {
            exports.push(("HSA_TOOLS_LIB".to_string(), "0".to_string()));
        }

        exports
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VAL-INSTALL-003: PyTorch installer correct wheel selection ---

    #[test]
    fn test_index_url_rocm_72() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("7.2");
        assert_eq!(url, "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/");
    }

    #[test]
    fn test_index_url_rocm_70() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("7.0");
        assert_eq!(url, "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/");
    }

    #[test]
    fn test_index_url_rocm_64() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("6.4");
        assert_eq!(url, "https://download.pytorch.org/whl/nightly/rocm6.4");
    }

    #[test]
    fn test_index_url_rocm_63() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("6.3");
        assert_eq!(url, "https://download.pytorch.org/whl/rocm6.3");
    }

    #[test]
    fn test_index_url_rocm_62() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("6.2");
        assert_eq!(url, "https://download.pytorch.org/whl/rocm6.2");
    }

    #[test]
    fn test_index_url_rocm_57() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("5.7");
        assert_eq!(url, "https://download.pytorch.org/whl/rocm5.7");
    }

    #[test]
    fn test_index_url_fallback() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.index_url_for_rocm("4.0");
        assert_eq!(url, "https://download.pytorch.org/whl/rocm6.3");
    }

    #[test]
    fn test_nightly_index_url() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.nightly_index_url("7.2");
        assert_eq!(url, "https://download.pytorch.org/whl/nightly/rocm7.2");
    }

    #[test]
    fn test_radeon_index_url() {
        let installer = PyTorchInstaller::with_defaults();
        let url = installer.radeon_index_url("7.2");
        assert_eq!(url, "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/");
    }

    #[test]
    fn test_build_install_command_uv_global_uses_pip() {
        // Global + uv: should use python3 -m pip (not uv --system) to avoid
        // "externally managed" errors on uv-managed Python installations.
        let installer = PyTorchInstaller::new(PyTorchConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command("7.2", true);
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"--index-url".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("rocm-rel-7.2")));
        assert!(cmd.args.contains(&"torch".to_string()));
    }

    #[test]
    fn test_build_install_command_pip() {
        let installer = PyTorchInstaller::new(PyTorchConfig {
            method: InstallMethod::Global,
            python_bin: "python3".to_string(),
            ..Default::default()
        });
        let cmd = installer.build_install_command("6.4", false);
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("rocm6.4")));
    }

    #[test]
    fn test_build_install_command_venv_no_break_system() {
        let installer = PyTorchInstaller::new(PyTorchConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command("7.2", true);
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(!cmd.args.contains(&"--system".to_string()));
    }

    #[test]
    fn test_build_fallback_command() {
        let installer = PyTorchInstaller::new(PyTorchConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_fallback_command(true);
        assert!(cmd.args.iter().any(|a| a.contains("rocm7.0")));
        assert!(cmd.args.contains(&"torch".to_string()));
    }

    #[test]
    fn test_build_common_deps_command() {
        let installer = PyTorchInstaller::with_defaults();
        let cmd = installer.build_common_deps_command(true);
        assert!(cmd.args.contains(&"torchsde".to_string()));
        assert!(cmd.args.contains(&"sentencepiece".to_string()));
    }

    #[test]
    fn test_rocm_env_exports() {
        let installer = PyTorchInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(
            Some(std::path::PathBuf::from("/opt/rocm")),
            "7.2.0".to_string(),
        );
        let exports = installer.rocm_env_exports(&rocm_env);
        assert!(exports
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
        assert!(exports
            .iter()
            .any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
        assert!(exports
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(exports.iter().any(|(k, _)| k == "HSA_TOOLS_LIB"));
    }

    #[test]
    fn test_channel_from_choice() {
        assert_eq!(TorchChannel::from_choice(1), Some(TorchChannel::Stable));
        assert_eq!(TorchChannel::from_choice(2), Some(TorchChannel::Latest));
        assert_eq!(TorchChannel::from_choice(3), Some(TorchChannel::Nightly));
        assert_eq!(TorchChannel::from_choice(0), None);
    }
}
