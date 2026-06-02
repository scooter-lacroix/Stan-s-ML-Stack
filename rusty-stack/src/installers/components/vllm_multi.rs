//! vLLM installer — ports `scripts/install_vllm_multi.sh`.
//!
//! Constructs correct git clone URL/branch + pip install commands for vLLM
//! with ROCm support. vLLM depends on PyTorch (validated via preflight check).
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-015**: vLLM correct git clone and pip install
//! - **VAL-INSTALL-042**: vLLM declares dependency on PyTorch

use crate::installers::common::RocmEnv;

// ===========================================================================
// Types
// ===========================================================================

/// Installation method for vLLM.
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

/// Configuration for the vLLM installer.
#[derive(Debug, Clone)]
pub struct VllmConfig {
    /// ROCm version string (e.g., "7.2.0").
    pub rocm_version: String,
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_arch: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether strict ROCm mode is enabled.
    pub strict_rocm: bool,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            rocm_version: "7.2.0".to_string(),
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
            strict_rocm: true,
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

/// The vLLM installer.
pub struct VllmInstaller {
    config: VllmConfig,
}

impl VllmInstaller {
    /// Create a new vLLM installer with the given config.
    pub fn new(config: VllmConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(VllmConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependencies (VAL-INSTALL-042)
    // -----------------------------------------------------------------------

    /// vLLM depends on PyTorch.
    pub fn dependencies(&self) -> &[&str] {
        &["pytorch"]
    }

    /// Validate that all dependencies are satisfied.
    pub fn validate_dependencies(&self, installed_components: &[&str]) -> anyhow::Result<()> {
        for dep in self.dependencies() {
            if !installed_components.contains(dep) {
                anyhow::bail!("vLLM requires '{}' to be installed first", dep);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ROCm detection helpers
    // -----------------------------------------------------------------------

    /// Detect ROCm major.minor version.
    pub fn detect_rocm_mm(&self) -> String {
        let v = &self.config.rocm_version;
        // Extract major.minor from version string
        let parts: Vec<&str> = v.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            "7.2".to_string()
        }
    }

    /// Get the ROCm PyTorch wheel index URL.
    pub fn rocm_index_url(&self) -> String {
        let mm = self.detect_rocm_mm();
        format!("https://repo.radeon.com/rocm/manylinux/rocm-rel-{mm}/")
    }

    /// Get the vLLM ROCm wheels extra index URL.
    pub fn vllm_wheels_url(&self) -> &'static str {
        "https://wheels.vllm.ai/rocm/"
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-015)
    // -----------------------------------------------------------------------

    /// Construct the ROCm build environment variables.
    pub fn rocm_build_env(&self, rocm_env: &RocmEnv) -> Vec<(String, String)> {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        vec![
            ("ROCM_HOME".to_string(), rocm_path.clone()),
            ("ROCM_PATH".to_string(), rocm_path.clone()),
            ("HIP_PATH".to_string(), rocm_path.clone()),
            ("HIP_ROOT_DIR".to_string(), rocm_path),
            (
                "PYTORCH_ROCM_ARCH".to_string(),
                self.config.gpu_arch.clone(),
            ),
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string()),
            ("VLLM_TARGET_DEVICE".to_string(), "rocm".to_string()),
        ]
    }

    /// Construct the pip install command for vLLM from ROCm wheels.
    ///
    /// The original script uses:
    /// `pip install --no-cache-dir --no-deps vllm --extra-index-url https://wheels.vllm.ai/rocm/`
    pub fn build_vllm_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--no-cache-dir".to_string(),
            "--no-deps".to_string(),
            "vllm".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the pip install command for vLLM dependencies.
    ///
    /// Installs the core dependencies needed by vLLM (excluding torch and xformers).
    pub fn build_deps_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--no-cache-dir".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
        ]);

        // Core vLLM dependencies
        let deps = [
            "accelerate",
            "aiohttp",
            "cloudpickle",
            "fastapi",
            "msgspec",
            "prometheus-client",
            "psutil",
            "py-cpuinfo",
            "pyzmq",
            "requests",
            "sentencepiece",
            "tiktoken",
            "uvicorn",
            "einops",
            "transformers",
            "huggingface-hub",
            "cachetools",
            "cbor2",
            "gguf",
            "pybase64",
            "ijson",
            "python-json-logger",
            "setproctitle",
            "watchfiles",
            "six",
            "openai",
            "blake3",
            "lark",
            "amdsmi",
            "lm-format-enforcer",
            "partial-json-parser",
            "prometheus-fastapi-instrumentator",
            "datasets",
            "diskcache",
            "timm",
            "peft",
            "numba",
        ];
        for dep in deps {
            args.push(dep.to_string());
        }

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the pip install command for vLLM versioned dependencies.
    ///
    /// These have specific version pins from the original script.
    pub fn build_versioned_deps_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--no-cache-dir".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
        ]);

        // Versioned deps from original script
        let deps = [
            "openai-harmony>=0.0.3",
            "mistral-common[image]>=1.9.0",
            "triton-kernels==1.0.0",
            "outlines-core==0.2.11",
            "xgrammar==0.1.29",
            "llguidance>=1.3.0,<1.4.0",
        ];
        for dep in deps {
            args.push(dep.to_string());
        }

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the Triton cache environment setup command.
    ///
    /// Sets up writable cache directories for Triton kernel compilation.
    pub fn triton_cache_env(&self) -> Vec<(String, String)> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let triton_home = format!("{home}/.cache/mlstack/triton");

        vec![
            ("MLSTACK_TRITON_HOME".to_string(), triton_home.clone()),
            ("TRITON_HOME".to_string(), triton_home.clone()),
            (
                "TRITON_CACHE_DIR".to_string(),
                format!("{triton_home}/cache"),
            ),
            ("TRITON_DUMP_DIR".to_string(), format!("{triton_home}/dump")),
            (
                "TRITON_OVERRIDE_DIR".to_string(),
                format!("{triton_home}/override"),
            ),
        ]
    }

    /// Construct the force-reinstall command for vLLM wheel repair.
    pub fn build_force_reinstall_command(&self) -> ShellCommand {
        let args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
            "--no-cache-dir".to_string(),
            "--force-reinstall".to_string(),
            "--no-deps".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
            "vllm".to_string(),
        ];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the source rebuild command for vLLM.
    ///
    /// Used as a fallback when wheel install fails.
    pub fn build_source_rebuild_command(&self) -> ShellCommand {
        let args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
            "--no-cache-dir".to_string(),
            "--force-reinstall".to_string(),
            "--no-deps".to_string(),
            "--no-build-isolation".to_string(),
            "--no-binary".to_string(),
            "vllm".to_string(),
        ];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![
                ("VLLM_TARGET_DEVICE".to_string(), "rocm".to_string()),
                ("VLLM_USE_ROCM".to_string(), "1".to_string()),
                ("USE_ROCM".to_string(), "1".to_string()),
            ],
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-015: vLLM correct git clone and pip install ---

    #[test]
    fn test_vllm_install_command() {
        let installer = VllmInstaller::new(VllmConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_vllm_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"--no-cache-dir".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"vllm".to_string()));
        assert!(cmd.args.contains(&"--extra-index-url".to_string()));
        assert!(cmd.args.iter().any(|a| a == "https://wheels.vllm.ai/rocm/"));
    }

    #[test]
    fn test_vllm_install_command_venv() {
        let installer = VllmInstaller::new(VllmConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_vllm_install_command();
        // Venv should NOT have --break-system-packages
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"vllm".to_string()));
    }

    #[test]
    fn test_vllm_wheels_url() {
        let installer = VllmInstaller::with_defaults();
        assert_eq!(installer.vllm_wheels_url(), "https://wheels.vllm.ai/rocm/");
    }

    #[test]
    fn test_rocm_index_url() {
        let installer = VllmInstaller::new(VllmConfig {
            rocm_version: "7.2.0".to_string(),
            ..Default::default()
        });
        assert_eq!(
            installer.rocm_index_url(),
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"
        );
    }

    #[test]
    fn test_rocm_index_url_legacy_version() {
        let installer = VllmInstaller::new(VllmConfig {
            rocm_version: "7.0.0".to_string(),
            ..Default::default()
        });
        assert_eq!(
            installer.rocm_index_url(),
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/"
        );
    }

    #[test]
    fn test_detect_rocm_mm() {
        let installer = VllmInstaller::new(VllmConfig {
            rocm_version: "7.2.0".to_string(),
            ..Default::default()
        });
        assert_eq!(installer.detect_rocm_mm(), "7.2");
    }

    #[test]
    fn test_rocm_build_env() {
        let installer = VllmInstaller::new(VllmConfig {
            gpu_arch: "gfx1100".to_string(),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let env = installer.rocm_build_env(&rocm_env);
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_HOME" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(env.iter().any(|(k, v)| k == "HIP_PATH" && v == "/opt/rocm"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "PYTORCH_ROCM_ARCH" && v == "gfx1100"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "VLLM_TARGET_DEVICE" && v == "rocm"));
    }

    #[test]
    fn test_deps_install_command() {
        let installer = VllmInstaller::new(VllmConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_deps_install_command();
        assert!(cmd.args.contains(&"accelerate".to_string()));
        assert!(cmd.args.contains(&"transformers".to_string()));
        assert!(cmd.args.contains(&"einops".to_string()));
        assert!(cmd.args.contains(&"fastapi".to_string()));
        assert!(cmd.args.contains(&"amdsmi".to_string()));
    }

    #[test]
    fn test_versioned_deps_command() {
        let installer = VllmInstaller::with_defaults();
        let cmd = installer.build_versioned_deps_command();
        assert!(cmd.args.iter().any(|a| a.starts_with("triton-kernels==")));
        assert!(cmd.args.iter().any(|a| a.starts_with("xgrammar==")));
        assert!(cmd.args.iter().any(|a| a.starts_with("outlines-core==")));
    }

    #[test]
    fn test_force_reinstall_command() {
        let installer = VllmInstaller::with_defaults();
        let cmd = installer.build_force_reinstall_command();
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"vllm".to_string()));
    }

    #[test]
    fn test_source_rebuild_command() {
        let installer = VllmInstaller::with_defaults();
        let cmd = installer.build_source_rebuild_command();
        assert!(cmd.args.contains(&"--no-build-isolation".to_string()));
        assert!(cmd.args.contains(&"--no-binary".to_string()));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "VLLM_TARGET_DEVICE" && v == "rocm"));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "VLLM_USE_ROCM" && v == "1"));
    }

    #[test]
    fn test_triton_cache_env() {
        let installer = VllmInstaller::with_defaults();
        let env = installer.triton_cache_env();
        assert!(env.iter().any(|(k, _)| k == "TRITON_HOME"));
        assert!(env.iter().any(|(k, _)| k == "TRITON_CACHE_DIR"));
        assert!(env.iter().any(|(k, _)| k == "TRITON_DUMP_DIR"));
    }

    // --- VAL-INSTALL-042: vLLM declares dependency on PyTorch ---

    #[test]
    fn test_dependencies() {
        let installer = VllmInstaller::with_defaults();
        assert!(installer.dependencies().contains(&"pytorch"));
    }

    #[test]
    fn test_validate_dependencies_success() {
        let installer = VllmInstaller::with_defaults();
        assert!(installer
            .validate_dependencies(&["pytorch", "rocm"])
            .is_ok());
    }

    #[test]
    fn test_validate_dependencies_missing_pytorch() {
        let installer = VllmInstaller::with_defaults();
        let result = installer.validate_dependencies(&["rocm"]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pytorch"));
    }
}
