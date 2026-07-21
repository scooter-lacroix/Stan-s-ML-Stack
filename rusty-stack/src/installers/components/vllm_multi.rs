//! vLLM installer — ports `scripts/install_vllm_multi.sh`.
//!
//! Constructs correct git clone URL/branch + pip install commands for vLLM
//! with ROCm support. vLLM depends on PyTorch (validated via preflight check).
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-015**: vLLM correct git clone and pip install
//! - **VAL-INSTALL-042**: vLLM declares dependency on PyTorch

/// Fallback vLLM version — used ONLY when PyPI is unreachable or no release
/// satisfies the user's version/age gate. Otherwise the installer resolves a
/// live version from PyPI (see `resolve_version_with_policy`) so no manual bump
/// is needed across releases. Also provides the `VLLM_VERSION_OVERRIDE` value
/// that suppresses vLLM's `+rocmNNN` build label (avoids pip's filename-vs-
/// metadata cascade).
const VLLM_SOURCE_VERSION: &str = "0.25.1";

/// Python: query PyPI for vLLM releases, apply the version-lag + age gate, and
/// print the selected version (empty if none qualifies). Run as
/// `python -c <this> <lag> <age>` (argv[1]=lag, argv[2]=age). Drops pre-releases,
/// sorts by version descending, then picks the newest release that is BOTH
/// beyond the newest `lag` AND at least `age` days old.
const VLLM_VERSION_POLICY_PY: &str = r#"
import json, urllib.request, sys, datetime
lag = int(sys.argv[1]); age = int(sys.argv[2])
try:
    from packaging.version import Version
except Exception:
    Version = None
data = json.load(urllib.request.urlopen('https://pypi.org/pypi/vllm/json', timeout=20))
rels = data.get('releases', {})
now = datetime.datetime.now(datetime.timezone.utc)
cands = []
for ver, files in rels.items():
    if not files:
        continue
    if Version:
        try:
            lv = Version(ver)
        except Exception:
            continue
        if lv.is_prerelease:
            continue
    ut = files[0].get('upload_time_iso_8601') or files[0].get('upload_time') or ''
    try:
        dt = datetime.datetime.fromisoformat(ut.replace('Z', '+00:00'))
    except Exception:
        continue
    # upload_time (legacy) is timezone-naive; coerce to UTC so now-dt works.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    cands.append((ver, dt, lv if Version else None))
cands.sort(key=lambda x: (x[2] if Version else x[1]), reverse=True)
for idx, (ver, dt, _lv) in enumerate(cands):
    if idx < lag:
        continue
    if age > 0 and (now - dt).days < age:
        continue
    print(ver)
    sys.exit(0)
print('')
"#;

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
            "--no-deps".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
        ]);

        // vLLM 0.25.x direct deps from package metadata, excluding torch/triton/
        // xformers and CUDA/NVIDIA packages. `--no-deps` is intentional: vLLM is
        // torch-adjacent, so Rusty must not let transitive deps replace ROCm
        // stack packages or pull CUDA runtime wheels.
        let deps = [
            "aiohttp>=3.13.3",
            "amd-quark>=0.8.99",
            "anthropic>=0.71.0",
            "astor",
            "blake3",
            "boto3",
            "botocore",
            "cachetools",
            "cbor2",
            "cloudpickle",
            "datasets",
            "docstring-parser",
            "einops",
            "evaluate",
            "fastsafetensors>=0.3.2",
            "filelock>=3.16.1",
            "googleapis-common-protos",
            "hiredis",
            "httpx-sse",
            "humanize",
            "ijson",
            "importlib-metadata",
            "jmespath",
            "joblib",
            "jsonschema>=4.23.0",
            "libnacl",
            "loguru",
            "mcp",
            "mistral-common[image]>=1.11.5",
            "ml-dtypes",
            "model-hosting-container-standards>=0.1.14,<1.0.0",
            "msgspec",
            "colorama",
            "narwhals",
            "onnx-ir",
            "onnxscript",
            "onnxslim",
            "openai>=2.0.0",
            "openai-harmony>=0.0.3",
            "opentelemetry-api>=1.27.0",
            "opentelemetry-exporter-otlp>=1.27.0",
            "opentelemetry-exporter-otlp-proto-common>=1.27.0",
            "opentelemetry-exporter-otlp-proto-grpc>=1.27.0",
            "opentelemetry-exporter-otlp-proto-http>=1.27.0",
            "opentelemetry-proto>=1.27.0",
            "opentelemetry-sdk>=1.27.0",
            "opentelemetry-semantic-conventions>=0.59b0",
            "opentelemetry-semantic-conventions-ai>=0.4.1",
            "opentelemetry-util-http>=0.59b0",
            "opencv-python-headless>=4.13.0",
            "packaging>=24.2",
            "partial-json-parser",
            "peft",
            "pillow",
            "plotly",
            "prometheus-client>=0.18.0",
            "prometheus-fastapi-instrumentator>=8.0.0",
            "protobuf>=6.33.5,<7.0.0",
            "psutil",
            "py-cpuinfo",
            "pybase64",
            "pydantic-settings",
            "pytest-asyncio",
            "pyjwt",
            "python-json-logger",
            "python-dotenv",
            "python-multipart",
            "pyyaml",
            "pyzmq>=25.0.0",
            "redis",
            "regex",
            "requests>=2.26.0",
            "safetensors>=0.6.2",
            "s3transfer",
            "sentencepiece",
            "setproctitle",
            "setuptools-rust>=1.9.0",
            "setuptools-scm>=8",
            "six>=1.16.0",
            "sse-starlette",
            "starlette>=1.0.1",
            "supervisor",
            "tiktoken>=0.6.0",
            "timm>=1.0.17",
            "tokenizers>=0.21.1",
            "tqdm",
            "transformers>=5.5.3",
            "typing-extensions>=4.10",
            "uvicorn",
            "uvloop",
            "watchfiles",
            "zipp",
            "zstandard",
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
            "--no-deps".to_string(),
            "--extra-index-url".to_string(),
            self.vllm_wheels_url().to_string(),
        ]);

        // Versioned vLLM 0.25.x deps from package metadata, excluding torch/CUDA.
        let deps = [
            "apache-tvm-ffi==0.1.10",
            "compressed-tensors==0.17.0",
            "conch-triton-kernels==1.2.1",
            "depyf==0.20.0",
            "diskcache==5.6.3",
            "fastapi[standard]>=0.133.0,<0.137.0",
            "grpcio==1.78.0",
            "grpcio-reflection==1.78.0",
            "lark==1.2.2",
            "llguidance>=1.7.0,<1.8.0",
            "llvmlite>=0.47.0,<0.48.0",
            "lm-format-enforcer==0.11.3",
            "numba==0.65.0",
            "outlines-core==0.2.14",
            "runai-model-streamer[azure,gcs,s3]==0.15.7",
            "setuptools>=77.0.3,<80.0.0",
            "tensorizer==2.10.1",
            "tilelang==0.1.10",
            "torch-c-dlpack-ext",
            "xgrammar>=0.2.1,<1.0.0",
            "z3-solver>=4.13.0,<4.15.5",
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

    /// Install the build tools vLLM's source build needs (cmake/ninja/wheel/
    /// setuptools/patchelf). The env already supplies ROCm dev headers + torch;
    /// these are the remaining pip-installable build tools. (rust/cargo, if an
    /// optional dep ever needs it, is a system package — not pip-installable.)
    pub fn build_tools_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;
        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--no-cache-dir".to_string(),
            "cmake".to_string(),
            "ninja".to_string(),
            "wheel".to_string(),
            "setuptools".to_string(),
            // vLLM's sdist setup.py imports these at metadata-generation time
            // (`ModuleNotFoundError: No module named 'setuptools_rust'`). cargo/
            // rustc are a system dep (present at /usr/bin/cargo) — not pip.
            "setuptools-rust".to_string(),
            "setuptools-scm".to_string(),
            "patchelf".to_string(),
        ]);
        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Build vLLM FROM SOURCE against the env's installed torch.
    ///
    /// The prebuilt ROCm wheel is compiled against a specific torch (e.g.
    /// 2.11.0) and ABI-breaks against the env's torch (e.g. 2.12.1+rocm7.2) —
    /// `vllm._C.abi3.so: undefined symbol: _ZNR5torch7Library4_defE…`. Building
    /// the sdist with `--no-build-isolation` compiles the C/HIP extensions
    /// against the ACTUAL installed torch, so the symbols match. `--no-deps`
    /// guarantees pip NEVER swaps the env's torch (vLLM pins torch==<other>).
    /// Heavy: compiles C++/HIP (tens of minutes). VLLM_TARGET_DEVICE=rocm picks
    /// the ROCm backend. Non-negotiable: the env's torch is the only torch.
    pub fn build_source_install_command(&self, version: &str) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;
        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        // `--no-binary vllm` = build from sdist (NOT the prebuilt CUDA wheel);
        // the pinned `vllm==<ver>` is the requirement pip installs.
        let req = format!("vllm=={version}");
        args.extend([
            "--no-cache-dir".to_string(),
            "--no-deps".to_string(),
            "--no-build-isolation".to_string(),
            "--no-binary".to_string(),
            "vllm".to_string(),
            req,
        ]);
        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![
                // ROCm build backend — the REAL ROCm selector (independent of the
                // version label). Guarantees C/HIP compilation, never CUDA.
                ("VLLM_TARGET_DEVICE".to_string(), "rocm".to_string()),
                ("VLLM_USE_ROCM".to_string(), "1".to_string()),
                ("USE_ROCM".to_string(), "1".to_string()),
                // Suppress vLLM's `+rocmNNN` build label. setup.py appends
                // +rocmNNN (from the local ROCm version) which mismatches the
                // PyPI sdist filename (X.Y.Z) → pip cascades through every
                // release downloading 36MB each. VLLM_VERSION_OVERRIDE makes
                // get_vllm_version() return the exact version EARLY (before the
                // append), so metadata == filename → no cascade. The version is
                // pinned (VLLM_SOURCE_VERSION) so the override matches the sdist
                // pip fetches. ROCm is still guaranteed by VLLM_TARGET_DEVICE.
                ("VLLM_VERSION_OVERRIDE".to_string(), version.to_string()),
            ],
        }
    }

    /// Resolve the vLLM version to build, applying the user's supply-chain gate.
    ///
    /// Queries PyPI's release list (versions + upload dates), drops pre-releases,
    /// then picks the newest release satisfying BOTH: (a) not within the
    /// `version_lag` newest (lag=1 = skip the latest), AND (b) at least
    /// `min_age_days` old (0 = no age requirement). Falls back to
    /// `VLLM_SOURCE_VERSION` if PyPI is unreachable or no release qualifies.
    pub fn resolve_version_with_policy(&self, version_lag: u32, min_age_days: u32) -> String {
        let out = std::process::Command::new(&self.config.python_bin)
            .args([
                "-c",
                VLLM_VERSION_POLICY_PY,
                &version_lag.to_string(),
                &min_age_days.to_string(),
            ])
            .output();
        if let Ok(o) = out {
            if o.status.success() {
                let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if v.chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                {
                    return v;
                }
            }
        }
        VLLM_SOURCE_VERSION.to_string()
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
    fn test_deps_install_command() {
        let installer = VllmInstaller::new(VllmConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_deps_install_command();
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"transformers>=5.5.3".to_string()));
        assert!(cmd.args.contains(&"einops".to_string()));
        assert!(cmd.args.contains(&"amd-quark>=0.8.99".to_string()));
        assert!(cmd.args.contains(&"evaluate".to_string()));
        assert!(cmd.args.contains(&"colorama".to_string()));
        assert!(cmd.args.contains(&"onnx-ir".to_string()));
        assert!(cmd.args.contains(&"onnxscript".to_string()));
        assert!(cmd.args.contains(&"boto3".to_string()));
        assert!(cmd.args.contains(&"httpx-sse".to_string()));
        assert!(cmd
            .args
            .contains(&"opentelemetry-semantic-conventions>=0.59b0".to_string()));
        assert!(cmd.args.contains(&"protobuf>=6.33.5,<7.0.0".to_string()));
        assert!(cmd
            .args
            .contains(&"model-hosting-container-standards>=0.1.14,<1.0.0".to_string()));
        assert!(cmd.args.contains(&"uvloop".to_string()));
        assert!(!cmd.args.iter().any(|arg| arg == "torch"));
        assert!(!cmd.args.iter().any(|arg| arg == "triton"));
        assert!(!cmd.args.iter().any(|arg| arg == "triton-kernels"));
        assert!(!cmd
            .args
            .iter()
            .any(|arg| arg.to_ascii_lowercase().contains("nvidia")));
        assert!(!cmd
            .args
            .iter()
            .any(|arg| arg.to_ascii_lowercase().contains("cuda")));
    }

    #[test]
    fn test_versioned_deps_command() {
        let installer = VllmInstaller::with_defaults();
        let cmd = installer.build_versioned_deps_command();
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.starts_with("conch-triton-kernels==")));
        assert!(cmd.args.iter().any(|a| a.starts_with("xgrammar>=")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.starts_with("outlines-core==0.2.14")));
        assert!(cmd.args.iter().any(|a| a.starts_with("llguidance>=1.7.0")));
        assert!(cmd.args.iter().any(|a| a.starts_with("llvmlite>=0.47.0")));
        assert!(cmd.args.iter().any(|a| a.starts_with("grpcio==1.78.0")));
        assert!(cmd.args.contains(&"torch-c-dlpack-ext".to_string()));
        assert!(cmd.args.contains(&"z3-solver>=4.13.0,<4.15.5".to_string()));
        assert!(!cmd.args.iter().any(|arg| arg == "torch"));
        assert!(!cmd.args.iter().any(|arg| arg == "triton"));
        assert!(!cmd.args.iter().any(|arg| arg == "triton-kernels"));
        assert!(!cmd
            .args
            .iter()
            .any(|arg| arg.to_ascii_lowercase().contains("nvidia")));
        assert!(!cmd
            .args
            .iter()
            .any(|arg| arg.to_ascii_lowercase().contains("cuda")));
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
