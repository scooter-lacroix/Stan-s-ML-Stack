//! ONNX Runtime installer — ports `scripts/build_onnxruntime_multi.sh`.
//!
//! Constructs correct cmake commands with ROCm integration and Python bindings,
//! builds from source using ONNX Runtime's build.sh, and pip installs from
//! the built wheel. Falls back to prebuilt wheel if available.
//!
//! ONNX Runtime depends on ROCm.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-010**: ONNX Runtime correct cmake command
//! - **VAL-INSTALL-011**: ONNX Runtime pip install from build output
//! - **VAL-INSTALL-014**: Source builds parse build output for errors
//! - **VAL-INSTALL-045**: ONNX Runtime declares dependency on ROCm

use crate::installers::common::RocmEnv;
use std::path::PathBuf;

// ===========================================================================
// Types
// ===========================================================================

/// GPU architecture specification for HIP builds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HipArchs {
    /// gfx1030 family.
    Gfx1030,
    /// gfx1100 family (gfx1100-gfx1103).
    Gfx1100,
    /// gfx1200 family (gfx1200-gfx1201).
    Gfx1200,
    /// Default fallback.
    Default,
}

impl HipArchs {
    /// Detect HIP architectures from a GPU arch string.
    pub fn from_gpu_arch(gpu_arch: &str) -> Self {
        if gpu_arch.starts_with("gfx103") {
            HipArchs::Gfx1030
        } else if gpu_arch.starts_with("gfx110") {
            HipArchs::Gfx1100
        } else if gpu_arch.starts_with("gfx120") {
            HipArchs::Gfx1200
        } else {
            HipArchs::Default
        }
    }

    /// Get the CMAKE_HIP_ARCHITECTURES string.
    pub fn cmake_hip_architectures(&self) -> &str {
        match self {
            HipArchs::Gfx1030 => "gfx1030",
            HipArchs::Gfx1100 => "gfx1100;gfx1101;gfx1102;gfx1103",
            HipArchs::Gfx1200 => "gfx1200;gfx1201",
            HipArchs::Default => "gfx1100",
        }
    }
}

/// ONNX Runtime install method.
///
/// **Option ordering:** [`MigraphxWheel`] is the primary/default (PyPI
/// `onnxruntime-migraphx`, current release); [`PrebuiltWheel`] is the legacy
/// fallback; [`SourceBuild`] is the last resort (custom/when no wheel exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnnxInstallMethod {
    /// **Option 1 (primary, default):** install the pinned
    /// `onnxruntime-migraphx` wheel from PyPI ([`PREBUILT_MIGRAPHX_VERSION`],
    /// currently 1.27.1). PyPI's Microsoft-published wheel ships the
    /// MIGraphXExecutionProvider built against ROCm and tracks the current
    /// release. The AMD manylinux repo (`repo.radeon.com`) is NOT used — it lags
    /// far behind (1.23.2 for ROCm 7.2.4) and fabricates 404 URLs for newer
    /// versions.
    #[default]
    MigraphxWheel,
    /// **Option 3 (last resort):** build from source with ROCm + MIGraphX EP
    /// (includes ROCMExecutionProvider), targeting [`DEFAULT_ONNXRUNTIME_VERSION`].
    /// Use only when the PyPI wheel is unavailable or a custom build is required.
    SourceBuild,
    /// **Option 2 (legacy):** install prebuilt `onnxruntime-rocm` from PyPI.
    /// May be ABI-incompatible with the installed ROCm; prefer [`MigraphxWheel`].
    PrebuiltWheel,
}

/// Current ROCm-compatible ONNX Runtime release: the source-build target and
/// the version the bundled manifest proposes.
pub const DEFAULT_ONNXRUNTIME_VERSION: &str = "1.27.1";

/// Pinned prebuilt `onnxruntime-migraphx` wheel version installed from PyPI by
/// the default ([`OnnxInstallMethod::MigraphxWheel`]) path — the primary,
/// production-proven ROCm build that ships the MIGraphXExecutionProvider.
///
/// Tracks the current PyPI release line (1.27.1) and is kept in lock-step with
/// [`DEFAULT_ONNXRUNTIME_VERSION`] and the manifest's `onnx` target version
/// (`baseline_manifest.json`). That lock-step is load-bearing: the default
/// MIGraphX-wheel install path must land on exactly the version the post-install
/// honesty guard verifies against — otherwise an explicit
/// `rusty-stack update onnx` would install this pinned wheel but be reported as
/// failed because the detected version mismatches the manifest target.
///
/// The Rust `ort` crate's MIGraphX execution-provider builder exposes no
/// `with_model_cache_dir` / model-cache API in ANY released version (newest is
/// `ort 2.0.0-rc.12`), and the v2.0 "arbitrarily configurable" EP change left
/// MIGraphX out — so the version choice is driven by manifest/honesty-guard
/// alignment rather than an `ort` API benefit. The source-build path
/// ([`OnnxInstallMethod::SourceBuild`]) targets
/// [`DEFAULT_ONNXRUNTIME_VERSION`] when a custom build is actually required.
/// The AMD manylinux repo (`repo.radeon.com/rocm-rel-<release>`) is NOT used —
/// it lags far behind (still 1.23.2 for ROCm 7.2.4) and fabricates 404 URLs for
/// newer versions. Bump only when a newer wheel is verified to add real value
/// (and bump all three in lock-step: this constant,
/// [`DEFAULT_ONNXRUNTIME_VERSION`], and the manifest target).
pub const PREBUILT_MIGRAPHX_VERSION: &str = "1.27.1";

/// Env var selecting the ONNX install method
/// (`migraphx`/`source`/`prebuilt`). Default: [`OnnxInstallMethod::MigraphxWheel`].
pub const ONNX_INSTALL_METHOD_ENV: &str = "MLSTACK_ONNX_INSTALL_METHOD";

/// Env var overriding the prebuilt `onnxruntime-migraphx` version (else the
/// pinned [`PREBUILT_MIGRAPHX_VERSION`] is used).
pub const ONNX_VERSION_ENV: &str = "MLSTACK_ONNX_VERSION";

/// Resolve the ONNX install method from [`ONNX_INSTALL_METHOD_ENV`].
pub fn install_method_from_env() -> OnnxInstallMethod {
    match std::env::var(ONNX_INSTALL_METHOD_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "source" | "source-build" | "src" => OnnxInstallMethod::SourceBuild,
        "prebuilt" | "rocm" | "onnxruntime-rocm" => OnnxInstallMethod::PrebuiltWheel,
        _ => OnnxInstallMethod::MigraphxWheel,
    }
}

/// Resolve an overridden prebuilt version from [`ONNX_VERSION_ENV`], else `None`.
pub fn prebuilt_version_from_env() -> Option<String> {
    std::env::var(ONNX_VERSION_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Configuration for the ONNX Runtime installer.
#[derive(Debug, Clone)]
pub struct OnnxRuntimeConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// ROCm version string (e.g., "7.2.0").
    pub rocm_version: Option<String>,
    /// ROCm release for AMD repo URL (e.g., "7.2.4").
    pub rocm_release: Option<String>,
    /// GPU architecture string (e.g., "gfx1100").
    pub gpu_arch: Option<String>,
    /// Working directory for the build (defaults to /tmp/onnxruntime-rocm).
    pub workdir: Option<PathBuf>,
    /// Preinstalled Eigen path.
    pub eigen_path: Option<PathBuf>,
    /// Whether to use preinstalled Eigen.
    pub use_preinstalled_eigen: bool,
    /// Install method (default: MIGraphX wheel from PyPI).
    pub install_method: OnnxInstallMethod,
    /// ONNX Runtime wheel version. `None` uses the current bundled default.
    pub runtime_version: Option<String>,
    /// Pinned prebuilt `onnxruntime-migraphx` version for the default path.
    /// `None` uses [`PREBUILT_MIGRAPHX_VERSION`].
    pub prebuilt_version: Option<String>,
}

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            dry_run: false,
            rocm_version: None,
            rocm_release: None,
            gpu_arch: None,
            workdir: None,
            eigen_path: None,
            use_preinstalled_eigen: false,
            install_method: OnnxInstallMethod::default(),
            runtime_version: None,
            prebuilt_version: None,
        }
    }
}

impl OnnxRuntimeConfig {
    /// Get the working directory for the build.
    pub fn workdir(&self) -> PathBuf {
        self.workdir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("onnxruntime-rocm"))
    }

    /// Get the effective GPU arch string.
    pub fn gpu_arch(&self) -> &str {
        self.gpu_arch.as_deref().unwrap_or("gfx1100")
    }

    /// Get the effective ROCm version string.
    pub fn rocm_version(&self) -> &str {
        self.rocm_version.as_deref().unwrap_or("7.2")
    }

    /// Get the effective ROCm release string (for AMD repo URL).
    pub fn rocm_release(&self) -> &str {
        self.rocm_release.as_deref().unwrap_or("7.2.4")
    }

    /// Get the effective ONNX Runtime wheel version.
    ///
    /// Filters an empty/whitespace `Some("")` to the default: a stale or
    /// untrimmed `ctx.target_version` would otherwise produce an invalid
    /// `git checkout v` (no tag suffix) on the source-build path. Mirrors
    /// [`prebuilt_version_from_env`]'s empty-filtering.
    pub fn runtime_version(&self) -> &str {
        match self.runtime_version.as_deref() {
            Some(v) if !v.trim().is_empty() => v,
            _ => DEFAULT_ONNXRUNTIME_VERSION,
        }
    }

    /// Get the pinned prebuilt `onnxruntime-migraphx` version for the default
    /// PyPI install path.
    pub fn prebuilt_version(&self) -> &str {
        self.prebuilt_version
            .as_deref()
            .unwrap_or(PREBUILT_MIGRAPHX_VERSION)
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
    /// Working directory for the command.
    pub working_dir: Option<PathBuf>,
}

/// The ONNX Runtime installer.
pub struct OnnxRuntimeInstaller {
    config: OnnxRuntimeConfig,
}

impl OnnxRuntimeInstaller {
    /// Create a new ONNX Runtime installer with the given config.
    pub fn new(config: OnnxRuntimeConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(OnnxRuntimeConfig::default())
    }

    /// Get the requested/effective runtime version for status reporting.
    pub fn runtime_version(&self) -> &str {
        self.config.runtime_version()
    }

    /// Get the pinned prebuilt `onnxruntime-migraphx` version installed by the
    /// default path.
    pub fn prebuilt_version(&self) -> &str {
        self.config.prebuilt_version()
    }

    // -----------------------------------------------------------------------
    // Dependencies (VAL-INSTALL-045)
    // -----------------------------------------------------------------------

    /// Get the list of required dependencies.
    ///
    /// ONNX Runtime depends on ROCm.
    pub fn dependencies(&self) -> &[&str] {
        &["rocm"]
    }

    // -----------------------------------------------------------------------
    // Prebuilt wheel install command
    // -----------------------------------------------------------------------

    /// Construct the pip install command for prebuilt onnxruntime-rocm wheel.
    ///
    /// The original script tries this first:
    /// `pip install --upgrade --prefer-binary onnxruntime-rocm`
    pub fn build_prebuilt_install_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--upgrade".to_string(),
                "--force-reinstall".to_string(),
                "--no-deps".to_string(),
                "--prefer-binary".to_string(),
                "onnxruntime-rocm".to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // MIGraphX wheel install from PyPI (default)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for the pinned `onnxruntime-migraphx`
    /// wheel from PyPI.
    ///
    /// This is the default install method. The Microsoft-published PyPI wheel
    /// ships the MIGraphX execution provider built against ROCm. `--no-deps`
    /// honors the No-CUDA hard-prime (prevents pulling nvidia/cuda runtime
    /// transitive deps); `--no-cache-dir` forces a fresh fetch.
    pub fn build_migraphx_install_command(&self) -> ShellCommand {
        let spec = format!("onnxruntime-migraphx=={}", self.config.prebuilt_version());
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--upgrade".to_string(),
                "--force-reinstall".to_string(),
                "--no-deps".to_string(),
                "--no-cache-dir".to_string(),
                spec,
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Construct a provider validation command for AMD ONNX Runtime.
    pub fn build_provider_validation_command(&self) -> ShellCommand {
        let script = r#"
import ctypes
import os
import onnxruntime as ort

if not getattr(ort, "__version__", None) or not hasattr(ort, "get_available_providers"):
    raise SystemExit(f"ONNX Runtime import incomplete: {getattr(ort, '__file__', '<unknown>')}")

priority = ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]
available = ort.get_available_providers()
selected = next((p for p in priority if p in available), None)
if selected is None:
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    loader_errors = []
    for lib in ("libonnxruntime_providers_migraphx.so", "libonnxruntime_providers_rocm.so"):
        path = os.path.join(capi, lib)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path)
            except OSError as exc:
                loader_errors.append(f"{lib}: {exc}")
    raise SystemExit(
        "ONNX Runtime AMD provider unavailable; expected MIGraphXExecutionProvider "
        f"or legacy ROCMExecutionProvider; available={available}; "
        f"loader_errors={loader_errors or 'none'}"
    )
print(f"ONNX Runtime AMD provider ready: {selected}; available={available}")
"#;

        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), script.trim().to_string()],
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // Model optimizer (ORT graph optimization for quantized models)
    // -----------------------------------------------------------------------

    /// Construct the Python command to run ORT graph optimization on an ONNX model.
    ///
    /// Applies `ORT_ENABLE_ALL` optimization level which fuses quantized ops
    /// (DynamicQuantizeLinear, MatMulInteger, etc.) into custom ops that bypass
    /// MIGraphX's broken kernels. The optimized model is saved alongside the original.
    pub fn build_model_optimizer_command(&self, model_path: &str) -> ShellCommand {
        let optimized_path = format!("{}.optimized", model_path);

        let script = format!(
            "import onnxruntime as ort; \
             providers = [p for p in ['MIGraphXExecutionProvider', 'ROCMExecutionProvider'] if p in ort.get_available_providers()]; \
             assert providers, f'No AMD ONNX Runtime provider available: {{ort.get_available_providers()}}'; \
             opts = ort.SessionOptions(); \
             opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; \
             opts.optimized_model_filepath = '{optimized_path}'; \
             ort.InferenceSession('{model_path}', opts, providers=providers); \
             print('Optimized: {model_path} -> {optimized_path} using ' + providers[0])"
        );

        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), script],
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // Git clone command
    // -----------------------------------------------------------------------

    /// Construct the git clone command for ONNX Runtime.
    ///
    /// The original script clones:
    /// `git clone --recursive https://github.com/microsoft/onnxruntime.git`
    pub fn build_git_clone_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--recursive".to_string(),
                "https://github.com/microsoft/onnxruntime.git".to_string(),
            ],
            env: vec![],
            working_dir: Some(self.config.workdir()),
        }
    }

    /// Construct the git checkout command for the release tag matching the
    /// configured runtime version (e.g. `v1.27.1`).
    pub fn build_git_checkout_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "checkout".to_string(),
                format!("v{}", self.config.runtime_version()),
            ],
            env: vec![],
            working_dir: Some(self.config.workdir().join("onnxruntime")),
        }
    }

    /// Construct the git submodule update command.
    pub fn build_git_submodule_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "submodule".to_string(),
                "update".to_string(),
                "--init".to_string(),
                "--recursive".to_string(),
            ],
            env: vec![],
            working_dir: Some(self.config.workdir().join("onnxruntime")),
        }
    }

    // -----------------------------------------------------------------------
    // Build command (VAL-INSTALL-010)
    // -----------------------------------------------------------------------

    /// Construct the ONNX Runtime build.sh command with ROCm flags.
    ///
    /// The original script runs `./build.sh` with many flags:
    /// - `--config Release`
    /// - `--build_wheel`
    /// - `--parallel $(nproc)-1`
    /// - `--use_rocm --rocm_home /opt/rocm`
    /// - `--rocm_version 70200`
    /// - `--use_migraphx --migraphx_home /opt/rocm`
    /// - `--cmake_extra_defines CMAKE_HIP_ARCHITECTURES=...`
    /// - Various CMAKE isolation defines
    /// - `--allow_running_as_root`
    pub fn build_build_command(&self, rocm_env: &RocmEnv) -> ShellCommand {
        let rocm_path = rocm_env
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/opt/rocm".to_string());

        let hip_archs = HipArchs::from_gpu_arch(self.config.gpu_arch());
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(3);

        // ONNX Runtime v1.23+ dropped `--use_rocm`/`--rocm_home`/`--rocm_version`
        // (verified against v1.27.1's build.py — zero `use_rocm` references).
        // The MIGraphX EP is built with `--use_migraphx --migraphx_home`; CMake
        // locates ROCm via CMAKE_PREFIX_PATH (below) and targets the GPU arch
        // via CMAKE_HIP_ARCHITECTURES. Passing the removed flags would make
        // argparse reject the build outright.
        let mut args = vec![
            "--config".to_string(),
            "Release".to_string(),
            "--build_wheel".to_string(),
            "--parallel".to_string(),
            nproc.to_string(),
            "--skip_tests".to_string(),
            "--use_migraphx".to_string(),
            "--migraphx_home".to_string(),
            rocm_path.clone(),
        ];

        if self.config.use_preinstalled_eigen {
            if let Some(ref eigen_path) = self.config.eigen_path {
                args.push("--use_preinstalled_eigen".to_string());
                args.push("--eigen_path".to_string());
                args.push(eigen_path.to_string_lossy().to_string());
            }
        }

        // CMAKE extra defines
        let cmake_extra_defines = vec![
            format!(
                "CMAKE_HIP_ARCHITECTURES={}",
                hip_archs.cmake_hip_architectures()
            ),
            "CMAKE_CXX_STANDARD=20".to_string(),
            "onnxruntime_USE_EXTERNAL_ABSEIL=OFF".to_string(),
            "CMAKE_DISABLE_FIND_PACKAGE_re2=ON".to_string(),
            "CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON".to_string(),
            "CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON".to_string(),
            "CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF".to_string(),
            "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF".to_string(),
            format!("CMAKE_PREFIX_PATH={}", rocm_path),
            "re2_DIR=RE2_DIR-NOTFOUND".to_string(),
            "CMAKE_POLICY_VERSION_MINIMUM=3.5".to_string(),
        ];

        for define in cmake_extra_defines {
            args.push("--cmake_extra_defines".to_string());
            args.push(define);
        }

        if self.config.use_preinstalled_eigen {
            if let Some(ref eigen_path) = self.config.eigen_path {
                args.push("--cmake_extra_defines".to_string());
                args.push(format!(
                    "FETCHCONTENT_SOURCE_DIR_EIGEN={}",
                    eigen_path.to_string_lossy()
                ));
                args.push("--cmake_extra_defines".to_string());
                args.push("FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER".to_string());
            }
        }

        args.push("--allow_running_as_root".to_string());

        let env = vec![];

        ShellCommand {
            program: "./build.sh".to_string(),
            args,
            env,
            working_dir: Some(self.config.workdir().join("onnxruntime")),
        }
    }

    // -----------------------------------------------------------------------
    // Pip install from build output (VAL-INSTALL-011)
    // -----------------------------------------------------------------------

    /// Construct the pip uninstall command to remove all onnxruntime variants.
    pub fn build_uninstall_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "uninstall".to_string(),
                "-y".to_string(),
                "onnxruntime".to_string(),
                "onnxruntime-rocm".to_string(),
                "onnxruntime-gpu".to_string(),
                "onnxruntime-migraphx".to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Construct the pip install command for the built wheel.
    ///
    /// The original script installs from:
    /// `build/Linux/Release/dist/*.whl`
    pub fn build_wheel_install_command(&self) -> ShellCommand {
        let wheel_path = self
            .config
            .workdir()
            .join("onnxruntime")
            .join("build/Linux/Release/dist/*.whl");

        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                wheel_path.to_string_lossy().to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // Build error detection (VAL-INSTALL-014)
    // -----------------------------------------------------------------------

    /// Check build output for errors.
    ///
    /// Detects common build failure patterns and propagates as an error.
    pub fn check_build_output(
        &self,
        stdout: &str,
        stderr: &str,
        exit_code: i32,
    ) -> Result<(), String> {
        if exit_code != 0 {
            let error_patterns = [
                "CMake Error",
                "error:",
                "FAILED:",
                "fatal error:",
                "Build failed",
                "ninja: build stopped",
                "Could NOT find",
            ];

            let combined = format!("{}\n{}", stdout, stderr);
            let detected_errors: Vec<&str> = error_patterns
                .iter()
                .filter(|p| combined.contains(*p))
                .copied()
                .collect();

            if detected_errors.is_empty() {
                return Err(format!(
                    "ONNX Runtime build failed with exit code {} (no specific error pattern detected)",
                    exit_code
                ));
            }

            return Err(format!(
                "ONNX Runtime build failed with exit code {}. Detected errors: {}",
                exit_code,
                detected_errors.join(", ")
            ));
        }

        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // --- VAL-INSTALL-010: ONNX Runtime correct cmake command ---

    #[test]
    fn test_build_command_uses_migraphx_recipe_not_use_rocm() {
        // ONNX Runtime v1.23+ dropped --use_rocm/--rocm_home/--rocm_version
        // (verified: v1.27.1 build.py has zero `use_rocm`). The source build
        // uses --use_migraphx --migraphx_home + cmake defines (CMAKE_PREFIX_PATH
        // + CMAKE_HIP_ARCHITECTURES).
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            rocm_version: Some("7.2.0".to_string()),
            gpu_arch: Some("gfx1100".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert_eq!(cmd.program, "./build.sh");
        assert!(cmd.args.contains(&"--use_migraphx".to_string()));
        assert!(cmd.args.contains(&"--migraphx_home".to_string()));
        assert!(cmd.args.contains(&"--build_wheel".to_string()));
        assert!(cmd.args.contains(&"--skip_tests".to_string()));
        assert!(cmd.args.contains(&"--allow_running_as_root".to_string()));
        assert!(cmd.args.iter().any(|a| a == "/opt/rocm"));
        // Removed flags must NOT appear (argparse would reject them outright).
        for removed in ["--use_rocm", "--rocm_home", "--rocm_version"] {
            assert!(
                !cmd.args.contains(&removed.to_string()),
                "removed flag {removed} must not appear: {:?}",
                cmd.args
            );
        }
    }

    #[test]
    fn test_build_command_has_hip_architectures() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            gpu_arch: Some("gfx1100".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101;gfx1102;gfx1103")));
    }

    #[test]
    fn test_build_command_has_cmake_extra_defines() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("onnxruntime_USE_EXTERNAL_ABSEIL=OFF")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("CMAKE_DISABLE_FIND_PACKAGE_re2=ON")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("CMAKE_POLICY_VERSION_MINIMUM=3.5")));
    }

    // --- C++20 standard fix (fix-onnx-cmake-cxx20) ---

    #[test]
    fn test_build_command_includes_cmake_cxx_standard_20() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        // The cmake command must include CMAKE_CXX_STANDARD=20 to fix
        // C++20 standard library test failures during cmake configuration.
        assert!(
            cmd.args.iter().any(|a| a.contains("CMAKE_CXX_STANDARD=20")),
            "build command must include CMAKE_CXX_STANDARD=20 cmake extra define, got args: {:?}",
            cmd.args
        );
    }

    #[test]
    fn test_build_command_cmake_cxx_standard_is_cmake_extra_define() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        // Find the CMAKE_CXX_STANDARD=20 arg and verify it's preceded by --cmake_extra_defines
        let define_idx = cmd
            .args
            .iter()
            .position(|a| a.contains("CMAKE_CXX_STANDARD=20"));
        assert!(
            define_idx.is_some(),
            "CMAKE_CXX_STANDARD=20 not found in args"
        );

        let idx = define_idx.unwrap();
        assert!(
            idx > 0 && cmd.args[idx - 1] == "--cmake_extra_defines",
            "CMAKE_CXX_STANDARD=20 must be preceded by --cmake_extra_defines"
        );
    }

    #[test]
    fn test_build_command_does_not_override_env_vars() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(
            cmd.env.is_empty(),
            "ONNX build must inherit ~/.mlstack_env, got env: {:?}",
            cmd.env
        );
    }

    #[test]
    fn test_build_command_has_parallel_flag() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd.args.contains(&"--parallel".to_string()));
    }

    #[test]
    fn test_build_command_env_vars() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd.env.is_empty());
    }

    #[test]
    fn test_build_command_with_eigen() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            use_preinstalled_eigen: true,
            eigen_path: Some(PathBuf::from("/usr/include/eigen3")),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd.args.contains(&"--use_preinstalled_eigen".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("/usr/include/eigen3")));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("FETCHCONTENT_SOURCE_DIR_EIGEN")));
    }

    // --- VAL-INSTALL-011: ONNX Runtime pip install from build output ---

    #[test]
    fn test_wheel_install_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_wheel_install_command();

        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd
            .args
            .iter()
            .any(|a| a.contains("build/Linux/Release/dist/*.whl")));
    }

    #[test]
    fn test_uninstall_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_uninstall_command();

        assert!(cmd.args.contains(&"uninstall".to_string()));
        assert!(cmd.args.contains(&"-y".to_string()));
        assert!(cmd.args.contains(&"onnxruntime".to_string()));
        assert!(cmd.args.contains(&"onnxruntime-rocm".to_string()));
        assert!(cmd.args.contains(&"onnxruntime-gpu".to_string()));
        assert!(cmd.args.contains(&"onnxruntime-migraphx".to_string()));
    }

    // --- Prebuilt wheel install ---

    #[test]
    fn test_prebuilt_install_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_prebuilt_install_command();

        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"--upgrade".to_string()));
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"--prefer-binary".to_string()));
        assert!(cmd.args.contains(&"onnxruntime-rocm".to_string()));
        assert!(cmd.env.is_empty());
    }

    // --- Git commands ---

    #[test]
    fn test_git_clone_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&"--recursive".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("microsoft/onnxruntime")));
    }

    #[test]
    fn test_git_checkout_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_git_checkout_command();
        // Checks out the tag matching the configured runtime version (1.27.1),
        // not a stale hardcoded tag.
        assert!(cmd
            .args
            .contains(&format!("v{}", DEFAULT_ONNXRUNTIME_VERSION)));
        assert!(cmd.args.contains(&"v1.27.1".to_string()));
    }

    #[test]
    fn test_git_checkout_command_follows_runtime_version() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            runtime_version: Some("1.25.0".to_string()),
            ..Default::default()
        });
        let cmd = installer.build_git_checkout_command();
        assert!(cmd.args.contains(&"v1.25.0".to_string()));
    }

    #[test]
    fn test_runtime_version_filters_empty_to_default() {
        // An untrimmed/stale ctx.target_version of Some("") must NOT fall through
        // as the tag — it would produce an invalid `git checkout v` (no suffix).
        // Empty/whitespace values resolve to DEFAULT_ONNXRUNTIME_VERSION.
        let empty = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            runtime_version: Some(String::new()),
            ..Default::default()
        });
        assert_eq!(empty.config.runtime_version(), DEFAULT_ONNXRUNTIME_VERSION);
        assert_ne!(empty.config.runtime_version(), "");
        let ws = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            runtime_version: Some("   ".to_string()),
            ..Default::default()
        });
        assert_eq!(ws.config.runtime_version(), DEFAULT_ONNXRUNTIME_VERSION);
        // And a real value is preserved.
        let real = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            runtime_version: Some("1.25.0".to_string()),
            ..Default::default()
        });
        assert_eq!(real.config.runtime_version(), "1.25.0");
    }

    // --- HipArchs ---

    #[test]
    fn test_hip_archs_gfx1030() {
        let archs = HipArchs::from_gpu_arch("gfx1030");
        assert_eq!(archs.cmake_hip_architectures(), "gfx1030");
    }

    #[test]
    fn test_hip_archs_gfx1100() {
        let archs = HipArchs::from_gpu_arch("gfx1100");
        assert_eq!(
            archs.cmake_hip_architectures(),
            "gfx1100;gfx1101;gfx1102;gfx1103"
        );
    }

    #[test]
    fn test_hip_archs_gfx1200() {
        let archs = HipArchs::from_gpu_arch("gfx1200");
        assert_eq!(archs.cmake_hip_architectures(), "gfx1200;gfx1201");
    }

    #[test]
    fn test_hip_archs_default() {
        let archs = HipArchs::from_gpu_arch("gfx900");
        assert_eq!(archs.cmake_hip_architectures(), "gfx1100");
    }

    // --- VAL-INSTALL-045: ONNX Runtime declares dependency on ROCm ---

    #[test]
    fn test_dependencies() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let deps = installer.dependencies();
        assert!(deps.contains(&"rocm"));
    }

    // --- VAL-INSTALL-014: Build error detection ---

    #[test]
    fn test_build_error_cmake() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let result = installer.check_build_output("", "CMake Error at cmake/CMakeLists.txt:42", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CMake Error"));
    }

    #[test]
    fn test_build_error_ninja() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let result =
            installer.check_build_output("ninja: build stopped: subcommand failed.", "", 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ninja: build stopped"));
    }

    #[test]
    fn test_build_error_success() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let result = installer.check_build_output("Build succeeded", "", 0);
        assert!(result.is_ok());
    }

    // --- Config defaults ---

    #[test]
    fn test_config_defaults() {
        let config = OnnxRuntimeConfig::default();
        assert_eq!(config.python_bin, "python3");
        assert!(!config.dry_run);
        assert!(config.rocm_version.is_none());
        assert!(config.rocm_release.is_none());
        assert!(config.gpu_arch.is_none());
        assert!(config.workdir.is_none());
        assert!(!config.use_preinstalled_eigen);
        assert!(config.eigen_path.is_none());
        assert_eq!(config.install_method, OnnxInstallMethod::MigraphxWheel);
        assert_eq!(config.runtime_version(), DEFAULT_ONNXRUNTIME_VERSION);
        assert_eq!(config.prebuilt_version(), PREBUILT_MIGRAPHX_VERSION);
    }

    // --- MIGraphX wheel install (PyPI prebuilt, default path) ---

    #[test]
    fn test_prebuilt_version_defaults_to_pinned() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        assert_eq!(installer.prebuilt_version(), PREBUILT_MIGRAPHX_VERSION);
        // Lock-step invariant: the default wheel version matches the manifest
        // target + DEFAULT_ONNXRUNTIME_VERSION so the honesty guard verifies the
        // version the default path actually installs.
        assert_eq!(installer.prebuilt_version(), DEFAULT_ONNXRUNTIME_VERSION);
        assert_eq!(installer.prebuilt_version(), "1.27.1");
    }

    #[test]
    fn test_install_method_from_env() {
        let _env = crate::test_support::lock_env();
        std::env::remove_var(ONNX_INSTALL_METHOD_ENV);
        assert_eq!(install_method_from_env(), OnnxInstallMethod::MigraphxWheel);
        for (v, expected) in [
            ("source", OnnxInstallMethod::SourceBuild),
            ("SOURCE-BUILD", OnnxInstallMethod::SourceBuild),
            ("prebuilt", OnnxInstallMethod::PrebuiltWheel),
            ("rocm", OnnxInstallMethod::PrebuiltWheel),
            ("migraphx", OnnxInstallMethod::MigraphxWheel),
            ("", OnnxInstallMethod::MigraphxWheel),
        ] {
            if v.is_empty() {
                std::env::remove_var(ONNX_INSTALL_METHOD_ENV);
            } else {
                std::env::set_var(ONNX_INSTALL_METHOD_ENV, v);
            }
            assert_eq!(install_method_from_env(), expected, "for {v:?}");
        }
    }

    #[test]
    fn test_prebuilt_version_from_env_override() {
        let _env = crate::test_support::lock_env();
        std::env::remove_var(ONNX_VERSION_ENV);
        assert_eq!(prebuilt_version_from_env(), None);
        std::env::set_var(ONNX_VERSION_ENV, "  1.30.0  ");
        assert_eq!(prebuilt_version_from_env(), Some("1.30.0".to_string()));
        std::env::set_var(ONNX_VERSION_ENV, "   ");
        assert_eq!(prebuilt_version_from_env(), None);
    }

    #[test]
    fn test_prebuilt_version_overridable() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            prebuilt_version: Some("1.23.2".to_string()),
            ..Default::default()
        });
        assert_eq!(installer.prebuilt_version(), "1.23.2");
    }

    #[test]
    fn test_migraphx_install_command_pypi() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_migraphx_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"--upgrade".to_string()));
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
        // --no-deps honors the No-CUDA hard-prime (no nvidia/cuda transitive deps)
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"--no-cache-dir".to_string()));
        // Pins the prebuilt version from PyPI — NOT an AMD repo URL.
        assert!(cmd
            .args
            .iter()
            .any(|a| a == &format!("onnxruntime-migraphx=={}", PREBUILT_MIGRAPHX_VERSION)));
        assert!(
            !cmd.args.iter().any(|a| a.contains("repo.radeon.com")),
            "must not use the AMD manylinux repo (it lags and 404s): {:?}",
            cmd.args
        );
        assert!(cmd.env.is_empty());
    }

    // --- Provider validation / model optimizer ---

    #[test]
    fn test_provider_validation_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_provider_validation_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-c".to_string()));
        let script = &cmd.args[1];
        assert!(script.contains("MIGraphXExecutionProvider"));
        assert!(script.contains("ROCMExecutionProvider"));
        assert!(script.contains("ctypes.CDLL"));
        assert!(script.contains("loader_errors"));
        assert!(cmd.env.is_empty());
    }

    #[test]
    fn test_model_optimizer_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_model_optimizer_command("/path/to/model.onnx");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-c".to_string()));
        let script = &cmd.args[1];
        assert!(script.contains("MIGraphXExecutionProvider"));
        assert!(script.contains("ROCMExecutionProvider"));
        assert!(!script.contains("CPUExecutionProvider"));
        assert!(script.contains("ORT_ENABLE_ALL"));
        assert!(script.contains("/path/to/model.onnx"));
        assert!(script.contains("/path/to/model.onnx.optimized"));
        assert!(cmd.env.is_empty());
    }

    // --- OnnxInstallMethod ---

    #[test]
    fn test_install_method_default() {
        assert_eq!(
            OnnxInstallMethod::default(),
            OnnxInstallMethod::MigraphxWheel
        );
    }
}
