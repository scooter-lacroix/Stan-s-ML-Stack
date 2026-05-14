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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxInstallMethod {
    /// Install onnxruntime-migraphx from AMD manylinux repo (default).
    MigraphxWheel,
    /// Build from source with ROCm + MIGraphX EP (includes ROCMExecutionProvider).
    SourceBuild,
    /// Install prebuilt onnxruntime-rocm from PyPI (legacy, may be ABI-incompatible).
    PrebuiltWheel,
}

impl Default for OnnxInstallMethod {
    fn default() -> Self {
        OnnxInstallMethod::MigraphxWheel
    }
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
    /// ROCm release for AMD repo URL (e.g., "7.2.3").
    pub rocm_release: Option<String>,
    /// GPU architecture string (e.g., "gfx1100").
    pub gpu_arch: Option<String>,
    /// Working directory for the build (defaults to /tmp/onnxruntime-rocm).
    pub workdir: Option<PathBuf>,
    /// Preinstalled Eigen path.
    pub eigen_path: Option<PathBuf>,
    /// Whether to use preinstalled Eigen.
    pub use_preinstalled_eigen: bool,
    /// Install method (default: MIGraphX wheel from AMD repo).
    pub install_method: OnnxInstallMethod,
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
        self.rocm_release.as_deref().unwrap_or("7.2.3")
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
    // ROCm version formatting
    // -----------------------------------------------------------------------

    /// Format ROCm version for ONNX Runtime (e.g., "7.2.0" -> "70200").
    ///
    /// The original script formats as Mmmpp: `printf "%d%02d%02d" major minor patch`.
    pub fn format_rocm_version_for_ort(&self, rocm_version: &str) -> String {
        let parts: Vec<&str> = rocm_version.split('.').collect();
        let major: u32 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(7);
        let minor: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
        let patch: u32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
        format!("{:02}{:02}{:02}", major, minor, patch)
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
                "--prefer-binary".to_string(),
                "onnxruntime-rocm".to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    // -----------------------------------------------------------------------
    // MIGraphX wheel install from AMD repo
    // -----------------------------------------------------------------------

    /// Construct the AMD manylinux repo URL for onnxruntime-migraphx.
    ///
    /// Pattern: `https://repo.radeon.com/rocm/manylinux/rocm-rel-{release}/`
    ///
    /// The URL points to a specific wheel matching the Python version and ROCm release.
    /// ROCm 7.2.3 ships onnxruntime_migraphx 1.23.2.
    pub fn build_migraphx_wheel_url(&self) -> String {
        let release = self.config.rocm_release();
        let python_bin = &self.config.python_bin;
        // Extract python version from binary (e.g., "python3.12" -> "312")
        let py_ver = if python_bin.contains('.') {
            let parts: Vec<&str> = python_bin.split('.').collect();
            if parts.len() >= 2 {
                format!("{}{}", parts[0], parts[1])
            } else {
                "312".to_string()
            }
        } else {
            "312".to_string()
        };

        format!(
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-{release}/\
             onnxruntime_migraphx-1.23.2-cp{py_ver}-cp{py_ver}-\
             manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        )
    }

    /// Construct the pip install command for onnxruntime-migraphx from AMD repo.
    ///
    /// This is the default install method — the AMD-provided wheel includes the
    /// MIGraphX execution provider and is built against the matching ROCm version,
    /// avoiding ABI incompatibility issues with PyPI's onnxruntime-rocm.
    pub fn build_migraphx_install_command(&self) -> ShellCommand {
        let url = self.build_migraphx_wheel_url();
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--upgrade".to_string(),
                url,
            ],
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
        let optimized_path = if model_path.ends_with(".onnx") {
            format!("{}.optimized", model_path)
        } else {
            format!("{}.optimized", model_path)
        };

        let script = format!(
            "import onnxruntime as ort; \
             opts = ort.SessionOptions(); \
             opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; \
             opts.optimized_model_filepath = '{optimized_path}'; \
             ort.InferenceSession('{model_path}', opts, providers=['CPUExecutionProvider']); \
             print('Optimized: {model_path} -> {optimized_path}')"
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

    /// Construct the git checkout command for the stable tag.
    ///
    /// The original script checks out tag `v1.20.1`.
    pub fn build_git_checkout_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec!["checkout".to_string(), "v1.20.1".to_string()],
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

        let rocm_version = self.config.rocm_version();
        let ort_rocm_version = self.format_rocm_version_for_ort(rocm_version);
        let hip_archs = HipArchs::from_gpu_arch(self.config.gpu_arch());
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(3);

        let mut args = vec![
            "--config".to_string(),
            "Release".to_string(),
            "--build_wheel".to_string(),
            "--parallel".to_string(),
            nproc.to_string(),
            "--use_rocm".to_string(),
            "--rocm_home".to_string(),
            rocm_path.clone(),
            "--rocm_version".to_string(),
            ort_rocm_version,
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

        let env = vec![
            ("ROCM_HOME".to_string(), rocm_path.clone()),
            ("ROCM_PATH".to_string(), rocm_path.clone()),
            ("HIP_PATH".to_string(), rocm_path),
            ("PYTHONPATH".to_string(), String::new()),
            ("CMAKE_PREFIX_PATH".to_string(), "/opt/rocm".to_string()),
            ("CMAKE_CXX_STANDARD".to_string(), "20".to_string()),
            (
                "CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY".to_string(),
                "ON".to_string(),
            ),
            (
                "CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY".to_string(),
                "ON".to_string(),
            ),
            (
                "CMAKE_FIND_USE_PACKAGE_REGISTRY".to_string(),
                "OFF".to_string(),
            ),
            (
                "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY".to_string(),
                "OFF".to_string(),
            ),
        ];

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
    fn test_build_command_has_rocm_flags() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            rocm_version: Some("7.2.0".to_string()),
            gpu_arch: Some("gfx1100".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert_eq!(cmd.program, "./build.sh");
        assert!(cmd.args.contains(&"--use_rocm".to_string()));
        assert!(cmd.args.contains(&"--rocm_home".to_string()));
        assert!(cmd.args.iter().any(|a| a == "/opt/rocm"));
        assert!(cmd.args.contains(&"--use_migraphx".to_string()));
        assert!(cmd.args.contains(&"--build_wheel".to_string()));
        assert!(cmd.args.contains(&"--allow_running_as_root".to_string()));
    }

    #[test]
    fn test_build_command_has_rocm_version() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            rocm_version: Some("7.2.0".to_string()),
            ..Default::default()
        });
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        assert!(cmd.args.contains(&"--rocm_version".to_string()));
        // 7.2.0 -> "070200"
        assert!(cmd.args.contains(&"070200".to_string()));
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
    fn test_build_command_cmake_cxx_standard_env_var() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let rocm_env = RocmEnv::from_known(Some(PathBuf::from("/opt/rocm")), "7.2.0".to_string());
        let cmd = installer.build_build_command(&rocm_env);

        // The CXX standard should also be set as an environment variable for
        // cmake to pick up during the C++20 standard library test.
        assert!(
            cmd.env
                .iter()
                .any(|(k, v)| k == "CMAKE_CXX_STANDARD" && v == "20"),
            "build command env must include CMAKE_CXX_STANDARD=20, got env: {:?}",
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

        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "ROCM_HOME" && v == "/opt/rocm"));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "ROCM_PATH" && v == "/opt/rocm"));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "HIP_PATH" && v == "/opt/rocm"));
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "PYTHONPATH" && v.is_empty()));
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
        assert!(cmd.args.contains(&"--prefer-binary".to_string()));
        assert!(cmd.args.contains(&"onnxruntime-rocm".to_string()));
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
        assert!(cmd.args.contains(&"v1.20.1".to_string()));
    }

    // --- ROCm version formatting ---

    #[test]
    fn test_rocm_version_formatting() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        assert_eq!(installer.format_rocm_version_for_ort("7.2.0"), "070200");
        assert_eq!(installer.format_rocm_version_for_ort("6.4.3"), "060403");
        assert_eq!(installer.format_rocm_version_for_ort("7.1.0"), "070100");
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
    }

    // --- MIGraphX wheel install ---

    #[test]
    fn test_migraphx_wheel_url_default() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let url = installer.build_migraphx_wheel_url();
        assert!(url.contains("repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/"));
        assert!(url.contains("onnxruntime_migraphx-1.23.2"));
        assert!(url.contains("manylinux_2_27_x86_64"));
    }

    #[test]
    fn test_migraphx_wheel_url_custom_release() {
        let installer = OnnxRuntimeInstaller::new(OnnxRuntimeConfig {
            rocm_release: Some("7.1.0".to_string()),
            ..Default::default()
        });
        let url = installer.build_migraphx_wheel_url();
        assert!(url.contains("rocm-rel-7.1.0/"));
    }

    #[test]
    fn test_migraphx_install_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_migraphx_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.iter().any(|a| a.contains("repo.radeon.com")));
        assert!(cmd.args.iter().any(|a| a.contains("onnxruntime_migraphx")));
    }

    // --- Model optimizer ---

    #[test]
    fn test_model_optimizer_command() {
        let installer = OnnxRuntimeInstaller::with_defaults();
        let cmd = installer.build_model_optimizer_command("/path/to/model.onnx");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-c".to_string()));
        let script = &cmd.args[1];
        assert!(script.contains("ORT_ENABLE_ALL"));
        assert!(script.contains("/path/to/model.onnx"));
        assert!(script.contains("/path/to/model.onnx.optimized"));
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
