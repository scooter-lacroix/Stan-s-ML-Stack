//! MIGraphX Python bindings installer — ports `scripts/install_migraphx_python.sh`.
//!
//! Installs the MIGraphX Python module via pip. Requires MIGraphX system
//! package to be installed first.
//!
//! # Arch Linux / CachyOS Handling
//!
//! On Arch-family distros, the `migraphx` pip wheel is not available.
//! The installer should be skipped on these distros with a clear message
//! informing the user. Use `is_available_on_distro()` to check before
//! attempting installation.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-030**: MIGraphX Python correct pip command

use crate::installers::common::DistroFacade;
use crate::platform::detection::DistroFamily;

// ===========================================================================
// Types
// ===========================================================================

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

impl std::fmt::Display for InstallMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallMethod::Global => write!(f, "global"),
            InstallMethod::Venv => write!(f, "venv"),
            InstallMethod::Auto => write!(f, "auto"),
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

/// Configuration for the MIGraphX Python installer.
#[derive(Debug, Clone)]
pub struct MigraphxPythonConfig {
    /// Python binary to use.
    pub python_bin: String,
    /// Installation method.
    pub method: InstallMethod,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// Whether to force reinstall.
    pub force: bool,
}

impl Default for MigraphxPythonConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            method: InstallMethod::Auto,
            dry_run: false,
            force: false,
        }
    }
}

/// Package name for MIGraphX Python bindings.
pub const PACKAGE_NAME: &str = "migraphx";

/// Env var that opts into building the MIGraphX Python bindings **from source**
/// (AMDMIGraphX) — the only path on Arch/CachyOS, where no pip/system wheel
/// exists. Off by default (the C++ core + onnxruntime MIGraphX EP cover ONNX
/// inference). Values: `1`/`true`/`yes`/`on`.
pub const MIGRAPHX_BUILD_PYTHON_ENV: &str = "MLSTACK_MIGRAPHX_BUILD_PYTHON";

/// Env var that opts into building a PATCHED MIGraphX C++ core from source
/// (AMDMIGraphX `rocm-7.2.3` + backport of upstream PR #5106 "Fix
/// find_concat_transpose with non-transposed inputs"). This is the LIBRARY-level
/// fix for the MIGraphX 2.15.0 `repeat_while_changes` compile hang /
/// `simplify_reshapes.cpp:845 find_concat_transpose: Assertion s.transposed()
/// failed` crash — the same defect the offline pre-opt workaround dodges.
///
/// The patched core is installed into `~/.mlstack/migraphx` (same SONAME
/// `2015000` as the distro package, so ORT's provider loads it unchanged); the
/// caller must prepend its `lib` dir to `LD_LIBRARY_PATH` (the ORT provider
/// uses RUNPATH, which is searched AFTER `LD_LIBRARY_PATH`, so the fixed lib
/// wins without touching `/opt/rocm`). Opt-in only — off by default, because
/// the pre-optimization workaround already ships and a full core build is heavy
/// (~30-90 min). Values: `1`/`true`/`yes`/`on`.
pub const MIGRAPHX_CORE_FIX_ENV: &str = "MLSTACK_MIGRAPHX_CORE_FIX";

/// Env var overriding the AMDMIGraphX branch to build. The standalone
/// python bindings must match the installed C++ core, so the default tracks the
/// Arch `migraphx` package line.
pub const MIGRAPHX_SOURCE_BRANCH_ENV: &str = "MLSTACK_MIGRAPHX_SOURCE_BRANCH";

/// Default AMDMIGraphX branch (matches the Arch `migraphx 7.2.3` package).
pub const AMDMIGRAPHX_DEFAULT_BRANCH: &str = "rocm-7.2.3";

/// Upstream source for the MIGraphX Python bindings.
pub const AMDMIGRAPHX_REPO: &str = "https://github.com/ROCm/AMDMIGraphX";

/// Install prefix for the patched MIGraphX core — the canonical `~/.mlstack/migraphx`
/// directory (the sole MIGraphX install managed by the stack). Kept
/// version-agnostic so a future branch bump does not strand old builds.
pub const MIGRAPHX_FIXED_PREFIX: &str = ".mlstack/migraphx";

/// Whether the user opted into building the patched MIGraphX core from source.
pub fn core_fix_requested() -> bool {
    let v = std::env::var(MIGRAPHX_CORE_FIX_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(v.as_str(), "1" | "true" | "yes" | "on")
}

/// Whether the user opted into building the Python bindings from source.
pub fn build_python_source_requested() -> bool {
    let v = std::env::var(MIGRAPHX_BUILD_PYTHON_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(v.as_str(), "1" | "true" | "yes" | "on")
}

/// The AMDMIGraphX branch to build: the `MLSTACK_MIGRAPHX_SOURCE_BRANCH` env
/// override if set, else [`AMDMIGRAPHX_DEFAULT_BRANCH`].
pub fn source_branch() -> String {
    let v = std::env::var(MIGRAPHX_SOURCE_BRANCH_ENV)
        .unwrap_or_default()
        .trim()
        .to_string();
    if v.is_empty() {
        AMDMIGRAPHX_DEFAULT_BRANCH.to_string()
    } else {
        v
    }
}

/// The MIGraphX Python installer.
pub struct MigraphxPythonInstaller {
    config: MigraphxPythonConfig,
}

impl MigraphxPythonInstaller {
    /// Create a new MIGraphX Python installer with the given config.
    pub fn new(config: MigraphxPythonConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MigraphxPythonConfig::default())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-030)
    // -----------------------------------------------------------------------

    /// Construct the pip install command for MIGraphX Python.
    ///
    /// The original script runs:
    /// `uv pip install migraphx` or `pip install migraphx`
    /// With optional `--break-system-packages` for global installs.
    pub fn build_install_command(&self) -> ShellCommand {
        let use_break = self.config.method == InstallMethod::Global
            || self.config.method == InstallMethod::Auto;

        let mut args = vec!["-m".to_string(), "pip".to_string(), "install".to_string()];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        if self.config.force {
            args.push("--force-reinstall".to_string());
        }
        args.push(PACKAGE_NAME.to_string());

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
        }
    }

    /// Construct the uv pip install command (preferred method).
    pub fn build_uv_install_command(&self) -> ShellCommand {
        let mut args = vec!["pip".to_string(), "install".to_string()];
        if self.config.force {
            args.push("--force-reinstall".to_string());
        }
        args.push(PACKAGE_NAME.to_string());

        ShellCommand {
            program: "uv".to_string(),
            args,
            env: vec![
                ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
                ("AMD_LOG_LEVEL".to_string(), "0".to_string()),
            ],
        }
    }

    // -----------------------------------------------------------------------
    // Opt-in source build (Arch/CachyOS — no pip/system wheel)
    // -----------------------------------------------------------------------

    /// Ordered shell commands to build the MIGraphX Python bindings **from
    /// source** (AMDMIGraphX) against the installed `/opt/rocm` migraphx core.
    ///
    /// This is the only path to the standalone `migraphx` python API
    /// (`migraphx.parse_onnx`) on Arch/CachyOS, where no pip/system wheel
    /// exists. Opt-in only via [`MIGRAPHX_BUILD_PYTHON_ENV`] — off by default,
    /// because the C++ core + onnxruntime `MIGraphXExecutionProvider` already
    /// cover ONNX inference. Heavy build (~20-40 min).
    ///
    /// - `workdir`     — scratch dir for the clone + build (e.g. `/tmp/...`)
    /// - `rocm_prefix` — `/opt/rocm`
    /// - `gpu_arch`    — `gfx1100` (or detected `GPU_ARCH`)
    /// - `branch`      — matching ROCm branch, e.g. `rocm-7.2.3`
    pub fn build_source_commands(
        &self,
        workdir: &str,
        rocm_prefix: &str,
        gpu_arch: &str,
        branch: &str,
    ) -> Vec<ShellCommand> {
        let src = format!("{workdir}/AMDMIGraphX");
        let build = format!("{src}/build");
        let clang = format!("{rocm_prefix}/llvm/bin/clang");
        let clangxx = format!("{rocm_prefix}/llvm/bin/clang++");
        vec![
            // 1. Shallow clone of the matching ROCm branch.
            ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "clone".to_string(),
                    "--depth".to_string(),
                    "1".to_string(),
                    "--single-branch".to_string(),
                    "--branch".to_string(),
                    branch.to_string(),
                    AMDMIGRAPHX_REPO.to_string(),
                    src.clone(),
                ],
                env: vec![],
            },
            // 2. Configure: python on, against /opt/rocm, ROCm clang, GPU arch.
            ShellCommand {
                program: "cmake".to_string(),
                args: vec![
                    "-S".to_string(),
                    src.clone(),
                    "-B".to_string(),
                    build.clone(),
                    "-DMIGRAPHX_ENABLE_PYTHON=On".to_string(),
                    format!("-DCMAKE_PREFIX_PATH={rocm_prefix}"),
                    format!("-DGPU_TARGETS={gpu_arch}"),
                    format!("-DCMAKE_C_COMPILER={clang}"),
                    format!("-DCMAKE_CXX_COMPILER={clangxx}"),
                    "-DCMAKE_BUILD_TYPE=Release".to_string(),
                ],
                env: vec![],
            },
            // 3. Build the python extension target.
            ShellCommand {
                program: "cmake".to_string(),
                args: vec![
                    "--build".to_string(),
                    build,
                    "--target".to_string(),
                    "migraphx_py".to_string(),
                    "-j".to_string(),
                ],
                env: vec![],
            },
            // 4. Install the built python package into the active interpreter.
            ShellCommand {
                program: self.config.python_bin.clone(),
                args: vec![
                    "-m".to_string(),
                    "pip".to_string(),
                    "install".to_string(),
                    format!("{src}/python"),
                ],
                env: vec![],
            },
        ]
    }

    // -----------------------------------------------------------------------
    // Patched C++ core build (library-level compile-hang fix, env-gated)
    // -----------------------------------------------------------------------

    /// Ordered shell commands to build a PATCHED MIGraphX C++ core from source
    /// (AMDMIGraphX `rocm-7.2.3` + backport of upstream PR #5106) and install it
    /// into [`MIGRAPHX_FIXED_PREFIX`] under `user_home`. Opt-in via
    /// [`MIGRAPHX_CORE_FIX_ENV`]; see its doc comment for the defect this fixes.
    ///
    /// The built libs keep the distro SONAME (`libmigraphx.so.2015000` for
    /// MIGraphX 2.15.0), so the ORT MIGraphX EP loads them unchanged. The caller
    /// should export `LD_LIBRARY_PATH=<prefix>/lib:$LD_LIBRARY_PATH` (the EP
    /// provider uses RUNPATH, which is searched AFTER `LD_LIBRARY_PATH`, so the
    /// patched lib wins without modifying `/opt/rocm`).
    ///
    /// Config flags mirror what the distro packages actually ship so the build
    /// matches the installed toolchain: `MIGRAPHX_USE_COMPOSABLEKERNEL=Off`
    /// (distro composable-kernel 7.2.4 dropped the `jit_library` component the
    /// 7.2.3 branch requires) and `MIGRAPHX_ENABLE_MLIR=Off` (the `migraphx`
    /// Arch package has no `rocmlir` dependency). The fix itself is a pure C++
    /// reshape-simplification change, unaffected by either option.
    ///
    /// - `workdir`     — scratch dir for the clone + build (e.g. `/tmp/...`)
    /// - `rocm_prefix` — `/opt/rocm`
    /// - `gpu_arch`    — `gfx1100` (or detected `GPU_ARCH`)
    /// - `branch`      — matching ROCm branch, e.g. `rocm-7.2.3`
    /// - `patch_path`  — path to the vendored PR #5106 backport patch
    ///   (`rusty-stack/patches/amdmigraphx-5106-find_concat_transpose.patch`)
    pub fn build_fixed_core_commands(
        &self,
        workdir: &str,
        rocm_prefix: &str,
        gpu_arch: &str,
        branch: &str,
        patch_path: &str,
        user_home: &str,
    ) -> Vec<ShellCommand> {
        let src = format!("{workdir}/AMDMIGraphX");
        let build = format!("{src}/build");
        let prefix = format!("{user_home}/{}", MIGRAPHX_FIXED_PREFIX);
        let deps = format!("{workdir}/deps");
        let nlohmann = format!("{deps}/nlohmann-json");
        let clang = format!("{rocm_prefix}/llvm/bin/clang");
        let clangxx = format!("{rocm_prefix}/llvm/bin/clang++");
        let patch_abs = std::path::Path::new(patch_path);
        let patch_arg = if patch_abs.is_absolute() {
            patch_path.to_string()
        } else {
            // Resolve relative to the crate root: `rusty-stack/patches/...`
            // when invoked from the repo root.
            let cwd = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            cwd.join(patch_path).to_string_lossy().to_string()
        };
        vec![
            // 1. Vendor nlohmann-json (header-only, no system package needed) into
            //    the build prefix so `find_package(nlohmann_json)` succeeds.
            //    Idempotent: reuses a previous clone (like the verified system
            //    script) so a partial/failed run can be resumed.
            ShellCommand {
                program: "bash".to_string(),
                args: vec![
                    "-c".to_string(),
                    format!(
                        "mkdir -p \"{deps}\" && cd \"{deps}\" && \
                         [ -d \"{nlohmann}/.git\" ] || git clone --depth 1 --branch v3.12.0 \
                         https://github.com/nlohmann/json.git \"{nlohmann}\" && \
                         cmake -S \"{nlohmann}\" -B \"{nlohmann}/build\" \
                         -DCMAKE_INSTALL_PREFIX=\"{prefix}\" -DJSON_BuildTests=Off \
                         -DJSON_MultipleHeaders=Off && \
                         cmake --install \"{nlohmann}/build\""
                    ),
                ],
                env: vec![],
            },
            // 2. Shallow clone of the matching ROCm branch. Idempotent: skip if a
            //    previous clone already exists (resume after a partial/failed run).
            ShellCommand {
                program: "bash".to_string(),
                args: vec![
                    "-c".to_string(),
                    format!(
                        "[ -d \"{src}/.git\" ] || git clone --depth 1 --single-branch \
                         --branch \"{branch}\" {repo} \"{src}\"",
                        repo = AMDMIGRAPHX_REPO,
                    ),
                ],
                env: vec![],
            },
            // 3. Apply the vendored PR #5106 backport (fixes
            //    find_concat_transpose: Assertion s.transposed() + the
            //    repeat_while_changes non-convergence it drives). Idempotent: a
            //    re-run after a partial apply must not fail — check whether the
            //    fix marker is already present before applying.
            ShellCommand {
                program: "bash".to_string(),
                args: vec![
                    "-c".to_string(),
                    format!(
                        "if ! grep -q 'get_permutation' \"{src}/src/simplify_reshapes.cpp\"; then \
                         git -C \"{src}\" apply \"{patch_arg}\"; fi"
                    ),
                ],
                env: vec![],
            },
            // 4. Configure: GPU only (CK JIT lib absent from distro CK 7.2.4),
            //    no MLIR (distro migraphx has no rocmlir dep), python/tests/
            //    examples off, ROCm clang, GPU arch, user install prefix.
            ShellCommand {
                program: "cmake".to_string(),
                args: vec![
                    "-S".to_string(),
                    src.clone(),
                    "-B".to_string(),
                    build.clone(),
                    format!("-DCMAKE_PREFIX_PATH={rocm_prefix}"),
                    format!("-Dnlohmann_json_DIR={prefix}/share/cmake/nlohmann_json"),
                    format!("-DGPU_TARGETS={gpu_arch}"),
                    format!("-DCMAKE_C_COMPILER={clang}"),
                    format!("-DCMAKE_CXX_COMPILER={clangxx}"),
                    "-DCMAKE_BUILD_TYPE=Release".to_string(),
                    "-DMIGRAPHX_ENABLE_PYTHON=Off".to_string(),
                    "-DMIGRAPHX_ENABLE_CPP_EXAMPLES=Off".to_string(),
                    "-DMIGRAPHX_ENABLE_TEST=Off".to_string(),
                    "-DMIGRAPHX_ENABLE_CPU=Off".to_string(),
                    "-DMIGRAPHX_USE_COMPOSABLEKERNEL=Off".to_string(),
                    "-DMIGRAPHX_ENABLE_MLIR=Off".to_string(),
                    format!("-DCMAKE_INSTALL_PREFIX={prefix}"),
                ],
                env: vec![],
            },
            // 5. Build + 6. install the patched core into the user prefix.
            ShellCommand {
                program: "cmake".to_string(),
                args: vec![
                    "--build".to_string(),
                    build.clone(),
                    "-j".to_string(),
                ],
                env: vec![],
            },
            ShellCommand {
                program: "cmake".to_string(),
                args: vec![
                    "--install".to_string(),
                    build.clone(),
                ],
                env: vec![],
            },
        ]
    }

    /// Construct the command to check if MIGraphX system package is installed.
    pub fn build_migraphx_check_command(&self) -> ShellCommand {
        ShellCommand {
            program: "migraphx-driver".to_string(),
            args: vec!["--version".to_string()],
            env: vec![],
        }
    }

    /// Construct the command to verify MIGraphX Python module import.
    pub fn build_python_import_check(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-c".to_string(),
                "import migraphx; print(getattr(migraphx, '__version__', 'unknown'))".to_string(),
            ],
            env: vec![],
        }
    }

    /// Get the package name.
    pub fn package_name(&self) -> &'static str {
        PACKAGE_NAME
    }

    // -----------------------------------------------------------------------
    // Distro availability (Arch-specific handling)
    // -----------------------------------------------------------------------

    /// Check whether MIGraphX Python bindings are available on the given distro.
    ///
    /// On Arch-family distros, the `migraphx` pip wheel does not exist in PyPI
    /// and there is no `python3-migraphx` system package. The installation
    /// should be skipped on these distros.
    pub fn is_available_on_distro(&self, distro: &DistroFacade) -> bool {
        !matches!(distro.family(), DistroFamily::Arch)
    }

    /// Build a human-readable message explaining why MIGraphX Python bindings
    /// are not available on the given distro. Returns `None` if available.
    pub fn build_unavailable_message(&self, distro: &DistroFacade) -> Option<String> {
        if self.is_available_on_distro(distro) {
            return None;
        }
        Some(format!(
            "MIGraphX Python bindings (pip install migraphx) are not available on {}. \
             The migraphx pip wheel is not published for Arch-family distros. \
             If you need Python bindings, consider using the ROCm Docker image \
             or building MIGraphX from source with Python bindings enabled.",
            distro.id()
        ))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-INSTALL-030: MIGraphX Python correct pip command
    // -----------------------------------------------------------------------

    #[test]
    fn test_package_name_matches_original_script() {
        assert_eq!(
            PACKAGE_NAME, "migraphx",
            "Package name must match install_migraphx_python.sh package"
        );
    }

    #[test]
    fn test_install_command_auto_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Auto,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_install_command_global_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Global,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_install_command_venv_method() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        // Venv method should NOT include --break-system-packages
        assert!(!cmd.args.contains(&"--break-system-packages".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
    }

    #[test]
    fn test_install_command_with_force() {
        let installer = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            force: true,
            method: InstallMethod::Venv,
            ..Default::default()
        });
        let cmd = installer.build_install_command();
        assert!(cmd.args.contains(&"--force-reinstall".to_string()));
    }

    #[test]
    fn test_install_command_string() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_install_command();
        let s = cmd.to_command_string();
        assert!(s.contains("python3"));
        assert!(s.contains("pip install"));
        assert!(s.contains("migraphx"));
    }

    #[test]
    fn test_uv_install_command() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_uv_install_command();
        assert_eq!(cmd.program, "uv");
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"migraphx".to_string()));
        // Should have ROCm env vars
        assert!(cmd.env.iter().any(|(k, _)| k == "ROCM_PATH"));
        assert!(cmd.env.iter().any(|(k, _)| k == "AMD_LOG_LEVEL"));
    }

    #[test]
    fn test_migraphx_check_command() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_migraphx_check_command();
        assert_eq!(cmd.program, "migraphx-driver");
        assert!(cmd.args.contains(&"--version".to_string()));
    }

    #[test]
    fn test_python_import_check() {
        let installer = MigraphxPythonInstaller::with_defaults();
        let cmd = installer.build_python_import_check();
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.iter().any(|a| a.contains("import migraphx")));
    }

    // --- Arch-specific availability tests ---

    #[test]
    fn test_is_available_debian() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(
            installer.is_available_on_distro(&distro),
            "MIGraphX Python should be available on Debian"
        );
    }

    #[test]
    fn test_is_available_rhel() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "fedora".to_string(),
            family: DistroFamily::Rhel,
            pkg_manager: PackageManager::Dnf,
            ..Default::default()
        });
        assert!(
            installer.is_available_on_distro(&distro),
            "MIGraphX Python should be available on RHEL"
        );
    }

    #[test]
    fn test_not_available_on_arch() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert!(
            !installer.is_available_on_distro(&distro),
            "MIGraphX Python should NOT be available on Arch"
        );
    }

    #[test]
    fn test_not_available_on_cachyos() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "cachyos".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        assert!(
            !installer.is_available_on_distro(&distro),
            "MIGraphX Python should NOT be available on CachyOS"
        );
    }

    #[test]
    fn test_unavailable_message_debian_is_none() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "ubuntu".to_string(),
            family: DistroFamily::Debian,
            pkg_manager: PackageManager::Apt,
            ..Default::default()
        });
        assert!(
            installer.build_unavailable_message(&distro).is_none(),
            "Debian should not have an unavailable message"
        );
    }

    #[test]
    fn test_unavailable_message_arch_is_informative() {
        use crate::platform::detection::{DistroInfo, PackageManager};
        let installer = MigraphxPythonInstaller::with_defaults();
        let distro = DistroFacade::from_info(DistroInfo {
            id: "arch".to_string(),
            family: DistroFamily::Arch,
            pkg_manager: PackageManager::Pacman,
            ..Default::default()
        });
        let msg = installer
            .build_unavailable_message(&distro)
            .expect("Arch should have an unavailable message");
        assert!(
            msg.contains("not available"),
            "Message should say not available: {msg}"
        );
        assert!(msg.contains("Arch"), "Message should mention Arch: {msg}");
        assert!(msg.contains("pip"), "Message should mention pip: {msg}");
        assert!(
            msg.contains("Docker") || msg.contains("source"),
            "Message should suggest alternatives: {msg}"
        );
    }

    // --- Opt-in source build ---

    #[test]
    fn test_build_python_source_requested_env_gated() {
        let _env = crate::test_support::lock_env();
        std::env::remove_var(MIGRAPHX_BUILD_PYTHON_ENV);
        assert!(!build_python_source_requested(), "default must be OFF");
        for v in ["1", "true", "TRUE", "yes", "on"] {
            std::env::set_var(MIGRAPHX_BUILD_PYTHON_ENV, v);
            assert!(build_python_source_requested(), "should be ON for {v:?}");
        }
        std::env::set_var(MIGRAPHX_BUILD_PYTHON_ENV, "0");
        assert!(!build_python_source_requested());
    }

    #[test]
    fn test_source_branch_default_and_override() {
        let _env = crate::test_support::lock_env();
        std::env::remove_var(MIGRAPHX_SOURCE_BRANCH_ENV);
        assert_eq!(source_branch(), AMDMIGRAPHX_DEFAULT_BRANCH);
        std::env::set_var(MIGRAPHX_SOURCE_BRANCH_ENV, "rocm-7.2.4");
        assert_eq!(source_branch(), "rocm-7.2.4");
    }

    // --- Patched C++ core build (library-level fix) ---

    #[test]
    fn test_core_fix_requested_env_gated() {
        let _env = crate::test_support::lock_env();
        std::env::remove_var(MIGRAPHX_CORE_FIX_ENV);
        assert!(!core_fix_requested(), "default must be OFF");
        for v in ["1", "true", "TRUE", "yes", "on"] {
            std::env::set_var(MIGRAPHX_CORE_FIX_ENV, v);
            assert!(core_fix_requested(), "should be ON for {v:?}");
        }
        std::env::set_var(MIGRAPHX_CORE_FIX_ENV, "0");
        assert!(!core_fix_requested());
    }

    #[test]
    fn test_build_fixed_core_commands_sequence() {
        let inst = MigraphxPythonInstaller::with_defaults();
        let cmds = inst.build_fixed_core_commands(
            "/tmp/wd",
            "/opt/rocm",
            "gfx1100",
            "rocm-7.2.3",
            "patches/amdmigraphx-5106-find_concat_transpose.patch",
            "/home/user",
        );
        // nlohmann vendor, clone, apply patch, configure, build, install
        assert_eq!(cmds.len(), 6);

        // 0. vendor nlohmann-json (header-only) into the build prefix
        assert_eq!(cmds[0].program, "bash");
        assert!(cmds[0]
            .args
            .iter()
            .any(|a| a.contains("nlohmann/json.git")));
        assert!(cmds[0]
            .args
            .iter()
            .any(|a| a.contains("|| git clone")), // idempotent guard
        );

        // 1. shallow clone of the matching ROCm branch (bash -c, idempotent)
        assert_eq!(cmds[1].program, "bash");
        let clone = cmds[1].args.iter().find(|a| a.contains("git clone")).unwrap();
        assert!(clone.contains("--branch"));
        assert!(clone.contains("rocm-7.2.3"));
        assert!(clone.contains(AMDMIGRAPHX_REPO));
        assert!(clone.contains("|| git clone")); // skip-if-present

        // 2. apply the vendored PR #5106 backport patch (bash -c, idempotent)
        assert_eq!(cmds[2].program, "bash");
        let apply = cmds[2]
            .args
            .iter()
            .find(|a| a.contains("git -C"))
            .unwrap();
        assert!(apply.contains("get_permutation")); // grep guard
        assert!(
            apply.contains("amdmigraphx-5106-find_concat_transpose.patch"),
            "must apply the vendored backport patch: {apply}"
        );

        // 3. configure: GPU only, CK off (distro CK 7.2.4 has no jit_library),
        //    MLIR off (distro migraphx has no rocmlir dep), user install prefix
        assert_eq!(cmds[3].program, "cmake");
        assert!(cmds[3]
            .args
            .contains(&"-DMIGRAPHX_USE_COMPOSABLEKERNEL=Off".to_string()));
        assert!(cmds[3]
            .args
            .contains(&"-DMIGRAPHX_ENABLE_MLIR=Off".to_string()));
        assert!(cmds[3]
            .args
            .contains(&"-DMIGRAPHX_ENABLE_PYTHON=Off".to_string()));
        assert!(cmds[3]
            .args
            .iter()
            .any(|a| a.contains("-Dnlohmann_json_DIR=")
                && a.contains("migraphx")));
        assert!(cmds[3]
            .args
            .iter()
            .any(|a| a.contains("-DCMAKE_INSTALL_PREFIX=")
                && a.contains("/home/user/")
                && a.contains("migraphx")));

        // 4. build + 5. install
        assert_eq!(cmds[4].program, "cmake");
        assert!(cmds[4].args.contains(&"--build".to_string()));
        assert!(cmds[4].args.contains(&"-j".to_string()));
        assert_eq!(cmds[5].program, "cmake");
        assert!(cmds[5].args.contains(&"--install".to_string()));
    }

    #[test]
    fn test_build_fixed_core_commands_absolute_patch() {
        let inst = MigraphxPythonInstaller::with_defaults();
        let cmds = inst.build_fixed_core_commands(
            "/tmp/wd",
            "/opt/rocm",
            "gfx1100",
            "rocm-7.2.3",
            "/abs/path/amdmigraphx-5106.patch",
            "/home/user",
        );
        let apply = cmds[2]
            .args
            .iter()
            .find(|a| a.contains("git -C"))
            .unwrap();
        assert!(apply.contains("/abs/path/amdmigraphx-5106.patch"));
    }

    #[test]
    fn test_build_source_commands_sequence() {
        let inst = MigraphxPythonInstaller::new(MigraphxPythonConfig {
            python_bin: "/venv/bin/python".to_string(),
            ..Default::default()
        });
        let cmds = inst.build_source_commands("/tmp/wd", "/opt/rocm", "gfx1100", "rocm-7.2.3");
        assert_eq!(
            cmds.len(),
            4,
            "expected: clone, configure, build, pip install"
        );

        // 1. shallow clone of the matching ROCm branch
        assert_eq!(cmds[0].program, "git");
        assert!(cmds[0].args.contains(&"clone".to_string()));
        assert!(cmds[0].args.contains(&"--branch".to_string()));
        assert!(cmds[0].args.contains(&"rocm-7.2.3".to_string()));
        assert!(cmds[0].args.contains(&AMDMIGRAPHX_REPO.to_string()));

        // 2. configure: python on, ROCm clang, GPU arch, /opt/rocm prefix
        assert_eq!(cmds[1].program, "cmake");
        assert!(cmds[1]
            .args
            .contains(&"-DMIGRAPHX_ENABLE_PYTHON=On".to_string()));
        assert!(cmds[1]
            .args
            .contains(&"-DCMAKE_PREFIX_PATH=/opt/rocm".to_string()));
        assert!(cmds[1].args.contains(&"-DGPU_TARGETS=gfx1100".to_string()));
        assert!(cmds[1]
            .args
            .contains(&"-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++".to_string()));

        // 3. build the python extension target
        assert_eq!(cmds[2].program, "cmake");
        assert!(cmds[2].args.contains(&"--target".to_string()));
        assert!(cmds[2].args.contains(&"migraphx_py".to_string()));

        // 4. pip install the built package into the configured interpreter
        assert_eq!(cmds[3].program, "/venv/bin/python");
        assert!(cmds[3].args.contains(&"install".to_string()));
        assert!(cmds[3]
            .args
            .iter()
            .any(|a| a.ends_with("/AMDMIGraphX/python")));
    }
}
