//! FastVideo installer — native Rust build for ROCm gfx11 support.
//!
//! Clones scooter-lacroix/FastVideo feature/rocm-gfx11-support branch,
//! builds fastvideo-kernel with ROCm support using cmake, then pip installs.
//!
//! Instead of delegating to `./build.sh --rocm` (which calls `uv pip install`
//! without `--system` and fails outside a venv), this module replicates the
//! build steps individually with proper env var injection and pip prefix
//! handling consistent with the rest of the rusty-stack installer ecosystem.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-050**: FastVideo builds from fork with ROCm gfx11 support

use std::path::PathBuf;

const FASTVIDEO_REPO: &str = "https://github.com/scooter-lacroix/FastVideo.git";
const FASTVIDEO_BRANCH: &str = "feature/rocm-gfx11-support";
const FASTVIDEO_BUILD_DIR: &str = "/tmp/FastVideo_ROCm_build";

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

/// Configuration for the FastVideo installer.
#[derive(Debug, Clone)]
pub struct FastVideoConfig {
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_arch: String,
    /// Python binary to use.
    pub python_bin: String,
}

impl Default for FastVideoConfig {
    fn default() -> Self {
        Self {
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
        }
    }
}

/// FastVideo installer.
#[derive(Debug, Clone)]
pub struct FastVideoInstaller {
    repo_url: String,
    branch: String,
    build_dir: String,
    config: FastVideoConfig,
}

impl Default for FastVideoInstaller {
    fn default() -> Self {
        Self::new(FastVideoConfig::default())
    }
}

impl FastVideoInstaller {
    pub fn new(config: FastVideoConfig) -> Self {
        Self {
            repo_url: FASTVIDEO_REPO.to_string(),
            branch: FASTVIDEO_BRANCH.to_string(),
            build_dir: FASTVIDEO_BUILD_DIR.to_string(),
            config,
        }
    }

    /// Build the sequence of commands to install FastVideo with ROCm support.
    pub fn build_commands(&self) -> Vec<ShellCommand> {
        vec![
            self.mkdir_build_dir(),
            self.git_clone(),
            self.git_checkout(),
            self.git_submodule_init(),
            self.install_build_deps(),
            self.pip_install_kernel(),
            self.cleanup(),
        ]
    }

    pub fn mkdir_build_dir(&self) -> ShellCommand {
        ShellCommand {
            program: "mkdir".to_string(),
            args: vec!["-p".to_string(), self.build_dir.clone()],
            env: vec![],
            working_dir: None,
        }
    }

    pub fn git_clone(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                self.repo_url.clone(),
                ".".to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    pub fn git_checkout(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec!["checkout".to_string(), self.branch.clone()],
            env: vec![],
            working_dir: None,
        }
    }

    /// Initialize git submodules (cutlass, ThunderKittens, composable_kernel).
    ///
    /// The `--recursive` flag ensures nested submodules like composable_kernel
    /// (required by flash_attn_rocm for MHA device implementations) are also
    /// initialized.
    pub fn git_submodule_init(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "submodule".to_string(),
                "update".to_string(),
                "--init".to_string(),
                "--recursive".to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Install build dependencies (scikit-build-core, cmake, ninja).
    ///
    /// Uses `python -m pip install` with `--break-system-packages` when needed,
    /// matching the pattern used by Triton, AITER, and other source-build installers.
    pub fn install_build_deps(&self) -> ShellCommand {
        let use_break = std::env::var("VIRTUAL_ENV").is_err()
            && std::env::var("CONDA_PREFIX").is_err();
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "--upgrade".to_string(),
            "--no-cache-dir".to_string(),
            "scikit-build-core".to_string(),
            "cmake".to_string(),
            "ninja".to_string(),
        ]);

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env: vec![],
            working_dir: None,
        }
    }

    /// Build and install fastvideo-kernel with ROCm support.
    ///
    /// Sets CMAKE_ARGS for ROCm:
    /// - `-DCMAKE_HIP_ARCHITECTURES=<gpu_arch>`
    /// - `-DFASTVIDEO_KERNEL_BUILD_TK=OFF` (ThunderKittens not supported on ROCm)
    /// - `-DGPU_BACKEND=ROCM`
    ///
    /// Uses `pip install --no-build-isolation` matching the upstream build.sh.
    pub fn pip_install_kernel(&self) -> ShellCommand {
        let use_break = std::env::var("VIRTUAL_ENV").is_err()
            && std::env::var("CONDA_PREFIX").is_err();
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        if use_break {
            args.push("--break-system-packages".to_string());
        }
        args.extend([
            "-v".to_string(),
            "--no-build-isolation".to_string(),
            ".".to_string(),
        ]);

        let cmake_args = format!(
            "-DCMAKE_HIP_ARCHITECTURES={} -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=ROCM",
            self.config.gpu_arch
        );

        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(4);

        let env = vec![
            ("CMAKE_ARGS".to_string(), cmake_args),
            ("GPU_ARCHS".to_string(), self.config.gpu_arch.clone()),
            ("MAX_JOBS".to_string(), nproc.to_string()),
            ("GPU_BACKEND".to_string(), "ROCM".to_string()),
        ];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env,
            working_dir: None,
        }
    }

    pub fn cleanup(&self) -> ShellCommand {
        ShellCommand {
            program: "rm".to_string(),
            args: vec!["-rf".to_string(), self.build_dir.clone()],
            env: vec![],
            working_dir: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_commands_count() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cmds = inst.build_commands();
        assert_eq!(cmds.len(), 7, "Should produce 7 commands");
    }

    #[test]
    fn test_git_clone_uses_fork() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cmds = inst.build_commands();
        let clone_cmd = &cmds[1];
        assert!(clone_cmd.args.contains(&"clone".into()));
        assert!(
            clone_cmd.args.contains(&FASTVIDEO_REPO.into()),
            "Should clone from scooter-lacroix fork"
        );
    }

    #[test]
    fn test_submodule_init_present() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cmds = inst.build_commands();
        let submod_cmd = &cmds[3];
        assert!(submod_cmd.args.contains(&"submodule".into()));
        assert!(submod_cmd.args.contains(&"--recursive".into()));
    }

    #[test]
    fn test_pip_install_has_rocm_cmake_args() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        assert!(cmd.args.contains(&"--no-build-isolation".into()));
        let cmake_env = cmd.env.iter().find(|(k, _)| k == "CMAKE_ARGS").unwrap();
        assert!(cmake_env.1.contains("CMAKE_HIP_ARCHITECTURES=gfx1100"));
        assert!(cmake_env.1.contains("GPU_BACKEND=ROCM"));
        assert!(cmake_env.1.contains("FASTVIDEO_KERNEL_BUILD_TK=OFF"));
    }

    #[test]
    fn test_pip_install_sets_gpu_archs() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_arch: "gfx1200".to_string(),
            python_bin: "python3".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        let gpu_env = cmd.env.iter().find(|(k, _)| k == "GPU_ARCHS").unwrap();
        assert_eq!(gpu_env.1, "gfx1200");
    }

    #[test]
    fn test_cleanup_removes_build_dir() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cmds = inst.build_commands();
        let cleanup = &cmds[6];
        assert!(cleanup.args.contains(&"-rf".into()));
        assert!(cleanup.args.contains(&FASTVIDEO_BUILD_DIR.into()));
    }

    #[test]
    fn test_config_propagates_python_bin() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_arch: "gfx1100".to_string(),
            python_bin: "python3.12".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        assert_eq!(cmd.program, "python3.12");
    }
}
