//! FastVideo installer — ports `scripts/build_fastvideo_rocm.sh`.
//!
//! Clones scooter-lacroix/FastVideo feature/rocm-gfx11-support branch,
//! builds with ROCm support via `./build.sh --rocm`, then pip installs.
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

/// FastVideo installer.
#[derive(Debug, Clone)]
pub struct FastVideoInstaller {
    repo_url: String,
    branch: String,
    build_dir: String,
    python_bin: String,
}

impl Default for FastVideoInstaller {
    fn default() -> Self {
        Self::new()
    }
}

impl FastVideoInstaller {
    pub fn new() -> Self {
        Self {
            repo_url: FASTVIDEO_REPO.to_string(),
            branch: FASTVIDEO_BRANCH.to_string(),
            build_dir: FASTVIDEO_BUILD_DIR.to_string(),
            python_bin: "python3".to_string(),
        }
    }

    /// Build the sequence of commands to install FastVideo with ROCm support.
    pub fn build_commands(&self) -> Vec<ShellCommand> {
        vec![
            self.mkdir_build_dir(),
            self.git_clone(),
            self.git_checkout(),
            self.build_rocm(),
            self.pip_install(),
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

    pub fn build_rocm(&self) -> ShellCommand {
        ShellCommand {
            program: "./build.sh".to_string(),
            args: vec!["--rocm".to_string()],
            env: vec![],
            working_dir: None,
        }
    }

    pub fn pip_install(&self) -> ShellCommand {
        ShellCommand {
            program: self.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                ".".to_string(),
            ],
            env: vec![],
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
        let inst = FastVideoInstaller::new();
        let cmds = inst.build_commands();
        assert_eq!(cmds.len(), 6, "Should produce 6 commands");
    }

    #[test]
    fn test_git_clone_uses_fork() {
        let inst = FastVideoInstaller::new();
        let cmds = inst.build_commands();
        let clone_cmd = &cmds[1];
        assert!(clone_cmd.args.contains(&"clone".into()));
        assert!(
            clone_cmd.args.contains(&FASTVIDEO_REPO.into()),
            "Should clone from scooter-lacroix fork"
        );
    }

    #[test]
    fn test_build_uses_rocm_flag() {
        let inst = FastVideoInstaller::new();
        let cmds = inst.build_commands();
        let build_cmd = &cmds[3];
        assert!(build_cmd.args.contains(&"--rocm".into()));
    }

    #[test]
    fn test_cleanup_removes_build_dir() {
        let inst = FastVideoInstaller::new();
        let cmds = inst.build_commands();
        let cleanup = &cmds[5];
        assert!(cleanup.args.contains(&"-rf".into()));
        assert!(cleanup.args.contains(&FASTVIDEO_BUILD_DIR.into()));
    }
}
