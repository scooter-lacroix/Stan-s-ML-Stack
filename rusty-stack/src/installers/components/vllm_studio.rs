//! vLLM Studio installer — ports `scripts/install_vllm_studio.sh`.
//!
//! Clones the vLLM Studio repository to the correct target directory,
//! installs controller and frontend dependencies via bun/npm.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-022**: vLLM Studio correct git clone
//! - **VAL-INSTALL-025**: App installers correct target directory

use crate::installers::common::command_exists;

// ===========================================================================
// Types
// ===========================================================================

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

/// Configuration for the vLLM Studio installer.
#[derive(Debug, Clone)]
pub struct VllmStudioConfig {
    /// Target installation directory.
    pub install_dir: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for VllmStudioConfig {
    fn default() -> Self {
        Self {
            install_dir: default_install_dir(),
            python_bin: "python3".to_string(),
            dry_run: false,
        }
    }
}

/// vLLM Studio Git repository URL.
pub const REPO_URL: &str = "https://github.com/0xSero/vllm-studio.git";

/// Default branch for vLLM Studio (uses default from remote).
pub const DEFAULT_BRANCH: &str = "main";

/// Default vLLM Studio installation directory.
pub fn default_install_dir() -> String {
    format!("{}/vllm-studio", std::env::var("HOME").unwrap_or_default())
}

/// The vLLM Studio installer.
pub struct VllmStudioInstaller {
    config: VllmStudioConfig,
}

impl VllmStudioInstaller {
    /// Create a new vLLM Studio installer with the given config.
    pub fn new(config: VllmStudioConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(VllmStudioConfig::default())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-022, VAL-INSTALL-025)
    // -----------------------------------------------------------------------

    /// Construct the git clone command for vLLM Studio.
    ///
    /// The original script runs:
    /// `git clone "https://github.com/0xSero/vllm-studio.git" "$VLLM_STUDIO_DIR"`
    pub fn build_git_clone_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                REPO_URL.to_string(),
                self.config.install_dir.clone(),
            ],
            env: vec![],
        }
    }

    /// Construct the git update commands for an existing clone.
    ///
    /// The original script runs:
    /// 1. `git -C "$DIR" remote set-head origin -a`
    /// 2. `git -C "$DIR" checkout "$DEFAULT_BRANCH"`
    /// 3. `git -C "$DIR" fetch --all`
    /// 4. `git -C "$DIR" reset --hard "origin/$DEFAULT_BRANCH"`
    pub fn build_git_update_commands(&self) -> Vec<ShellCommand> {
        let dir = &self.config.install_dir;
        vec![
            ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "remote".to_string(),
                    "set-head".to_string(),
                    "origin".to_string(),
                    "-a".to_string(),
                ],
                env: vec![],
            },
            ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "checkout".to_string(),
                    DEFAULT_BRANCH.to_string(),
                ],
                env: vec![],
            },
            ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "fetch".to_string(),
                    "--all".to_string(),
                ],
                env: vec![],
            },
            ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "reset".to_string(),
                    "--hard".to_string(),
                    format!("origin/{DEFAULT_BRANCH}"),
                ],
                env: vec![],
            },
        ]
    }

    /// Detect which package manager to use (bun preferred, npm fallback).
    pub fn detect_pkg_manager(&self) -> Option<String> {
        if command_exists("bun") {
            Some("bun".to_string())
        } else if command_exists("npm") {
            Some("npm".to_string())
        } else {
            None
        }
    }

    /// Construct the package install command for controller/frontend.
    pub fn build_pkg_install_command(&self, pkg_mgr: &str) -> ShellCommand {
        ShellCommand {
            program: pkg_mgr.to_string(),
            args: vec!["install".to_string()],
            env: vec![],
        }
    }

    /// Construct the frontend build command.
    pub fn build_frontend_build_command(&self, pkg_mgr: &str) -> ShellCommand {
        ShellCommand {
            program: pkg_mgr.to_string(),
            args: vec!["run".to_string(), "build".to_string()],
            env: vec![],
        }
    }

    /// Construct the shim creation command.
    ///
    /// Creates a shell script at `/usr/local/bin/vllm-studio` that
    /// runs `cd "$DIR/controller" && $PKG_MGR run start`.
    pub fn build_shim_content(&self, pkg_mgr: &str) -> String {
        format!(
            "#!/bin/bash\ncd \"{}\" && {} run start\n",
            format!("{}/controller", self.config.install_dir),
            pkg_mgr,
        )
    }

    /// Get the install directory.
    pub fn install_dir(&self) -> &str {
        &self.config.install_dir
    }

    /// Get the repository URL.
    pub fn repo_url(&self) -> &str {
        REPO_URL
    }

    /// Get the default branch.
    pub fn default_branch(&self) -> &str {
        DEFAULT_BRANCH
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_installer(dir: &str) -> VllmStudioInstaller {
        VllmStudioInstaller::new(VllmStudioConfig {
            install_dir: dir.to_string(),
            python_bin: "python3".to_string(),
            dry_run: false,
        })
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-022: vLLM Studio correct git clone
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_clone_url_matches_original_script() {
        let installer = make_installer("/home/user/vllm-studio");
        let cmd = installer.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&REPO_URL.to_string()));
        assert!(cmd.args.contains(&"/home/user/vllm-studio".to_string()));
    }

    #[test]
    fn test_git_clone_uses_correct_repo_url() {
        assert_eq!(
            REPO_URL, "https://github.com/0xSero/vllm-studio.git",
            "URL must match install_vllm_studio.sh REPO_URL"
        );
    }

    #[test]
    fn test_git_clone_command_string() {
        let installer = make_installer("/home/user/vllm-studio");
        let cmd = installer.build_git_clone_command();
        let s = cmd.to_command_string();
        assert!(s.contains("git clone"));
        assert!(s.contains(REPO_URL));
        assert!(s.contains("/home/user/vllm-studio"));
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-025: App installers correct target directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_install_dir_is_home_vllm_studio() {
        let config = VllmStudioConfig::default();
        assert!(
            config.install_dir.ends_with("/vllm-studio"),
            "Default dir must end with /vllm-studio, got: {}",
            config.install_dir,
        );
    }

    #[test]
    fn test_custom_install_dir() {
        let installer = make_installer("/opt/apps/vllm-studio");
        assert_eq!(installer.install_dir(), "/opt/apps/vllm-studio");
    }

    #[test]
    fn test_git_update_commands_reference_correct_dir() {
        let installer = make_installer("/home/test/vllm-studio");
        let cmds = installer.build_git_update_commands();
        assert_eq!(cmds.len(), 4);
        // All commands should reference the install dir
        for cmd in &cmds {
            assert!(
                cmd.args.contains(&"/home/test/vllm-studio".to_string()),
                "Command {} should reference install dir",
                cmd.to_command_string(),
            );
        }
    }

    #[test]
    fn test_pkg_install_command_bun() {
        let installer = make_installer("/home/user/vllm-studio");
        let cmd = installer.build_pkg_install_command("bun");
        assert_eq!(cmd.program, "bun");
        assert_eq!(cmd.args, vec!["install"]);
    }

    #[test]
    fn test_pkg_install_command_npm() {
        let installer = make_installer("/home/user/vllm-studio");
        let cmd = installer.build_pkg_install_command("npm");
        assert_eq!(cmd.program, "npm");
        assert_eq!(cmd.args, vec!["install"]);
    }

    #[test]
    fn test_frontend_build_command() {
        let installer = make_installer("/home/user/vllm-studio");
        let cmd = installer.build_frontend_build_command("npm");
        assert_eq!(cmd.program, "npm");
        assert!(cmd.args.contains(&"run".to_string()));
        assert!(cmd.args.contains(&"build".to_string()));
    }

    #[test]
    fn test_shim_content_references_controller_dir() {
        let installer = make_installer("/home/user/vllm-studio");
        let shim = installer.build_shim_content("npm");
        assert!(shim.contains("/home/user/vllm-studio/controller"));
        assert!(shim.contains("npm run start"));
        assert!(shim.starts_with("#!/bin/bash"));
    }

    #[test]
    fn test_default_branch_is_main() {
        assert_eq!(DEFAULT_BRANCH, "main");
    }
}
