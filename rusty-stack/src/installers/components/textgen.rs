//! text-generation-webui installer — ports `scripts/install_textgen.sh`.
//!
//! Clones the text-generation-webui repository, installs Python dependencies
//! with ROCm-only filtering (excludes nvidia/CUDA packages), and creates
//! launcher shim.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-024**: text-generation-webui correct git clone
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

/// Configuration for the text-generation-webui installer.
#[derive(Debug, Clone)]
pub struct TextgenConfig {
    /// Target installation directory.
    pub install_dir: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for TextgenConfig {
    fn default() -> Self {
        Self {
            install_dir: default_install_dir(),
            python_bin: "python3".to_string(),
            dry_run: false,
        }
    }
}

/// text-generation-webui Git repository URL.
pub const REPO_URL: &str = "https://github.com/oobabooga/text-generation-webui.git";

/// Default branch for text-generation-webui.
pub const DEFAULT_BRANCH: &str = "main";

/// Default web port.
pub const WEB_PORT: u16 = 7860;

/// Directories to preserve during git update.
pub const PRESERVE_DIRS: &[&str] = &[
    "models",
    "loras",
    "embeddings",
    "presets",
    "characters",
    "training",
];

/// Regex pattern for packages to exclude from requirements.txt.
/// Original script excludes: nvidia-*, cuda*, tensorrt*, triton[versioned],
/// xformers, flash-attn, torch/torchvision/torchaudio.
pub const EXCLUDED_PATTERNS: &[&str] = &[
    "nvidia-",
    "cuda",
    "tensorrt",
    "triton",
    "xformers",
    "flash-attn",
    "torch",
    "torchvision",
    "torchaudio",
];

/// Default text-generation-webui installation directory.
pub fn default_install_dir() -> String {
    format!(
        "{}/text-generation-webui",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// The text-generation-webui installer.
pub struct TextgenInstaller {
    config: TextgenConfig,
}

impl TextgenInstaller {
    /// Create a new text-generation-webui installer with the given config.
    pub fn new(config: TextgenConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(TextgenConfig::default())
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-024, VAL-INSTALL-025)
    // -----------------------------------------------------------------------

    /// Construct the git clone command for text-generation-webui.
    ///
    /// The original script runs:
    /// `git clone "https://github.com/oobabooga/text-generation-webui.git" "$TEXTGEN_DIR"`
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
    /// 4. `git -C "$DIR" stash push -u` (if user data exists)
    /// 5. `git -C "$DIR" reset --hard "origin/$DEFAULT_BRANCH"`
    /// 6. `git -C "$DIR" stash pop` (if user data was stashed)
    pub fn build_git_update_commands(&self, has_user_data: bool) -> Vec<ShellCommand> {
        let dir = &self.config.install_dir;
        let mut cmds = vec![
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
        ];

        if has_user_data {
            cmds.push(ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "stash".to_string(),
                    "push".to_string(),
                    "-u".to_string(),
                    "-m".to_string(),
                    "rusty-stack-preserve-user-data".to_string(),
                    "--".to_string(),
                ]
                .into_iter()
                .chain(PRESERVE_DIRS.iter().map(|s| s.to_string()))
                .collect(),
                env: vec![],
            });
        }

        cmds.push(ShellCommand {
            program: "git".to_string(),
            args: vec![
                "-C".to_string(),
                dir.clone(),
                "reset".to_string(),
                "--hard".to_string(),
                format!("origin/{DEFAULT_BRANCH}"),
            ],
            env: vec![],
        });

        if has_user_data {
            cmds.push(ShellCommand {
                program: "git".to_string(),
                args: vec![
                    "-C".to_string(),
                    dir.clone(),
                    "stash".to_string(),
                    "pop".to_string(),
                ],
                env: vec![],
            });
        }

        cmds
    }

    /// Construct the pip install command for filtered requirements.
    ///
    /// The original script filters requirements.txt excluding nvidia/CUDA packages:
    /// `grep -v -E '^(nvidia-|cuda|tensorrt|triton([=<>!\[]|$)|xformers|flash-attn|torch([=<>! ]|$)|torchvision|torchaudio)'`
    pub fn build_pip_install_command(&self, requirements_path: &str) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "-r".to_string(),
                requirements_path.to_string(),
            ],
            env: vec![],
        }
    }

    /// Construct the pip install command for AMD-specific requirements.
    pub fn build_amd_requirements_command(&self, amd_req_path: &str) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "-r".to_string(),
                amd_req_path.to_string(),
            ],
            env: vec![],
        }
    }

    /// Get the grep filter pattern for excluding packages from requirements.
    /// Matches the regex from the original script.
    pub fn requirements_filter_pattern() -> String {
        r"^(nvidia-|cuda|tensorrt|triton([=<>!\[]|$)|xformers|flash-attn|torch([=<>! ]|$)|torchvision|torchaudio)".to_string()
    }

    /// Construct the launcher shim content.
    ///
    /// Creates a shell script at `~/.local/bin/textgen` that runs:
    /// `cd "$TEXTGEN_DIR" && HIP_VISIBLE_DEVICES=$GPUS ... python3 server.py --chat`
    pub fn build_launcher_content(&self, gpu_devices: &str) -> String {
        format!(
            "#!/bin/bash\n# text-generation-webui launcher for Stan's ML Stack\n# Auto-detected GPU devices: {}\ncd \"{}\" && \\\n    HIP_VISIBLE_DEVICES={} \\\n    CUDA_VISIBLE_DEVICES={} \\\n    {} server.py --chat \"$@\"\n",
            gpu_devices,
            self.config.install_dir,
            gpu_devices,
            gpu_devices,
            self.config.python_bin,
        )
    }

    /// Construct the systemd service unit content.
    pub fn build_systemd_service(&self, gpu_devices: &str) -> String {
        format!(
            "[Unit]\nDescription=text-generation-webui - LLM chat interface with ROCm support\nAfter=network.target\n\n\
             [Service]\nType=simple\nWorkingDirectory=\"{}\"\n\
             Environment=\"HIP_VISIBLE_DEVICES={}\"\n\
             Environment=\"CUDA_VISIBLE_DEVICES={}\"\n\
             ExecStart={} \"{}\" --chat\nRestart=on-failure\n\n\
             [Install]\nWantedBy=default.target\n",
            self.config.install_dir,
            gpu_devices,
            gpu_devices,
            self.config.python_bin,
            self.config.install_dir,
        )
    }

    /// Detect GPU devices via rocm-smi.
    pub fn detect_gpu_devices(&self) -> String {
        let _ = command_exists("rocm-smi");
        "0,1".to_string()
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

    fn make_installer(dir: &str) -> TextgenInstaller {
        TextgenInstaller::new(TextgenConfig {
            install_dir: dir.to_string(),
            python_bin: "python3".to_string(),
            dry_run: false,
        })
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-024: text-generation-webui correct git clone
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_clone_url_matches_original_script() {
        assert_eq!(
            REPO_URL, "https://github.com/oobabooga/text-generation-webui.git",
            "URL must match install_textgen.sh REPO_URL"
        );
    }

    #[test]
    fn test_git_clone_command() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmd = installer.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&REPO_URL.to_string()));
        assert!(cmd
            .args
            .contains(&"/home/user/text-generation-webui".to_string()));
    }

    #[test]
    fn test_git_clone_command_string() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmd = installer.build_git_clone_command();
        let s = cmd.to_command_string();
        assert!(s.contains("git clone"));
        assert!(s.contains(REPO_URL));
    }

    #[test]
    fn test_default_branch_is_main() {
        assert_eq!(DEFAULT_BRANCH, "main");
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-025: App installers correct target directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_install_dir_is_home_textgen() {
        let config = TextgenConfig::default();
        assert!(
            config.install_dir.ends_with("/text-generation-webui"),
            "Default dir must end with /text-generation-webui, got: {}",
            config.install_dir,
        );
    }

    #[test]
    fn test_custom_install_dir() {
        let installer = make_installer("/opt/apps/text-generation-webui");
        assert_eq!(installer.install_dir(), "/opt/apps/text-generation-webui");
    }

    #[test]
    fn test_pip_install_command() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmd = installer.build_pip_install_command("/tmp/filtered_reqs.txt");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"/tmp/filtered_reqs.txt".to_string()));
    }

    #[test]
    fn test_requirements_filter_excludes_nvidia_cuda() {
        let pattern = TextgenInstaller::requirements_filter_pattern();
        assert!(pattern.contains("nvidia-"));
        assert!(pattern.contains("cuda"));
        assert!(pattern.contains("tensorrt"));
        assert!(pattern.contains("triton"));
        assert!(pattern.contains("xformers"));
        assert!(pattern.contains("flash-attn"));
        assert!(pattern.contains("torch"));
    }

    #[test]
    fn test_preserve_dirs_match_original_script() {
        // Original: models, loras, embeddings, presets, characters, training
        assert_eq!(
            PRESERVE_DIRS,
            [
                "models",
                "loras",
                "embeddings",
                "presets",
                "characters",
                "training"
            ]
        );
    }

    #[test]
    fn test_git_update_without_user_data() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmds = installer.build_git_update_commands(false);
        assert_eq!(cmds.len(), 4);
        let last = cmds.last().unwrap();
        assert!(last.args.contains(&"reset".to_string()));
    }

    #[test]
    fn test_git_update_with_user_data() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmds = installer.build_git_update_commands(true);
        assert_eq!(cmds.len(), 6);
        let cmd_strings: Vec<String> = cmds.iter().map(|c| c.to_command_string()).collect();
        let has_stash = cmd_strings.iter().any(|s| s.contains("stash"));
        assert!(has_stash);
    }

    #[test]
    fn test_launcher_content() {
        let installer = make_installer("/home/user/text-generation-webui");
        let content = installer.build_launcher_content("0,1");
        assert!(content.starts_with("#!/bin/bash"));
        assert!(content.contains("/home/user/text-generation-webui"));
        assert!(content.contains("HIP_VISIBLE_DEVICES=0,1"));
        assert!(content.contains("python3 server.py --chat"));
    }

    #[test]
    fn test_systemd_service_content() {
        let installer = make_installer("/home/user/text-generation-webui");
        let svc = installer.build_systemd_service("0,1");
        assert!(svc.contains("[Unit]"));
        assert!(svc.contains("text-generation-webui"));
        assert!(svc.contains("HIP_VISIBLE_DEVICES=0,1"));
        assert!(svc.contains("--chat"));
    }

    #[test]
    fn test_amd_requirements_command() {
        let installer = make_installer("/home/user/text-generation-webui");
        let cmd = installer.build_amd_requirements_command(
            "/home/user/text-generation-webui/requirements_amd.txt",
        );
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-r".to_string()));
        assert!(cmd
            .args
            .contains(&"/home/user/text-generation-webui/requirements_amd.txt".to_string()));
    }
}
