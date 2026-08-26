//! ComfyUI installer — ports `scripts/install_comfyui.sh`.
//!
//! Clones the ComfyUI repository, installs Python dependencies (excluding
//! torch packages), and creates launcher shim. ComfyUI depends on PyTorch.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-023**: ComfyUI correct git clone and pip install
//! - **VAL-INSTALL-025**: App installers correct target directory
//! - **VAL-INSTALL-047**: ComfyUI declares dependency on PyTorch

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

/// Configuration for the ComfyUI installer.
#[derive(Debug, Clone)]
pub struct ComfyuiConfig {
    /// Target installation directory.
    pub install_dir: String,
    /// Python binary to use.
    pub python_bin: String,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
}

impl Default for ComfyuiConfig {
    fn default() -> Self {
        Self {
            install_dir: default_install_dir(),
            python_bin: "python3".to_string(),
            dry_run: false,
        }
    }
}

/// ComfyUI Git repository URL.
pub const REPO_URL: &str = "https://github.com/comfyanonymous/ComfyUI.git";

/// Default branch for ComfyUI.
pub const DEFAULT_BRANCH: &str = "master";

/// Default web port.
pub const WEB_PORT: u16 = 8188;

/// Packages to exclude from requirements.txt filtering.
/// The original script excludes: torch, torchvision, torchaudio, torchsde, sentencepiece.
pub const EXCLUDED_PACKAGES: &[&str] = &[
    "torch",
    "torchvision",
    "torchaudio",
    "torchsde",
    "sentencepiece",
];

/// Rusty-owned pinned ComfyUI Node Manager package.
///
/// Upstream instructs `pip install -U --pre comfyui-manager`; a floating or
/// pre-channel install inside the managed environment would violate exact-pin
/// ownership, so the version is pinned here and bumped consciously.
pub const COMFYUI_MANAGER_PACKAGE: &str = "comfyui-manager==4.2.2";

/// Append Rusty-managed extras to a filtered requirements body.
///
/// Idempotent: skips insertion when any comfyui-manager requirement line is
/// already present (e.g. re-running against an already-extended body).
pub fn inject_managed_extras(filtered_body: &str) -> String {
    let has_manager = filtered_body.lines().any(|line| {
        line.trim()
            .to_ascii_lowercase()
            .starts_with("comfyui-manager")
    });
    if has_manager {
        return filtered_body.to_string();
    }
    let mut body = String::from(filtered_body);
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    body.push_str("# Rusty-owned extra: ComfyUI Node Manager (upstream asks for `pip install -U --pre comfyui-manager`); pinned per managed-env ownership\n");
    body.push_str(COMFYUI_MANAGER_PACKAGE);
    body.push('\n');
    body
}

/// Marker file storing the requirements.txt fingerprint captured at the time
/// dependencies were last installed by Rusty.
pub const DEPS_BASELINE_FILE: &str = ".mlstack-requirements.sha256";

/// Exit code used by the launcher when ComfyUI sources drifted past the last
/// managed dependency install (78 = EX_CONFIG convention).
pub const DRIFT_EXIT_CODE: u32 = 78;

/// Bash snippet enforcing that `requirements.txt` still matches the baseline
/// recorded at managed-install time. Out-of-band repo updates change the file;
/// without re-running Rusty's dependency step, imports inside newer code fail
/// at launch with missing modules (e.g. sqlalchemy/alembic for the asset DB).
pub fn drift_guard_snippet(install_dir: &str) -> String {
    format!(
        r#"# Drift guard: requirements changed since Rusty's last managed install?
if [ -f "{dir}/requirements.txt" ]; then
    CUR_REQS=$(sha256sum "{dir}/requirements.txt" 2>/dev/null | cut -d' ' -f1)
    OLD_REQS=$(cat "{dir}/{baseline}" 2>/dev/null)
    if [ -n "$CUR_REQS" ] && [ "$CUR_REQS" != "$OLD_REQS" ]; then
        echo "ComfyUI sources drifted since the last managed install (requirements fingerprint mismatch)." >&2
        echo "Dependencies were NOT refreshed; launch refuses to avoid import-time breakage." >&2
        echo "Fix: run Rusty Stack -> Reinstall 'ComfyUI' (re-syncs deps + refreshes this guard), then relaunch." >&2
        exit {code}
    fi
fi
"#,
        dir = install_dir,
        baseline = DEPS_BASELINE_FILE,
        code = DRIFT_EXIT_CODE
    )
}

/// Directories to preserve during git update.
pub const PRESERVE_DIRS: &[&str] = &["models", "input", "output", "user"];

/// Default ComfyUI installation directory.
pub fn default_install_dir() -> String {
    format!("{}/ComfyUI", std::env::var("HOME").unwrap_or_default())
}

/// The ComfyUI installer.
pub struct ComfyuiInstaller {
    config: ComfyuiConfig,
}

impl ComfyuiInstaller {
    /// Create a new ComfyUI installer with the given config.
    pub fn new(config: ComfyuiConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ComfyuiConfig::default())
    }

    // -----------------------------------------------------------------------
    // Dependency declaration (VAL-INSTALL-047)
    // -----------------------------------------------------------------------

    /// ComfyUI depends on PyTorch with ROCm support.
    pub fn dependencies() -> Vec<&'static str> {
        vec!["pytorch"]
    }

    /// Check if PyTorch is available (via import check command).
    pub fn build_pytorch_check_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), "import torch".to_string()],
            env: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Command construction (VAL-INSTALL-023, VAL-INSTALL-025)
    // -----------------------------------------------------------------------

    /// Construct the git clone command for ComfyUI.
    ///
    /// The original script runs:
    /// `git clone "https://github.com/comfyanonymous/ComfyUI.git" "$COMFYUI_DIR"`
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
    /// 2. Detect default branch
    /// 3. `git -C "$DIR" checkout "$DEFAULT_BRANCH"`
    /// 4. `git -C "$DIR" fetch --all`
    /// 5. `git -C "$DIR" stash push -u` (if user data exists)
    /// 6. `git -C "$DIR" reset --hard "origin/$DEFAULT_BRANCH"`
    /// 7. `git -C "$DIR" stash pop` (if user data was stashed)
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
    /// The original script filters out torch/torchvision/torchaudio/torchsde/sentencepiece
    /// from requirements.txt then runs:
    /// `python3 -m pip install -r "$FILTERED_REQS"`
    pub fn build_pip_install_command(&self, requirements_path: &str) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--break-system-packages".to_string(),
                "-r".to_string(),
                requirements_path.to_string(),
            ],
            env: vec![],
        }
    }

    /// Get the grep filter pattern for excluding packages.
    /// Returns the regex pattern used to filter requirements.txt.
    pub fn requirements_filter_pattern() -> String {
        EXCLUDED_PACKAGES
            .iter()
            .map(|p| format!("^{}", p))
            .collect::<Vec<_>>()
            .join("|")
    }

    /// Construct the launcher shim content.
    ///
    /// Creates a shell script at `~/.local/bin/comfy` that runs:
    /// `cd "$COMFYUI_DIR" && HIP_VISIBLE_DEVICES=$GPUS ... python3 main.py --enable-manager`
    /// preceded by the requirements-drift guard (see [`drift_guard_snippet`]).
    pub fn build_launcher_content(&self, gpu_devices: &str) -> String {
        format!(
            "#!/bin/bash\n# ComfyUI launcher for Stan's ML Stack\n# Auto-detected GPU devices: {}\n{}\ncd \"{}\" && \\\n    HIP_VISIBLE_DEVICES={} \\\n    CUDA_VISIBLE_DEVICES={} \\\n    {} main.py --enable-manager \"$@\"\n",
            gpu_devices,
            drift_guard_snippet(&self.config.install_dir),
            self.config.install_dir,
            gpu_devices,
            gpu_devices,
            self.config.python_bin,
        )
    }

    /// Construct a command that records the requirements fingerprint baseline
    /// after a successful managed dependency install. Launcher drift-guard
    /// comparisons run against this marker.
    pub fn build_record_requirements_baseline_command(&self) -> ShellCommand {
        let dir = &self.config.install_dir;
        ShellCommand {
            program: "bash".to_string(),
            args: vec![
                "-c".to_string(),
                format!(
                    "cd \"{dir}\" && sha256sum requirements.txt | cut -d' ' -f1 > \"{dir}/{tmp}\" && mv \"{dir}/{tmp}\" \"{dir}/{baseline}\"",
                    dir = dir,
                    tmp = format!("{DEPS_BASELINE_FILE}.tmp"),
                    baseline = DEPS_BASELINE_FILE,
                ),
            ],
            env: vec![],
        }
    }

    /// Construct the systemd service unit content.
    ///
    /// ExecStart routes through the launcher shim so the drift guard applies
    /// to systemd launches identically to manual ones.
    pub fn build_systemd_service(&self, gpu_devices: &str, home: &str) -> String {
        format!(
            "[Unit]\nDescription=ComfyUI - Node-based UI for Stable Diffusion\nAfter=network.target\n\n\
             [Service]\nType=simple\nWorkingDirectory=\"{}\"\n\
             Environment=\"HIP_VISIBLE_DEVICES={}\"\n\
             Environment=\"CUDA_VISIBLE_DEVICES={}\"\n\
             ExecStart={}/.local/bin/comfy\nRestart=on-failure\nRestartPreventExitStatus=78\n\n\
             [Install]\nWantedBy=default.target\n",
            self.config.install_dir,
            gpu_devices,
            gpu_devices,
            home,
        )
    }

    /// Detect GPU devices — the canonical, iGPU-filtered discrete list.
    ///
    /// Routes through `crate::installer::detect_gpu_list` (Stage 1) so the
    /// integrated GPU is never included and no dGPU is missed. Baked into the
    /// launcher shim + systemd unit as `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`.
    pub fn detect_gpu_devices(&self) -> String {
        crate::installer::detect_gpu_list()
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

    fn make_installer(dir: &str) -> ComfyuiInstaller {
        ComfyuiInstaller::new(ComfyuiConfig {
            install_dir: dir.to_string(),
            python_bin: "python3".to_string(),
            dry_run: false,
        })
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-023: ComfyUI correct git clone and pip install
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_clone_url_matches_original_script() {
        assert_eq!(
            REPO_URL, "https://github.com/comfyanonymous/ComfyUI.git",
            "URL must match install_comfyui.sh REPO_URL"
        );
    }

    #[test]
    fn test_git_clone_command() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmd = installer.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"clone".to_string()));
        assert!(cmd.args.contains(&REPO_URL.to_string()));
        assert!(cmd.args.contains(&"/home/user/ComfyUI".to_string()));
    }

    #[test]
    fn test_git_clone_command_string() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmd = installer.build_git_clone_command();
        let s = cmd.to_command_string();
        assert!(s.contains("git clone"));
        assert!(s.contains(REPO_URL));
    }

    #[test]
    fn test_pip_install_command() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmd = installer.build_pip_install_command("/tmp/filtered_reqs.txt");
        assert_eq!(cmd.program, "python3");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.contains(&"pip".to_string()));
        assert!(cmd.args.contains(&"install".to_string()));
        assert!(cmd.args.contains(&"-r".to_string()));
        assert!(cmd.args.contains(&"/tmp/filtered_reqs.txt".to_string()));
    }

    #[test]
    fn test_requirements_filter_excludes_torch_packages() {
        let pattern = ComfyuiInstaller::requirements_filter_pattern();
        // Pattern should match all excluded packages
        for pkg in EXCLUDED_PACKAGES {
            assert!(pattern.contains(pkg), "Pattern should contain '{pkg}'");
        }
    }

    #[test]
    fn test_excluded_packages_match_original_script() {
        // Original script excludes: torch, torchvision, torchaudio, torchsde, sentencepiece
        assert_eq!(
            EXCLUDED_PACKAGES,
            [
                "torch",
                "torchvision",
                "torchaudio",
                "torchsde",
                "sentencepiece"
            ]
        );
    }

    #[test]
    fn test_default_branch_is_master() {
        // ComfyUI uses 'master' as default branch
        assert_eq!(DEFAULT_BRANCH, "master");
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-025: App installers correct target directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_install_dir_is_home_comfyui() {
        let config = ComfyuiConfig::default();
        assert!(
            config.install_dir.ends_with("/ComfyUI"),
            "Default dir must end with /ComfyUI, got: {}",
            config.install_dir,
        );
    }

    #[test]
    fn test_custom_install_dir() {
        let installer = make_installer("/opt/apps/ComfyUI");
        assert_eq!(installer.install_dir(), "/opt/apps/ComfyUI");
    }

    // -----------------------------------------------------------------------
    // VAL-INSTALL-047: ComfyUI declares dependency on PyTorch
    // -----------------------------------------------------------------------

    #[test]
    fn test_declares_pytorch_dependency() {
        let deps = ComfyuiInstaller::dependencies();
        assert!(
            deps.contains(&"pytorch"),
            "ComfyUI must declare pytorch as a dependency"
        );
    }

    #[test]
    fn test_git_update_without_user_data() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmds = installer.build_git_update_commands(false);
        // Should have: set-head, checkout, fetch, reset (4 commands, no stash)
        assert_eq!(cmds.len(), 4);
        // Last command should be reset
        let last = cmds.last().unwrap();
        assert!(last.args.contains(&"reset".to_string()));
    }

    #[test]
    fn test_git_update_with_user_data() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmds = installer.build_git_update_commands(true);
        // Should have: set-head, checkout, fetch, stash, reset, stash pop (6 commands)
        assert_eq!(cmds.len(), 6);
        // Should contain stash commands
        let cmd_strings: Vec<String> = cmds.iter().map(|c| c.to_command_string()).collect();
        let has_stash = cmd_strings.iter().any(|s| s.contains("stash"));
        assert!(
            has_stash,
            "Should contain stash commands when user data exists"
        );
    }

    #[test]
    fn test_launcher_content() {
        let installer = make_installer("/home/user/ComfyUI");
        let content = installer.build_launcher_content("0,1");
        assert!(content.starts_with("#!/bin/bash"));
        assert!(content.contains("/home/user/ComfyUI"));
        assert!(content.contains("HIP_VISIBLE_DEVICES=0,1"));
        assert!(content.contains("python3 main.py --enable-manager"));
    }

    #[test]
    fn test_launcher_enforces_requirements_drift_guard() {
        let installer = make_installer("/home/user/ComfyUI");
        let content = installer.build_launcher_content("0,1");
        assert!(content.contains(DEPS_BASELINE_FILE));
        assert!(content.contains(&format!("exit {DRIFT_EXIT_CODE}")));
        assert!(content.contains("Reinstall 'ComfyUI'"));
    }

    #[test]
    fn test_managed_extras_inject_manager_pin_idempotently() {
        let once = inject_managed_extras("comfy-kitchen==0.2.31\n");
        assert!(once.contains(COMFYUI_MANAGER_PACKAGE));
        assert_eq!(once.lines().last().unwrap(), COMFYUI_MANAGER_PACKAGE);
        let twice = inject_managed_extras(&once);
        assert_eq!(
            twice.matches(COMFYUI_MANAGER_PACKAGE).count(),
            1,
            "re-injection must stay idempotent"
        );
        // guard rails: exact pin, no float ranges, no nvidia channel drift
        assert!(COMFYUI_MANAGER_PACKAGE.starts_with("comfyui-manager=="));
    }

    #[test]
    fn test_inject_respects_existing_manager_line() {
        let body = "# manual\ncomfyui-manager==9.9.9\n";
        assert!(inject_managed_extras(body).contains("comfyui-manager==9.9.9"));
        assert!(!inject_managed_extras(body).contains("==4.2.2"));
    }

    #[test]
    fn test_baseline_command_writes_atomic_marker() {
        let installer = make_installer("/home/user/ComfyUI");
        let cmd = installer.build_record_requirements_baseline_command();
        assert_eq!(cmd.program, "bash");
        let joined = cmd.args.join(" ");
        assert!(joined.contains("sha256sum requirements.txt"));
        assert!(joined.contains(DEPS_BASELINE_FILE));
        assert!(joined.contains(".tmp")); // atomic tmp+rename
    }

    #[test]
    fn test_preserve_dirs_match_original_script() {
        // Original script preserves: models, input, output, user
        assert_eq!(PRESERVE_DIRS, ["models", "input", "output", "user"]);
    }

    #[test]
    fn test_systemd_service_content() {
        let installer = make_installer("/home/user/ComfyUI");
        let svc = installer.build_systemd_service("0,1", "/home/user");
        assert!(svc.contains("[Unit]"));
        assert!(svc.contains("ComfyUI"));
        assert!(svc.contains("HIP_VISIBLE_DEVICES=0,1"));
        // routes through the shim so the drift guard covers systemd launches
        assert!(svc.contains("ExecStart=/home/user/.local/bin/comfy"));
        // drift-guard exit must NOT trigger restart churn
        assert!(svc.contains("RestartPreventExitStatus=78"));
    }
}
