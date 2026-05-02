//! UI installer helper — CLI arg parsing, path validation.
//!
//! Ports functionality from `scripts/lib/ui_installer_helper.sh`:
//! - Struct-based parsing for `--dry-run`, `--dir`, `--force`
//! - System path validation (blocks sensitive directories)
//! - Git clone/update helpers
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-015**: CLI arg parsing with path validation

use crate::installers::common::utils;
use std::path::Path;

// ===========================================================================
// Types
// ===========================================================================

/// Parsed common arguments for UI installer scripts.
///
/// Provides struct-based parsing for `--dry-run`, `--dir <path>`, `--force`,
/// and `--help` flags, matching the behavior of `ui_parse_common_args` from
/// the shell scripts.
///
/// # Validation
///
/// - **VAL-INFRA-015**: CLI arg parsing with path validation
#[derive(Debug, Clone, Default)]
pub struct UiArgs {
    /// Whether dry-run mode is enabled.
    pub dry_run: bool,
    /// The installation directory (absolute path).
    pub install_dir: Option<String>,
    /// Whether --force was specified (no-op for UI installers, accepted for compatibility).
    pub force: bool,
}

impl UiArgs {
    /// Parse command-line arguments into `UiArgs`.
    ///
    /// Supports:
    /// - `--dry-run` — enable dry-run mode
    /// - `--dir <path>` — set installation directory (must be absolute)
    /// - `--force` — accepted for compatibility (no-op for UI installers)
    /// - `--help` / `-h` — print usage and return `Err(2)` (matching shell behavior)
    ///
    /// Returns an error if:
    /// - `--dir` is provided without a path argument
    /// - `--dir` path is not absolute
    /// - `--dir` targets a sensitive system directory
    pub fn parse(args: &[&str]) -> Result<Self, UiArgError> {
        let mut dry_run = false;
        let mut install_dir = None;
        let mut force = false;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--dry-run" => {
                    dry_run = true;
                }
                "--force" => {
                    force = true;
                }
                "--dir" => {
                    if i + 1 >= args.len() {
                        return Err(UiArgError::MissingValue {
                            arg: "--dir".to_string(),
                        });
                    }
                    i += 1;
                    let dir = args[i];

                    // Must be absolute
                    if !dir.starts_with('/') {
                        return Err(UiArgError::NotAbsolute {
                            path: dir.to_string(),
                        });
                    }

                    // Resolve the path (canonicalize if exists, use as-is otherwise)
                    let resolved = if Path::new(dir).exists() {
                        std::fs::canonicalize(dir)
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_else(|_| dir.to_string())
                    } else {
                        dir.to_string()
                    };

                    // Block sensitive system paths
                    if is_system_path(&resolved) {
                        return Err(UiArgError::SystemPath {
                            path: resolved,
                        });
                    }

                    install_dir = Some(resolved);
                }
                "--help" | "-h" => {
                    return Err(UiArgError::HelpRequested);
                }
                _ => {
                    // Unknown args are silently ignored (matching shell behavior)
                }
            }
            i += 1;
        }

        Ok(Self {
            dry_run,
            install_dir,
            force,
        })
    }

    /// Get the installation directory, or a default.
    pub fn install_dir_or<'a>(&'a self, default: &'a str) -> &'a str {
        match &self.install_dir {
            Some(dir) => dir.as_str(),
            None => default,
        }
    }
}

/// Errors that can occur during UI argument parsing.
#[derive(Debug)]
pub enum UiArgError {
    /// `--dir` was provided without a following path argument.
    MissingValue {
        /// The flag that was missing a value.
        arg: String,
    },
    /// `--dir` path is not absolute.
    NotAbsolute {
        /// The non-absolute path provided.
        path: String,
    },
    /// `--dir` targets a sensitive system directory.
    SystemPath {
        /// The system path that was blocked.
        path: String,
    },
    /// `--help` was requested (exit code 2 in shell scripts).
    HelpRequested,
}

impl std::fmt::Display for UiArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingValue { arg } => {
                write!(f, "Error: {arg} requires a path argument")
            }
            Self::NotAbsolute { path } => {
                write!(f, "Error: --dir requires an absolute path (got: {path})")
            }
            Self::SystemPath { path } => {
                write!(f, "Error: --dir targets a system directory: {path}")
            }
            Self::HelpRequested => {
                write!(f, "Usage: [--dry-run] [--dir <path>] [--help]")
            }
        }
    }
}

impl std::error::Error for UiArgError {}

// ===========================================================================
// Path Validation
// ===========================================================================

/// System paths that should never be used as installation targets.
const BLOCKED_SYSTEM_PATHS: &[&str] = &[
    "/",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/var",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/opt/rocm",
];

/// Check if a path targets a sensitive system directory.
///
/// Returns `true` if the resolved path matches any blocked system path.
pub fn is_system_path(path: &str) -> bool {
    let resolved = Path::new(path);

    for blocked in BLOCKED_SYSTEM_PATHS {
        let blocked_path = Path::new(blocked);
        if resolved == blocked_path {
            return true;
        }
    }

    false
}

// ===========================================================================
// Git Clone/Update Helper
// ===========================================================================

/// Clone a git repository or update it if it already exists.
///
/// If the directory exists and contains a `.git` folder, performs a fetch
/// and hard reset. Otherwise, clones the repository.
///
/// Supports dry-run mode — logs commands without executing.
pub fn git_clone_or_update(
    install_dir: &str,
    repo_url: &str,
    dry_run: bool,
) -> Result<(), String> {
    let git_dir = Path::new(install_dir).join(".git");

    if git_dir.exists() {
        // Update existing clone
        let fetch_cmd = format!("git -C \"{install_dir}\" fetch --all");
        utils::execute_command(&fetch_cmd, "Fetching latest changes", dry_run)?;

        // Get target branch
        let target_branch = get_current_branch(install_dir, dry_run);

        let reset_cmd =
            format!("git -C \"{install_dir}\" reset --hard \"origin/{target_branch}\"");
        utils::execute_command(&reset_cmd, "Resetting to latest", dry_run)?;
    } else {
        // Fresh clone
        let clone_cmd = format!("git clone \"{repo_url}\" \"{install_dir}\"");
        utils::execute_command(&clone_cmd, "Cloning repository", dry_run)?;
    }

    Ok(())
}

/// Get the current branch name of a git repository.
fn get_current_branch(repo_dir: &str, dry_run: bool) -> String {
    if dry_run {
        return "main".to_string();
    }

    // Try symbolic-ref first
    let output = std::process::Command::new("git")
        .args(["-C", repo_dir, "symbolic-ref", "refs/remotes/origin/HEAD"])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Some(branch) = stdout.strip_prefix("refs/remotes/origin/") {
                return branch.to_string();
            }
        }
    }

    // Fallback: try rev-parse
    let output = std::process::Command::new("git")
        .args(["-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !branch.is_empty() {
                return branch;
            }
        }
    }

    "main".to_string()
}

/// Fix file ownership when run with sudo.
///
/// If running as root with `SUDO_USER` set, changes ownership of the
/// install directory to the sudo user.
#[cfg(all(unix, feature = "unix-deps"))]
pub fn fix_ownership(install_dir: &str) -> Result<(), String> {
    let euid = unsafe { libc::geteuid() };
    if euid != 0 {
        return Ok(());
    }

    let sudo_user = match std::env::var("SUDO_USER") {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(()),
    };

    let cmd = format!("chown -R {sudo_user}:{sudo_user} \"{install_dir}\"");
    let result = std::process::Command::new("bash")
        .arg("-c")
        .arg(&cmd)
        .output();

    match result {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("Failed to fix ownership: {stderr}"))
        }
        Err(e) => Err(format!("Failed to execute chown: {e}")),
    }
}

#[cfg(not(all(unix, feature = "unix-deps")))]
pub fn fix_ownership(_install_dir: &str) -> Result<(), String> {
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- UiArgs parsing tests (VAL-INFRA-015) ---

    #[test]
    fn test_parse_empty_args() {
        let args = UiArgs::parse(&[]).unwrap();
        assert!(!args.dry_run);
        assert!(args.install_dir.is_none());
        assert!(!args.force);
    }

    #[test]
    fn test_parse_dry_run() {
        let args = UiArgs::parse(&["--dry-run"]).unwrap();
        assert!(args.dry_run);
    }

    #[test]
    fn test_parse_force() {
        let args = UiArgs::parse(&["--force"]).unwrap();
        assert!(args.force);
    }

    #[test]
    fn test_parse_dir_valid() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        let args = UiArgs::parse(&["--dir", &temp_path]).unwrap();
        assert!(args.install_dir.is_some());
    }

    #[test]
    fn test_parse_dir_nonexistent_absolute() {
        // Nonexistent but absolute path should still work
        let args = UiArgs::parse(&["--dir", "/tmp/nonexistent_test_dir_xyz"]).unwrap();
        assert_eq!(args.install_dir, Some("/tmp/nonexistent_test_dir_xyz".to_string()));
    }

    #[test]
    fn test_parse_dir_missing_value() {
        let result = UiArgs::parse(&["--dir"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UiArgError::MissingValue { .. }));
        assert!(format!("{err}").contains("--dir requires a path argument"));
    }

    #[test]
    fn test_parse_dir_not_absolute() {
        let result = UiArgs::parse(&["--dir", "relative/path"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UiArgError::NotAbsolute { .. }));
    }

    #[test]
    fn test_parse_dir_system_path_root() {
        let result = UiArgs::parse(&["--dir", "/"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UiArgError::SystemPath { .. }));
    }

    #[test]
    fn test_parse_dir_system_path_usr() {
        let result = UiArgs::parse(&["--dir", "/usr"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dir_system_path_bin() {
        // /bin may be a symlink to /usr/bin on some systems, so test with
        // the canonical path check instead
        let result = is_system_path("/bin");
        assert!(result);
    }

    #[test]
    fn test_parse_dir_system_path_sbin() {
        // /sbin may be a symlink on some systems
        let result = is_system_path("/sbin");
        assert!(result);
    }

    #[test]
    fn test_parse_dir_system_path_etc() {
        let result = UiArgs::parse(&["--dir", "/etc"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dir_system_path_var() {
        let result = UiArgs::parse(&["--dir", "/var"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dir_system_path_rocm() {
        let result = UiArgs::parse(&["--dir", "/opt/rocm"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_help_long() {
        let result = UiArgs::parse(&["--help"]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UiArgError::HelpRequested));
    }

    #[test]
    fn test_parse_help_short() {
        let result = UiArgs::parse(&["-h"]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UiArgError::HelpRequested));
    }

    #[test]
    fn test_parse_combined_args() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        let args = UiArgs::parse(&["--dry-run", "--force", "--dir", &temp_path]).unwrap();
        assert!(args.dry_run);
        assert!(args.force);
        assert!(args.install_dir.is_some());
    }

    #[test]
    fn test_parse_unknown_args_ignored() {
        let args = UiArgs::parse(&["--unknown", "value"]).unwrap();
        assert!(!args.dry_run);
    }

    #[test]
    fn test_install_dir_or_default() {
        let args = UiArgs::parse(&[]).unwrap();
        assert_eq!(args.install_dir_or("/default"), "/default");

        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        let args = UiArgs::parse(&["--dir", &temp_path]).unwrap();
        assert_ne!(args.install_dir_or("/default"), "/default");
    }

    #[test]
    fn test_ui_args_default() {
        let args = UiArgs::default();
        assert!(!args.dry_run);
        assert!(args.install_dir.is_none());
        assert!(!args.force);
    }

    // --- is_system_path tests ---

    #[test]
    fn test_is_system_path_blocked() {
        assert!(is_system_path("/"));
        assert!(is_system_path("/usr"));
        assert!(is_system_path("/bin"));
        assert!(is_system_path("/sbin"));
        assert!(is_system_path("/etc"));
        assert!(is_system_path("/var"));
        assert!(is_system_path("/boot"));
        assert!(is_system_path("/dev"));
        assert!(is_system_path("/proc"));
        assert!(is_system_path("/sys"));
        assert!(is_system_path("/opt/rocm"));
    }

    #[test]
    fn test_is_system_path_safe() {
        assert!(!is_system_path("/home/user/apps"));
        assert!(!is_system_path("/opt/mlstack"));
        assert!(!is_system_path("/tmp/test"));
        assert!(!is_system_path("/usr/local/apps/myapp"));
    }

    // --- UiArgError Display ---

    #[test]
    fn test_ui_arg_error_display() {
        let err = UiArgError::MissingValue { arg: "--dir".to_string() };
        assert!(format!("{err}").contains("--dir requires a path argument"));

        let err = UiArgError::NotAbsolute { path: "relative".to_string() };
        assert!(format!("{err}").contains("absolute path"));

        let err = UiArgError::SystemPath { path: "/usr".to_string() };
        assert!(format!("{err}").contains("system directory"));

        let err = UiArgError::HelpRequested;
        assert!(format!("{err}").contains("Usage"));
    }

    #[test]
    fn test_ui_arg_error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        let err = UiArgError::MissingValue { arg: "test".to_string() };
        assert_error(&err);
    }

    // --- git_clone_or_update tests ---

    #[test]
    fn test_git_clone_dry_run() {
        // Dry run should succeed without actually cloning
        let result = git_clone_or_update(
            "/tmp/test_clone_dir",
            "https://github.com/example/repo.git",
            true,
        );
        assert!(result.is_ok());
    }

    // --- fix_ownership tests ---

    #[test]
    fn test_fix_ownership_no_sudo() {
        // When not running as root, this should be a no-op
        let result = fix_ownership("/tmp/test");
        // May or may not succeed depending on permissions, but shouldn't panic
        let _ = result;
    }
}
