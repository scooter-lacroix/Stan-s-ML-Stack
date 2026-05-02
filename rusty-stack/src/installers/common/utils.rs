//! Common utility functions — color-aware output, command existence checks, print helpers.
//!
//! Ports functionality from `scripts/common_utils.sh`:
//! - Color constants that auto-disable for non-TTY and `NO_COLOR`
//! - `command_exists()` utility function
//! - Print helpers (header, section, step, success, warning, error)
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-016**: Color-aware output auto-disabling for non-TTY, command_exists utility

use std::fmt;
use std::io::IsTerminal;
use std::process::Command;

// ===========================================================================
// Color Support
// ===========================================================================

/// ANSI color constants that auto-disable for non-TTY and `NO_COLOR`.
///
/// When color is disabled, all constants are empty strings.
///
/// # Validation
///
/// - **VAL-INFRA-016**: Color-aware terminal output auto-disabling for non-TTY
#[derive(Debug, Clone, Copy)]
pub struct Colors {
    pub red: &'static str,
    pub green: &'static str,
    pub yellow: &'static str,
    pub blue: &'static str,
    pub magenta: &'static str,
    pub cyan: &'static str,
    pub bold: &'static str,
    pub reset: &'static str,
}

/// Color constants with ANSI escape codes.
const COLOR_ENABLED: Colors = Colors {
    red: "\x1b[0;31m",
    green: "\x1b[0;32m",
    yellow: "\x1b[0;33m",
    blue: "\x1b[0;34m",
    magenta: "\x1b[0;35m",
    cyan: "\x1b[0;36m",
    bold: "\x1b[1m",
    reset: "\x1b[0m",
};

/// Color constants when color is disabled (all empty).
const COLOR_DISABLED: Colors = Colors {
    red: "",
    green: "",
    yellow: "",
    blue: "",
    magenta: "",
    cyan: "",
    bold: "",
    reset: "",
};

/// Determine if color output should be used.
///
/// Returns `false` if:
/// - `NO_COLOR` environment variable is set
/// - stderr is not connected to a TTY
pub fn color_enabled() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    std::io::stderr().is_terminal()
}

/// Get the appropriate color constants based on terminal capabilities.
pub fn get_colors() -> &'static Colors {
    if color_enabled() {
        &COLOR_ENABLED
    } else {
        &COLOR_DISABLED
    }
}

// ===========================================================================
// Command Existence
// ===========================================================================

/// Check if a command exists in the system PATH.
///
/// Uses `which` on Unix systems to check for command availability.
/// Returns `true` if the command is found, `false` otherwise.
///
/// # Validation
///
/// - **VAL-INFRA-016**: command_exists utility function
///
/// # Examples
///
/// ```no_run
/// use rusty_stack::installers::common::utils::command_exists;
///
/// if command_exists("python3") {
///     println!("Python 3 is available");
/// }
/// ```
pub fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ===========================================================================
// Print Helpers
// ===========================================================================

/// Print a formatted header box to stderr.
///
/// Displays a box-drawing header with the given title, centered.
/// All output goes to stderr to avoid breaking command captures.
pub fn print_header(title: &str) {
    let c = get_colors();
    let width = 57;
    let inner = width - 2; // space for ║ on each side
    let padded = format!("=== {title} ===");
    let centered = center_str(&padded, inner);

    let border_line = "═".repeat(inner);
    let empty_line = " ".repeat(inner);

    eprintln!();
    eprintln!("{}╔{border_line}╗{}", c.cyan, c.reset);
    eprintln!("{}║{empty_line}║{}", c.cyan, c.reset);
    eprintln!("{}║{centered}║{}", c.cyan, c.reset);
    eprintln!("{}║{empty_line}║{}", c.cyan, c.reset);
    eprintln!("{}╚{border_line}╝{}", c.cyan, c.reset);
    eprintln!();
}

/// Print a section separator to stderr.
pub fn print_section(title: &str) {
    let c = get_colors();
    eprintln!();
    eprintln!(
        "{}┌─────────────────────────────────────────────────────────┐{}",
        c.blue, c.reset
    );
    eprintln!("{}│ {}{}{}", c.blue, c.bold, title, c.reset);
    eprintln!(
        "{}└─────────────────────────────────────────────────────────┘{}",
        c.blue, c.reset
    );
}

/// Print a step indicator to stderr.
pub fn print_step(message: &str) {
    let c = get_colors();
    eprintln!("{}➤ {}{}", c.magenta, message, c.reset);
}

/// Print a success message to stderr.
pub fn print_success(message: &str) {
    let c = get_colors();
    eprintln!("{}✓ {}{}", c.green, message, c.reset);
}

/// Print a warning message to stderr.
pub fn print_warning(message: &str) {
    let c = get_colors();
    eprintln!("{}⚠ {}{}", c.yellow, message, c.reset);
}

/// Print an error message to stderr.
pub fn print_error(message: &str) {
    let c = get_colors();
    eprintln!("{}✗ {}{}", c.red, message, c.reset);
}

/// Execute a command with dry-run support.
///
/// When `dry_run` is `true`, logs the command without executing it.
/// Returns `Ok(())` on success or `Err` with the error description.
pub fn execute_command(cmd: &str, description: &str, dry_run: bool) -> Result<(), String> {
    let c = get_colors();
    if dry_run {
        eprintln!("{}[DRY-RUN]{} {}: ", c.yellow, c.reset, description);
        eprintln!("  {}{}{}", c.bold, cmd, c.reset);
        return Ok(());
    }

    print_step(&format!("{description}..."));

    let result = Command::new("bash").arg("-c").arg(cmd).output();

    match result {
        Ok(output) if output.status.success() => {
            print_success(&format!("Done: {description}"));
            Ok(())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            print_error(&format!("Failed: {description}"));
            if !stderr.is_empty() {
                eprintln!("  {stderr}");
            }
            Err(format!("Command failed: {cmd}"))
        }
        Err(e) => {
            print_error(&format!("Failed: {description}: {e}"));
            Err(format!("Command execution error: {e}"))
        }
    }
}

// ===========================================================================
// Python Package Manager Detection (VAL-INFRA-018)
// ===========================================================================

/// Detected Python package manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythonPkgManager {
    /// uv package manager (preferred).
    Uv,
    /// pip (fallback).
    Pip,
}

impl fmt::Display for PythonPkgManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PythonPkgManager::Uv => write!(f, "uv"),
            PythonPkgManager::Pip => write!(f, "pip"),
        }
    }
}

/// Detect the available Python package manager.
///
/// Prefers `uv` if available, falls back to `pip`.
/// Respects `UV_PYTHON` environment variable.
///
/// # Validation
///
/// - **VAL-INFRA-018**: Python package manager detection (uv/pip)
pub fn detect_python_pkg_manager() -> PythonPkgManager {
    if command_exists("uv") {
        PythonPkgManager::Uv
    } else {
        PythonPkgManager::Pip
    }
}

/// Build a pip install command prefix using the detected package manager.
///
/// Returns the command parts (program + base args) for installing packages.
pub fn pip_install_prefix(python_bin: &str) -> Vec<String> {
    let mgr = detect_python_pkg_manager();
    match mgr {
        PythonPkgManager::Uv => {
            let mut args = vec!["uv".to_string(), "pip".to_string(), "install".to_string()];
            // Respect UV_PYTHON env var
            if std::env::var("UV_PYTHON").is_err() {
                args.push("--python".to_string());
                args.push(python_bin.to_string());
            }
            args
        }
        PythonPkgManager::Pip => {
            vec![
                python_bin.to_string(),
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
            ]
        }
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Center a string within a given width, padding with spaces.
fn center_str(s: &str, width: usize) -> String {
    if s.len() >= width {
        return s.to_string();
    }
    let left = (width - s.len()) / 2;
    let right = width - s.len() - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Color tests (VAL-INFRA-016) ---

    #[test]
    fn test_colors_enabled_has_escapes() {
        let c = &COLOR_ENABLED;
        assert!(!c.red.is_empty());
        assert!(!c.green.is_empty());
        assert!(!c.reset.is_empty());
        assert!(c.red.starts_with('\x1b'));
    }

    #[test]
    fn test_colors_disabled_is_empty() {
        let c = &COLOR_DISABLED;
        assert!(c.red.is_empty());
        assert!(c.green.is_empty());
        assert!(c.yellow.is_empty());
        assert!(c.blue.is_empty());
        assert!(c.magenta.is_empty());
        assert!(c.cyan.is_empty());
        assert!(c.bold.is_empty());
        assert!(c.reset.is_empty());
    }

    #[test]
    fn test_get_colors_returns_valid_ref() {
        let _colors = get_colors();
        // Just verify it doesn't panic
    }

    // --- command_exists tests (VAL-INFRA-016) ---

    #[test]
    fn test_command_exists_true() {
        // bash should exist on any Linux system
        assert!(command_exists("bash"));
    }

    #[test]
    fn test_command_exists_false() {
        assert!(!command_exists("nonexistent_command_xyz_12345"));
    }

    #[test]
    fn test_command_exists_which() {
        // which is used by the implementation itself
        assert!(command_exists("which"));
    }

    // --- Print helper tests ---

    #[test]
    fn test_print_header_no_panic() {
        // Just verify no panic
        print_header("Test Title");
    }

    #[test]
    fn test_print_section_no_panic() {
        print_section("Test Section");
    }

    #[test]
    fn test_print_step_no_panic() {
        print_step("Test step");
    }

    #[test]
    fn test_print_success_no_panic() {
        print_success("Test success");
    }

    #[test]
    fn test_print_warning_no_panic() {
        print_warning("Test warning");
    }

    #[test]
    fn test_print_error_no_panic() {
        print_error("Test error");
    }

    // --- execute_command tests ---

    #[test]
    fn test_execute_command_dry_run() {
        let result = execute_command("echo hello", "test dry run", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_command_success() {
        let result = execute_command("true", "test success", false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_command_failure() {
        let result = execute_command("false", "test failure", false);
        assert!(result.is_err());
    }

    // --- PythonPkgManager tests (VAL-INFRA-018) ---

    #[test]
    fn test_detect_python_pkg_manager() {
        // Should return either Uv or Pip without panicking
        let mgr = detect_python_pkg_manager();
        assert!(mgr == PythonPkgManager::Uv || mgr == PythonPkgManager::Pip);
    }

    #[test]
    fn test_python_pkg_manager_display() {
        assert_eq!(format!("{}", PythonPkgManager::Uv), "uv");
        assert_eq!(format!("{}", PythonPkgManager::Pip), "pip");
    }

    #[test]
    fn test_pip_install_prefix_returns_nonempty() {
        let prefix = pip_install_prefix("python3");
        assert!(!prefix.is_empty());
        // First element should be either "uv" or "python3"
        assert!(prefix[0] == "uv" || prefix[0] == "python3");
    }

    // --- Center string helper ---

    #[test]
    fn test_center_str_short() {
        let result = center_str("hi", 10);
        assert_eq!(result.len(), 10);
        assert_eq!(result.trim(), "hi");
    }

    #[test]
    fn test_center_str_exact() {
        let result = center_str("hello", 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_center_str_longer() {
        let result = center_str("hello world", 5);
        assert_eq!(result, "hello world");
    }
}
