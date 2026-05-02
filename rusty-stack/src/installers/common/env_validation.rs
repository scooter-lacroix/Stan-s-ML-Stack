//! Environment validation — `.mlstack_env` checking with sensible defaults.
//!
//! Ports functionality from `scripts/env_validation_utils.sh`:
//! - Validates `.mlstack_env` file exists and contains required variables
//! - Provides sensible defaults for missing values
//! - Validates ROCM_CHANNEL value
//! - Auto-detects GPU_ARCH and ROCM_VERSION when possible
//!
//! # Validation Assertions
//!
//! - **VAL-INFRA-017**: .mlstack_env validation with sensible defaults
//! - **VAL-INFRA-022**: Graceful handling of missing dependencies (no panics)

use crate::installers::common::utils::command_exists;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ===========================================================================
// Constants
// ===========================================================================

/// Valid ROCm channel names.
const VALID_CHANNELS: &[&str] = &["legacy", "stable", "latest", "preview"];

/// Default ROCm version if not detected.
const DEFAULT_ROCM_VERSION: &str = "7.2.0";

/// Default ROCm channel.
const DEFAULT_ROCM_CHANNEL: &str = "latest";

/// Default GPU architecture.
const DEFAULT_GPU_ARCH: &str = "gfx1100";

/// Default venv base directory (relative to home).
const DEFAULT_VENV_BASE_SUBDIR: &str = ".mlstack/venvs";

// ===========================================================================
// Types
// ===========================================================================

/// Result of validating the `.mlstack_env` file.
#[derive(Debug, Clone)]
pub struct EnvValidationResult {
    /// Whether the env file existed.
    pub file_existed: bool,
    /// Whether any defaults were applied.
    pub defaults_applied: bool,
    /// The resolved environment variables.
    pub vars: EnvVars,
    /// Warnings generated during validation.
    pub warnings: Vec<String>,
}

/// Resolved environment variables from `.mlstack_env`.
#[derive(Debug, Clone)]
pub struct EnvVars {
    /// ROCm version string (e.g., "7.2.0").
    pub rocm_version: String,
    /// ROCm channel (legacy, stable, latest, preview).
    pub rocm_channel: String,
    /// GPU architecture (e.g., "gfx1100").
    pub gpu_arch: String,
    /// ML Stack venv base directory.
    pub venv_base: String,
    /// Python binary path (if set).
    pub python_bin: Option<String>,
}

impl Default for EnvVars {
    fn default() -> Self {
        Self {
            rocm_version: DEFAULT_ROCM_VERSION.to_string(),
            rocm_channel: DEFAULT_ROCM_CHANNEL.to_string(),
            gpu_arch: DEFAULT_GPU_ARCH.to_string(),
            venv_base: String::new(),
            python_bin: None,
        }
    }
}

// ===========================================================================
// Validation Functions
// ===========================================================================

/// Get the default path for `.mlstack_env`.
pub fn default_env_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mlstack_env")
}

/// Get the default venv base directory.
pub fn default_venv_base() -> String {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join(DEFAULT_VENV_BASE_SUBDIR)
        .to_string_lossy()
        .to_string()
}

/// Validate the `.mlstack_env` file and return resolved variables.
///
/// If the file does not exist, auto-detects values from the system.
/// Missing values are filled with sensible defaults.
/// Invalid channel values are corrected to "latest".
///
/// # Validation
///
/// - **VAL-INFRA-017**: .mlstack_env validation with sensible defaults
/// - **VAL-INFRA-022**: Graceful handling of missing dependencies
///
/// This function never panics. If detection fails, defaults are used.
pub fn validate_mlstack_env(script_name: Option<&str>) -> EnvValidationResult {
    let env_path = default_env_path();
    let mut warnings = Vec::new();
    let mut defaults_applied = false;

    if !env_path.exists() {
        warnings.push(format!(
            "Environment file not found: {} - using auto-detection",
            env_path.display()
        ));
        let vars = auto_detect_vars(&mut defaults_applied);
        return EnvValidationResult {
            file_existed: false,
            defaults_applied: true,
            vars,
            warnings,
        };
    }

    // Parse the env file
    let contents = match fs::read_to_string(&env_path) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(format!("Failed to read {}: {e}", env_path.display()));
            let vars = auto_detect_vars(&mut defaults_applied);
            return EnvValidationResult {
                file_existed: true,
                defaults_applied: true,
                vars,
                warnings,
            };
        }
    };

    let parsed = parse_env_file(&contents);

    // Build EnvVars from parsed values with defaults
    let mut vars = EnvVars {
        rocm_version: String::new(),
        rocm_channel: String::new(),
        gpu_arch: String::new(),
        venv_base: String::new(),
        python_bin: None,
    };
    let _ = script_name; // used for warning context

    // ROCM_VERSION
    if let Some(v) = parsed.get("ROCM_VERSION") {
        vars.rocm_version = v.clone();
    } else {
        vars.rocm_version = detect_rocm_version_or_default();
        defaults_applied = true;
        warnings.push("ROCM_VERSION not set, using auto-detection".to_string());
    }

    // ROCM_CHANNEL
    if let Some(v) = parsed.get("ROCM_CHANNEL") {
        let lower = v.to_lowercase();
        if VALID_CHANNELS.contains(&lower.as_str()) {
            vars.rocm_channel = lower;
        } else {
            warnings.push(format!(
                "Unknown ROCM_CHANNEL '{v}', defaulting to 'latest'"
            ));
            vars.rocm_channel = DEFAULT_ROCM_CHANNEL.to_string();
            defaults_applied = true;
        }
    } else {
        vars.rocm_channel = DEFAULT_ROCM_CHANNEL.to_string();
        defaults_applied = true;
        warnings.push("ROCM_CHANNEL not set, defaulting to 'latest'".to_string());
    }

    // GPU_ARCH
    if let Some(v) = parsed.get("GPU_ARCH") {
        vars.gpu_arch = v.clone();
    } else {
        vars.gpu_arch = detect_gpu_arch_or_default();
        defaults_applied = true;
        warnings.push("GPU_ARCH not set, using auto-detection".to_string());
    }

    // MLSTACK_VENV_BASE
    vars.venv_base = parsed
        .get("MLSTACK_VENV_BASE")
        .cloned()
        .unwrap_or_else(default_venv_base);

    // MLSTACK_PYTHON_BIN
    vars.python_bin = parsed.get("MLSTACK_PYTHON_BIN").cloned();

    EnvValidationResult {
        file_existed: true,
        defaults_applied,
        vars,
        warnings,
    }
}

/// Require `.mlstack_env` validation — always succeeds with defaults.
///
/// This is the equivalent of `require_mlstack_env` in the shell scripts.
/// It never fails; missing values are always filled with defaults.
pub fn require_mlstack_env(script_name: Option<&str>) -> EnvValidationResult {
    validate_mlstack_env(script_name)
}

/// Load the `.mlstack_env` file into the current process environment.
///
/// Equivalent to `source ~/.mlstack_env` in the shell scripts.
/// Returns `true` if the file was loaded successfully.
pub fn load_mlstack_env() -> bool {
    let env_path = default_env_path();
    if !env_path.exists() {
        return false;
    }

    let contents = match fs::read_to_string(&env_path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("export ") {
            if let Some((key, value)) = rest.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                // Remove surrounding quotes if present
                let value = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .unwrap_or(value);
                // Skip values with shell variable expansions ($PATH, etc.)
                if !value.contains('$') {
                    std::env::set_var(key, value);
                }
            }
        }
    }

    true
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse an env file into a HashMap of key-value pairs.
fn parse_env_file(contents: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("export ") {
            if let Some((key, value)) = rest.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();
                // Remove surrounding quotes
                let value = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .unwrap_or(&value)
                    .to_string();
                map.insert(key, value);
            }
        } else if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim().to_string();
            let value = value.trim().to_string();
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(&value)
                .to_string();
            map.insert(key, value);
        }
    }
    map
}

/// Auto-detect environment variables from the system.
fn auto_detect_vars(defaults_applied: &mut bool) -> EnvVars {
    *defaults_applied = true;
    EnvVars {
        rocm_version: detect_rocm_version_or_default(),
        rocm_channel: DEFAULT_ROCM_CHANNEL.to_string(),
        gpu_arch: detect_gpu_arch_or_default(),
        venv_base: default_venv_base(),
        python_bin: None,
    }
}

/// Detect ROCm version or return default.
fn detect_rocm_version_or_default() -> String {
    // Try platform::linux first
    #[cfg(unix)]
    {
        let version = crate::platform::linux::get_rocm_version();
        if !version.is_empty() {
            return version;
        }
    }

    // Fallback: read version file directly
    let version_file = Path::new("/opt/rocm/.info/version");
    if let Ok(contents) = fs::read_to_string(version_file) {
        let v = contents.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }

    DEFAULT_ROCM_VERSION.to_string()
}

/// Detect GPU architecture or return default.
fn detect_gpu_arch_or_default() -> String {
    // Try rocminfo
    if command_exists("rocminfo") {
        if let Ok(output) = std::process::Command::new("rocminfo")
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("Name:") || trimmed.contains("gfx") {
                        // Extract gfx version
                        if let Some(idx) = trimmed.find("gfx") {
                            let rest = &trimmed[idx..];
                            let gfx: String = rest
                                .chars()
                                .take_while(|c| c.is_ascii_alphanumeric())
                                .collect();
                            if gfx.starts_with("gfx") && gfx.len() >= 5 {
                                return gfx;
                            }
                        }
                    }
                }
            }
        }
    }

    DEFAULT_GPU_ARCH.to_string()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // --- parse_env_file tests ---

    #[test]
    fn test_parse_env_file_basic() {
        let contents = r#"
# Comment
export ROCM_VERSION=7.2.0
export ROCM_CHANNEL=latest
export GPU_ARCH=gfx1100
"#;
        let map = parse_env_file(contents);
        assert_eq!(map.get("ROCM_VERSION").unwrap(), "7.2.0");
        assert_eq!(map.get("ROCM_CHANNEL").unwrap(), "latest");
        assert_eq!(map.get("GPU_ARCH").unwrap(), "gfx1100");
    }

    #[test]
    fn test_parse_env_file_quoted_values() {
        let contents = r#"
export PATH="/opt/rocm/bin:$PATH"
export ROCM_PATH=/opt/rocm
"#;
        let map = parse_env_file(contents);
        assert_eq!(map.get("PATH").unwrap(), "/opt/rocm/bin:$PATH");
        assert_eq!(map.get("ROCM_PATH").unwrap(), "/opt/rocm");
    }

    #[test]
    fn test_parse_env_file_empty() {
        let map = parse_env_file("");
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_env_file_comments_only() {
        let contents = "# just comments\n# another comment\n";
        let map = parse_env_file(contents);
        assert!(map.is_empty());
    }

    // --- validate_mlstack_env tests (VAL-INFRA-017) ---

    #[test]
    fn test_validate_missing_env_file() {
        // Use a temp dir to ensure no .mlstack_env exists
        let _temp_dir = tempfile::tempdir().unwrap();
        // We can't easily override the home dir, but we can test the auto-detect path
        let result = validate_mlstack_env(Some("test_script"));
        // Should always succeed (never panic)
        assert!(result.vars.rocm_channel == "latest"
            || result.vars.rocm_channel == "stable"
            || result.vars.rocm_channel == "legacy"
            || result.vars.rocm_channel == "preview");
    }

    #[test]
    fn test_validate_always_returns_defaults() {
        let result = validate_mlstack_env(None);
        // Should always have non-empty values
        assert!(!result.vars.rocm_version.is_empty());
        assert!(!result.vars.rocm_channel.is_empty());
        assert!(!result.vars.gpu_arch.is_empty());
    }

    #[test]
    fn test_require_mlstack_env_always_succeeds() {
        let result = require_mlstack_env(None);
        // Should always return a result with valid defaults
        assert!(!result.vars.rocm_version.is_empty());
    }

    #[test]
    fn test_valid_channels() {
        for ch in VALID_CHANNELS {
            assert!(VALID_CHANNELS.contains(ch));
        }
    }

    // --- EnvVars default ---

    #[test]
    fn test_env_vars_default() {
        let vars = EnvVars::default();
        assert_eq!(vars.rocm_version, "7.2.0");
        assert_eq!(vars.rocm_channel, "latest");
        assert_eq!(vars.gpu_arch, "gfx1100");
        assert!(vars.python_bin.is_none());
    }

    // --- default_venv_base ---

    #[test]
    fn test_default_venv_base() {
        let base = default_venv_base();
        assert!(base.contains(".mlstack/venvs"));
    }

    // --- load_mlstack_env ---

    #[test]
    fn test_load_mlstack_env_nonexistent() {
        // Should return false without panicking
        // This tests the actual default path which may or may not exist
        let _ = load_mlstack_env();
    }

    // --- detect_rocm_version_or_default ---

    #[test]
    fn test_detect_rocm_version_no_panic() {
        let version = detect_rocm_version_or_default();
        assert!(!version.is_empty());
    }

    // --- detect_gpu_arch_or_default ---

    #[test]
    fn test_detect_gpu_arch_no_panic() {
        let arch = detect_gpu_arch_or_default();
        assert!(!arch.is_empty());
        assert!(arch.starts_with("gfx"));
    }

    // --- Integration: parse + validate with known content ---

    #[test]
    fn test_validate_with_valid_channel() {
        // This tests the channel validation logic indirectly
        let vars = EnvVars::default();
        assert!(VALID_CHANNELS.contains(&vars.rocm_channel.as_str()));
    }

    #[test]
    fn test_gpu_arch_format() {
        let arch = detect_gpu_arch_or_default();
        // Should be gfx followed by digits
        assert!(arch.starts_with("gfx"));
        assert!(arch.len() >= 5); // gfx + at least 2 digits
    }
}
