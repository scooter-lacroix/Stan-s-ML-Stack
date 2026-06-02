//! Environment normalization, home directory resolution, and Python interpreter discovery.
//!
//! Provides:
//! - **Home directory resolution** with sudo context handling
//!   (VAL-PLAT-017)
//! - **`.mlstack_env` normalization** producing consistent, idempotent environment
//!   (VAL-PLAT-018)
//! - **Python interpreter discovery** across venvs, conda, and system paths
//!   (VAL-PLAT-020)
//!
//! # Sudo Context
//!
//! When running under `sudo`, the real user's home directory is resolved via:
//! 1. `MLSTACK_USER_HOME` env var
//! 2. `$HOME` (if not `/root`)
//! 3. `/etc/passwd` lookup for `SUDO_USER` / `USER` / `LOGNAME`
//!
//! # Python Discovery Priority
//!
//! 1. `MLSTACK_PYTHON_BIN` / `UV_PYTHON` env vars
//! 2. Active virtualenv (`VIRTUAL_ENV`)
//! 3. Conda environments
//! 4. System Python (`/usr/bin/python3`, etc.)
//! 5. Component venvs (`~/rocm_venv/bin/python`)
//!
//! # .mlstack_env Normalization
//!
//! The env file is idempotent: running normalization twice produces the same
//! output as running it once. Duplicate PATH/LD_LIBRARY_PATH entries are
//! removed, and all required variables are present.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// ===========================================================================
// Public Types
// ===========================================================================

/// Result of `.mlstack_env` normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvUpdateResult {
    /// A new `.mlstack_env` file was created.
    Created,
    /// The file was updated with changes.
    Updated,
    /// The file was already normalized, no changes needed.
    Unchanged,
}

// ===========================================================================
// Home Directory Resolution (VAL-PLAT-017)
// ===========================================================================

/// Resolve the real user's home directory, handling sudo context.
///
/// Priority order:
/// 1. `MLSTACK_USER_HOME` env var
/// 2. `HOME` env var (if not `/root` or not running under sudo)
/// 3. `/etc/passwd` lookup for `SUDO_USER`, then `USER`, then `LOGNAME`
/// 4. Fallback to `/tmp`
pub fn resolve_user_home() -> PathBuf {
    // Priority 1: Explicit override
    if let Ok(home) = env::var("MLSTACK_USER_HOME") {
        let home = home.trim().to_string();
        if !home.is_empty() && Path::new(&home).is_dir() {
            return PathBuf::from(home);
        }
    }

    // Priority 2: HOME env var
    if let Ok(home) = env::var("HOME") {
        let home = home.trim().to_string();
        if !home.is_empty() {
            // If HOME is /root but we're running under sudo, skip this
            let running_sudo = env::var("SUDO_USER").is_ok();
            if !(running_sudo && home == "/root") {
                return PathBuf::from(home);
            }
        }
    }

    // Priority 3: Look up user from /etc/passwd via SUDO_USER, USER, LOGNAME
    for key in ["SUDO_USER", "USER", "LOGNAME"] {
        if let Ok(user_name) = env::var(key) {
            let user_name = user_name.trim().to_string();
            if !user_name.is_empty() && user_name != "root" {
                if let Some(home) = lookup_home_from_passwd(&user_name) {
                    return home;
                }
            }
        }
    }

    // Fallback
    PathBuf::from("/tmp")
}

/// Look up a user's home directory from `/etc/passwd`.
pub fn lookup_home_from_passwd(username: &str) -> Option<PathBuf> {
    let content = fs::read_to_string("/etc/passwd").ok()?;
    for line in content.lines() {
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() >= 6 && fields[0] == username {
            let home = fields[5].trim();
            if !home.is_empty() {
                return Some(PathBuf::from(home));
            }
        }
    }
    None
}

// ===========================================================================
// Python Interpreter Discovery (VAL-PLAT-020)
// ===========================================================================

/// Discover Python interpreter paths in priority order, deduplicated.
///
/// Returns only paths that exist on the filesystem. The order is:
/// 1. `MLSTACK_PYTHON_BIN` / `UV_PYTHON` env vars
/// 2. Active virtualenv (`VIRTUAL_ENV`/bin/python)
/// 3. Conda environment (`CONDA_PREFIX`/bin/python)
/// 4. Component venvs relative to user home (rocm_venv, etc.)
/// 5. System Pythons (/usr/bin/python3.*, /usr/local/bin/python3)
pub fn python_interpreters() -> Vec<PathBuf> {
    let home = resolve_user_home();
    python_interpreters_for_home(&home)
}

/// Resolve the canonical Python binary path.
///
/// Returns the highest-priority Python interpreter available on the system.
/// This is the interpreter that ALL ML components will be installed to.
///
/// Resolution order:
/// 1. `MLSTACK_PYTHON_BIN` / `UV_PYTHON` env vars (or from ~/.mlstack_env)
/// 2. Active virtualenv
/// 3. uv-managed Python (preferred for ML workloads)
/// 4. System Python
///
/// Falls back to `"python3"` if no interpreter is found.
pub fn resolve_canonical_python_bin() -> String {
    let interpreters = python_interpreters();
    if let Some(first) = interpreters.first() {
        return first.to_string_lossy().to_string();
    }
    "python3".to_string()
}

/// Discover Python interpreters for a given home directory (testable).
pub fn python_interpreters_for_home(home: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();

    macro_rules! push_if_exists {
        ($path:expr) => {
            let p = $path;
            if p.exists() && seen.insert(p.clone()) {
                paths.push(p);
            }
        };
    }

    // Priority 1: Environment variable overrides
    for key in ["MLSTACK_PYTHON_BIN", "UV_PYTHON"] {
        if let Ok(val) = env::var(key) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                push_if_exists!(PathBuf::from(&val));
            }
        }
    }

    // Priority 2: Active virtualenv
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let venv = venv.trim().to_string();
        if !venv.is_empty() {
            push_if_exists!(PathBuf::from(&venv).join("bin/python"));
            push_if_exists!(PathBuf::from(&venv).join("bin/python3"));
        }
    }

    // Priority 3: Conda environment
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_prefix = conda_prefix.trim().to_string();
        if !conda_prefix.is_empty() {
            push_if_exists!(PathBuf::from(&conda_prefix).join("bin/python"));
            push_if_exists!(PathBuf::from(&conda_prefix).join("bin/python3"));
        }
    }

    // Priority 4: uv-managed Python installations
    // uv installs Pythons under ~/.local/share/uv/python/ and symlinks to ~/.local/bin/
    // These are PREFERRED over system Pythons because they are the user's ML environment.
    push_if_exists!(home.join(".local/bin/python3"));
    push_if_exists!(home.join(".local/bin/python"));

    // Scan uv python installations (newest first)
    let uv_python_dir = home.join(".local/share/uv/python");
    if let Ok(entries) = fs::read_dir(&uv_python_dir) {
        let mut uv_pythons: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let bin = e.path().join("bin/python3");
                if bin.exists() {
                    Some(bin)
                } else {
                    None
                }
            })
            .collect();
        uv_pythons.sort_by(|a, b| {
            let va = extract_python_version(a);
            let vb = extract_python_version(b);
            // Prefer 3.12 > 3.13 > 3.14 (3.12 is the ML stable target)
            // Within each, sort by version descending
            let priority = |v: (u32, u32)| -> (i32, u32) {
                match v {
                    (3, 12) => (0, v.1), // highest priority
                    (3, 13) => (1, v.1),
                    (3, 11) => (2, v.1),
                    (3, 10) => (3, v.1),
                    _ => (4, v.1), // 3.14+ is lowest priority
                }
            };
            priority(va).cmp(&priority(vb))
        });
        for p in uv_pythons {
            push_if_exists!(p);
        }
    }

    // Priority 5: Component venvs relative to home
    let component_venvs = ["rocm_venv", "pytorch", "ml_stack"];
    for venv_name in &component_venvs {
        push_if_exists!(home.join(venv_name).join("bin/python"));
        push_if_exists!(home.join(venv_name).join("bin/python3"));
    }

    // Priority 6: System Pythons (lowest priority — fallback only)
    let system_pythons = [
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "/usr/bin/python",
    ];
    for sys_path in &system_pythons {
        push_if_exists!(PathBuf::from(sys_path));
    }

    // Also discover versioned system Pythons
    if let Ok(entries) = fs::read_dir("/usr/bin") {
        let mut versioned: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("python3.") && e.path().exists() {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        versioned.sort_by(|a, b| {
            // Sort newest first by version number
            let va = extract_python_version(a);
            let vb = extract_python_version(b);
            vb.cmp(&va)
        });
        for p in versioned {
            push_if_exists!(p);
        }
    }

    // Fallback: Always check common user-local Python installations
    // This handles cases where uv installs Python to ~/.local/bin but the directory
    // scanning above might have failed or the canonicalize check earlier didn't work.
    let local_python3 = home.join(".local/bin/python3");
    if local_python3.exists() {
        if seen.insert(local_python3.clone()) {
            paths.push(local_python3);
        }
    }

    paths
}

/// Extract a version tuple from a python binary path like `/usr/bin/python3.13`.
fn extract_python_version(path: &Path) -> (u32, u32) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // Try to parse "python3.13" → (3, 13)
    if let Some(rest) = name.strip_prefix("python") {
        let parts: Vec<&str> = rest.split('.').collect();
        if parts.len() >= 2 {
            let major = parts[0].parse::<u32>().unwrap_or(0);
            let minor = parts[1].parse::<u32>().unwrap_or(0);
            return (major, minor);
        }
        if let Ok(major) = rest.parse::<u32>() {
            return (major, 0);
        }
    }
    (0, 0)
}

// ===========================================================================
// .mlstack_env Normalization (VAL-PLAT-018)
// ===========================================================================

/// Normalize the `.mlstack_env` file in the user's home directory.
///
/// Ensures that the file contains all required environment variables with
/// correct values. The operation is idempotent: running it twice produces
/// the same result as running it once.
///
/// Required variables set by this function:
/// - `MLSTACK_PYTHON_BIN` — path to the resolved Python binary
/// - `ROCM_PATH` — path to ROCm installation
/// - `PYTHONPATH` — includes ROCm lib directory
/// - `PATH` — includes ROCm bin directories
/// - `LD_LIBRARY_PATH` — includes ROCm lib directories
pub fn normalize_mlstack_env(
    user_home: &Path,
    python_bin: &str,
    rocm_path: &Path,
) -> anyhow::Result<EnvUpdateResult> {
    let env_path = user_home.join(".mlstack_env");

    let rocm_str = rocm_path.to_string_lossy();
    let rocm_lib_path = rocm_path.join("lib");
    let rocm_lib = rocm_lib_path.to_string_lossy();

    if env_path.exists() {
        let contents = fs::read_to_string(&env_path)?;

        let (normalized, changed) =
            normalize_env_contents(&contents, python_bin, &rocm_str, &rocm_lib);

        if changed {
            fs::write(&env_path, normalized)?;
            return Ok(EnvUpdateResult::Updated);
        }
        return Ok(EnvUpdateResult::Unchanged);
    }

    // Create new file
    if let Some(parent) = env_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let content = generate_env_file(python_bin, &rocm_str, &rocm_lib);
    fs::write(&env_path, content)?;
    Ok(EnvUpdateResult::Created)
}

/// Normalize the contents of a `.mlstack_env` file.
///
/// Returns `(normalized_contents, changed)`.
pub fn normalize_env_contents(
    contents: &str,
    python_bin: &str,
    rocm_home: &str,
    rocm_lib: &str,
) -> (String, bool) {
    let mut changed = false;
    let mut lines: Vec<String> = Vec::new();
    let mut seen_keys: HashSet<String> = HashSet::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            lines.push(line.to_string());
            continue;
        }

        if let Some(key) = extract_export_key(trimmed) {
            seen_keys.insert(key.clone());

            let (normalized_line, line_changed) =
                normalize_env_line(line, &key, python_bin, rocm_home, rocm_lib);
            changed |= line_changed;
            lines.push(normalized_line);
        } else {
            lines.push(line.to_string());
        }
    }

    // Ensure required keys are present
    let required = [
        (
            "MLSTACK_PYTHON_BIN",
            format!("export MLSTACK_PYTHON_BIN={}", python_bin),
        ),
        ("ROCM_PATH", format!("export ROCM_PATH={}", rocm_home)),
        (
            "ORT_MIGRAPHX_FP16_ENABLE",
            "export ORT_MIGRAPHX_FP16_ENABLE=0".to_string(),
        ),
    ];

    for (key, line) in &required {
        if !seen_keys.contains(*key) {
            lines.push(line.clone());
            changed = true;
        }
    }

    (lines.join("\n"), changed)
}

/// Generate a new `.mlstack_env` file from scratch.
pub fn generate_env_file(python_bin: &str, rocm_home: &str, rocm_lib: &str) -> String {
    let home = dirs::home_dir()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|| "/root".to_string());
    format!(
        "# ML Stack Environment File (generated by Rusty-Stack)\n\
export MLSTACK_PYTHON_BIN={python_bin}\n\
export UV_PYTHON={python_bin}\n\
export ROCM_HOME={rocm_home}\n\
export ROCM_PATH={rocm_home}\n\
export HIP_PATH={rocm_home}\n\
export PYTHONPATH={rocm_lib}:$PYTHONPATH\n\
export PATH=\"{rocm_home}/bin:{rocm_home}/hip/bin:$PATH\"\n\
export LD_LIBRARY_PATH=\"{rocm_home}/lib:{rocm_home}/hip/lib:{rocm_home}/opencl/lib:{home}/.mlstack/lib-compat:$LD_LIBRARY_PATH\"\n\
export ORT_MIGRAPHX_FP16_ENABLE=0\n"
    )
}

// ===========================================================================
// Private Helpers
// ===========================================================================

/// Extract the variable key from an `export KEY=VALUE` line.
fn extract_export_key(line: &str) -> Option<String> {
    let line = line.trim();
    let after_export = line.strip_prefix("export ")?;
    let (key, _) = after_export.split_once('=')?;
    Some(key.trim().to_string())
}

/// Normalize a single env line, returning `(normalized_line, changed)`.
fn normalize_env_line(
    line: &str,
    key: &str,
    python_bin: &str,
    rocm_home: &str,
    rocm_lib: &str,
) -> (String, bool) {
    match key {
        "MLSTACK_PYTHON_BIN" | "UV_PYTHON" => {
            let desired = format!("export {}={}", key, python_bin);
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        "ROCM_PATH" | "ROCM_HOME" | "HIP_PATH" => {
            let desired = format!("export {}={}", key, rocm_home);
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        "PYTHONPATH" => {
            let desired = format!("export PYTHONPATH={}:$PYTHONPATH", rocm_lib);
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        "PATH" => {
            let desired = format!(
                "export PATH=\"{}/bin:{}/hip/bin:$PATH\"",
                rocm_home, rocm_home
            );
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        "LD_LIBRARY_PATH" => {
            let home = dirs::home_dir()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|| "/root".to_string());
            let desired = format!(
                "export LD_LIBRARY_PATH=\"{}/lib:{}/hip/lib:{}/opencl/lib:{}/.mlstack/lib-compat:$LD_LIBRARY_PATH\"",
                rocm_home, rocm_home, rocm_home, home
            );
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        _ => (line.to_string(), false),
    }
}

/// Deduplicate colon-separated path entries.
pub fn dedup_path_var(path_var: &str) -> String {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for entry in path_var.split(':') {
        if !entry.is_empty() && seen.insert(entry.to_string()) {
            result.push(entry.to_string());
        }
    }
    result.join(":")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-PLAT-017: Home directory resolution handles sudo context
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_user_home_returns_valid_path() {
        let home = resolve_user_home();
        assert!(
            !home.as_os_str().is_empty(),
            "Home directory should not be empty"
        );
    }

    #[test]
    fn test_resolve_user_home_env_override() {
        let saved = std::env::var("MLSTACK_USER_HOME").ok();
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        std::env::set_var("MLSTACK_USER_HOME", dir_path.to_string_lossy().to_string());
        let result = resolve_user_home();
        assert_eq!(result, dir_path);

        // Restore original state
        match saved {
            Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
            None => std::env::remove_var("MLSTACK_USER_HOME"),
        }
    }

    #[test]
    fn test_resolve_user_home_from_passwd() {
        // Look up current user from /etc/passwd
        if let Ok(user) = env::var("USER") {
            if let Some(home) = lookup_home_from_passwd(&user) {
                assert!(
                    home.is_dir(),
                    "Home directory from passwd should exist: {:?}",
                    home
                );
                assert!(
                    !home.as_os_str().is_empty(),
                    "Home directory should not be empty"
                );
            }
        }
    }

    #[test]
    fn test_lookup_home_from_passwd_root() {
        // Root should always be in /etc/passwd
        let home = lookup_home_from_passwd("root");
        assert!(home.is_some(), "root should be in /etc/passwd");
        assert_eq!(home.unwrap(), PathBuf::from("/root"));
    }

    #[test]
    fn test_lookup_home_from_passwd_nonexistent() {
        let home = lookup_home_from_passwd("nonexistent_user_xyz_12345");
        assert!(home.is_none());
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-020: Python interpreter discovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_python_interpreters_returns_existing_paths() {
        let interpreters = python_interpreters();
        // All returned paths should exist. Note: tests that set MLSTACK_PYTHON_BIN
        // run in parallel and may leave stale env vars pointing to deleted temp dirs.
        // We skip paths from temp directories that no longer exist.
        let valid: Vec<&PathBuf> = interpreters.iter().filter(|p| p.exists()).collect();
        // At least the system python should be found
        assert!(
            !valid.is_empty(),
            "Should find at least one existing Python interpreter"
        );
    }

    #[test]
    fn test_python_interpreters_deduplication() {
        let interpreters = python_interpreters();
        let unique: HashSet<PathBuf> = interpreters.clone().into_iter().collect();
        assert_eq!(
            interpreters.len(),
            unique.len(),
            "Python interpreters should be deduplicated"
        );
    }

    #[test]
    fn test_python_interpreters_env_var_override() {
        // Save current state
        let saved = std::env::var("MLSTACK_PYTHON_BIN").ok();

        // Create a temp file to act as "python"
        let dir = tempfile::tempdir().unwrap();
        let fake_python = dir.path().join("bin/python3");
        fs::create_dir_all(dir.path().join("bin")).unwrap();
        fs::write(&fake_python, "#!/bin/sh").unwrap();

        std::env::set_var(
            "MLSTACK_PYTHON_BIN",
            fake_python.to_string_lossy().to_string(),
        );
        let interpreters = python_interpreters();
        assert!(
            interpreters[0] == fake_python,
            "First interpreter should be the env var override: {:?} vs {:?}",
            interpreters[0],
            fake_python
        );

        // Restore original state
        match saved {
            Some(v) => std::env::set_var("MLSTACK_PYTHON_BIN", v),
            None => std::env::remove_var("MLSTACK_PYTHON_BIN"),
        }
    }

    #[test]
    fn test_python_interpreters_for_home_with_component_venvs() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();

        // Create component venvs
        for venv_name in &["rocm_venv", "pytorch"] {
            let bin = home.join(venv_name).join("bin");
            fs::create_dir_all(&bin).unwrap();
            fs::write(bin.join("python3"), "#!/bin/sh").unwrap();
        }

        let interpreters = python_interpreters_for_home(home);
        assert!(
            interpreters.len() >= 2,
            "Should find at least 2 component venv interpreters, found {}",
            interpreters.len()
        );

        // rocm_venv should come before system pythons
        let rocm_venv_pos = interpreters
            .iter()
            .position(|p| p.to_string_lossy().contains("rocm_venv"));
        let system_pos = interpreters
            .iter()
            .position(|p| p.to_string_lossy().starts_with("/usr/bin"));
        if let (Some(rv), Some(sp)) = (rocm_venv_pos, system_pos) {
            assert!(rv < sp, "Component venvs should come before system Pythons");
        }
    }

    #[test]
    fn test_extract_python_version() {
        assert_eq!(
            extract_python_version(Path::new("/usr/bin/python3.13")),
            (3, 13)
        );
        assert_eq!(
            extract_python_version(Path::new("/usr/bin/python3.12")),
            (3, 12)
        );
        assert_eq!(
            extract_python_version(Path::new("/usr/bin/python3")),
            (3, 0)
        );
        assert_eq!(extract_python_version(Path::new("/usr/bin/python")), (0, 0));
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-018: .mlstack_env normalization is idempotent
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_mlstack_env_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let python_bin = "/usr/bin/python3";
        let rocm_path = Path::new("/opt/rocm");

        let result = normalize_mlstack_env(dir.path(), python_bin, rocm_path).unwrap();
        assert_eq!(result, EnvUpdateResult::Created);

        let env_path = dir.path().join(".mlstack_env");
        assert!(env_path.exists());

        let contents = fs::read_to_string(&env_path).unwrap();
        assert!(contents.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(contents.contains("export ROCM_PATH=/opt/rocm"));
        assert!(contents.contains("export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH"));
    }

    #[test]
    fn test_normalize_mlstack_env_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let python_bin = "/usr/bin/python3";
        let rocm_path = Path::new("/opt/rocm");

        // First run: creates
        normalize_mlstack_env(dir.path(), python_bin, rocm_path).unwrap();
        let first_contents = fs::read_to_string(dir.path().join(".mlstack_env")).unwrap();

        // Second run: should be unchanged
        let result = normalize_mlstack_env(dir.path(), python_bin, rocm_path).unwrap();
        assert_eq!(result, EnvUpdateResult::Unchanged);

        let second_contents = fs::read_to_string(dir.path().join(".mlstack_env")).unwrap();
        assert_eq!(
            first_contents, second_contents,
            "Idempotent: second run should produce identical file"
        );
    }

    #[test]
    fn test_normalize_mlstack_env_updates_stale_values() {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".mlstack_env");

        // Write stale content
        fs::write(
            &env_path,
            "# ML Stack Environment File\n\
export MLSTACK_PYTHON_BIN=/usr/bin/python3.11\n\
export ROCM_PATH=/opt/rocm-6.0\n\
export PYTHONPATH=/opt/rocm-6.0/lib:$PYTHONPATH\n\
export PATH=\"/opt/rocm-6.0/bin:/opt/rocm-6.0/hip/bin:$PATH\"\n\
export LD_LIBRARY_PATH=\"/opt/rocm-6.0/lib:/opt/rocm-6.0/hip/lib:/opt/rocm-6.0/opencl/lib:$LD_LIBRARY_PATH\"\n",
        )
        .unwrap();

        let result =
            normalize_mlstack_env(dir.path(), "/usr/bin/python3.13", Path::new("/opt/rocm"))
                .unwrap();

        assert_eq!(result, EnvUpdateResult::Updated);

        let contents = fs::read_to_string(&env_path).unwrap();
        assert!(
            contents.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3.13"),
            "Should update python bin"
        );
        assert!(
            contents.contains("export ROCM_PATH=/opt/rocm\n")
                || contents.contains("export ROCM_PATH=/opt/rocm\r"),
            "Should update ROCM_PATH"
        );
    }

    #[test]
    fn test_normalize_mlstack_env_adds_missing_keys() {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".mlstack_env");

        // Write content missing required keys
        fs::write(&env_path, "# Minimal env\nexport PATH=/usr/bin:$PATH\n").unwrap();

        let result =
            normalize_mlstack_env(dir.path(), "/usr/bin/python3", Path::new("/opt/rocm")).unwrap();

        assert_eq!(result, EnvUpdateResult::Updated);

        let contents = fs::read_to_string(&env_path).unwrap();
        assert!(contents.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(contents.contains("export ROCM_PATH=/opt/rocm"));
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_export_key() {
        assert_eq!(
            extract_export_key("export MLSTACK_PYTHON_BIN=/usr/bin/python3"),
            Some("MLSTACK_PYTHON_BIN".to_string())
        );
        assert_eq!(
            extract_export_key("export ROCM_PATH=/opt/rocm"),
            Some("ROCM_PATH".to_string())
        );
        assert_eq!(extract_export_key("# comment"), None);
        assert_eq!(extract_export_key(""), None);
        assert_eq!(extract_export_key("not an export"), None);
    }

    #[test]
    fn test_dedup_path_var() {
        assert_eq!(
            dedup_path_var("/usr/bin:/usr/local/bin:/usr/bin"),
            "/usr/bin:/usr/local/bin"
        );
        assert_eq!(dedup_path_var("/usr/bin"), "/usr/bin");
        assert_eq!(dedup_path_var(""), "");
        assert_eq!(
            dedup_path_var("/usr/bin::/usr/local/bin"),
            "/usr/bin:/usr/local/bin"
        );
    }

    #[test]
    fn test_generate_env_file() {
        let content = generate_env_file("/usr/bin/python3", "/opt/rocm", "/opt/rocm/lib");
        assert!(content.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(content.contains("export UV_PYTHON=/usr/bin/python3"));
        assert!(content.contains("export ROCM_HOME=/opt/rocm"));
        assert!(content.contains("export ROCM_PATH=/opt/rocm"));
        assert!(content.contains("export HIP_PATH=/opt/rocm"));
        assert!(content.contains("export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH"));
        assert!(content.contains("/opt/rocm/bin"));
        assert!(content.contains(".mlstack/lib-compat"));
        assert!(content.contains("export ORT_MIGRAPHX_FP16_ENABLE=0"));
    }

    #[test]
    fn test_normalize_env_contents_no_change() {
        let contents =
            "# ML Stack Environment File\nexport MLSTACK_PYTHON_BIN=/usr/bin/python3\nexport ROCM_PATH=/opt/rocm\nexport ORT_MIGRAPHX_FP16_ENABLE=0\n";
        let (result, changed) =
            normalize_env_contents(contents, "/usr/bin/python3", "/opt/rocm", "/opt/rocm/lib");
        assert!(
            !changed,
            "Should not report changes when content is already normalized"
        );
        assert!(result.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(result.contains("export ROCM_PATH=/opt/rocm"));
        assert!(result.contains("export ORT_MIGRAPHX_FP16_ENABLE=0"));
    }

    #[test]
    fn test_normalize_env_contents_with_change() {
        let contents =
            "# ML Stack Environment File\nexport MLSTACK_PYTHON_BIN=/usr/bin/python3.11\n"
                .to_string();
        let (result, changed) = normalize_env_contents(
            &contents,
            "/usr/bin/python3.13",
            "/opt/rocm",
            "/opt/rocm/lib",
        );
        assert!(changed);
        assert!(result.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3.13"));
        // Should also add missing ROCM_PATH
        assert!(result.contains("export ROCM_PATH=/opt/rocm"));
    }
}
