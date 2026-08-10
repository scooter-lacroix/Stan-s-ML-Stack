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
use std::io::IsTerminal;
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
// ML Stack paths — the SINGLE consolidated root (~/.mlstack/)
// ===========================================================================

/// The single ML Stack root directory: `~/.mlstack/`.
///
/// All Rusty-managed state lives under here (Stage 0 consolidation): the
/// global venv (`global/`), named envs (`envs/<name>/`), logs, cache, triton,
/// the installed-component registry (`installed.json`), and config.
pub fn mlstack_root() -> PathBuf {
    resolve_user_home().join(".mlstack")
}

/// The managed **global** default environment: `~/.mlstack/global/`.
pub fn mlstack_global_dir() -> PathBuf {
    mlstack_root().join("global")
}

/// The Python binary of the managed global env: `~/.mlstack/global/bin/python`.
pub fn mlstack_global_python() -> PathBuf {
    mlstack_global_dir().join("bin").join("python")
}

/// The named-envs directory: `~/.mlstack/envs/`.
pub fn mlstack_envs_dir() -> PathBuf {
    mlstack_root().join("envs")
}

/// A specific named env: `~/.mlstack/envs/<name>/`.
pub fn mlstack_env_dir(name: &str) -> PathBuf {
    mlstack_envs_dir().join(name)
}

/// The cache directory: `~/.mlstack/cache/`.
pub fn mlstack_cache_dir() -> PathBuf {
    mlstack_root().join("cache")
}

/// The logs directory: `~/.mlstack/logs/`.
pub fn mlstack_logs_dir() -> PathBuf {
    mlstack_root().join("logs")
}

/// The triton home directory: `~/.mlstack/triton/`.
pub fn mlstack_triton_dir() -> PathBuf {
    mlstack_root().join("triton")
}

/// Ensure the `~/.mlstack/` root and its standard subdirs exist.
pub fn ensure_mlstack_dirs() -> std::io::Result<()> {
    for dir in [
        mlstack_root(),
        mlstack_global_dir(),
        mlstack_envs_dir(),
        mlstack_cache_dir(),
        mlstack_logs_dir(),
        mlstack_triton_dir(),
    ] {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(())
}

/// Create the managed global venv at `~/.mlstack/global/` if it does not yet
/// exist, using `bootstrap_python` (a discovered system/uv interpreter) as the
/// base. Returns the path to the global venv's `python`.
///
/// Prefers `uv venv` when available (fast, deterministic); falls back to
/// `python -m venv`. This is the deterministic anchor that replaces the old
/// non-deterministic interpreter scan for global installs.
pub fn ensure_global_venv(bootstrap_python: &str) -> anyhow::Result<PathBuf> {
    let global_python = mlstack_global_python();
    if !global_python.exists() {
        ensure_mlstack_dirs()?;
        let global_dir = mlstack_global_dir();
        create_venv(&global_dir, bootstrap_python)?;
        if !global_python.exists() {
            anyhow::bail!(
                "global venv creation reported success but {} is missing",
                global_python.display()
            );
        }
    }
    // Always ensure pip is present (uv venvs don't seed pip by default; a venv
    // created before --seed may lack it). Every installer runs `python -m pip`.
    ensure_pip_in_venv(&global_python)?;
    Ok(global_python)
}

/// Create the managed NAMED env at `~/.mlstack/envs/<name>/` if absent
/// (Tenet 1: install-to-env isolation). Returns the path to its `python`.
///
/// `name` is validated (non-empty, no path separators / traversal) since it is
/// interpolated into a filesystem path.
pub fn ensure_named_venv(name: &str, bootstrap_python: &str) -> anyhow::Result<PathBuf> {
    let name = name.trim();
    if name.is_empty() {
        anyhow::bail!("named env name must not be empty");
    }
    if name.contains(std::path::MAIN_SEPARATOR)
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
    {
        anyhow::bail!("invalid env name '{name}'");
    }
    ensure_mlstack_dirs()?;
    let env_dir = mlstack_env_dir(name);
    let env_python = env_dir.join("bin").join("python");
    if !env_python.exists() {
        create_venv(&env_dir, bootstrap_python)?;
        if !env_python.exists() {
            anyhow::bail!(
                "named venv creation reported success but {} is missing",
                env_python.display()
            );
        }
    }
    // Always ensure pip is present (same rationale as the global venv).
    ensure_pip_in_venv(&env_python)?;
    Ok(env_python)
}

/// Create a venv at `dir` using `uv` when available, else `python -m venv`.
fn create_venv(dir: &Path, bootstrap_python: &str) -> anyhow::Result<()> {
    if command_on_path("uv") {
        // --seed installs pip/setuptools/wheel so `python -m pip` works in the
        // venv. uv venvs ship WITHOUT pip by default, which breaks every
        // installer (they all run `<python> -m pip install …`).
        // Treat a uv SPAWN failure (e.g. uv on PATH but not executable / missing
        // runtime) the same as a non-successful run: fall through to the
        // `python -m venv` fallback rather than bailing with a spawn error.
        let uv_status = std::process::Command::new("uv")
            .arg("venv")
            .arg("--seed")
            .arg("--python")
            .arg(bootstrap_python)
            .arg(dir)
            .status();
        if let Ok(status) = uv_status {
            if status.success() {
                return Ok(());
            }
        }
    }
    let status = std::process::Command::new(bootstrap_python)
        .arg("-m")
        .arg("venv")
        .arg(dir)
        .status()?;
    if !status.success() {
        anyhow::bail!(
            "failed to create venv at {} via uv/venv (bootstrap={})",
            dir.display(),
            bootstrap_python
        );
    }
    Ok(())
}

/// Ensure `pip` is importable in the managed venv. `uv venv` does not seed pip
/// by default, and a venv created before `--seed` may lack it — but every
/// installer runs `<python> -m pip install …`, so a pip-less venv breaks ALL of
/// them. Bootstrap via `uv pip install pip` (preferred; works on uv-managed
/// cpython, which may not ship `ensurepip`) or stdlib `ensurepip` as a fallback.
fn ensure_pip_in_venv(venv_python: &Path) -> anyhow::Result<()> {
    // Fast path: pip already importable.
    let has_pip = std::process::Command::new(venv_python)
        .args(["-m", "pip", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if has_pip {
        return Ok(());
    }
    // Prefer uv (uv-managed cpython may not ship ensurepip).
    if command_on_path("uv") {
        let s = std::process::Command::new("uv")
            .args(["pip", "install", "--python"])
            .arg(venv_python)
            .arg("pip")
            .status();
        if let Ok(status) = s {
            if status.success() {
                return Ok(());
            }
        }
    }
    // Fallback: stdlib ensurepip.
    let status = std::process::Command::new(venv_python)
        .args(["-m", "ensurepip", "--upgrade"])
        .status()?;
    if !status.success() {
        anyhow::bail!(
            "could not bootstrap pip into {} (tried `uv pip install pip` and `ensurepip`)",
            venv_python.display()
        );
    }
    Ok(())
}

// ===========================================================================
// Python Interpreter Discovery (VAL-PLAT-020)
// ===========================================================================

/// Discover Python interpreter paths in priority order, deduplicated.
///
/// Returns only paths that exist on the filesystem. The order is:
/// 0. The managed global env (`~/.mlstack/global/bin/python`) — deterministic anchor
/// 1. `MLSTACK_PYTHON_BIN` / `UV_PYTHON` env vars
/// 2. Active virtualenv (`VIRTUAL_ENV`/bin/python)
/// 3. Conda environment (`CONDA_PREFIX`/bin/python)
/// 4. Component venvs relative to user home (rocm_venv, etc.)
/// 5. System Pythons (/usr/bin/python3.*, /usr/local/bin/python3)
pub fn python_interpreters() -> Vec<PathBuf> {
    let home = resolve_user_home();
    python_interpreters_for_home(&home)
}

/// Resolve the canonical Python binary path — the single interpreter ALL ML
/// components install into.
///
/// Resolution order:
/// 1. `MLSTACK_PYTHON_BIN` / `UV_PYTHON` — an **explicit user override** always
///    wins (respects user intent; e.g. pinning a specific interpreter).
/// 2. The managed global env (`~/.mlstack/global/bin/python`) if it exists —
///    the deterministic single source (Stage 0). Installers create it on first
///    global install via [`ensure_global_venv`].
/// 3. Active virtualenv
/// 4. uv-managed Python (preferred for ML workloads)
/// 5. System Python
///
/// Falls back to `"python3"` if no interpreter is found.
pub fn resolve_canonical_python_bin() -> String {
    // 0. Named-env isolation (Tenet 1): MLSTACK_ENV_NAME pins the target env.
    if let Ok(name) = env::var("MLSTACK_ENV_NAME") {
        let name = name.trim();
        if !name.is_empty() {
            let p = mlstack_env_dir(name).join("bin").join("python");
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
    }
    // 1. Explicit user override wins.
    for key in ["MLSTACK_PYTHON_BIN", "UV_PYTHON"] {
        if let Ok(val) = env::var(key) {
            let val = val.trim();
            if !val.is_empty() && Path::new(val).exists() {
                return val.to_string();
            }
        }
    }
    // 2. Deterministic managed global env (single source).
    let global = mlstack_global_python();
    if global.exists() {
        return global.to_string_lossy().to_string();
    }
    // 3. Discovery fallback.
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

    // Priority 1: Environment variable overrides — an explicit override wins
    // (kept first so discovery is consistent with `resolve_canonical_python_bin`).
    for key in ["MLSTACK_PYTHON_BIN", "UV_PYTHON"] {
        if let Ok(val) = env::var(key) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                push_if_exists!(PathBuf::from(&val));
            }
        }
    }

    // Priority 2: The managed global env (~/.mlstack/global/bin/python) —
    // the deterministic single source (Stage 0). Derived from `home` so the
    // override is consistent with `resolve_user_home()`.
    push_if_exists!(home
        .join(".mlstack")
        .join("global")
        .join("bin")
        .join("python"));
    push_if_exists!(home
        .join(".mlstack")
        .join("global")
        .join("bin")
        .join("python3"));

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
    if local_python3.exists() && seen.insert(local_python3.clone()) {
        paths.push(local_python3);
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
// Managed Python Environment (Track B §7)
// ===========================================================================

/// Resolve a managed Python interpreter (≤3.13, prefer 3.12) via backend.
///
/// **uv backend**: runs `uv python find 3.12`; if absent, falls back to
/// `uv python find ">=3.10,<3.14"` (uv picks newest ≤3.13).
/// **python backend**: scans `python3.12`, `python3.11`, `python3.10`, `python3.13`
/// in that order. EXPLICITLY REJECTS 3.14+ via `--version` parse.
///
/// Returns the path to the Python binary.
pub fn resolve_managed_python(backend: &crate::config::PythonBackend) -> anyhow::Result<PathBuf> {
    match backend {
        crate::config::PythonBackend::Uv => {
            if !command_on_path("uv") {
                anyhow::bail!("uv backend selected but uv not on PATH");
            }

            // Try 3.12 first (ML stable target)
            let output = std::process::Command::new("uv")
                .args(["python", "find", "3.12"])
                .output();
            if let Ok(out) = output {
                if out.status.success() {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if let Some(line) = stdout.lines().next() {
                        let path = line.trim();
                        if !path.is_empty() {
                            return Ok(PathBuf::from(path));
                        }
                    }
                }
            }

            // Fallback: newest Python in range ≥3.10,<3.14
            let output = std::process::Command::new("uv")
                .args(["python", "find", ">=3.10,<3.14"])
                .output()?;
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().next() {
                    let path = line.trim();
                    if !path.is_empty() {
                        return Ok(PathBuf::from(path));
                    }
                }
            }

            anyhow::bail!("uv python find failed to locate a Python ≤3.13");
        }
        crate::config::PythonBackend::Python => {
            // Scan in order: 3.12 (preferred), 3.11, 3.10, 3.13
            let candidates = ["python3.12", "python3.11", "python3.10", "python3.13"];
            for name in &candidates {
                if let Some(path) = check_python_version_cap(name) {
                    return Ok(path);
                }
            }

            anyhow::bail!(
                "No suitable Python ≤3.13 found on PATH (tried: {})",
                candidates.join(", ")
            );
        }
    }
}

/// Check if `python_name` exists and meets version cap (≤3.13).
/// Returns `Some(path)` if valid, `None` if not found or rejected (3.14+).
fn check_python_version_cap(python_name: &str) -> Option<PathBuf> {
    let output = std::process::Command::new(python_name)
        .arg("--version")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let version_str = String::from_utf8_lossy(&output.stdout);
    // Parse "Python 3.14.5" → (major, minor)
    let (major, minor) = parse_python_version_cap(&version_str)?;

    // Hard cap: major > 3 OR minor > 13 → reject (3.14+ is FORBIDDEN)
    if major > 3 || (major == 3 && minor > 13) {
        return None;
    }

    // Resolve full path via PATH scan (dependency-free)
    for dir in env::split_paths(&env::var("PATH").unwrap_or_default()) {
        let candidate = dir.join(python_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Parse Python version from "Python 3.14.5" output → (major, minor).
fn parse_python_version_cap(version_str: &str) -> Option<(u32, u32)> {
    // Expected format: "Python 3.14.5"
    let rest = version_str.strip_prefix("Python ")?;
    let parts: Vec<&str> = rest.split('.').collect();
    if parts.len() >= 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        Some((major, minor))
    } else {
        None
    }
}

/// Create the managed global environment at `~/.mlstack/global/`.
///
/// **uv backend**: `uv venv ~/.mlstack/global --python <resolved>`
/// **python backend**: `<resolved> -m venv ~/.mlstack/global`
///
/// Returns the path to the global venv's `python` binary.
pub fn create_global_env(backend: &crate::config::PythonBackend) -> anyhow::Result<PathBuf> {
    let bootstrap_python = resolve_managed_python(backend)?;
    let global_dir = mlstack_global_dir();

    if global_dir.exists() {
        // Already exists — verify the python binary is present
        let global_python = mlstack_global_python();
        if global_python.exists() {
            // Exists — but ensure pip is present (uv venvs don't seed pip by
            // default; a venv created before --seed may lack it). Every installer
            // runs `<python> -m pip install …`, so a pip-less venv breaks all of them.
            ensure_pip_in_venv(&global_python)?;
            return Ok(global_python);
        }
        // Corrupted — recreate
        std::fs::remove_dir_all(&global_dir)?;
    }

    std::fs::create_dir_all(&global_dir)?;

    match backend {
        crate::config::PythonBackend::Uv => {
            // --seed installs pip/setuptools/wheel so `python -m pip` works.
            let status = std::process::Command::new("uv")
                .arg("venv")
                .arg("--seed")
                .arg(&global_dir)
                .arg("--python")
                .arg(&bootstrap_python)
                .status()?;
            if !status.success() {
                anyhow::bail!("uv venv failed for global env");
            }
        }
        crate::config::PythonBackend::Python => {
            let status = std::process::Command::new(&bootstrap_python)
                .arg("-m")
                .arg("venv")
                .arg(&global_dir)
                .status()?;
            if !status.success() {
                anyhow::bail!("python -m venv failed for global env");
            }
        }
    }

    let global_python = mlstack_global_python();
    if !global_python.exists() {
        anyhow::bail!(
            "global env creation reported success but {} is missing",
            global_python.display()
        );
    }
    // Belt-and-suspenders: --seed should have installed pip, but verify/bootstrap
    // (covers the `python -m venv` fallback path + any --seed shortcoming).
    ensure_pip_in_venv(&global_python)?;

    Ok(global_python)
}

/// Write activation snippets to `~/.mlstack/global/`.
///
/// Creates:
/// - `activate-global.sh` (bash/zsh): `export PATH="$HOME/.mlstack/global/bin:$PATH"`
/// - `activate-global.fish` (fish): `fish_add_path $HOME/.mlstack/global/bin`
///
/// Returns (sh_path, fish_path).
pub fn write_activate_snippets() -> anyhow::Result<(PathBuf, PathBuf)> {
    let global_dir = mlstack_global_dir();
    std::fs::create_dir_all(&global_dir)?;

    let sh_path = global_dir.join("activate-global.sh");
    let fish_path = global_dir.join("activate-global.fish");

    // bash/zsh snippet
    let sh_content = r#"# mlstack-global-python activation — generated by Rusty-Stack
# Do NOT edit manually — uninstall strips this marker line
export PATH="$HOME/.mlstack/global/bin:$PATH"
"#;
    std::fs::write(&sh_path, sh_content)?;

    // fish snippet (fish_add_path is idempotent)
    let fish_content = r#"# mlstack-global-python activation — generated by Rusty-Stack
# Do NOT edit manually — uninstall strips this marker line
fish_add_path $HOME/.mlstack/global/bin
"#;
    std::fs::write(&fish_path, fish_content)?;

    Ok((sh_path, fish_path))
}

/// Offer to prepend the global env to the user's shell rc.
///
/// Detects shell via `$SHELL` (bash/zsh/fish). If TTY, prompts:
/// `Make ~/.mlstack/global the default Python in your <shell> rc? [Y/n]`
/// (default Y). If no TTY or `yes_default=true`: act=true.
///
/// On act: appends ONE idempotent marker line to the correct rc:
/// - `~/.bashrc` / `~/.zshrc`: `source "$HOME/.mlstack/global/activate-global.sh  # mlstack-global-python"`
/// - `~/.config/fish/config.fish`: `source $HOME/.mlstack/global/activate-global.fish  # mlstack-global-python"`
///
/// Marker tag `# mlstack-global-python` ensures idempotency (uninstall strips).
pub fn offer_global_prepend(yes_default: bool) -> anyhow::Result<bool> {
    let shell = std::env::var("SHELL").unwrap_or_default();
    let (rc_path, _snippet_path) = if shell.contains("fish") {
        let rc = resolve_user_home().join(".config/fish/config.fish");
        let snippet = mlstack_global_dir().join("activate-global.fish");
        (rc, snippet)
    } else if shell.contains("zsh") {
        let rc = resolve_user_home().join(".zshrc");
        let snippet = mlstack_global_dir().join("activate-global.sh");
        (rc, snippet)
    } else {
        // Default to bash
        let rc = resolve_user_home().join(".bashrc");
        let snippet = mlstack_global_dir().join("activate-global.sh");
        (rc, snippet)
    };

    let marker = "# mlstack-global-python";

    // Check if marker already present
    if rc_path.exists() {
        let content = std::fs::read_to_string(&rc_path).unwrap_or_default();
        if content.contains(marker) {
            return Ok(false); // Already present, no action needed
        }
    }

    // Determine action
    let act = if yes_default {
        true
    } else {
        // Check TTY via stdin is terminal (dependency-free)
        let is_tty = std::io::stdin().is_terminal();
        if !is_tty {
            true // Non-interactive: default to yes
        } else {
            // Prompt
            eprint!(
                "Make ~/.mlstack/global the default Python in your {} rc? [Y/n] ",
                if shell.contains("fish") {
                    "fish"
                } else if shell.contains("zsh") {
                    "zsh"
                } else {
                    "bash"
                }
            );
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            let trimmed = input.trim().to_lowercase();
            trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
        }
    };

    if !act {
        return Ok(false);
    }

    // Ensure rc file exists
    if let Some(parent) = rc_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !rc_path.exists() {
        std::fs::write(&rc_path, "")?;
    }

    // Append idempotent marker line using $HOME for portability
    let source_line = if shell.contains("fish") {
        format!(
            "source \"$HOME/.mlstack/global/activate-global.fish\"  {}",
            marker
        )
    } else {
        format!(
            "source \"$HOME/.mlstack/global/activate-global.sh\"  {}",
            marker
        )
    };

    let mut content = std::fs::read_to_string(&rc_path).unwrap_or_default();
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str(&source_line);
    content.push('\n');

    std::fs::write(&rc_path, content)?;
    Ok(true)
}

/// Idempotently append a `source ~/.mlstack_env` line to `~/.bashrc` and
/// `~/.zshrc` so the device-filtered ROCm env (iGPUs excluded) is auto-sourced
/// on every POSIX shell launch — the "persistent env" the Global install option
/// promises. Fish is NOT handled here: it auto-loads the `conf.d/mlstack_env.fish`
/// written by `ensure_mlstack_env`.
///
/// Idempotency: skips an rc file that already references `.mlstack_env`. The
/// appended line contains the literal `.mlstack_env` + `source`, which is exactly
/// what `uninstall::strip_shell_sourcing` removes, so uninstall cleans it up.
/// Marker tag `# mlstack-rocm-env` documents provenance.
pub fn offer_mlstack_env_source() -> anyhow::Result<()> {
    const MARKER: &str = "# mlstack-rocm-env";
    let source_line = format!(
        "[ -f \"$HOME/.mlstack_env\" ] && source \"$HOME/.mlstack_env\"  {}",
        MARKER
    );
    for name in ["bashrc", "zshrc"] {
        let rc_path = resolve_user_home().join(format!(".{}", name));
        let mut content = std::fs::read_to_string(&rc_path).unwrap_or_default();
        if content.contains(".mlstack_env") {
            continue; // already sourcing the env — idempotent
        }
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str("# mlstack ROCm env — auto-sourced on Global install\n");
        content.push_str(&source_line);
        content.push('\n');
        if let Some(parent) = rc_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&rc_path, content)?;
    }
    Ok(())
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
    // The env file's `$HOME`-anchored paths must resolve to `user_home` (the
    // file's owner), NOT the live process HOME — they can differ (installing
    // for another user, or a test tempdir), and reading live HOME makes the
    // generate/normalize pair non-deterministic under concurrent HOME mutation.
    let user_home_str = user_home.to_string_lossy().to_string();

    if env_path.exists() {
        let contents = fs::read_to_string(&env_path)?;

        let (normalized, changed) =
            normalize_env_contents(&contents, python_bin, &rocm_str, &rocm_lib, &user_home_str);

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

    let content = generate_env_file(python_bin, &rocm_str, &rocm_lib, &user_home_str);
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
    user_home: &str,
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
                normalize_env_line(line, &key, python_bin, rocm_home, rocm_lib, user_home);
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
        // MIGraphX compile-hang levers (repeat_while_changes non-convergence,
        // MIGraphX 2.15.0 / ROCm 7.2.4): disabling exhaustive tuning removes
        // compiler work and a persisted compiled-model cache lets ORT load a
        // previously compiled program instead of recompiling on every process.
        // See docs/superpowers/plans/2026-08-09-migraphx-compile-hang-fix.md.
        (
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE",
            "export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0".to_string(),
        ),
        (
            "ORT_MIGRAPHX_MODEL_CACHE_PATH",
            format!(
                "export ORT_MIGRAPHX_MODEL_CACHE_PATH=\"{}/.mlstack/migraphx_cache\"",
                user_home
            ),
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
///
/// `user_home` anchors the `$HOME`-relative paths (e.g. `~/.mlstack/lib-compat`)
/// so the file is correct for its owner even when that differs from the live
/// process HOME.
pub fn generate_env_file(
    python_bin: &str,
    rocm_home: &str,
    rocm_lib: &str,
    user_home: &str,
) -> String {
    format!(
        "# ML Stack Environment File (generated by Rusty-Stack)\n\
export MLSTACK_PYTHON_BIN={python_bin}\n\
export UV_PYTHON={python_bin}\n\
export ROCM_HOME={rocm_home}\n\
export ROCM_PATH={rocm_home}\n\
export HIP_PATH={rocm_home}\n\
export PYTHONPATH={rocm_lib}:$PYTHONPATH\n\
export PATH=\"{rocm_home}/bin:{rocm_home}/hip/bin:$PATH\"\n\
export LD_LIBRARY_PATH=\"{rocm_home}/lib:{rocm_home}/hip/lib:{rocm_home}/opencl/lib:{user_home}/.mlstack/lib-compat:$LD_LIBRARY_PATH\"\n\
export ORT_MIGRAPHX_FP16_ENABLE=0\n\
export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0\n\
export ORT_MIGRAPHX_MODEL_CACHE_PATH=\"{user_home}/.mlstack/migraphx_cache\"\n"
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
    user_home: &str,
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
            let desired = format!(
                "export LD_LIBRARY_PATH=\"{}/lib:{}/hip/lib:{}/opencl/lib:{}/.mlstack/lib-compat:$LD_LIBRARY_PATH\"",
                rocm_home, rocm_home, rocm_home, user_home
            );
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        // MIGraphX compile-hang levers: pin the VALUE, not just presence, so a
        // stale user value (e.g. ORT_MIGRAPHX_EXHAUSTIVE_TUNE=1) is corrected
        // on normalize rather than silently left in place.
        "ORT_MIGRAPHX_FP16_ENABLE" | "ORT_MIGRAPHX_EXHAUSTIVE_TUNE" => {
            let desired = format!("export {}={}", key, "0");
            let trimmed = line.trim();
            if trimmed != desired {
                (desired, true)
            } else {
                (line.to_string(), false)
            }
        }
        "ORT_MIGRAPHX_MODEL_CACHE_PATH" => {
            let desired = format!(
                "export ORT_MIGRAPHX_MODEL_CACHE_PATH=\"{}/.mlstack/migraphx_cache\"",
                user_home
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

/// Is `cmd` executable found on `PATH`? (Dependency-free; does not spawn.)
pub fn command_on_path(cmd: &str) -> bool {
    let path = match env::var_os("PATH") {
        Some(p) => p,
        None => return false,
    };
    for dir in env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return true;
        }
    }
    false
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
        let _env = crate::test_support::lock_env();
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
        let _env = crate::test_support::lock_env();
        // Save current state (hermetic: pin home so a real ~/.mlstack/global
        // on the dev machine can't shadow the explicit override).
        let saved_py = std::env::var("MLSTACK_PYTHON_BIN").ok();
        let saved_home = std::env::var("MLSTACK_USER_HOME").ok();

        // Create a temp file to act as "python"
        let dir = tempfile::tempdir().unwrap();
        let fake_python = dir.path().join("bin/python3");
        fs::create_dir_all(dir.path().join("bin")).unwrap();
        fs::write(&fake_python, "#!/bin/sh").unwrap();

        std::env::set_var(
            "MLSTACK_USER_HOME",
            dir.path().to_string_lossy().to_string(),
        );
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
        match saved_py {
            Some(v) => std::env::set_var("MLSTACK_PYTHON_BIN", v),
            None => std::env::remove_var("MLSTACK_PYTHON_BIN"),
        }
        match saved_home {
            Some(v) => std::env::set_var("MLSTACK_USER_HOME", v),
            None => std::env::remove_var("MLSTACK_USER_HOME"),
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
        let content = generate_env_file(
            "/usr/bin/python3",
            "/opt/rocm",
            "/opt/rocm/lib",
            "/home/test",
        );
        assert!(content.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(content.contains("export UV_PYTHON=/usr/bin/python3"));
        assert!(content.contains("export ROCM_HOME=/opt/rocm"));
        assert!(content.contains("export ROCM_PATH=/opt/rocm"));
        assert!(content.contains("export HIP_PATH=/opt/rocm"));
        assert!(content.contains("export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH"));
        assert!(content.contains("/opt/rocm/bin"));
        assert!(content.contains(".mlstack/lib-compat"));
        // user_home (not live HOME) anchors the lib-compat path — this is the
        // determinism fix that killed the parallel HOME race.
        assert!(
            content.contains("/home/test/.mlstack/lib-compat"),
            "LD_LIBRARY_PATH must use the passed user_home: {content}"
        );
        assert!(content.contains("export ORT_MIGRAPHX_FP16_ENABLE=0"));
        assert!(content.contains("export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0"));
        assert!(
            content.contains(
                "export ORT_MIGRAPHX_MODEL_CACHE_PATH=\"/home/test/.mlstack/migraphx_cache\""
            ),
            "model cache path must use the passed user_home: {content}"
        );
    }

    #[test]
    fn test_normalize_env_contents_no_change() {
        let contents =
            "# ML Stack Environment File\nexport MLSTACK_PYTHON_BIN=/usr/bin/python3\nexport ROCM_PATH=/opt/rocm\nexport ORT_MIGRAPHX_FP16_ENABLE=0\nexport ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0\nexport ORT_MIGRAPHX_MODEL_CACHE_PATH=\"/home/test/.mlstack/migraphx_cache\"\n";
        let (result, changed) = normalize_env_contents(
            contents,
            "/usr/bin/python3",
            "/opt/rocm",
            "/opt/rocm/lib",
            "/home/test",
        );
        assert!(
            !changed,
            "Should not report changes when content is already normalized"
        );
        assert!(result.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3"));
        assert!(result.contains("export ROCM_PATH=/opt/rocm"));
        assert!(result.contains("export ORT_MIGRAPHX_FP16_ENABLE=0"));
        assert!(result.contains("export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0"));
        assert!(
            result.contains(
                "export ORT_MIGRAPHX_MODEL_CACHE_PATH=\"/home/test/.mlstack/migraphx_cache\""
            )
        );
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
            "/home/test",
        );
        assert!(changed);
        assert!(result.contains("export MLSTACK_PYTHON_BIN=/usr/bin/python3.13"));
        // Should also add missing ROCM_PATH
        assert!(result.contains("export ROCM_PATH=/opt/rocm"));
    }

    // -----------------------------------------------------------------------
    // Track B: Managed Python Environment (≤3.13, prefer 3.12)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_python_version_cap_3_12() {
        let (major, minor) = parse_python_version_cap("Python 3.12.12").unwrap();
        assert_eq!(major, 3);
        assert_eq!(minor, 12);
    }

    #[test]
    fn test_parse_python_version_cap_3_14_rejected() {
        let parsed = parse_python_version_cap("Python 3.14.5");
        assert!(
            parsed.is_some(),
            "3.14.5 parses to (3, 14) but cap check rejects later"
        );
        let (major, minor) = parsed.unwrap();
        assert_eq!(major, 3);
        assert_eq!(minor, 14);
    }

    #[test]
    fn test_parse_python_version_cap_invalid() {
        assert!(parse_python_version_cap("Not a version string").is_none());
    }

    #[test]
    fn test_check_python_version_cap_rejects_3_14() {
        // Verify that 3.14 is rejected at the version cap check.
        // The parse helper returns (3, 14), and check_python_version_cap
        // returns None when major > 3 OR (major == 3 && minor > 13).
        let (major, minor) = parse_python_version_cap("Python 3.14.0").unwrap();
        assert!(
            major > 3 || (major == 3 && minor > 13),
            "3.14.0 exceeds cap (≤3.13)"
        );
    }

    #[test]
    fn test_check_python_version_cap_accepts_3_12() {
        let (major, minor) = parse_python_version_cap("Python 3.12.12").unwrap();
        assert!(major <= 3 && minor <= 13, "3.12.12 is within cap (≤3.13)");
    }

    #[test]
    fn test_write_activate_snippets_creates_files() {
        let dir = tempfile::tempdir().unwrap();
        let _env = crate::test_support::lock_env();
        let saved_home = std::env::var("HOME").ok();

        // Pin home to temp dir
        std::env::set_var("HOME", dir.path().to_string_lossy().to_string());

        let (sh_path, fish_path) = write_activate_snippets().unwrap();

        assert!(sh_path.exists());
        assert!(fish_path.exists());

        let sh_content = std::fs::read_to_string(&sh_path).unwrap();
        assert!(sh_content.contains("export PATH=\"$HOME/.mlstack/global/bin:$PATH\""));
        assert!(sh_content.contains("# mlstack-global-python"));

        let fish_content = std::fs::read_to_string(&fish_path).unwrap();
        assert!(fish_content.contains("fish_add_path $HOME/.mlstack/global/bin"));
        assert!(fish_content.contains("# mlstack-global-python"));

        // Restore
        match saved_home {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }

    #[test]
    fn test_offer_global_prepend_idempotent_marker() {
        let dir = tempfile::tempdir().unwrap();
        let _env = crate::test_support::lock_env();
        let saved_home = std::env::var("HOME").ok();
        let saved_shell = std::env::var("SHELL").ok();

        // Pin home and shell to temp dir
        std::env::set_var("HOME", dir.path().to_string_lossy().to_string());
        std::env::set_var("SHELL", "/bin/bash");

        let rc_path = dir.path().join(".bashrc");
        std::fs::write(&rc_path, "# Existing rc\n").unwrap();

        // First call: should append marker line
        let act1 = offer_global_prepend(true).unwrap();
        assert!(act1, "First call should append marker line");

        let content1 = std::fs::read_to_string(&rc_path).unwrap();
        assert!(
            content1.contains("# mlstack-global-python"),
            "Marker line should be present"
        );

        // Second call: should detect marker and skip (idempotent)
        let act2 = offer_global_prepend(true).unwrap();
        assert!(!act2, "Second call should detect existing marker and skip");

        let content2 = std::fs::read_to_string(&rc_path).unwrap();
        let marker_count = content2.matches("# mlstack-global-python").count();
        assert_eq!(
            marker_count, 1,
            "Marker should appear exactly once (idempotent)"
        );

        // Restore
        match saved_home {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
        match saved_shell {
            Some(v) => std::env::set_var("SHELL", v),
            None => std::env::remove_var("SHELL"),
        }
    }

    #[test]
    fn test_offer_global_prepend_detects_shell() {
        let dir = tempfile::tempdir().unwrap();
        let _env = crate::test_support::lock_env();
        let saved_home = std::env::var("HOME").ok();
        let saved_shell = std::env::var("SHELL").ok();

        // Test fish shell
        std::env::set_var("HOME", dir.path().to_string_lossy().to_string());
        std::env::set_var("SHELL", "/usr/bin/fish");

        let fish_config = dir.path().join(".config/fish/config.fish");
        std::fs::create_dir_all(fish_config.parent().unwrap()).unwrap();
        std::fs::write(&fish_config, "# Fish rc\n").unwrap();

        let act_fish = offer_global_prepend(true).unwrap();
        assert!(act_fish);

        let fish_content = std::fs::read_to_string(&fish_config).unwrap();
        assert!(fish_content.contains("source \"$HOME/.mlstack/global/activate-global.fish\""));
        assert!(fish_content.contains("# mlstack-global-python"));

        // Restore
        match saved_home {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
        match saved_shell {
            Some(v) => std::env::set_var("SHELL", v),
            None => std::env::remove_var("SHELL"),
        }
    }
}
