//! Uninstall orchestrator — removes the Rusty-managed ML stack so a clean
//! reinstall is possible without manual purging.
//!
//! # Tenet (v0.3.0 remediation)
//!
//! "Reinstall functionality is currently broken and DOES NOT WORK. Rusty
//! uninstall AND force-reinstall options MUST be fully functional across shells
//! (fish, bash, zsh) AND across OS's (cross Linux distributions)."
//!
//! This module existed in v0.2.0 as dead primitives (`PackageManager::purge`,
//! `purge_component_packages`) that were never wired into a user-facing flow.
//! It is now a real `uninstall_stack()` reachable via `rusty uninstall`.
//!
//! # What it removes
//!
//! 1. Python ML packages (pip) installed by Rusty components.
//! 2. ROCm/amdgpu system packages via the detected package manager (apt/dnf/
//!    pacman/zypper) — cross-distro; skipped with `--keep-rocm`.
//! 3. `/opt/rocm` (sudo) — skipped with `--keep-rocm`.
//! 4. The env files Rusty wrote (`~/.mlstack_env`, the fish `conf.d` file) and
//!    the legacy `source ~/.mlstack_env` lines from `~/.bashrc` / `~/.zshrc` /
//!    `~/.config/fish/config.fish` (cross-shell).
//! 5. The installed-component registry (`~/.mlstack/installed.json`).
//!
//! Sudo steps shell out to `sudo` (the user authenticates); they are skipped
//! (with a warning) if not root and sudo is unavailable, so the command never
//! stalls on a hidden prompt.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

use crate::core::registry::InstalledComponentRegistry;
use crate::platform::environment::{
    command_on_path, mlstack_global_dir, resolve_canonical_python_bin, resolve_user_home,
};

/// ROCm/amdgpu system-package name candidates tried across distros. The package
/// manager ignores names that aren't installed.
const ROCM_SYSTEM_PACKAGES: &[&str] = &[
    "rocm-hip-sdk",
    "rocm-opencl-sdk",
    "rocm-core",
    "rocm-dev",
    "rocm-opencl-runtime",
    "amdgpu",
    "amdgpu-dkms",
];

/// Python ML packages Rusty installs (pip). Each uninstall is non-fatal.
const ML_PIP_PACKAGES: &[&str] = &[
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton-rocm",
    "pytorch-triton",
    "vllm",
    "deepspeed",
    "megatron-core",
    "bitsandbytes",
    "flash-attn",
    "flash_attn",
    "aiter",
    "onnxruntime",
    "onnxruntime-rocm",
    "migraphx",
    "mpi4py",
    "wandb",
    "xformers",
    "accelerate",
    "transformers",
];

/// Options for [`uninstall_stack`].
#[derive(Debug, Clone, Default)]
pub struct UninstallOptions {
    /// Keep ROCm/amdgpu system packages and `/opt/rocm` (only remove Python +
    /// env/registry). Useful when reinstalling just the Python stack.
    pub keep_rocm: bool,
    /// Skip the interactive confirmation prompt.
    pub yes: bool,
    /// Also remove the `~/.mlstack/` data root (logs, cache, global venv).
    /// Off by default — preserves logs/diagnostics across reinstalls.
    pub purge_mlstack_dir: bool,
    /// Optional sudo password for the privileged steps (system-package purge,
    /// `/opt/rocm` removal). When set, privileged commands run via `sudo -A`
    /// with an askpass helper so they work without a TTY or cached credentials
    /// (which `sudo -n` cannot). When unset, falls back to `sudo -n`.
    pub sudo_password: Option<String>,
}

/// A summary of what uninstall did.
#[derive(Debug, Clone, Default)]
pub struct UninstallReport {
    pub pip_uninstall_attempted: Vec<String>,
    pub system_packages_purged: bool,
    pub opt_rocm_removed: bool,
    pub env_files_removed: Vec<String>,
    pub sourcing_lines_stripped: Vec<String>,
    pub registry_cleared: bool,
    pub warnings: Vec<String>,
}

impl UninstallReport {
    fn note(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

/// Uninstall the Rusty-managed ML stack.
///
/// See the module docs for what is removed. Never panics; sudo steps degrade
/// gracefully when not root and sudo is unavailable.
pub fn uninstall_stack(opts: &UninstallOptions) -> anyhow::Result<UninstallReport> {
    let mut report = UninstallReport::default();
    let home = resolve_user_home();

    if !opts.yes {
        // Caller (CLI) is expected to confirm; this is the non-interactive guard.
        eprintln!(
            "[uninstall] Proceeding (use --yes to skip this notice). keep_rocm={}, purge_dir={}",
            opts.keep_rocm, opts.purge_mlstack_dir
        );
    }

    // 1. pip uninstall ML packages via the canonical Python.
    let python = resolve_canonical_python_bin();

    // 1a. Remove pip's corrupted `~`-prefixed leftovers BEFORE the pip
    // uninstall, so pip's own site-packages scan doesn't emit "Ignoring
    // invalid distribution ~pkg" warnings. These are frequently root-owned
    // (interrupted `sudo pip`), so this needs the privileged path.
    clean_corrupted_pip_distributions(&python, opts.sudo_password.as_deref(), &mut report);

    println!("[uninstall] Removing ML pip packages via {python} …");
    // Union the curated list with the installed-component registry's recorded
    // pip packages (Tenet 2: the registry is now populated at install time, so
    // uninstall consults it instead of relying solely on the hardcoded list).
    let mut pkgs: Vec<String> = ML_PIP_PACKAGES.iter().map(|s| s.to_string()).collect();
    pkgs.extend(crate::core::registry::InstalledComponentRegistry::load().all_pip_packages());
    pkgs.sort();
    pkgs.dedup();
    report.pip_uninstall_attempted = pkgs.clone();
    let mut pip_args = vec![
        "-m".to_string(),
        "pip".to_string(),
        "uninstall".to_string(),
        "-y".to_string(),
        // Rusty manages these ML packages; the resolved interpreter may be a
        // uv/system Python marked EXTERNALLY-MANAGED (PEP 668) — especially
        // when the managed ~/.mlstack/global venv is absent and we fall back
        // to a discovered interpreter holding legacy installs. Override the
        // guard so the curated ML set is actually removed (mirrors the install
        // side, which uses PIP_BREAK_SYSTEM_PACKAGES=1).
        "--break-system-packages".to_string(),
    ];
    pip_args.extend(pkgs);
    let pip_status = Command::new(&python).args(&pip_args).status();
    match pip_status {
        Ok(s) if s.success() => {}
        Ok(s) => report.note(format!(
            "pip uninstall exited non-zero ({s}) — some packages may not have been installed"
        )),
        Err(e) => report.note(format!("could not run {python} -m pip ({e})")),
    }

    // 2. ROCm/amdgpu system packages (cross-distro) unless --keep-rocm.
    if !opts.keep_rocm {
        // Pre-filter to only the installed packages: apt/pacman exit non-zero
        // if ANY named package in the purge list is absent (e.g. amdgpu-dkms on
        // a partial stack), which would abort the whole purge and leave valid
        // packages (e.g. rocm-core) installed. dnf/yum tolerate missing names,
        // but filtering is harmless there too. Only purge what's present.
        let installed = filter_installed_system_packages(ROCM_SYSTEM_PACKAGES);
        if installed.is_empty() {
            report.note(
                "no ROCm/amdgpu system packages are installed — skipping system-package purge",
            );
        } else if let Some((pm, purge_args)) = build_system_purge_cmd(&installed) {
            println!("[uninstall] Purging ROCm/amdgpu system packages via {pm} …");
            let ran = run_privileged(&pm, &purge_args, opts.sudo_password.as_deref(), &mut report);
            report.system_packages_purged = ran;
        } else {
            report.note("no supported system package manager detected (apt/dnf/pacman/zypper) — skipping ROCm system-package purge");
        }

        // 3. /opt/rocm (sudo).
        if PathBuf::from("/opt/rocm").exists() {
            println!("[uninstall] Removing /opt/rocm …");
            report.opt_rocm_removed = run_privileged(
                "rm",
                &["-rf".to_string(), "/opt/rocm".to_string()],
                opts.sudo_password.as_deref(),
                &mut report,
            );
        }
    } else {
        println!("[uninstall] --keep-rocm: leaving ROCm system packages and /opt/rocm in place");
    }

    // 4. Env files + shell sourcing lines.
    remove_env_files(&home, &mut report);
    strip_shell_sourcing(&home, &mut report);

    // 5. Registry.
    let mut registry = InstalledComponentRegistry::load();
    if !registry.is_empty() {
        registry.clear();
        if let Err(e) = registry.save() {
            report.note(format!("failed to clear registry: {e}"));
        } else {
            report.registry_cleared = true;
        }
    }

    // Optional: ~/.mlstack data root.
    if opts.purge_mlstack_dir {
        let root = crate::platform::environment::mlstack_root();
        if root.exists() {
            // The global venv / cache may contain root-owned bits; try plain,
            // then sudo if needed.
            if std::fs::remove_dir_all(&root).is_ok() {
                report
                    .env_files_removed
                    .push(format!("(dir) {}", root.display()));
            } else {
                let removed = run_privileged(
                    "rm",
                    &["-rf".to_string(), root.to_string_lossy().into_owned()],
                    opts.sudo_password.as_deref(),
                    &mut report,
                );
                if removed {
                    report
                        .env_files_removed
                        .push(format!("(dir, sudo) {}", root.display()));
                }
            }
        }
    }

    Ok(report)
}

/// Build a (program, args) purge command for the detected package manager, or
/// `None` if no supported manager is on PATH.
fn build_system_purge_cmd(packages: &[&str]) -> Option<(String, Vec<String>)> {
    let (pm, verb) = if command_on_path("apt") {
        (
            "apt",
            vec!["purge".to_string(), "-y".to_string(), "-qq".to_string()],
        )
    } else if command_on_path("pacman") {
        // `-Rcn` cascade is unsafe — it removes ALL dependents, including
        // non-ROCm packages that happen to depend on ROCm libs. Instead,
        // enumerate the full ROCm package set explicitly and use `-Rn` (no
        // cascade) to guarantee only ROCm is removed.
        let full_set = enumerate_rocm_explicit_purge_set(packages);
        if full_set.is_empty() {
            // Nothing to purge (or enumeration failed — logged inside).
            return None;
        }
        let mut args = vec!["-Rn".to_string(), "--noconfirm".to_string()];
        args.extend(full_set.into_iter().map(|s| s.to_string()));
        return Some(("pacman".to_string(), args));
    } else if command_on_path("dnf") {
        ("dnf", vec!["remove".to_string(), "-y".to_string()])
    } else if command_on_path("zypper") {
        ("zypper", vec!["remove".to_string(), "-y".to_string()])
    } else if command_on_path("yum") {
        ("yum", vec!["remove".to_string(), "-y".to_string()])
    } else {
        return None;
    };
    let mut args = verb;
    args.extend(packages.iter().map(|s| s.to_string()));
    Some((pm.to_string(), args))
}

/// Enumerate the full set of ROCm packages to remove explicitly (targets +
/// their ROCm-internal dependents). Returns a sorted Vec of package names to
/// pass to `pacman -Rn`. Only includes packages matching the ROCm allowlist.
/// Non-ROCm dependents are excluded (logged to stderr).
fn enumerate_rocm_explicit_purge_set(targets: &[&str]) -> Vec<String> {
    let allowlist = rocm_allowlist_patterns();
    let mut to_remove: HashSet<String> = targets.iter().map(|s| s.to_string()).collect();
    let mut skipped_non_rocm: Vec<String> = Vec::new();

    for target in targets {
        let dependents = reverse_tree_pacman(target);
        for dep in dependents {
            if matches_rocm_allowlist(&dep, &allowlist) {
                to_remove.insert(dep);
            } else {
                skipped_non_rocm.push(dep);
            }
        }
    }

    if !skipped_non_rocm.is_empty() {
        eprintln!(
            "[uninstall] Skipping {} non-ROCm dependent(s) that would be caught by cascade: {:?}. Only ROCm packages are removed.",
            skipped_non_rocm.len(),
            skipped_non_rocm
        );
    }

    let mut sorted: Vec<_> = to_remove.into_iter().collect();
    sorted.sort();
    sorted.dedup();
    sorted
}

/// ROCm package name patterns — any package matching these prefixes/names is
/// considered part of the ROCm stack. This covers the full ROCm package
/// namespace on Arch (rocm-core, hip-*-, roc-*-, comgr, migraphx, miopen-hip,
/// rccl, rocwmma, hsa-*, amdgpu-*, libdrm_amd, etc.).
fn rocm_allowlist_patterns() -> Vec<&'static str> {
    vec![
        "rocm-",
        "hip",
        "roc",
        "hsa",
        "comgr",
        "migraphx",
        "miopen",
        "rccl",
        "amdgpu",
        "libdrm_amd",
        "rocwmma",
        "amd-comgr",
        "hsakmt",
        "hipfft",
        "hipsparse",
        "hipcub",
        "hiprand",
        "rocsolver",
        "rocprim",
        "rocrand",
        "rocfft",
        "llvm-libs",
    ]
}

/// True if pkg matches any ROCm allowlist pattern (prefix or exact match).
fn matches_rocm_allowlist(pkg: &str, allowlist: &[&str]) -> bool {
    allowlist
        .iter()
        .any(|pat| pkg.starts_with(pat) || pkg == pat.trim_end_matches('-'))
}

/// Compute the reverse dependency tree for `package` via `pactree -r`. Returns
/// all packages that depend on `package` (directly or transitively). Empty on
/// failure (pactree missing, or package not installed).
///
/// `--unique` (`-u`) is required: `pactree -r` defaults to a *tree* view with
/// box-drawing prefixes and indentation, so a naive line-by-line parse would
/// capture the structural glyphs / duplicated entries and the allowlist would
/// never see bare package names. `--unique` implies `--linear` (one package per
/// line, deduplicated) which is exactly the format the caller filters.
fn reverse_tree_pacman(package: &str) -> Vec<String> {
    if !command_on_path("pactree") {
        // pactree is part of pacman-contrib — assumed present on any ROCm
        // system (Arch installs the contrib package with base-devel).
        // If absent, we can't compute dependents safely — return empty to
        // avoid partial removal.
        eprintln!("[uninstall] pactree not found — cannot enumerate ROCm dependents for {package}. Skipping dependent enumeration.");
        return Vec::new();
    }

    Command::new("pactree")
        .args(["-r", "-u", package])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| {
            s.lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Filter `candidates` to only those that are actually installed, using the
/// detected package manager's "is installed" query. This makes the purge
/// tolerant of partial stacks: apt/pacman exit non-zero if any named package in
/// a purge list is absent, so we only pass installed names. Returns the
/// original list verbatim if no manager is detected (let the purge path emit its
/// own "no manager" note).
fn filter_installed_system_packages<'a>(candidates: &[&'a str]) -> Vec<&'a str> {
    // Return the installed candidates with the same lifetime as the input
    // (ROCM_SYSTEM_PACKAGES is &'static, so the call site gets &'static strs).
    candidates
        .iter()
        .copied()
        .filter(|pkg| is_system_package_installed(pkg))
        .collect()
}

/// Is `pkg` installed according to the detected package manager?
fn is_system_package_installed(pkg: &str) -> bool {
    // dpkg-query: exit 0 iff installed. (apt-backed distros.)
    if command_on_path("dpkg-query") {
        return Command::new("dpkg-query")
            .args(["-W", "-f=${Status}", pkg])
            .output()
            .map(|o| {
                o.status.success()
                    && String::from_utf8_lossy(&o.stdout).contains("install ok installed")
            })
            .unwrap_or(false);
    }
    // pacman -Q: exit 0 iff installed.
    if command_on_path("pacman") {
        return Command::new("pacman")
            .args(["-Q", pkg])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
    }
    // rpm -q: exit 0 iff installed (dnf/yum/zypper all back onto rpm).
    if command_on_path("rpm") {
        return Command::new("rpm")
            .args(["-q", pkg])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
    }
    // No detector: assume installed so the purge is attempted (the purge path
    // will note "no manager" if none exists).
    true
}

/// Run a privileged command: directly if root, otherwise via sudo. Returns
/// `true` if it ran successfully. Never stalls on a hidden prompt.
///
/// When `sudo_password` is supplied, the command runs as `sudo -A` with a
/// temporary askpass helper (`SUDO_ASKPASS`), so it works with no TTY and no
/// cached credentials — the case `sudo -n` cannot handle (it fails whenever a
/// password is required, e.g. a normal desktop without NOPASSWD). When no
/// password is supplied, it falls back to `sudo -n` (cached/NOPASSWD only),
/// degrading to a warning rather than hanging in CI/CD or SSH-without-TTY.
fn run_privileged(
    program: &str,
    args: &[String],
    sudo_password: Option<&str>,
    report: &mut UninstallReport,
) -> bool {
    if is_root() {
        return run_command_status(program, args, None, report);
    }
    if !command_on_path("sudo") {
        report.note(format!(
            "skipping `{program} {args:?}` — not root and sudo unavailable",
        ));
        return false;
    }
    // Prefer askpass when a password is available — it works unattended where
    // `sudo -n` cannot. The Askpass guard keeps its temp files alive for the
    // duration of the (synchronous) command below.
    if let Some(pw) = sudo_password.filter(|p| !p.is_empty()) {
        match crate::installers::common::askpass::Askpass::new(pw) {
            Ok(helper) => {
                let mut sudo_args = vec!["-A".to_string(), program.to_string()];
                sudo_args.extend_from_slice(args);
                return run_command_status("sudo", &sudo_args, Some(helper.path()), report);
            }
            Err(e) => {
                report.note(format!(
                    "askpass setup failed ({e}); falling back to sudo -n"
                ));
            }
        }
    }
    // No password (or askpass failed) → non-interactive sudo: fail fast instead
    // of prompting when no cached credentials exist.
    let mut a = vec!["-n".to_string(), program.to_string()];
    a.extend_from_slice(args);
    run_command_status("sudo", &a, None, report)
}

/// Spawn `program args` (optionally with `SUDO_ASKPASS` set) and return whether
/// it succeeded. Records a note on failure.
fn run_command_status(
    program: &str,
    args: &[String],
    askpass_path: Option<&str>,
    report: &mut UninstallReport,
) -> bool {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(p) = askpass_path {
        cmd.env("SUDO_ASKPASS", p);
    }
    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(s) => {
            report.note(format!("`{program}` exited non-zero ({s})"));
            false
        }
        Err(e) => {
            report.note(format!(
                "could not run `{program}` ({e}) — sudo may require a password (no TTY)"
            ));
            false
        }
    }
}

#[cfg(unix)]
fn is_root() -> bool {
    // SAFETY: geteuid is always safe on Unix.
    unsafe { libc::geteuid() == 0 }
}
#[cfg(not(unix))]
fn is_root() -> bool {
    false
}

/// Remove pip's corrupted `~`-prefixed leftovers from interrupted uninstalls.
///
/// pip renames a package dir to `~pkg` (and `~pkg-VERSION.dist-info`) just
/// before deleting it; if the delete is interrupted (OOM, signal, a file
/// locked under a managed uv Python), those `~` entries survive and pip emits
/// "Ignoring invalid distribution ~pkg" warnings on every later invocation.
///
/// These leftovers are frequently **root-owned** (the interrupted uninstall
/// ran under `sudo pip`), so a user-level `remove_dir_all` silently fails. We
/// try the user-level removal first and fall back to a single privileged
/// `sudo rm -rf` (via askpass) for any that remain — typically all of them on
/// a stack that was ever installed with sudo.
fn clean_corrupted_pip_distributions(
    python: &str,
    sudo_password: Option<&str>,
    report: &mut UninstallReport,
) {
    // Ask the interpreter for its site-packages dirs (global + user).
    let sites_out = Command::new(python)
        .args([
            "-c",
            "import site,sys; sys.stdout.write('\\n'.join(site.getsitepackages()+[site.getusersitepackages()]))",
        ])
        .output();
    let sites: Vec<String> = match sites_out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => return,
    };
    let mut needs_sudo: Vec<String> = Vec::new();
    for site_dir in &sites {
        let Ok(entries) = std::fs::read_dir(site_dir) else {
            continue;
        };
        for ent in entries.flatten() {
            let file_name = ent.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if !name.starts_with('~') {
                continue;
            }
            let path = ent.path();
            // Try user-level removal first (covers user-owned leftovers).
            let user_removed = match std::fs::metadata(&path) {
                Ok(m) if m.is_dir() => std::fs::remove_dir_all(&path).is_ok(),
                Ok(_) => std::fs::remove_file(&path).is_ok(),
                Err(_) => false,
            };
            if user_removed {
                report
                    .env_files_removed
                    .push(format!("(corrupted pip dist) {site_dir}/{name}"));
            } else if let Some(p) = path.to_str() {
                // User-level failed (likely root-owned) — defer to a privileged rm.
                needs_sudo.push(p.to_string());
            }
        }
    }
    if needs_sudo.is_empty() {
        return;
    }
    // Batch-remove the root-owned leftovers in one privileged transaction.
    let count = needs_sudo.len();
    let mut args = vec!["-rf".to_string()];
    args.extend(needs_sudo.iter().cloned());
    if run_privileged("rm", &args, sudo_password, report) {
        for p in &needs_sudo {
            report
                .env_files_removed
                .push(format!("(corrupted pip dist, sudo) {p}"));
        }
    } else {
        report.note(format!(
            "failed to remove {count} root-owned corrupted pip dist dir(s) — sudo required"
        ));
    }
}

/// Remove the env files Rusty wrote.
fn remove_env_files(home: &std::path::Path, report: &mut UninstallReport) {
    let targets = [
        home.join(".mlstack_env"),
        home.join(".config/fish/conf.d/mlstack_env.fish"),
        home.join(".rocm_env"),
    ];
    for t in targets {
        if t.exists() {
            match std::fs::remove_file(&t) {
                Ok(()) => report
                    .env_files_removed
                    .push(t.to_string_lossy().into_owned()),
                Err(e) => report.note(format!("could not remove {}: {e}", t.display())),
            }
        }
    }
}

/// Strip legacy `source ~/.mlstack_env` / `.mlstack_env` lines from shell rc
/// files (fish/bash/zsh). Idempotent: only rewrites if a line was actually
/// removed, and preserves the original trailing newline.
///
/// Only drops lines that actually SOURCE the env file — `source .mlstack_env`,
/// `. ~/.mlstack_env` (POSIX dot), or `bass source ...mlstack_env` (fish). A
/// bare mention of `.mlstack_env` in a comment/echo/assignment is NOT a source
/// line and is preserved. (The previous `t.contains(".")` condition was always
/// true for any `.mlstack_env` line and dropped everything mentioning it.)
fn strip_shell_sourcing(home: &std::path::Path, report: &mut UninstallReport) {
    let rcs = [
        home.join(".bashrc"),
        home.join(".zshrc"),
        home.join(".config/fish/config.fish"),
    ];
    let had_trailing_newline = |content: &str| content.ends_with('\n');
    for rc in rcs {
        let Ok(content) = std::fs::read_to_string(&rc) else {
            continue;
        };
        let mut filtered = String::with_capacity(content.len());
        let mut changed = false;
        for line in content.lines() {
            let t = line.trim();
            // A source line: it mentions .mlstack_env AND uses a source command
            // — `source`/`.` (POSIX dot-source at the start of the line)/`bass`
            // (fish wrapper). NOT a bare `.mlstack_env` mention (the old
            // `t.contains(".")` matched the dot in the filename itself).
            let is_source_line = t.contains(".mlstack_env")
                && (t.contains("source") || t.contains("bass") || t.starts_with(". "));
            if is_source_line {
                changed = true;
            } else {
                filtered.push_str(line);
                filtered.push('\n');
            }
        }
        if changed {
            // Preserve the original trailing-newline state. `content.lines()`
            // drops a trailing newline; we always re-add '\n' per line above, so
            // a file that originally ended without one would gain one — trim it
            // back to match. A file that ended with '\n' keeps it (the last
            // pushed '\n' stands in for the original terminator).
            if !had_trailing_newline(&content) && filtered.ends_with('\n') {
                filtered.pop();
            }
            let _ = std::fs::write(&rc, filtered);
            report
                .sourcing_lines_stripped
                .push(rc.to_string_lossy().into_owned());
        }
    }
}

/// Path to the managed global env (used by callers/tests).
pub fn global_env_dir() -> PathBuf {
    mlstack_global_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_system_purge_cmd_apt_when_present() {
        // Only meaningful on a Debian host; just ensure it doesn't panic and
        // returns either Some(apt...) or None.
        let cmd = build_system_purge_cmd(&["rocm-hip-sdk"]);
        if command_on_path("apt") {
            let (pm, args) = cmd.expect("apt present");
            assert_eq!(pm, "apt");
            assert!(args.contains(&"purge".to_string()));
            assert!(args.contains(&"rocm-hip-sdk".to_string()));
        }
    }

    #[test]
    fn strip_sourcing_removes_mlstack_lines() {
        let dir = tempfile::tempdir().unwrap();
        let rc = dir.path().join(".bashrc");
        std::fs::write(
            &rc,
            "# my rc\nsource \"$HOME/.mlstack_env\"\nexport FOO=bar\n. ~/.mlstack_env\n",
        )
        .unwrap();
        let mut report = UninstallReport::default();
        strip_shell_sourcing(dir.path(), &mut report);
        let after = std::fs::read_to_string(&rc).unwrap();
        assert!(
            !after.contains(".mlstack_env"),
            "sourcing lines removed: {after}"
        );
        assert!(
            after.contains("export FOO=bar"),
            "other lines preserved: {after}"
        );
        assert_eq!(report.sourcing_lines_stripped.len(), 1);
        // Trailing newline preserved (original ended with '\n').
        assert!(
            after.ends_with('\n'),
            "trailing newline preserved: {after:?}"
        );
    }

    #[test]
    fn strip_sourcing_preserves_non_source_mentions() {
        // D1 regression: a comment / echo / assignment that merely MENTIONS
        // .mlstack_env is NOT a source line and must be kept. The old
        // `t.contains(".")` matched the dot in the filename and dropped these.
        let dir = tempfile::tempdir().unwrap();
        let rc = dir.path().join(".bashrc");
        let original = "# see ~/.mlstack_env for env\necho loaded .mlstack_env\nMLSTACK_ENV_FILE=.mlstack_env\n";
        std::fs::write(&rc, original).unwrap();
        let mut report = UninstallReport::default();
        strip_shell_sourcing(dir.path(), &mut report);
        let after = std::fs::read_to_string(&rc).unwrap();
        assert_eq!(
            report.sourcing_lines_stripped.len(),
            0,
            "no source lines, no rewrite: {after:?}"
        );
        // File untouched (changed == false → no write at all).
        assert!(
            after.contains("# see ~/.mlstack_env for env"),
            "comment preserved: {after}"
        );
        assert!(
            after.contains("MLSTACK_ENV_FILE=.mlstack_env"),
            "assignment preserved: {after}"
        );
    }

    #[test]
    fn remove_env_files_deletes_targets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".mlstack_env"), "x").unwrap();
        let mut report = UninstallReport::default();
        remove_env_files(dir.path(), &mut report);
        assert!(!dir.path().join(".mlstack_env").exists());
        assert!(!report.env_files_removed.is_empty());
    }
}
