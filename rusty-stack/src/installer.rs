use crate::benchmark_logs::{
    benchmark_log_directories, extract_benchmark_json_value, find_latest_log_in_dirs,
};
use crate::component_status::{
    component_verification_commands, is_component_installed_by_id, modules_available,
    python_interpreters, python_search_paths, verification_commands, VerificationCommand,
};
use crate::config::InstallerConfig;
use crate::state::{Category, Component};
use anyhow::{bail, Context, Result};
use chrono::Local;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Receiver,
    mpsc::Sender,
    Arc, Mutex,
};
use std::time::Duration;
use std::{fs, thread};

#[derive(Debug, Clone)]
pub enum InstallerEvent {
    Log(String, bool),
    Progress {
        component_id: String,
        progress: f32,
        message: String,
    },
    ComponentStart {
        component_id: String,
        name: String,
    },
    ComponentComplete {
        component_id: String,
        success: bool,
        message: String,
    },
    VerificationReport {
        component_id: String,
        lines: Vec<String>,
    },
    Finished {
        success: bool,
    },
}

pub fn run_installation(
    components: Vec<Component>,
    config: InstallerConfig,
    sudo_password: Option<String>,
    sender: Sender<InstallerEvent>,
    input_rx: Receiver<String>,
) {
    let scripts_dir = &config.scripts_dir;
    let batch_mode = config.batch_mode;
    let install_method = config.install_method.clone();
    let total = components.len() as f32;
    let mut index = 0usize;
    let mut python_candidates = python_interpreters();
    let user_home = resolve_mlstack_user_home();
    let input_rx = Arc::new(Mutex::new(input_rx));

    match ensure_mlstack_env(&user_home, &install_method) {
        Ok(EnvUpdate::Created) => {
            let _ = sender.send(InstallerEvent::Log(
                format!("Created .mlstack_env in {}", user_home),
                false,
            ));
        }
        Ok(EnvUpdate::Updated) => {
            let _ = sender.send(InstallerEvent::Log(
                format!("Updated .mlstack_env in {}", user_home),
                false,
            ));
        }
        Ok(EnvUpdate::Unchanged) => {}
        Err(err) => {
            let _ = sender.send(InstallerEvent::Log(
                format!("Failed to ensure .mlstack_env: {}", err),
                false,
            ));
        }
    }

    // Check if any selected component needs sudo
    let any_component_needs_sudo = components.iter().any(|c| c.needs_sudo);

    if needs_sudo() && any_component_needs_sudo {
        if let Some(password) = sudo_password.clone() {
            if let Err(err) = validate_sudo(password) {
                let _ = sender.send(InstallerEvent::Log(
                    format!("Sudo validation failed: {}", err),
                    false,
                ));
                let _ = sender.send(InstallerEvent::Finished { success: false });
                return;
            }
        } else {
            let _ = sender.send(InstallerEvent::Log(
                "Sudo password missing; cannot continue".into(),
                false,
            ));
            let _ = sender.send(InstallerEvent::Finished { success: false });
            return;
        }
    }

    // Tenet 1: anchor bare-`rusty` installs to the SINGLE managed global env
    // (~/.mlstack/global) when the user did not request a named env and did not
    // pin an explicit interpreter. Previously the bare TUI fell through to a
    // *discovered* interpreter, so "global installs ALL install to a SINGLE
    // default env" only held for `rusty install --global`. Best-effort: if venv
    // creation fails (no bootstrap python), resolution falls back to discovery.
    let no_named_env = std::env::var("MLSTACK_ENV_NAME")
        .map(|v| v.trim().is_empty())
        .unwrap_or(true);
    let no_explicit_python = std::env::var("MLSTACK_PYTHON_BIN")
        .map(|v| v.trim().is_empty())
        .unwrap_or(true);
    // A "global" install = no named env requested AND no interpreter pinned.
    // Captured here (before MLSTACK_PYTHON_BIN is set below) so the post-install
    // auto-source step can gate on it.
    let is_global_install = no_named_env && no_explicit_python;
    if is_global_install {
        let bootstrap = resolve_python_bin();
        if let Ok(global_py) = crate::platform::environment::ensure_global_venv(&bootstrap) {
            std::env::set_var(
                "MLSTACK_PYTHON_BIN",
                global_py.to_string_lossy().to_string(),
            );
        } else if std::env::var("MLSTACK_PYTHON_BIN").is_err() {
            // ensure failed and nothing pinned — surface it so the user knows
            // the install is landing on a discovered interpreter, not global.
            let _ = sender.send(InstallerEvent::Log(
                "[native] could not create managed global venv (~/.mlstack/global); \
                 falling back to a discovered interpreter"
                    .into(),
                false,
            ));
        }
    }

    let mut overall_success = true;
    let env_exports = load_mlstack_env_exports(&user_home);
    let persistent_python = env_exports
        .get("MLSTACK_PYTHON_BIN")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    // Tenet 1: MLSTACK_ENV_NAME (set by `rusty install --env <name>`) isolates
    // ALL components into ~/.mlstack/envs/<name>/. Falls back to the persisted
    // target, then discovery.
    let python_bin = std::env::var("MLSTACK_ENV_NAME")
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .map(|n| {
            crate::platform::environment::mlstack_env_dir(&n)
                .join("bin")
                .join("python")
                .to_string_lossy()
                .to_string()
        })
        .or(persistent_python)
        .unwrap_or_else(resolve_python_bin);
    python_candidates.retain(|candidate| candidate != &python_bin);
    python_candidates.insert(0, python_bin.clone());
    let _ = sender.send(InstallerEvent::Log(
        format!("Selected python interpreter: {}", python_bin),
        false,
    ));

    // VAL-INSTALL-049: Sort components in dependency order so dependencies
    // are installed before dependents. Only sorts installer components;
    // verification and performance components maintain their original order.
    let components = sort_components_by_dependencies(components);

    for component in components {
        index += 1;
        let _ = sender.send(InstallerEvent::ComponentStart {
            component_id: component.id.clone(),
            name: component.name.clone(),
        });

        if component.category == Category::Maintenance && component.id.starts_with("verify-") {
            let outcome = run_verification(&component, &python_candidates, &sender, &user_home);
            let _ = sender.send(InstallerEvent::VerificationReport {
                component_id: component.id.clone(),
                lines: outcome.report_lines.clone(),
            });
            if !outcome.success {
                overall_success = false;
            }
            let _ = sender.send(InstallerEvent::ComponentComplete {
                component_id: component.id.clone(),
                success: outcome.success,
                message: if outcome.success {
                    format!("{} verification complete", component.name)
                } else {
                    format!("{} verification completed with issues", component.name)
                },
            });

            let overall_progress = index as f32 / total;
            let _ = sender.send(InstallerEvent::Progress {
                component_id: "__overall__".into(),
                progress: overall_progress,
                message: format!(
                    "Finished {} ({:.0}%)",
                    component.name,
                    overall_progress * 100.0
                ),
            });
            continue;
        }

        // VAL-INSTALL-031: For native Rust installer components, dispatch
        // to the native module instead of spawning a bash subprocess.
        let is_native = crate::installers::components::is_native_component(&component.id);

        let mut install_success = true;
        let mut error_msg = String::new();

        if is_native {
            // VAL-INSTALL-032: Dispatch to correct Rust module per component ID.
            // The native installer constructs commands and executes them directly,
            // preserving sudo behavior, environment variable injection, and
            // InstallerEvent progress reporting.
            let ctx = NativeInstallerContext {
                sudo_password: sudo_password.clone(),
                batch_mode,
                install_method: &install_method,
                sender: &sender,
                user_home: &user_home,
                env_exports: &env_exports,
                vllm_version_lag: config.vllm_version_lag,
                vllm_version_min_age_days: config.vllm_version_min_age_days,
            };
            // T2: sealed-core no-override gate + registry population.
            match registry_gate(&component, &sender) {
                Ok(false) => { /* sealed core already installed — reuse, skip reinstall */ }
                Ok(true) => {
                    if let Err(err) = run_native_installer(&component, &ctx) {
                        install_success = false;
                        let chain = err
                            .chain()
                            .map(|e| e.to_string())
                            .collect::<Vec<_>>()
                            .join(": ");
                        error_msg = format_user_friendly_error(&component.id, &chain);
                        // Log the full error chain for debugging
                        let err_label = if component.category == Category::Performance {
                            "benchmarks"
                        } else {
                            "installation"
                        };
                        let _ = sender.send(InstallerEvent::Log(
                            format!("[ERROR] {} {} failed: {}", component.name, err_label, chain),
                            false,
                        ));
                    } else {
                        // Success — record in the registry (seals core components).
                        registry_record(&component);
                    }
                }
                Err(blocked) => {
                    install_success = false;
                    error_msg = format_user_friendly_error(&component.id, &blocked.to_string());
                    let _ = sender.send(InstallerEvent::Log(
                        format!("[ERROR] {} blocked: {}", component.name, blocked),
                        false,
                    ));
                }
            }
        } else {
            // Legacy path: spawn bash subprocess for non-ported components
            // (verification, performance, or any future shell-based components)
            let script_path = format!("{}/{}", scripts_dir, component.script);
            if fs::metadata(&script_path).is_err() {
                overall_success = false;
                let _ = sender.send(InstallerEvent::ComponentComplete {
                    component_id: component.id.clone(),
                    success: false,
                    message: format!("Script not found: {}", script_path),
                });
                continue;
            }

            if let Err(err) = run_script(
                &component,
                &script_path,
                sudo_password.clone(),
                batch_mode,
                &install_method,
                &sender,
                Arc::clone(&input_rx),
            ) {
                install_success = false;
                let chain = err
                    .chain()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(": ");
                error_msg = format_user_friendly_error(&component.id, &chain);
                // Log the full error chain for debugging
                let err_label = if component.category == Category::Performance {
                    "benchmarks"
                } else {
                    "installation"
                };
                let _ = sender.send(InstallerEvent::Log(
                    format!("[ERROR] {} {} failed: {}", component.name, err_label, chain),
                    false,
                ));
            } else {
                // Tenet 2: record legacy-script installs in the registry too.
                registry_record(&component);
            }
        }

        let verify_label = if component.category == Category::Performance {
            "Validating"
        } else {
            "Verifying"
        };
        let _ = sender.send(InstallerEvent::Progress {
            component_id: component.id.clone(),
            progress: 0.8,
            message: format!("{} {}", verify_label, component.name),
        });

        // Benchmarks are their own verification — run_native_benchmark already
        // returned a definitive SUCCESS/FAILED. The file-based verification
        // below checks for benchmark log files, which the native benchmark
        // runners do not write, so it would spuriously fail a benchmark that
        // succeeded (e.g. "no benchmark logs found"). Use the benchmark's own
        // result as the verdict for Performance components.
        let verification_outcome = if component.category == Category::Performance {
            VerificationOutcome {
                success: install_success,
                report_lines: vec![format!(
                    "Benchmark {}",
                    if install_success {
                        "completed successfully"
                    } else {
                        "reported failures"
                    }
                )],
            }
        } else if is_native_maintenance_action(&component) {
            native_maintenance_action_outcome(&component, install_success)
        } else {
            run_component_verification(&component, &python_candidates, &sender, &user_home)
        };
        let _ = sender.send(InstallerEvent::VerificationReport {
            component_id: component.id.clone(),
            lines: verification_outcome.report_lines.clone(),
        });

        if !verification_outcome.success {
            overall_success = false;
        }

        // Verification can only make the verdict STRICTER, never rescue a
        // failed install. A failed install step means the operation did not
        // complete as intended; verification passing on pre-existing artifacts
        // (e.g. a stale `/opt/rocm` left by an incomplete purge) must NOT seal
        // the component as installed — otherwise a broken install is recorded
        // as good (Tenet 5: honest verification). The previous behavior sealed
        // ROCm as installed after a fully-aborted install merely because
        // rocminfo still ran against the surviving `/opt/rocm`.
        let final_success = install_success && verification_outcome.success;
        if !install_success && verification_outcome.success {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[{}] install step failed — NOT marking installed despite passing \
                     verification (verification likely ran against pre-existing/stale artifacts)",
                    component.name
                ),
                true,
            ));
        }
        if !final_success {
            overall_success = false;
        }

        // Debug logging for environment components
        if component.id == "permanent-env"
            || component.id == "enhanced-env"
            || component.id == "basic-env"
        {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[DEBUG] Component {}: install_success={}, verification_success={}, final={}",
                    component.id, install_success, verification_outcome.success, final_success
                ),
                false,
            ));
            if !install_success {
                let _ = sender.send(InstallerEvent::Log(
                    format!("[DEBUG] Install error: {}", error_msg),
                    false,
                ));
            }
        }

        let completion_msg = if !install_success {
            error_msg
        } else if component.category == Category::Performance {
            if verification_outcome.success {
                format!("{} benchmarks completed", component.name)
            } else {
                format!("{} benchmarks completed with errors", component.name)
            }
        } else if verification_outcome.success {
            format!("{} completed", component.name)
        } else {
            format!("{} completed with verification errors", component.name)
        };

        let _ = sender.send(InstallerEvent::ComponentComplete {
            component_id: component.id.clone(),
            success: final_success,
            message: completion_msg,
        });

        let overall_progress = index as f32 / total;
        let _ = sender.send(InstallerEvent::Progress {
            component_id: "__overall__".into(),
            progress: overall_progress,
            message: format!(
                "Finished {} ({:.0}%)",
                component.name,
                overall_progress * 100.0
            ),
        });
    }

    // Refresh ~/.mlstack_env from POST-INSTALL state. ensure_mlstack_env ran once
    // at the start of the run, but components install the files it derives values
    // from — e.g. /opt/rocm/.info/version is absent until the ROCm component lands,
    // so a start-of-run capture would freeze ROCM_VERSION at the fallback (7.2.0).
    // Re-running here re-derives ROCM_VERSION / GPU_ARCH / HSA_OVERRIDE /
    // HIP_VISIBLE_DEVICES / MLSTACK_PYTHON_BIN from the now-installed reality.
    // Best-effort: never fail the install over an env refresh.
    match ensure_mlstack_env(&user_home, &install_method) {
        Ok(EnvUpdate::Created) | Ok(EnvUpdate::Updated) => {
            let _ = sender.send(InstallerEvent::Log(
                "[post-install] Refreshed ~/.mlstack_env from installed state \
                 (ROCM_VERSION / GPU_ARCH / etc.)"
                    .into(),
                false,
            ));
        }
        Ok(EnvUpdate::Unchanged) => {}
        Err(e) => {
            let _ = sender.send(InstallerEvent::Log(
                format!("[post-install] Warning: could not refresh ~/.mlstack_env: {e}"),
                false,
            ));
        }
    }

    // Auto-source ~/.mlstack_env into bash/zsh (fish auto-loads the conf.d file
    // written above) on shell launch — but ONLY for a Global install. Named /
    // env-isolated installs must not mutate the global shell; their env is sourced
    // manually. Idempotent + uninstall-safe (see offer_mlstack_env_source).
    if is_global_install {
        if let Err(e) = crate::platform::environment::offer_mlstack_env_source() {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[post-install] Warning: could not auto-source ~/.mlstack_env into shell rc: {e}"
                ),
                false,
            ));
        }
    }

    if overall_success && config.star_repos {
        let _ = star_repo(&config, &sender);
    }

    // Check for reboot-required marker (set by ROCm force-reinstall)
    if std::path::Path::new("/tmp/mlstack-reboot-required").exists() {
        let _ = sender.send(InstallerEvent::Log(
            "⚠ REBOOT REQUIRED: ROCm packages were reinstalled. A reboot is needed for kernel module changes to take effect.".to_string(),
            false,
        ));
        let _ = std::fs::remove_file("/tmp/mlstack-reboot-required");
    }

    // Fix pip cache directory ownership after all installations complete.
    // Some installation steps use sudo, which can create pip cache files
    // owned by root. This prevents the unprivileged user from writing to
    // the cache, causing permission errors on subsequent pip install runs.
    {
        use crate::installers::common::fix_pip_cache_ownership;
        let result = fix_pip_cache_ownership(&user_home);
        match &result {
            crate::installers::common::PipCacheFixResult::Fixed => {
                let _ = sender.send(InstallerEvent::Log(
                    "[post-install] Fixed pip cache directory ownership (~/.cache/pip)".into(),
                    false,
                ));
            }
            crate::installers::common::PipCacheFixResult::FixFailed(err) => {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[post-install] Warning: could not fix pip cache ownership: {}",
                        err
                    ),
                    false,
                ));
            }
            _ => {
                // NoCacheDir or OwnershipCorrect — nothing to report
            }
        }
    }

    let _ = sender.send(InstallerEvent::Finished {
        success: overall_success,
    });
}

fn validate_sudo(password: String) -> Result<()> {
    let mut child = Command::new("sudo")
        .arg("-S")
        .arg("-v")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn sudo")?;

    let mut stdin = child.stdin.take().context("Failed to open stdin")?;
    stdin.write_all(password.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    drop(stdin);

    let output = child.wait_with_output()?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let detail = stderr.lines().next().unwrap_or("sudo validation failed");
    anyhow::bail!(detail.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvUpdate {
    Created,
    Updated,
    Unchanged,
}

struct VerificationCheckResult {
    status: VerificationResult,
    output: Vec<String>,
}

#[derive(Debug, Clone)]
struct VerificationOutcome {
    success: bool,
    report_lines: Vec<String>,
}

/// Create a symlink so libdrm can find amdgpu.ids at /opt/amdgpu/share/libdrm/.
///
/// ROCm packages (comgr, etc.) compile libdrm with a custom prefix of /opt/amdgpu,
/// causing libdrm to look for amdgpu.ids at /opt/amdgpu/share/libdrm/amdgpu.ids first.
/// On Arch and most distros the file lives at /usr/share/libdrm/amdgpu.ids.
/// Without the symlink, every libdrm call prints:
///   `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory`
///
/// This function tries direct filesystem creation first, then falls back to sudo.
fn fix_libdrm_amdgpu_ids() {
    let target = Path::new("/usr/share/libdrm/amdgpu.ids");
    let link = Path::new("/opt/amdgpu/share/libdrm/amdgpu.ids");

    if !target.exists() {
        return;
    }
    if link.exists() {
        return; // Already fixed
    }

    // Try direct creation (works if user has write access to /opt)
    if let Some(parent) = link.parent() {
        if fs::create_dir_all(parent).is_ok() {
            #[cfg(unix)]
            {
                let relative = compute_relative_symlink(link, target);
                if std::os::unix::fs::symlink(&relative, link).is_ok() {
                    tracing::info!(
                        "Created libdrm symlink: {} -> {}",
                        link.display(),
                        relative.display()
                    );
                    return;
                }
            }
            #[cfg(not(unix))]
            {
                let _ = (link, target);
            }
        }
    }

    // Fallback: try with sudo (non-interactive, uses cached credentials if available)
    let result = Command::new("sudo")
        .args(["mkdir", "-p", "/opt/amdgpu/share/libdrm/"])
        .stdin(std::process::Stdio::null())
        .output();

    if let Ok(out) = result {
        if out.status.success() {
            let result2 = Command::new("sudo")
                .args([
                    "ln",
                    "-sf",
                    "/usr/share/libdrm/amdgpu.ids",
                    "/opt/amdgpu/share/libdrm/amdgpu.ids",
                ])
                .stdin(std::process::Stdio::null())
                .output();
            if let Ok(out2) = result2 {
                if out2.status.success() {
                    tracing::info!(
                        "Created libdrm symlink via sudo: {} -> {}",
                        link.display(),
                        target.display()
                    );
                    return;
                }
            }
        }
    }

    // Couldn't fix — log a suggestion
    tracing::warn!(
        "Cannot create /opt/amdgpu/share/libdrm/amdgpu.ids symlink (need sudo). \
         Run manually: sudo mkdir -p /opt/amdgpu/share/libdrm && \
         sudo ln -sf /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids"
    );
}

/// Compute a relative path from `link` to `target` for symlink creation.
fn compute_relative_symlink(link: &Path, target: &Path) -> PathBuf {
    // Go up from link's parent to /, then down to target
    let mut ups = 0u32;
    for _ in link.parent().unwrap_or(Path::new(".")).ancestors().skip(1) {
        ups += 1;
    }
    let mut result = PathBuf::new();
    for _ in 0..ups.saturating_sub(1) {
        result.push("..");
    }
    result.push(target.strip_prefix("/").unwrap_or(target));
    result
}

fn ensure_mlstack_env(user_home: &str, install_method: &str) -> Result<EnvUpdate> {
    let env_path = PathBuf::from(user_home).join(".mlstack_env");
    let normalized_install_method = match install_method.trim().to_ascii_lowercase().as_str() {
        "global" => "global",
        "venv" => "venv",
        "auto" => "auto",
        _ => "auto",
    };
    let env_exports = load_mlstack_env_exports(user_home);
    let persistent_python = env_exports
        .get("MLSTACK_PYTHON_BIN")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    // Tenet 1: MLSTACK_ENV_NAME (set by `rusty install --env <name>`) isolates
    // ALL components into ~/.mlstack/envs/<name>/. Falls back to the persisted
    // target, then global managed python, then discovery.
    let python_bin = std::env::var("MLSTACK_ENV_NAME")
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .map(|n| {
            crate::platform::environment::mlstack_env_dir(&n)
                .join("bin")
                .join("python")
                .to_string_lossy()
                .to_string()
        })
        .or(persistent_python)
        .unwrap_or_else(|| {
            // Prefer managed global python (~/.mlstack/global/bin/python)
            let global_py = crate::platform::environment::mlstack_global_python();
            if global_py.exists() {
                global_py.to_string_lossy().to_string()
            } else {
                resolve_python_bin()
            }
        });

    // FAIL-LOUD: require discrete AMD GPU via sysfs detector
    let dgpus = crate::gpu::detect_discrete_amd_gpus();
    if dgpus.is_empty() {
        bail!("cannot detect AMD dGPU arch via sysfs — ensure amdgpu driver loaded");
    }
    let primary = &dgpus[0];
    let gpu_arch = primary.gfx_arch.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "could not resolve gfx arch for PCI device {} — set GPU_ARCH manually",
            primary.pci_device_id
        )
    })?;
    let discrete_indices = dgpus
        .iter()
        .enumerate()
        .map(|(i, _)| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let primary_gpu = "0";
    // Per-GPU identity emitted to the env so the benchmark renders a real GPU
    // inventory from ONE source (~/.mlstack_env) — no ad-hoc detection elsewhere.
    // VRAM from the sysfs detector (dgpus.vram_bytes); clean marketing names +
    // compute units from rocminfo (crate::gpu's lspci names are ugly PCI strings).
    let gpu_vram_gb_csv = dgpus
        .iter()
        .map(|g| match g.vram_bytes {
            Some(b) => format!("{:.0}", b as f64 / (1024.0 * 1024.0 * 1024.0)),
            None => "0".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",");
    let (gpu_names_csv, gpu_cus_csv) = gpu_identity_from_rocminfo(&dgpus);
    let gpu_pairs = [
        ("MLSTACK_GPU_NAMES", gpu_names_csv.as_str()),
        ("MLSTACK_GPU_VRAM_GB", gpu_vram_gb_csv.as_str()),
        ("MLSTACK_GPU_CUS", gpu_cus_csv.as_str()),
    ];

    // Precise HSA override: use GpuArchInfo mapping for gfx1101→11.0.1 etc.
    let hsa_override =
        crate::bootstrap::env_setup::GpuArchInfo::from_arch(gpu_arch).hsa_override_gfx_version;

    if env_path.exists() {
        let contents = fs::read_to_string(&env_path).context("Failed to read .mlstack_env")?;
        let rocm_version = detect_rocm_version();
        let rocm_home = "/opt/rocm";
        let rocm_lib = "/opt/rocm/lib";
        let defaults = [
            ("ROCM_VERSION", rocm_version.as_str()),
            ("ROCM_CHANNEL", "latest"),
            ("GPU_ARCH", gpu_arch),
            ("PYTORCH_ROCM_ARCH", gpu_arch),
            ("GPU_ARCHS", gpu_arch),
            ("ROCM_HOME", rocm_home),
            ("ROCM_PATH", rocm_home),
            ("HIP_PATH", rocm_home),
            ("HSA_OVERRIDE_GFX_VERSION", hsa_override.as_str()),
            ("HIP_VISIBLE_DEVICES", discrete_indices.as_str()),
            ("CUDA_VISIBLE_DEVICES", discrete_indices.as_str()),
            // ROCR_VISIBLE_DEVICES — the Linux ROCr (HSA) runtime filter, the
            // lowest level. AMD's guidance: on Linux use ROCR_VISIBLE_DEVICES; it
            // filters before HIP. CUDA becomes usable on AMD via HIP only when the
            // device is visible at this layer, so emit all three (HIP/CUDA/ROCR).
            ("ROCR_VISIBLE_DEVICES", discrete_indices.as_str()),
            ("PYTORCH_ROCM_DEVICE", primary_gpu),
            ("MLSTACK_PYTHON_BIN", python_bin.as_str()),
            ("UV_PYTHON", python_bin.as_str()),
            ("MLSTACK_INSTALL_METHOD", normalized_install_method),
            ("INSTALL_METHOD", normalized_install_method),
            ("PYTHONPATH", rocm_lib),
            // PYTORCH_CUDA_ALLOC_CONF — PyTorch's REAL caching-allocator var (the
            // ROCm backend reads it as-is via the HIP shim; there is no _HIP_ or
            // bare-ALLOC variant that works). Append-on-missing only — NOT
            // normalized — so a user's custom allocator config is preserved rather
            // than clobbered with this default.
            ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512"),
            // PYTORCH_HIP_ALLOC_CONF — the ROCm/HIP-canonical allocator var
            // (docs.pytorch.org/.../hip). Emitted alongside the CUDA name so the
            // config is honored regardless of which a given PyTorch build reads.
            ("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512"),
            // libdrm amdgpu ASIC id table (ROCm's amdgpu.ids lookup) — constant path.
            ("AMDGPU_ASIC_ID_TABLE_PATH", "/usr/share/libdrm/amdgpu.ids"),
            ("AMDGPU_ASIC_ID_TABLE_PATHS", "/usr/share/libdrm"),
        ];
        let (sanitized, _changed) = sanitize_mlstack_env(
            &contents,
            &defaults,
            discrete_indices.as_str(),
            rocm_home,
            python_bin.as_str(),
            normalized_install_method,
            hsa_override.as_str(),
        );
        // Per-GPU metadata (quoted — names contain spaces; the unquoted sanitize
        // path truncates at the first space). Canonical GPU info source.
        let updated = ensure_gpu_metadata_exports(&sanitized, &gpu_pairs);
        if updated != contents {
            fs::write(&env_path, &updated).context("Failed to update .mlstack_env")?;
            return Ok(EnvUpdate::Updated);
        }
        return Ok(EnvUpdate::Unchanged);
    }

    if let Some(parent) = env_path.parent() {
        fs::create_dir_all(parent).context("Failed to create home directory")?;
    }

    let rocm_version = detect_rocm_version();
    let rocm_home = "/opt/rocm";

    // Single canonical env generator (bash) — shared with the bootstrap path so
    // there is ONE source of truth for the ROCm env (device filter, arch, paths,
    // perf/triton/vLLM/MPI). Only rusty-stack install metadata is appended.
    let mut content = crate::bootstrap::env_setup::generate_env_file_content(
        &discrete_indices,
        rocm_home,
        &rocm_version,
        "latest",
        gpu_arch,
        &hsa_override,
        &python_bin,
    );
    content.push_str(&format!(
        "\n# Rusty-Stack install metadata\nexport MLSTACK_INSTALL_METHOD={}\nexport INSTALL_METHOD={}\n",
        normalized_install_method, normalized_install_method,
    ));
    // Per-GPU metadata (quoted — names contain spaces). Canonical GPU info source.
    content = ensure_gpu_metadata_exports(&content, &gpu_pairs);

    fs::write(&env_path, &content).context("Failed to create .mlstack_env")?;

    // Fish mirror — auto-loaded from ~/.config/fish/conf.d/mlstack_env.fish so
    // fish shells get the identical device-filtered ROCm env without sourcing.
    let fish_dir = PathBuf::from(user_home).join(".config/fish/conf.d");
    if fs::create_dir_all(&fish_dir).is_ok() {
        let fish_content = crate::bootstrap::env_setup::generate_fish_env_file_content(
            &discrete_indices,
            rocm_home,
            &rocm_version,
            "latest",
            gpu_arch,
            &hsa_override,
            &python_bin,
        );
        let fish_path = fish_dir.join("mlstack_env.fish");
        fs::write(&fish_path, fish_content)
            .context("Failed to write fish conf.d/mlstack_env.fish")?;
    }

    // Fix libdrm amdgpu.ids symlink for ROCm packages that look in /opt/amdgpu
    // The amdgpu.ids file is typically at /usr/share/libdrm/amdgpu.ids but
    // libdrm compiled with ROCm looks at /opt/amdgpu/share/libdrm/ first.
    fix_libdrm_amdgpu_ids();

    Ok(EnvUpdate::Created)
}

/// Clean GPU marketing names + compute-unit counts from rocminfo, for the
/// discrete GPUs (Device Type: GPU, iGPUs filtered), ordered to match
/// ROCR_VISIBLE_DEVICES. lspci descriptions (what crate::gpu's marketing_name
/// holds) are ugly PCI strings; rocminfo's Marketing Name is the clean product
/// name + Compute Unit is the CU count. Returns (names_csv, cus_csv). Falls
/// back to dgpus' lspci name + 0 CUs if rocminfo is unavailable or mismatches.
fn gpu_identity_from_rocminfo(dgpus: &[crate::gpu::AmdGpu]) -> (String, String) {
    let count = dgpus.len();
    let text =
        match std::process::Command::new(crate::installers::common::utils::resolve_rocminfo_path())
            .output()
        {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => return gpu_identity_fallback(dgpus),
        };
    let mut entries: Vec<(String, u32)> = Vec::new();
    let mut name = String::new();
    let mut cus: u32 = 0;
    let mut is_gpu = false;
    let mut started = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("Agent ") && t.contains(char::is_numeric) {
            if started && is_gpu && !name.is_empty() && !crate::gpu::is_integrated_gpu_name(&name) {
                entries.push((name.clone(), cus));
            }
            name.clear();
            cus = 0;
            is_gpu = false;
            started = false;
        } else if let Some(v) = t.strip_prefix("Marketing Name:") {
            name = v.trim().to_string();
            started = true;
        } else if let Some(v) = t.strip_prefix("Device Type:") {
            is_gpu = v.trim().contains("GPU");
            started = true;
        } else if let Some(v) = t.strip_prefix("Compute Unit:") {
            cus = v.trim().parse().unwrap_or(0);
            started = true;
        }
    }
    if started && is_gpu && !name.is_empty() && !crate::gpu::is_integrated_gpu_name(&name) {
        entries.push((name, cus));
    }
    entries.truncate(count);
    if entries.len() == count {
        let names = entries
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>()
            .join(",");
        let cus = entries
            .iter()
            .map(|(_, c)| c.to_string())
            .collect::<Vec<_>>()
            .join(",");
        (names, cus)
    } else {
        gpu_identity_fallback(dgpus)
    }
}

/// Fallback: dgpus' lspci marketing names + 0 CUs (when rocminfo is unavailable
/// or its GPU-agent count doesn't match the sysfs detector's discrete count).
fn gpu_identity_fallback(dgpus: &[crate::gpu::AmdGpu]) -> (String, String) {
    let names = dgpus
        .iter()
        .map(|g| {
            g.marketing_name
                .clone()
                .unwrap_or_else(|| format!("AMD GPU {}", g.pci_device_id))
        })
        .collect::<Vec<_>>()
        .join(",");
    let cus = dgpus
        .iter()
        .map(|_| "0".to_string())
        .collect::<Vec<_>>()
        .join(",");
    (names, cus)
}

/// Ensure `MLSTACK_GPU_NAMES` + `MLSTACK_GPU_VRAM_GB` are present as QUOTED
/// exports. GPU marketing names contain spaces ("AMD Radeon RX 7900 XTX"), and
/// the sanitize path writes `export KEY=VALUE` unquoted (truncating at the first
/// space), so these two are emitted/normalized here instead. Values are
/// comma-separated, ordered to match ROCR_VISIBLE_DEVICES. This is the canonical
/// GPU info (names + VRAM + arch via GPU_ARCH) that the benchmark + verification
/// read from the env rather than re-detecting.
fn ensure_gpu_metadata_exports(contents: &str, pairs: &[(&str, &str)]) -> String {
    let mut lines: Vec<String> = contents.lines().map(|l| l.to_string()).collect();
    for (key, value) in pairs {
        let quoted = format!("export {key}=\"{value}\"");
        let prefix = format!("export {key}=");
        match lines
            .iter()
            .position(|l| l.trim_start().starts_with(&prefix))
        {
            Some(idx) => lines[idx] = quoted,
            None => lines.push(quoted),
        }
    }
    let mut out = lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn sanitize_mlstack_env(
    contents: &str,
    defaults: &[(&str, &str)],
    gpu_list: &str,
    rocm_home: &str,
    python_bin: &str,
    install_method: &str,
    hsa_override: &str,
) -> (String, bool) {
    let mut changed = false;
    let mut lines = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Defaults lookup for keys not covered by explicit params (ROCM_VERSION,
    // GPU_ARCH, PYTORCH_ROCM_ARCH, GPU_ARCHS, ROCM_HOME, PYTORCH_ROCM_DEVICE) so
    // they are normalized — not preserved verbatim — and the env cannot drift
    // stale (e.g. ROCM_VERSION=7.2.0 frozen in the file while 7.2.4 is installed).
    let default_for = |key: &str| -> &str {
        defaults
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or("")
    };

    for line in contents.lines() {
        let trimmed = line.trim_start();
        if let Some(key) = extract_env_key(trimmed) {
            seen.insert(key.to_string());
        }
        // Stage 2: strip vars that don't belong on a ROCm stack — decided by
        // FUNCTION, not name. CUDA-named vars ROCm actually uses are KEPT
        // (CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF — PyTorch's real caching
        // allocator var, read as-is by the ROCm backend via the HIP shim). Strip:
        //   - TORCH_CUDA_ARCH_LIST: CUDA compute-capability BUILD targets — ROCm
        //     uses PYTORCH_ROCM_ARCH (gfx), so this is meaningless/harmful here.
        //   - PYTORCH_ALLOC_CONF: a non-standard no-op rename (PyTorch never
        //     dropped the _CUDA) — strip from legacy files so only the functional
        //     PYTORCH_CUDA_ALLOC_CONF remains.
        if trimmed.starts_with("export TORCH_CUDA_ARCH_LIST=")
            || trimmed.starts_with("export PYTORCH_ALLOC_CONF=")
        {
            changed = true;
            continue;
        }
        let (line, changed_hip) = normalize_env_value(line, "HIP_PATH", "/opt/rocm");
        let (line, changed_rocm) = normalize_env_value(&line, "ROCM_PATH", "/opt/rocm");
        let (line, changed_python) = normalize_env_value(&line, "MLSTACK_PYTHON_BIN", python_bin);
        let (line, changed_uv_python) = normalize_env_value(&line, "UV_PYTHON", python_bin);
        let (line, changed_method) =
            normalize_env_value(&line, "MLSTACK_INSTALL_METHOD", install_method);
        let (line, changed_method_alias) =
            normalize_env_value(&line, "INSTALL_METHOD", install_method);
        let (line, changed_hsa) =
            normalize_env_value(&line, "HSA_OVERRIDE_GFX_VERSION", hsa_override);
        let (line, changed_hip_vis) =
            normalize_visible_devices(&line, "HIP_VISIBLE_DEVICES", gpu_list);
        let (line, changed_cuda_vis) =
            normalize_visible_devices(&line, "CUDA_VISIBLE_DEVICES", gpu_list);
        let (line, changed_rocr_vis) =
            normalize_visible_devices(&line, "ROCR_VISIBLE_DEVICES", gpu_list);
        // Normalize the remaining defaults that were previously preserved verbatim
        // and could drift stale. ROCM_VERSION was the reported bug; the arch/home
        // cluster is the same class.
        let (line, changed_rocm_ver) =
            normalize_env_value(&line, "ROCM_VERSION", default_for("ROCM_VERSION"));
        let (line, changed_gpu_arch) =
            normalize_env_value(&line, "GPU_ARCH", default_for("GPU_ARCH"));
        let (line, changed_pytorch_arch) =
            normalize_env_value(&line, "PYTORCH_ROCM_ARCH", default_for("PYTORCH_ROCM_ARCH"));
        let (line, changed_gpu_archs) =
            normalize_env_value(&line, "GPU_ARCHS", default_for("GPU_ARCHS"));
        let (line, changed_rocm_home) =
            normalize_env_value(&line, "ROCM_HOME", default_for("ROCM_HOME"));
        let (line, changed_pytorch_dev) = normalize_env_value(
            &line,
            "PYTORCH_ROCM_DEVICE",
            default_for("PYTORCH_ROCM_DEVICE"),
        );
        let (line, changed_pythonpath) = if line.starts_with("export PYTHONPATH=") {
            let desired = format!("export PYTHONPATH={}/lib:$PYTHONPATH", rocm_home);
            if line.trim() != desired {
                (desired, true)
            } else {
                (line, false)
            }
        } else {
            (line, false)
        };

        // Enforce safe PATH and LD_LIBRARY_PATH
        let (line, changed_path) = if line.starts_with("export PATH=") {
            let desired = format!("export PATH=\"/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:{}/bin:{}/hip/bin:$PATH\"", rocm_home, rocm_home);
            if line.trim() != desired {
                (desired, true)
            } else {
                (line, false)
            }
        } else {
            (line, false)
        };

        let (line, changed_ld) = if line.starts_with("export LD_LIBRARY_PATH=") {
            let desired = format!(
                "export LD_LIBRARY_PATH=\"$HOME/.mlstack/libmpi-compat:$HOME/.mlstack/libmpi-compat-user-$(id -u):{}/lib:{}/hip/lib:{}/opencl/lib:$LD_LIBRARY_PATH\"",
                rocm_home, rocm_home, rocm_home
            );
            if line.trim() != desired {
                (desired, true)
            } else {
                (line, false)
            }
        } else {
            (line, false)
        };

        changed |= changed_hip
            || changed_rocm
            || changed_python
            || changed_uv_python
            || changed_method
            || changed_method_alias
            || changed_hsa
            || changed_hip_vis
            || changed_cuda_vis
            || changed_rocr_vis
            || changed_rocm_ver
            || changed_gpu_arch
            || changed_pytorch_arch
            || changed_gpu_archs
            || changed_rocm_home
            || changed_pytorch_dev
            || changed_pythonpath
            || changed_path
            || changed_ld;
        lines.push(line);
    }

    for (key, value) in defaults {
        if !seen.contains(*key) {
            lines.push(format!("export {}={}", key, value));
            changed = true;
        }
    }

    let mut output = lines.join("\n");
    output.push('\n');
    (output, changed)
}

fn extract_env_key(line: &str) -> Option<&str> {
    // Use rfind to handle both direct exports and if-guarded exports
    let export_idx = line.rfind("export ")?;
    let rest = &line[export_idx + 7..]; // skip "export "
    rest.split('=').next().map(str::trim)
}

fn normalize_env_value(line: &str, key: &str, desired: &str) -> (String, bool) {
    let marker = format!("export {}=", key);
    // Use rfind to handle both direct exports and if-guarded exports
    if let Some(export_idx) = line.rfind(&marker) {
        let after_marker = &line[export_idx + marker.len()..];
        let current = after_marker.trim().trim_matches('"');
        // Strip shell suffixes before comparing
        let current = strip_shell_suffixes_from_value(current);
        if current != desired {
            // Replace the entire line with a direct export (removing if-guard)
            return (format!("export {}={}", key, desired), true);
        }
    }
    (line.to_string(), false)
}

fn normalize_visible_devices(line: &str, key: &str, desired: &str) -> (String, bool) {
    let desired = desired.trim();
    if desired.is_empty() {
        return (line.to_string(), false);
    }

    let marker = format!("export {}=", key);
    // Use rfind to handle both direct exports and if-guarded exports
    if let Some(export_idx) = line.rfind(&marker) {
        let after_marker = &line[export_idx + marker.len()..];
        let current = after_marker.trim().trim_matches('"');
        // Strip shell suffixes before comparing
        let current = strip_shell_suffixes_from_value(current);

        // Always update if current differs from desired
        // This handles:
        // 1. "0" -> "0,1" (adding more GPUs)
        // 2. "0,1,2" -> "0,1" (filtering out iGPUs)
        // 3. "0,1" -> "0,1" (no change)
        if current != desired {
            // Replace the entire line with a direct export (removing if-guard)
            return (format!("export {}={}", key, desired), true);
        }
    }
    (line.to_string(), false)
}

fn detect_rocm_version() -> String {
    if let Ok(version) = fs::read_to_string("/opt/rocm/.info/version") {
        return version.trim().to_string();
    }
    "7.2.0".to_string()
}

fn run_verification(
    component: &Component,
    python_candidates: &[String],
    sender: &Sender<InstallerEvent>,
    user_home: &str,
) -> VerificationOutcome {
    let steps = verification_commands(&component.id, python_candidates);
    let mut results: HashMap<String, VerificationCheckResult> = HashMap::new();
    let mut executed = 0usize;
    let mut failed = 0usize;

    for step in &steps {
        if let Some((benchmark_ok, benchmark_output)) = verify_benchmark_target(&step.target_id) {
            executed += 1;
            if benchmark_ok {
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Verified,
                        output: benchmark_output,
                    },
                );
            } else {
                failed += 1;
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Failed,
                        output: benchmark_output,
                    },
                );
                let _ = sender.send(InstallerEvent::Log(
                    format!("Verification failed for {}", step.label),
                    false,
                ));
            }
            continue;
        }

        if !is_component_installed_by_id(&step.target_id, python_candidates) {
            results.insert(
                step.target_id.clone(),
                VerificationCheckResult {
                    status: VerificationResult::Missing,
                    output: vec!["Component not found in path or site-packages".to_string()],
                },
            );
            let _ = sender.send(InstallerEvent::Log(
                format!("Skipping {} (not installed)", step.label),
                false,
            ));
            continue;
        }
        if !modules_available(&step.modules, python_candidates) {
            results.insert(
                step.target_id.clone(),
                VerificationCheckResult {
                    status: VerificationResult::Missing,
                    output: vec![format!(
                        "Required Python modules missing: {}",
                        step.modules.join(", ")
                    )],
                },
            );
            let _ = sender.send(InstallerEvent::Log(
                format!("Skipping {} (module not available)", step.label),
                false,
            ));
            continue;
        }
        executed += 1;
        let _ = sender.send(InstallerEvent::Log(
            format!("Verifying {}...", step.label),
            false,
        ));
        match run_verification_command(step, sender) {
            Ok((success, output)) => {
                if success {
                    results.insert(
                        step.target_id.clone(),
                        VerificationCheckResult {
                            status: VerificationResult::Verified,
                            output,
                        },
                    );
                } else {
                    failed += 1;
                    results.insert(
                        step.target_id.clone(),
                        VerificationCheckResult {
                            status: VerificationResult::Failed,
                            output,
                        },
                    );
                    let _ = sender.send(InstallerEvent::Log(
                        format!("Verification failed for {}", step.label),
                        false,
                    ));
                }
            }
            Err(err) => {
                failed += 1;
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Failed,
                        output: vec![err.to_string()],
                    },
                );
                let _ = sender.send(InstallerEvent::Log(
                    format!("Failed to execute verification for {}: {}", step.label, err),
                    false,
                ));
            }
        }
    }

    let report_lines = build_verification_report(python_candidates, user_home, &results, &steps);
    let success = executed > 0 && failed == 0;

    if executed == 0 {
        let _ = sender.send(InstallerEvent::Log(
            "No installed components found to verify".into(),
            false,
        ));
    }

    if failed > 0 {
        let _ = sender.send(InstallerEvent::Log(
            format!("Verification completed with {} failed checks", failed),
            false,
        ));
    }

    VerificationOutcome {
        success,
        report_lines,
    }
}

fn build_verification_report(
    python_candidates: &[String],
    user_home: &str,
    results: &HashMap<String, VerificationCheckResult>,
    steps: &[VerificationCommand],
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("Environment:".to_string());
    for (key, value) in collect_env_info(user_home) {
        lines.push(format!("  {}: {}", key, value));
    }

    lines.push("".to_string());
    lines.push("Python Interpreters:".to_string());
    for py in python_candidates {
        lines.push(format!("  - {}", py));
    }

    lines.push("".to_string());
    lines.push("Checks:".to_string());
    for step in steps {
        if let Some(res) = results.get(&step.target_id) {
            lines.push(format!("  - {}: {}", step.label, res.status.label()));
            for out_line in &res.output {
                lines.push(format!("    > {}", out_line));
            }
        } else {
            lines.push(format!("  - {}: Skipped", step.label));
        }
    }

    lines
}

fn run_component_verification(
    component: &Component,
    python_candidates: &[String],
    sender: &Sender<InstallerEvent>,
    user_home: &str,
) -> VerificationOutcome {
    let steps = component_verification_commands(&component.id, python_candidates);
    let mut results: HashMap<String, VerificationCheckResult> = HashMap::new();
    let mut executed = 0usize;
    let mut failed = 0usize;

    for step in &steps {
        if let Some((benchmark_ok, benchmark_output)) = verify_benchmark_target(&step.target_id) {
            executed += 1;
            if benchmark_ok {
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Verified,
                        output: benchmark_output,
                    },
                );
            } else {
                failed += 1;
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Failed,
                        output: benchmark_output,
                    },
                );
            }
            continue;
        }

        if !modules_available(&step.modules, python_candidates) {
            results.insert(
                step.target_id.clone(),
                VerificationCheckResult {
                    status: VerificationResult::Missing,
                    output: vec![format!(
                        "Required Python modules missing: {}",
                        step.modules.join(", ")
                    )],
                },
            );
            continue;
        }
        executed += 1;
        match run_verification_command(step, sender) {
            Ok((success, output)) => {
                if success {
                    results.insert(
                        step.target_id.clone(),
                        VerificationCheckResult {
                            status: VerificationResult::Verified,
                            output,
                        },
                    );
                } else {
                    failed += 1;
                    results.insert(
                        step.target_id.clone(),
                        VerificationCheckResult {
                            status: VerificationResult::Failed,
                            output,
                        },
                    );
                }
            }
            Err(err) => {
                failed += 1;
                results.insert(
                    step.target_id.clone(),
                    VerificationCheckResult {
                        status: VerificationResult::Failed,
                        output: vec![err.to_string()],
                    },
                );
            }
        }
    }

    let report_lines = build_component_report(component, user_home, &results, &steps);
    let success = executed > 0 && failed == 0;

    VerificationOutcome {
        success,
        report_lines,
    }
}

fn is_native_maintenance_action(component: &Component) -> bool {
    component.category == Category::Maintenance && !component.id.starts_with("verify-")
}

fn native_maintenance_action_outcome(
    component: &Component,
    install_success: bool,
) -> VerificationOutcome {
    VerificationOutcome {
        success: install_success,
        report_lines: vec![format!(
            "{} native action {}",
            component.name,
            if install_success {
                "reported success"
            } else {
                "reported failure"
            }
        )],
    }
}

fn build_component_report(
    component: &Component,
    user_home: &str,
    results: &HashMap<String, VerificationCheckResult>,
    steps: &[VerificationCommand],
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Component: {}", component.name));

    for (key, value) in collect_env_info(user_home) {
        lines.push(format!("  {}: {}", key, value));
    }

    // Add benchmark results for benchmark components
    if let Some(benchmark_lines) = extract_benchmark_results(&component.id) {
        lines.push("".to_string());
        lines.push("Benchmark Results:".to_string());
        lines.extend(benchmark_lines);
    }

    lines.push("".to_string());
    lines.push("Verification Checks:".to_string());

    for step in steps {
        if let Some(res) = results.get(&step.target_id) {
            lines.push(format!("  - {}: {}", step.label, res.status.label()));
            for out_line in &res.output {
                lines.push(format!("    > {}", out_line));
            }
        } else {
            lines.push(format!("  - {}: Skipped", step.label));
        }
    }

    lines
}

fn extract_benchmark_results(component_id: &str) -> Option<Vec<String>> {
    let log_dirs = benchmark_log_directories();
    if log_dirs.is_empty() {
        return None;
    }

    // Map component IDs to log file patterns
    let pattern = match component_id {
        "mlperf-inference" => "all_benchmarks",
        "rocm-benchmarks" => "gpu-capability_benchmarks",
        "gpu-memory-bandwidth" => "memory-bandwidth_benchmarks",
        "rocm-smi-bench" => "gpu-capability_benchmarks",
        "pytorch-performance" => "pytorch_performance",
        "vllm-performance" => "vllm_benchmarks",
        "deepspeed-performance" => "deepspeed_benchmarks",
        "megatron-performance" => "megatron_benchmarks",
        "onnx-performance" => "onnx_benchmarks",
        "all-benchmarks" => "all_benchmarks",
        _ => return None,
    };

    let log_file = find_latest_log_in_dirs(&log_dirs, pattern)?;
    let contents = std::fs::read_to_string(log_file).ok()?;

    let Some(json) = extract_benchmark_json_value(&contents) else {
        return Some(vec!["  No metrics found in log".to_string()]);
    };

    let mut result_lines = benchmark_result_lines_from_json(&json);
    if result_lines.is_empty() {
        result_lines.push("  No metrics found in log".to_string());
    }

    Some(result_lines)
}

fn benchmark_result_lines_from_json(json: &serde_json::Value) -> Vec<String> {
    let mut result_lines = Vec::new();

    // Handle nested results object (for "all" benchmarks) or a single
    // benchmark's scalar metric object persisted under `results`.
    if let Some(results) = json.get("results").and_then(|r| r.as_object()) {
        let nested_suite = results.values().any(|value| {
            value.as_object().is_some_and(|obj| {
                obj.contains_key("success")
                    || obj.contains_key("metrics")
                    || obj.contains_key("execution_time_ms")
            })
        });
        if nested_suite {
            for (bench_name, bench_data) in results {
                if bench_data.is_object() {
                    let status = bench_data
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .map(|ok| if ok { "SUCCESS" } else { "FAILED" })
                        .unwrap_or("UNKNOWN");
                    let time_ms = bench_data
                        .get("execution_time_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    result_lines.push(format!(
                        "  === {} ({}, {} ms) ===",
                        bench_name.replace('_', " ").to_uppercase(),
                        status,
                        time_ms
                    ));
                    if let Some(metrics) = bench_data.get("metrics").and_then(|m| m.as_object()) {
                        append_benchmark_metric_lines(&mut result_lines, metrics, "    ");
                    }
                    append_benchmark_error_lines(&mut result_lines, bench_data, "    ");
                    result_lines.push(String::new());
                }
            }
        } else {
            append_benchmark_metric_lines(&mut result_lines, results, "  ");
            append_benchmark_error_lines(&mut result_lines, json, "  ");
        }
    } else if let Some(metrics) = json.get("metrics").and_then(|m| m.as_object()) {
        // Handle single benchmark results
        append_benchmark_metric_lines(&mut result_lines, metrics, "  ");
        append_benchmark_error_lines(&mut result_lines, json, "  ");
    }

    result_lines
}

fn append_benchmark_metric_lines(
    lines: &mut Vec<String>,
    metrics: &serde_json::Map<String, serde_json::Value>,
    indent: &str,
) {
    for (key, value) in metrics {
        let value_str = match value {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Array(arr) => format!("{} values", arr.len()),
            _ => continue,
        };
        let formatted_key = key.replace('_', " ").to_uppercase();
        lines.push(format!("{}{}: {}", indent, formatted_key, value_str));
    }
}

fn append_benchmark_error_lines(lines: &mut Vec<String>, value: &serde_json::Value, indent: &str) {
    if let Some(errors) = value.get("errors").and_then(|v| v.as_array()) {
        for err in errors.iter().filter_map(|v| v.as_str()).take(3) {
            lines.push(format!("{}ERROR: {}", indent, err));
        }
    }
}

fn benchmark_pattern_for_target(target_id: &str) -> Option<&'static str> {
    match target_id {
        "mlperf-inference" => Some("all_benchmarks"),
        "rocm-benchmarks" => Some("gpu-capability_benchmarks"),
        "gpu-memory-bandwidth" => Some("memory-bandwidth_benchmarks"),
        "rocm-smi-bench" => Some("gpu-capability_benchmarks"),
        "pytorch-performance" => Some("pytorch_performance"),
        "vllm-performance" => Some("vllm_benchmarks"),
        "deepspeed-performance" => Some("deepspeed_benchmarks"),
        "megatron-performance" => Some("megatron_benchmarks"),
        "onnx-performance" => Some("onnx_benchmarks"),
        "all-benchmarks" => Some("all_benchmarks"),
        _ => None,
    }
}

fn benchmark_success_and_errors(value: &serde_json::Value) -> (bool, Vec<String>) {
    if let Some(success) = value.get("success").and_then(|v| v.as_bool()) {
        let mut details = Vec::new();
        if !success {
            if let Some(reason) = value.get("reason").and_then(|v| v.as_str()) {
                let reason = reason.trim();
                if !reason.is_empty() {
                    details.push(reason.to_string());
                }
            }
            if let Some(errors) = value.get("errors").and_then(|v| v.as_array()) {
                for err in errors.iter().filter_map(|entry| entry.as_str()).take(4) {
                    let trimmed = err.trim();
                    if !trimmed.is_empty() {
                        details.push(trimmed.to_string());
                    }
                }
            }
            if details.is_empty() {
                details.push("Benchmark JSON reported success=false".to_string());
            }
        }
        return (success, details);
    }

    if let Some(results) = value.get("results").and_then(|v| v.as_object()) {
        let mut failed = Vec::new();
        let mut seen_result = false;
        for (name, item) in results {
            if let Some(item_success) = item.get("success").and_then(|v| v.as_bool()) {
                seen_result = true;
                if !item_success {
                    let mut reason = item
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| format!("{name}: success=false"));
                    if reason == format!("{name}: success=false") {
                        if let Some(errors) = item.get("errors").and_then(|v| v.as_array()) {
                            if let Some(first) = errors.iter().filter_map(|v| v.as_str()).next() {
                                let first = first.trim();
                                if !first.is_empty() {
                                    reason = format!("{name}: {first}");
                                }
                            }
                        }
                    }
                    failed.push(reason);
                }
            }
        }
        if seen_result {
            return (failed.is_empty(), failed);
        }
    }

    (
        false,
        vec!["Benchmark JSON does not contain a recognizable success field".to_string()],
    )
}

fn verify_benchmark_target(target_id: &str) -> Option<(bool, Vec<String>)> {
    let pattern = benchmark_pattern_for_target(target_id)?;
    let log_dirs = benchmark_log_directories();
    if log_dirs.is_empty() {
        return Some((
            false,
            vec!["No benchmark log directories found".to_string()],
        ));
    }

    let Some(log_file) = find_latest_log_in_dirs(&log_dirs, pattern) else {
        return Some((
            false,
            vec![format!("No benchmark logs found for pattern '{pattern}'")],
        ));
    };

    let mut output = vec![format!("Latest benchmark log: {}", log_file.display())];
    let contents = match std::fs::read_to_string(&log_file) {
        Ok(contents) => contents,
        Err(err) => {
            output.push(format!("Could not read benchmark log: {err}"));
            return Some((false, output));
        }
    };

    let Some(json) = extract_benchmark_json_value(&contents) else {
        output.push("Could not parse benchmark JSON payload from log".to_string());
        return Some((false, output));
    };

    let (success, mut details) = benchmark_success_and_errors(&json);
    output.append(&mut details);
    Some((success, output))
}

fn collect_env_info(user_home: &str) -> Vec<(String, String)> {
    let mut values = load_mlstack_env_exports(user_home);

    let keys = [
        "ROCM_VERSION",
        "ROCM_CHANNEL",
        "GPU_ARCH",
        "GPU_ARCHS",
        "PYTORCH_ROCM_ARCH",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTORCH_ROCM_DEVICE",
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCM_HOME",
        "ROCM_PATH",
        "HIP_PATH",
        "MLSTACK_PYTHON_BIN",
        "PYTHONPATH",
    ];

    for key in keys {
        if !values.contains_key(key) {
            if let Ok(value) = std::env::var(key) {
                values.insert(key.to_string(), value);
            }
        }
    }

    keys.iter()
        .map(|key| {
            let default_val = if *key == "ROCM_HOME" || *key == "ROCM_PATH" || *key == "HIP_PATH" {
                "/opt/rocm".to_string()
            } else {
                "unknown".to_string()
            };
            (
                key.to_string(),
                values.get(*key).cloned().unwrap_or(default_val),
            )
        })
        .collect()
}

fn load_mlstack_env_exports(user_home: &str) -> HashMap<String, String> {
    let mut values: HashMap<String, String> = HashMap::new();
    for home in candidate_mlstack_homes(user_home) {
        let env_path = PathBuf::from(home).join(".mlstack_env");
        if let Ok(contents) = fs::read_to_string(&env_path) {
            for line in contents.lines() {
                if let Some((key, value)) = parse_env_export(line) {
                    values.entry(key).or_insert(value);
                }
            }
        }
    }

    values
}

fn parse_env_export(line: &str) -> Option<(String, String)> {
    let export_idx = line.rfind("export ")?;
    let rest = &line[export_idx + 7..];
    let mut parts = rest.splitn(2, '=');
    let key = parts.next()?.trim().to_string();
    let value = parts.next()?.trim().trim_matches('"').to_string();
    // Strip shell suffixes from if-guard lines: "; fi", "&& ..."
    let value = strip_shell_suffixes_from_value(&value);
    Some((key, value))
}

/// Strip shell command suffixes from an env file value.
///
/// The `.mlstack_env` file uses if-guard patterns:
/// `if [ -z "${VAR:-}" ]; then export VAR=value; fi`
///
/// When parsing the `export VAR=value` part, the value may contain
/// the trailing `; fi`. This function strips it.
fn strip_shell_suffixes_from_value(value: &str) -> String {
    let mut v = value.to_string();
    // Strip trailing "; fi" (from if-guard lines)
    if let Some(idx) = v.find("; fi") {
        v.truncate(idx);
    }
    // Strip trailing "&& ..." (from chained commands)
    if let Some(idx) = v.find("&& ") {
        v.truncate(idx);
    }
    v.trim().to_string()
}

fn detect_onnx_rocm_provider(python: &str) -> Option<String> {
    let output = Command::new(python)
        .arg("-c")
        .arg(
            "import onnxruntime, pathlib; base=pathlib.Path(onnxruntime.__file__).parent; libs=list(base.rglob('libonnxruntime_providers_rocm.so')); print(libs[0] if libs else '')",
        )
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn push_unique_env_path(paths: &mut Vec<String>, value: &str) {
    let candidate = value.trim();
    if candidate.is_empty() {
        return;
    }
    if paths.iter().any(|existing| existing == candidate) {
        return;
    }
    paths.push(candidate.to_string());
}

fn user_home_from_passwd(user_name: &str) -> Option<String> {
    let target = user_name.trim();
    if target.is_empty() {
        return None;
    }
    let passwd = fs::read_to_string("/etc/passwd").ok()?;
    passwd.lines().find_map(|line| {
        if line.trim().is_empty() || line.starts_with('#') {
            return None;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 || fields[0] != target {
            return None;
        }
        let home = fields[5].trim();
        if home.is_empty() {
            None
        } else {
            Some(home.to_string())
        }
    })
}

fn passwd_homes_with_mlstack() -> Vec<String> {
    let mut homes = Vec::new();
    let Ok(passwd) = fs::read_to_string("/etc/passwd") else {
        return homes;
    };

    for line in passwd.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 {
            continue;
        }
        let home = fields[5].trim();
        if home.is_empty() {
            continue;
        }
        if Path::new(home).join(".mlstack").is_dir()
            || Path::new(home).join(".mlstack_env").is_file()
        {
            push_unique_env_path(&mut homes, home);
        }
    }

    homes
}

fn candidate_mlstack_homes(preferred_home: &str) -> Vec<String> {
    let mut homes = Vec::new();
    push_unique_env_path(&mut homes, preferred_home);

    if let Ok(value) = std::env::var("MLSTACK_USER_HOME") {
        push_unique_env_path(&mut homes, &value);
    }
    if let Ok(value) = std::env::var("HOME") {
        push_unique_env_path(&mut homes, &value);
    }

    for key in ["SUDO_USER", "USER", "LOGNAME"] {
        if let Ok(user_name) = std::env::var(key) {
            if let Some(home) = user_home_from_passwd(&user_name) {
                push_unique_env_path(&mut homes, &home);
            }
        }
    }

    for home in passwd_homes_with_mlstack() {
        push_unique_env_path(&mut homes, &home);
    }

    homes
}

fn resolve_mlstack_user_home() -> String {
    let fallback = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let candidates = candidate_mlstack_homes(&fallback);

    for home in &candidates {
        if Path::new(home).join(".mlstack_env").is_file() {
            return home.clone();
        }
    }
    for home in &candidates {
        if Path::new(home).join(".mlstack").is_dir() {
            return home.clone();
        }
    }

    candidates.into_iter().next().unwrap_or(fallback)
}

fn mlstack_mpi_compat_dirs(user_home: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    for home in candidate_mlstack_homes(user_home) {
        let mlstack_dir = PathBuf::from(home).join(".mlstack");

        let primary = mlstack_dir.join("libmpi-compat");
        if primary.is_dir() {
            dirs.push(primary.to_string_lossy().to_string());
        }

        if let Ok(entries) = fs::read_dir(&mlstack_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                    continue;
                };
                if name.starts_with("libmpi-compat-user-") {
                    dirs.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    dirs.sort();
    dirs.dedup();
    dirs
}

fn build_verification_ld_library_path(user_home: &str) -> String {
    let mut paths: Vec<String> = Vec::new();

    for compat_dir in mlstack_mpi_compat_dirs(user_home) {
        push_unique_env_path(&mut paths, &compat_dir);
    }

    for rocm_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib", "/opt/rocm/opencl/lib"] {
        if Path::new(rocm_path).exists() {
            push_unique_env_path(&mut paths, rocm_path);
        }
    }

    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        for part in existing.split(':') {
            push_unique_env_path(&mut paths, part);
        }
    }

    paths.join(":")
}

fn run_verification_command(
    step: &VerificationCommand,
    sender: &Sender<InstallerEvent>,
) -> Result<(bool, Vec<String>)> {
    let user_home = resolve_mlstack_user_home();
    let user_name = std::env::var("USER").unwrap_or_else(|_| "user".into());
    let env_exports = load_mlstack_env_exports(&user_home);

    let mut command = Command::new(&step.program);
    command.args(&step.args);

    let extra_paths = python_search_paths();
    let mut pythonpath = std::env::var("PYTHONPATH").unwrap_or_default();
    if !extra_paths.is_empty() {
        if !pythonpath.is_empty() {
            pythonpath.push(':');
        }
        pythonpath.push_str(&extra_paths.join(":"));
    }

    // Clean PYTHONPATH of any /tmp paths to avoid importing from build dirs
    pythonpath = pythonpath
        .split(':')
        .filter(|part| !part.starts_with("/tmp/"))
        .collect::<Vec<_>>()
        .join(":");

    if !pythonpath.is_empty() {
        command.env("PYTHONPATH", pythonpath);
    }

    let verification_ld = build_verification_ld_library_path(&user_home);
    if !verification_ld.is_empty() {
        command.env("LD_LIBRARY_PATH", verification_ld);
    }

    if step.target_id == "onnx" {
        if let Some(lib) = detect_onnx_rocm_provider(&step.program) {
            command.env("ORT_ROCM_EP_PROVIDER_PATH", lib);
        }
    }

    let selected_python = if !step.modules.is_empty() {
        step.program.clone()
    } else {
        resolve_python_bin()
    };
    let gpu_arch = env_exports
        .get("GPU_ARCH")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            detect_gpu_arch().unwrap_or_else(|| {
                eprintln!("Warning: no GPU arch detected, falling back to gfx1100");
                "gfx1100".to_string()
            })
        });
    let py_rocm_arch = env_exports
        .get("PYTORCH_ROCM_ARCH")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| gpu_arch.clone());
    let mut gpu_archs = env_exports
        .get("GPU_ARCHS")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| gpu_arch.clone());
    if step.target_id == "aiter" {
        gpu_archs = py_rocm_arch
            .split(';')
            .next()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| gpu_arch.clone());
    }
    let mut hip_visible = env_exports
        .get("HIP_VISIBLE_DEVICES")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let mut cuda_visible = env_exports
        .get("CUDA_VISIBLE_DEVICES")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| hip_visible.clone());
    let mut rocr_visible = env_exports
        .get("ROCR_VISIBLE_DEVICES")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| hip_visible.clone());
    if step.target_id == "aiter" {
        let base_list = hip_visible
            .clone()
            .or_else(|| cuda_visible.clone())
            .unwrap_or_else(detect_gpu_list);
        let ordered = prioritize_gpu_list_for_arch(&base_list, &gpu_archs);
        hip_visible = Some(ordered.clone());
        cuda_visible = Some(ordered.clone());
        rocr_visible = Some(ordered);
    }
    let hsa_override = env_exports
        .get("HSA_OVERRIDE_GFX_VERSION")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    command
        .env("HOME", &user_home)
        .env("USER", &user_name)
        .env("LOGNAME", &user_name)
        .env("MLSTACK_USER_HOME", &user_home)
        .env("MLSTACK_PYTHON_BIN", &selected_python)
        .env("UV_PYTHON", &selected_python)
        .env("PYTHONWARNINGS", "ignore")
        .env("PYTHONUNBUFFERED", "1")
        .env("ROCM_HOME", "/opt/rocm")
        .env("HIP_PATH", "/opt/rocm")
        .env("GPU_ARCH", &gpu_arch)
        .env("PYTORCH_ROCM_ARCH", &py_rocm_arch)
        .env("GPU_ARCHS", &gpu_archs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(value) = hip_visible {
        command.env("HIP_VISIBLE_DEVICES", value);
    }
    if let Some(value) = cuda_visible {
        command.env("CUDA_VISIBLE_DEVICES", value);
    }
    if let Some(value) = rocr_visible {
        command.env("ROCR_VISIBLE_DEVICES", value);
    }
    if let Some(value) = hsa_override {
        command.env("HSA_OVERRIDE_GFX_VERSION", value);
    }
    if step.target_id == "aiter" {
        let aiter_jit_dir = PathBuf::from(&user_home)
            .join(".mlstack")
            .join("aiter")
            .join("jit");
        let _ = fs::create_dir_all(&aiter_jit_dir);
        command.env("AITER_JIT_DIR", aiter_jit_dir);
    }

    if step.target_id == "mpi4py" {
        // Verification imports should avoid GPU-aware OpenMPI accelerator paths that can segfault.
        command
            .env("OMPI_MCA_btl", "^smcuda")
            .env("OMPI_MCA_opal_cuda_support", "0")
            .env("OMPI_MCA_opal_accelerator_rocm_enable", "0");
    }

    let output = command
        .output()
        .with_context(|| format!("Failed to run verification for {}", step.label))?;

    let mut captured_output = Vec::new();
    const MAX_CAPTURED_OUTPUT_LINES: usize = 24;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for raw in stdout.replace('\r', "\n").lines() {
        let clean_line = sanitize_output_line(raw);
        if should_suppress_verification_line(&clean_line, &step.target_id) {
            continue;
        }
        if !clean_line.trim().is_empty() && captured_output.len() < MAX_CAPTURED_OUTPUT_LINES {
            let line = format!("{}: {}", step.label, clean_line);
            let _ = sender.send(InstallerEvent::Log(line.clone(), false));
            captured_output.push(clean_line);
        }
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    for raw in stderr.replace('\r', "\n").lines() {
        let clean_line = sanitize_output_line(raw);
        if should_suppress_verification_line(&clean_line, &step.target_id) {
            continue;
        }
        if !clean_line.trim().is_empty() && captured_output.len() < MAX_CAPTURED_OUTPUT_LINES {
            let line = format!("{}: {}", step.label, clean_line);
            let _ = sender.send(InstallerEvent::Log(line.clone(), false));
            captured_output.push(clean_line);
        }
    }

    if output.status.success() && captured_output.len() == MAX_CAPTURED_OUTPUT_LINES {
        captured_output.push("Output truncated".to_string());
    }

    Ok((output.status.success(), captured_output))
}

// ===========================================================================
// Dependency-aware component sorting (VAL-INSTALL-049)
// ===========================================================================

/// Sort installer components in topological order so dependencies are
/// installed before dependents. Verification and performance components
/// maintain their original relative order at the end.
///
/// # Validation Assertions
///
/// - **VAL-INSTALL-048**: Dependency graph is acyclic
/// - **VAL-INSTALL-049**: installer.rs respects dependency ordering
fn sort_components_by_dependencies(components: Vec<Component>) -> Vec<Component> {
    use crate::installers::components::topological_sort;

    // Separate installer components from verification/performance components
    let mut installer_comps: Vec<Component> = Vec::new();
    let mut other_comps: Vec<Component> = Vec::new();

    for comp in components {
        if comp.category == Category::Maintenance || comp.category == Category::Performance {
            other_comps.push(comp);
        } else {
            installer_comps.push(comp);
        }
    }

    // Get IDs for topological sort
    let ids: Vec<String> = installer_comps.iter().map(|c| c.id.clone()).collect();

    match topological_sort(&ids) {
        Ok(sorted_ids) => {
            // Reorder installer_comps to match sorted IDs
            let mut reordered = Vec::with_capacity(installer_comps.len());
            for id in &sorted_ids {
                if let Some(pos) = installer_comps.iter().position(|c| c.id == *id) {
                    reordered.push(installer_comps.remove(pos));
                }
            }
            // Append any remaining (shouldn't happen if topological_sort is correct)
            reordered.extend(installer_comps);

            // Append verification/performance components at the end
            reordered.extend(other_comps);
            reordered
        }
        Err(_) => {
            // If topological sort fails (cycle), keep original order
            // but still append other components at the end
            installer_comps.extend(other_comps);
            installer_comps
        }
    }
}

// ===========================================================================
// Native Rust installer dispatch (VAL-INSTALL-031, VAL-INSTALL-032)
// ===========================================================================

/// Context for native installer execution. Groups related parameters
/// to keep function signatures under clippy's 7-argument threshold.
struct NativeInstallerContext<'a> {
    sudo_password: Option<String>,
    batch_mode: bool,
    install_method: &'a str,
    sender: &'a Sender<InstallerEvent>,
    user_home: &'a str,
    env_exports: &'a HashMap<String, String>,
    /// vLLM source-build supply-chain gate (version lag + min age), from config.
    vllm_version_lag: u32,
    vllm_version_min_age_days: u32,
}

// ---------------------------------------------------------------------------
// Command execution primitives (VAL-INSTALL-031)
// ---------------------------------------------------------------------------

/// A command that can be executed natively without bash.
///
/// This enum wraps the different command types produced by installer modules
/// (`ShellCommand`, `PipCommand`, `SystemCommand`, `PackageCommand`) into a
/// single type for uniform execution.
enum NativeCommand {
    /// A shell command (program + args + env vars + optional working dir).
    Shell {
        program: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        working_dir: Option<PathBuf>,
    },
    /// A pip/uv install command (program + args, no extra env).
    Pip { program: String, args: Vec<String> },
    /// A system package command (program + args, no extra env).
    System { program: String, args: Vec<String> },
    /// A package manager command (program + args, may already include sudo).
    Package { program: String, args: Vec<String> },
}

impl NativeCommand {
    /// Create a NativeCommand from any type that exposes `program`, `args`, and `env` fields.
    /// This works with all component-specific ShellCommand types since they all share
    /// the same field layout: `program: String`, `args: Vec<String>`, `env: Vec<(String, String)>`.
    fn from_shell_cmd(program: &str, args: &[String], env: &[(String, String)]) -> Self {
        NativeCommand::Shell {
            program: program.to_string(),
            args: args.to_vec(),
            env: env.to_vec(),
            working_dir: None,
        }
    }

    /// Create a NativeCommand from a ShellCommand that has a working_dir.
    fn from_shell_cmd_with_dir(
        program: &str,
        args: &[String],
        env: &[(String, String)],
        working_dir: Option<PathBuf>,
    ) -> Self {
        NativeCommand::Shell {
            program: program.to_string(),
            args: args.to_vec(),
            env: env.to_vec(),
            working_dir,
        }
    }
}

/// Execute a native command, streaming output to the event sender.
///
/// This function:
/// - Converts the command to `std::process::Command`
/// - Wraps with `sudo -S -p ''` when `sudo_pw` is `Some` and the command
///   does not already include sudo (VAL-INSTALL-033)
/// - Injects environment variables from the command struct (VAL-INSTALL-034)
/// - Spawns the process and streams stdout/stderr line-by-line as
///   `InstallerEvent::Log` (VAL-INSTALL-035)
/// - Checks exit status and returns `Err` on non-zero (VAL-INSTALL-036)
///
/// # Validation Assertions
/// Check if command output indicates a package is already up to date.
///
/// yay/pacman output patterns that should be treated as success even with
/// non-zero exit codes:
/// - `warning: <pkg> is up to date -- skipping`
/// - `there is nothing to do`
/// - `already installed`
fn is_up_to_date_output(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("up to date -- skipping")
        || lower.contains("up to date, skipping")
        || lower.contains("up to date -- reinstalling")
        || lower.contains("there is nothing to do")
        || lower.contains("already installed")
        || (lower.contains("warning:") && lower.contains("up to date"))
}

///
/// - **VAL-INSTALL-031**: No `Command::new("bash")` for native components
/// - **VAL-INSTALL-033**: sudo behavior preserved
/// - **VAL-INSTALL-034**: Environment variable injection preserved
/// - **VAL-INSTALL-035**: InstallerEvent progress reporting preserved
fn execute_native_command(
    cmd: &NativeCommand,
    sudo_pw: Option<&str>,
    sender: &Sender<InstallerEvent>,
    component_name: &str,
) -> Result<()> {
    let (program, args, envs, working_dir) = match cmd {
        NativeCommand::Shell {
            program,
            args,
            env,
            working_dir,
        } => (
            program.clone(),
            args.clone(),
            env.clone(),
            working_dir.clone(),
        ),
        NativeCommand::Pip { program, args } => (program.clone(), args.clone(), vec![], None),
        NativeCommand::System { program, args } => (program.clone(), args.clone(), vec![], None),
        NativeCommand::Package { program, args } => (program.clone(), args.clone(), vec![], None),
    };

    // ── No-CUDA chokepoint (hard-prime tenet) ──────────────────────────
    // Reject ANY explicit nvidia/cuda runtime package or CUDA wheel URL from
    // being pip-installed by ANY component, regardless of command type (covers
    // NativeCommand::Pip AND NativeCommand::Shell `python3 -m pip install …`).
    // Permits torch/triton (core, ROCm-managed, from the ROCm index) but
    // rejects nvidia-*/cuda*/cudnn/cublas/.../cupy-cuda* and +cuXXX wheels.
    // Per-component `--no-deps` + requirements filtering handle the transitive
    // case; this is the universal, unbypassable explicit-package guard.
    let invokes_pip = program == "uv"
        || program.ends_with("pip")
        || program.ends_with("pip3")
        || args.iter().any(|a| a == "pip");
    let is_install = args.iter().any(|a| a == "install");
    if invokes_pip && is_install {
        let mut skip_next = false;
        for a in &args {
            if skip_next {
                // The argument following -r/--requirement/-c/--constraint is a
                // requirements/constraint FILENAME, not a package — never block
                // it (a file named `cuda.txt` is not a CUDA package).
                skip_next = false;
                continue;
            }
            if a == "-r" || a == "--requirement" || a == "-c" || a == "--constraint" {
                skip_next = true;
                continue;
            }
            if a.starts_with('-') || a == "install" || a == "." || a == "-e" || a.is_empty() {
                continue;
            }
            if crate::installers::common::nvidia_blocklist::is_nvidia_cuda_runtime_package(a) {
                let _ = sender.send(InstallerEvent::Log(
                    format!("[native] BLOCKED nvidia/cuda package in {component_name}: {a}"),
                    true,
                ));
                anyhow::bail!(
                    "NVIDIA/CUDA contamination blocked: component '{component_name}' attempted \
                     to pip install '{a}' — violates the No-CUDA hard-prime tenet"
                );
            }
            if a.starts_with("http")
                && crate::installers::common::nvidia_blocklist::is_cuda_wheel_url(a)
            {
                let _ = sender.send(InstallerEvent::Log(
                    format!("[native] BLOCKED cuda wheel URL in {component_name}: {a}"),
                    true,
                ));
                anyhow::bail!("NVIDIA/CUDA wheel URL blocked in {component_name}: {a}");
            }
        }
    }

    // Build the std::process::Command, wrapping with sudo if needed.
    // Package commands may already include "sudo" as the program.
    let already_sudo = program == "sudo";
    let mut command = if let Some(_pw) = sudo_pw {
        if already_sudo {
            // Command already has sudo — just inject password via stdin
            let mut c = Command::new(&program);
            c.args(&args);
            c
        } else {
            let mut c = Command::new("sudo");
            c.arg("-S").arg("-p").arg("").arg("--").arg(&program);
            c.args(&args);
            // Log that we're using sudo
            let _ = sender.send(InstallerEvent::Log(
                format!("[native] Running with sudo: {} {}", program, args.join(" ")),
                false,
            ));
            c
        }
    } else {
        let mut c = Command::new(&program);
        c.args(&args);
        c
    };

    // Inject environment variables (VAL-INSTALL-034)
    for (key, value) in &envs {
        command.env(key, value);
    }

    // Set working directory if specified (e.g., for pip install -e .)
    if let Some(ref dir) = working_dir {
        command.current_dir(dir);
    }

    // Set up stdio for streaming
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Log the command being executed
    let cmd_str = if args.is_empty() {
        program.clone()
    } else {
        format!("{} {}", program, args.join(" "))
    };
    let _ = sender.send(InstallerEvent::Log(
        format!("[native] $ {}", cmd_str),
        false,
    ));

    let mut child = command.spawn().with_context(|| {
        let mut hint = String::new();
        // Provide actionable hints for common spawn failures
        if cmd_str.starts_with("sudo") {
            hint = " — ensure sudo is installed and you have the correct permissions".to_string();
        } else if cmd_str.starts_with("pip")
            || cmd_str.starts_with("uv")
            || cmd_str.starts_with("python")
        {
            hint = " — ensure Python/pip is installed and accessible".to_string();
        } else if cmd_str.starts_with("git") {
            hint = " — ensure git is installed".to_string();
        } else if cmd_str.starts_with("cmake") || cmd_str.starts_with("make") {
            hint = " — ensure build tools (cmake, make, gcc) are installed".to_string();
        } else if cmd_str.starts_with("rocminfo") || cmd_str.starts_with("rocm-smi") {
            hint = " — ensure ROCm is installed and /opt/rocm/bin is in PATH".to_string();
        }
        format!(
            "Failed to spawn command '{}' for {}.{}",
            cmd_str, component_name, hint
        )
    })?;

    // Write sudo password to stdin if needed, then close stdin
    // Always close stdin to signal to the child that no more input is coming
    if let Some(pw) = sudo_pw {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(pw.as_bytes());
            let _ = stdin.write_all(b"\n");
            let _ = stdin.flush();
            drop(stdin);
        }
    } else {
        // Close stdin even when no sudo password to prevent child from
        // blocking on stdin read
        drop(child.stdin.take());
    }

    // Stream stdout and stderr in separate threads (VAL-INSTALL-035)
    // Also capture combined output for up-to-date pattern detection.
    let sender_stdout = sender.clone();
    let stdout_stream = match child.stdout.take() {
        Some(s) => s,
        None => {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[ERROR] {} — failed to capture stdout from spawned process",
                    component_name
                ),
                false,
            ));
            bail!(
                "{} — failed to capture stdout from spawned process. \
                 This usually indicates a system resource issue.",
                component_name
            );
        }
    };
    let stdout_captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let stdout_captured_clone = stdout_captured.clone();
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout_stream);
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    let clean = strip_ansi_codes(&l);
                    if !clean.trim().is_empty() {
                        let _ = sender_stdout.send(InstallerEvent::Log(clean.clone(), false));
                        if let Ok(mut buf) = stdout_captured_clone.lock() {
                            buf.push_str(&clean);
                            buf.push('\n');
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });

    let sender_stderr = sender.clone();
    let stderr_stream = match child.stderr.take() {
        Some(s) => s,
        None => {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[ERROR] {} — failed to capture stderr from spawned process",
                    component_name
                ),
                false,
            ));
            bail!(
                "{} — failed to capture stderr from spawned process. \
                 This usually indicates a system resource issue.",
                component_name
            );
        }
    };
    let stderr_captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let stderr_captured_clone = stderr_captured.clone();
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr_stream);
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    let clean = strip_ansi_codes(&l);
                    if !clean.trim().is_empty() {
                        let _ = sender_stderr.send(InstallerEvent::Log(clean.clone(), false));
                        if let Ok(mut buf) = stderr_captured_clone.lock() {
                            buf.push_str(&clean);
                            buf.push('\n');
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Wait for completion
    let status = child.wait().with_context(|| {
        format!(
            "{} — command process wait failed. \
             The process may have been killed or the system is under heavy load.",
            component_name
        )
    })?;
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();

    if !status.success() {
        // Check if the output indicates the package is already up to date.
        // yay/pacman may return non-zero exit codes for warnings like
        // "warning: rocminfo is up to date -- skipping" which should be
        // treated as success, not failure.
        let combined_output = {
            let stdout_buf = stdout_captured.lock().unwrap_or_else(|e| e.into_inner());
            let stderr_buf = stderr_captured.lock().unwrap_or_else(|e| e.into_inner());
            format!("{}{}", *stdout_buf, *stderr_buf)
        };

        if is_up_to_date_output(&combined_output) {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {} — package already up to date, treating as success",
                    component_name
                ),
                false,
            ));
            return Ok(());
        }

        let code = status.code().unwrap_or(-1);
        let error_detail = match code {
            1 => "General error — check the command output above for details".to_string(),
            2 => "Misuse of shell builtins — check command arguments".to_string(),
            126 => "Command not executable — check file permissions".to_string(),
            127 => "Command not found — ensure the required tool is installed".to_string(),
            130 => "Process terminated by Ctrl+C (SIGINT)".to_string(),
            137 => "Process killed (SIGKILL) — possibly out of memory".to_string(),
            139 => "Segmentation fault — this is a bug in the invoked program".to_string(),
            n if n > 128 => format!(
                "Process terminated by signal {} — check system logs",
                n - 128
            ),
            _ => format!(
                "Command exited with code {} — check the output above for details",
                code
            ),
        };
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[ERROR] {} failed: {} (exit code {} at {})",
                component_name,
                error_detail,
                code,
                Local::now().format("%H:%M:%S")
            ),
            false,
        ));
        bail!(
            "{} failed: {} (exit code {})",
            component_name,
            error_detail,
            code
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: build base env vars shared by all native installers
// ---------------------------------------------------------------------------

/// Build the common environment variables that all native installers receive.
///
/// These mirror the environment injected by `run_script()` to ensure
/// behavioral parity (VAL-INSTALL-034).
fn build_common_env(ctx: &NativeInstallerContext) -> Vec<(String, String)> {
    let mut env = Vec::new();

    // Core environment from .mlstack_env
    env.push(("HOME".into(), ctx.user_home.to_string()));
    if let Ok(user) = std::env::var("USER") {
        env.push(("USER".into(), user.clone()));
        env.push(("LOGNAME".into(), user));
    }
    env.push(("MLSTACK_USER_HOME".into(), ctx.user_home.to_string()));
    env.push((
        "MLSTACK_BATCH_MODE".into(),
        if ctx.batch_mode { "1" } else { "0" }.into(),
    ));
    env.push((
        "MLSTACK_INSTALL_METHOD".into(),
        ctx.install_method.to_string(),
    ));
    env.push(("INSTALL_METHOD".into(), ctx.install_method.to_string()));
    env.push(("RUSTY_STACK".into(), "true".into()));

    // ROCm / GPU environment from .mlstack_env exports
    for key in &[
        "ROCM_VERSION",
        "ROCM_CHANNEL",
        "GPU_ARCH",
        "GPU_ARCHS",
        "PYTORCH_ROCM_ARCH",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTORCH_ROCM_DEVICE",
        "HSA_OVERRIDE_GFX_VERSION",
        "MLSTACK_PYTHON_BIN",
        "UV_PYTHON",
    ] {
        if let Some(value) = ctx.env_exports.get(*key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                env.push((key.to_string(), trimmed.to_string()));
            }
        }
    }

    // ROCm paths (always set)
    env.push(("ROCM_HOME".into(), "/opt/rocm".into()));
    env.push(("ROCM_PATH".into(), "/opt/rocm".into()));
    env.push(("HIP_PATH".into(), "/opt/rocm".into()));
    env.push(("HIP_ROOT_DIR".into(), "/opt/rocm".into()));
    env.push(("ROCM_ROOT".into(), "/opt/rocm".into()));
    env.push(("HIPCC_BIN_DIR".into(), "/opt/rocm/bin".into()));

    // Python/pip settings
    env.push(("PIP_BREAK_SYSTEM_PACKAGES".into(), "1".into()));
    env.push(("PIP_ROOT_USER_ACTION".into(), "ignore".into()));
    env.push(("UV_PIP_BREAK_SYSTEM_PACKAGES".into(), "1".into()));
    env.push(("UV_SYSTEM_PYTHON".into(), "1".into()));
    env.push(("PYTHONUNBUFFERED".into(), "1".into()));

    env
}

// ---------------------------------------------------------------------------
// Git clone idempotency helper (fix-git-clone-idempotency)
// ---------------------------------------------------------------------------

/// Check if a directory exists and contains a `.git` subdirectory,
/// indicating it is an existing git repository.
///
/// Uses `symlink_metadata` instead of `is_dir()` to handle root-owned
/// directories that may not be readable by the current user under sudo.
fn is_existing_git_repo(target_dir: &str) -> bool {
    let path = Path::new(target_dir);
    // Use symlink_metadata which doesn't follow symlinks and works
    // even when the directory owner differs from the current user
    match std::fs::symlink_metadata(target_dir) {
        Ok(meta) if meta.is_dir() => {
            // Also use symlink_metadata for .git check to handle permission edge cases
            path.join(".git")
                .symlink_metadata()
                .map(|m| m.is_dir() || m.is_file())
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// Clone a git repository, or pull updates if the directory already exists.
///
/// This function makes git clone idempotent:
/// - If `target_dir` does **not** exist (or has no `.git` subdirectory),
///   it runs `git clone <extra_args> <repo_url> <target_dir>`.
/// - If `target_dir` exists and contains a `.git` subdirectory, it runs
///   `git -C <target_dir> pull --ff-only` to update. If that fails, it
///   falls back to `git -C <target_dir> fetch --all` followed by
///   `git -C <target_dir> reset --hard origin/HEAD`.
///
/// All commands are executed via `execute_native_command()`.
/// Fix ownership of a directory tree that may have been created by root via sudo.
///
/// When `git_clone_or_pull` runs git operations with sudo, the resulting files
/// are owned by root. This prevents subsequent user-space builds from writing
/// to the directory. This function chowns the tree back to the real user.
fn fix_directory_ownership(dir: &str, sudo_pw: Option<&str>, sender: &Sender<InstallerEvent>) {
    // Check if the directory is actually owned by root
    let stat_output = std::process::Command::new("stat")
        .args(["-c", "%U", dir])
        .output();

    let needs_fix = match stat_output {
        Ok(out) if out.status.success() => {
            let owner = String::from_utf8_lossy(&out.stdout).trim().to_string();
            owner == "root"
        }
        _ => true, // Can't determine — try anyway
    };

    if !needs_fix {
        return;
    }

    let real_user = std::env::var("SUDO_USER")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "scooter".to_string());

    let _ = sender.send(InstallerEvent::Log(
        format!("[native] Fixing ownership of {} -> {}", dir, real_user),
        false,
    ));

    if sudo_pw.is_some() {
        let chown_cmd = NativeCommand::from_shell_cmd(
            "chown",
            &[
                "-R".to_string(),
                format!("{}:{}", real_user, real_user),
                dir.to_string(),
            ],
            &[],
        );
        let _ = execute_native_command(&chown_cmd, sudo_pw, sender, "ownership-fix");
    }
    // If no sudo_pw available, the directory should already be user-owned
}

fn git_clone_or_pull(
    repo_url: &str,
    target_dir: &str,
    extra_clone_args: &[&str],
    sudo_pw: Option<&str>,
    sender: &Sender<InstallerEvent>,
    component_name: &str,
) -> Result<()> {
    if is_existing_git_repo(target_dir) {
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {} — directory '{}' already exists, pulling updates instead of cloning",
                component_name, target_dir
            ),
            false,
        ));

        // Try fast-forward pull first
        let pull_cmd = NativeCommand::from_shell_cmd(
            "git",
            &[
                "-C".to_string(),
                target_dir.to_string(),
                "pull".to_string(),
                "--ff-only".to_string(),
            ],
            &[],
        );
        let pull_result = execute_native_command(&pull_cmd, sudo_pw, sender, component_name);

        if pull_result.is_ok() {
            return Ok(());
        }

        // Pull failed — log warning and try fetch + reset
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {} — git pull --ff-only failed, trying fetch + reset --hard origin/HEAD",
                component_name
            ),
            false,
        ));

        let fetch_cmd = NativeCommand::from_shell_cmd(
            "git",
            &[
                "-C".to_string(),
                target_dir.to_string(),
                "fetch".to_string(),
                "--all".to_string(),
            ],
            &[],
        );
        let fetch_ok = execute_native_command(&fetch_cmd, sudo_pw, sender, component_name).is_ok();

        let reset_ok = if fetch_ok {
            let reset_cmd = NativeCommand::from_shell_cmd(
                "git",
                &[
                    "-C".to_string(),
                    target_dir.to_string(),
                    "reset".to_string(),
                    "--hard".to_string(),
                    "origin/HEAD".to_string(),
                ],
                &[],
            );
            execute_native_command(&reset_cmd, sudo_pw, sender, component_name).is_ok()
        } else {
            false
        };

        if fetch_ok && reset_ok {
            return Ok(());
        }

        // .git exists but pull + fetch + reset all failed → the repo is corrupt
        // or unusable (e.g. a purged/empty .git left by a prior run). Remove it
        // and re-clone fresh: the recursive call sees no .git → clone branch.
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {} — existing .git is corrupt/unusable (pull+fetch+reset failed); removing and re-cloning",
                component_name
            ),
            false,
        ));
        if std::fs::remove_dir_all(target_dir).is_err() {
            let rm_cmd = NativeCommand::from_shell_cmd(
                "rm",
                &["-rf".to_string(), target_dir.to_string()],
                &[],
            );
            let _ = execute_native_command(&rm_cmd, sudo_pw, sender, component_name);
        }
        git_clone_or_pull(
            repo_url,
            target_dir,
            extra_clone_args,
            sudo_pw,
            sender,
            component_name,
        )
    } else {
        // Directory may exist but not be a git repo, or may not exist at all.
        // If a non-empty directory exists without .git, git clone will fail with
        // "fatal: destination path already exists". Handle this by removing
        // the incomplete directory before cloning.
        let target_path = Path::new(target_dir);
        if target_path.is_dir() && !target_path.join(".git").exists() {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {} — directory '{}' exists but is not a git repo; removing incomplete clone",
                    component_name, target_dir
                ),
                false,
            ));
            // Try plain remove_dir_all first (works if user owns all files)
            if std::fs::remove_dir_all(target_dir).is_err() {
                // Fallback: sudo rm -rf for root-owned files
                let rm_cmd = NativeCommand::from_shell_cmd(
                    "rm",
                    &["-rf".to_string(), target_dir.to_string()],
                    &[],
                );
                let rm_result = execute_native_command(&rm_cmd, sudo_pw, sender, component_name);
                if let Err(e) = rm_result {
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] [WARN] {} — could not remove stale dir: {}. Clone may fail.",
                            component_name, e
                        ),
                        false,
                    ));
                }
            }
        }

        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {} — cloning '{}' into '{}'",
                component_name, repo_url, target_dir
            ),
            false,
        ));

        let mut args: Vec<String> = vec!["clone".to_string()];
        for extra in extra_clone_args {
            args.push(extra.to_string());
        }
        args.push(repo_url.to_string());
        args.push(target_dir.to_string());

        let clone_cmd = NativeCommand::from_shell_cmd("git", &args, &[]);
        execute_native_command(&clone_cmd, sudo_pw, sender, component_name)
    }
}

// ---------------------------------------------------------------------------
// Per-component dispatch (VAL-INSTALL-032)
// ---------------------------------------------------------------------------

/// Execute a native Rust installer for a ported component.
///
/// This function dispatches to the appropriate Rust installer module based
/// on the component ID. It preserves:
/// - **Sudo behavior** (VAL-INSTALL-033): Components with `needs_sudo:true`
///   still elevate privileges via `sudo -S`.
/// - **Environment variable injection** (VAL-INSTALL-034): Same env vars
///   as the bash path (ROCM_VERSION, GPU_ARCH, etc.).
/// - **InstallerEvent progress reporting** (VAL-INSTALL-035): Same event
///   types emitted (Log, Progress, ComponentStart, ComponentComplete).
/// - **Error message format** (VAL-INSTALL-036): Error messages match
///   original scripts.
///
/// # Validation Assertions
///
/// - **VAL-INSTALL-031**: No bash subprocess for ported components
///
/// Filter a pip requirements file to exclude CUDA/nvidia packages and wheels.
///
/// Delegates to the single canonical blocklist
/// [`crate::installers::common::nvidia_blocklist::filter_requirements_file`]
/// (Stage 2: one unified blocklist replaces the three former divergent copies
/// in `installer.rs`, `megatron.rs`, and `textgen.rs`). Excludes `nvidia-*`,
/// `cuda*`, `cudnn`/`cublas`/`cufft`/`curand`/`cusolver`/`cusparse`/`nccl`/
/// `nvtx`/`nvjitlink`/`tensorrt`, the torch family (ROCm-managed), and any
/// URL wheel carrying a CUDA marker (`+cu1`/`cu124`/`cu128`/…).
fn filter_cuda_requirements(req_path: &str) -> std::io::Result<String> {
    crate::installers::common::nvidia_blocklist::filter_requirements_file(req_path)
}

/// Check whether force-reinstall mode is active.
///
/// Reads `MLSTACK_FORCE_REINSTALL` first, falls back to `FORCE`.
/// Recognises common truthy values: 1, true, yes, on (case-insensitive).
fn is_force_reinstall() -> bool {
    std::env::var("MLSTACK_FORCE_REINSTALL")
        .or_else(|_| std::env::var("FORCE"))
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Build a pip uninstall command that works with both `uv` and `pip3`.
///
/// `uv` requires `uv pip uninstall` (not `uv uninstall` which doesn't exist).
/// Note: `uv pip uninstall` does NOT accept `-y` — it's non-interactive by default.
fn build_pip_uninstall(packages: &[&str]) -> NativeCommand {
    let use_uv = crate::installers::common::utils::command_exists("uv");
    if use_uv {
        let mut args = vec!["pip".to_string(), "uninstall".to_string()];
        for pkg in packages {
            args.push(pkg.to_string());
        }
        NativeCommand::Pip {
            program: "uv".to_string(),
            args,
        }
    } else {
        let mut args = vec!["uninstall".to_string(), "-y".to_string()];
        for pkg in packages {
            args.push(pkg.to_string());
        }
        NativeCommand::Pip {
            program: "pip3".to_string(),
            args,
        }
    }
}

/// Thoroughly purge a Python package and all its related packages.
///
/// This is the nuclear option for force-reinstall: removes the main package,
/// its known aliases, related dist-info directories, and common build artifacts.
/// Each uninstall is non-fatal (logged but doesn't fail the install).
fn purge_component_packages(
    packages: &[&str],
    sender: &Sender<InstallerEvent>,
    component_name: &str,
) {
    let _ = sender.send(InstallerEvent::Log(
        format!(
            "[native] {} — purging packages: {}",
            component_name,
            packages.join(", ")
        ),
        false,
    ));

    // Run the uninstall command (non-fatal)
    let uninstall_cmd = build_pip_uninstall(packages);
    let _ = execute_native_command(&uninstall_cmd, None, sender, component_name);
}

/// Purge PyTorch and all related packages.
fn purge_pytorch(sender: &Sender<InstallerEvent>, component_name: &str) {
    purge_component_packages(
        &[
            "torch",
            "torchvision",
            "torchaudio",
            "pytorch-triton-rocm",
            "pytorch-triton",
        ],
        sender,
        component_name,
    );
}

/// Purge Triton and related packages.
fn purge_triton(sender: &Sender<InstallerEvent>, component_name: &str) {
    purge_component_packages(
        &["triton", "triton-rocm", "pytorch-triton-rocm"],
        sender,
        component_name,
    );
}

/// Purge AITER and related packages.
fn purge_aiter(sender: &Sender<InstallerEvent>, component_name: &str) {
    purge_component_packages(&["aiter", "amd-aiter"], sender, component_name);
}

/// Purge Flash Attention and related packages.
fn purge_flash_attn(sender: &Sender<InstallerEvent>, component_name: &str) {
    purge_component_packages(
        &["flash-attn", "flash_attn", "flash_attention_amd"],
        sender,
        component_name,
    );
}

/// Purge vLLM and related packages.
fn purge_vllm(sender: &Sender<InstallerEvent>, component_name: &str) {
    purge_component_packages(&["vllm"], sender, component_name);
}

/// Has the user explicitly requested to unseal core components (force-reinstall
/// past the sealed-core no-override gate)? Env: `MLSTACK_UNSEAL_CORE=1`.
fn unseal_core_requested() -> bool {
    std::env::var("MLSTACK_UNSEAL_CORE")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Sealed-core no-override gate (Tenet 2). Called before each component install.
///
/// Returns:
/// - `Ok(true)`  — proceed with the install.
/// - `Ok(false)` — the component is a sealed core that is already installed;
///   REUSE it (skip reinstall) instead of overriding.
/// - `Err`       — blocked: a force-reinstall targeted a sealed core without
///   `MLSTACK_UNSEAL_CORE=1` (refuse to clobber an installed core component).
fn registry_gate(component: &Component, sender: &Sender<InstallerEvent>) -> Result<bool> {
    let registry = crate::core::registry::InstalledComponentRegistry::load();
    if !registry.is_sealed(&component.id) {
        return Ok(true);
    }
    if is_force_reinstall() && !unseal_core_requested() {
        bail!(
            "{} is a sealed CORE component — force-reinstall refused to prevent override. \
             Set MLSTACK_UNSEAL_CORE=1 to intentionally reinstall it.",
            component.name
        );
    }
    if !is_force_reinstall() {
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {} already installed (sealed core) — reusing, NOT reinstalling",
                component.name
            ),
            false,
        ));
        return Ok(false);
    }
    Ok(true)
}

/// Record a successful install in the registry (Tenet 2). Seals core AND
/// install-once components (rocm/pytorch/triton/aiter/flash-attn/…) so later
/// installs reuse them and never override. Populates source/location/pip-package
/// metadata so the registry is the real single-source-of-truth (consulted by
/// dep-reuse and uninstall), not a hollow id+seal stub.
fn registry_record(component: &Component) {
    use crate::core::registry::{InstalledComponent, InstalledComponentRegistry};
    let mut registry = InstalledComponentRegistry::load();
    let record = InstalledComponent {
        id: component.id.clone(),
        version: component_version(&component.id),
        source_index: component_source_index(&component.id),
        location: install_target_location(),
        sealed: crate::core::registry::should_seal_component(&component.id),
        installed_at: Some(chrono::Utc::now().to_rfc3339()),
        pip_packages: component_pip_packages(&component.id),
    };
    registry.mark_installed(record);
    if let Err(e) = registry.save() {
        // Tenet 2: a failed save means the component is NOT recorded, so a
        // later install could override a sealed core. Surface it loudly rather
        // than silently dropping the result.
        eprintln!("[WARN] Failed to save installed component registry: {e}");
    }
}

/// The pip package names a component owns (for registry dep-tracking +
/// uninstall enumeration). Empty for system/repo components (rocm, comfyui,
/// textgen) which don't install a single owned pip package.
fn component_pip_packages(id: &str) -> Vec<String> {
    match id {
        "pytorch" => vec!["torch", "torchvision", "torchaudio", "pytorch-triton-rocm"],
        "triton" => vec!["triton", "pytorch-triton-rocm"],
        "vllm" | "vllm-multi" => vec!["vllm"],
        "deepspeed" => vec!["deepspeed", "einops"],
        "megatron" => vec!["megatron-core"],
        "bitsandbytes" => vec!["bitsandbytes"],
        "flash-attn" | "flash-attn-triton" | "flash-attn-ck" | "flash_attention" => {
            vec!["flash-attn", "flash_attn"]
        }
        "aiter" => vec!["aiter"],
        "onnx" | "onnxruntime" => vec!["onnxruntime", "onnxruntime-rocm"],
        "migraphx" => vec!["migraphx"],
        "mpi4py" => vec!["mpi4py"],
        "wandb" => vec!["wandb"],
        "fastvideo" => vec!["fastvideo"],
        _ => Vec::new(),
    }
    .into_iter()
    .map(String::from)
    .collect()
}

/// Where a component was installed — the named env, the managed global env, or
/// the pinned interpreter. Recorded so the registry is the single source of
/// truth for "where the single source lives" (Tenet 2).
fn install_target_location() -> Option<String> {
    if let Ok(name) = std::env::var("MLSTACK_ENV_NAME") {
        let name = name.trim();
        if !name.is_empty() {
            return Some(
                crate::platform::environment::mlstack_env_dir(name)
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    let global = crate::platform::environment::mlstack_global_dir();
    if global.exists() {
        return Some(global.to_string_lossy().into_owned());
    }
    std::env::var("MLSTACK_PYTHON_BIN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The wheel index / source a component was installed from, where statically
/// known. PyTorch comes from PyTorch's own ROCm wheel index
/// (download.pytorch.org/whl/rocmX.Y); triton is the `triton-rocm` wheel bundled
/// as a PyTorch dependency; ROCm lives at /opt/rocm. `None` for components with
/// no fixed index.
fn component_source_index(id: &str) -> Option<String> {
    match id {
        "pytorch" => Some("https://download.pytorch.org/whl/rocm".to_string()),
        "triton" => Some("triton-rocm (bundled with PyTorch)".to_string()),
        "rocm" => Some("/opt/rocm".to_string()),
        _ => None,
    }
}

/// Best-effort installed version, where cheaply detectable. ROCm version from
/// `/opt/rocm/.info/version`; `None` otherwise (version is not load-bearing for
/// the seal/no-override gate — pip_packages/location/source_index are).
fn component_version(id: &str) -> Option<String> {
    match id {
        "rocm" => {
            let v = detect_rocm_version();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    }
}

/// Ensure the invoking user is a member of the given system groups, adding them
/// via `sudo usermod -aG` when missing.
///
/// Restores a step the original `install_rocm.sh` had (`sudo usermod -aG
/// render,video "$target_user"`) that the Rust port dropped: without `render`
/// membership a user cannot open `/dev/kfd` / render nodes on distros that mode
/// them `0660` (the documented ROCm requirement; works by accident on distros
/// that mode them `0666`).
///
/// System-aware + best-effort:
/// - Resolves the *real* target user via `SUDO_USER` (then `USER`), so an elevated
///   `rusty-stack` adds the invoking user, not root.
/// - Reads the user's *configured* groups (`id -nG <user>` → `/etc/group`), not the
///   running process's supplemental groups, so it's correct even under sudo.
/// - Silently skips any group the distro doesn't define (`getent group`).
/// - Never fails the install: every decision is logged via `sender`.
fn ensure_user_in_groups(
    groups: &[&str],
    sudo_pw: Option<&str>,
    sender: &Sender<InstallerEvent>,
    component_name: &str,
) {
    use std::process::Command;

    // Prefer SUDO_USER: if rusty-stack is elevated, USER==root but we want the
    // invoking user added to the groups.
    let user = std::env::var("SUDO_USER")
        .ok()
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty())
        .or_else(|| {
            std::env::var("USER")
                .ok()
                .map(|u| u.trim().to_string())
                .filter(|u| !u.is_empty())
        });
    let user = match user {
        Some(u) => u,
        None => {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {component_name} — could not determine target user; skipping \
                     group check. Add yourself manually: sudo usermod -aG {} $USER",
                    groups.join(",")
                ),
                false,
            ));
            return;
        }
    };

    // The user's configured groups (resolved from /etc/group via `id -nG`), NOT
    // the running process's supplemental groups — correct even under sudo.
    let current_groups: Vec<String> = Command::new("id")
        .args(["-nG", &user])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    // Keep only groups that EXIST on this system AND that the user lacks.
    let mut to_add: Vec<&str> = Vec::new();
    for g in groups {
        let exists = Command::new("getent")
            .args(["group", g])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !exists {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {component_name} — group '{g}' not present on this system; skipping"
                ),
                false,
            ));
            continue;
        }
        if !current_groups.iter().any(|c| c == g) {
            to_add.push(*g);
        }
    }

    if to_add.is_empty() {
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {component_name} — user '{user}' already in all required group(s); nothing to add"
            ),
            false,
        ));
        return;
    }

    let csv = to_add.join(",");

    // Add via `sudo usermod -aG`. Needs root → set up askpass so it's non-interactive.
    let askpass = sudo_pw.filter(|p| !p.is_empty()).and_then(|p| {
        match crate::installers::common::askpass::Askpass::new(p) {
            Ok(g) => Some(g),
            Err(e) => {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[native] {component_name} — askpass setup failed ({e}); cannot add \
                         groups non-interactively"
                    ),
                    true,
                ));
                None
            }
        }
    });

    let added = if let Some(ap) = askpass {
        // `sudo -A usermod -aG …` — pass sudo_pw = None so execute_native_command
        // does not double-wrap in `sudo -S`; SUDO_ASKPASS supplies the password.
        let args = vec![
            "-A".to_string(),
            "usermod".to_string(),
            "-aG".to_string(),
            csv.clone(),
            user.clone(),
        ];
        let envs = vec![("SUDO_ASKPASS".to_string(), ap.path().to_string())];
        let cmd = NativeCommand::from_shell_cmd("sudo", &args, &envs);
        execute_native_command(&cmd, None, sender, component_name).is_ok()
    } else {
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {component_name} — no sudo password available; please run: \
                 sudo usermod -aG {csv} {user}"
            ),
            true,
        ));
        return;
    };

    if added {
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[native] {component_name} — added user '{user}' to group(s): {csv}. \
                 A re-login (or reboot) is required for the new groups to take effect."
            ),
            false,
        ));
    }
}

/// Probe and repair RCCL without installing external dependencies.
fn run_rccl_remediation(ctx: &NativeInstallerContext, component_name: &str) -> Result<()> {
    let sender = ctx.sender;
    let rccl_remediation = crate::installers::components::rccl::RcclInstaller::new(
        ctx.user_home,
        resolve_python_bin(),
        ctx.env_exports
            .get("ROCM_PATH")
            .cloned()
            .unwrap_or_else(|| "/opt/rocm".to_string()),
        ctx.env_exports
            .get("ROCM_VERSION")
            .cloned()
            .unwrap_or_else(|| "7.2.0".to_string()),
        ctx.env_exports
            .get("ROCM_CHANNEL")
            .cloned()
            .unwrap_or_else(|| "latest".to_string()),
        ctx.env_exports
            .get("ROCR_VISIBLE_DEVICES")
            .cloned()
            .or_else(|| ctx.env_exports.get("HIP_VISIBLE_DEVICES").cloned()),
    );

    let _ = sender.send(InstallerEvent::Log(
        "[native] Probing RCCL multi-GPU collectives...".into(),
        false,
    ));
    if let Some(build) = rccl_remediation.prepare_remediation()? {
        let _ = sender.send(InstallerEvent::Log(
            "[native] Known RCCL collective failure detected; building sealed overlay...".into(),
            false,
        ));
        execute_native_command(
            &NativeCommand::from_shell_cmd(&build.program, &build.args, &build.env),
            None,
            sender,
            component_name,
        )?;
        rccl_remediation.activate_and_verify()?;
        let _ = sender.send(InstallerEvent::Log(
            "[native] RCCL sealed overlay activated and validated on both GPUs".into(),
            false,
        ));
    } else {
        let _ = sender.send(InstallerEvent::Log(
            "[native] RCCL probe healthy or existing sealed overlay revalidated; no repair needed"
                .into(),
            false,
        ));
    }
    Ok(())
}

/// Execute the native Rust installer dispatch for a ported component.
///
/// - **VAL-INSTALL-032**: installer.rs dispatches to correct Rust module per ID
/// - **VAL-INSTALL-033**: sudo behavior preserved for needs_sudo:true components
/// - **VAL-INSTALL-034**: Environment variable injection preserved
/// - **VAL-INSTALL-035**: InstallerEvent progress reporting preserved
/// - **VAL-INSTALL-036**: Error message format matches original scripts
fn run_native_installer(component: &Component, ctx: &NativeInstallerContext) -> Result<()> {
    let sender = ctx.sender;
    let _ = sender.send(InstallerEvent::Log(
        format!("[native] Installing {} via Rust module...", component.name),
        false,
    ));

    let _ = sender.send(InstallerEvent::Progress {
        component_id: component.id.clone(),
        progress: 0.1,
        message: format!("Starting {} installation", component.name),
    });

    // VAL-INSTALL-033: Preserve sudo behavior for components that need it
    let component_needs_sudo = component.needs_sudo && needs_sudo();
    let sudo_pw: Option<&str> = if component_needs_sudo {
        ctx.sudo_password.as_deref()
    } else {
        None
    };

    // VAL-INSTALL-036: Error message format matches original scripts
    if component_needs_sudo && ctx.sudo_password.is_none() {
        bail!("{} requires sudo but no password provided", component.name);
    }

    // Build common env vars for all commands (VAL-INSTALL-034)
    let common_env = build_common_env(ctx);

    // VAL-INSTALL-032: Per-component match dispatch covering all 24 components.
    // Each arm instantiates the correct installer, calls build_*_command()
    // methods in the correct sequence, and executes via execute_native_command().
    match component.id.as_str() {
        // ── permanent-env ──────────────────────────────────────────────
        "permanent-env" => {
            use crate::installers::components::permanent_env::PermanentEnvInstaller;
            // REGENERATE ~/.mlstack_env from current system state. This is the
            // component's namesake job: running it — even after initial creation,
            // even with force-reinstall — must rewrite the env file so values like
            // ROCM_VERSION track what is actually installed. (run_installation also
            // refreshes at start + post-loop, but the component owns this explicitly
            // so the option does what its name says.) Best-effort: log on error,
            // never fail the component over an env refresh.
            match ensure_mlstack_env(ctx.user_home, ctx.install_method) {
                Ok(EnvUpdate::Created) | Ok(EnvUpdate::Updated) => {
                    let _ = sender.send(InstallerEvent::Log(
                        "[native] Permanent ROCm Env — regenerated ~/.mlstack_env from current state".into(),
                        false,
                    ));
                }
                Ok(EnvUpdate::Unchanged) => {
                    let _ = sender.send(InstallerEvent::Log(
                        "[native] Permanent ROCm Env — ~/.mlstack_env already up to date".into(),
                        false,
                    ));
                }
                Err(e) => {
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] Permanent ROCm Env — could not regenerate ~/.mlstack_env: {e}"
                        ),
                        true,
                    ));
                }
            }
            // Build triton directories, write activate snippets, offer global prepend
            crate::platform::environment::write_activate_snippets()?;
            crate::platform::environment::offer_global_prepend(true)?;
            let inst = PermanentEnvInstaller::default();
            let cmd = inst.build_mkdir_triton_dirs_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;
        }

        // ── rocm ──────────────────────────────────────────────────────
        "rocm" => {
            use crate::installers::components::rocm::{RocmConfig, RocmInstaller};
            let rocm_force = is_force_reinstall();
            let inst = RocmInstaller::new(RocmConfig {
                force_reinstall: rocm_force,
                ..RocmConfig::default()
            });
            let distro = crate::installers::common::DistroFacade::detect();
            let is_arch = !(distro.uses_apt()
                || distro.uses_dnf()
                || distro.uses_yum()
                || distro.uses_zypper());
            if is_arch {
                // Arch/pacman path. The AUR helper (`yay`) MUST run as the
                // user — makepkg refuses root, and `yay` warns "Avoid running
                // yay as root/sudo". This branch therefore does NOT reuse the
                // sudo-wrapped package loop below; it runs a single user-space
                // `yay` and feeds its internal `sudo pacman` via SUDO_ASKPASS.
                let all_pkgs = inst.pacman_rocm_packages();
                let force_reinstall = is_force_reinstall();
                let pkgs_to_install = if force_reinstall {
                    all_pkgs.clone()
                } else {
                    let need_install = RocmInstaller::filter_already_installed_pacman(&all_pkgs);
                    if need_install.is_empty() {
                        let _ = sender.send(InstallerEvent::Log(
                            format!(
                                "[native] {} — all ROCm packages already installed, skipping",
                                component.name
                            ),
                            false,
                        ));
                        return Ok(());
                    }
                    need_install
                };

                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[native] {} — installing {} of {} ROCm packages via yay \
                         (user-space, non-interactive): {}",
                        component.name,
                        pkgs_to_install.len(),
                        all_pkgs.len(),
                        pkgs_to_install.join(", ")
                    ),
                    false,
                ));

                // Build a fully non-interactive `yay` command. `--noconfirm`
                // skips pacman + yay menus. `--sudoflags=-A` combined with
                // SUDO_ASKPASS hands the password to yay's internal sudo with
                // no TTY. Force-reinstall omits `--needed` so yay reinstalls in
                // place — the previous explicit `pacman -Rns` pre-removal was
                // removed: it aborted on dependency conflicts
                // (hip-runtime-amd/hipblaslt/migraphx/miopen-hip depend on the
                // ROCm libs) and was redundant, since `yay -S` without
                // `--needed` already reinstalls.
                let mut args = vec!["-S".to_string()];
                if !force_reinstall {
                    args.push("--needed".to_string());
                }
                args.push("--noconfirm".to_string());

                // Askpass guard — keeps the password file alive for the op.
                let askpass = sudo_pw.filter(|p| !p.is_empty()).and_then(|p| {
                    match crate::installers::common::askpass::Askpass::new(p) {
                        Ok(g) => Some(g),
                        Err(e) => {
                            let _ = sender.send(InstallerEvent::Log(
                                format!(
                                    "[native] {} — askpass setup failed ({e}); yay's \
                                     privileged steps may not authenticate",
                                    component.name
                                ),
                                true,
                            ));
                            None
                        }
                    }
                });
                let mut envs: Vec<(String, String)> = Vec::new();
                if let Some(ref ap) = askpass {
                    envs.push(("SUDO_ASKPASS".to_string(), ap.path().to_string()));
                    args.push("--sudoflags=-A".to_string());
                } else {
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] {} — no sudo password supplied; yay will only \
                             authenticate if sudo is cached / passwordless",
                            component.name
                        ),
                        false,
                    ));
                }
                args.extend(pkgs_to_install);

                let yay_cmd = NativeCommand::Shell {
                    program: "yay".to_string(),
                    args,
                    env: envs,
                    working_dir: None,
                };
                // Pass sudo_pw = None: `execute_native_command` would otherwise
                // wrap the program in `sudo -S` (running yay as root → refused).
                execute_native_command(&yay_cmd, None, sender, &component.name)?;
            } else {
                let commands = if distro.uses_apt() {
                    inst.apt_install_commands()
                } else if distro.uses_dnf() || distro.uses_yum() {
                    let rhel_ver = distro.version().split('.').next().unwrap_or("9");
                    inst.dnf_install_commands(rhel_ver)
                } else {
                    inst.zypper_install_commands()
                };
                for pkg_cmd in &commands {
                    let native_cmd = NativeCommand::Package {
                        program: pkg_cmd.program.clone(),
                        args: pkg_cmd.args.clone(),
                    };
                    // Package commands may already include "sudo" as the program;
                    // don't double-wrap with sudo.
                    execute_native_command(&native_cmd, sudo_pw, sender, &component.name)?;
                }
            }

            // After successful ROCm install, create reboot marker when force-reinstalling
            if is_force_reinstall() {
                let marker_path = std::path::Path::new("/tmp/mlstack-reboot-required");
                let _ = std::fs::write(
                    marker_path,
                    "ROCm packages were reinstalled. Reboot required for kernel module changes.\n",
                );
                let _ = sender.send(InstallerEvent::Log(
                    "[native] ROCm — reboot recommended for kernel module changes".to_string(),
                    false,
                ));
            }

            // Ensure the user can actually access GPU compute devices: add to
            // render/video if missing. The original install_rocm.sh did
            // `sudo usermod -aG render,video "$target_user"`; the Rust port had
            // dropped it. Idempotent + system-aware (skips absent groups).
            ensure_user_in_groups(&["render", "video"], sudo_pw, sender, &component.name);
        }

        // ── pytorch ───────────────────────────────────────────────────
        "pytorch" => {
            use crate::installers::components::pytorch::{PyTorchConfig, PyTorchInstaller};
            let force = is_force_reinstall();
            let python_bin = resolve_python_bin();
            let config = PyTorchConfig {
                python_bin: python_bin.clone(),
                force_reinstall: force,
                ..PyTorchConfig::default()
            };
            let inst = PyTorchInstaller::new(config);
            let use_uv = crate::installers::common::utils::command_exists("uv");
            let rocm_mm = ctx
                .env_exports
                .get("ROCM_VERSION")
                .map(|v| {
                    let parts: Vec<&str> = v.split('.').collect();
                    if parts.len() >= 2 {
                        format!("{}.{}", parts[0], parts[1])
                    } else {
                        v.clone()
                    }
                })
                .unwrap_or_else(|| "7.2".to_string());

            // Force reinstall: thoroughly purge previous PyTorch (non-fatal)
            if force {
                purge_pytorch(sender, &component.name);
            }

            // Step 1: Install PyTorch FIRST, from the ROCm wheel index. This MUST
            // precede the common-deps step: `torchsde` depends on `torch>=1.6.0`,
            // and if torch isn't already installed, pip resolves that dep from the
            // DEFAULT PyPI index — which is the CUDA build, dragging in the entire
            // nvidia-*-cu13 / cuda-toolkit stack. Installing the ROCm torch first
            // satisfies torchsde's dep so the deps step never pulls CUDA torch.
            // (The original install_pytorch_rocm.sh installed torchsde/sentencepiece
            // AFTER PyTorch — the port had inverted the order.)
            let install_cmd = inst.build_install_command(&rocm_mm, use_uv);
            execute_native_command(
                &NativeCommand::Pip {
                    program: install_cmd.program.clone(),
                    args: install_cmd.args.clone(),
                },
                None,
                sender,
                &component.name,
            )?;

            // Step 2: Install common deps (torchsde, sentencepiece) from PyPI. With
            // the ROCm torch already installed, torchsde's torch dep is satisfied —
            // pip (default --upgrade-strategy=only-if-needed) won't replace it with
            // the CUDA build.
            let deps_cmd = inst.build_common_deps_command(use_uv);
            let _ = common_env.clone(); // Available for future env injection
            execute_native_command(
                &NativeCommand::Pip {
                    program: deps_cmd.program.clone(),
                    args: deps_cmd.args.clone(),
                },
                None, // pip commands don't need sudo
                sender,
                &component.name,
            )?;

            // Step 3: Rusty centrally owns FastVideo's build/import prerequisites.
            // Install exact pins without dependency resolution, then verify them
            // read-only. Component installers may consume these packages but may
            // never mutate or resolve the managed environment themselves.
            let managed_python = resolve_python_bin();
            let managed_install =
                crate::installers::common::managed_python_install_command(managed_python.clone());
            execute_native_command(
                &NativeCommand::Pip {
                    program: managed_install.program,
                    args: managed_install.args,
                },
                None,
                sender,
                &component.name,
            )?;

            let managed_verify =
                crate::installers::common::managed_python_verify_command(managed_python);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&managed_verify.program, &managed_verify.args, &[]),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── triton ────────────────────────────────────────────────────
        "triton" => {
            use crate::installers::components::triton::{TritonConfig, TritonInstaller};

            // Skip the from-source build if `triton` is already importable.
            // PyTorch's ROCm wheels bundle `triton-rocm` (the prebuilt ROCm
            // triton), so once PyTorch is installed `import triton` already works.
            // The ROCm/triton from-source build is heavy (pulls the NVIDIA
            // toolchain, builds LLVM) and brittle (permission hazard on
            // ~/.triton/llvm when prior steps ran under sudo), so it is both
            // redundant and risky when triton-rocm already provides it.
            let triton_present = std::process::Command::new(resolve_python_bin())
                .arg("-c")
                .arg("import triton")
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if triton_present {
                let _ = sender.send(InstallerEvent::Log(
                    "[native] Triton — already importable (triton-rocm bundled with PyTorch); \
                     skipping the redundant from-source ROCm/triton build"
                        .into(),
                    false,
                ));
                return Ok(());
            }

            let triton_force = is_force_reinstall();
            let inst = TritonInstaller::new(TritonConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });
            let target_dir = format!("{}/.mlstack/triton", ctx.user_home);

            // Force reinstall: thoroughly purge previous Triton (non-fatal)
            if triton_force {
                purge_triton(sender, &component.name);
            }

            // Step 1: git clone (idempotent — pulls if already exists)
            // NOTE: git operations may use sudo for the clone/pull, but we fix
            // ownership afterward so the build (which runs as the user) can succeed.
            git_clone_or_pull(
                inst.git_clone_url(),
                &target_dir,
                &[],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Fix ownership: if the triton dir was created by sudo, the user can't build in it.
            // Chown recursively to the real user.
            fix_directory_ownership(&target_dir, sudo_pw, sender);

            // Add safe.directory exception for dubious ownership (git runs under different uid)
            let safe_dir_cmd = NativeCommand::from_shell_cmd(
                "git",
                &[
                    "config".to_string(),
                    "--global".to_string(),
                    "--add".to_string(),
                    "safe.directory".to_string(),
                    target_dir.clone(),
                ],
                &[],
            );
            let _ = execute_native_command(&safe_dir_cmd, None, sender, &component.name);

            // Step 2: git fetch the target branch
            let branch = inst.select_branch().to_string();
            let fetch_branch_cmd = NativeCommand::from_shell_cmd(
                "git",
                &[
                    "-C".to_string(),
                    target_dir.clone(),
                    "fetch".to_string(),
                    "origin".to_string(),
                    branch.clone(),
                ],
                &[],
            );
            let _ = execute_native_command(&fetch_branch_cmd, None, sender, &component.name);

            // Step 3: git checkout
            let checkout_cmd = inst.build_git_checkout_command();
            let mut checkout_args = vec!["-C".to_string(), target_dir.clone()];
            checkout_args.extend(checkout_cmd.args.iter().cloned());
            let checkout_result = execute_native_command(
                &NativeCommand::from_shell_cmd(
                    &checkout_cmd.program,
                    &checkout_args,
                    &checkout_cmd.env,
                ),
                None,
                sender,
                &component.name,
            );
            if let Err(err) = checkout_result {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[native] [WARN] {} git checkout failed (non-fatal): {}",
                        component.name, err
                    ),
                    false,
                ));
            }

            // Step 4: prerequisites — run WITHOUT sudo so pip targets the user's Python
            let cmd = inst.build_prerequisites_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 5: pip install — run WITHOUT sudo so it installs into user's Python
            let cmd = inst.build_pip_install_command(&target_dir);
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── mpi4py ────────────────────────────────────────────────────
        "mpi4py" => {
            use crate::installers::components::mpi4py::{Mpi4PyConfig, Mpi4PyInstaller};
            let force = is_force_reinstall();
            let inst = Mpi4PyInstaller::new(Mpi4PyConfig {
                python_bin: resolve_python_bin(),
                force_reinstall: force,
                ..Mpi4PyConfig::default()
            });
            let distro = crate::installers::common::DistroFacade::detect();

            // Force reinstall: purge previous mpi4py (non-fatal)
            if force {
                purge_component_packages(&["mpi4py"], sender, &component.name);
            }

            // Step 1: Install system MPI
            let cmd = inst.build_system_mpi_install_command(&distro);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: pip install mpi4py
            let mpi_path = std::path::PathBuf::from("/usr");
            let cmd = inst.build_pip_install_command(&mpi_path);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── deepspeed ─────────────────────────────────────────────────
        "deepspeed" => {
            use crate::installers::components::deepspeed::{DeepSpeedConfig, DeepSpeedInstaller};
            let force = is_force_reinstall();
            let config = DeepSpeedConfig {
                python_bin: resolve_python_bin(),
                force_reinstall: force,
                ..DeepSpeedConfig::default()
            };
            let inst = DeepSpeedInstaller::new(config);

            // Force reinstall: purge previous deepspeed (non-fatal)
            if force {
                purge_component_packages(&["deepspeed"], sender, &component.name);
            }

            // Step 1: Install dependencies
            let deps_cmd = inst.build_deps_install_command();
            execute_native_command(
                &NativeCommand::Pip {
                    program: deps_cmd.program.clone(),
                    args: deps_cmd.args.clone(),
                },
                None,
                sender,
                &component.name,
            )?;

            // Step 2: Install DeepSpeed
            let install_cmd = inst.build_install_command();
            execute_native_command(
                &NativeCommand::Pip {
                    program: install_cmd.program.clone(),
                    args: install_cmd.args.clone(),
                },
                None,
                sender,
                &component.name,
            )?;
        }

        // ── ml-stack-core ─────────────────────────────────────────────
        "ml-stack-core" => {
            use crate::installers::components::ml_stack::{MlStackConfig, MlStackInstaller};
            let inst = MlStackInstaller::new(MlStackConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });
            let distro = crate::installers::common::DistroFacade::detect();

            // Step 1: Install system dependencies
            let migraphx_cmd = inst.build_migraphx_install_command(&distro);
            execute_native_command(
                &NativeCommand::System {
                    program: migraphx_cmd.program.clone(),
                    args: migraphx_cmd.args.clone(),
                },
                sudo_pw,
                sender,
                &component.name,
            )?;

            let rccl_cmd = inst.build_rccl_install_command(&distro);
            execute_native_command(
                &NativeCommand::System {
                    program: rccl_cmd.program.clone(),
                    args: rccl_cmd.args.clone(),
                },
                sudo_pw,
                sender,
                &component.name,
            )?;

            // The distro package stays authoritative. Only the exact ROCm#6074
            // two-GPU failure triggers a sealed, stack-owned RCCL overlay.
            run_rccl_remediation(ctx, &component.name)?;

            let mpi_cmd = inst.build_mpi_install_command(&distro);
            execute_native_command(
                &NativeCommand::System {
                    program: mpi_cmd.program.clone(),
                    args: mpi_cmd.args.clone(),
                },
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: Clone Megatron into a fixed dir, then `pip install -e .` FROM that
            // dir. Previously the clone had no destination (landed in the inherited CWD)
            // and the pip step had no working_dir, so `pip install -e .` would target
            // whatever dir rusty-stack was launched from — not Megatron-LM.
            let megatron_dir = format!("{}/.mlstack/src/Megatron-LM", ctx.user_home);
            let megatron_clone = inst.build_megatron_clone_command(&megatron_dir);
            execute_native_command(
                &NativeCommand::System {
                    program: megatron_clone.program.clone(),
                    args: megatron_clone.args.clone(),
                },
                sudo_pw,
                sender,
                &component.name,
            )?;

            let megatron_install = inst.build_megatron_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &megatron_install.program,
                    &megatron_install.args,
                    &[],
                    Some(PathBuf::from(&megatron_dir)),
                ),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── flash-attn-triton / flash-attn-ck ─────────────────────────
        // Both backends build from the SAME official ROCm/flash-attention repo.
        // CK is the repo's default backend; Triton is enabled via
        // FLASH_ATTENTION_TRITON_AMD_ENABLE. They install the identical
        // `flash_attn` package → mutually exclusive; only the last-installed is
        // active (tracked by ~/.mlstack/flash-attention/.backend).
        //   - triton: full forward + backward (recommended for RDNA3)
        //   - ck:     forward-only on RDNA3 (backward unimplemented —
        //             ROCm/composable_kernel#1434)
        "flash-attn" | "flash-attn-triton" | "flash-attn-ck" => {
            let force = is_force_reinstall();
            // The generic "flash-attn" id (e.g. from migration/update) defaults to
            // the recommended Triton backend.
            let is_triton = matches!(component.id.as_str(), "flash-attn" | "flash-attn-triton");
            let backend_name = if is_triton { "triton" } else { "ck" };

            // Force reinstall: thoroughly purge previous Flash Attention (non-fatal)
            if force {
                purge_flash_attn(sender, &component.name);
            }

            let target_dir = format!("{}/.mlstack/flash-attention", ctx.user_home);

            git_clone_or_pull(
                "https://github.com/ROCm/flash-attention.git",
                &target_dir,
                &[],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Fix ownership if cloned with sudo
            fix_directory_ownership(&target_dir, sudo_pw, sender);

            // Add safe.directory for git operations
            let safe_dir_cmd = NativeCommand::from_shell_cmd(
                "git",
                &[
                    "config".to_string(),
                    "--global".to_string(),
                    "--add".to_string(),
                    "safe.directory".to_string(),
                    target_dir.clone(),
                ],
                &[],
            );
            let _ = execute_native_command(&safe_dir_cmd, None, sender, &component.name);

            // Resolve GPU arch and HSA version from env
            let gpu_arch = ctx
                .env_exports
                .get("GPU_ARCH")
                .cloned()
                .unwrap_or_else(|| "gfx1100".to_string());
            let hsa_version = ctx
                .env_exports
                .get("HSA_OVERRIDE_GFX_VERSION")
                .cloned()
                .unwrap_or_else(|| "11.0.0".to_string());

            // Flash Attention's real build/runtime deps — einops, ninja, packaging,
            // psutil — are NOT guaranteed by the dep graph (get_dependencies =
            // [pytorch, rocm]; no deepspeed), so install them explicitly first.
            // None are CUDA runtime packages, and the universal pip chokepoint still
            // blocks nvidia-*/cuda*. torch itself comes from the pytorch dependency.
            let fa_deps_args = vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--break-system-packages".to_string(),
                "einops".to_string(),
                "ninja".to_string(),
                "packaging".to_string(),
                "psutil".to_string(),
            ];
            execute_native_command(
                &NativeCommand::Shell {
                    program: resolve_python_bin(),
                    args: fa_deps_args,
                    env: vec![],
                    working_dir: None,
                },
                None, // pip runs as user, not sudo
                sender,
                &component.name,
            )?;

            // --no-deps on the FA package itself: with einops/ninja/packaging/psutil
            // now installed explicitly + torch from pytorch, --no-deps scopes this to
            // flash_attn and can NEVER pull a CUDA torch to satisfy a torch requirement.
            // -v surfaces ninja/compiler output (e.g. "[N/M] Compiling …") so the
            // build log is detailed AND parse_progress can drive the bar with real
            // per-TU progress instead of pip's opaque "still running…" heartbeat.
            let mut pip_args = vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "-v".to_string(),
                "--no-build-isolation".to_string(),
                "--no-deps".to_string(),
                "--break-system-packages".to_string(),
                ".".to_string(),
            ];
            if force {
                pip_args.push("--force-reinstall".to_string());
            }

            let mut pip_env = vec![
                ("GPU_ARCH".to_string(), gpu_arch.clone()),
                ("PYTORCH_ROCM_ARCH".to_string(), gpu_arch),
                ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
                ("HSA_OVERRIDE_GFX_VERSION".to_string(), hsa_version),
            ];
            if is_triton {
                // Triton backend opt-in (full forward + backward on RDNA3/4).
                pip_env.push((
                    "FLASH_ATTENTION_TRITON_AMD_ENABLE".to_string(),
                    "TRUE".to_string(),
                ));
            } else {
                // CK backend (repo default — Triton flag omitted). The
                // composable-kernel compile is RAM-heavy; cap parallel jobs so
                // builds don't OOM on boxes with <96 GB RAM.
                let jobs = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .min(8);
                pip_env.push(("MAX_JOBS".to_string(), jobs.to_string()));
            }

            execute_native_command(
                &NativeCommand::Shell {
                    // Managed interpreter (~/.mlstack/global) — same value the
                    // config was built with. NEVER bare system python3.
                    program: resolve_python_bin(),
                    args: pip_args,
                    env: pip_env,
                    working_dir: Some(PathBuf::from(&target_dir)),
                },
                None, // pip must run as user, not sudo
                sender,
                &component.name,
            )?;

            // Record the active backend so detection is HONEST: "installed" requires
            // `flash_attn` importable AND a known-backend marker. Only the
            // last-installed backend reports installed (mutual exclusivity). The
            // marker is part of the detection contract, so a write failure MUST fail
            // the install — otherwise the package is installed but untrackable.
            let marker = PathBuf::from(&target_dir).join(".backend");
            std::fs::write(&marker, backend_name).map_err(|err| {
                anyhow::anyhow!(
                    "{}: failed to write Flash Attention backend marker {}: {} \
                     — package installed but untrackable; failing install",
                    component.name,
                    marker.display(),
                    err
                )
            })?;
        }

        // ── repair-stack ──────────────────────────────────────────────
        // VAL-INSTALL-051: Repair dispatches to native Rust installer
        // modules for each step, not bash scripts.
        "repair-stack" => {
            use crate::installers::components::repair::RepairInstaller;
            let repair_ids = RepairInstaller::native_component_ids();

            let _ = sender.send(InstallerEvent::Log(
                "[native] Starting ML Stack repair (8 steps via native installers)...".to_string(),
                false,
            ));

            for step_id in &repair_ids {
                // Build a synthetic Component for the sub-installer
                // and dispatch through run_native_installer recursively.
                let sub_component = Component {
                    id: (*step_id).to_string(),
                    name: (*step_id).to_string(),
                    description: String::new(),
                    script: String::new(),
                    category: Category::Core,
                    required: true,
                    selected: true,
                    installed: false,
                    progress: 0.0,
                    estimate: String::new(),
                    needs_sudo: component.needs_sudo,
                    experimental: false,
                    note: None,
                };

                let _ = sender.send(InstallerEvent::Progress {
                    component_id: component.id.clone(),
                    progress: 0.5,
                    message: format!("Repairing {}...", step_id),
                });

                if let Err(err) = run_native_installer(&sub_component, ctx) {
                    // Log but continue — repair tries all steps
                    let _ = sender.send(InstallerEvent::Log(
                        format!("[native] Repair step '{}' failed: {}", step_id, err),
                        false,
                    ));
                }
            }
        }

        "rccl-repair" => {
            run_rccl_remediation(ctx, &component.name)?;
        }

        // ── megatron ──────────────────────────────────────────────────
        "megatron" => {
            use crate::installers::components::megatron::{MegatronConfig, MegatronInstaller};
            let force = is_force_reinstall();
            let inst = MegatronInstaller::new(MegatronConfig {
                python_bin: resolve_python_bin(),
                force_reinstall: force,
                ..MegatronConfig::default()
            });

            // Force reinstall: purge previous megatron-core (non-fatal)
            if force {
                purge_component_packages(&["megatron-core", "megatron"], sender, &component.name);
            }

            // Step 1: Install dependencies
            for dep_pkg in &["ninja", "packaging", "wheel"] {
                let cmd = inst.build_dep_install_command(dep_pkg);
                execute_native_command(
                    &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                    None,
                    sender,
                    &component.name,
                )?;
            }

            // Step 2: git clone (idempotent — pulls if already exists)
            let clone_cmd = inst.build_git_clone_command();
            let clone_target_str = clone_cmd.args.last().cloned().unwrap_or_default();
            git_clone_or_pull(
                "https://github.com/NVIDIA/Megatron-LM.git",
                &clone_target_str,
                &[],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2b: Fix ownership — git_clone_or_pull runs with sudo, leaving
            // the tree root-owned. pip install -e . runs as the user and must
            // create megatron_core.egg-info in the clone dir; without chown it
            // fails "could not create 'megatron_core.egg-info': Permission denied".
            // (Mirrors the aiter installer's post-clone chown.)
            let run_user = std::env::var("SUDO_USER")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default();
            if !run_user.is_empty() {
                let chown_cmd = NativeCommand::from_shell_cmd(
                    "chown",
                    &[
                        "-R".to_string(),
                        format!("{run_user}:{run_user}"),
                        clone_target_str.clone(),
                    ],
                    &[],
                );
                let _ = execute_native_command(&chown_cmd, sudo_pw, sender, &component.name);
            }

            // Step 3: Remove stale egg-info directories that may be root-owned
            // from a prior sudo install. Without this, `pip install -e .` fails with
            // "Cannot update time stamp of directory 'megatron_core.egg-info'".
            let clone_target_path = PathBuf::from(&clone_target_str);
            if let Ok(entries) = std::fs::read_dir(&clone_target_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".egg-info") {
                        let egg_path = entry.path();
                        // Try plain remove first
                        if std::fs::remove_dir_all(&egg_path).is_err() {
                            // Root-owned: try sudo rm -rf
                            let rm_cmd = NativeCommand::from_shell_cmd(
                                "rm",
                                &["-rf".to_string(), egg_path.to_string_lossy().to_string()],
                                &[],
                            );
                            let _ =
                                execute_native_command(&rm_cmd, sudo_pw, sender, &component.name);
                        }
                    }
                }
            }

            // Step 4: pip install (uses working_dir to run in the cloned Megatron-LM dir)
            let cmd = inst.build_pip_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            )?;

            // Step 5: Install filtered Megatron requirements without CUDA/NVIDIA deps.
            let requirements_candidates = [
                clone_target_path
                    .join("requirements")
                    .join("requirements.txt"),
                clone_target_path.join("requirements.txt"),
                clone_target_path
                    .join("requirements")
                    .join("requirements_amd.txt"),
            ];
            for req_path in requirements_candidates.iter().filter(|p| p.exists()) {
                let filtered = match filter_cuda_requirements(&req_path.to_string_lossy()) {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = sender.send(InstallerEvent::Log(
                            format!(
                                "[native] WARN: could not read Megatron requirements {}: {e} — skipping requirements install",
                                req_path.display()
                            ),
                            true,
                        ));
                        anyhow::bail!(
                            "failed to read Megatron requirements {}: {e}",
                            req_path.display()
                        );
                    }
                };
                if filtered.is_empty() {
                    continue;
                }
                let tmp_dir = std::env::temp_dir().join("rusty-stack-megatron");
                let _ = std::fs::create_dir_all(&tmp_dir);
                let filtered_path = tmp_dir.join("requirements_filtered.txt");
                std::fs::write(&filtered_path, &filtered).with_context(|| {
                    format!(
                        "Failed to write filtered Megatron requirements to {:?}",
                        filtered_path
                    )
                })?;
                let filtered_path_str = filtered_path.to_string_lossy().to_string();
                let filtered_cmd = inst.build_requirements_install_command(&filtered_path_str);
                execute_native_command(
                    &NativeCommand::from_shell_cmd(
                        &filtered_cmd.program,
                        &filtered_cmd.args,
                        &filtered_cmd.env,
                    ),
                    None,
                    sender,
                    &component.name,
                )?;
                let _ = std::fs::remove_file(&filtered_path);
                break;
            }
        }

        // ── vllm ──────────────────────────────────────────────────────
        "vllm" => {
            use crate::installers::components::vllm_multi::{VllmConfig, VllmInstaller};
            let vllm_force = is_force_reinstall();
            let inst = VllmInstaller::new(VllmConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Force reinstall: thoroughly purge previous vLLM (non-fatal)
            if vllm_force {
                purge_vllm(sender, &component.name);
            }

            // Step 0: build tools (cmake/ninja/wheel/setuptools/patchelf) for the
            // source compile of vLLM's C/HIP extensions.
            let cmd = inst.build_tools_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Resolve the vLLM version per the user's supply-chain gate (default:
            // skip the newest release = lag 1). Queries PyPI for releases + dates,
            // picks the newest satisfying the version-lag + age policy. Falls back
            // to a pinned constant if PyPI is unreachable.
            let vllm_version = inst
                .resolve_version_with_policy(ctx.vllm_version_lag, ctx.vllm_version_min_age_days);
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] vLLM: selected {vllm_version} (gate: lag={}, min_age_days={}) — \
                     building from source (ROCm, env torch)",
                    ctx.vllm_version_lag, ctx.vllm_version_min_age_days
                ),
                false,
            ));
            // Build vLLM FROM SOURCE against the env's torch (--no-deps so pip
            // never swaps torch; --no-build-isolation compiles against the
            // installed torch). Pinned + VLLM_VERSION_OVERRIDE so metadata matches
            // the sdist filename (no +rocmNNN cascade). Heavy compile.
            let cmd = inst.build_source_install_command(&vllm_version);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 2: versioned runtime deps (mistral-common, openai-harmony,
            // triton-kernels, outlines-core, xgrammar, llguidance) — none pin
            // torch, so the env torch stays intact.
            let cmd = inst.build_versioned_deps_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 3: core runtime deps (accelerate, fastapi, transformers, …).
            let cmd = inst.build_deps_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 4: fail loud if the explicit --no-deps closure is incomplete.
            // This is read-only: it prevents a false "vLLM completed" without
            // letting pip mutate torch/ROCm/NVIDIA/CUDA dependencies.
            let cmd = inst.build_integrity_check_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── aiter ─────────────────────────────────────────────────────
        "aiter" => {
            use crate::installers::components::aiter::{AiterConfig, AiterInstaller};
            // Source the build/HSA arch from the detector (env file) — NEVER default to
            // gfx1100, which would mis-compile the wheel on RDNA2/RDNA4 hardware.
            let detected_gpu_arch = ctx
                .env_exports
                .get("GPU_ARCH")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| detect_gpu_arch().unwrap_or_else(|| "gfx1100".to_string()));
            let inst = AiterInstaller::new(AiterConfig {
                python_bin: resolve_python_bin(),
                gpu_arch: detected_gpu_arch,
                ..Default::default()
            });
            let target_dir = format!("{}/.mlstack/aiter", ctx.user_home);

            // Force reinstall: thoroughly purge previous AITER (non-fatal)
            if is_force_reinstall() {
                purge_aiter(sender, &component.name);
            }

            // Step 1: git clone (idempotent — pulls if already exists)
            git_clone_or_pull(
                inst.git_clone_url(),
                &target_dir,
                &["--recursive"],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 1b: Fix ownership of cloned directory.
            // git_clone_or_pull may run with sudo, leaving files root-owned.
            // pip install runs as the user and needs write access for metadata.
            let run_user = std::env::var("SUDO_USER")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default();
            if !run_user.is_empty() {
                let chown_cmd = NativeCommand::from_shell_cmd(
                    "chown",
                    &[
                        "-R".to_string(),
                        format!("{run_user}:{run_user}"),
                        target_dir.clone(),
                    ],
                    &[],
                );
                let _ = execute_native_command(&chown_cmd, sudo_pw, sender, &component.name);
            }

            // Step 2: Install dependencies
            let cmd = inst.build_deps_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 3: pip install
            let cmd = inst.build_pip_install_command(&target_dir);
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            )?;

            // Step 4: Fix ownership of aiter directories unconditionally.
            // AITER's JIT compiler writes to ~/.aiter/jit which fails if root-owned
            // from a prior sudo install or if pip's subprocess created it as root.
            let run_user = std::env::var("SUDO_USER")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default();
            if !run_user.is_empty() {
                let user_home = &ctx.user_home;
                for aiter_dir in [
                    format!("{user_home}/.aiter"),
                    format!("{user_home}/.mlstack/aiter"),
                ] {
                    let dir_path = PathBuf::from(&aiter_dir);
                    if dir_path.exists() {
                        // Check if we can't write to it — then fix permissions
                        let needs_fix = std::fs::read_dir(&dir_path)
                            .map(|mut entries| entries.next().is_some())
                            .unwrap_or(false)
                            && std::fs::write(dir_path.join(".permission_check"), b"").is_err();
                        if needs_fix {
                            let chown_cmd = NativeCommand::from_shell_cmd(
                                "chown",
                                &[
                                    "-R".to_string(),
                                    format!("{run_user}:{run_user}"),
                                    aiter_dir.clone(),
                                ],
                                &[],
                            );
                            let _ = execute_native_command(
                                &chown_cmd,
                                sudo_pw,
                                sender,
                                &component.name,
                            );
                        } else {
                            // Clean up the check file
                            let _ = std::fs::remove_file(dir_path.join(".permission_check"));
                        }
                    }
                }
            }
        }
        "vllm-studio" => {
            use crate::installers::components::vllm_studio::{
                VllmStudioConfig, VllmStudioInstaller,
            };
            let inst = VllmStudioInstaller::new(VllmStudioConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Step 1: git clone (idempotent — pulls if already exists)
            let clone_cmd = inst.build_git_clone_command();
            let target_dir = clone_cmd.args.last().cloned().unwrap_or_default();
            git_clone_or_pull(
                inst.repo_url(),
                &target_dir,
                &[],
                None,
                sender,
                &component.name,
            )?;

            // Step 2: Install packages
            let pkg_mgr = if crate::installers::common::utils::command_exists("npm") {
                "npm"
            } else {
                "yarn"
            };
            let cmd = inst.build_pkg_install_command(pkg_mgr);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 3: Build frontend (non-fatal — build script may not exist yet)
            // Run from the frontend/ subdirectory where package.json with "build" script lives.
            let frontend_dir = PathBuf::from(&target_dir).join("frontend");
            let cmd = inst.build_frontend_build_command(pkg_mgr);
            let build_result = execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    Some(frontend_dir),
                ),
                None,
                sender,
                &component.name,
            );
            if let Err(err) = build_result {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[native] [WARN] {} frontend build failed (non-fatal): {}",
                        component.name, err
                    ),
                    false,
                ));
            }
        }

        // ── comfyui ───────────────────────────────────────────────────
        "comfyui" => {
            use crate::installers::components::comfyui::{ComfyuiConfig, ComfyuiInstaller};
            let inst = ComfyuiInstaller::new(ComfyuiConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Step 1: Check PyTorch
            let cmd = inst.build_pytorch_check_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Step 2: git clone (idempotent — pulls if already exists)
            let clone_cmd = inst.build_git_clone_command();
            let target_dir = clone_cmd.args.last().cloned().unwrap_or_default();
            git_clone_or_pull(
                inst.repo_url(),
                &target_dir,
                &[],
                None,
                sender,
                &component.name,
            )?;

            // Step 3: pip install — filter requirements to exclude CUDA deps.
            // (Stage 2 tenet: no nvidia deps EVER. ComfyUI previously installed
            // requirements.txt verbatim, which could pull xformers/flash-attn
            // CUDA wheels. The ROCm torch is already present, so torch lines are
            // stripped and reused, not reinstalled from a CUDA index.)
            let req_path = format!("{}/requirements.txt", target_dir);
            if std::path::Path::new(&req_path).exists() {
                let filtered = match filter_cuda_requirements(&req_path) {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = sender.send(InstallerEvent::Log(
                            format!(
                                "[native] WARN: could not read ComfyUI requirements {req_path}: {e} — skipping requirements install"
                            ),
                            true,
                        ));
                        anyhow::bail!("failed to read ComfyUI requirements {req_path}: {e}");
                    }
                };
                if !filtered.is_empty() {
                    let tmp_dir = std::env::temp_dir().join("rusty-stack-comfyui");
                    let _ = std::fs::create_dir_all(&tmp_dir);
                    let filtered_path = tmp_dir.join("requirements_filtered.txt");
                    std::fs::write(&filtered_path, &filtered).with_context(|| {
                        format!(
                            "Failed to write filtered requirements to {:?}",
                            filtered_path
                        )
                    })?;
                    let filtered_path_str = filtered_path.to_string_lossy().to_string();
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] Installing filtered requirements (CUDA deps excluded) from {}",
                            req_path
                        ),
                        false,
                    ));
                    let cmd = inst.build_pip_install_command(&filtered_path_str);
                    execute_native_command(
                        &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                        None,
                        sender,
                        &component.name,
                    )?;
                    let _ = std::fs::remove_file(&filtered_path);
                } else {
                    let _ = sender.send(InstallerEvent::Log(
                        "[native] [WARN] All ComfyUI requirements were filtered out, nothing to install".to_string(),
                        false,
                    ));
                }
            } else {
                let _ = sender.send(InstallerEvent::Log(
                    "[native] [WARN] ComfyUI requirements.txt not found, skipping dependency installation".to_string(),
                    false,
                ));
            }
        }

        // ── textgen ───────────────────────────────────────────────────
        "textgen" => {
            use crate::installers::components::textgen::{TextgenConfig, TextgenInstaller};
            // Route through the canonical managed interpreter (~/.mlstack/global),
            // NOT bare system python3 — every installer must respect the created env.
            let textgen_python = resolve_python_bin();
            // textgen: always use python -m pip --break-system-packages.
            // uv pip install --system fails on uv-managed Pythons ("externally managed");
            // in the managed venv `python -m pip` works and --break-system-packages is a
            // harmless no-op.
            let inst = TextgenInstaller::new(TextgenConfig {
                python_bin: textgen_python.clone(),
                use_uv: false,
                break_system_packages: true,
                ..Default::default()
            });

            // Step 1: git clone (idempotent — pulls if already exists)
            let clone_cmd = inst.build_git_clone_command();
            let target_dir = clone_cmd.args.last().cloned().unwrap_or_default();
            git_clone_or_pull(
                inst.repo_url(),
                &target_dir,
                &[],
                None,
                sender,
                &component.name,
            )?;

            // Step 2: Install AMD requirements FIRST.
            // AMD requirements contain ROCm-specific wheels (e.g. llama-cpp+rocm7.2)
            // that must be installed before the filtered generic requirements.
            // This ensures ROCm wheels take precedence over any CUDA fallbacks.
            let amd_req_full = format!("{}/requirements/full/requirements_amd.txt", target_dir);
            let amd_req_subdir = format!("{}/requirements/requirements_amd.txt", target_dir);
            let amd_req_path = if std::path::Path::new(&amd_req_full).exists() {
                amd_req_full
            } else if std::path::Path::new(&amd_req_subdir).exists() {
                amd_req_subdir
            } else {
                String::new()
            };
            if !amd_req_path.is_empty() {
                let _ = sender.send(InstallerEvent::Log(
                    "[native] Installing AMD-specific requirements first (ROCm wheels)".to_string(),
                    false,
                ));
                let cmd = inst.build_amd_requirements_command(&amd_req_path);
                execute_native_command(
                    &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                    None,
                    sender,
                    &component.name,
                )?;
            }

            // Step 3: Install filtered main requirements.
            // The original script filters with grep -v to exclude CUDA/nvidia packages:
            //   grep -v -E '^(nvidia-|cuda|tensorrt|triton([=<>!\[]|$)|xformers|
            //                flash-attn|torch([=<>! ]|$)|torchvision|torchaudio)'
            // Additionally, we exclude any .whl URLs containing CUDA markers (+cu\d)
            // and any ik_llama lines (CUDA-only builds).
            let req_path_root = format!("{}/requirements.txt", target_dir);
            let req_path_full = format!("{}/requirements/full/requirements.txt", target_dir);
            let req_path_subdir = format!("{}/requirements/requirements.txt", target_dir);
            let raw_req_path = if std::path::Path::new(&req_path_full).exists() {
                req_path_full
            } else if std::path::Path::new(&req_path_root).exists() {
                req_path_root
            } else if std::path::Path::new(&req_path_subdir).exists() {
                req_path_subdir
            } else {
                String::new()
            };

            if !raw_req_path.is_empty() {
                // Read and filter the requirements file to exclude CUDA deps
                let filtered = match filter_cuda_requirements(&raw_req_path) {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = sender.send(InstallerEvent::Log(
                            format!(
                                "[native] WARN: could not read textgen requirements {raw_req_path}: {e} — skipping requirements install"
                            ),
                            true,
                        ));
                        anyhow::bail!("failed to read textgen requirements {raw_req_path}: {e}");
                    }
                };
                if !filtered.is_empty() {
                    // Write filtered requirements to a temp file
                    let tmp_dir = std::env::temp_dir().join("rusty-stack-textgen");
                    let _ = std::fs::create_dir_all(&tmp_dir);
                    let filtered_path = tmp_dir.join("requirements_filtered.txt");
                    std::fs::write(&filtered_path, &filtered).with_context(|| {
                        format!(
                            "Failed to write filtered requirements to {:?}",
                            filtered_path
                        )
                    })?;
                    let filtered_path_str = filtered_path.to_string_lossy().to_string();
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] Installing filtered requirements (CUDA deps excluded) from {}",
                            raw_req_path
                        ),
                        false,
                    ));
                    let cmd = inst.build_pip_install_command(&filtered_path_str);
                    execute_native_command(
                        &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                        None,
                        sender,
                        &component.name,
                    )?;
                    let _ = std::fs::remove_file(&filtered_path);
                } else {
                    let _ = sender.send(InstallerEvent::Log(
                        "[native] [WARN] All requirements were filtered out, nothing to install"
                            .to_string(),
                        false,
                    ));
                }
            } else {
                let _ = sender.send(InstallerEvent::Log(
                    "[native] [WARN] requirements.txt not found, skipping dependency installation"
                        .to_string(),
                    false,
                ));
            }
        }

        // ── onnx ──────────────────────────────────────────────────────
        "onnx" => {
            use crate::installers::components::onnxruntime::{
                OnnxInstallMethod, OnnxRuntimeConfig, OnnxRuntimeInstaller,
            };

            let detected_rocm_version = detect_rocm_version();
            let detected_gpu_arch = detect_gpu_arch().unwrap_or_else(|| {
                eprintln!("Warning: no GPU arch detected for ONNX, falling back to gfx1100");
                "gfx1100".to_string()
            });

            // Build config with detected hardware info
            let config = OnnxRuntimeConfig {
                rocm_version: Some(detected_rocm_version.clone()),
                rocm_release: Some({
                    // Derive release from version: keep full semver when available.
                    // If only major.minor is detected, default ROCm 7.2 to current 7.2.4.
                    let parts: Vec<&str> = detected_rocm_version.split('.').collect();
                    if parts.len() >= 3 {
                        detected_rocm_version.clone()
                    } else if parts.len() >= 2 && parts[0] == "7" && parts[1] == "2" {
                        "7.2.4".to_string()
                    } else if parts.len() >= 2 {
                        format!("{}.{}.3", parts[0], parts[1])
                    } else {
                        "7.2.4".to_string()
                    }
                }),
                gpu_arch: Some(detected_gpu_arch),
                python_bin: resolve_python_bin(),
                install_method: OnnxInstallMethod::MigraphxWheel,
                ..Default::default()
            };
            let inst = OnnxRuntimeInstaller::new(config);

            // Step 1: Uninstall ALL existing onnxruntime variants
            let cmd = inst.build_uninstall_command();
            let _ = execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            );

            // Step 2: Install onnxruntime-migraphx from AMD repo (default)
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {} — installing onnxruntime-migraphx from AMD repo (ROCm {})",
                    component.name, detected_rocm_version
                ),
                false,
            ));
            let cmd = inst.build_migraphx_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            )?;

            // Step 3: Validate AMD execution provider availability before accepting install
            let cmd = inst.build_provider_validation_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &cmd.program,
                    &cmd.args,
                    &cmd.env,
                    cmd.working_dir.clone(),
                ),
                None,
                sender,
                &component.name,
            )?;

            // Step 4: Run model optimizer on known .onnx model paths
            let model_dirs = [
                dirs::home_dir().map(|h| h.join(".mlstack/models")),
                dirs::home_dir().map(|h| h.join(".local/share/models")),
            ];
            for model_dir in model_dirs.iter().flatten() {
                if model_dir.exists() {
                    if let Ok(entries) = std::fs::read_dir(model_dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                                let model_path = path.to_string_lossy().to_string();
                                // Skip already-optimized models
                                if model_path.ends_with(".onnx.optimized") {
                                    continue;
                                }
                                let _ = sender.send(InstallerEvent::Log(
                                    format!(
                                        "[native] {} — optimizing model: {}",
                                        component.name, model_path
                                    ),
                                    false,
                                ));
                                let cmd = inst.build_model_optimizer_command(&model_path);
                                let _ = execute_native_command(
                                    &NativeCommand::from_shell_cmd_with_dir(
                                        &cmd.program,
                                        &cmd.args,
                                        &cmd.env,
                                        cmd.working_dir.clone(),
                                    ),
                                    None,
                                    sender,
                                    &component.name,
                                );
                            }
                        }
                    }
                }
            }

            let _ = sender.send(InstallerEvent::Log(
                format!("[native] {} — ONNX Runtime (MIGraphX) installed successfully. \
                    Source build available via OnnxInstallMethod::SourceBuild for ROCMExecutionProvider.",
                    component.name),
                false,
            ));
        }

        // ── bitsandbytes ──────────────────────────────────────────────
        "bitsandbytes" => {
            use crate::installers::components::bitsandbytes_multi::{
                BitsAndBytesConfig, BitsAndBytesInstaller,
            };
            let inst = BitsAndBytesInstaller::new(BitsAndBytesConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });
            let target_dir = format!("{}/.mlstack/bitsandbytes", ctx.user_home);

            // Force reinstall: purge previous bitsandbytes (non-fatal)
            if is_force_reinstall() {
                purge_component_packages(&["bitsandbytes"], sender, &component.name);
            }

            // Step 1: git clone (idempotent — pulls if already exists)
            git_clone_or_pull(
                "https://github.com/ROCm/bitsandbytes.git",
                &target_dir,
                &["--recursive"],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: Install from PyPI
            let cmd = inst.build_pypi_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── rocm-smi ──────────────────────────────────────────────────
        "rocm-smi" => {
            use crate::installers::components::rocm_smi::RocmSmiInstaller;
            let inst = RocmSmiInstaller::new();
            let distro = crate::installers::common::DistroFacade::detect();
            let target_dir = format!("{}/.mlstack/rocm_smi", ctx.user_home);

            // Step 1: git clone (idempotent — pulls if already exists)
            git_clone_or_pull(
                inst.git_clone_url(),
                &target_dir,
                &[],
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: package install (skip on Arch if rocm-smi already on PATH)
            if !(distro.family() == crate::platform::detection::DistroFamily::Arch
                && crate::installers::common::utils::command_exists("rocm-smi"))
            {
                let cmd = inst.build_package_install_command(&distro);
                execute_native_command(
                    &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                    sudo_pw,
                    sender,
                    &component.name,
                )?;
            } else {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "[native] {} — rocm-smi already on PATH, skipping package install",
                        component.name
                    ),
                    false,
                ));
            }
        }

        // ── migraphx ──────────────────────────────────────────────────
        "migraphx" => {
            use crate::installers::components::migraphx_multi::{
                MigraphxConfig, MigraphxInstaller,
            };
            let inst = MigraphxInstaller::new(MigraphxConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });
            let distro = crate::installers::common::DistroFacade::detect();
            let packages = inst.required_packages(&distro);

            // Step 1: system package install
            let cmd = inst.build_package_install_command(&distro, &packages);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: pip install (skip on Arch — no pip wheel available)
            if inst.should_skip_pip_install(&distro) {
                if let Some(msg) = inst.build_limitation_message(&distro) {
                    let _ = sender.send(InstallerEvent::Log(
                        format!("[native] [INFO] {}", msg),
                        false,
                    ));
                }
            } else {
                let cmd = inst.build_pip_install_command();
                execute_native_command(
                    &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                    None,
                    sender,
                    &component.name,
                )?;
            }

            // Step 3: functional check — migraphx must actually LOAD. On Arch a
            // partial upgrade can leave migraphx installed but unrunnable (e.g.
            // libprotobuf.so.35.1.0 / libabsl SONAME desync). Surface the missing
            // libs + the fix instead of a vague "verification errors".
            match crate::installers::components::migraphx_multi::migraphx_driver_status() {
                Ok(()) => {}
                Err(missing) => {
                    let detail = if missing.is_empty() {
                        "migraphx-driver would not run".to_string()
                    } else {
                        format!("missing shared libraries: {}", missing.join(", "))
                    };
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "[native] [WARN] {} installed but NOT functional — {detail}. \
                             This is usually a library version desync from a partial upgrade. \
                             Run `sudo pacman -Syu` (Arch/CachyOS) or the equivalent system \
                             update to sync dependencies, then re-verify.",
                            component.name
                        ),
                        true,
                    ));
                }
            }
        }

        // ── pytorch-profiler ──────────────────────────────────────────
        "pytorch-profiler" => {
            use crate::installers::components::pytorch_profiler::{
                PytorchProfilerConfig, PytorchProfilerInstaller,
            };
            let inst = PytorchProfilerInstaller::new(PytorchProfilerConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Step 1: pip install
            let cmd = inst.build_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── wandb ─────────────────────────────────────────────────────
        "wandb" => {
            use crate::installers::components::wandb::{WandbConfig, WandbInstaller};
            let inst = WandbInstaller::new(WandbConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Step 1: pip install
            let cmd = inst.build_install_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;
        }

        // ── amdgpu-drivers ────────────────────────────────────────────
        "amdgpu-drivers" => {
            use crate::installers::components::amdgpu_drivers::{AmdgpuConfig, AmdgpuInstaller};
            let inst = AmdgpuInstaller::new(AmdgpuConfig::default());
            let distro = crate::installers::common::DistroFacade::detect();
            let packages = inst.required_packages(&distro);

            // Step 1: package install
            let cmd = inst.build_package_install_command(&distro, &packages);
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Step 2: env vars (HSA_OVERRIDE_GFX_VERSION / PYTORCH_ROCM_ARCH) are sourced
            // from the persistent env file (~/.mlstack_env via ensure_mlstack_env) — they
            // belong to ROCm RUNTIME, not driver install. The previous per-step `export X=Y`
            // was doubly broken: a bare `export` run as a subprocess is a no-op (it cannot
            // mutate the parent shell), and HSA_OVERRIDE was hardcoded to 11.0.0.

            // Step 3: verify
            let cmd = inst.build_verify_command();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                None,
                sender,
                &component.name,
            )?;

            // Driver access requires render/video group membership on distros that
            // mode /dev/kfd + render nodes 0660. Idempotent + system-aware.
            ensure_user_in_groups(&["render", "video"], sudo_pw, sender, &component.name);
        }

        // ── migraphx-python ───────────────────────────────────────────
        "migraphx-python" => {
            use crate::installers::components::migraphx_python::{
                MigraphxPythonConfig, MigraphxPythonInstaller,
            };
            let inst = MigraphxPythonInstaller::new(MigraphxPythonConfig {
                python_bin: resolve_python_bin(),
                ..Default::default()
            });

            // Check distro availability — skip on Arch (no pip wheel)
            let distro = crate::installers::common::DistroFacade::detect();
            if !inst.is_available_on_distro(&distro) {
                if let Some(msg) = inst.build_unavailable_message(&distro) {
                    let _ = sender.send(InstallerEvent::Log(
                        format!("[native] [SKIP] {}", msg),
                        false,
                    ));
                }
            } else {
                // Step 1: pip install
                let cmd = inst.build_install_command();
                execute_native_command(
                    &NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env),
                    None,
                    sender,
                    &component.name,
                )?;
            }
        }

        // ── enhanced-env ──────────────────────────────────────────────
        "enhanced-env" => {
            // Enhanced env setup is handled by ensure_mlstack_env() in
            // run_installation() above. This is a no-op placeholder for
            // the dispatch table to ensure complete coverage.
            let _ = sender.send(InstallerEvent::Log(
                "[native] Enhanced environment setup handled during pre-install phase".into(),
                false,
            ));
        }

        // ── Benchmark components ──────────────────────────────────────
        "mlperf-inference"
        | "rocm-benchmarks"
        | "gpu-memory-bandwidth"
        | "rocm-smi-bench"
        | "pytorch-performance"
        | "vllm-performance"
        | "deepspeed-performance"
        | "megatron-performance"
        | "onnx-performance"
        | "rusty-llama-performance"
        | "flash-attention-ck-performance"
        | "all-benchmarks" => {
            run_native_benchmark(&component.id, sender)?;
        }

        // ── FastVideo ──────────────────────────────────────────────────
        "fastvideo" => {
            use crate::installers::components::fastvideo::{
                detect_discrete_gpu_archs_from_sysfs, fastvideo_build_warning, post_checkout_plan,
                resolve_gpu_archs, resolve_install_user, select_visible_gpu_archs, FastVideoConfig,
                FastVideoInstallStep, FastVideoInstaller,
            };
            let detected_archs =
                detect_discrete_gpu_archs_from_sysfs(PathBuf::from("/sys/class/drm").as_path());
            let selected_archs = select_visible_gpu_archs(
                &detected_archs,
                ctx.env_exports
                    .get("ROCR_VISIBLE_DEVICES")
                    .map(String::as_str),
                ctx.env_exports
                    .get("HIP_VISIBLE_DEVICES")
                    .map(String::as_str),
                ctx.env_exports
                    .get("CUDA_VISIBLE_DEVICES")
                    .map(String::as_str),
            );
            let gpu_archs = resolve_gpu_archs(
                &selected_archs,
                ctx.env_exports.get("GPU_ARCHS").map(String::as_str),
                ctx.env_exports.get("GPU_ARCH").map(String::as_str),
                None,
            );
            let python_bin = ctx
                .env_exports
                .get("MLSTACK_PYTHON_BIN")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(resolve_python_bin);
            let inst = FastVideoInstaller::new(FastVideoConfig {
                gpu_archs: gpu_archs.clone(),
                python_bin: python_bin.clone(),
            });
            let build_dir = PathBuf::from("/tmp/FastVideo_ROCm_build");

            // Always start from a fresh directory. Reusing a failed build could
            // execute stale sitecustomize/build files or contaminate the wheel.
            let initial_cleanup = inst.cleanup();
            execute_native_command(
                &NativeCommand::from_shell_cmd(
                    &initial_cleanup.program,
                    &initial_cleanup.args,
                    &initial_cleanup.env,
                ),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // mkdir -p build dir
            let mkdir_cmd = inst.mkdir_build_dir();
            execute_native_command(
                &NativeCommand::from_shell_cmd(&mkdir_cmd.program, &mkdir_cmd.args, &mkdir_cmd.env),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // Clone only into the freshly emptied directory.
            let clone_cmd = inst.git_clone();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &clone_cmd.program,
                    &clone_cmd.args,
                    &clone_cmd.env,
                    clone_cmd.working_dir,
                ),
                sudo_pw,
                sender,
                &component.name,
            )?;

            // git checkout branch
            let checkout_cmd = inst.git_checkout();
            execute_native_command(
                &NativeCommand::from_shell_cmd_with_dir(
                    &checkout_cmd.program,
                    &checkout_cmd.args,
                    &checkout_cmd.env,
                    Some(build_dir.clone()),
                ),
                sudo_pw,
                sender,
                &component.name,
            )?;

            let sudo_user = std::env::var("SUDO_USER").ok();
            let user = std::env::var("USER").ok();
            let run_user = resolve_install_user(sudo_user.as_deref(), user.as_deref())
                .context("FastVideo install user unavailable from SUDO_USER or USER")?;

            for step in post_checkout_plan() {
                let step_sudo = if step.requires_sudo() { sudo_pw } else { None };
                if step == FastVideoInstallStep::KernelInstall {
                    let snapshot = inst.distribution_snapshot(
                        "before",
                        crate::installers::common::nvidia_blocklist::NVIDIA_RUNTIME_PREFIXES,
                    );
                    execute_native_command(
                        &NativeCommand::from_shell_cmd(
                            &snapshot.program,
                            &snapshot.args,
                            &snapshot.env,
                        ),
                        None,
                        sender,
                        &component.name,
                    )?;
                }

                let command = match step {
                    FastVideoInstallStep::Chown => {
                        let cmd = inst.chown_build_dir(&run_user);
                        NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env)
                    }
                    FastVideoInstallStep::ManagedDependenciesPreflight => {
                        let cmd = inst.managed_dependencies_preflight();
                        NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env)
                    }
                    FastVideoInstallStep::FlashAttentionPreflight => {
                        let cmd = inst.flash_attention_preflight();
                        NativeCommand::from_shell_cmd(&cmd.program, &cmd.args, &cmd.env)
                    }
                    FastVideoInstallStep::SourcePolicyPreflight => {
                        let cmd = inst.source_policy_preflight();
                        NativeCommand::from_shell_cmd_with_dir(
                            &cmd.program,
                            &cmd.args,
                            &cmd.env,
                            cmd.working_dir,
                        )
                    }
                    FastVideoInstallStep::KernelInstall => {
                        let _ = sender.send(InstallerEvent::Log(
                            fastvideo_build_warning(&gpu_archs),
                            false,
                        ));
                        let cmd = inst.pip_install_kernel();
                        NativeCommand::from_shell_cmd_with_dir(
                            &cmd.program,
                            &cmd.args,
                            &cmd.env,
                            cmd.working_dir,
                        )
                    }
                    FastVideoInstallStep::PackageInstall => {
                        let cmd = inst.pip_install_package();
                        NativeCommand::from_shell_cmd_with_dir(
                            &cmd.program,
                            &cmd.args,
                            &cmd.env,
                            cmd.working_dir,
                        )
                    }
                };
                execute_native_command(&command, step_sudo, sender, &component.name)?;
            }

            let snapshot = inst.distribution_snapshot(
                "after",
                crate::installers::common::nvidia_blocklist::NVIDIA_RUNTIME_PREFIXES,
            );
            execute_native_command(
                &NativeCommand::from_shell_cmd(&snapshot.program, &snapshot.args, &snapshot.env),
                None,
                sender,
                &component.name,
            )?;

            // cleanup
            let cleanup_cmd = inst.cleanup();
            execute_native_command(
                &NativeCommand::from_shell_cmd(
                    &cleanup_cmd.program,
                    &cleanup_cmd.args,
                    &cleanup_cmd.env,
                ),
                sudo_pw,
                sender,
                &component.name,
            )?;
        }
        // ── llama-cpp ──────────────────────────────────────────────────
        "llama-cpp" => {
            use crate::installers::components::llama_cpp::{LlamaCppConfig, LlamaCppInstaller};
            let gpu_arch = ctx
                .env_exports
                .get("GPU_ARCH")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| {
                    detect_gpu_arch().unwrap_or_else(|| {
                        eprintln!("Warning: no GPU arch detected, falling back to gfx1100");
                        "gfx1100".to_string()
                    })
                });
            let channel = ctx
                .env_exports
                .get("ROCM_CHANNEL")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "latest".to_string());
            let inst = LlamaCppInstaller::new(LlamaCppConfig {
                gpu_arch: gpu_arch.clone(),
                channel: channel.clone(),
                ..Default::default()
            });
            let home = ctx.user_home;
            let manifest = inst.check_latest_release();
            let strategy = inst.resolve_install_strategy(home, manifest.as_ref());
            match &strategy {
                crate::installers::components::llama_cpp::InstallStrategy::Prebuilt(plan) => {
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "llama-cpp install strategy selected: prebuilt ({})",
                            plan.arch
                        ),
                        false,
                    ));
                    if let Err(err) = inst.download_prebuilt_binary(plan) {
                        let _ = sender.send(InstallerEvent::Log(
                            format!(
                                "llama-cpp prebuilt download failed; falling back to source: {}",
                                err
                            ),
                            false,
                        ));
                        inst.run_source_install(home).map_err(|source_err| {
                            anyhow::anyhow!(
                                "llama-cpp prebuilt install failed and source fallback failed: {}",
                                source_err
                            )
                        })?;
                    }
                }
                crate::installers::components::llama_cpp::InstallStrategy::Source(plan) => {
                    let _ = sender.send(InstallerEvent::Log(
                        format!(
                            "llama-cpp install strategy selected: source ({})",
                            plan.reason
                        ),
                        false,
                    ));
                    inst.run_source_install(home).map_err(|err| {
                        anyhow::anyhow!("llama-cpp source install failed: {}", err)
                    })?;
                }
            }

            // ── Post-install verification (VAL-CROSS-008/009/010/011) ──
            // Binaries resolve from the managed install dir inside verify; the
            // clone dir is only a best-effort search root for a verification model.
            let clone_dir = inst.build_dir_path().to_string();
            let verification =
                crate::installers::components::llama_cpp::verify_installed_binary(home, &clone_dir);
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "llama-cpp post-install verification: {}",
                    verification.summary
                ),
                false,
            ));
            if verification.stronger_check_passed {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "llama-cpp llama-bench: prefill={:.0} t/s, decode={:.0} t/s",
                        verification.bench_prefill_tps.unwrap_or(0.0),
                        verification.bench_decode_tps.unwrap_or(0.0),
                    ),
                    false,
                ));
            }
            if verification.rdna3_is_valid.unwrap_or(false) {
                let _ = sender.send(InstallerEvent::Log(
                    format!(
                        "llama-cpp RDNA3 proof: device=RDNA3, WMMA={:.0} t/s",
                        verification.rdna3_wmma_tps.unwrap_or(0.0),
                    ),
                    false,
                ));
            }
        }

        // ── Unknown component ─────────────────────────────────────────
        _ => {
            bail!(
                "No native installer dispatch for component '{}'. \
                 This should not happen — is_native_component() should return false for this ID.",
                component.id
            );
        }
    }

    let _ = sender.send(InstallerEvent::Progress {
        component_id: component.id.clone(),
        progress: 0.9,
        message: format!("{} installation complete", component.name),
    });

    Ok(())
}

fn run_script(
    component: &Component,
    script_path: &str,
    sudo_password: Option<String>,
    batch_mode: bool,
    install_method: &str,
    sender: &Sender<InstallerEvent>,
    input_rx: Arc<Mutex<Receiver<String>>>,
) -> Result<()> {
    let user_home = resolve_mlstack_user_home();
    let user_name = std::env::var("USER").unwrap_or_else(|_| "user".into());
    let preserve_env = "HOME,USER,LOGNAME,PATH,MLSTACK_USER_HOME,MLSTACK_SKIP_TORCH_INSTALL,MLSTACK_PYTHON_BIN,MLSTACK_ENV_NAME,MLSTACK_INSTALL_METHOD,INSTALL_METHOD,PIP_BREAK_SYSTEM_PACKAGES,PIP_ROOT_USER_ACTION,UV_PIP_BREAK_SYSTEM_PACKAGES,UV_SYSTEM_PYTHON,PYTHONPATH,LD_LIBRARY_PATH,ROCM_HOME,ROCM_PATH,ROCM_VERSION,ROCM_CHANNEL,GPU_ARCH,GPU_ARCHS,PYTORCH_ROCM_ARCH,HSA_OVERRIDE_GFX_VERSION,HIP_VISIBLE_DEVICES,CUDA_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES,AITER_JIT_DIR,HIP_PATH,HIP_ROOT_DIR,ROCM_ROOT,HIPCC_BIN_DIR,VLLM_TARGET_DEVICE,VLLM_USE_ROCM,USE_ROCM,VLLM_VERSION,UV_PYTHON,DS_ACCELERATOR,FORCE,PYTORCH_REINSTALL,MLSTACK_FORCE_REINSTALL,MLSTACK_SUDO_PASSWORD,MLSTACK_TARGET_UID,MLSTACK_TARGET_GID";

    let venv_bin = PathBuf::from(&user_home).join("rocm_venv").join("bin");
    let local_bin = PathBuf::from(&user_home).join(".local").join("bin");
    let mut path = std::env::var("PATH").unwrap_or_default();

    // Add .local/bin if missing
    if local_bin.exists() {
        let local_str = local_bin.to_string_lossy();
        if !path.split(':').any(|entry| entry == local_str) {
            if path.is_empty() {
                path = local_str.to_string();
            } else {
                path = format!("{}:{}", path, local_str);
            }
        }
    }

    if venv_bin.exists() {
        let venv_str = venv_bin.to_string_lossy();
        if !path.split(':').any(|entry| entry == venv_str) {
            if path.is_empty() {
                path = venv_str.to_string();
            } else {
                path = format!("{}:{}", venv_str, path);
            }
        }
    }

    let env_exports = load_mlstack_env_exports(&user_home);
    let persistent_python = env_exports
        .get("MLSTACK_PYTHON_BIN")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    // Tenet 1: MLSTACK_ENV_NAME (set by `rusty install --env <name>`) isolates
    // ALL components into ~/.mlstack/envs/<name>/. Falls back to the persisted
    // target, then discovery.
    let python_bin = std::env::var("MLSTACK_ENV_NAME")
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .map(|n| {
            crate::platform::environment::mlstack_env_dir(&n)
                .join("bin")
                .join("python")
                .to_string_lossy()
                .to_string()
        })
        .or(persistent_python)
        .unwrap_or_else(resolve_python_bin);
    let force_reinstall = is_force_reinstall();
    let force_env_value = if force_reinstall { "true" } else { "false" };
    let sudo_password_env = sudo_password.clone();
    let user_home_meta = fs::metadata(&user_home).ok();
    #[cfg(unix)]
    let (mlstack_target_uid, mlstack_target_gid) = {
        use std::os::unix::fs::MetadataExt;
        (
            user_home_meta.as_ref().map(|meta| meta.uid().to_string()),
            user_home_meta.as_ref().map(|meta| meta.gid().to_string()),
        )
    };
    #[cfg(not(unix))]
    let (mlstack_target_uid, mlstack_target_gid): (Option<String>, Option<String>) = (None, None);

    // Check if this specific component needs sudo
    let component_needs_sudo = component.needs_sudo && needs_sudo();

    let mut command = if component_needs_sudo {
        let mut cmd = Command::new("sudo");
        cmd.arg("-S")
            .arg("-p")
            .arg("")
            .arg(format!("--preserve-env={}", preserve_env))
            .arg("bash")
            .arg(script_path);
        cmd
    } else {
        let mut cmd = Command::new("bash");
        cmd.arg(script_path);
        cmd
    };

    let mut pythonpath = std::env::var("PYTHONPATH").unwrap_or_default();
    let rocm_lib = "/opt/rocm/lib";
    if !pythonpath.split(':').any(|entry| entry == rocm_lib) {
        if pythonpath.is_empty() {
            pythonpath = rocm_lib.to_string();
        } else {
            pythonpath = format!("{}:{}", rocm_lib, pythonpath);
        }
    }

    // Clean PYTHONPATH of any /tmp paths to avoid importing from build dirs
    pythonpath = pythonpath
        .split(':')
        .filter(|part| !part.starts_with("/tmp/"))
        .collect::<Vec<_>>()
        .join(":");

    let rocm_version = env_exports
        .get("ROCM_VERSION")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(detect_rocm_version);
    let gpu_arch = env_exports
        .get("GPU_ARCH")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            env_exports
                .get("PYTORCH_ROCM_ARCH")
                .map(|v| v.split(';').next().unwrap_or("").trim().to_string())
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_else(|| {
            detect_gpu_arch().unwrap_or_else(|| {
                eprintln!("Warning: no GPU arch detected, falling back to gfx1100");
                "gfx1100".to_string()
            })
        });
    let py_rocm_arch = env_exports
        .get("PYTORCH_ROCM_ARCH")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| gpu_arch.clone());
    let mut gpu_archs = env_exports
        .get("GPU_ARCHS")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| gpu_arch.clone());
    let mut gpu_list = env_exports
        .get("HIP_VISIBLE_DEVICES")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            env_exports
                .get("CUDA_VISIBLE_DEVICES")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_else(detect_gpu_list);
    let hsa_override = env_exports
        .get("HSA_OVERRIDE_GFX_VERSION")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let aiter_jit_dir = PathBuf::from(&user_home)
        .join(".mlstack")
        .join("aiter")
        .join("jit");

    if component.id == "aiter" {
        gpu_archs = py_rocm_arch
            .split(';')
            .next()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| gpu_arch.clone());
        gpu_list = prioritize_gpu_list_for_arch(&gpu_list, &gpu_archs);
        let _ = fs::create_dir_all(&aiter_jit_dir);
    }

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("HOME", &user_home)
        .env("USER", &user_name)
        .env("LOGNAME", &user_name)
        .env("PATH", &path)
        .env("PYTHONPATH", &pythonpath)
        .env("MLSTACK_USER_HOME", &user_home)
        .env("MLSTACK_PYTHON_BIN", &python_bin)
        .env("UV_PYTHON", &python_bin)
        .env("ROCM_HOME", "/opt/rocm")
        .env("ROCM_PATH", "/opt/rocm")
        .env("ROCM_VERSION", &rocm_version)
        .env("ROCM_CHANNEL", "latest")
        .env("GPU_ARCH", &gpu_arch)
        .env("GPU_ARCHS", &gpu_archs)
        .env("PYTORCH_ROCM_ARCH", &py_rocm_arch)
        .env("HIP_PATH", "/opt/rocm")
        .env("HIP_ROOT_DIR", "/opt/rocm")
        .env("ROCM_ROOT", "/opt/rocm")
        .env("HIPCC_BIN_DIR", "/opt/rocm/bin")
        .env("HIP_VISIBLE_DEVICES", &gpu_list)
        .env("CUDA_VISIBLE_DEVICES", &gpu_list)
        .env("ROCR_VISIBLE_DEVICES", &gpu_list)
        .env("MLSTACK_BATCH_MODE", if batch_mode { "1" } else { "0" })
        .env("MLSTACK_INSTALL_METHOD", install_method)
        .env("INSTALL_METHOD", install_method)
        .env("PIP_BREAK_SYSTEM_PACKAGES", "1")
        .env("PIP_ROOT_USER_ACTION", "ignore")
        .env("UV_PIP_BREAK_SYSTEM_PACKAGES", "1")
        .env("UV_SYSTEM_PYTHON", "1")
        .env("FORCE", force_env_value)
        .env("PYTORCH_REINSTALL", force_env_value)
        .env("MLSTACK_FORCE_REINSTALL", force_env_value)
        .env(
            "MLSTACK_SKIP_TORCH_INSTALL",
            if component.id == "pytorch" { "0" } else { "1" },
        )
        .env("RUSTY_STACK", "true");

    if let Some(uid) = mlstack_target_uid.as_deref() {
        command.env("MLSTACK_TARGET_UID", uid);
    }
    if let Some(gid) = mlstack_target_gid.as_deref() {
        command.env("MLSTACK_TARGET_GID", gid);
    }

    if let Some(password) = sudo_password_env {
        command.env("MLSTACK_SUDO_PASSWORD", password);
    }

    if force_reinstall {
        if let Ok(script_contents) = fs::read_to_string(script_path) {
            let supports_force_flag = script_contents.contains("--force)")
                || script_contents.contains("--force ")
                || script_contents.contains("\"--force\"")
                || script_contents.contains("'--force'");
            if supports_force_flag {
                command.arg("--force");
            }
        }
    }

    if let Some(value) = hsa_override {
        command.env("HSA_OVERRIDE_GFX_VERSION", value);
    }
    if component.id == "aiter" {
        command.env("AITER_JIT_DIR", aiter_jit_dir);
    }

    let mut child = command.spawn().with_context(|| {
        format!(
            "Failed to spawn installer script '{}' for {}. \
                 Ensure bash is installed and the script file is accessible.",
            script_path, component.name
        )
    })?;

    let mut child_stdin = child.stdin.take().with_context(|| {
        format!(
            "Failed to open stdin for script '{}' — \
             this usually indicates a system resource issue.",
            script_path
        )
    })?;
    if let Some(password) = sudo_password {
        let _ = child_stdin.write_all(password.as_bytes());
        let _ = child_stdin.write_all(b"\n");
        let _ = child_stdin.flush();
    }

    // Send default input if any
    if let Some(default_input) = default_component_input(&component.id) {
        let _ = child_stdin.write_all(default_input.as_bytes());
        let _ = child_stdin.flush();
    }

    let stop_input = Arc::new(AtomicBool::new(false));

    // Handle input forwarding
    let stop_input_clone = Arc::clone(&stop_input);
    let input_thread = thread::spawn(move || {
        while !stop_input_clone.load(Ordering::Relaxed) {
            let rx = match input_rx.lock() {
                Ok(guard) => guard,
                Err(_) => break, // Mutex poisoned, stop trying
            };
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(input) => {
                    let _ = child_stdin.write_all(input.as_bytes());
                    let _ = child_stdin.flush();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });

    let sender_stdout = sender.clone();
    let component_id = component.id.clone();
    let component_name = component_name(component);
    let stdout_stream = match child.stdout.take() {
        Some(s) => s,
        None => {
            stop_input.store(true, Ordering::Relaxed);
            let _ = input_thread.join();
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[ERROR] {} — failed to capture stdout from script process",
                    component.name
                ),
                false,
            ));
            bail!(
                "{} — failed to capture stdout from script process. \
                 This usually indicates a system resource issue.",
                component.name
            );
        }
    };
    let stdout_handle = thread::spawn(move || {
        let mut reader = BufReader::new(stdout_stream);
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];

        while let Ok(n) = reader.read(&mut chunk) {
            if n == 0 {
                break;
            }
            for &b in chunk.iter().take(n) {
                if b == b'\n' || b == b'\r' {
                    let is_transient = b == b'\r';
                    if !buffer.is_empty() {
                        let line = String::from_utf8_lossy(&buffer).to_string();
                        let clean_line = strip_ansi_codes(&line);
                        if !clean_line.trim().is_empty() {
                            let _ = sender_stdout
                                .send(InstallerEvent::Log(clean_line.clone(), is_transient));
                            if let Some(progress) = parse_progress(&clean_line) {
                                let _ = sender_stdout.send(InstallerEvent::Progress {
                                    component_id: component_id.clone(),
                                    progress,
                                    message: format!(
                                        "{}: {}%",
                                        component_name,
                                        (progress * 100.0) as i32
                                    ),
                                });
                            }
                        }
                        buffer.clear();
                    }
                } else {
                    buffer.push(b);
                    if buffer.len() > 8192 {
                        buffer.clear();
                    }
                }
            }
        }
    });

    let sender_stderr = sender.clone();
    let stderr_stream = match child.stderr.take() {
        Some(s) => s,
        None => {
            stop_input.store(true, Ordering::Relaxed);
            let _ = input_thread.join();
            let _ = stdout_handle.join();
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[ERROR] {} — failed to capture stderr from script process",
                    component.name
                ),
                false,
            ));
            bail!(
                "{} — failed to capture stderr from script process. \
                 This usually indicates a system resource issue.",
                component.name
            );
        }
    };
    let stderr_handle = thread::spawn(move || {
        let mut reader = BufReader::new(stderr_stream);
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];

        while let Ok(n) = reader.read(&mut chunk) {
            if n == 0 {
                break;
            }
            for &b in chunk.iter().take(n) {
                if b == b'\n' || b == b'\r' {
                    let is_transient = b == b'\r';
                    if !buffer.is_empty() {
                        let line = String::from_utf8_lossy(&buffer).to_string();
                        let clean_line = strip_ansi_codes(&line);
                        if !clean_line.trim().is_empty() {
                            let _ =
                                sender_stderr.send(InstallerEvent::Log(clean_line, is_transient));
                        }
                        buffer.clear();
                    }
                } else {
                    buffer.push(b);
                    if buffer.len() > 8192 {
                        buffer.clear();
                    }
                }
            }
        }
    });

    let status = child.wait().with_context(|| {
        format!(
            "{} — script process wait failed. \
             The process may have been killed or the system is under heavy load.",
            component.name
        )
    })?;
    stop_input.store(true, Ordering::Relaxed);
    let _ = input_thread.join();
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        let error_detail = match code {
            1 => "General error — check the script output above for details".to_string(),
            2 => "Misuse of shell builtins — check script arguments".to_string(),
            126 => "Script not executable — check file permissions (chmod +x)".to_string(),
            127 => "Command not found in script — ensure required tools are installed".to_string(),
            130 => "Script terminated by Ctrl+C (SIGINT)".to_string(),
            137 => "Script killed (SIGKILL) — possibly out of memory".to_string(),
            139 => "Segmentation fault in script subprocess".to_string(),
            n if n > 128 => format!(
                "Script process terminated by signal {} — check system logs",
                n - 128
            ),
            _ => format!(
                "Script exited with code {} — check the output above for details",
                code
            ),
        };
        let _ = sender.send(InstallerEvent::Log(
            format!(
                "[ERROR] {} failed: {} (exit code {} at {})",
                component.name,
                error_detail,
                code,
                Local::now().format("%H:%M:%S")
            ),
            false,
        ));
        bail!(
            "{} failed: {} (exit code {})",
            component.name,
            error_detail,
            code
        );
    }

    Ok(())
}

fn star_repo(config: &InstallerConfig, sender: &Sender<InstallerEvent>) -> Result<()> {
    if !config.star_repos {
        return Ok(());
    }

    let repo_url = "https://github.com/scooter-lacroix/Stan-s-ML-Stack";

    // Check if gh cli is available and authenticated
    let gh_check = Command::new("gh")
        .arg("auth")
        .arg("status")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if let Ok(status) = gh_check {
        if status.success() {
            let _ = sender.send(InstallerEvent::Log(
                "Starring ML Stack repository on GitHub...".into(),
                false,
            ));
            let star_status = Command::new("gh")
                .arg("repo")
                .arg("star")
                .arg(repo_url)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            if let Ok(s) = star_status {
                if s.success() {
                    let _ = sender.send(InstallerEvent::Log(
                        "Thank you for starring the repo! ✦".into(),
                        false,
                    ));
                }
            }
        }
    }

    Ok(())
}

fn component_name(comp: &Component) -> String {
    comp.name.clone()
}

/// Format a user-friendly error message with actionable guidance based on
/// the component ID and the raw error chain.
///
/// This function translates cryptic error messages into helpful guidance
/// similar to what the original shell scripts provided with `set -e` and
/// explicit error handling.
///
/// Order of checks: component-specific → error-type-specific → generic fallback
fn format_user_friendly_error(component_id: &str, raw_error: &str) -> String {
    let error_lower = raw_error.to_lowercase();

    // ── Component-specific guidance (highest priority) ──────────────

    match component_id {
        "permanent-env" => {
            if error_lower.contains("rocminfo") {
                return format!(
                    "{}: Failed to detect GPU — rocminfo command failed. \
                     Ensure ROCm is properly installed and /opt/rocm/bin is in your PATH. \
                     Original error: {}",
                    component_id, raw_error
                );
            }
            return format!(
                "{}: Environment setup failed. \
                 Ensure ROCm is installed and GPU is accessible. \
                 Original error: {}",
                component_id, raw_error
            );
        }
        "rocm" if error_lower.contains("no such file") || error_lower.contains("not found") => {
            return format!(
                "{}: ROCm installation failed — required packages or repositories not found. \
                 Ensure your system's package manager is configured correctly and ROCm repositories are added. \
                 Original error: {}",
                component_id, raw_error
            );
        }
        "pytorch" => {
            if error_lower.contains("no such file") || error_lower.contains("not found") {
                return format!(
                    "{}: PyTorch installation failed — required packages not found. \
                     Ensure ROCm is installed first and Python 3.10+ is available. \
                     Original error: {}",
                    component_id, raw_error
                );
            }
            return format!(
                "{}: PyTorch installation failed. \
                 Ensure ROCm is installed first, Python 3.10-3.13 is available, \
                 and pip/uv can reach the PyTorch package index. \
                 Original error: {}",
                component_id, raw_error
            );
        }
        "triton" | "flash-attn" | "flash-attn-triton" | "flash-attn-ck" | "vllm" | "aiter" => {
            return format!(
                "{}: Installation failed. \
                 Ensure PyTorch and ROCm are properly installed first \
                 (these are required dependencies). \
                 Original error: {}",
                component_id, raw_error
            );
        }
        "megatron" => {
            return format!(
                "{}: Megatron-LM installation failed. \
                 Ensure PyTorch and MPI4Py are installed first (required dependencies). \
                 Original error: {}",
                component_id, raw_error
            );
        }
        "deepspeed" => {
            return format!(
                "{}: DeepSpeed installation failed. \
                 Ensure PyTorch is installed first (required dependency). \
                 Original error: {}",
                component_id, raw_error
            );
        }
        _ => {}
    }

    // ── Error-type-specific guidance ────────────────────────────────

    if error_lower.contains("permission denied") || error_lower.contains("sudo") {
        return format!(
            "{}: Permission denied. Try running with elevated privileges or check file/directory permissions. \
             Original error: {}",
            component_id, raw_error
        );
    }

    if error_lower.contains("network")
        || error_lower.contains("connection")
        || error_lower.contains("timeout")
        || error_lower.contains("resolve")
    {
        return format!(
            "{}: Network error — failed to download packages or clone repositories. \
             Check your internet connection and try again. \
             Original error: {}",
            component_id, raw_error
        );
    }

    if error_lower.contains("out of memory")
        || error_lower.contains("cannot allocate")
        || error_lower.contains("killed")
    {
        return format!(
            "{}: Out of memory — the build process was killed due to insufficient memory. \
             Try closing other applications or increasing swap space. \
             Original error: {}",
            component_id, raw_error
        );
    }

    if error_lower.contains("exit code") {
        // Already has exit code context from our execute_native_command improvements
        return raw_error.to_string();
    }

    if error_lower.contains("no such file") || error_lower.contains("not found") {
        return format!(
            "{}: Required file or command not found. \
             Ensure all dependencies are installed before this component. \
             Original error: {}",
            component_id, raw_error
        );
    }

    // ── Generic fallback ────────────────────────────────────────────

    format!(
        "{}: Installation failed. Check the log output above for details. \
         Original error: {}",
        component_id, raw_error
    )
}

/// Detect GPU architecture via sysfs (NO rocminfo).
///
/// Returns `None` if no discrete AMD GPU is detected or gfx arch cannot be resolved.
/// Callers MUST handle the Option — NEVER silently default to gfx000.
pub(crate) fn detect_gpu_arch() -> Option<String> {
    crate::gpu::detect_discrete_amd_gpus()
        .first()
        .and_then(|g| g.gfx_arch.clone())
}

// NOTE (v0.3.0): iGPU classification is unified in `crate::gpu`. The former
// local `IGPU_PATTERNS` + `is_igpu_name` lived here and diverged from the
// copies in `bootstrap/env_setup`, `hardware`, and `platform/linux`. All call
// sites now route through `crate::gpu::is_integrated_gpu_name` /
// `crate::gpu::is_integrated_by_pci_id` / `crate::gpu::device_is_integrated`.

/// Detects the list of discrete AMD GPUs, filtering out integrated GPUs.
/// Returns a comma-separated string of GPU indices (e.g., "0,1" for first and second GPUs).
///
/// This is the SINGLE entry point every consumer should use for the
/// `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` device mask (Stage 1).
/// Detects the list of discrete AMD GPUs, filtering out integrated GPUs.
/// Returns a comma-separated string of GPU indices (e.g., "0,1" for first and second GPUs).
///
/// This is the SINGLE entry point every consumer should use for the
/// `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` device mask (Stage 1).
///
/// Returns `"0"` only as the legacy default when NO discrete GPU is detected —
/// preserving backward-compatible visibility for env-var defaulting. Consumers
/// that must distinguish "no dGPU" from "device 0" should call
/// [`detect_discrete_gpus`] instead, which returns an empty `Vec` when none is
/// found.
pub(crate) fn detect_gpu_list() -> String {
    let gpus = detect_discrete_gpus();
    if gpus.is_empty() {
        "0".to_string()
    } else {
        gpus.join(",")
    }
}

/// Detect the discrete AMD GPU indices, **without** the legacy `"0"` fallback.
///
/// Returns an empty `Vec` when no discrete GPU is confirmed by any detector
/// (rocminfo → rocm-smi → lspci → sysfs). This is the canonical "how many real
/// dGPUs are present" answer — use it for `gpu_count` and for emitting device
/// masks only when a dGPU is actually proven (never leak a default `"0"` onto an
/// iGPU-only host). The resolution order mirrors [`detect_gpu_list`].
pub(crate) fn detect_discrete_gpus() -> Vec<String> {
    crate::gpu::detect_discrete_amd_gpus()
        .iter()
        .enumerate()
        .map(|(i, _)| i.to_string())
        .collect()
}

/// Builds a map of ROCm GPU ID -> gfx architecture (e.g., "1" -> "gfx1100").
fn detect_gpu_arch_by_id() -> HashMap<String, String> {
    let mut mapping: HashMap<String, String> = HashMap::new();

    let rocminfo_path = crate::installers::common::utils::resolve_rocminfo_path();
    let Ok(output) = Command::new(&rocminfo_path).output() else {
        return mapping;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut current_gpu_id: Option<String> = None;

    for line in stdout.lines() {
        let line = line.trim();
        if line.contains("GPU ID:") {
            if let Some(id_part) = line.split(':').nth(1) {
                let id: String = id_part.chars().filter(|c| c.is_ascii_digit()).collect();
                if !id.is_empty() {
                    current_gpu_id = Some(id);
                }
            }
            continue;
        }

        if !line.contains("Name:") || !line.contains("gfx") {
            continue;
        }

        if let Some(id) = current_gpu_id.clone() {
            for token in line.split_whitespace() {
                let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric());
                if cleaned.starts_with("gfx") {
                    mapping.entry(id).or_insert_with(|| cleaned.to_string());
                    break;
                }
            }
        }
    }

    mapping
}

/// Reorders a comma-separated GPU list so the first GPU matches the requested arch when possible.
/// It preserves all original indices and only changes ordering.
fn prioritize_gpu_list_for_arch(gpu_list: &str, target_arch: &str) -> String {
    let target_arch = target_arch.trim();
    if target_arch.is_empty() {
        return gpu_list.to_string();
    }

    let arch_by_id = detect_gpu_arch_by_id();
    if arch_by_id.is_empty() {
        return gpu_list.to_string();
    }

    let mut ids: Vec<String> = gpu_list
        .split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    if ids.len() < 2 {
        return gpu_list.to_string();
    }

    let preferred_pos = ids.iter().position(|id| {
        arch_by_id
            .get(id)
            .map(|v| v == target_arch)
            .unwrap_or(false)
    });
    if let Some(pos) = preferred_pos {
        if pos > 0 {
            let preferred = ids.remove(pos);
            ids.insert(0, preferred);
            return ids.join(",");
        }
    }

    gpu_list.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationResult {
    Verified,
    Failed,
    Missing,
}

impl VerificationResult {
    fn label(&self) -> &'static str {
        match self {
            VerificationResult::Verified => "Verified",
            VerificationResult::Failed => "Failed",
            VerificationResult::Missing => "Missing",
        }
    }
}

fn resolve_python_bin() -> String {
    if let Ok(value) = std::env::var("MLSTACK_PYTHON_BIN") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }

    if let Ok(value) = std::env::var("UV_PYTHON") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }

    // Tenet 1: prefer the managed global env (~/.mlstack/global) — the SINGLE
    // default env — over discovered interpreters, so the bare `rusty` TUI
    // anchors to it once it has been created (by run_installation / `rusty
    // install`). Explicit MLSTACK_PYTHON_BIN/UV_PYTHON above still win.
    let global_py = crate::platform::environment::mlstack_global_python();
    if global_py.exists() {
        return global_py.to_string_lossy().to_string();
    }

    let home = std::env::var("HOME").unwrap_or_default();

    // uv-managed Pythons — PREFERRED over system Pythons for ML workloads.
    // uv installs to ~/.local/share/uv/python/cpython-3.12.*/bin/python3
    // and symlinks to ~/.local/bin/python3
    // Priority order: 3.12 > 3.13 > 3.11 > 3.10 > everything else (3.14+ is fallback)
    let uv_python_dir = PathBuf::from(&home).join(".local/share/uv/python");
    if let Ok(entries) = std::fs::read_dir(&uv_python_dir) {
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
        // Sort: 3.12 first, then 3.13, then 3.11, then 3.10, then others
        uv_pythons.sort_by(|a, b| {
            let va = python_version_tuple(a);
            let vb = python_version_tuple(b);
            python_ml_priority(&va).cmp(&python_ml_priority(&vb))
        });
        // First try: find one with ROCm torch
        for python in &uv_pythons {
            if python_has_rocm_torch(&python.to_string_lossy()) {
                return python.to_string_lossy().to_string();
            }
        }
        // Fallback: use first uv python (any will do)
        if let Some(first) = uv_pythons.first() {
            return first.to_string_lossy().to_string();
        }
    }

    // ~/.local/bin/python3 — the uv symlink
    let local_bin_python = format!("{home}/.local/bin/python3");
    if Path::new(&local_bin_python).exists() {
        if python_has_rocm_torch(&local_bin_python) {
            return local_bin_python;
        }
        return local_bin_python;
    }

    // Legacy candidates (component venvs, system pythons)
    let candidates = [
        Path::new(&home)
            .join("rocm_venv")
            .join("bin")
            .join("python"),
        PathBuf::from("/usr/local/bin/python3"),
        PathBuf::from("/usr/bin/python3"),
        PathBuf::from("python3"),
        PathBuf::from("python"),
    ];

    for candidate in &candidates {
        let candidate_str = candidate.to_string_lossy();
        if candidate_str.is_empty() {
            continue;
        }
        if candidate.is_absolute() && !candidate.exists() {
            continue;
        }
        if python_has_rocm_torch(&candidate_str) {
            return candidate_str.to_string();
        }
    }

    for candidate in &candidates {
        let candidate_str = candidate.to_string_lossy();
        if candidate.is_absolute() && !candidate.exists() {
            continue;
        }
        return candidate_str.to_string();
    }

    "python3".to_string()
}

/// Extract (major, minor) from a python binary path.
fn python_version_tuple(path: &Path) -> (u32, u32) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // Walk up to parent dir name like "cpython-3.12.12-linux-x86_64-gnu"
    let parent = path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    // Try parent dir first (more reliable for uv installs)
    if let Some(v) = parse_cpython_version(parent) {
        return v;
    }
    // Fallback to binary name
    if let Some(rest) = name.strip_prefix("python") {
        let parts: Vec<&str> = rest.split('.').collect();
        if parts.len() >= 2 {
            if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return (major, minor);
            }
        }
    }
    (0, 0)
}

/// Parse version from uv cpython dir name like "cpython-3.12.12-linux-x86_64-gnu".
fn parse_cpython_version(dir_name: &str) -> Option<(u32, u32)> {
    let rest = dir_name.strip_prefix("cpython-")?;
    let parts: Vec<&str> = rest.split('.').collect();
    if parts.len() >= 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        return Some((major, minor));
    }
    None
}

/// Priority ordering for ML workloads: 3.12 > 3.13 > 3.11 > 3.10 > everything else.
fn python_ml_priority(version: &(u32, u32)) -> (i32, u32) {
    match version {
        (3, 12) => (0, version.1), // highest — ML stable target
        (3, 13) => (1, version.1),
        (3, 11) => (2, version.1),
        (3, 10) => (3, version.1),
        _ => (4, version.1), // 3.14+ is lowest priority
    }
}

fn python_has_rocm_torch(python: &str) -> bool {
    Command::new(python)
        .arg("-c")
        .arg(
            "import torch, sys; ok=hasattr(torch,'__version__') and hasattr(torch,'version') and getattr(torch.version,'hip', None); sys.exit(0 if ok else 1)",
        )
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn default_component_input(component_id: &str) -> Option<&'static str> {
    match component_id {
        "ml-stack-core" => Some("9\n0\n"),
        _ => None,
    }
}

fn parse_progress(line: &str) -> Option<f32> {
    let lower = line.to_lowercase();
    if let Some(percent_idx) = lower.find('%') {
        let prefix = &lower[..percent_idx];
        let digits: String = prefix
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            let value: String = digits.chars().rev().collect();
            if let Ok(percent) = value.parse::<f32>() {
                return Some((percent / 100.0).min(1.0));
            }
        }
    }
    if lower.contains("progress") && lower.contains("100") {
        return Some(1.0);
    }
    // ninja/make build progress: "[N/M] action" -> N/M. Surfaced to the bar so
    // long native compiles (e.g. Flash Attention CK's ~2800 translation units)
    // show real per-TU progress instead of a frozen wheel-build spinner.
    if let Some(bracket) = line.find('[') {
        if let Some(close) = line[bracket..].find(']') {
            let inside = &line[bracket + 1..bracket + close];
            if let Some(slash) = inside.find('/') {
                let (n_str, m_str) = inside.split_at(slash);
                if let (Ok(n), Ok(m)) = (
                    n_str.trim().parse::<f32>(),
                    m_str[1..].trim().parse::<f32>(),
                ) {
                    if m > 0.0 && n <= m {
                        return Some((n / m).min(1.0));
                    }
                }
            }
        }
    }
    None
}

fn needs_sudo() -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::geteuid() != 0 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn strip_ansi_codes(s: &str) -> String {
    let mut clean = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if ('@'..='~').contains(&c) || c == 'm' {
                in_escape = false;
            }
        } else {
            clean.push(c);
        }
    }
    clean
}

fn sanitize_output_line(line: &str) -> String {
    let stripped = strip_ansi_codes(line);
    let mut clean = String::with_capacity(stripped.len());
    for c in stripped.chars() {
        if c == '\t' {
            clean.push(' ');
        } else if !c.is_control() {
            clean.push(c);
        }
    }
    clean
}

fn should_suppress_verification_line(line: &str, target_id: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_lowercase();
    if is_exception_or_traceback_line(&lower) {
        return false;
    }

    if lower.contains("amdgpu.ids")
        || lower.contains("tool-lib")
        || lower.contains("cannot load tool")
        || lower.contains("libdrm/amdgpu.ids: no such file or directory")
    {
        return true;
    }

    if (target_id == "rocm" || target_id == "rocm-smi")
        && (lower.starts_with("======")
            || lower.starts_with("====")
            || lower.starts_with("---")
            || lower.contains("the info is not accurate")
            || lower.contains("================================")
            || lower.starts_with('{')
            || lower.starts_with('[')
            || lower.starts_with(']')
            || lower.starts_with('}')
            || lower.starts_with("\"card")
            || lower.starts_with("device ")
            || lower.starts_with("*****"))
    {
        return true;
    }

    false
}

fn is_exception_or_traceback_line(lowercase_line: &str) -> bool {
    lowercase_line.contains("traceback")
        || lowercase_line.contains("exception")
        || lowercase_line.contains("segmentation fault")
        || lowercase_line.contains("fatal:")
        || lowercase_line.contains("error:")
        || lowercase_line.starts_with("error ")
}

/// Run a benchmark component natively via the `benchmark_runners` module.
///
/// Maps component IDs to benchmark names and dispatches through the
/// native Rust benchmark runner instead of spawning bash scripts.
fn run_native_benchmark(component_id: &str, sender: &Sender<InstallerEvent>) -> Result<()> {
    use crate::benchmark_runners;

    let bench_name = match component_id {
        "mlperf-inference" => "all",
        "rocm-benchmarks" => "gpu-capability",
        "gpu-memory-bandwidth" => "memory-bandwidth",
        "rocm-smi-bench" => "gpu-capability",
        "pytorch-performance" => "pytorch",
        "vllm-performance" => "vllm",
        "deepspeed-performance" => "deepspeed",
        "megatron-performance" => "megatron",
        "onnx-performance" => "onnx",
        "rusty-llama-performance" => "llama-cpp",
        "flash-attention-ck-performance" => "flash-attention-ck",
        "all-benchmarks" => "all",
        _ => bail!("Unknown benchmark component: {}", component_id),
    };

    let _ = sender.send(InstallerEvent::Log(
        format!("[native] Running {} benchmark...", component_id),
        false,
    ));

    let _ = sender.send(InstallerEvent::Progress {
        component_id: component_id.to_string(),
        progress: 0.3,
        message: format!("Running {} benchmark", component_id),
    });

    match benchmark_runners::run_benchmark(bench_name) {
        Ok(output) => {
            let _ = sender.send(InstallerEvent::Log(
                format!(
                    "[native] {} benchmark completed: {} ({} ms)",
                    component_id,
                    if output.success { "SUCCESS" } else { "FAILED" },
                    output.execution_time_ms
                ),
                false,
            ));
            // Persist the result as JSON to the benchmark log dir so the
            // Benchmarks page + HTML export pick it up (otherwise the summary
            // shows "No Data available"). Reshapes metrics into the top-level
            // "results" object that parse_benchmark_json expects.
            persist_benchmark_result(bench_name, &output);
            if !output.errors.is_empty() {
                for err in &output.errors {
                    let _ = sender.send(InstallerEvent::Log(
                        format!("[native] {} error: {}", component_id, err),
                        false,
                    ));
                }
            }
            if !output.success {
                let error_details = output.errors.join("; ");
                bail!(
                    "{} benchmark failed: {}",
                    component_id,
                    if error_details.is_empty() {
                        "unknown error".to_string()
                    } else {
                        error_details
                    }
                );
            }
            Ok(())
        }
        Err(err) => {
            let _ = sender.send(InstallerEvent::Log(
                format!("[native] {} benchmark error: {}", component_id, err),
                false,
            ));
            bail!("{} benchmark error: {}", component_id, err)
        }
    }
}

/// Persist a benchmark result as JSON to the benchmark log dir so the Benchmarks
/// page (`load_benchmark_results`) and HTML export pick it up.
///
/// `convert_result` nests the metrics under `results.metrics`, but
/// `parse_benchmark_json` looks for the `prefill_*_tps_gpu*` / `decode_*_tps_gpu*`
/// keys at the `results` level (it does not recurse into `metrics`), so the
/// payload is reshaped to `{"results": <metrics>, ...}`.
///
/// Tries each scanned log dir in turn and writes to the first that succeeds —
/// `~/.rusty-stack/logs` can be root-owned (from a prior sudo run), so we fall
/// back through `$TMPDIR/rusty-stack/logs` (also scanned) rather than silently
/// dropping the result.
fn persist_benchmark_result(bench_name: &str, output: &crate::benchmark_runners::BenchmarkOutput) {
    let metrics = output
        .results
        .get("metrics")
        .cloned()
        .unwrap_or_else(|| output.results.clone());
    let payload = serde_json::json!({
        "name": output.name.clone(),
        "success": output.success,
        "execution_time_ms": output.execution_time_ms as u64,
        "results": metrics,
        "errors": serde_json::to_value(&output.errors).unwrap_or_default(),
    });
    let body = match serde_json::to_string_pretty(&payload) {
        Ok(s) => s,
        Err(_) => return,
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let fname = format!("{bench_name}_benchmarks_{ts}.json");

    let mut candidates: Vec<std::path::PathBuf> =
        crate::benchmark_logs::benchmark_log_directories();
    if let Some(home) = std::env::var("HOME").ok().filter(|s| !s.is_empty()) {
        candidates.push(
            std::path::Path::new(&home)
                .join(".rusty-stack")
                .join("logs"),
        );
    }
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    candidates.push(std::path::Path::new(&tmp).join("rusty-stack").join("logs"));

    for dir in candidates {
        let _ = std::fs::create_dir_all(&dir);
        if std::fs::write(dir.join(&fname), &body).is_ok() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── iGPU filtering: parse_rocminfo_for_discrete_gpus (tenet 3) ─────
    // Target topology: 2 dGPU (gfx1100 RX 7900 XTX, gfx1101 RX 7800 XT) +
    // 1 iGPU (gfx1036 Raphael). The iGPU must NEVER appear; dGPUs never missed.

    // -------------------------------------------------------------------
    // Error handling tests (fix-installation-silent-failures)
    // -------------------------------------------------------------------

    #[test]
    fn test_format_user_friendly_error_permission_denied() {
        let msg = format_user_friendly_error("rocm", "Permission denied: /opt/rocm");
        assert!(
            msg.contains("Permission denied"),
            "Should mention permission issue"
        );
        assert!(
            msg.contains("elevated privileges"),
            "Should suggest running with sudo"
        );
    }

    #[test]
    fn test_native_maintenance_actions_use_native_verdict() {
        let repair = Component {
            id: "rccl-repair".into(),
            name: "Repair RCCL Multi-GPU".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Maintenance,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        };
        let verify = Component {
            id: "verify-basic".into(),
            name: "Verify Installation".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Maintenance,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        };
        let normal = Component {
            id: "pytorch".into(),
            name: "PyTorch with ROCm".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        };

        assert!(is_native_maintenance_action(&repair));
        assert!(!is_native_maintenance_action(&verify));
        assert!(!is_native_maintenance_action(&normal));

        let success = native_maintenance_action_outcome(&repair, true);
        assert!(success.success);
        assert!(success.report_lines[0].contains("reported success"));

        let failure = native_maintenance_action_outcome(&repair, false);
        assert!(!failure.success);
        assert!(failure.report_lines[0].contains("reported failure"));
    }

    #[test]
    fn test_benchmark_patterns_match_native_output_filenames() {
        assert_eq!(
            benchmark_pattern_for_target("rocm-benchmarks"),
            Some("gpu-capability_benchmarks")
        );
        assert_eq!(
            benchmark_pattern_for_target("gpu-memory-bandwidth"),
            Some("memory-bandwidth_benchmarks")
        );
        assert_eq!(
            benchmark_pattern_for_target("all-benchmarks"),
            Some("all_benchmarks")
        );
        assert_eq!(
            benchmark_pattern_for_target("onnx-performance"),
            Some("onnx_benchmarks")
        );
    }

    #[test]
    fn test_single_benchmark_results_render_metrics_and_errors() {
        let lines = benchmark_result_lines_from_json(&serde_json::json!({
            "name": "vllm",
            "success": false,
            "execution_time_ms": 248842,
            "results": {
                "mode": "degraded",
                "reason": "model timed out",
                "throughput_tokens_per_sec": 0.0,
                "visible_devices": "0,1"
            },
            "errors": ["model timed out after 240 seconds"]
        }));
        let joined = lines.join("\n");

        assert!(joined.contains("MODE: degraded"));
        assert!(joined.contains("REASON: model timed out"));
        assert!(joined.contains("THROUGHPUT TOKENS PER SEC: 0"));
        assert!(joined.contains("ERROR: model timed out after 240 seconds"));
    }

    #[test]
    fn test_persist_full_suite_keeps_nested_results() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_home = std::env::var("HOME").ok();
        let old_tmpdir = std::env::var("TMPDIR").ok();
        let old_log_dir = std::env::var("MLSTACK_LOG_DIR").ok();
        let root = std::env::temp_dir().join(format!(
            "rusty-stack-persist-full-suite-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let logs = root.join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::env::set_var("HOME", &root);
        std::env::set_var("TMPDIR", root.join("tmp"));
        std::env::set_var("MLSTACK_LOG_DIR", &logs);

        let output = crate::benchmark_runners::BenchmarkOutput {
            name: "all".into(),
            success: true,
            execution_time_ms: 1,
            results: serde_json::json!({
                "memory_bandwidth": {
                    "success": true,
                    "execution_time_ms": 1,
                    "metrics": {"hbm_peak_gb_s": 123.0}
                }
            }),
            errors: Vec::new(),
        };
        persist_benchmark_result("all", &output);

        let written = std::fs::read_dir(&logs)
            .unwrap()
            .find_map(|entry| {
                let path = entry.ok()?.path();
                (path
                    .file_name()?
                    .to_string_lossy()
                    .starts_with("all_benchmarks_"))
                .then_some(path)
            })
            .expect("all benchmark log should be written");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(written).unwrap()).unwrap();

        if let Some(home) = old_home {
            std::env::set_var("HOME", home);
        } else {
            std::env::remove_var("HOME");
        }
        if let Some(tmpdir) = old_tmpdir {
            std::env::set_var("TMPDIR", tmpdir);
        } else {
            std::env::remove_var("TMPDIR");
        }
        if let Some(log_dir) = old_log_dir {
            std::env::set_var("MLSTACK_LOG_DIR", log_dir);
        } else {
            std::env::remove_var("MLSTACK_LOG_DIR");
        }
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(
            json["results"]["memory_bandwidth"]["metrics"]["hbm_peak_gb_s"],
            123.0
        );
    }

    #[test]
    fn test_format_user_friendly_error_not_found() {
        let msg = format_user_friendly_error("pytorch", "No such file or directory: python3");
        assert!(msg.contains("not found"), "Should mention file not found");
        assert!(
            msg.contains("ROCm") || msg.contains("Python"),
            "Should mention ROCm or Python dependency, got: {}",
            msg
        );
    }

    #[test]
    fn test_format_user_friendly_error_network() {
        // Use a generic component to test network error detection
        let msg = format_user_friendly_error(
            "onnxruntime",
            "network error: Could not resolve host: github.com",
        );
        assert!(
            msg.contains("Network error"),
            "Should mention network issue, got: {}",
            msg
        );
        assert!(
            msg.contains("internet connection"),
            "Should suggest checking internet, got: {}",
            msg
        );
    }

    #[test]
    fn test_format_user_friendly_error_out_of_memory() {
        // Use a generic component to test OOM error detection
        let msg = format_user_friendly_error("onnxruntime", "out of memory during build");
        assert!(
            msg.contains("Out of memory"),
            "Should mention memory issue, got: {}",
            msg
        );
        assert!(
            msg.contains("swap space"),
            "Should suggest increasing swap, got: {}",
            msg
        );
    }

    #[test]
    fn test_format_user_friendly_error_permanent_env_rocminfo() {
        let msg = format_user_friendly_error("permanent-env", "rocminfo command not found");
        assert!(
            msg.contains("rocminfo"),
            "Should mention rocminfo specifically"
        );
        assert!(
            msg.contains("ROCm is properly installed"),
            "Should suggest ROCm installation"
        );
    }

    #[test]
    fn test_format_user_friendly_error_pytorch_specific() {
        // Use a generic error (not "not found") to hit the general pytorch handler
        let msg = format_user_friendly_error("pytorch", "pip install failed: build error");
        assert!(msg.contains("PyTorch"), "Should mention PyTorch by name");
        assert!(
            msg.contains("ROCm is installed first"),
            "Should mention ROCm dependency"
        );
        assert!(
            msg.contains("Python 3.10-3.13"),
            "Should mention Python version, got: {}",
            msg
        );
    }

    #[test]
    fn test_format_user_friendly_error_triton_dependency() {
        let msg = format_user_friendly_error("triton", "git clone failed");
        assert!(
            msg.contains("PyTorch and ROCm"),
            "Should mention required dependencies"
        );
    }

    #[test]
    fn test_format_user_friendly_error_megatron_dependency() {
        let msg = format_user_friendly_error("megatron", "pip install failed");
        assert!(
            msg.contains("PyTorch and MPI4Py"),
            "Should mention Megatron's specific dependencies"
        );
    }

    #[test]
    fn test_format_user_friendly_error_deepspeed_dependency() {
        let msg = format_user_friendly_error("deepspeed", "build failed");
        assert!(
            msg.contains("PyTorch"),
            "Should mention DeepSpeed's dependency on PyTorch"
        );
    }

    #[test]
    fn test_format_user_friendly_error_exit_code_passthrough() {
        let msg = format_user_friendly_error("rocm", "ROCm failed: General error (exit code 1)");
        // Exit code errors should be passed through as-is
        assert!(
            msg.contains("exit code 1"),
            "Should preserve exit code context"
        );
    }

    #[test]
    fn test_format_user_friendly_error_unknown_component() {
        let msg = format_user_friendly_error("some-unknown-component", "something went wrong");
        assert!(
            msg.contains("Installation failed"),
            "Should have generic failure message"
        );
        assert!(
            msg.contains("something went wrong"),
            "Should include original error"
        );
    }

    #[test]
    fn test_format_user_friendly_error_rocm_not_found() {
        let msg = format_user_friendly_error("rocm", "package not found: amdgpu-dkms");
        assert!(msg.contains("ROCm"), "Should mention ROCm specifically");
        assert!(msg.contains("repositories"), "Should mention repositories");
    }

    #[test]
    fn test_execute_native_command_nonexistent_program() {
        // Verify that executing a non-existent command returns a proper error
        // (not a panic) and includes context about the failure
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "/nonexistent/program/that/does/not/exist".to_string(),
            args: vec![],
            env: vec![],
            working_dir: None,
        };
        let result = execute_native_command(&cmd, None, &tx, "test-component");
        assert!(result.is_err(), "Should fail for non-existent program");
        let err_msg = result.unwrap_err().to_string();
        assert!(!err_msg.is_empty(), "Error message should not be empty");

        // Verify events were sent (no silent failure)
        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            !events.is_empty(),
            "Should have sent at least one event before failing"
        );
    }

    #[test]
    fn test_execute_native_command_captures_exit_code() {
        // Verify that a failing command sends error events with context
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "bash".to_string(),
            args: vec!["-c".to_string(), "echo test_output; exit 42".to_string()],
            env: vec![],
            working_dir: None,
        };
        let result = execute_native_command(&cmd, None, &tx, "test-component");
        assert!(
            result.is_err(),
            "Should fail for non-zero exit code, got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("42"),
            "Error should include exit code 42, got: {}",
            err_msg
        );

        // Collect events to verify output was captured
        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        let log_messages: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                InstallerEvent::Log(msg, _) => Some(msg.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            log_messages.iter().any(|m| m.contains("test_output")),
            "Should have captured stdout output before failure"
        );
        assert!(
            log_messages
                .iter()
                .any(|m| m.contains("[ERROR]") || m.contains("exit code")),
            "Should have logged error with exit code"
        );
    }

    #[test]
    fn test_execute_native_command_success_sends_events() {
        // Verify that a successful command sends log events
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "bash".to_string(),
            args: vec![
                "-c".to_string(),
                "echo 'hello world'; echo 'second line'".to_string(),
            ],
            env: vec![],
            working_dir: None,
        };
        let result = execute_native_command(&cmd, None, &tx, "test-component");
        assert!(result.is_ok(), "Should succeed for zero exit code");

        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        let log_messages: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                InstallerEvent::Log(msg, _) => Some(msg.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            log_messages.iter().any(|m| m.contains("hello world")),
            "Should have captured first line of output"
        );
        assert!(
            log_messages.iter().any(|m| m.contains("second line")),
            "Should have captured second line of output"
        );
    }

    // -----------------------------------------------------------------------
    // fix-textgen-git-clone-args: sudo wrapping only when sudo_pw is Some
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_sudo_wrap_when_password_is_none() {
        // Verify that when sudo_pw is None, the command is NOT wrapped with sudo.
        // This is the core fix for the textgen git clone failure where git received
        // empty arguments due to the sudo wrapper inserting "--" separator.
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "echo".to_string(),
            args: vec!["hello".to_string()],
            env: vec![],
            working_dir: None,
        };
        let result = execute_native_command(&cmd, None, &tx, "test-component");
        assert!(result.is_ok(), "Should succeed without sudo: {:?}", result);

        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        let log_messages: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                InstallerEvent::Log(msg, _) => Some(msg.clone()),
                _ => None,
            })
            .collect();

        // The logged command should NOT contain "sudo" since we passed None
        let cmd_log = log_messages.iter().find(|m| m.contains("[native] $"));
        assert!(cmd_log.is_some(), "Should have logged the command");
        let cmd_log = cmd_log.unwrap();
        assert!(
            !cmd_log.contains("sudo"),
            "Command should NOT be wrapped with sudo when sudo_pw is None, got: {}",
            cmd_log
        );
        assert!(
            cmd_log.contains("echo hello"),
            "Command should be 'echo hello', got: {}",
            cmd_log
        );
    }

    #[test]
    fn test_sudo_wrap_when_password_is_some() {
        // Verify that when sudo_pw is Some, the command IS wrapped with sudo.
        // We use a password value but run a command that doesn't actually need sudo
        // (echo) to avoid actual privilege escalation in tests.
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "echo".to_string(),
            args: vec!["hello".to_string()],
            env: vec![],
            working_dir: None,
        };
        // Pass a dummy password — echo doesn't need sudo but we test the wrapping
        let _result = execute_native_command(&cmd, Some("dummy_pw"), &tx, "test-component");
        // This may fail since sudo -S with a dummy password won't work, but we
        // just want to verify the command was constructed with sudo
        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        let log_messages: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                InstallerEvent::Log(msg, _) => Some(msg.clone()),
                _ => None,
            })
            .collect();

        // The logged command SHOULD contain "sudo" since we passed Some
        let has_sudo_log = log_messages.iter().any(|m| m.contains("Running with sudo"));
        assert!(
            has_sudo_log,
            "Should have logged 'Running with sudo' when sudo_pw is Some"
        );
    }

    #[test]
    fn test_git_clone_without_sudo_works() {
        // Simulate the exact textgen git clone scenario:
        // git clone <URL> <dir> with sudo_pw = None
        // This should execute "git --version" successfully (we use --version
        // instead of clone to avoid actually cloning)
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = NativeCommand::Shell {
            program: "git".to_string(),
            args: vec!["--version".to_string()],
            env: vec![],
            working_dir: None,
        };
        let result = execute_native_command(&cmd, None, &tx, "textgen");
        assert!(
            result.is_ok(),
            "git --version should succeed without sudo wrapping: {:?}",
            result
        );

        drop(tx);
        let events: Vec<_> = rx.try_iter().collect();
        let log_messages: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                InstallerEvent::Log(msg, _) => Some(msg.clone()),
                _ => None,
            })
            .collect();

        // Verify git version output was captured
        assert!(
            log_messages.iter().any(|m| m.contains("git version")),
            "Should have captured 'git version' output, got: {:?}",
            log_messages
        );

        // Verify no sudo wrapping in the command log
        let cmd_log = log_messages.iter().find(|m| m.contains("[native] $"));
        assert!(cmd_log.is_some(), "Should have logged the command");
        assert!(
            !cmd_log.unwrap().contains("sudo"),
            "git command should NOT be wrapped with sudo"
        );
    }

    #[test]
    fn test_textgen_git_clone_args_not_empty() {
        // Verify that the textgen git clone command has proper args
        // (not empty strings that would cause git to print usage)
        use crate::installers::components::textgen::{TextgenConfig, TextgenInstaller};
        let inst = TextgenInstaller::new(TextgenConfig {
            install_dir: "/tmp/test-textgen".to_string(),
            python_bin: "python3".to_string(),
            use_uv: false,
            break_system_packages: true,
            dry_run: false,
        });
        let cmd = inst.build_git_clone_command();

        // Verify no empty arguments
        for (i, arg) in cmd.args.iter().enumerate() {
            assert!(
                !arg.is_empty(),
                "Argument {} should not be empty in git clone command",
                i
            );
        }

        // Verify the command structure
        assert_eq!(cmd.program, "git");
        assert_eq!(
            cmd.args.len(),
            3,
            "git clone should have 3 args: clone, URL, dir"
        );
        assert_eq!(cmd.args[0], "clone");
        assert_eq!(
            cmd.args[1],
            "https://github.com/oobabooga/text-generation-webui.git"
        );
        assert_eq!(cmd.args[2], "/tmp/test-textgen");

        // Verify no "--" separator in args (which would be from sudo wrapping)
        assert!(
            !cmd.args.contains(&"--".to_string()),
            "git clone args should not contain '--' separator (from sudo wrapping)"
        );
    }

    // -------------------------------------------------------------------
    // fix-pytorch-strict-validation: verification result tests
    // -------------------------------------------------------------------

    #[test]
    fn test_verification_result_labels() {
        assert_eq!(VerificationResult::Verified.label(), "Verified");
        assert_eq!(VerificationResult::Failed.label(), "Failed");
        assert_eq!(VerificationResult::Missing.label(), "Missing");
    }
}
