use crate::config::InstallerConfig;
use crate::component_status::{
    component_verification_commands, is_component_installed_by_id, modules_available,
    python_interpreters, python_search_paths, verification_commands, VerificationCommand,
};
use crate::state::{Category, Component};
use anyhow::{bail, Context, Result};
use chrono::Local;
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
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
    let total = components.len() as f32;
    let mut index = 0usize;
    let mut python_candidates = python_interpreters();
    let user_home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let input_rx = Arc::new(Mutex::new(input_rx));

    match ensure_mlstack_env(&user_home) {
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

    if needs_sudo() {
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

    let mut overall_success = true;
    let python_bin = resolve_python_bin();
    if !python_candidates.contains(&python_bin) {
        python_candidates.insert(0, python_bin.clone());
    }
    let _ = sender.send(InstallerEvent::Log(
        format!("Selected python interpreter: {}", python_bin),
        false,
    ));

    for component in components {
        index += 1;
        let _ = sender.send(InstallerEvent::ComponentStart {
            component_id: component.id.clone(),
            name: component.name.clone(),
        });

        if component.category == Category::Verification {
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

        let mut install_success = true;
        let mut error_msg = String::new();

        if let Err(err) = run_script(
            &component,
            &script_path,
            sudo_password.clone(),
            batch_mode,
            &sender,
            Arc::clone(&input_rx),
        ) {
            overall_success = false;
            install_success = false;
            error_msg = err.to_string();
        }

        let _ = sender.send(InstallerEvent::Progress {
            component_id: component.id.clone(),
            progress: 0.8,
            message: format!("Verifying {}", component.name),
        });

        let verification_outcome =
            run_component_verification(&component, &python_candidates, &sender, &user_home);
        let _ = sender.send(InstallerEvent::VerificationReport {
            component_id: component.id.clone(),
            lines: verification_outcome.report_lines.clone(),
        });

        if !verification_outcome.success {
            overall_success = false;
        }

        let _ = sender.send(InstallerEvent::ComponentComplete {
            component_id: component.id.clone(),
            success: install_success && verification_outcome.success,
            message: if !install_success {
                error_msg
            } else if verification_outcome.success {
                format!("{} completed", component.name)
            } else {
                format!("{} completed with verification errors", component.name)
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
    }

    if overall_success && config.star_repos {
        let _ = star_repo(&config, &sender);
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

fn ensure_mlstack_env(user_home: &str) -> Result<EnvUpdate> {
    let env_path = PathBuf::from(user_home).join(".mlstack_env");
    if env_path.exists() {
        let contents = fs::read_to_string(&env_path).context("Failed to read .mlstack_env")?;
        let rocm_version = detect_rocm_version();
        let gpu_arch = detect_gpu_arch();
        let gpu_list = detect_gpu_list();
        let primary_gpu = first_gpu_index(&gpu_list);
        let rocm_home = "/opt/rocm";
        let rocm_lib = "/opt/rocm/lib";
        let defaults = [
            ("ROCM_VERSION", rocm_version.as_str()),
            ("ROCM_CHANNEL", "latest"),
            ("GPU_ARCH", gpu_arch.as_str()),
            ("ROCM_HOME", rocm_home),
            ("ROCM_PATH", rocm_home),
            ("HIP_PATH", rocm_home),
            ("HIP_VISIBLE_DEVICES", gpu_list.as_str()),
            ("CUDA_VISIBLE_DEVICES", gpu_list.as_str()),
            ("PYTORCH_ROCM_DEVICE", primary_gpu.as_str()),
            ("PYTHONPATH", rocm_lib),
        ];
        let (updated, changed) = sanitize_mlstack_env(&contents, &defaults, gpu_list.as_str(), rocm_home);
        if changed {
            fs::write(&env_path, updated).context("Failed to update .mlstack_env")?;
            return Ok(EnvUpdate::Updated);
        }
        return Ok(EnvUpdate::Unchanged);
    }

    if let Some(parent) = env_path.parent() {
        fs::create_dir_all(parent).context("Failed to create home directory")?;
    }

    let rocm_version = detect_rocm_version();
    let gpu_arch = detect_gpu_arch();
    let gpu_list = detect_gpu_list();
    let primary_gpu = first_gpu_index(&gpu_list);
    let rocm_home = "/opt/rocm";
    let rocm_lib = "/opt/rocm/lib";

    let content = format!(
        "# ML Stack Environment File (generated by Rusty-Stack)\n\
export ROCM_VERSION={}\n\
export ROCM_CHANNEL=latest\n\
export GPU_ARCH={}\n\
export ROCM_HOME={}\n\
export ROCM_PATH={}\n\
export HIP_PATH={}\n\
export HIP_VISIBLE_DEVICES={}\n\
export CUDA_VISIBLE_DEVICES={}\n\
export PYTORCH_ROCM_DEVICE={}\n\
export PYTHONPATH={}:$PYTHONPATH\n\
export TORCH_CUDA_ARCH_LIST=\"7.0;8.0;9.0\"\n\
export PYTORCH_CUDA_ALLOC_CONF=\"max_split_size_mb:512\"\n\
export PATH=\"/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:{}/bin:{}/hip/bin:$PATH\"\n\
export LD_LIBRARY_PATH=\"{}/lib:{}/hip/lib:{}/opencl/lib:${{LD_LIBRARY_PATH:-}}\"\n",
        rocm_version,
        gpu_arch,
        rocm_home,
        rocm_home,
        rocm_home,
        gpu_list,
        gpu_list,
        primary_gpu,
        rocm_lib,
        rocm_home,
        rocm_home,
        rocm_home,
        rocm_home,
        rocm_home
    );

    fs::write(&env_path, content).context("Failed to create .mlstack_env")?;
    Ok(EnvUpdate::Created)
}

fn sanitize_mlstack_env(
    contents: &str,
    defaults: &[(&str, &str)],
    gpu_list: &str,
    rocm_home: &str,
) -> (String, bool) {
    let mut changed = false;
    let mut lines = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for line in contents.lines() {
        let trimmed = line.trim_start();
        if let Some(key) = extract_env_key(trimmed) {
            seen.insert(key.to_string());
        }
        let (line, changed_arch) = fix_env_assignment(line, "TORCH_CUDA_ARCH_LIST");
        let (line, changed_alloc) = fix_env_assignment(&line, "PYTORCH_CUDA_ALLOC_CONF");
        let (line, changed_hip) = normalize_env_value(&line, "HIP_PATH", "/opt/rocm");
        let (line, changed_rocm) = normalize_env_value(&line, "ROCM_PATH", "/opt/rocm");
        let (line, changed_hip_vis) =
            normalize_visible_devices(&line, "HIP_VISIBLE_DEVICES", gpu_list);
        let (line, changed_cuda_vis) =
            normalize_visible_devices(&line, "CUDA_VISIBLE_DEVICES", gpu_list);
        
        // Enforce safe PATH and LD_LIBRARY_PATH
        let (line, changed_path) = if line.starts_with("export PATH=") {
            let desired = format!("export PATH=\"/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:{}/bin:{}/hip/bin:$PATH\"", rocm_home, rocm_home);
            if line.trim() != desired { (desired, true) } else { (line, false) }
        } else { (line, false) };
        
        let (line, changed_ld) = if line.starts_with("export LD_LIBRARY_PATH=") {
            let desired = format!("export LD_LIBRARY_PATH=\"{}/lib:{}/hip/lib:{}/opencl/lib:${{LD_LIBRARY_PATH:-}}\"", rocm_home, rocm_home, rocm_home);
            if line.trim() != desired { (desired, true) } else { (line, false) }
        } else { (line, false) };

        changed |=
            changed_arch || changed_alloc || changed_hip || changed_rocm || changed_hip_vis || changed_cuda_vis || changed_path || changed_ld;
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
    let line = line.strip_prefix("export ")?;
    line.split('=').next().map(str::trim)
}

fn fix_env_assignment(line: &str, key: &str) -> (String, bool) {
    let marker = format!("{}=", key);
    let Some(idx) = line.find(&marker) else {
        return (line.to_string(), false);
    };

    let before = &line[..idx + marker.len()];
    let after = &line[idx + marker.len()..];

    if after.starts_with('"') {
        return (line.to_string(), false);
    }

    (format!("{}\"{}\"", before, after), true)
}

fn normalize_env_value(line: &str, key: &str, desired: &str) -> (String, bool) {
    let marker = format!("export {}=", key);
    if let Some(rest) = line.trim_start().strip_prefix(&marker) {
        let current = rest.trim().trim_matches('"');
        if current != desired {
            return (format!("export {}={}", key, desired), true);
        }
    }
    (line.to_string(), false)
}

fn normalize_visible_devices(line: &str, key: &str, desired: &str) -> (String, bool) {
    if !desired.contains(',') {
        return (line.to_string(), false);
    }
    let marker = format!("export {}=", key);
    if let Some(rest) = line.trim_start().strip_prefix(&marker) {
        let current = rest.trim().trim_matches('"');
        if current == "0" {
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

fn collect_env_info(user_home: &str) -> Vec<(String, String)> {
    let mut values: HashMap<String, String> = HashMap::new();
    let env_path = PathBuf::from(user_home).join(".mlstack_env");

    if let Ok(contents) = fs::read_to_string(&env_path) {
        for line in contents.lines() {
            if let Some((key, value)) = parse_env_export(line) {
                values.entry(key).or_insert(value);
            }
        }
    }

    let keys = [
        "ROCM_VERSION",
        "ROCM_CHANNEL",
        "GPU_ARCH",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ROCM_DEVICE",
        "ROCM_HOME",
        "ROCM_PATH",
        "HIP_PATH",
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

fn parse_env_export(line: &str) -> Option<(String, String)> {
    let export_idx = line.find("export ")?;
    let rest = &line[export_idx + 7..];
    let mut parts = rest.splitn(2, '=');
    let key = parts.next()?.trim().to_string();
    let value = parts.next()?.trim().trim_matches('"').to_string();
    Some((key, value))
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

fn run_verification_command(
    step: &VerificationCommand,
    sender: &Sender<InstallerEvent>,
) -> Result<(bool, Vec<String>)> {
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
    pythonpath = pythonpath.split(':')
        .filter(|part| !part.starts_with("/tmp/"))
        .collect::<Vec<_>>()
        .join(":");
    
    if !pythonpath.is_empty() {
        command.env("PYTHONPATH", pythonpath);
    }

    if step.target_id == "onnx" {
        if let Some(lib) = detect_onnx_rocm_provider(&step.program) {
            command.env("ORT_ROCM_EP_PROVIDER_PATH", lib);
        }
        let mut ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if !ld_path.split(':').any(|entry| entry == "/opt/rocm/lib") {
            if ld_path.is_empty() {
                ld_path = "/opt/rocm/lib".into();
            } else {
                ld_path = format!("/opt/rocm/lib:{}", ld_path);
            }
        }
        command.env("LD_LIBRARY_PATH", ld_path);
    }

    let user_home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let user_name = std::env::var("USER").unwrap_or_else(|_| "user".into());

    command
        .env("HOME", &user_home)
        .env("USER", &user_name)
        .env("LOGNAME", &user_name)
        .env("MLSTACK_USER_HOME", &user_home)
        .env("PYTHONWARNINGS", "ignore")
        .env("PYTHONUNBUFFERED", "1")
        .env("ROCM_HOME", "/opt/rocm")
        .env("HIP_PATH", "/opt/rocm")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = command
        .output()
        .with_context(|| format!("Failed to run verification for {}", step.label))?;

    let mut captured_output = Vec::new();

    let stdout = String::from_utf8_lossy(&output.stdout);
    for raw in stdout.replace('\r', "\n").lines() {
        let clean_line = sanitize_output_line(raw);
        if !clean_line.trim().is_empty() {
            let line = format!("{}: {}", step.label, clean_line);
            let _ = sender.send(InstallerEvent::Log(line.clone(), false));
            captured_output.push(clean_line);
        }
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    for raw in stderr.replace('\r', "\n").lines() {
        let clean_line = sanitize_output_line(raw);
        if !clean_line.trim().is_empty() {
            let line = format!("{}: {}", step.label, clean_line);
            let _ = sender.send(InstallerEvent::Log(line.clone(), false));
            captured_output.push(clean_line);
        }
    }

    Ok((output.status.success(), captured_output))
}

fn run_script(
    component: &Component,
    script_path: &str,
    sudo_password: Option<String>,
    batch_mode: bool,
    sender: &Sender<InstallerEvent>,
    input_rx: Arc<Mutex<Receiver<String>>>,
) -> Result<()> {
    let user_home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let user_name = std::env::var("USER").unwrap_or_else(|_| "user".into());
    let preserve_env = "HOME,USER,LOGNAME,PATH,MLSTACK_USER_HOME,MLSTACK_SKIP_TORCH_INSTALL,MLSTACK_PYTHON_BIN,PIP_BREAK_SYSTEM_PACKAGES,PIP_ROOT_USER_ACTION,UV_PIP_BREAK_SYSTEM_PACKAGES,UV_SYSTEM_PYTHON,PYTHONPATH,ROCM_HOME,ROCM_PATH,ROCM_VERSION,ROCM_CHANNEL,GPU_ARCH,HIP_PATH,HIP_ROOT_DIR,ROCM_ROOT,HIPCC_BIN_DIR,VLLM_TARGET_DEVICE,VLLM_USE_ROCM,USE_ROCM,VLLM_VERSION,UV_PYTHON,DS_ACCELERATOR,FORCE,PYTORCH_REINSTALL";

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

    let python_bin = resolve_python_bin();

    let mut command = if needs_sudo() {
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
    pythonpath = pythonpath.split(':')
        .filter(|part| !part.starts_with("/tmp/"))
        .collect::<Vec<_>>()
        .join(":");

    let rocm_version = detect_rocm_version();
    let gpu_arch = detect_gpu_arch();

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
        .env("HIP_PATH", "/opt/rocm")
        .env("HIP_ROOT_DIR", "/opt/rocm")
        .env("ROCM_ROOT", "/opt/rocm")
        .env("HIPCC_BIN_DIR", "/opt/rocm/bin")
        .env("MLSTACK_BATCH_MODE", if batch_mode { "1" } else { "0" })
        .env("PIP_BREAK_SYSTEM_PACKAGES", "1")
        .env("PIP_ROOT_USER_ACTION", "ignore")
        .env("UV_PIP_BREAK_SYSTEM_PACKAGES", "1")
        .env("UV_SYSTEM_PYTHON", "1")
        .env(
            "MLSTACK_SKIP_TORCH_INSTALL",
            if component.id == "pytorch" { "0" } else { "1" },
        )
        .env("RUSTY_STACK", "true");

    let mut child = command
        .spawn()
        .context("Failed to spawn installer script")?;

    let mut child_stdin = child.stdin.take().context("Failed to open stdin")?;
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
            let rx = input_rx.lock().unwrap();
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
    let stdout_stream = child.stdout.take().unwrap();
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
    let stderr_stream = child.stderr.take().unwrap();
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
                            let _ = sender_stderr
                                .send(InstallerEvent::Log(clean_line, is_transient));
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

    let status = child.wait().context("Installer script failed")?;
    stop_input.store(true, Ordering::Relaxed);
    let _ = input_thread.join();
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        let _ = sender.send(InstallerEvent::Log(
            format!("{} exited with code {} at {}", component.name, code, Local::now().format("%H:%M:%S")),
            false,
        ));
        bail!("{} failed with exit code {}", component.name, code);
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
            let _ = sender.send(InstallerEvent::Log("Starring ML Stack repository on GitHub...".into(), false));
            let star_status = Command::new("gh")
                .arg("repo")
                .arg("star")
                .arg(repo_url)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            
            if let Ok(s) = star_status {
                if s.success() {
                    let _ = sender.send(InstallerEvent::Log("Thank you for starring the repo! ✦".into(), false));
                }
            }
        }
    }
    
    Ok(())
}

fn component_name(comp: &Component) -> String {
    comp.name.clone()
}

fn detect_gpu_arch() -> String {
    if let Ok(output) = Command::new("rocminfo").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for token in stdout.split_whitespace() {
            let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric());
            if cleaned.starts_with("gfx") {
                return cleaned.to_string();
            }
        }
    }

    "gfx000".to_string()
}

fn detect_gpu_list() -> String {
    if let Ok(output) = Command::new("rocminfo").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let count = stdout
            .lines()
            .filter(|line| line.contains("Device Type") && line.contains("GPU"))
            .count();
        if count > 0 {
            return (0..count)
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(",");
        }
    }

    if let Ok(output) = Command::new("lspci").output() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
        let count = stdout
            .lines()
            .filter(|line| line.contains("vga") && line.contains("amd"))
            .count();
        if count > 0 {
            return (0..count)
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(",");
        }
    }

    if let Some(count) = detect_gpu_count_sysfs() {
        if count > 0 {
            return (0..count)
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join(",");
        }
    }

    "0".to_string()
}

fn detect_gpu_count_sysfs() -> Option<usize> {
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    let mut count = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains("render") {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        let vendor = match fs::read_to_string(&vendor_path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if vendor.trim().eq_ignore_ascii_case("0x1002") {
            count += 1;
        }
    }
    if count > 0 {
        Some(count)
    } else {
        None
    }
}

fn first_gpu_index(list: &str) -> String {
    list.split(',').next().unwrap_or("0").to_string()
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
    // 1. Check for the new global 3.12 priority link
    let global_312 = "/usr/local/bin/python3";
    if Path::new(global_312).exists() {
        // Double check it's actually 3.12
        if let Ok(output) = Command::new(global_312).arg("--version").output() {
            let version = String::from_utf8_lossy(&output.stdout);
            if version.contains("3.12") {
                return global_312.to_string();
            }
        }
    }

    if let Ok(value) = std::env::var("MLSTACK_PYTHON_BIN") {
        if !value.trim().is_empty() && python_has_rocm_torch(&value) {
            return value;
        }
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        PathBuf::from("/usr/local/bin/python3"),
        PathBuf::from("python3.12"),
        Path::new(&home)
            .join("rocm_venv")
            .join("bin")
            .join("python"),
        PathBuf::from("/usr/bin/python3"),
        PathBuf::from("python3"),
        PathBuf::from("python"),
    ];

    for candidate in candidates {
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

    if Path::new("/usr/bin/python3").exists() {
        return "/usr/bin/python3".to_string();
    }
    "python3".to_string()
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
    None
}

fn needs_sudo() -> bool {
    unsafe { libc::geteuid() != 0 }
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
