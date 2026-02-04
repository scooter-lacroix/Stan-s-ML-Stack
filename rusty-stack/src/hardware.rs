use crate::state::{
    GPUInfo, HardwareState, PreflightCheck, PreflightResult, PreflightStatus, PreflightType,
    SystemInfo,
};
use anyhow::Result;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use sysinfo::{Disks, System};

pub fn detect_hardware() -> Result<HardwareState> {
    let mut state = HardwareState {
        status: "Detecting system info".into(),
        progress: 0.2,
        ..Default::default()
    };

    let mut sys = System::new_all();
    sys.refresh_all();

    state.system.os = System::name().unwrap_or_else(|| "Unknown".into());
    state.system.kernel = System::kernel_version().unwrap_or_else(|| "Unknown".into());
    state.system.distribution = System::long_os_version().unwrap_or_else(|| "Unknown".into());
    if state.system.distribution.trim().is_empty() || state.system.distribution == "Unknown" {
        if let Some(dist) = fallback_distribution() {
            state.system.distribution = dist;
        }
    }
    state.system.cpu_model = sys.global_cpu_info().brand().to_string();
    if state.system.cpu_model.trim().is_empty() {
        if let Some(model) = read_cpu_model_name() {
            state.system.cpu_model = model;
        }
    }
    state.system.memory_gb = (sys.total_memory() as f32) / 1024.0 / 1024.0;
    if state.system.memory_gb < 0.1 {
        if let Some(mem) = fallback_memory_gb() {
            state.system.memory_gb = mem;
        }
    }

    let disks = Disks::new_with_refreshed_list();
    let total_storage: u64 = disks.iter().map(|d| d.total_space()).sum();
    state.system.storage_gb = total_storage as f32 / 1024.0 / 1024.0 / 1024.0;
    let total_available: u64 = disks.iter().map(|d| d.available_space()).sum();
    state.system.storage_available_gb = total_available as f32 / 1024.0 / 1024.0 / 1024.0;
    if state.system.storage_gb < 0.1 || state.system.storage_available_gb < 0.1 {
        if let Some((available, total)) = fallback_disk_space_gb() {
            state.system.storage_available_gb = available;
            state.system.storage_gb = total;
        }
    }

    state.status = "Detecting GPU info".into();
    state.progress = 0.6;
    state.gpu = detect_gpu();

    state.status = "Hardware detection complete".into();
    state.progress = 1.0;

    Ok(state)
}

pub fn run_preflight_checks(
    system: &SystemInfo,
    gpu: &GPUInfo,
    sudo_password: Option<&str>,
) -> PreflightResult {
    let mut result = PreflightResult {
        can_continue: true,
        ..Default::default()
    };

    let checks = vec![
        check_root_privileges(sudo_password),
        check_disk_space(system),
        check_network_connectivity(),
        check_gpu_detection(gpu),
        check_driver_compatibility(gpu),
        check_cpu_compatibility(system),
        check_memory_requirements(system),
        check_package_manager(),
        check_python_availability(),
        check_system_dependencies(),
        check_distribution_compatibility(system),
    ];

    for check in checks {
        match check.status {
            PreflightStatus::Passed => result.passed_count += 1,
            PreflightStatus::Warning => result.warning_count += 1,
            PreflightStatus::Failed => {
                result.failed_count += 1;
                if check.check_type == PreflightType::Critical {
                    result.can_continue = false;
                }
            }
        }
        result.total_score += check.score;
        result.checks.push(check);
    }

    result.passed = result.failed_count == 0;
    result.summary = if result.passed {
        format!("Preflight checks passed (score: {})", result.total_score)
    } else {
        format!(
            "Preflight checks reported issues ({} failed)",
            result.failed_count
        )
    };

    result
}

fn check_root_privileges(sudo_password: Option<&str>) -> PreflightCheck {
    let is_root = unsafe { libc::geteuid() == 0 };

    if is_root {
        return PreflightCheck {
            name: "Root privileges".into(),
            status: PreflightStatus::Passed,
            check_type: PreflightType::Critical,
            message: "Running as root".into(),
            details: "Installer has direct root access".into(),
            score: 10,
        };
    }

    let (status, message, details, score) = if let Some(password) = sudo_password {
        match validate_sudo_password(password) {
            Ok(true) => (
                PreflightStatus::Passed,
                "Sudo privileges validated".into(),
                "Sudo password validated".into(),
                10,
            ),
            Ok(false) => (
                PreflightStatus::Failed,
                "Sudo validation failed".into(),
                "Check the sudo password".into(),
                0,
            ),
            Err(err) => (
                PreflightStatus::Warning,
                "Sudo validation unavailable".into(),
                format!("Sudo validation skipped: {}", err),
                5,
            ),
        }
    } else if check_cached_sudo() {
        (
            PreflightStatus::Passed,
            "Sudo privileges validated".into(),
            "Cached sudo credentials available".into(),
            10,
        )
    } else {
        (
            PreflightStatus::Failed,
            "Sudo privileges unavailable".into(),
            "Enter sudo password on the welcome screen".into(),
            0,
        )
    };

    PreflightCheck {
        name: "Root privileges".into(),
        status,
        check_type: PreflightType::Critical,
        message,
        details,
        score,
    }
}

fn validate_sudo_password(password: &str) -> Result<bool> {
    let mut command = Command::new("sudo");
    command
        .arg("-S")
        .arg("-p")
        .arg("")
        .arg("-k")
        .arg("-v")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(password.as_bytes())?;
        stdin.write_all(b"\n")?;
    }

    let output = child.wait_with_output()?;
    Ok(output.status.success())
}

fn check_cached_sudo() -> bool {
    Command::new("sudo")
        .arg("-n")
        .arg("-v")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn check_disk_space(system: &SystemInfo) -> PreflightCheck {
    let free_gb = system.storage_available_gb;
    let total_gb = system.storage_gb;

    if total_gb < 0.1 {
        return PreflightCheck {
            name: "Disk space".into(),
            status: PreflightStatus::Warning,
            check_type: PreflightType::Critical,
            message: "Disk space unavailable".into(),
            details: "Unable to detect disk capacity".into(),
            score: 2,
        };
    }

    let (status, score, details) = if free_gb >= 100.0 {
        (
            PreflightStatus::Passed,
            10,
            "Sufficient disk space for full ML stack".into(),
        )
    } else if free_gb >= 50.0 {
        (
            PreflightStatus::Warning,
            5,
            "Limited disk space (100GB recommended)".into(),
        )
    } else {
        (
            PreflightStatus::Failed,
            0,
            "Insufficient disk space (minimum 50GB required)".into(),
        )
    };

    PreflightCheck {
        name: "Disk space".into(),
        status,
        check_type: PreflightType::Critical,
        message: format!("{:.1} GB free of {:.1} GB", free_gb, total_gb),
        details,
        score,
    }
}

fn check_network_connectivity() -> PreflightCheck {
    let network_ok = Command::new("bash")
        .arg("-c")
        .arg("ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    PreflightCheck {
        name: "Network connectivity".into(),
        status: if network_ok {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Failed
        },
        check_type: PreflightType::Critical,
        message: if network_ok {
            "Network reachable".into()
        } else {
            "Network connectivity failed".into()
        },
        details: "Connectivity required for package downloads".into(),
        score: if network_ok { 10 } else { 0 },
    }
}

fn check_gpu_detection(gpu: &GPUInfo) -> PreflightCheck {
    let ok = gpu.gpu_count > 0;
    let score = if ok { 15 } else { 0 };
    PreflightCheck {
        name: "GPU detection".into(),
        status: if ok {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Failed
        },
        check_type: PreflightType::Critical,
        message: if ok {
            format!("Detected {} AMD GPU(s)", gpu.gpu_count)
        } else {
            "No AMD GPU detected".into()
        },
        details: format!("Primary GPU: {}", gpu.model),
        score,
    }
}

fn check_driver_compatibility(gpu: &GPUInfo) -> PreflightCheck {
    let ok = !gpu.rocm_version.is_empty();
    PreflightCheck {
        name: "Driver compatibility".into(),
        status: if ok {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Warning
        },
        check_type: PreflightType::Warning,
        message: if ok {
            "ROCm drivers detected".into()
        } else {
            "ROCm drivers not detected".into()
        },
        details: if ok {
            format!("ROCm version {}", gpu.rocm_version)
        } else {
            "Install ROCm drivers for GPU acceleration".into()
        },
        score: if ok { 10 } else { 5 },
    }
}

fn check_cpu_compatibility(system: &SystemInfo) -> PreflightCheck {
    let is_amd = is_amd_cpu(system);
    PreflightCheck {
        name: "CPU compatibility".into(),
        status: if is_amd {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Warning
        },
        check_type: PreflightType::Warning,
        message: if is_amd {
            "AMD CPU detected".into()
        } else {
            "Non-AMD CPU detected".into()
        },
        details: if system.cpu_model.trim().is_empty() {
            "CPU model unavailable".into()
        } else {
            system.cpu_model.clone()
        },
        score: if is_amd { 10 } else { 5 },
    }
}

fn is_amd_cpu(system: &SystemInfo) -> bool {
    let model = system.cpu_model.to_lowercase();
    if model.contains("amd") || model.contains("ryzen") || model.contains("epyc") {
        return true;
    }

    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            let lower = line.to_lowercase();
            if lower.starts_with("vendor_id") && lower.contains("authenticamd") {
                return true;
            }
            if lower.starts_with("model name") && lower.contains("amd") {
                return true;
            }
        }
    }

    false
}

fn read_cpu_model_name() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in cpuinfo.lines() {
        let lower = line.to_lowercase();
        if lower.starts_with("model name") {
            if let Some(model) = line.split(':').nth(1) {
                let trimmed = model.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

fn fallback_memory_gb() -> Option<f32> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f32>() {
                    return Some(kb / 1024.0 / 1024.0);
                }
            }
        }
    }
    None
}

fn fallback_disk_space_gb() -> Option<(f32, f32)> {
    let output = Command::new("df")
        .args(["-k", "--output=avail,size", "/"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout.lines().last()?;
    let parts: Vec<&str> = last_line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let avail_kb = parts[0].parse::<f32>().ok()?;
    let total_kb = parts[1].parse::<f32>().ok()?;
    Some((avail_kb / 1024.0 / 1024.0, total_kb / 1024.0 / 1024.0))
}

fn fallback_distribution() -> Option<String> {
    let contents = fs::read_to_string("/etc/os-release").ok()?;
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("PRETTY_NAME=") {
            let cleaned = value.trim().trim_matches('"');
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }
    None
}

fn detect_gpu_from_sysfs() -> Option<(usize, String)> {
    let mut count = 0usize;
    let mut model = String::new();
    let entries = fs::read_dir("/sys/class/drm").ok()?;
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
            if model.is_empty() {
                let device_path = entry.path().join("device");
                let uevent_path = device_path.join("uevent");
                if let Ok(uevent) = fs::read_to_string(&uevent_path) {
                    for line in uevent.lines() {
                        if let Some(driver) = line.strip_prefix("DRIVER=") {
                            model = format!("AMD GPU ({})", driver);
                            break;
                        }
                    }
                }
                if model.is_empty() {
                    model = "AMD GPU".to_string();
                }
            }
        }
    }
    if count > 0 {
        Some((count, model))
    } else {
        None
    }
}

fn check_memory_requirements(system: &SystemInfo) -> PreflightCheck {
    let mem = system.memory_gb;

    if mem < 0.1 {
        return PreflightCheck {
            name: "Memory requirements".into(),
            status: PreflightStatus::Warning,
            check_type: PreflightType::Warning,
            message: "Memory unavailable".into(),
            details: "Unable to detect system memory".into(),
            score: 2,
        };
    }

    let (status, score, details) = if mem >= 32.0 {
        (
            PreflightStatus::Passed,
            10,
            "Sufficient memory for large models".into(),
        )
    } else if mem >= 16.0 {
        (
            PreflightStatus::Warning,
            7,
            "Moderate memory (consider 32GB+)".into(),
        )
    } else {
        (
            PreflightStatus::Failed,
            0,
            "Insufficient memory (16GB minimum)".into(),
        )
    };

    PreflightCheck {
        name: "Memory requirements".into(),
        status,
        check_type: PreflightType::Warning,
        message: format!("{:.1} GB RAM", mem),
        details,
        score,
    }
}

fn check_package_manager() -> PreflightCheck {
    let managers = ["apt", "dnf", "yum", "zypper", "pacman"];
    let found = managers
        .iter()
        .find(|m| {
            Command::new("bash")
                .arg("-c")
                .arg(format!("command -v {}", m))
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
        .cloned();

    PreflightCheck {
        name: "Package manager".into(),
        status: if found.is_some() {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Failed
        },
        check_type: PreflightType::Critical,
        message: match found {
            Some(manager) => format!("{} detected", manager),
            None => "No supported package manager detected".into(),
        },
        details: "Required for dependency installation".into(),
        score: if found.is_some() { 10 } else { 0 },
    }
}

fn is_version_at_least(version: &str, min_version: &str) -> bool {
    let v_parts: Vec<i32> = version
        .split('.')
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();
    let m_parts: Vec<i32> = min_version
        .split('.')
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();

    for i in 0..std::cmp::max(v_parts.len(), m_parts.len()) {
        let v = *v_parts.get(i).unwrap_or(&0);
        let m = *m_parts.get(i).unwrap_or(&0);
        if v > m {
            return true;
        }
        if v < m {
            return false;
        }
    }
    true
}

fn check_python_availability() -> PreflightCheck {
    let output = Command::new("python3").arg("--version").output();
    let (status, message, details, score) = if let Ok(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let version_str = if stdout.is_empty() { stderr } else { stdout };
        let version = version_str
            .split_whitespace()
            .nth(1)
            .unwrap_or("")
            .to_string();
        if is_version_at_least(&version, "3.9") {
            (
                PreflightStatus::Passed,
                "Python 3.9+ available".into(),
                version,
                10,
            )
        } else {
            (
                PreflightStatus::Warning,
                "Python version too old".into(),
                version,
                5,
            )
        }
    } else {
        (
            PreflightStatus::Failed,
            "Python not available".into(),
            "python3 missing".into(),
            0,
        )
    };

    PreflightCheck {
        name: "Python availability".into(),
        status,
        check_type: PreflightType::Critical,
        message,
        details,
        score,
    }
}

fn check_system_dependencies() -> PreflightCheck {
    let deps = [
        "curl", "wget", "git", "cmake", "unzip", "gcc", "g++", "make",
    ];
    let mut missing = Vec::new();
    for dep in deps {
        let ok = Command::new("bash")
            .arg("-c")
            .arg(format!("command -v {}", dep))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            missing.push(dep);
        }
    }

    PreflightCheck {
        name: "System dependencies".into(),
        status: if missing.is_empty() {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Failed
        },
        check_type: PreflightType::Critical,
        message: if missing.is_empty() {
            "All dependencies available".into()
        } else {
            "Missing dependencies".into()
        },
        details: if missing.is_empty() {
            "Required tools are installed".into()
        } else {
            format!("Missing: {}", missing.join(", "))
        },
        score: if missing.is_empty() { 10 } else { 0 },
    }
}

fn check_distribution_compatibility(system: &SystemInfo) -> PreflightCheck {
    let supported = ["ubuntu", "debian", "linux mint", "pop!_os", "fedora"];
    let dist = system.distribution.to_lowercase();
    let compatible = supported.iter().any(|name| dist.contains(name));

    PreflightCheck {
        name: "Distribution compatibility".into(),
        status: if compatible {
            PreflightStatus::Passed
        } else {
            PreflightStatus::Warning
        },
        check_type: PreflightType::Info,
        message: if compatible {
            format!("Compatible distribution: {}", system.distribution)
        } else {
            format!("Unofficial distribution: {}", system.distribution)
        },
        details: "Supported distributions offer best compatibility".into(),
        score: if compatible { 5 } else { 2 },
    }
}

fn detect_gpu() -> GPUInfo {
    let mut info = GPUInfo::default();
    let rocminfo_cmd = if Path::new("/opt/rocm/bin/rocminfo").exists() {
        "/opt/rocm/bin/rocminfo"
    } else {
        "rocminfo"
    };

    if let Ok(output) = Command::new(rocminfo_cmd).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpu_count = 0usize;
            for line in stdout.lines() {
                if line.contains("Name:") {
                    let name = line.replace("Name:", "").trim().to_string();
                    if !name.is_empty() && !name.to_lowercase().contains("cpu") {
                        gpu_count += 1;
                        if info.model.is_empty() {
                            info.model = name;
                        }
                    }
                }
                if line.contains("gfx") && info.architecture.is_empty() {
                    // Extract only the gfx architecture identifier (e.g., gfx1100, gfx906)
                    let trimmed = line.trim();
                    if let Some(gfx_start) = trimmed.find("gfx") {
                        let after_gfx = &trimmed[gfx_start + 3..];
                        let gfx_num: String = after_gfx
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect();
                        if !gfx_num.is_empty() {
                            info.architecture = format!("gfx{}", gfx_num);
                        }
                    }
                }
            }
            if gpu_count > 0 {
                info.gpu_count = gpu_count;
            }
        }
    }

    if info.rocm_version.is_empty() {
        if let Ok(output) = Command::new("bash")
            .arg("-c")
            .arg("cat /opt/rocm/.info/version 2>/dev/null")
            .output()
        {
            if output.status.success() {
                info.rocm_version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }
    }

    if info.model.is_empty() {
        let lspci_cmd = if Path::new("/usr/bin/lspci").exists() {
            "/usr/bin/lspci"
        } else {
            "lspci"
        };
        if let Ok(output) = Command::new(lspci_cmd).output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.to_lowercase().contains("amd") && line.to_lowercase().contains("vga") {
                    if info.model.is_empty() {
                        info.model = line.trim().to_string();
                    }
                    info.gpu_count += 1;
                }
            }
        }
    }

    if info.gpu_count == 0 {
        if let Some((count, model)) = detect_gpu_from_sysfs() {
            info.gpu_count = count;
            if info.model.is_empty() {
                info.model = model;
            }
        }
    }

    if info.gpu_count == 0 && !info.model.is_empty() {
        info.gpu_count = 1;
    }

    let rocm_smi_cmd = if Path::new("/opt/rocm/bin/rocm-smi").exists() {
        "/opt/rocm/bin/rocm-smi"
    } else {
        "rocm-smi"
    };

    if let Ok(output) = Command::new(rocm_smi_cmd)
        .arg("--showmeminfo")
        .arg("vram")
        .arg("--csv")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    if let Ok(value) = parts[1].trim().parse::<f32>() {
                        info.memory_gb = value / 1024.0;
                        break;
                    }
                }
            }
        }
    }

    if let Ok(output) = Command::new(rocm_smi_cmd)
        .arg("--showtemp")
        .arg("--csv")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    if let Ok(value) = parts[1].trim().parse::<f32>() {
                        info.temperature_c = Some(value);
                        break;
                    }
                }
            }
        }
    }

    if let Ok(output) = Command::new(rocm_smi_cmd)
        .arg("--showpower")
        .arg("--csv")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    if let Ok(value) = parts[1].trim().parse::<f32>() {
                        info.power_watts = Some(value);
                        break;
                    }
                }
            }
        }
    }

    info
}
