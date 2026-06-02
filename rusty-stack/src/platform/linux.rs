//! Linux-specific platform detection: GPU, ROCm, and hardware introspection.
//!
//! Provides GPU detection with a three-tier fallback chain (rocminfo → lspci → sysfs),
//! GPU architecture correction from marketing names, ROCm version detection from
//! `/opt/rocm/.info/version`, and ROCm path search with priority ordering.
//!
//! # Fallback Chain
//!
//! 1. **rocminfo** — parses agent entries for AMD GPU devices
//! 2. **lspci** — scans PCI bus for AMD VGA devices
//! 3. **sysfs** (`/sys/class/drm`) — reads vendor IDs from DRM card entries
//!
//! # ROCm Path Priority
//!
//! 1. `ROCm_PATH` environment variable
//! 2. `/opt/rocm` (standard location)
//! 3. `/opt/rocm-*` versioned directories (sorted newest-first)
//! 4. `/usr/lib/rocm`, `/usr/local/rocm` (Arch AUR paths)
//! 5. `/usr/lib/rocm-*` versioned paths (Arch AUR)
//! 6. Derived from `which rocminfo`

use crate::core::types::GPUInfo;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// Public API
// ===========================================================================

/// Detect AMD GPUs using the three-tier fallback chain (rocminfo → lspci → sysfs).
///
/// Returns a [`GPUInfo`] with detected GPU details. If no AMD GPU is found,
/// returns a default `GPUInfo` with `gpu_count = 0`.
///
/// # Hardware Test
///
/// On systems with real AMD GPUs, run with `cargo test -- --ignored` to execute
/// hardware-dependent tests.
pub fn detect_gpu() -> GPUInfo {
    let mut info = GPUInfo::default();

    // Detect ROCm path first
    let rocm_path = detect_rocm_path();

    // Determine rocminfo path
    let rocminfo_cmd = resolve_rocminfo_path(&rocm_path);

    // Tier 1: rocminfo
    let rocminfo_ok = detect_gpu_from_rocminfo(&rocminfo_cmd, &mut info);

    // Detect ROCm version
    if info.rocm_version.is_empty() {
        info.rocm_version = detect_rocm_version_from_file(&rocm_path);
    }

    // Tier 2: lspci (only if rocminfo didn't find a model)
    if !rocminfo_ok || info.model.is_empty() {
        detect_gpu_from_lspci(&mut info);
    }

    // Tier 3: sysfs (only if still no GPU found)
    if info.gpu_count == 0 {
        detect_gpu_from_sysfs(&mut info);
    }

    // Ensure at least 1 GPU if a model was found but count is 0
    if info.gpu_count == 0 && !info.model.is_empty() {
        info.gpu_count = 1;
    }

    // Enrich with runtime metrics from rocm-smi
    if info.gpu_count > 0 {
        let rocm_smi_cmd = resolve_rocm_smi_path(&rocm_path);
        enrich_gpu_metrics(&rocm_smi_cmd, &mut info);
    }

    info
}

/// Map GPU marketing name to correct gfx architecture.
///
/// This corrects rocminfo bugs where RDNA3 cards report `gfx1030` instead of
/// `gfx1100`/`gfx1101`. When the marketing name matches a known GPU, the correct
/// architecture is returned. For unknown names, the `fallback_gfx` value is used.
///
/// # Examples
///
/// ```
/// use rusty_stack::platform::get_correct_gfx_from_marketing_name;
///
/// assert_eq!(get_correct_gfx_from_marketing_name("AMD Radeon RX 7900 XTX", "1030"), "gfx1100");
/// assert_eq!(get_correct_gfx_from_marketing_name("AMD Radeon RX 7800 XT", "1030"), "gfx1101");
/// assert_eq!(get_correct_gfx_from_marketing_name("Unknown GPU", "999"), "gfx999");
/// ```
pub fn get_correct_gfx_from_marketing_name(marketing_name: &str, fallback_gfx: &str) -> String {
    let name_lower = marketing_name.to_lowercase();

    // RDNA 3 (Navi 3x) — gfx1100/gfx1101/gfx1102
    // These are commonly misreported as gfx1030 by buggy ROCm versions
    if name_lower.contains("7900 xtx") || name_lower.contains("7900xtx") {
        return "gfx1100".to_string();
    }
    if name_lower.contains("7900 gre") || name_lower.contains("7900gre") {
        return "gfx1100".to_string();
    }
    if name_lower.contains("7900 xt") || name_lower.contains("7900xt") {
        return "gfx1100".to_string();
    }
    if name_lower.contains("7800 xt") || name_lower.contains("7800xt") {
        return "gfx1101".to_string();
    }
    if name_lower.contains("7800 gre") || name_lower.contains("7800gre") {
        return "gfx1101".to_string();
    }
    if name_lower.contains("7700 xt") || name_lower.contains("7700xt") {
        return "gfx1101".to_string();
    }
    if name_lower.contains("7600 xt")
        || name_lower.contains("7600xt")
        || name_lower.contains("7600")
    {
        return "gfx1102".to_string();
    }

    // RDNA 4 (Navi 4x) — gfx1200/gfx1201
    if name_lower.contains("9070 xt")
        || name_lower.contains("9070xt")
        || name_lower.contains("9070 gre")
        || name_lower.contains("9070gre")
    {
        return "gfx1200".to_string();
    }
    if name_lower.contains("9060") {
        return "gfx1201".to_string();
    }

    // RDNA 2 (Navi 2x) — gfx1030/gfx1031/gfx1032/gfx1034
    if name_lower.contains("6950 xt")
        || name_lower.contains("6950xt")
        || name_lower.contains("6900 xt")
        || name_lower.contains("6900xt")
    {
        return "gfx1030".to_string();
    }
    if name_lower.contains("6800 xt")
        || name_lower.contains("6800xt")
        || name_lower.contains("6800")
        || name_lower.contains("6900")
    {
        return "gfx1030".to_string();
    }
    if name_lower.contains("6700 xt")
        || name_lower.contains("6700xt")
        || name_lower.contains("6750 xt")
        || name_lower.contains("6750xt")
    {
        return "gfx1031".to_string();
    }
    if name_lower.contains("6600 xt")
        || name_lower.contains("6600xt")
        || name_lower.contains("6600")
        || name_lower.contains("6650")
    {
        return "gfx1032".to_string();
    }
    if name_lower.contains("6500 xt") || name_lower.contains("6500xt") {
        return "gfx1034".to_string();
    }

    // CDNA (MI accelerators) — trust rocminfo for these
    if name_lower.contains("instinct")
        || name_lower.contains("mi60")
        || name_lower.contains("mi100")
        || name_lower.contains("mi200")
        || name_lower.contains("mi250")
        || name_lower.contains("mi300")
    {
        return format!("gfx{}", fallback_gfx);
    }

    // Default: use the fallback value
    format!("gfx{}", fallback_gfx)
}

/// Detect ROCm installation path by searching common locations in priority order.
///
/// # Priority Order
///
/// 1. `ROCm_PATH` environment variable
/// 2. `/opt/rocm` (standard location)
/// 3. `/opt/rocm-*` versioned directories (sorted newest-first)
/// 4. `/usr/lib/rocm`, `/usr/local/rocm` (Arch AUR paths)
/// 5. `/usr/lib/rocm-*` versioned paths (Arch AUR)
/// 6. Derived from `which rocminfo`
///
/// Returns `None` if no valid ROCm installation is found.
pub fn detect_rocm_path() -> Option<PathBuf> {
    let mut search_paths: Vec<PathBuf> = Vec::new();

    // Priority 1: Environment variable override
    if let Ok(env_path) = std::env::var("ROCm_PATH") {
        search_paths.push(PathBuf::from(env_path));
    }

    // Priority 2: Standard location
    search_paths.push(PathBuf::from("/opt/rocm"));

    // Check explicit paths first
    for path in &search_paths {
        if validate_rocm_path(path) {
            return Some(path.clone());
        }
    }

    // Priority 3: Versioned ROCm directories in /opt (sorted newest-first)
    if let Some(path) = find_versioned_rocm_in_dir("/opt") {
        return Some(path);
    }

    // Priority 4: Arch AUR package locations
    let arch_paths = ["/usr/lib/rocm", "/usr/local/rocm"];
    for path_str in &arch_paths {
        let path = PathBuf::from(path_str);
        if validate_rocm_path(&path) {
            return Some(path);
        }
    }

    // Priority 5: Versioned paths in /usr/lib (Arch AUR)
    if let Some(path) = find_versioned_rocm_in_dir("/usr/lib") {
        return Some(path);
    }

    // Priority 6: Try to derive from rocminfo in PATH
    if let Ok(output) = Command::new("which").arg("rocminfo").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                if let Some(parent) = Path::new(&path).parent() {
                    if let Some(rocm_root) = parent.parent() {
                        if validate_rocm_path(rocm_root) {
                            return Some(rocm_root.to_path_buf());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Detect ROCm version using multiple fallback methods.
///
/// Tries in order:
/// 1. `$ROCm_PATH/.info/version`
/// 2. `$ROCm_PATH/version`
/// 3. `rocminfo --version`
/// 4. `rocm-smi --version`
/// 5. Version extracted from ROCm path
///
/// Returns an empty string if no method succeeds.
pub fn get_rocm_version() -> String {
    let rocm_path = detect_rocm_path();

    // Method 1: $ROCm_PATH/.info/version
    if let Some(ref path) = rocm_path {
        let version_file = path.join(".info/version");
        if let Ok(version) = read_version_file(&version_file) {
            return version;
        }
    }

    // Method 2: $ROCm_PATH/version
    if let Some(ref path) = rocm_path {
        let version_file = path.join("version");
        if let Ok(version) = read_version_file(&version_file) {
            return version;
        }
    }

    // Method 3: rocminfo --version
    if let Ok(output) = Command::new("rocminfo").arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = parse_version_from_output(&stdout) {
                return version;
            }
        }
    }

    // Method 4: rocm-smi --version
    if let Ok(output) = Command::new("rocm-smi").arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = parse_version_from_output(&stdout) {
                return version;
            }
        }
    }

    // Method 5: Version from path
    if let Some(ref path) = rocm_path {
        if let Some(version) = extract_version_from_path(path) {
            return version;
        }
    }

    String::new()
}

// ===========================================================================
// Private Implementation
// ===========================================================================

/// Resolve the rocminfo binary path from the ROCm installation.
fn resolve_rocminfo_path(rocm_path: &Option<PathBuf>) -> String {
    if let Some(ref path) = rocm_path {
        let rocminfo_path = path.join("bin/rocminfo");
        if rocminfo_path.exists() {
            return rocminfo_path.to_string_lossy().to_string();
        }
    }
    if Path::new("/opt/rocm/bin/rocminfo").exists() {
        "/opt/rocm/bin/rocminfo".to_string()
    } else {
        "rocminfo".to_string()
    }
}

/// Resolve the rocm-smi binary path from the ROCm installation.
fn resolve_rocm_smi_path(rocm_path: &Option<PathBuf>) -> String {
    if let Some(ref path) = rocm_path {
        let rocm_smi_path = path.join("bin/rocm-smi");
        if rocm_smi_path.exists() {
            return rocm_smi_path.to_string_lossy().to_string();
        }
    }
    if Path::new("/opt/rocm/bin/rocm-smi").exists() {
        "/opt/rocm/bin/rocm-smi".to_string()
    } else {
        "rocm-smi".to_string()
    }
}

/// Tier 1: Detect GPU from rocminfo output.
///
/// Returns `true` if rocminfo ran successfully and found at least one GPU.
fn detect_gpu_from_rocminfo(rocminfo_cmd: &str, info: &mut GPUInfo) -> bool {
    let output = match Command::new(rocminfo_cmd).output() {
        Ok(o) if o.status.success() => o,
        _ => return false,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpu_count = 0usize;
    let mut current_marketing_name = String::new();
    let mut current_device_type = String::new();

    for line in stdout.lines() {
        let trimmed = line.trim();

        // Track marketing name (within Agent section)
        if trimmed.starts_with("Marketing Name:") {
            current_marketing_name = trimmed
                .strip_prefix("Marketing Name:")
                .unwrap_or("")
                .trim()
                .to_string();
        }

        // Track device type to distinguish GPU from CPU
        if trimmed.starts_with("Device Type:") {
            current_device_type = trimmed
                .strip_prefix("Device Type:")
                .unwrap_or("")
                .trim()
                .to_string();
        }

        // Count GPU agents (Name: lines that aren't CPU)
        if line.contains("Name:") {
            let name = line.replace("Name:", "").trim().to_string();
            if !name.is_empty() && !name.to_lowercase().contains("cpu") {
                gpu_count += 1;
                if info.model.is_empty() {
                    info.model = name;
                }
            }
        }

        // Detect GPU architecture from Name field
        if trimmed.starts_with("Name:") && trimmed.contains("gfx") {
            if let Some(name_value) = trimmed.strip_prefix("Name:") {
                let name_value = name_value.trim();
                if let Some(after_gfx) = name_value.strip_prefix("gfx") {
                    let gfx_num: String = after_gfx
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if !gfx_num.is_empty() && gfx_num.len() >= 3 {
                        // Only set architecture for dGPUs (not iGPUs, not CPUs)
                        let is_igpu = current_marketing_name.to_lowercase().contains("ryzen")
                            || current_marketing_name.to_lowercase().contains("apu")
                            || current_marketing_name.to_lowercase().contains("integrated");
                        if !is_igpu && current_device_type == "GPU" && info.architecture.is_empty()
                        {
                            info.architecture = get_correct_gfx_from_marketing_name(
                                &current_marketing_name,
                                &gfx_num,
                            );
                        }
                    }
                }
            }
        }
    }

    if gpu_count > 0 {
        info.gpu_count = gpu_count;
        true
    } else {
        false
    }
}

/// Tier 2: Detect GPU from lspci output.
///
/// Scans for AMD VGA devices on the PCI bus.
fn detect_gpu_from_lspci(info: &mut GPUInfo) {
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

/// Tier 3: Detect GPU from sysfs (`/sys/class/drm`).
///
/// Reads vendor IDs from DRM card entries to find AMD GPU devices.
fn detect_gpu_from_sysfs(info: &mut GPUInfo) {
    let mut count = 0usize;
    let mut model = String::new();

    let entries = match fs::read_dir("/sys/class/drm") {
        Ok(e) => e,
        Err(_) => return,
    };

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

        // AMD vendor ID: 0x1002
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
        info.gpu_count = count;
        if info.model.is_empty() {
            info.model = model;
        }
    }
}

/// Detect ROCm version from the `.info/version` file.
fn detect_rocm_version_from_file(rocm_path: &Option<PathBuf>) -> String {
    let version_path = if let Some(ref path) = rocm_path {
        path.join(".info/version")
    } else {
        PathBuf::from("/opt/rocm/.info/version")
    };

    read_version_file(&version_path).unwrap_or_default()
}

/// Read and trim a version file.
fn read_version_file(path: &Path) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(path)?;
    Ok(content.trim().to_string())
}

/// Parse a version number from command output (e.g., "ROCm version 7.2.1").
fn parse_version_from_output(output: &str) -> Option<String> {
    // Look for version-like patterns: X.Y.Z
    for part in output.split_whitespace() {
        let parts: Vec<&str> = part.split('.').collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.parse::<u32>().is_ok()) {
            return Some(part.to_string());
        }
    }
    None
}

/// Extract version from a path like `/opt/rocm-7.2.1`.
fn extract_version_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    if let Some(version) = name.strip_prefix("rocm-") {
        let version = version.trim().to_string();
        if !version.is_empty() {
            return Some(version);
        }
    }
    None
}

/// Validate that a path is a valid ROCm installation directory.
fn validate_rocm_path(path: &Path) -> bool {
    if !path.exists() || !path.is_dir() {
        return false;
    }
    // Should have at least bin or lib directory, or contain ROCm tools directly
    path.join("bin").exists()
        || path.join("lib").exists()
        || path.join("rocminfo").exists()
        || path.join("rocm-smi").exists()
}

/// Find versioned ROCm directories in a given parent directory, sorted newest-first.
fn find_versioned_rocm_in_dir(parent_dir: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(parent_dir).ok()?;
    let mut versioned_paths: Vec<(Vec<i32>, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("rocm-") {
                let version = name.trim_start_matches("rocm-").to_string();
                let parts: Vec<i32> = version.split('.').filter_map(|s| s.parse().ok()).collect();
                if !parts.is_empty() {
                    Some((parts, e.path()))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Sort by version descending (newest first)
    versioned_paths.sort_by(|a, b| {
        for i in 0..std::cmp::max(a.0.len(), b.0.len()) {
            let a_val = *a.0.get(i).unwrap_or(&0);
            let b_val = *b.0.get(i).unwrap_or(&0);
            if a_val != b_val {
                return b_val.cmp(&a_val);
            }
        }
        std::cmp::Ordering::Equal
    });

    for (_version_parts, path) in versioned_paths {
        if validate_rocm_path(&path) {
            return Some(path);
        }
    }

    None
}

/// Enrich GPU info with runtime metrics from rocm-smi (VRAM, temperature, power).
fn enrich_gpu_metrics(rocm_smi_cmd: &str, info: &mut GPUInfo) {
    // VRAM usage
    if let Ok(output) = Command::new(rocm_smi_cmd)
        .args(["--showmeminfo", "vram", "--csv"])
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

    // Temperature
    if let Ok(output) = Command::new(rocm_smi_cmd)
        .args(["--showtemp", "--csv"])
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

    // Power draw
    if let Ok(output) = Command::new(rocm_smi_cmd)
        .args(["--showpower", "--csv"])
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
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-PLAT-002: GPU architecture correction from marketing name
    // -----------------------------------------------------------------------

    #[test]
    fn test_gfx_correction_rdna3_7900_xtx() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7900 XTX", "1030"),
            "gfx1100"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7900_xtx_no_space() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7900XTX", "1030"),
            "gfx1100"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7900_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7900 XT", "1030"),
            "gfx1100"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7900_gre() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7900 GRE", "1030"),
            "gfx1100"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7800_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7800 XT", "1030"),
            "gfx1101"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7700_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7700 XT", "1030"),
            "gfx1101"
        );
    }

    #[test]
    fn test_gfx_correction_rdna3_7600() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 7600", "1030"),
            "gfx1102"
        );
    }

    #[test]
    fn test_gfx_correction_rdna4_9070_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 9070 XT", "1200"),
            "gfx1200"
        );
    }

    #[test]
    fn test_gfx_correction_rdna4_9060() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 9060", "1201"),
            "gfx1201"
        );
    }

    #[test]
    fn test_gfx_correction_rdna2_6900_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 6900 XT", "1030"),
            "gfx1030"
        );
    }

    #[test]
    fn test_gfx_correction_rdna2_6800() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 6800", "1030"),
            "gfx1030"
        );
    }

    #[test]
    fn test_gfx_correction_rdna2_6700_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 6700 XT", "1030"),
            "gfx1031"
        );
    }

    #[test]
    fn test_gfx_correction_rdna2_6600() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 6600", "1030"),
            "gfx1032"
        );
    }

    #[test]
    fn test_gfx_correction_rdna2_6500_xt() {
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Radeon RX 6500 XT", "1030"),
            "gfx1034"
        );
    }

    #[test]
    fn test_gfx_correction_cdna_instinct() {
        // CDNA accelerators trust rocminfo
        assert_eq!(
            get_correct_gfx_from_marketing_name("AMD Instinct MI250X", "90a0"),
            "gfx90a0"
        );
    }

    #[test]
    fn test_gfx_correction_unknown_passthrough() {
        // Unknown GPU names pass through the fallback gfx
        assert_eq!(
            get_correct_gfx_from_marketing_name("Unknown GPU Model", "999"),
            "gfx999"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-003: GPU detection fallback chain
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_gpu_from_sysfs_no_drm_dir() {
        // When /sys/class/drm doesn't exist, no GPU detected via sysfs
        let mut info = GPUInfo::default();
        detect_gpu_from_sysfs(&mut info);
        // On this system, DRM may or may not have AMD entries,
        // but the function should not panic
        // Test passes if we get here without panicking
    }

    #[test]
    fn test_detect_gpu_from_lspci_no_panic() {
        // lspci may or may not find AMD GPUs, but should never panic
        let mut info = GPUInfo::default();
        detect_gpu_from_lspci(&mut info);
        // No assertion on count — just ensuring no panic
    }

    #[test]
    fn test_detect_gpu_from_rocminfo_bad_command() {
        // Non-existent rocminfo command should return false cleanly
        let mut info = GPUInfo::default();
        let result = detect_gpu_from_rocminfo("/nonexistent/rocminfo", &mut info);
        assert!(!result);
        assert_eq!(info.gpu_count, 0);
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-004: ROCm version detection from .info/version
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_rocm_version_from_file_with_real_install() {
        // On this system, ROCm should be installed at /opt/rocm
        let rocm_path = Some(PathBuf::from("/opt/rocm"));
        let version = detect_rocm_version_from_file(&rocm_path);
        if Path::new("/opt/rocm/.info/version").exists() {
            assert!(
                !version.is_empty(),
                "ROCm version should be non-empty when file exists"
            );
            // Version should look like X.Y.Z
            let parts: Vec<&str> = version.split('.').collect();
            assert!(
                parts.len() >= 2,
                "Version should have at least major.minor: got '{}'",
                version
            );
        }
    }

    #[test]
    fn test_detect_rocm_version_from_file_missing() {
        let rocm_path = Some(PathBuf::from("/nonexistent/rocm"));
        let version = detect_rocm_version_from_file(&rocm_path);
        assert!(
            version.is_empty(),
            "Version should be empty for nonexistent path"
        );
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-005: ROCm path search priority
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_rocm_path_finds_standard_location() {
        // On this system, /opt/rocm should exist
        let path = detect_rocm_path();
        if Path::new("/opt/rocm").is_dir() {
            assert!(path.is_some(), "Should detect ROCm at /opt/rocm");
            let p = path.unwrap();
            assert!(
                p.as_os_str() == "/opt/rocm" || p.to_string_lossy().contains("rocm"),
                "Path should contain 'rocm': got {:?}",
                p
            );
        }
    }

    #[test]
    fn test_detect_rocm_path_env_override() {
        // Set ROCm_PATH env var — should take priority
        // We test this by setting to a known valid path
        if Path::new("/opt/rocm").is_dir() {
            std::env::set_var("ROCm_PATH", "/opt/rocm");
            let path = detect_rocm_path();
            assert_eq!(path, Some(PathBuf::from("/opt/rocm")));
            std::env::remove_var("ROCm_PATH");
        }
    }

    #[test]
    fn test_validate_rocm_path_rejects_nonexistent() {
        assert!(!validate_rocm_path(Path::new("/nonexistent/path")));
    }

    #[test]
    fn test_validate_rocm_path_accepts_standard() {
        if Path::new("/opt/rocm/bin").is_dir() {
            assert!(validate_rocm_path(Path::new("/opt/rocm")));
        }
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-006: ROCm version detection multi-method fallback
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_rocm_version_returns_nonempty_on_installed_system() {
        if Path::new("/opt/rocm/.info/version").exists() {
            let version = get_rocm_version();
            assert!(
                !version.is_empty(),
                "Should detect ROCm version on installed system"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_version_from_output() {
        assert_eq!(
            parse_version_from_output("ROCm version 7.2.1"),
            Some("7.2.1".to_string())
        );
        assert_eq!(
            parse_version_from_output("rocm-smi version 4.0.0"),
            Some("4.0.0".to_string())
        );
        assert_eq!(parse_version_from_output("no version here"), None);
    }

    #[test]
    fn test_extract_version_from_path() {
        assert_eq!(
            extract_version_from_path(Path::new("/opt/rocm-7.2.1")),
            Some("7.2.1".to_string())
        );
        assert_eq!(extract_version_from_path(Path::new("/opt/rocm")), None);
    }

    #[test]
    fn test_read_version_file() {
        let dir = tempfile::tempdir().unwrap();
        let version_file = dir.path().join("version");
        fs::write(&version_file, "7.2.2\n").unwrap();
        let version = read_version_file(&version_file).unwrap();
        assert_eq!(version, "7.2.2");
    }

    #[test]
    fn test_read_version_file_trims_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let version_file = dir.path().join("version");
        fs::write(&version_file, "  7.2.2  \n").unwrap();
        let version = read_version_file(&version_file).unwrap();
        assert_eq!(version, "7.2.2");
    }

    #[test]
    fn test_find_versioned_rocm_no_dir() {
        let result = find_versioned_rocm_in_dir("/nonexistent/dir");
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Hardware tests (require real AMD GPU + ROCm)
    // Run with: cargo test -- --ignored
    // -----------------------------------------------------------------------

    /// VAL-PLAT-001: GPU detection identifies AMD GPUs via rocminfo
    #[test]
    #[ignore]
    fn test_detect_gpu_real_hardware() {
        let info = detect_gpu();
        assert!(
            info.gpu_count >= 1,
            "Expected at least 1 AMD GPU, got gpu_count = {}",
            info.gpu_count
        );
        assert!(
            !info.model.is_empty(),
            "GPU model should be non-empty on real hardware"
        );
    }

    /// VAL-PLAT-001: GPU detection on real hardware has correct architecture
    #[test]
    #[ignore]
    fn test_detect_gpu_real_hardware_architecture() {
        let info = detect_gpu();
        if info.gpu_count > 0 {
            assert!(
                !info.architecture.is_empty(),
                "Architecture should be detected on real AMD GPU"
            );
            assert!(
                info.architecture.starts_with("gfx"),
                "Architecture should start with 'gfx': got '{}'",
                info.architecture
            );
        }
    }

    /// VAL-PLAT-004: ROCm version detected on real system
    #[test]
    #[ignore]
    fn test_detect_rocm_version_real_system() {
        let info = detect_gpu();
        assert!(
            !info.rocm_version.is_empty(),
            "ROCm version should be detected on installed system"
        );
    }

    /// VAL-PLAT-005: ROCm path found on real system
    #[test]
    #[ignore]
    fn test_detect_rocm_path_real_system() {
        let path = detect_rocm_path();
        assert!(
            path.is_some(),
            "ROCm path should be detected on installed system"
        );
        let p = path.unwrap();
        assert!(
            p.exists() && p.is_dir(),
            "ROCm path should be an existing directory"
        );
    }

    /// VAL-PLAT-006: Multi-method ROCm version on real system
    #[test]
    #[ignore]
    fn test_get_rocm_version_real_system() {
        let version = get_rocm_version();
        assert!(
            !version.is_empty(),
            "Should detect ROCm version via multi-method fallback"
        );
        // Version should be parseable as X.Y.Z
        let parts: Vec<&str> = version.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Version should have at least major.minor: got '{}'",
            version
        );
    }

    /// GPU detection returns GPUInfo with runtime metrics on real hardware
    #[test]
    #[ignore]
    fn test_detect_gpu_real_hardware_metrics() {
        let info = detect_gpu();
        if info.gpu_count > 0 {
            // Memory should be positive for real GPUs
            assert!(
                info.memory_gb > 0.0,
                "GPU memory should be positive: got {} GB",
                info.memory_gb
            );
        }
    }

    /// Verify GPU detection matches rocm-smi output
    #[test]
    #[ignore]
    fn test_detect_gpu_matches_rocm_smi() {
        let info = detect_gpu();

        // Cross-check with rocm-smi
        let output = Command::new("/opt/rocm/bin/rocm-smi")
            .args(["--showproductinfo", "--csv"])
            .output()
            .expect("rocm-smi should be available");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // If rocm-smi reports a card, our detection should also find one
            if stdout.contains("card") {
                assert!(
                    info.gpu_count >= 1,
                    "GPU count should be >= 1 if rocm-smi reports cards"
                );
            }
        }
    }
}
