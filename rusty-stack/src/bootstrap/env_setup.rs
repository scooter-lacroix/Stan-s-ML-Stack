//! enhanced_setup_environment.sh equivalent — GPU filtering, system deps, env file, ROCm optimization.
//!
//! Ports `scripts/enhanced_setup_environment.sh` (1411 lines) to native Rust.
//! This is the most complex bootstrap script, handling:
//! - GPU detection and filtering (discrete vs integrated)
//! - System dependency checking (kernel, ROCm, packages, Python)
//! - Environment file creation (`~/.mlstack_env`)
//! - ROCm optimization settings (HSA_OVERRIDE_GFX_VERSION, MIOpen, channel selection)
//!
//! # Validation Assertions
//!
//! - **VAL-VBA-014**: GPU filtering detects AMD GPUs via ROCm, sets HIP_VISIBLE_DEVICES
//! - **VAL-VBA-015**: System dependency checking covers kernel, ROCm, packages, Python
//! - **VAL-VBA-016**: Environment file creation produces shell-sourceable ~/.mlstack_env
//! - **VAL-VBA-017**: ROCm optimization: HSA_OVERRIDE_GFX_VERSION, ROCM_MATH_LIBS, MIOPEN_DEBUG, channel selection

use crate::installers::common::utils::{
    command_exists, print_error, print_step, print_success, print_warning,
};
use std::fmt;

// ===========================================================================
// Types
// ===========================================================================

/// GPU architecture information detected from the system.
#[derive(Debug, Clone)]
pub struct GpuArchInfo {
    /// The GPU architecture string (e.g., "gfx1100").
    pub gpu_arch: String,
    /// The HSA override GFX version (e.g., "11.0.0").
    pub hsa_override_gfx_version: String,
}

impl Default for GpuArchInfo {
    fn default() -> Self {
        Self {
            gpu_arch: "gfx1100".to_string(),
            hsa_override_gfx_version: "11.0.0".to_string(),
        }
    }
}

impl GpuArchInfo {
    /// Create GpuArchInfo from a GPU architecture string.
    ///
    /// Maps the architecture to the correct HSA override version.
    pub fn from_arch(arch: &str) -> Self {
        let hsa_version = match arch {
            "gfx1100" => "11.0.0",
            "gfx1101" => "11.0.1",
            "gfx1102" => "11.0.2",
            "gfx1200" => "12.0.0",
            "gfx1201" => "12.0.1",
            "gfx1030" => "10.3.0",
            "gfx1031" => "10.3.1",
            "gfx1032" => "10.3.2",
            _ => "11.0.0", // default fallback
        };
        Self {
            gpu_arch: arch.to_string(),
            hsa_override_gfx_version: hsa_version.to_string(),
        }
    }
}

/// Result of GPU detection and filtering.
#[derive(Debug, Clone)]
pub struct GpuFilterResult {
    /// Indices of discrete GPUs (integrated GPUs filtered out).
    pub discrete_gpu_indices: Vec<u32>,
    /// Comma-separated string for HIP_VISIBLE_DEVICES.
    pub hip_visible_devices: String,
    /// Number of detected GPUs.
    pub gpu_count: usize,
}

/// A system dependency to check.
#[derive(Debug, Clone)]
pub struct SystemDependency {
    /// The package name (generic, will be mapped per distro).
    pub package: &'static str,
    /// Human-readable description.
    pub description: &'static str,
}

impl SystemDependency {
    /// Returns the list of required system packages.
    ///
    /// Mirrors the `required_packages` array from enhanced_setup_environment.sh.
    pub fn required_packages() -> Vec<&'static str> {
        vec![
            "build-essential",
            "cmake",
            "git",
            "python3-dev",
            "python3-pip",
            "libnuma-dev",
            "pciutils",
            "mesa-utils",
            "clinfo",
        ]
    }
}

impl fmt::Display for SystemDependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.package, self.description)
    }
}

/// Result of checking a single system dependency.
#[derive(Debug, Clone)]
pub struct DependencyCheckResult {
    /// The generic package name.
    pub package: String,
    /// Whether the package is installed.
    pub installed: bool,
    /// The distro-mapped package name.
    pub mapped_name: String,
}

/// Result of environment file creation.
#[derive(Debug, Clone)]
pub struct EnvFileResult {
    /// Whether the operation was successful.
    pub success: bool,
    /// Path to the created environment file.
    pub path: Option<String>,
    /// Any errors encountered.
    pub errors: Vec<String>,
}

impl EnvFileResult {
    /// Create a successful result.
    pub fn success(path: String) -> Self {
        Self {
            success: true,
            path: Some(path),
            errors: Vec::new(),
        }
    }

    /// Create a failure result.
    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            path: None,
            errors: vec![error],
        }
    }
}

// ===========================================================================
// GPU Detection and Filtering (VAL-VBA-014)
// ===========================================================================

/// Detect discrete AMD GPUs, filtering out integrated GPUs.
///
/// Uses rocminfo to enumerate GPUs and filters out known integrated GPUs
/// (Raphael, Ryzen Graphics, etc.). Falls back to lspci if rocminfo is
/// unavailable.
///
/// # Validation
///
/// - **VAL-VBA-014**: GPU filtering detects AMD GPUs via ROCm, sets HIP_VISIBLE_DEVICES
pub fn detect_discrete_gpus() -> GpuFilterResult {
    let mut discrete_indices: Vec<u32> = Vec::new();

    // Try rocminfo first
    let rocminfo_path = crate::installers::common::utils::resolve_rocminfo_path();
    if command_exists(&rocminfo_path) || command_exists("rocminfo") {
        discrete_indices = detect_gpus_from_rocminfo();
    }

    // Fallback to lspci
    if discrete_indices.is_empty() {
        discrete_indices = detect_gpus_from_lspci();
    }

    // Fallback to render nodes
    if discrete_indices.is_empty() {
        discrete_indices = detect_gpus_from_render_nodes();
    }

    // Default to GPU 0 if nothing detected
    if discrete_indices.is_empty() {
        discrete_indices.push(0);
    }

    let gpu_count = discrete_indices.len();
    let hip_visible_devices = discrete_indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    GpuFilterResult {
        discrete_gpu_indices: discrete_indices,
        hip_visible_devices,
        gpu_count,
    }
}

/// Detect GPUs from rocminfo output, filtering out integrated GPUs.
fn detect_gpus_from_rocminfo() -> Vec<u32> {
    let rocminfo_path = crate::installers::common::utils::resolve_rocminfo_path();
    let output = match std::process::Command::new(&rocminfo_path).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return Vec::new(),
    };

    let mut discrete_indices: Vec<u32> = Vec::new();
    let mut current_gpu_id: Option<u32> = None;
    let mut is_igpu = false;
    let mut gpu_agent_counter = 0u32; // Fallback counter for systems without GPU ID lines

    for line in output.lines() {
        let trimmed = line.trim();

        // Detect new Agent sections (lines like "Agent N")
        if trimmed.starts_with("Agent") && trimmed.contains(char::is_numeric) {
            // Don't increment here — only for GPU agents
        }

        if trimmed.starts_with("GPU ID:") {
            // Save previous GPU if it was discrete
            if let Some(id) = current_gpu_id {
                if !is_igpu && !discrete_indices.contains(&id) {
                    discrete_indices.push(id);
                }
            }

            // Parse new GPU ID
            current_gpu_id = trimmed
                .strip_prefix("GPU ID:")
                .and_then(|s| s.trim().parse::<u32>().ok());
            is_igpu = false;
        } else if trimmed.starts_with("Marketing Name:") {
            let marketing_name = trimmed
                .strip_prefix("Marketing Name:")
                .map(|s| s.trim())
                .unwrap_or("");

            // Check if this is an integrated GPU
            if is_integrated_gpu_name(marketing_name) {
                is_igpu = true;
            }

            // If no GPU ID lines exist, track by agent counter
            if current_gpu_id.is_none() {
                // Only count GPU agents (those with Device Type: GPU)
                // We'll finalize on Device Type line
            }
        } else if trimmed.starts_with("Device Type:") {
            let device_type = trimmed
                .strip_prefix("Device Type:")
                .map(|s| s.trim())
                .unwrap_or("");

            // CPU devices are also not discrete GPUs
            if device_type.contains("CPU") {
                is_igpu = true;
            }

            // Finalize this GPU
            if let Some(id) = current_gpu_id {
                if !is_igpu && !discrete_indices.contains(&id) {
                    discrete_indices.push(id);
                }
            } else if device_type.contains("GPU") {
                // No GPU ID lines in this rocminfo output — use counter
                if !is_igpu && !discrete_indices.contains(&gpu_agent_counter) {
                    discrete_indices.push(gpu_agent_counter);
                }
                gpu_agent_counter += 1;
            }
        }
    }

    // Handle last GPU
    if let Some(id) = current_gpu_id {
        if !is_igpu && !discrete_indices.contains(&id) {
            discrete_indices.push(id);
        }
    }

    discrete_indices.sort();
    discrete_indices.dedup();
    discrete_indices
}

/// Detect GPUs from lspci output.
fn detect_gpus_from_lspci() -> Vec<u32> {
    if !command_exists("lspci") {
        return Vec::new();
    }

    let output = match std::process::Command::new("lspci").output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return Vec::new(),
    };

    let mut discrete_count = 0u32;
    for line in output.lines() {
        let lower = line.to_lowercase();
        if !((lower.contains("amd")
            || lower.contains("radeon")
            || lower.contains("advanced micro devices"))
            && (lower.contains("vga") || lower.contains("3d") || lower.contains("display")))
        {
            continue;
        }
        // Filter out integrated GPUs
        if is_integrated_gpu_name(line) {
            continue;
        }
        discrete_count += 1;
    }

    if discrete_count > 0 {
        (0..discrete_count).collect()
    } else {
        Vec::new()
    }
}

/// Detect GPUs from /dev/dri/render* nodes.
/// Filters out iGPU render nodes by checking the associated device description.
fn detect_gpus_from_render_nodes() -> Vec<u32> {
    let render_entries: Vec<_> = std::fs::read_dir("/dev/dri")
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("render"))
                .collect()
        })
        .unwrap_or_default();

    if render_entries.is_empty() {
        return Vec::new();
    }

    // Try to filter iGPUs via sysfs device description
    let discrete_count = render_entries
        .iter()
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            // Extract render node number
            let node_num: String = name
                .trim_start_matches("render")
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if node_num.is_empty() {
                return true; // Can't determine, include it
            }

            // Check sysfs for the associated device description
            let sysfs_path = format!("/sys/class/drm/card{}-render/device/uevent", node_num);
            if let Ok(uevent) = std::fs::read_to_string(&sysfs_path) {
                // Look for PCI_ID to identify the device
                for line in uevent.lines() {
                    if line.starts_with("PCI_ID=") {
                        // Could check specific IDs here if needed
                        return true;
                    }
                }
            }

            // Alternative: check via /sys/class/drm/card<N>/device/device
            let card_sysfs = format!("/sys/class/drm/card{}/device/device", node_num);
            if let Ok(pci_id) = std::fs::read_to_string(&card_sysfs) {
                let pci_id = pci_id.trim();
                // 0x164e = Raphael iGPU (7800X3D)
                if pci_id == "0x164e" {
                    return false; // Skip iGPU
                }
            }

            true // Include by default
        })
        .count();

    if discrete_count > 0 {
        (0..discrete_count as u32).collect()
    } else if !render_entries.is_empty() {
        // Fallback: use all render nodes (safer than returning empty)
        (0..render_entries.len() as u32).collect()
    } else {
        Vec::new()
    }
}

/// Check if a GPU name indicates an integrated GPU.
///
/// Delegates to the canonical classifier `crate::gpu::is_integrated_gpu_name`
/// (Stage 1: single source of truth for dGPU/iGPU classification).
pub fn is_integrated_gpu_name(name: &str) -> bool {
    crate::gpu::is_integrated_gpu_name(name)
}

// ===========================================================================
// GPU Architecture Detection (VAL-VBA-017)
// ===========================================================================

/// Detect the correct GPU architecture from marketing name.
///
/// This corrects rocminfo bugs where RDNA3 cards report gfx1030 instead of
/// gfx1100. Maps marketing names to correct architectures.
///
/// # Validation
///
/// - **VAL-VBA-017**: ROCm optimization — correct GPU arch detection
pub fn detect_correct_gpu_arch(marketing_name: Option<&str>) -> GpuArchInfo {
    let name = match marketing_name {
        Some(n) => n,
        None => return GpuArchInfo::default(),
    };

    // Skip integrated GPUs
    if is_integrated_gpu_name(name) {
        return GpuArchInfo::default();
    }

    // Delegate to canonical name→gfx lookup; default to gfx1100 for unknown
    let arch = crate::gpu::gfx_from_marketing_name(name).unwrap_or("gfx1100");

    GpuArchInfo::from_arch(arch)
}

// ===========================================================================
// System Dependency Checking (VAL-VBA-015)
// ===========================================================================

/// Check a single system dependency.
///
/// Returns whether the package is installed and the mapped name.
/// Uses distro detection to map generic package names to distro-specific names.
///
/// # Validation
///
/// - **VAL-VBA-015**: System dependency checking covers kernel, ROCm, packages, Python
pub fn check_system_dependency(package: &str) -> DependencyCheckResult {
    use crate::installers::common::distro::DistroFacade;
    use crate::installers::common::package_mappings::map_package_name;

    let distro = DistroFacade::detect();
    let pkg_mgr = distro.package_manager();
    let mapped = map_package_name(package, pkg_mgr);
    let installed = check_package_installed(&mapped);

    DependencyCheckResult {
        package: package.to_string(),
        installed,
        mapped_name: mapped,
    }
}

/// Check all required system dependencies.
///
/// Returns results for each dependency, indicating which are missing.
pub fn check_all_dependencies() -> Vec<DependencyCheckResult> {
    SystemDependency::required_packages()
        .into_iter()
        .map(check_system_dependency)
        .collect()
}

/// Check if a package is installed (best-effort, distro-agnostic).
fn check_package_installed(package: &str) -> bool {
    // Try dpkg first (Debian/Ubuntu)
    if command_exists("dpkg-query") {
        let output = std::process::Command::new("dpkg-query")
            .args(["-W", "-f=${Status}", package])
            .output();
        if let Ok(o) = output {
            let stdout = String::from_utf8_lossy(&o.stdout);
            if stdout.contains("install ok installed") {
                return true;
            }
        }
    }

    // Try rpm (Fedora/RHEL/SUSE)
    if command_exists("rpm") {
        let output = std::process::Command::new("rpm")
            .args(["-q", package])
            .output();
        if let Ok(o) = output {
            if o.status.success() {
                return true;
            }
        }
    }

    // Try pacman (Arch)
    if command_exists("pacman") {
        let output = std::process::Command::new("pacman")
            .args(["-Q", package])
            .output();
        if let Ok(o) = output {
            if o.status.success() {
                return true;
            }
        }
    }

    false
}

// ===========================================================================
// Environment File Creation (VAL-VBA-016)
// ===========================================================================

/// Create the `~/.mlstack_env` file with all required variables.
///
/// Generates a shell-sourceable file with GPU settings, ROCm paths,
/// performance optimizations, and PyTorch/MPI settings.
///
/// # Validation
///
/// - **VAL-VBA-016**: Environment file creation produces shell-sourceable ~/.mlstack_env
pub fn create_env_file(
    hip_visible_devices: &str,
    rocm_path: &str,
    rocm_version: &str,
    rocm_channel: &str,
    gpu_arch: &str,
    hsa_override_gfx_version: &str,
    python_bin: &str,
) -> EnvFileResult {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let env_path = format!("{home}/.mlstack_env");
    let fish_env_path = format!("{home}/.config/fish/conf.d/mlstack_env.fish");

    let content = generate_env_file_content(
        hip_visible_devices,
        rocm_path,
        rocm_version,
        rocm_channel,
        gpu_arch,
        hsa_override_gfx_version,
        python_bin,
    );

    // Write the bash env file
    if let Err(e) = std::fs::write(&env_path, &content) {
        let msg = format!("Failed to write environment file: {e}");
        print_error(&msg);
        return EnvFileResult::failure(msg);
    }

    // Write the fish-compatible env file
    let fish_content = generate_fish_env_file_content(
        hip_visible_devices,
        rocm_path,
        rocm_version,
        rocm_channel,
        gpu_arch,
        hsa_override_gfx_version,
        python_bin,
    );
    // Ensure fish conf.d directory exists
    if let Some(parent) = std::path::Path::new(&fish_env_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&fish_env_path, &fish_content) {
        // Non-fatal — fish env is best-effort
        print_warning(&format!("Failed to write fish env file: {e}"));
    } else {
        print_success(&format!("Fish env file created: {fish_env_path}"));
    }

    print_success(&format!("Environment file created: {env_path}"));
    EnvFileResult::success(env_path)
}

/// Generate the environment file content.
///
/// This produces a shell-sourceable file with all required environment
/// variables. The content uses `if [ -z ... ]` guards to avoid overriding
/// existing values, matching the original script behavior exactly.
///
/// # Validation
///
/// - **VAL-VBA-016**: Produces shell-sourceable content
/// - **VAL-VBA-017**: Includes ROCm optimization settings
pub fn generate_env_file_content(
    hip_visible_devices: &str,
    rocm_path: &str,
    rocm_version: &str,
    rocm_channel: &str,
    gpu_arch: &str,
    hsa_override_gfx_version: &str,
    python_bin: &str,
) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "$HOME".to_string());
    // RCCL overlay env is MANAGED here (the persistent-env generator is the
    // single source of truth for env vars — components no longer write env files).
    let rccl_block = rccl_overlay_bash_block(&home);
    // Patched MIGraphX core overlay (library-level compile-hang fix, opt-in via
    // MLSTACK_MIGRAPHX_CORE_FIX) — emitted only when the fixed core is present.
    let migx_block = migraphx_fixed_core_bash_block(&home);

    format!(
        r#"# ML Stack Environment File
# Created by Enhanced ML Stack Environment Setup (Rust native)
# Date: {date}

# GPU Selection — device filter is MANAGED (unconditional). It must reflect the
# detected discrete GPUs on every source so a stale override can never re-expose
# an integrated GPU (the iGPU segfault). ROCR_VISIBLE_DEVICES is the authoritative
# ROCr (HSA) runtime filter; HIP/CUDA alias it for the HIP + CUDA-compat layers.
# Discrete GPUs only (iGPUs filtered): {hip_visible_devices}
export ROCR_VISIBLE_DEVICES={hip_visible_devices}
export HIP_VISIBLE_DEVICES={hip_visible_devices}
export CUDA_VISIBLE_DEVICES={hip_visible_devices}
export PYTORCH_ROCM_DEVICE={hip_visible_devices}

# ROCm Settings
# Only set if not already set
if [ -z "${{ROCM_HOME:-}}" ]; then export ROCM_HOME={rocm_path}; fi
if [ -z "${{ROCM_PATH:-}}" ]; then export ROCM_PATH={rocm_path}; fi
if [ -z "${{HIP_PATH:-}}" ]; then export HIP_PATH={rocm_path}; fi
if [ -z "${{CUDA_HOME:-}}" ]; then export CUDA_HOME={rocm_path}; fi
if [ -z "${{ROCM_VERSION:-}}" ]; then export ROCM_VERSION={rocm_version}; fi
if [ -z "${{ROCM_CHANNEL:-}}" ]; then export ROCM_CHANNEL={rocm_channel}; fi
# GPU arch identity is MANAGED (unconditional): must match detected hardware.
export GPU_ARCH={gpu_arch}
export PYTORCH_ROCM_ARCH={gpu_arch}
export GPU_ARCHS={gpu_arch}
# libdrm amdgpu ASIC id table (ROCm's amdgpu.ids lookup)
export AMDGPU_ASIC_ID_TABLE_PATH=/usr/share/libdrm/amdgpu.ids
export AMDGPU_ASIC_ID_TABLE_PATHS=/usr/share/libdrm

# Path Settings - Hardcoded safe paths to prevent "command not found" errors.
# Idempotent: prepend the ROCm bin/lib blocks only if not already on the var, so
# re-sourcing the env never grows PATH/LD_LIBRARY_PATH (the old un-guarded
# `:$PATH` form accumulated a duplicate entry per source).
case ":${{PATH:-}}:" in *:"{rocm_path}/bin":*) ;; *) export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:{rocm_path}/bin:{rocm_path}/hip/bin:${{PATH:-}}";; esac
case ":${{LD_LIBRARY_PATH:-}}:" in *:"{rocm_path}/lib":*) ;; *) export LD_LIBRARY_PATH="$HOME/.mlstack/libmpi-compat:$HOME/.mlstack/libmpi-compat-user-$(id -u):{rocm_path}/lib:{rocm_path}/hip/lib:{rocm_path}/opencl/lib:${{LD_LIBRARY_PATH:-}}";; esac
{rccl_block}
{migx_block}

# Performance Settings
# HSA_OVERRIDE_GFX_VERSION is set based on detected GPU_ARCH
export HSA_OVERRIDE_GFX_VERSION={hsa_override_gfx_version}
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_TOOLS_LIB=/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so

# MIOpen Settings
# Only set if not already set
if [ -z "${{MIOPEN_DEBUG_CONV_IMPLICIT_GEMM:-}}" ]; then export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1; fi
if [ -z "${{MIOPEN_FIND_MODE:-}}" ]; then export MIOPEN_FIND_MODE=3; fi
if [ -z "${{MIOPEN_FIND_ENFORCE:-}}" ]; then export MIOPEN_FIND_ENFORCE=3; fi

# PyTorch Settings
# Only set if not already set
# (Stage 2: the CUDA build-arch env var is intentionally NOT exported on a
# ROCm stack — ROCm uses PYTORCH_ROCM_ARCH. No CUDA env leakage.)
if [ -z "${{PYTORCH_CUDA_ALLOC_CONF:-}}" ]; then export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"; fi
if [ -z "${{PYTORCH_HIP_ALLOC_CONF:-}}" ]; then export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512"; fi
if [ -z "${{VLLM_WORKER_MULTIPROC_METHOD:-}}" ]; then export VLLM_WORKER_MULTIPROC_METHOD=spawn; fi
if [ -z "${{VLLM_ROCM_USE_AITER:-}}" ]; then export VLLM_ROCM_USE_AITER=0; fi
if [ -z "${{MLSTACK_TRITON_HOME:-}}" ]; then export MLSTACK_TRITON_HOME="$HOME/.cache/mlstack/triton"; fi
if [ -z "${{TRITON_HOME:-}}" ]; then export TRITON_HOME="$MLSTACK_TRITON_HOME"; fi
if [ -z "${{TRITON_CACHE_DIR:-}}" ]; then export TRITON_CACHE_DIR="$TRITON_HOME/cache"; fi
if [ -z "${{TRITON_DUMP_DIR:-}}" ]; then export TRITON_DUMP_DIR="$TRITON_HOME/dump"; fi
if [ -z "${{TRITON_OVERRIDE_DIR:-}}" ]; then export TRITON_OVERRIDE_DIR="$TRITON_HOME/override"; fi
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR" "$TRITON_OVERRIDE_DIR" 2>/dev/null || true

# MPI Settings
# Only set if not already set
# (Stage 2: the OMPI CUDA-opal support vars are intentionally NOT set — that is
# the CUDA-built OpenMPI path; a ROCm stack uses the ROCm accelerator path.)
if [ -z "${{OMPI_MCA_btl_openib_allow_ib:-}}" ]; then export OMPI_MCA_btl_openib_allow_ib=true; fi
if [ -z "${{OMPI_MCA_btl_openib_warn_no_device_params_found:-}}" ]; then export OMPI_MCA_btl_openib_warn_no_device_params_found=0; fi
if [ -z "${{OMPI_MCA_coll_hcoll_enable:-}}" ]; then export OMPI_MCA_coll_hcoll_enable=0; fi
if [ -z "${{OMPI_MCA_pml:-}}" ]; then export OMPI_MCA_pml=ucx; fi
if [ -z "${{OMPI_MCA_osc:-}}" ]; then export OMPI_MCA_osc=ucx; fi
if [ -z "${{OMPI_MCA_btl:-}}" ]; then export OMPI_MCA_btl=^openib,uct; fi
# Prevent UCX ROCm memory registration segfault on multi-GPU systems
export UCX_MEMTYPE_CACHE=no
export UCX_LOG_LEVEL=error

# Global flags for seamless pip installs (PEP 668 override)
export PIP_BREAK_SYSTEM_PACKAGES=1
export UV_PIP_BREAK_SYSTEM_PACKAGES=1
export UV_SYSTEM_PYTHON=1

# Python Interpreter
# Canonical Python binary — all ML components install to this interpreter.
# Ensures a single unified environment regardless of how many Pythons exist.
export MLSTACK_PYTHON_BIN={python_bin}
export UV_PYTHON={python_bin}

# Flash Attention AMD
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# ONNX Runtime Settings
# Only add if not already in PYTHONPATH
if ! echo "${{PYTHONPATH:-}}" | grep -q "{home}/onnxruntime_build/onnxruntime/build/Linux/Release"; then
  export PYTHONPATH={home}/onnxruntime_build/onnxruntime/build/Linux/Release:${{PYTHONPATH:-}}
fi
"#,
        date = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        hip_visible_devices = hip_visible_devices,
        rocm_path = rocm_path,
        rocm_version = rocm_version,
        rocm_channel = rocm_channel,
        gpu_arch = gpu_arch,
        hsa_override_gfx_version = hsa_override_gfx_version,
        home = home,
        python_bin = python_bin,
        rccl_block = rccl_block,
        migx_block = migx_block,
    )
}

/// Generate a fish-compatible environment file.
///
/// Produces `~/.config/fish/conf.d/mlstack_env.fish` that natively loads
/// all ML Stack environment variables without needing to source the bash file.
/// Fish doesn't support bash syntax (`if [ -z ... ]`, `${VAR:-}`, `fi`, etc.),
/// so this uses idiomatic fish constructs.
pub fn generate_fish_env_file_content(
    hip_visible_devices: &str,
    rocm_path: &str,
    rocm_version: &str,
    rocm_channel: &str,
    gpu_arch: &str,
    hsa_override_gfx_version: &str,
    python_bin: &str,
) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "$HOME".to_string());

    // Build ONNX Runtime PYTHONPATH check
    let onnx_path = format!("{home}/onnxruntime_build/onnxruntime/build/Linux/Release");
    // RCCL overlay env is MANAGED here (single source of truth — no component env files).
    let rccl_block = rccl_overlay_fish_block(&home);
    // Patched MIGraphX core overlay — emitted only when the fixed core is present.
    let migx_block = migraphx_fixed_core_fish_block(&home);

    format!(
        r##"# ML Stack Environment File — Fish Shell Native
# Generated by Enhanced ML Stack Environment Setup (Rust native)
# Date: {date}
#
# This file is auto-generated. Edit via rusty-stack's env setup, not by hand.

# --- GPU Selection — device filter is MANAGED (unconditional) so a stale
# override can never re-expose an integrated GPU. ROCR_VISIBLE_DEVICES is the
# authoritative ROCr (HSA) filter. ---
# Discrete GPUs only (iGPUs filtered): {hip_visible_devices}
set -gx ROCR_VISIBLE_DEVICES {hip_visible_devices}
set -gx HIP_VISIBLE_DEVICES {hip_visible_devices}
set -gx CUDA_VISIBLE_DEVICES {hip_visible_devices}
set -gx PYTORCH_ROCM_DEVICE {hip_visible_devices}

# --- ROCm Settings ---
set -q ROCM_HOME; or set -gx ROCM_HOME {rocm_path}
set -q ROCM_PATH; or set -gx ROCM_PATH {rocm_path}
set -q HIP_PATH; or set -gx HIP_PATH {rocm_path}
set -q CUDA_HOME; or set -gx CUDA_HOME {rocm_path}
set -q ROCM_VERSION; or set -gx ROCM_VERSION {rocm_version}
set -q ROCM_CHANNEL; or set -gx ROCM_CHANNEL {rocm_channel}
# GPU arch identity is MANAGED (unconditional): must match detected hardware.
set -gx GPU_ARCH {gpu_arch}
set -gx PYTORCH_ROCM_ARCH {gpu_arch}
set -gx GPU_ARCHS {gpu_arch}
set -gx AMDGPU_ASIC_ID_TABLE_PATH /usr/share/libdrm/amdgpu.ids
set -gx AMDGPU_ASIC_ID_TABLE_PATHS /usr/share/libdrm

# --- Path Settings ---
# Idempotent: prepend the ROCm bin/lib blocks only if not already on the var, so
# re-sourcing never grows PATH/LD_LIBRARY_PATH (no per-source duplicate buildup).
contains -- {rocm_path}/bin $PATH; or set -gx PATH /usr/local/bin /usr/bin /bin /usr/local/games /usr/games {rocm_path}/bin {rocm_path}/hip/bin $PATH
contains -- {rocm_path}/lib $LD_LIBRARY_PATH; or set -gx LD_LIBRARY_PATH $HOME/.mlstack/libmpi-compat $HOME/.mlstack/libmpi-compat-user-(id -u) {rocm_path}/lib {rocm_path}/hip/lib {rocm_path}/opencl/lib $LD_LIBRARY_PATH
{rccl_block}
{migx_block}

# --- Performance Settings ---
set -gx HSA_OVERRIDE_GFX_VERSION {hsa_override_gfx_version}
set -gx HSA_ENABLE_SDMA 0
set -gx GPU_MAX_HEAP_SIZE 100
set -gx GPU_MAX_ALLOC_PERCENT 100
set -gx HSA_TOOLS_LIB /opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so

# --- MIOpen Settings ---
set -q MIOPEN_DEBUG_CONV_IMPLICIT_GEMM; or set -gx MIOPEN_DEBUG_CONV_IMPLICIT_GEMM 1
set -q MIOPEN_FIND_MODE; or set -gx MIOPEN_FIND_MODE 3
set -q MIOPEN_FIND_ENFORCE; or set -gx MIOPEN_FIND_ENFORCE 3

# --- PyTorch Settings ---
# (Stage 2: the CUDA build-arch env var is intentionally NOT set on a ROCm
# stack — ROCm uses PYTORCH_ROCM_ARCH. No CUDA env leakage.)
set -q PYTORCH_CUDA_ALLOC_CONF; or set -gx PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:512"
set -q PYTORCH_HIP_ALLOC_CONF; or set -gx PYTORCH_HIP_ALLOC_CONF "max_split_size_mb:512"
set -q VLLM_WORKER_MULTIPROC_METHOD; or set -gx VLLM_WORKER_MULTIPROC_METHOD spawn
set -q VLLM_ROCM_USE_AITER; or set -gx VLLM_ROCM_USE_AITER 0
set -q MLSTACK_TRITON_HOME; or set -gx MLSTACK_TRITON_HOME "$HOME/.cache/mlstack/triton"
set -q TRITON_HOME; or set -gx TRITON_HOME $MLSTACK_TRITON_HOME
set -q TRITON_CACHE_DIR; or set -gx TRITON_CACHE_DIR "$TRITON_HOME/cache"
set -q TRITON_DUMP_DIR; or set -gx TRITON_DUMP_DIR "$TRITON_HOME/dump"
set -q TRITON_OVERRIDE_DIR; or set -gx TRITON_OVERRIDE_DIR "$TRITON_HOME/override"
mkdir -p $TRITON_CACHE_DIR $TRITON_DUMP_DIR $TRITON_OVERRIDE_DIR 2>/dev/null; or true

# --- Global flags for seamless pip installs (PEP 668 override) ---
set -q PIP_BREAK_SYSTEM_PACKAGES; or set -gx PIP_BREAK_SYSTEM_PACKAGES 1
set -q UV_PIP_BREAK_SYSTEM_PACKAGES; or set -gx UV_PIP_BREAK_SYSTEM_PACKAGES 1
set -gx UV_SYSTEM_PYTHON 1

# --- Python Interpreter ---
# Canonical Python binary — all ML components install to this interpreter.
# Ensures a single unified environment regardless of how many Pythons exist.
set -gx MLSTACK_PYTHON_BIN {python_bin}
set -gx UV_PYTHON {python_bin}

# --- Flash Attention AMD ---
set -gx FLASH_ATTENTION_TRITON_AMD_ENABLE TRUE

# --- MPI/UCX Settings ---
# (Stage 2: the OMPI CUDA-opal support vars are intentionally NOT set —
# CUDA-built OpenMPI path; a ROCm stack uses the ROCm accelerator path.)
set -q OMPI_MCA_btl_openib_allow_ib; or set -gx OMPI_MCA_btl_openib_allow_ib true
set -q OMPI_MCA_btl_openib_warn_no_device_params_found; or set -gx OMPI_MCA_btl_openib_warn_no_device_params_found 0
set -q OMPI_MCA_coll_hcoll_enable; or set -gx OMPI_MCA_coll_hcoll_enable 0
set -q OMPI_MCA_pml; or set -gx OMPI_MCA_pml ucx
set -q OMPI_MCA_osc; or set -gx OMPI_MCA_osc ucx
set -q OMPI_MCA_btl; or set -gx OMPI_MCA_btl '^openib,uct'
set -gx UCX_MEMTYPE_CACHE no
set -gx UCX_LOG_LEVEL error

# --- ONNX Runtime Settings ---
if not string match -q '*{onnx_path}*' "$PYTHONPATH"
    set -gx PYTHONPATH {onnx_path} $PYTHONPATH
end
"##,
        date = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        hip_visible_devices = hip_visible_devices,
        rocm_path = rocm_path,
        rocm_version = rocm_version,
        rocm_channel = rocm_channel,
        gpu_arch = gpu_arch,
        hsa_override_gfx_version = hsa_override_gfx_version,
        python_bin = python_bin,
        onnx_path = onnx_path,
        rccl_block = rccl_block,
        migx_block = migx_block,
    )
}

// ===========================================================================
// Component env contributions — MANAGED by the persistent-env generator.
// The generator is the SINGLE source of truth for env vars; components must
// NOT write their own env files (the old rccl `active/env.sh` did an un-guarded
// `export PYTHONPATH=X${PYTHONPATH:+:$PYTHONPATH}` that accumulated a duplicate
// entry on every `source ~/.mlstack_env`). Components instead declare state via
// the filesystem (the `active` symlink + manifest); the generator reads it and
// emits an idempotent block here.
// ===========================================================================

/// Path to the sealed RCCL overlay's `active` dir for `home`.
fn rccl_active_dir(home: &str) -> String {
    format!("{home}/.mlstack/components/rccl/active")
}

/// Read the RCCL overlay sha256 from `<active>/manifest` (line `sha256=…`), if
/// the overlay is installed and the manifest is readable.
fn rccl_overlay_sha256(active: &str) -> Option<String> {
    let manifest = std::fs::read_to_string(format!("{active}/manifest")).ok()?;
    let raw = manifest.lines().find_map(|l| {
        l.strip_prefix("sha256=")
            .map(str::trim)
            .filter(|s| !s.is_empty())
    })?;
    // Defense-in-depth: the SHA is written by the installer (always a 64-char
    // hex string), but this value is interpolated unquoted into a file that is
    // sourced on EVERY shell login. A corrupted/tampered manifest containing
    // shell metacharacters (`;`, backticks, `$()`, …) would execute when the
    // env file is sourced. Accept ONLY a 64-char lowercase-hex SHA-256; any
    // other shape is omitted (preserving the missing-manifest behavior).
    let is_hex64 = raw.len() == 64 && raw.chars().all(|c| c.is_ascii_hexdigit());
    if is_hex64 {
        Some(raw.to_string())
    } else {
        None
    }
}

/// Emit the **idempotent** bash block for the sealed RCCL overlay, or an empty
/// string when no overlay is installed.
///
/// Uses the `$HOME`-relative `active` symlink path in the emitted exports so
/// the block tracks the current RCCL build without an env refresh. The
/// PYTHONPATH prepend is guarded (`grep -qF`) so re-sourcing the env never
/// accumulates duplicates — the generator, not a component-written file, owns
/// these vars.
fn rccl_overlay_bash_block(home: &str) -> String {
    if !std::path::Path::new(&rccl_active_dir(home)).exists() {
        return String::new();
    }
    let sha_line = match rccl_overlay_sha256(&rccl_active_dir(home)) {
        // Quoted for defense-in-depth (the value is already validated 64-hex
        // upstream, so it carries no metacharacters — but quoting keeps the
        // sourced line robust if the validation contract ever loosens).
        Some(sha) => format!("  export MLSTACK_RCCL_OVERLAY_SHA256=\"{sha}\""),
        None => "  # MLSTACK_RCCL_OVERLAY_SHA256 omitted (manifest missing or invalid)".to_string(),
    };
    format!(
        r#"
# Sealed RCCL overlay — MANAGED by the persistent-env generator (single source of
# truth for env vars; the generator, not a component file, owns these). Follows
# the `active` symlink; the PYTHONPATH prepend is guarded so re-sourcing the env
# never accumulates duplicate entries.
if [ -r "$HOME/.mlstack/components/rccl/active/lib/librccl.so.1.0" ]; then
  export MLSTACK_RCCL_OVERLAY_LIB="$HOME/.mlstack/components/rccl/active/lib/librccl.so.1.0"
{sha_line}
  export NCCL_P2P_DISABLE=1
  export RCCL_P2P_DISABLE=1
  if ! echo "${{PYTHONPATH:-}}" | grep -qF "$HOME/.mlstack/components/rccl/active/python"; then
    export PYTHONPATH="$HOME/.mlstack/components/rccl/active/python:${{PYTHONPATH:-}}"
  fi
fi
"#
    )
}

/// Emit the **idempotent** fish block for the sealed RCCL overlay, or empty.
/// Fish analogue of [`rccl_overlay_bash_block`]; `contains` is the membership guard.
fn rccl_overlay_fish_block(home: &str) -> String {
    if !std::path::Path::new(&rccl_active_dir(home)).exists() {
        return String::new();
    }
    let sha_line = match rccl_overlay_sha256(&rccl_active_dir(home)) {
        Some(sha) => format!("    set -gx MLSTACK_RCCL_OVERLAY_SHA256 \"{sha}\""),
        None => {
            "    # MLSTACK_RCCL_OVERLAY_SHA256 omitted (manifest missing or invalid)".to_string()
        }
    };
    format!(
        r#"
# --- Sealed RCCL overlay — MANAGED by the persistent-env generator (single
# source of truth for env vars). Follows the `active` symlink; idempotent. ---
if test -r $HOME/.mlstack/components/rccl/active/lib/librccl.so.1.0
    set -gx MLSTACK_RCCL_OVERLAY_LIB $HOME/.mlstack/components/rccl/active/lib/librccl.so.1.0
{sha_line}
    set -gx NCCL_P2P_DISABLE 1
    set -gx RCCL_P2P_DISABLE 1
    if not contains -- $HOME/.mlstack/components/rccl/active/python $PYTHONPATH
        set -gx PYTHONPATH $HOME/.mlstack/components/rccl/active/python $PYTHONPATH
    end
end
"#
    )
}

// ===========================================================================
// Patched MIGraphX core overlay (library-level compile-hang fix, opt-in)
// ===========================================================================

/// `$HOME`-relative path to the patched MIGraphX core's lib dir.
///
/// The ORT MIGraphX EP provider resolves `libmigraphx.so.2015000` from
/// `<rocm>/lib/migraphx/lib` (its RUNPATH), so the fixed core must be laid out
/// at the mirror subpath `~/.mlstack/migraphx-fixed/lib/migraphx/lib` — that is
/// the directory we prepend to `LD_LIBRARY_PATH`, not the bare `lib`. Emitted
/// blocks reference `$HOME` at source-time (never the literal interpolated
/// `home`), matching the RCCL overlay precedent so a home containing spaces or
/// glob metacharacters can never break the sourced env file.
fn migraphx_fixed_core_lib_dir(home: &str) -> String {
    format!("{home}/.mlstack/migraphx-fixed/lib/migraphx/lib")
}

/// Emit the **idempotent** bash block that prepends the patched MIGraphX core
/// lib dir to `LD_LIBRARY_PATH`, or an empty string when no fixed core is
/// installed.
///
/// The patched core (built by the `MLSTACK_MIGRAPHX_CORE_FIX` env-gated path,
/// AMDMIGraphX `rocm-7.2.3` + PR #5106 backport) keeps the distro SONAME
/// (`libmigraphx.so.2015000`), so ORT's MIGraphX EP resolves it without a
/// rebuild. The EP provider uses RUNPATH (which the loader searches AFTER
/// `LD_LIBRARY_PATH`) and resolves the SONAME from `lib/migraphx/lib` — so
/// prepending the fixed-core `lib/migraphx/lib` dir makes the patched library
/// win without modifying `/opt/rocm`. Guarded (idempotent): re-sourcing the env
/// never accumulates a duplicate entry.
fn migraphx_fixed_core_bash_block(home: &str) -> String {
    let lib = migraphx_fixed_core_lib_dir(home);
    let so = format!("{lib}/libmigraphx.so.2015000");
    if !std::path::Path::new(&so).exists() {
        return String::new();
    }
    format!(
        r#"
# Patched MIGraphX core (PR #5106 find_concat_transpose fix) — MANAGED by the
# persistent-env generator. Prepended to LD_LIBRARY_PATH (guarded, idempotent)
# so the loader resolves the patched libmigraphx over the distro one via the
# RUNPATH-after-LD_LIBRARY_PATH precedence rule; /opt/rocm is untouched.
# Paths are $HOME-relative (resolved at source-time; never a literal home).
if [ -r "$HOME/.mlstack/migraphx-fixed/lib/migraphx/lib/libmigraphx.so.2015000" ]; then
  case ":${{LD_LIBRARY_PATH:-}}:" in *:"$HOME/.mlstack/migraphx-fixed/lib/migraphx/lib":*) ;; *) export LD_LIBRARY_PATH="$HOME/.mlstack/migraphx-fixed/lib/migraphx/lib:$HOME/.mlstack/migraphx-fixed/lib:${{LD_LIBRARY_PATH:-}}";; esac
fi
"#
    )
}

/// Emit the **idempotent** fish block for the patched MIGraphX core, or empty.
/// Fish analogue of [`migraphx_fixed_core_bash_block`]; `contains` is the
/// membership guard.
fn migraphx_fixed_core_fish_block(home: &str) -> String {
    let lib = migraphx_fixed_core_lib_dir(home);
    let so = format!("{lib}/libmigraphx.so.2015000");
    if !std::path::Path::new(&so).exists() {
        return String::new();
    }
    format!(
        r#"
# --- Patched MIGraphX core (PR #5106 find_concat_transpose fix) — MANAGED by
# the persistent-env generator; prepended to LD_LIBRARY_PATH; idempotent.
# Paths are $HOME-relative (resolved at source-time; never a literal home). ---
if test -r $HOME/.mlstack/migraphx-fixed/lib/migraphx/lib/libmigraphx.so.2015000
    contains -- $HOME/.mlstack/migraphx-fixed/lib/migraphx/lib $LD_LIBRARY_PATH; or set -gx LD_LIBRARY_PATH $HOME/.mlstack/migraphx-fixed/lib/migraphx/lib $HOME/.mlstack/migraphx-fixed/lib $LD_LIBRARY_PATH
end
"#
    )
}

// ===========================================================================
// Full Setup Orchestration
// ===========================================================================

/// Run the complete enhanced setup environment.
///
/// This is the Rust equivalent of the `main()` function in
/// `enhanced_setup_environment.sh`:
/// 1. Check system dependencies
/// 2. Detect AMD GPUs
/// 3. Detect ROCm
/// 4. Configure environment variables
/// 5. Create environment file
///
/// # Returns
///
/// A tuple of (env_file_result, gpu_filter_result, gpu_arch_info).
pub fn setup_environment() -> (EnvFileResult, GpuFilterResult, GpuArchInfo) {
    print_step("Enhanced ML Stack Environment Setup");

    // Step 1: Detect GPUs and filter
    print_step("Detecting AMD GPUs...");
    let gpu_filter = detect_discrete_gpus();
    if gpu_filter.gpu_count > 0 {
        print_success(&format!(
            "Detected {} discrete GPU(s): {}",
            gpu_filter.gpu_count, gpu_filter.hip_visible_devices
        ));
    } else {
        print_warning("No discrete GPUs detected, using defaults");
    }

    // Step 2: Detect ROCm
    let rocm_env = crate::installers::common::rocm_env::RocmEnv::detect();
    let rocm_path = rocm_env
        .path()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "/opt/rocm".to_string());
    let rocm_version = if rocm_env.version().is_empty() {
        "7.2.0".to_string()
    } else {
        rocm_env.version().to_string()
    };
    let rocm_channel = std::env::var("ROCM_CHANNEL").unwrap_or_else(|_| "latest".to_string());

    if rocm_env.is_detected() {
        print_success(&format!("ROCm {rocm_version} detected at {rocm_path}"));
    } else {
        print_warning("ROCm not detected, using default paths");
    }

    // Step 3: Detect GPU architecture
    let gpu_arch_info = detect_gpu_arch_from_system();

    // Step 4: Resolve canonical Python and create environment file
    print_step("Creating environment file...");
    let python_bin = crate::platform::environment::resolve_canonical_python_bin();
    let env_result = create_env_file(
        &gpu_filter.hip_visible_devices,
        &rocm_path,
        &rocm_version,
        &rocm_channel,
        &gpu_arch_info.gpu_arch,
        &gpu_arch_info.hsa_override_gfx_version,
        &python_bin,
    );

    if env_result.success {
        print_success("Environment setup complete");
    }

    (env_result, gpu_filter, gpu_arch_info)
}

/// Detect GPU architecture from the system (rocminfo-based).
fn detect_gpu_arch_from_system() -> GpuArchInfo {
    let rocminfo_path = crate::installers::common::utils::resolve_rocminfo_path();
    if !command_exists(&rocminfo_path) && !command_exists("rocminfo") {
        return GpuArchInfo::default();
    }

    let output = match std::process::Command::new(&rocminfo_path).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return GpuArchInfo::default(),
    };

    let mut marketing_name = String::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Marketing Name:") {
            let name = trimmed
                .strip_prefix("Marketing Name:")
                .map(|s| s.trim())
                .unwrap_or("");

            // Skip integrated GPUs
            if is_integrated_gpu_name(name) {
                continue;
            }

            if !name.is_empty() {
                marketing_name = name.to_string();
                break;
            }
        }
    }

    detect_correct_gpu_arch(if marketing_name.is_empty() {
        None
    } else {
        Some(&marketing_name)
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- GPU Arch Info (VAL-VBA-017) ---

    #[test]
    fn test_gpu_arch_info_default() {
        let info = GpuArchInfo::default();
        assert_eq!(info.gpu_arch, "gfx1100");
        assert_eq!(info.hsa_override_gfx_version, "11.0.0");
    }

    #[test]
    fn test_gpu_arch_info_from_arch_all_variants() {
        let cases = vec![
            ("gfx1100", "11.0.0"),
            ("gfx1101", "11.0.1"),
            ("gfx1102", "11.0.2"),
            ("gfx1200", "12.0.0"),
            ("gfx1201", "12.0.1"),
            ("gfx1030", "10.3.0"),
            ("gfx1031", "10.3.1"),
            ("gfx1032", "10.3.2"),
            ("gfx9999", "11.0.0"), // unknown defaults
        ];

        for (arch, expected_hsa) in cases {
            let info = GpuArchInfo::from_arch(arch);
            assert_eq!(info.gpu_arch, arch, "GPU arch mismatch for {arch}");
            assert_eq!(
                info.hsa_override_gfx_version, expected_hsa,
                "HSA version mismatch for {arch}"
            );
        }
    }

    // --- Integrated GPU Detection (VAL-VBA-014) ---

    #[test]
    fn test_is_integrated_gpu_name_positive() {
        assert!(is_integrated_gpu_name("Raphael"));
        assert!(is_integrated_gpu_name("AMD Ryzen Graphics"));
        assert!(is_integrated_gpu_name("Integrated Graphics"));
        assert!(is_integrated_gpu_name("iGPU Device"));
        assert!(is_integrated_gpu_name("AMD Radeon Graphics"));
    }

    #[test]
    fn test_is_integrated_gpu_name_negative() {
        assert!(!is_integrated_gpu_name("Radeon RX 7900 XTX"));
        assert!(!is_integrated_gpu_name("Radeon RX 7800 XT"));
        assert!(!is_integrated_gpu_name("Radeon RX 9070 XT"));
        assert!(!is_integrated_gpu_name("Unknown GPU"));
    }

    #[test]
    fn test_is_integrated_gpu_name_case_insensitive() {
        assert!(is_integrated_gpu_name("raphael"));
        assert!(is_integrated_gpu_name("INTEGRATED"));
        assert!(is_integrated_gpu_name("AMD RADEON GRAPHICS"));
    }

    // --- GPU Detection (VAL-VBA-014) ---

    #[test]
    fn test_detect_discrete_gpus_returns_result() {
        let result = detect_discrete_gpus();
        // Should always return at least the default GPU 0
        assert!(!result.hip_visible_devices.is_empty());
        assert!(result.gpu_count >= 1);
    }

    #[test]
    fn test_detect_gpus_from_lspci_returns_vec() {
        // Just verify it doesn't panic
        let result = detect_gpus_from_lspci();
        // May be empty if no AMD GPUs, that's fine
        let _ = result;
    }

    #[test]
    fn test_detect_gpus_from_render_nodes_returns_vec() {
        let result = detect_gpus_from_render_nodes();
        let _ = result;
    }

    // --- Marketing Name to Architecture (VAL-VBA-017) ---

    #[test]
    fn test_detect_correct_gpu_arch_7900_xtx() {
        let info = detect_correct_gpu_arch(Some("Radeon RX 7900 XTX"));
        assert_eq!(info.gpu_arch, "gfx1100");
        assert_eq!(info.hsa_override_gfx_version, "11.0.0");
    }

    #[test]
    fn test_detect_correct_gpu_arch_7800_xt() {
        let info = detect_correct_gpu_arch(Some("Radeon RX 7800 XT"));
        assert_eq!(info.gpu_arch, "gfx1101");
    }

    #[test]
    fn test_detect_correct_gpu_arch_7600() {
        let info = detect_correct_gpu_arch(Some("Radeon RX 7600"));
        assert_eq!(info.gpu_arch, "gfx1102");
    }

    #[test]
    fn test_detect_correct_gpu_arch_9070_xt() {
        let info = detect_correct_gpu_arch(Some("Radeon RX 9070 XT"));
        assert_eq!(info.gpu_arch, "gfx1201");
    }

    #[test]
    fn test_detect_correct_gpu_arch_none() {
        let info = detect_correct_gpu_arch(None);
        assert_eq!(info.gpu_arch, "gfx1100"); // default
    }

    #[test]
    fn test_detect_correct_gpu_arch_unknown() {
        let info = detect_correct_gpu_arch(Some("Unknown GPU Model"));
        assert_eq!(info.gpu_arch, "gfx1100"); // default fallback
    }

    #[test]
    fn test_detect_correct_gpu_arch_integrated_skipped() {
        let info = detect_correct_gpu_arch(Some("AMD Ryzen Graphics"));
        assert_eq!(info.gpu_arch, "gfx1100"); // default (skipped iGPU)
    }

    #[test]
    fn test_detect_correct_gpu_arch_case_insensitive() {
        let info = detect_correct_gpu_arch(Some("radeon rx 7900 xtx"));
        assert_eq!(info.gpu_arch, "gfx1100");
    }

    // --- System Dependencies (VAL-VBA-015) ---

    #[test]
    fn test_system_dependency_required_packages() {
        let packages = SystemDependency::required_packages();
        assert!(!packages.is_empty());
        assert!(packages.contains(&"git"));
        assert!(packages.contains(&"cmake"));
        assert!(packages.contains(&"build-essential"));
    }

    #[test]
    fn test_check_system_dependency_git() {
        let result = check_system_dependency("git");
        assert_eq!(result.package, "git");
        // git should be installed on any dev system
        assert!(result.installed, "git should be installed");
    }

    #[test]
    fn test_check_system_dependency_nonexistent() {
        let result = check_system_dependency("nonexistent-pkg-xyz-12345");
        assert!(!result.installed);
    }

    #[test]
    fn test_check_all_dependencies_returns_results() {
        let results = check_all_dependencies();
        assert!(!results.is_empty(), "Should check at least one dependency");
        assert_eq!(results.len(), SystemDependency::required_packages().len());
    }

    // --- Environment File Content (VAL-VBA-016, VAL-VBA-017) ---

    #[test]
    fn test_generate_env_file_has_header() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.starts_with("# ML Stack Environment File"));
    }

    #[test]
    fn test_generate_env_file_has_gpu_selection() {
        let content = generate_env_file_content(
            "0,1",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("HIP_VISIBLE_DEVICES"));
        assert!(content.contains("CUDA_VISIBLE_DEVICES"));
        assert!(content.contains("PYTORCH_ROCM_DEVICE"));
        assert!(content.contains("0,1"));
    }

    #[test]
    fn test_generate_env_file_has_rocm_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("ROCM_HOME"));
        assert!(content.contains("CUDA_HOME"));
        assert!(content.contains("ROCM_VERSION"));
        assert!(content.contains("ROCM_CHANNEL"));
        assert!(content.contains("GPU_ARCH"));
        assert!(content.contains("gfx1100"));
    }

    #[test]
    fn test_generate_env_file_has_path_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("PATH="));
        assert!(content.contains("LD_LIBRARY_PATH"));
        assert!(content.contains("/opt/rocm/bin"));
    }

    #[test]
    fn test_generate_env_file_has_performance_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("HSA_OVERRIDE_GFX_VERSION"));
        assert!(content.contains("HSA_ENABLE_SDMA"));
        assert!(content.contains("GPU_MAX_HEAP_SIZE"));
        assert!(content.contains("GPU_MAX_ALLOC_PERCENT"));
        assert!(content.contains("HSA_TOOLS_LIB"));
    }

    #[test]
    fn test_generate_env_file_has_miopen_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"));
        assert!(content.contains("MIOPEN_FIND_MODE"));
        assert!(content.contains("MIOPEN_FIND_ENFORCE"));
    }

    #[test]
    fn test_generate_env_file_has_pytorch_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(!content.contains("TORCH_CUDA_ARCH_LIST"));
        assert!(content.contains("PYTORCH_CUDA_ALLOC_CONF"));
        assert!(content.contains("PYTORCH_HIP_ALLOC_CONF"));
        assert!(content.contains("VLLM_WORKER_MULTIPROC_METHOD"));
    }

    #[test]
    fn test_generate_env_file_has_triton_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("MLSTACK_TRITON_HOME"));
        assert!(content.contains("TRITON_HOME"));
        assert!(content.contains("TRITON_CACHE_DIR"));
    }

    #[test]
    fn test_generate_env_file_has_mpi_settings() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(
            !content.contains("opal_cuda_support"),
            "must NOT set any OMPI CUDA-opal MCA var (Stage 2: ROCm uses the ROCm accelerator path)"
        );
        assert!(content.contains("OMPI_MCA_pml"));
        assert!(content.contains("OMPI_MCA_osc"));
        assert!(content.contains("OMPI_MCA_btl"));
    }

    #[test]
    fn test_generate_env_file_has_onnx_runtime() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("PYTHONPATH"));
        assert!(content.contains("onnxruntime"));
    }

    #[test]
    fn test_generate_env_file_uses_guard_syntax() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        // Device filter is MANAGED/unconditional (stale override must never
        // re-expose an iGPU): ROCR/HIP/CUDA_VISIBLE_DEVICES are bare exports.
        assert!(content.contains("export ROCR_VISIBLE_DEVICES=0"));
        assert!(content.contains("export HIP_VISIBLE_DEVICES=0"));
        assert!(!content.contains("if [ -z \"${HIP_VISIBLE_DEVICES:-}\" ]"));
        // Tuning vars still use "if [ -z ... ]" guards (respect user overrides).
        assert!(content.contains("if [ -z \"${ROCM_HOME:-}\" ]"));
    }

    #[test]
    fn test_generate_env_file_gpu_arch_not_guarded() {
        // GPU_ARCH should be exported unconditionally (not guarded)
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("export GPU_ARCH=gfx1100"));
    }

    #[test]
    fn test_generate_env_file_custom_values() {
        let content = generate_env_file_content(
            "0,1",
            "/opt/rocm-7.0",
            "7.0.0",
            "legacy",
            "gfx1030",
            "10.3.0",
            "python3",
        );
        assert!(content.contains("0,1"));
        assert!(content.contains("/opt/rocm-7.0"));
        assert!(content.contains("7.0.0"));
        assert!(content.contains("legacy"));
        assert!(content.contains("gfx1030"));
        assert!(content.contains("10.3.0"));
    }

    // --- create_env_file (VAL-VBA-016) ---

    #[test]
    fn test_create_env_file_creates_file_and_fish() {
        let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let _env_path = tmp_dir.path().join(".mlstack_env");
        let _fish_env_path = tmp_dir.path().join(".config/fish/conf.d/mlstack_env.fish");

        // Temporarily set HOME to the temp dir
        let original_home = std::env::var("HOME").ok();
        std::env::set_var("HOME", tmp_dir.path());

        let result = create_env_file(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );

        // Restore HOME
        if let Some(home) = original_home {
            std::env::set_var("HOME", home);
        } else {
            std::env::remove_var("HOME");
        }

        assert!(result.success, "create_env_file should succeed");
        assert!(result.path.is_some());

        // Verify the bash file exists and has content
        let path = result.path.unwrap();
        let content = std::fs::read_to_string(&path).expect("Should be able to read env file");
        assert!(content.contains("HIP_VISIBLE_DEVICES"));
        assert!(content.contains("ROCM_HOME"));
        assert!(content.contains("PIP_BREAK_SYSTEM_PACKAGES=1"));
        assert!(content.contains("UV_SYSTEM_PYTHON=1"));
    }

    // --- Fish env file generation ---

    #[test]
    fn test_generate_fish_env_file_has_fish_syntax() {
        let content = generate_fish_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        // Must NOT contain bash syntax
        assert!(!content.contains("if [ -z"));
        assert!(!content.contains("; fi"));
        assert!(!content.contains("export "));
        assert!(!content.contains("${"));
        // Must use fish syntax
        assert!(content.contains("set -gx"));
        assert!(content.contains("set -q"));
        assert!(content.contains("; or set -gx"));
    }

    #[test]
    fn test_generate_fish_env_file_has_required_vars() {
        let content = generate_fish_env_file_content(
            "0,1",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("HIP_VISIBLE_DEVICES"));
        assert!(content.contains("ROCM_HOME"));
        assert!(content.contains("ROCM_VERSION"));
        assert!(content.contains("GPU_ARCH"));
        assert!(content.contains("HSA_OVERRIDE_GFX_VERSION"));
        assert!(content.contains("PIP_BREAK_SYSTEM_PACKAGES"));
        assert!(content.contains("UV_PIP_BREAK_SYSTEM_PACKAGES"));
        assert!(content.contains("UV_SYSTEM_PYTHON"));
        assert!(content.contains("FLASH_ATTENTION_TRITON_AMD_ENABLE"));
        // Centralization: the generator must NOT source a component-written
        // env.fish — RCCL overlay env is emitted inline + idempotently by the
        // generator (rccl_overlay_fish_block), the single source of truth.
        assert!(!content.contains("source $HOME/.mlstack/components/rccl/active/env.fish"));
        assert!(content.contains("0,1"));
    }

    #[test]
    fn test_generate_fish_env_file_has_no_bash_patterns() {
        let content = generate_fish_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        // Should never contain bash-specific constructs
        assert!(!content.contains("if ! "));
        assert!(!content.contains("fi\n"));
        assert!(!content.contains("; then"));
    }

    #[test]
    fn test_generate_bash_env_file_has_pip_break_system_packages() {
        let content = generate_env_file_content(
            "0",
            "/opt/rocm",
            "7.2.0",
            "latest",
            "gfx1100",
            "11.0.0",
            "python3",
        );
        assert!(content.contains("export PIP_BREAK_SYSTEM_PACKAGES=1"));
        assert!(content.contains("export UV_PIP_BREAK_SYSTEM_PACKAGES=1"));
        assert!(content.contains("export UV_SYSTEM_PYTHON=1"));
        assert!(content.contains("export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"));
        // Centralization: the generator must NOT source a component-written
        // env.sh — RCCL overlay env is emitted inline + idempotently here.
        assert!(!content.contains(". \"$HOME/.mlstack/components/rccl/active/env.sh\""));
    }

    #[test]
    fn test_rccl_overlay_block_is_centralized_and_idempotent() {
        // No active RCCL overlay → generator emits nothing for it.
        let tmp = std::env::temp_dir();
        let home_no_rccl = tmp.join("rusty_rccl_no_active_test");
        let _ = std::fs::remove_dir_all(&home_no_rccl);
        assert_eq!(rccl_overlay_bash_block(&home_no_rccl.to_string_lossy()), "");
        assert_eq!(rccl_overlay_fish_block(&home_no_rccl.to_string_lossy()), "");

        // Active RCCL overlay → generator emits a guarded, idempotent block that
        // follows the `active` symlink (no component-written env file).
        let home = tmp.join("rusty_rccl_active_test");
        let active = home.join(".mlstack/components/rccl/active");
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(active.join("python")).unwrap();
        std::fs::create_dir_all(active.join("lib")).unwrap();
        std::fs::write(active.join("lib/librccl.so.1.0"), b"").unwrap();
        // A valid SHA-256 (64 lowercase hex chars) — the installer always writes
        // this shape. The generator validates 64-hex before splicing (defense-
        // in-depth against a tampered manifest), so use a realistic value here.
        let valid_sha = "a".repeat(64);
        std::fs::write(active.join("manifest"), format!("sha256={valid_sha}\n")).unwrap();

        let bash = rccl_overlay_bash_block(&home.to_string_lossy());
        assert!(bash.contains("MLSTACK_RCCL_OVERLAY_LIB"));
        // SHA is quoted + validated (64-hex). An invalid SHA would be omitted.
        assert!(bash.contains(&format!("MLSTACK_RCCL_OVERLAY_SHA256=\"{valid_sha}\"")));
        // Idempotent guard present (no bare un-guarded prepend).
        assert!(bash.contains("grep -qF"));
        // Single source of truth: does NOT delegate to a component env file.
        assert!(!bash.contains("env.sh"));

        let fish = rccl_overlay_fish_block(&home.to_string_lossy());
        assert!(fish.contains("MLSTACK_RCCL_OVERLAY_LIB"));
        assert!(fish.contains(&format!("MLSTACK_RCCL_OVERLAY_SHA256 \"{valid_sha}\"")));
        assert!(fish.contains("contains --"));
        assert!(!fish.contains("env.fish"));

        // Invalid SHA (not 64-hex) → omitted, never spliced into the sourced
        // block. Guards against a corrupted/tampered manifest injecting shell.
        std::fs::write(active.join("manifest"), "sha256=NOT-HEX\n").unwrap();
        let bash_bad = rccl_overlay_bash_block(&home.to_string_lossy());
        assert!(
            !bash_bad.contains("MLSTACK_RCCL_OVERLAY_SHA256=\""),
            "invalid SHA must be omitted, not spliced: {bash_bad}"
        );
        assert!(bash_bad.contains("omitted"));

        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&home_no_rccl);
    }

    #[test]
    fn test_migraphx_fixed_core_overlay_block_is_centralized_and_idempotent() {
        // No patched MIGraphX core → generator emits nothing for it.
        let tmp = std::env::temp_dir();
        let home_no_core = tmp.join("rusty_migx_no_core_test");
        let _ = std::fs::remove_dir_all(&home_no_core);
        assert_eq!(
            migraphx_fixed_core_bash_block(&home_no_core.to_string_lossy()),
            ""
        );
        assert_eq!(
            migraphx_fixed_core_fish_block(&home_no_core.to_string_lossy()),
            ""
        );

        // Patched core present → both blocks emit an idempotent LD_LIBRARY_PATH
        // prepend gated on the SONAME-matching lib file, at the `lib/migraphx/lib`
        // subpath the ORT provider's RUNPATH resolves (not the bare `lib`).
        let home = tmp.join("rusty_migx_core_test");
        let lib = home.join(".mlstack/migraphx-fixed/lib/migraphx/lib");
        let lib_c = home.join(".mlstack/migraphx-fixed/lib");
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("libmigraphx.so.2015000"), b"").unwrap();
        // Secondary entry: the bare `lib` holds libmigraphx_c.so.3 (the other
        // SONAME the provider resolves).
        std::fs::create_dir_all(&lib_c).unwrap();
        std::fs::write(lib_c.join("libmigraphx_c.so.3"), b"").unwrap();

        let bash = migraphx_fixed_core_bash_block(&home.to_string_lossy());
        assert!(bash.contains("libmigraphx.so.2015000"));
        // Idempotent: guard against duplicate accumulation on re-source.
        assert!(bash.contains("case \":${LD_LIBRARY_PATH:-}:\""));
        // Must prepend the RUNPATH-mirror subpath (patched lib wins via
        // RUNPATH-after-LD_LIBRARY_PATH rule), with the bare lib dir second.
        // Paths are $HOME-relative at source-time — never the literal `home`
        // (a home with spaces/glob metacharacters must not break sourcing).
        assert!(bash.contains("$HOME/.mlstack/migraphx-fixed/lib/migraphx/lib"));
        assert!(bash.contains("$HOME/.mlstack/migraphx-fixed/lib:"));
        assert!(!bash.contains(&lib.to_string_lossy().as_ref().to_string()));
        assert!(!bash.contains(&lib_c.to_string_lossy().as_ref().to_string()));

        let fish = migraphx_fixed_core_fish_block(&home.to_string_lossy());
        assert!(fish.contains("libmigraphx.so.2015000"));
        assert!(fish.contains("contains --"));
        assert!(fish.contains("$HOME/.mlstack/migraphx-fixed/lib/migraphx/lib"));
        assert!(fish.contains("$HOME/.mlstack/migraphx-fixed/lib"));
        assert!(!fish.contains(&lib.to_string_lossy().as_ref().to_string()));

        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&home_no_core);
    }

    // --- setup_environment integration ---

    #[test]
    fn test_setup_environment_returns_results() {
        let (env_result, gpu_result, gpu_arch) = setup_environment();
        // Should always return results without panicking
        let _ = env_result.success;
        assert!(!gpu_result.hip_visible_devices.is_empty());
        assert!(!gpu_arch.gpu_arch.is_empty());
    }
}
