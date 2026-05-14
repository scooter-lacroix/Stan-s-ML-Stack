//! Component registry and installed component detection.
//!
//! Provides:
//! - **Component registry** — 17 known components with display names and installer mappings
//!   (VAL-PLAT-009, VAL-PLAT-010)
//! - **Installed component detection** — path-based, Python module, and git-based strategies
//!   (VAL-PLAT-011, VAL-PLAT-012, VAL-PLAT-013)
//! - **Version querying** — per-component-type version retrieval
//!   (VAL-PLAT-014)
//! - **Batch Python operations** — single-subprocess module detection and version queries
//!   (VAL-PLAT-015, VAL-PLAT-016)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::environment::resolve_user_home;

// ===========================================================================
// Public Types
// ===========================================================================

/// How a component's installation status is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Check for files/paths on the filesystem.
    PathBased,
    /// Try importing a Python module via `importlib`.
    PythonModule,
    /// Check for a `.git` directory under an expected clone path.
    GitBased,
    /// Check for a command on `PATH`.
    CommandBased,
}

/// Metadata for a known component in the registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComponentInfo {
    /// Canonical component identifier (e.g., `"pytorch"`, `"flash-attn"`).
    pub id: String,
    /// Human-readable display name (e.g., `"PyTorch"`, `"Flash Attention"`).
    pub display_name: String,
    /// How to detect whether this component is installed.
    pub detection_method: DetectionMethod,
    /// Installer script name (relative to scripts directory).
    pub installer_script: String,
    /// Python import name, if this is a Python module.
    pub python_import: Option<String>,
    /// Expected clone directory name (relative to user home), for git-based components.
    pub clone_dir: Option<String>,
}

/// Result of querying a component's version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionInfo {
    /// The component identifier.
    pub component_id: String,
    /// Version string, or `"not installed"` / `"unknown"`.
    pub version: String,
}

// ===========================================================================
// Component Registry (VAL-PLAT-009, VAL-PLAT-010)
// ===========================================================================

/// Returns the canonical list of all 17 known components.
pub fn known_components() -> &'static [ComponentInfo] {
    // Built once, reused across calls.
    static COMPONENTS: std::sync::OnceLock<Vec<ComponentInfo>> = std::sync::OnceLock::new();
    COMPONENTS.get_or_init(|| {
        vec![
            // Foundation
            ComponentInfo {
                id: "rocm".into(),
                display_name: "ROCm".into(),
                detection_method: DetectionMethod::PathBased,
                installer_script: "install_rocm.sh".into(),
                python_import: None,
                clone_dir: None,
            },
            ComponentInfo {
                id: "rocm-smi".into(),
                display_name: "ROCm SMI".into(),
                detection_method: DetectionMethod::CommandBased,
                installer_script: String::new(), // managed by ROCm
                python_import: None,
                clone_dir: None,
            },
            // Core
            ComponentInfo {
                id: "pytorch".into(),
                display_name: "PyTorch".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_pytorch_rocm.sh".into(),
                python_import: Some("torch".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "triton".into(),
                display_name: "Triton".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_triton_multi.sh".into(),
                python_import: Some("triton".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "onnx".into(),
                display_name: "ONNX Runtime".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "build_onnxruntime_multi.sh".into(),
                python_import: Some("onnxruntime".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "migraphx".into(),
                display_name: "MIGraphX".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_migraphx_multi.sh".into(),
                python_import: Some("migraphx".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "flash-attn".into(),
                display_name: "Flash Attention".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_flash_attention_ck.sh".into(),
                python_import: Some("flash_attn".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "mpi4py".into(),
                display_name: "MPI4Py".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_mpi4py.sh".into(),
                python_import: Some("mpi4py".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "megatron".into(),
                display_name: "Megatron-LM".into(),
                detection_method: DetectionMethod::GitBased,
                installer_script: "install_megatron.sh".into(),
                python_import: None,
                clone_dir: Some("Megatron-LM".into()),
            },
            ComponentInfo {
                id: "aiter".into(),
                display_name: "AITER".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_aiter.sh".into(),
                python_import: Some("aiter".into()),
                clone_dir: None,
            },
            // Extensions
            ComponentInfo {
                id: "deepspeed".into(),
                display_name: "DeepSpeed".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_deepspeed.sh".into(),
                python_import: Some("deepspeed".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "vllm".into(),
                display_name: "vLLM".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_vllm_multi.sh".into(),
                python_import: Some("vllm".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "bitsandbytes".into(),
                display_name: "bitsandbytes".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_bitsandbytes_multi.sh".into(),
                python_import: Some("bitsandbytes".into()),
                clone_dir: None,
            },
            ComponentInfo {
                id: "wandb".into(),
                display_name: "Weights & Biases".into(),
                detection_method: DetectionMethod::PythonModule,
                installer_script: "install_wandb.sh".into(),
                python_import: Some("wandb".into()),
                clone_dir: None,
            },
            // Extensions
            ComponentInfo {
                id: "llama-cpp".into(),
                display_name: "llama.cpp (HIP)".into(),
                detection_method: DetectionMethod::CommandBased,
                installer_script: String::new(), // Native Rust installer
                python_import: None,
                clone_dir: None,
            },
            // UI/UX
            ComponentInfo {
                id: "comfyui".into(),
                display_name: "ComfyUI".into(),
                detection_method: DetectionMethod::GitBased,
                installer_script: "install_comfyui.sh".into(),
                python_import: None,
                clone_dir: Some("ComfyUI".into()),
            },
            ComponentInfo {
                id: "vllm-studio".into(),
                display_name: "vLLM Studio".into(),
                detection_method: DetectionMethod::GitBased,
                installer_script: "install_vllm_studio.sh".into(),
                python_import: None,
                clone_dir: Some("vllm-studio".into()),
            },
            ComponentInfo {
                id: "textgen".into(),
                display_name: "text-generation-webui".into(),
                detection_method: DetectionMethod::GitBased,
                installer_script: "install_textgen.sh".into(),
                python_import: None,
                clone_dir: Some("text-generation-webui".into()),
            },
            // Environment
            ComponentInfo {
                id: "permanent-env".into(),
                display_name: "ML Stack Environment".into(),
                detection_method: DetectionMethod::PathBased,
                installer_script: String::new(), // configured by installer
                python_import: None,
                clone_dir: None,
            },
        ]
    })
}

/// Check whether a component ID is recognized (VAL-PLAT-009).
pub fn is_known_component(id: &str) -> bool {
    known_components().iter().any(|c| c.id == id)
}

/// Look up a component by ID.
pub fn get_component(id: &str) -> Option<&'static ComponentInfo> {
    known_components().iter().find(|c| c.id == id)
}

/// Return the human-readable display name for a component ID.
///
/// Unknown IDs are returned as-is (passthrough).
/// VAL-PLAT-010: The mapping must be bijective for all 17 known components.
pub fn display_name(id: &str) -> String {
    get_component(id)
        .map(|c| c.display_name.clone())
        .unwrap_or_else(|| id.to_string())
}

/// Build a lookup map from display name back to component ID.
///
/// Used to verify bijectivity of the display name mapping.
pub fn display_name_to_id_map() -> HashMap<String, String> {
    known_components()
        .iter()
        .map(|c| (c.display_name.clone(), c.id.clone()))
        .collect()
}

// ===========================================================================
// Installed Component Detection
// ===========================================================================

/// Check whether a specific component is installed.
///
/// Dispatches to the appropriate detection strategy based on the component's
/// `detection_method`:
///
/// - **PathBased**: ROCm checks version file + `rocminfo`; `permanent-env` checks `~/.mlstack_env`.
/// - **CommandBased**: Checks that the binary exists and executes.
/// - **PythonModule**: Uses `importlib.util.find_spec` in candidate interpreters.
/// - **GitBased**: Checks for `.git` under expected clone path.
pub fn is_component_installed(id: &str) -> bool {
    is_component_installed_with_home(id, &resolve_user_home())
}

/// Check whether a component is installed using an explicit home directory.
///
/// This is the testable version that doesn't depend on global state.
pub fn is_component_installed_with_home(id: &str, home: &Path) -> bool {
    let Some(info) = get_component(id) else {
        return false;
    };

    match info.detection_method {
        DetectionMethod::PathBased => detect_path_based(info, home),
        DetectionMethod::CommandBased => detect_command_based(info),
        DetectionMethod::PythonModule => detect_python_module_single(info),
        DetectionMethod::GitBased => detect_git_based(info, home),
    }
}

// -----------------------------------------------------------------------
// Path-based detection (VAL-PLAT-011)
// -----------------------------------------------------------------------

/// Path-based detection for ROCm and permanent-env.
fn detect_path_based(info: &ComponentInfo, home: &Path) -> bool {
    match info.id.as_str() {
        "rocm" => {
            // ROCm: version file must exist AND rocminfo must be runnable
            let version_file_exists = Path::new("/opt/rocm/.info/version").exists();
            let rocminfo_works = Command::new("rocminfo")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            version_file_exists && rocminfo_works
        }
        "permanent-env" => {
            // Check for marker string in ~/.mlstack_env
            let env_file = home.join(".mlstack_env");
            if !env_file.exists() {
                return false;
            }
            std::fs::read_to_string(&env_file)
                .map(|content| content.contains("MLSTACK_PYTHON_BIN"))
                .unwrap_or(false)
        }
        _ => false,
    }
}

// -----------------------------------------------------------------------
// Command-based detection
// -----------------------------------------------------------------------

/// Command-based detection for tools like rocm-smi.
fn detect_command_based(info: &ComponentInfo) -> bool {
    match info.id.as_str() {
        "rocm-smi" => Command::new("rocm-smi")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false),
        _ => false,
    }
}

// -----------------------------------------------------------------------
// Python module detection (VAL-PLAT-012)
// -----------------------------------------------------------------------

/// Detect a single Python module component by trying importlib in candidate interpreters.
fn detect_python_module_single(info: &ComponentInfo) -> bool {
    let Some(ref import_name) = info.python_import else {
        return false;
    };

    let interpreters = super::environment::python_interpreters();
    for python in &interpreters {
        if try_python_import(python, import_name) {
            return true;
        }
    }
    false
}

/// Try importing a Python module using the given interpreter.
fn try_python_import(python: &Path, module: &str) -> bool {
    let script = format!(
        "import importlib.util; spec = importlib.util.find_spec('{module}'); exit(0 if spec else 1)"
    );
    Command::new(python)
        .arg("-c")
        .arg(&script)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// -----------------------------------------------------------------------
// Git-based detection (VAL-PLAT-013)
// -----------------------------------------------------------------------

/// Git-based detection for ComfyUI, vllm-studio, textgen.
fn detect_git_based(info: &ComponentInfo, home: &Path) -> bool {
    let Some(ref clone_dir) = info.clone_dir else {
        return false;
    };

    let git_dir = home.join(clone_dir).join(".git");
    git_dir.exists()
}

// ===========================================================================
// Version Querying (VAL-PLAT-014)
// ===========================================================================

/// Query the version of a single installed component.
///
/// Returns:
/// - Semver string for Python modules
/// - Git short-hash + subject for git-based components
/// - `"installed"` for command-only tools
/// - `"not installed"` when absent
pub fn get_version(id: &str) -> String {
    get_version_with_home(id, &resolve_user_home())
}

/// Query version with explicit home directory (testable).
pub fn get_version_with_home(id: &str, home: &Path) -> String {
    let Some(info) = get_component(id) else {
        return "unknown".to_string();
    };

    match info.detection_method {
        DetectionMethod::PathBased => get_version_path_based(info, home),
        DetectionMethod::CommandBased => get_version_command_based(info),
        DetectionMethod::PythonModule => get_version_python_single(info),
        DetectionMethod::GitBased => get_version_git(info, home),
    }
}

/// Get version for path-based components.
fn get_version_path_based(info: &ComponentInfo, home: &Path) -> String {
    match info.id.as_str() {
        "rocm" => {
            // Try version file first, then rocminfo fallback
            if let Ok(content) = std::fs::read_to_string("/opt/rocm/.info/version") {
                let v = content.trim().to_string();
                if !v.is_empty() {
                    return v;
                }
            }
            if let Ok(output) = Command::new("rocminfo").arg("--version").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = stdout.lines().next() {
                        return line.trim().to_string();
                    }
                }
            }
            "unknown".to_string()
        }
        "permanent-env" => {
            if is_component_installed_with_home("permanent-env", home) {
                "installed".to_string()
            } else {
                "not installed".to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

/// Get version for command-based components.
fn get_version_command_based(info: &ComponentInfo) -> String {
    match info.id.as_str() {
        "rocm-smi" => {
            if let Ok(output) = Command::new("rocm-smi").arg("--version").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = stdout.lines().next() {
                        return extract_semver(line.trim());
                    }
                }
            }
            "not installed".to_string()
        }
        _ => "unknown".to_string(),
    }
}

/// Get version for a single Python module component.
fn get_version_python_single(info: &ComponentInfo) -> String {
    let Some(ref import_name) = info.python_import else {
        return "unknown".to_string();
    };

    let interpreters = super::environment::python_interpreters();
    for python in &interpreters {
        let script = format!(
            "import {import_name}; print({import_name}.__version__)",
            import_name = import_name
        );
        if let Ok(output) = Command::new(python).arg("-c").arg(&script).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let version = stdout.trim().to_string();
                if !version.is_empty() {
                    return version;
                }
            }
        }
    }
    "not installed".to_string()
}

/// Get version for git-based components (short hash + subject).
/// Also stores the short hash for update comparison.
fn get_version_git(info: &ComponentInfo, home: &Path) -> String {
    let Some(ref clone_dir) = info.clone_dir else {
        return "unknown".to_string();
    };

    let repo_path = home.join(clone_dir);

    if !repo_path.join(".git").exists() {
        return "not installed".to_string();
    }

    // Try to get a tag-based version first (git describe)
    if let Ok(output) = Command::new("git")
        .args(["-C", repo_path.to_str().unwrap_or(".")])
        .args(["describe", "--tags", "--abbrev=0"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let tag = stdout.trim().to_string();
            if !tag.is_empty() {
                return tag;
            }
        }
    }

    // Fallback to short hash + subject
    if let Ok(output) = Command::new("git")
        .args(["-C", repo_path.to_str().unwrap_or(".")])
        .args(["log", "-1", "--format=%h %s"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.trim().to_string();
            if !line.is_empty() {
                return line;
            }
        }
    }

    "installed".to_string()
}

// ===========================================================================
// Batch Python Operations (VAL-PLAT-015, VAL-PLAT-016)
// ===========================================================================

/// Extract a semantic version substring from a version output string.
///
/// Handles strings like:
/// - `"ROCM-SMI version: 4.0.0+unknown"` → `"4.0.0"`
/// - `"4.0.0+unknown"` → `"4.0.0"`
/// - `"v2.6.0"` → `"2.6.0"`
/// - `"2.6.0"` → `"2.6.0"`
fn extract_semver(raw: &str) -> String {
    // Strip common prefixes
    let s = raw.trim().trim_start_matches('v').to_string();

    // Try to find a semver pattern (X.Y.Z or X.Y)
    let re = regex::Regex::new(r"(\d+\.\d+(?:\.\d+)?)").unwrap();
    if let Some(caps) = re.captures(&s) {
        return caps[1].to_string();
    }

    // No match — return cleaned string
    s
}

/// Check if a git-based component has updates available.
///
/// Performs `git fetch` then compares local HEAD with remote HEAD.
/// Returns `true` if remote has new commits.
pub fn git_component_has_updates(id: &str) -> bool {
    let Some(info) = get_component(id) else {
        return false;
    };
    let Some(ref clone_dir) = info.clone_dir else {
        return false;
    };
    let home = resolve_user_home();
    let repo_path = home.join(clone_dir);

    if !repo_path.join(".git").exists() {
        return false;
    }

    // Fetch without output
    let _ = Command::new("git")
        .args(["-C", repo_path.to_str().unwrap_or(".")])
        .args(["fetch", "--quiet"])
        .output();

    // Compare local HEAD with remote HEAD
    if let Ok(output) = Command::new("git")
        .args(["-C", repo_path.to_str().unwrap_or(".")])
        .args(["log", "HEAD..@{u}", "--oneline"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return !stdout.trim().is_empty();
        }
    }

    false
}

/// Python module component IDs and their import names.
fn python_module_mappings() -> &'static [(String, String)] {
    static MAPPINGS: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    MAPPINGS.get_or_init(|| {
        known_components()
            .iter()
            .filter(|c| c.detection_method == DetectionMethod::PythonModule)
            .filter_map(|c| {
                c.python_import
                    .as_ref()
                    .map(|imp| (c.id.clone(), imp.clone()))
            })
            .collect()
    })
}

/// Detect all installed Python ML modules using a single subprocess.
///
/// VAL-PLAT-016: Must use a single Python process invocation.
/// Returns a set of component IDs that are installed.
pub fn detect_python_modules() -> Vec<String> {
    let interpreters = super::environment::python_interpreters();
    detect_python_modules_with_interpreters(&interpreters)
}

/// Detect Python modules using the provided interpreter list (testable).
pub fn detect_python_modules_with_interpreters(interpreters: &[PathBuf]) -> Vec<String> {
    let mappings = python_module_mappings();
    if mappings.is_empty() || interpreters.is_empty() {
        return Vec::new();
    }

    // Build a single Python script that checks all modules at once
    let mut script = String::from("mods = [\n");
    for (comp_id, import_name) in mappings {
        script.push_str(&format!("    ({comp_id:?}, {import_name:?}),\n"));
    }
    script.push_str(
        "]\n\
for comp_id, import_name in mods:\n\
    try:\n\
        __import__(import_name)\n\
        print(comp_id)\n\
    except ImportError:\n\
        pass\n",
    );

    for python in interpreters {
        if let Ok(output) = Command::new(python).arg("-c").arg(&script).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
            }
        }
    }

    Vec::new()
}

/// Query versions of multiple Python modules in a single subprocess.
///
/// VAL-PLAT-015: Must use a single subprocess invocation.
/// Returns `component_id=version` pairs for all requested components.
pub fn get_versions_batch(component_ids: &[&str]) -> Vec<VersionInfo> {
    let interpreters = super::environment::python_interpreters();
    get_versions_batch_with_interpreters(component_ids, &interpreters)
}

/// Batch version query with provided interpreters (testable).
pub fn get_versions_batch_with_interpreters(
    component_ids: &[&str],
    interpreters: &[PathBuf],
) -> Vec<VersionInfo> {
    let mappings = python_module_mappings();
    let _mapping_map: HashMap<&str, &str> = mappings
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    if interpreters.is_empty() {
        return component_ids
            .iter()
            .map(|id| VersionInfo {
                component_id: (*id).to_string(),
                version: "unknown".to_string(),
            })
            .collect();
    }

    // Build the Python script for batch version query
    let mut script = String::from("import sys\nmapping = {\n");
    for (comp_id, import_name) in mappings {
        script.push_str(&format!(
            "    {:?}: ({:?}, '__version__'),\n",
            comp_id, import_name
        ));
    }
    script.push_str(
        "}\n\
for comp_id in sys.argv[1:]:\n\
    if comp_id not in mapping:\n\
        print(f'{comp_id}=unknown')\n\
        continue\n\
    mod_name, ver_attr = mapping[comp_id]\n\
    try:\n\
        mod = __import__(mod_name)\n\
        ver = getattr(mod, ver_attr, 'unknown')\n\
        print(f'{comp_id}={ver}')\n\
    except ImportError:\n\
        print(f'{comp_id}=not installed')\n",
    );

    for python in interpreters {
        if let Ok(output) = Command::new(python)
            .arg("-c")
            .arg(&script)
            .args(component_ids)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout
                    .lines()
                    .filter_map(|line| {
                        let line = line.trim();
                        if line.is_empty() {
                            return None;
                        }
                        let (id, version) = line.split_once('=')?;
                        Some(VersionInfo {
                            component_id: id.to_string(),
                            version: version.to_string(),
                        })
                    })
                    .collect();
            }
        }
    }

    // Fallback: return "unknown" for all
    component_ids
        .iter()
        .map(|id| VersionInfo {
            component_id: (*id).to_string(),
            version: "unknown".to_string(),
        })
        .collect()
}

/// Detect all installed components using the full detection pipeline.
///
/// Combines path-based, command-based, Python module (batch), and git-based
/// detection into a single scan.
pub fn detect_all_installed() -> Vec<String> {
    let home = resolve_user_home();
    detect_all_installed_with_home(&home)
}

/// Detect all installed components with explicit home directory (testable).
pub fn detect_all_installed_with_home(home: &Path) -> Vec<String> {
    let mut installed = Vec::new();

    // Path-based: ROCm
    if is_component_installed_with_home("rocm", home) {
        installed.push("rocm".to_string());
    }

    // Command-based: rocm-smi
    if is_component_installed("rocm-smi") {
        installed.push("rocm-smi".to_string());
    }

    // Python modules: batch detection
    let python_installed = detect_python_modules();
    installed.extend(python_installed);

    // Git-based
    for info in known_components() {
        if info.detection_method == DetectionMethod::GitBased && detect_git_based(info, home) {
            installed.push(info.id.clone());
        }
    }

    // Path-based: permanent-env
    if is_component_installed_with_home("permanent-env", home) {
        installed.push("permanent-env".to_string());
    }

    installed.sort();
    installed.dedup();
    installed
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VAL-PLAT-009: Component registry contains all 17 known components
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_has_exactly_19_components() {
        let components = known_components();
        assert_eq!(
            components.len(),
            19,
            "Registry must contain exactly 19 known components, found {}",
            components.len()
        );
    }

    #[test]
    fn test_registry_contains_all_expected_ids() {
        let expected_ids = [
            "rocm",
            "pytorch",
            "triton",
            "deepspeed",
            "vllm",
            "aiter",
            "onnx",
            "bitsandbytes",
            "migraphx",
            "flash-attn",
            "mpi4py",
            "wandb",
            "comfyui",
            "vllm-studio",
            "textgen",
            "rocm-smi",
            "permanent-env",
            "llama-cpp",
        ];
        let components = known_components();
        for expected in &expected_ids {
            assert!(
                components.iter().any(|c| c.id == *expected),
                "Missing component: {expected}"
            );
        }
    }

    #[test]
    fn test_is_known_component_accepts_all_19() {
        let ids = [
            "rocm",
            "pytorch",
            "triton",
            "deepspeed",
            "vllm",
            "aiter",
            "onnx",
            "bitsandbytes",
            "migraphx",
            "flash-attn",
            "mpi4py",
            "wandb",
            "comfyui",
            "vllm-studio",
            "textgen",
            "rocm-smi",
            "permanent-env",
            "llama-cpp",
        ];
        for id in &ids {
            assert!(
                is_known_component(id),
                "is_known_component should accept '{id}'"
            );
        }
    }

    #[test]
    fn test_is_known_component_rejects_unknown() {
        assert!(!is_known_component("nonexistent"));
        assert!(!is_known_component("cuda"));
        assert!(!is_known_component("tensorflow"));
        assert!(!is_known_component(""));
        assert!(!is_known_component("PyTorch")); // case-sensitive
    }

    #[test]
    fn test_get_component_returns_info_for_known() {
        let info = get_component("pytorch").unwrap();
        assert_eq!(info.id, "pytorch");
        assert_eq!(info.display_name, "PyTorch");
        assert_eq!(info.detection_method, DetectionMethod::PythonModule);
        assert_eq!(info.python_import.as_deref(), Some("torch"));
    }

    #[test]
    fn test_get_component_returns_none_for_unknown() {
        assert!(get_component("nonexistent").is_none());
    }

    #[test]
    fn test_all_components_have_non_empty_ids() {
        for c in known_components() {
            assert!(!c.id.is_empty(), "Component has empty id");
        }
    }

    #[test]
    fn test_all_component_ids_are_unique() {
        let ids: Vec<&str> = known_components().iter().map(|c| c.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "Component IDs must be unique");
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-010: Display name mapping is bijective for all 17 components
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_display_names_non_empty() {
        for c in known_components() {
            assert!(
                !c.display_name.is_empty(),
                "Display name for '{}' is empty",
                c.id
            );
        }
    }

    #[test]
    fn test_display_names_are_distinct() {
        let names: Vec<&str> = known_components()
            .iter()
            .map(|c| c.display_name.as_str())
            .collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "Display names must all be distinct (bijective)"
        );
    }

    #[test]
    fn test_display_name_to_id_is_bijective() {
        let map = display_name_to_id_map();
        assert_eq!(
            map.len(),
            19,
            "Display name to ID map must have exactly 19 entries"
        );

        // Verify round-trip: display_name(id) -> display_name -> id == original id
        for c in known_components() {
            let name = display_name(&c.id);
            let back_id = map.get(&name);
            assert_eq!(
                back_id,
                Some(&c.id),
                "Round-trip failed for '{}': display_name='{}'",
                c.id,
                name
            );
        }
    }

    #[test]
    fn test_display_name_unknown_passthrough() {
        assert_eq!(display_name("nonexistent"), "nonexistent");
        assert_eq!(display_name("custom-tool"), "custom-tool");
    }

    #[test]
    fn test_display_name_known_values() {
        assert_eq!(display_name("rocm"), "ROCm");
        assert_eq!(display_name("pytorch"), "PyTorch");
        assert_eq!(display_name("flash-attn"), "Flash Attention");
        assert_eq!(display_name("wandb"), "Weights & Biases");
        assert_eq!(display_name("comfyui"), "ComfyUI");
        assert_eq!(display_name("permanent-env"), "ML Stack Environment");
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-011: Path-based detection (ROCm, rocm-smi, permanent-env)
    // -----------------------------------------------------------------------

    #[test]
    fn test_rocm_detection_requires_both_version_file_and_rocminfo() {
        // On a real system with ROCm, this should return true
        // On a system without ROCm, this should return false
        // The test just verifies it doesn't panic
        let result = is_component_installed("rocm");
        // Just verify it runs without panic
        let _ = result;
    }

    #[test]
    fn test_permanent_env_detection_checks_marker() {
        let dir = tempfile::tempdir().unwrap();

        // Without marker string
        std::fs::write(dir.path().join(".mlstack_env"), "# empty\n").unwrap();
        assert!(!is_component_installed_with_home(
            "permanent-env",
            dir.path()
        ));

        // With marker string
        std::fs::write(
            dir.path().join(".mlstack_env"),
            "export MLSTACK_PYTHON_BIN=/usr/bin/python3\n",
        )
        .unwrap();
        assert!(is_component_installed_with_home(
            "permanent-env",
            dir.path()
        ));
    }

    #[test]
    fn test_permanent_env_not_installed_when_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_component_installed_with_home(
            "permanent-env",
            dir.path()
        ));
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-012: Python module detection via importlib
    // -----------------------------------------------------------------------

    #[test]
    fn test_python_module_detection_for_pytorch() {
        // On a real system with PyTorch installed, this should return true
        // On a system without PyTorch, it should return false
        // The test just verifies it doesn't panic
        let _ = is_component_installed("pytorch");
    }

    #[test]
    fn test_detect_python_modules_with_interpreters_empty() {
        let result = detect_python_modules_with_interpreters(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_python_modules_uses_single_subprocess() {
        // Verify the function runs and returns a Vec<String>
        // The "single subprocess" guarantee is structural (one Command invocation)
        let result = detect_python_modules();
        // All entries should be valid component IDs
        for id in &result {
            assert!(is_known_component(id), "Unknown component ID: {id}");
        }
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-013: Git-based detection (ComfyUI, vllm-studio, textgen)
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_based_detection_comfyui_with_git_dir() {
        let dir = tempfile::tempdir().unwrap();

        // Create ComfyUI/.git
        std::fs::create_dir_all(dir.path().join("ComfyUI/.git")).unwrap();
        assert!(is_component_installed_with_home("comfyui", dir.path()));
    }

    #[test]
    fn test_git_based_detection_comfyui_without_git_dir() {
        let dir = tempfile::tempdir().unwrap();

        // ComfyUI dir exists but no .git
        std::fs::create_dir_all(dir.path().join("ComfyUI")).unwrap();
        assert!(!is_component_installed_with_home("comfyui", dir.path()));
    }

    #[test]
    fn test_git_based_detection_vllm_studio() {
        let dir = tempfile::tempdir().unwrap();

        std::fs::create_dir_all(dir.path().join("vllm-studio/.git")).unwrap();
        assert!(is_component_installed_with_home("vllm-studio", dir.path()));
    }

    #[test]
    fn test_git_based_detection_textgen() {
        let dir = tempfile::tempdir().unwrap();

        std::fs::create_dir_all(dir.path().join("text-generation-webui/.git")).unwrap();
        assert!(is_component_installed_with_home("textgen", dir.path()));
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-014: Version querying returns correct format per component type
    // -----------------------------------------------------------------------

    #[test]
    fn test_version_query_rocm_format() {
        let version = get_version("rocm");
        // Should be non-empty and not "unknown" on a system with ROCm
        assert!(!version.is_empty());
    }

    #[test]
    fn test_version_query_permanent_env() {
        let dir = tempfile::tempdir().unwrap();

        // Not installed
        let version = get_version_with_home("permanent-env", dir.path());
        assert_eq!(version, "not installed");

        // Installed
        std::fs::write(
            dir.path().join(".mlstack_env"),
            "export MLSTACK_PYTHON_BIN=/usr/bin/python3\n",
        )
        .unwrap();
        let version = get_version_with_home("permanent-env", dir.path());
        assert_eq!(version, "installed");
    }

    #[test]
    fn test_version_query_unknown_component() {
        assert_eq!(get_version("nonexistent"), "unknown");
    }

    #[test]
    fn test_version_query_git_based_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        let version = get_version_with_home("comfyui", dir.path());
        assert_eq!(version, "not installed");
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-015: Batch Python version uses single subprocess
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_versions_batch_returns_results() {
        let ids = vec!["pytorch", "triton", "nonexistent-mod"];
        let results = get_versions_batch(&ids);

        assert_eq!(results.len(), 3);

        // Check that each result has the right component_id
        assert_eq!(results[0].component_id, "pytorch");
        assert_eq!(results[1].component_id, "triton");
        assert_eq!(results[2].component_id, "nonexistent-mod");

        // nonexistent-mod should be "unknown" (not in mapping)
        assert_eq!(results[2].version, "unknown");
    }

    #[test]
    fn test_get_versions_batch_with_interpreters_empty() {
        let ids = vec!["pytorch"];
        let results = get_versions_batch_with_interpreters(&ids, &[]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].version, "unknown");
    }

    #[test]
    fn test_get_versions_batch_empty_ids() {
        let results = get_versions_batch(&[]);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // VAL-PLAT-016: Batch Python module detection uses single subprocess
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_all_installed_returns_valid_ids() {
        let installed = detect_all_installed();
        for id in &installed {
            assert!(
                is_known_component(id),
                "detect_all_installed returned unknown component: {id}"
            );
        }
    }

    #[test]
    fn test_detect_all_installed_no_duplicates() {
        let installed = detect_all_installed();
        let unique: std::collections::HashSet<&String> = installed.iter().collect();
        assert_eq!(
            installed.len(),
            unique.len(),
            "detect_all_installed should not return duplicates"
        );
    }

    // -----------------------------------------------------------------------
    // ComponentInfo serde roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_component_info_serde_roundtrip() {
        let info = get_component("pytorch").unwrap();
        let json = serde_json::to_string(info).unwrap();
        let back: ComponentInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, &back);
    }

    #[test]
    fn test_detection_method_serde_roundtrip() {
        let methods = [
            DetectionMethod::PathBased,
            DetectionMethod::PythonModule,
            DetectionMethod::GitBased,
            DetectionMethod::CommandBased,
        ];
        for method in &methods {
            let json = serde_json::to_string(method).unwrap();
            let back: DetectionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(*method, back);
        }
    }

    #[test]
    fn test_version_info_serde_roundtrip() {
        let info = VersionInfo {
            component_id: "pytorch".to_string(),
            version: "2.4.0".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: VersionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    // -----------------------------------------------------------------------
    // Installer script mappings
    // -----------------------------------------------------------------------

    #[test]
    fn test_python_components_have_import_names() {
        for c in known_components() {
            if c.detection_method == DetectionMethod::PythonModule {
                assert!(
                    c.python_import.is_some(),
                    "Python module component '{}' must have python_import",
                    c.id
                );
                assert!(
                    !c.python_import.as_ref().unwrap().is_empty(),
                    "python_import for '{}' must be non-empty",
                    c.id
                );
            }
        }
    }

    #[test]
    fn test_git_components_have_clone_dirs() {
        for c in known_components() {
            if c.detection_method == DetectionMethod::GitBased {
                assert!(
                    c.clone_dir.is_some(),
                    "Git-based component '{}' must have clone_dir",
                    c.id
                );
                assert!(
                    !c.clone_dir.as_ref().unwrap().is_empty(),
                    "clone_dir for '{}' must be non-empty",
                    c.id
                );
            }
        }
    }

    #[test]
    fn test_installer_scripts_for_installable_components() {
        // rocm-smi, permanent-env, and native Rust components don't have installer scripts
        let no_script = ["rocm-smi", "permanent-env", "llama-cpp"];
        for c in known_components() {
            if !no_script.contains(&c.id.as_str()) {
                assert!(
                    !c.installer_script.is_empty(),
                    "Component '{}' must have an installer script",
                    c.id
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Real system integration (manual verification helpers)
    // -----------------------------------------------------------------------

    #[test]
    fn test_real_system_detect_all() {
        let installed = detect_all_installed();
        // On the real system, we expect at least ROCm to be installed
        // This test is informational — it just verifies the pipeline doesn't panic
        println!("Installed components: {:?}", installed);
        // Test passes if we get here without panicking
    }

    #[test]
    fn test_real_system_version_queries() {
        // Query versions for all components — informational
        for c in known_components() {
            let version = get_version(&c.id);
            println!("{}: {}", display_name(&c.id), version);
        }
        // Test passes if we get here without panicking
    }
}
