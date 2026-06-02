//! Windows-native platform support.
//!
//! Provides cfg-gated Windows implementations that compile cleanly on the
//! `x86_64-pc-windows-msvc` target without any Unix-only dependencies.
//!
//! # Platform Detection
//!
//! - `detect_backend_mode()` — classifies the current platform as
//!   `LinuxNative`, `WslBackedLinux`, or `WindowsNative`
//!
//! # Backend Routing
//!
//! - [`BackendRouter`] — selects the correct execution backend per component
//!   based on OS and WSL2 state (VAL-WIN-007)
//! - Unsupported components produce clear messages without attempting install
//!   (VAL-WIN-008)
//!
//! # Component Routing Matrix
//!
//! | Component     | Linux Native      | WSL-backed Linux  | Windows Native    |
//! |---------------|-------------------|--------------------|-------------------|
//! | rocm          | LegacyScript      | LegacyScript       | Unsupported       |
//! | pytorch       | LegacyScript      | LegacyScript       | Unsupported       |
//! | triton        | LegacyScript      | LegacyScript       | Unsupported       |
//! | deepspeed     | LegacyScript      | LegacyScript       | Unsupported       |
//! | vllm          | LegacyScript      | LegacyScript       | Unsupported       |
//! | aiter         | LegacyScript      | LegacyScript       | Unsupported       |
//! | onnx          | LegacyScript      | LegacyScript       | Unsupported       |
//! | bitsandbytes  | LegacyScript      | LegacyScript       | Unsupported       |
//! | migraphx      | LegacyScript      | LegacyScript       | Unsupported       |
//! | flash-attn    | LegacyScript      | LegacyScript       | Unsupported       |
//! | mpi4py        | LegacyScript      | LegacyScript       | Unsupported       |
//! | wandb         | ExternalPackageManager | ExternalPackageManager | ExternalPackageManager |
//! | comfyui       | LegacyScript      | LegacyScript       | Unsupported       |
//! | vllm-studio   | LegacyScript      | LegacyScript       | Unsupported       |
//! | textgen       | LegacyScript      | LegacyScript       | Unsupported       |
//! | rocm-smi      | LegacyScript      | LegacyScript       | Unsupported       |
//! | permanent-env | LegacyScript      | LegacyScript       | Unsupported       |

use crate::core::types::{BackendMode, ExecutorKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Backend Router
// ===========================================================================

/// Routes components to appropriate execution backends based on platform.
///
/// The router considers the [`BackendMode`] (Linux native, WSL-backed, or
/// Windows native) and selects the best [`ExecutorKind`] for each component.
///
/// # Examples
///
/// ```
/// use rusty_stack::platform::BackendRouter;
/// use rusty_stack::core::types::{BackendMode, ExecutorKind};
///
/// let router = BackendRouter::new(BackendMode::LinuxNative);
/// assert_ne!(router.executor_for("rocm"), ExecutorKind::Unsupported);
///
/// let win_router = BackendRouter::new(BackendMode::WindowsNative);
/// assert_eq!(win_router.executor_for("rocm"), ExecutorKind::Unsupported);
/// ```
#[derive(Debug, Clone)]
pub struct BackendRouter {
    mode: BackendMode,
    /// Component-specific overrides (component_id -> ExecutorKind).
    overrides: HashMap<String, ExecutorKind>,
}

impl BackendRouter {
    /// Create a new router for the given backend mode.
    pub fn new(mode: BackendMode) -> Self {
        Self {
            mode,
            overrides: HashMap::new(),
        }
    }

    /// Set an executor override for a specific component.
    pub fn set_override(&mut self, component_id: &str, kind: ExecutorKind) {
        self.overrides.insert(component_id.to_string(), kind);
    }

    /// Get the appropriate executor kind for the given component.
    ///
    /// Returns [`ExecutorKind::Unsupported`] for unknown components.
    pub fn executor_for(&self, component_id: &str) -> ExecutorKind {
        // Check overrides first
        if let Some(kind) = self.overrides.get(component_id) {
            return *kind;
        }

        // Check if the component is known
        if !is_known_component(component_id) {
            return ExecutorKind::Unsupported;
        }

        // Route based on backend mode and component
        match self.mode {
            BackendMode::LinuxNative => self.route_linux_native(component_id),
            BackendMode::WslBackedLinux => self.route_wsl_backed(component_id),
            BackendMode::WindowsNative => self.route_windows_native(component_id),
        }
    }

    /// Get the current backend mode.
    pub fn mode(&self) -> BackendMode {
        self.mode
    }

    /// Get a human-readable message explaining why a component is unsupported.
    ///
    /// Returns an empty string if the component is not unsupported.
    pub fn unsupported_message(&self, component_id: &str) -> String {
        let kind = self.executor_for(component_id);
        if kind != ExecutorKind::Unsupported {
            return String::new();
        }

        if !is_known_component(component_id) {
            return format!(
                "Component '{}' is not recognized by Rusty Stack.",
                component_id
            );
        }

        match self.mode {
            BackendMode::WindowsNative => format!(
                "Component '{}' requires a Linux environment with ROCm support. \
                 It cannot be installed natively on Windows. \
                 Consider using WSL2 with ROCm to enable Linux GPU computing on Windows.",
                display_name(component_id)
            ),
            BackendMode::WslBackedLinux => format!(
                "Component '{}' is not supported in the current WSL2 configuration.",
                display_name(component_id)
            ),
            BackendMode::LinuxNative => format!(
                "Component '{}' is not supported on this Linux configuration.",
                display_name(component_id)
            ),
        }
    }

    /// Route component on Linux native.
    fn route_linux_native(&self, component_id: &str) -> ExecutorKind {
        // On Linux native, all known ML components use legacy scripts
        // Python-only tools could use ExternalPackageManager
        match component_id {
            "wandb" => ExecutorKind::ExternalPackageManager,
            _ => ExecutorKind::LegacyScript,
        }
    }

    /// Route component on WSL-backed Linux.
    fn route_wsl_backed(&self, component_id: &str) -> ExecutorKind {
        // WSL-backed Linux can run the same scripts as native Linux
        match component_id {
            "wandb" => ExecutorKind::ExternalPackageManager,
            _ => ExecutorKind::LegacyScript,
        }
    }

    /// Route component on Windows native.
    fn route_windows_native(&self, component_id: &str) -> ExecutorKind {
        // On Windows native, only pure Python packages work
        match component_id {
            // Pure Python packages that work on Windows
            "wandb" => ExecutorKind::ExternalPackageManager,
            // Everything else requires Linux/ROCm
            _ => ExecutorKind::Unsupported,
        }
    }
}

// ===========================================================================
// Platform Detection
// ===========================================================================

/// Detect the current platform's backend mode.
///
/// Returns:
/// - `LinuxNative` on Linux without WSL indicators
/// - `WslBackedLinux` on Linux with WSL indicators (running inside WSL)
/// - `WindowsNative` on Windows
pub fn detect_backend_mode() -> BackendMode {
    #[cfg(target_os = "windows")]
    {
        BackendMode::WindowsNative
    }

    #[cfg(not(target_os = "windows"))]
    {
        if is_wsl_environment() {
            BackendMode::WslBackedLinux
        } else {
            BackendMode::LinuxNative
        }
    }
}

/// Check if the current Linux environment is running under WSL.
#[cfg(not(target_os = "windows"))]
fn is_wsl_environment() -> bool {
    if let Ok(version) = std::fs::read_to_string("/proc/version") {
        let version_lower = version.to_lowercase();
        return version_lower.contains("microsoft") || version_lower.contains("wsl");
    }
    false
}

// ===========================================================================
// Component Registry Helpers
// ===========================================================================

/// Check if a component ID is known.
fn is_known_component(id: &str) -> bool {
    matches!(
        id,
        "rocm"
            | "pytorch"
            | "triton"
            | "deepspeed"
            | "vllm"
            | "aiter"
            | "onnx"
            | "bitsandbytes"
            | "migraphx"
            | "flash-attn"
            | "mpi4py"
            | "wandb"
            | "comfyui"
            | "vllm-studio"
            | "textgen"
            | "rocm-smi"
            | "permanent-env"
    )
}

/// Get display name for a component.
fn display_name(id: &str) -> &str {
    match id {
        "rocm" => "ROCm",
        "pytorch" => "PyTorch",
        "triton" => "Triton",
        "deepspeed" => "DeepSpeed",
        "vllm" => "vLLM",
        "aiter" => "AITER",
        "onnx" => "ONNX Runtime",
        "bitsandbytes" => "bitsandbytes",
        "migraphx" => "MIGraphX",
        "flash-attn" => "Flash Attention",
        "mpi4py" => "MPI4Py",
        "wandb" => "Weights & Biases",
        "comfyui" => "ComfyUI",
        "vllm-studio" => "vLLM Studio",
        "textgen" => "Text Generation WebUI",
        "rocm-smi" => "ROCm SMI",
        "permanent-env" => "Permanent Environment",
        _ => id,
    }
}

// ===========================================================================
// Windows-Native Platform Info
// ===========================================================================

/// Windows-specific platform information.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WindowsInfo {
    /// Windows version string (e.g., "10.0.19045").
    pub version: String,
    /// Windows build number.
    pub build: u32,
    /// Whether WSL2 is available.
    pub wsl2_available: bool,
    /// Whether the system has a supported AMD GPU.
    pub has_amd_gpu: bool,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Backend Router Tests ----

    #[test]
    fn test_router_linux_native_rocm() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        assert_eq!(router.executor_for("rocm"), ExecutorKind::LegacyScript);
    }

    #[test]
    fn test_router_linux_native_all_components() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        let components = [
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
            "comfyui",
            "vllm-studio",
            "textgen",
            "rocm-smi",
            "permanent-env",
        ];
        for c in &components {
            assert_ne!(
                router.executor_for(c),
                ExecutorKind::Unsupported,
                "{} should not be unsupported on Linux",
                c
            );
        }
    }

    #[test]
    fn test_router_linux_native_wandb() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        assert_eq!(
            router.executor_for("wandb"),
            ExecutorKind::ExternalPackageManager
        );
    }

    #[test]
    fn test_router_windows_native_rocm_unsupported() {
        let router = BackendRouter::new(BackendMode::WindowsNative);
        assert_eq!(router.executor_for("rocm"), ExecutorKind::Unsupported);
    }

    #[test]
    fn test_router_windows_native_all_gpu_unsupported() {
        let router = BackendRouter::new(BackendMode::WindowsNative);
        let gpu_components = [
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
            "comfyui",
            "vllm-studio",
            "textgen",
            "rocm-smi",
            "permanent-env",
        ];
        for c in &gpu_components {
            assert_eq!(
                router.executor_for(c),
                ExecutorKind::Unsupported,
                "{} should be unsupported on Windows native",
                c
            );
        }
    }

    #[test]
    fn test_router_windows_native_wandb() {
        let router = BackendRouter::new(BackendMode::WindowsNative);
        assert_eq!(
            router.executor_for("wandb"),
            ExecutorKind::ExternalPackageManager
        );
    }

    #[test]
    fn test_router_wsl_backed_rocm() {
        let router = BackendRouter::new(BackendMode::WslBackedLinux);
        assert_eq!(router.executor_for("rocm"), ExecutorKind::LegacyScript);
    }

    #[test]
    fn test_router_unknown_component() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        assert_eq!(
            router.executor_for("nonexistent"),
            ExecutorKind::Unsupported
        );
    }

    #[test]
    fn test_router_override() {
        let mut router = BackendRouter::new(BackendMode::LinuxNative);
        router.set_override("rocm", ExecutorKind::Rust);
        assert_eq!(router.executor_for("rocm"), ExecutorKind::Rust);
    }

    // ---- Unsupported Message Tests ----

    #[test]
    fn test_unsupported_message_windows_rocm() {
        let router = BackendRouter::new(BackendMode::WindowsNative);
        let msg = router.unsupported_message("rocm");
        assert!(!msg.is_empty());
        assert!(msg.contains("ROCm"));
        assert!(msg.contains("Windows") || msg.contains("WSL"));
    }

    #[test]
    fn test_unsupported_message_unknown_component() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        let msg = router.unsupported_message("nonexistent");
        assert!(msg.contains("not recognized"));
    }

    #[test]
    fn test_unsupported_message_supported_component_empty() {
        let router = BackendRouter::new(BackendMode::LinuxNative);
        let msg = router.unsupported_message("rocm");
        assert!(
            msg.is_empty(),
            "Supported component should return empty message"
        );
    }

    // ---- Platform Detection Tests ----

    #[test]
    fn test_detect_backend_mode() {
        let mode = detect_backend_mode();
        // On this Linux system, should be LinuxNative or WslBackedLinux
        match mode {
            BackendMode::LinuxNative | BackendMode::WslBackedLinux => {}
            BackendMode::WindowsNative => {
                panic!("Should not detect WindowsNative on Linux")
            }
        }
    }

    // ---- Component Registry Tests ----

    #[test]
    fn test_is_known_component_all_17() {
        let known = [
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
        ];
        for c in &known {
            assert!(is_known_component(c), "{} should be known", c);
        }
    }

    #[test]
    fn test_is_known_component_unknowns() {
        let unknowns = ["foo", "bar", "cuda", "nvidia", ""];
        for c in &unknowns {
            assert!(!is_known_component(c), "{} should be unknown", c);
        }
    }

    #[test]
    fn test_display_name_known() {
        assert_eq!(display_name("rocm"), "ROCm");
        assert_eq!(display_name("pytorch"), "PyTorch");
        assert_eq!(display_name("wandb"), "Weights & Biases");
    }

    #[test]
    fn test_display_name_unknown_passthrough() {
        assert_eq!(display_name("custom-component"), "custom-component");
    }

    // ---- WindowsInfo Serde Tests ----

    #[test]
    fn test_windows_info_serde_roundtrip() {
        let info = WindowsInfo {
            version: "10.0.19045".to_string(),
            build: 19045,
            wsl2_available: true,
            has_amd_gpu: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: WindowsInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn test_windows_info_default() {
        let info = WindowsInfo::default();
        assert!(info.version.is_empty());
        assert_eq!(info.build, 0);
        assert!(!info.wsl2_available);
        assert!(!info.has_amd_gpu);
    }

    // ---- Router Mode Accessor ----

    #[test]
    fn test_router_mode() {
        let router = BackendRouter::new(BackendMode::WindowsNative);
        assert_eq!(router.mode(), BackendMode::WindowsNative);
    }
}
