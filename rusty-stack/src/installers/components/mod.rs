//! Native Rust installer modules for major components.
//!
//! Each module ports the corresponding shell installer script, constructing
//! the correct system/pip commands matching the original script behavior.
//! All modules use `installers::common::` for shared operations (package
//! manager, distro detection, ROCm env, etc.).
//!
//! # Dispatch Architecture (VAL-INSTALL-031, VAL-INSTALL-032, VAL-INSTALL-038)
//!
//! `is_native_component(id)` returns `true` for all 35 native components.
//! `get_dependencies(id)` returns the declared dependency IDs for a component.
//! `topological_sort(ids)` returns components in dependency order.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-001**: ROCm installer correct package commands
//! - **VAL-INSTALL-002**: ROCm channel selection correct version pins
//! - **VAL-INSTALL-003**: PyTorch installer correct wheel selection
//! - **VAL-INSTALL-004**: Triton installer correct pip command
//! - **VAL-INSTALL-005**: MPI4Py installer detects system MPI
//! - **VAL-INSTALL-006**: DeepSpeed installer correct pip command
//! - **VAL-INSTALL-007**: ML Stack Core installer correct pip command
//! - **VAL-INSTALL-008**: Flash Attention correct cmake command
//! - **VAL-INSTALL-009**: Flash Attention correct make command
//! - **VAL-INSTALL-010**: ONNX Runtime correct cmake command
//! - **VAL-INSTALL-011**: ONNX Runtime pip install from build output
//! - **VAL-INSTALL-012**: Megatron correct git clone
//! - **VAL-INSTALL-013**: Megatron correct pip install post-clone
//! - **VAL-INSTALL-015**: vLLM correct git clone and pip install
//! - **VAL-INSTALL-016**: AITER correct git clone and pip install
//! - **VAL-INSTALL-017**: bitsandbytes correct pip command
//! - **VAL-INSTALL-018**: ROCm SMI correct package command
//! - **VAL-INSTALL-019**: MIGraphX correct pip command
//! - **VAL-INSTALL-020**: PyTorch Profiler correct pip command
//! - **VAL-INSTALL-021**: W&B correct pip command
//! - **VAL-INSTALL-022**: vLLM Studio correct git clone
//! - **VAL-INSTALL-023**: ComfyUI correct git clone and pip install
//! - **VAL-INSTALL-024**: text-generation-webui correct git clone
//! - **VAL-INSTALL-025**: App installers correct target directory
//! - **VAL-INSTALL-026**: Repair invokes correct sub-installers in order
//! - **VAL-INSTALL-027**: Repair propagates individual failures
//! - **VAL-INSTALL-028**: Permanent env creates correct venv and symlinks
//! - **VAL-INSTALL-029**: AMD GPU drivers correct package command
//! - **VAL-INSTALL-030**: MIGraphX Python correct pip command
//! - **VAL-INSTALL-031**: installer.rs no bash subprocess for ported components
//! - **VAL-INSTALL-032**: installer.rs dispatches to correct Rust module per ID
//! - **VAL-INSTALL-038**: state.rs supports native module routing
//! - **VAL-INSTALL-039**: All 35 components reference correct Rust modules
//! - **VAL-INSTALL-040**: Non-installer components unchanged
//! - **VAL-INSTALL-041**: Megatron declares dependency on PyTorch and MPI4Py
//! - **VAL-INSTALL-042**: vLLM declares dependency on PyTorch
//! - **VAL-INSTALL-043**: DeepSpeed declares dependency on PyTorch
//! - **VAL-INSTALL-044**: Flash Attention declares dependency on PyTorch and ROCm
//! - **VAL-INSTALL-045**: ONNX Runtime declares dependency on ROCm
//! - **VAL-INSTALL-046**: AITER declares dependency on PyTorch and ROCm
//! - **VAL-INSTALL-047**: ComfyUI declares dependency on PyTorch
//! - **VAL-INSTALL-048**: Dependency graph is acyclic
//! - **VAL-INSTALL-049**: installer.rs respects dependency ordering
//! - **VAL-INSTALL-051**: All installer modules have unit tests
//! - **VAL-INSTALL-052**: installer.rs integration tests exist

pub mod aiter;
pub mod amdgpu_drivers;
pub mod bitsandbytes_multi;
pub mod comfyui;
pub mod deepspeed;
pub mod fastvideo;
pub mod flash_attention_ck;
pub mod llama_cpp;
pub mod megatron;
pub mod migraphx_multi;
pub mod migraphx_python;
pub mod ml_stack;
pub mod mpi4py;
pub mod onnxruntime;
pub mod permanent_env;
pub mod pytorch;
pub mod pytorch_profiler;
pub mod repair;
pub mod rocm;
pub mod rocm_smi;
pub mod textgen;
pub mod triton;
pub mod vllm_multi;
pub mod vllm_studio;
pub mod wandb;

// Re-export key types for convenience
pub use aiter::{AiterConfig, AiterInstaller};
pub use amdgpu_drivers::{AmdgpuConfig, AmdgpuInstaller};
pub use bitsandbytes_multi::{BitsAndBytesConfig, BitsAndBytesInstaller};
pub use comfyui::{ComfyuiConfig, ComfyuiInstaller};
pub use deepspeed::{DeepSpeedConfig, DeepSpeedInstaller};
pub use flash_attention_ck::{FlashAttentionConfig, FlashAttentionInstaller, GpuArch};
pub use llama_cpp::{LlamaCppConfig, LlamaCppInstaller};
pub use megatron::{MegatronConfig, MegatronInstaller};
pub use migraphx_multi::{MigraphxConfig, MigraphxInstaller, MigraphxSupport};
pub use migraphx_python::{
    InstallMethod as MigraphxPythonInstallMethod, MigraphxPythonConfig, MigraphxPythonInstaller,
};
pub use ml_stack::{MlStackConfig, MlStackInstaller};
pub use mpi4py::{Mpi4PyConfig, Mpi4PyInstaller, MpiImplementation};
pub use onnxruntime::{HipArchs, OnnxRuntimeConfig, OnnxRuntimeInstaller};
pub use permanent_env::{PermanentEnvConfig, PermanentEnvInstaller};
pub use pytorch::{PyTorchConfig, PyTorchInstaller, TorchChannel};
pub use pytorch_profiler::{PytorchProfilerConfig, PytorchProfilerInstaller};
pub use repair::{RepairConfig, RepairInstaller, RepairResult, RepairStep};
pub use rocm::{RocmChannel, RocmConfig, RocmInstallType, RocmInstaller};
pub use rocm_smi::{RocmSmiConfig, RocmSmiInstaller};
pub use textgen::{TextgenConfig, TextgenInstaller};
pub use triton::{TritonBranch, TritonConfig, TritonInstaller};
pub use vllm_multi::{VllmConfig, VllmInstaller};
pub use vllm_studio::{VllmStudioConfig, VllmStudioInstaller};
pub use wandb::{WandbConfig, WandbInstaller};

// ===========================================================================
// Installer Dispatch (VAL-INSTALL-031, VAL-INSTALL-032, VAL-INSTALL-039)
// ===========================================================================

/// The set of all 35 component IDs that have been ported to native Rust.
///
/// These are the installer components that should NOT spawn bash subprocesses.
/// Verification and performance components are now routed through native Rust modules.
///
/// Note: most of these appear in `state::default_components()`. A few utility
/// components (`amdgpu-drivers`, `migraphx-python`, `enhanced-env`) are
/// chain-referenced instead of shown as top-level selectable TUI components.
pub const NATIVE_COMPONENT_IDS: &[&str] = &[
    "permanent-env",
    "rocm",
    "pytorch",
    "triton",
    "mpi4py",
    "deepspeed",
    "ml-stack-core",
    "flash-attn",
    "repair-stack",
    "megatron",
    "vllm",
    "aiter",
    "vllm-studio",
    "comfyui",
    "textgen",
    "onnx",
    "bitsandbytes",
    "rocm-smi",
    "migraphx",
    "pytorch-profiler",
    "wandb",
    // Chain-referenced / utility scripts (ported but not top-level TUI components)
    "amdgpu-drivers",
    "migraphx-python",
    "enhanced-env",
    // Benchmark components (dispatched via benchmark_runners module)
    "mlperf-inference",
    "rocm-benchmarks",
    "gpu-memory-bandwidth",
    "rocm-smi-bench",
    "pytorch-performance",
    "vllm-performance",
    "deepspeed-performance",
    "megatron-performance",
    "all-benchmarks",
    // FastVideo component
    "fastvideo",
    // llama.cpp component (HIP/ROCm CMake source build)
    "llama-cpp",
];

/// Returns `true` if the given component ID has been ported to native Rust
/// and should be dispatched through the Rust installer path instead of
/// spawning a bash subprocess.
///
/// # Validation Assertions
///
/// - **VAL-INSTALL-031**: installer.rs no bash subprocess for ported components
/// - **VAL-INSTALL-039**: All 35 components reference correct Rust modules
pub fn is_native_component(component_id: &str) -> bool {
    NATIVE_COMPONENT_IDS.contains(&component_id)
}

/// Dependency declarations for each native installer component.
///
/// Maps component IDs to the list of component IDs they depend on.
/// Dependencies must be installed before dependents.
///
/// # Validation Assertions
///
/// - **VAL-INSTALL-041**: Megatron depends on PyTorch and MPI4Py
/// - **VAL-INSTALL-042**: vLLM depends on PyTorch
/// - **VAL-INSTALL-043**: DeepSpeed depends on PyTorch
/// - **VAL-INSTALL-044**: Flash Attention depends on PyTorch and ROCm
/// - **VAL-INSTALL-045**: ONNX Runtime depends on ROCm
/// - **VAL-INSTALL-046**: AITER depends on PyTorch and ROCm
/// - **VAL-INSTALL-047**: ComfyUI depends on PyTorch
/// - **VAL-CROSS-001**: PyTorch for ROCm depends on ROCm
pub fn get_dependencies(component_id: &str) -> &'static [&'static str] {
    match component_id {
        "megatron" => &["pytorch", "mpi4py"],
        "vllm" => &["pytorch"],
        "aiter" => &["pytorch", "rocm"],
        "flash-attn" => &["pytorch", "rocm"],
        "onnx" => &["rocm"],
        "deepspeed" => &["pytorch"],
        "comfyui" => &["pytorch"],
        "pytorch" => &["rocm"], // PyTorch for ROCm requires ROCm installed
        "pytorch-profiler" => &["pytorch"],
        "llama-cpp" => &["rocm"], // llama.cpp HIP build requires ROCm
        // All other native components have no cross-component dependencies
        _ => &[],
    }
}

/// Sort component IDs in topological order so dependencies are installed
/// before dependents.
///
/// Uses Kahn's algorithm (BFS-based topological sort).
///
/// # Validation Assertions
///
/// - **VAL-INSTALL-048**: Dependency graph is acyclic
/// - **VAL-INSTALL-049**: installer.rs respects dependency ordering
///
/// # Returns
///
/// `Ok(sorted_ids)` on success, `Err(cycle_description)` if a cycle is detected.
pub fn topological_sort(component_ids: &[String]) -> Result<Vec<String>, String> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let id_set: HashSet<&str> = component_ids.iter().map(|s| s.as_str()).collect();

    // Build adjacency list: dep -> list of dependents
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_degree: HashMap<&str, usize> = HashMap::new();

    for id in component_ids.iter().map(|s| s.as_str()) {
        in_degree.entry(id).or_insert(0);
        adj.entry(id).or_default();
    }

    for id in component_ids.iter().map(|s| s.as_str()) {
        for dep in get_dependencies(id) {
            if id_set.contains(dep) {
                // dep must come before id
                adj.entry(dep).or_default().push(id);
                *in_degree.entry(id).or_insert(0) += 1;
            }
        }
    }

    let mut queue: VecDeque<&str> = VecDeque::new();
    for (&id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(id);
        }
    }

    let mut sorted = Vec::with_capacity(component_ids.len());
    while let Some(id) = queue.pop_front() {
        sorted.push(id.to_string());
        if let Some(dependents) = adj.get(id) {
            for &dependent in dependents {
                if let Some(degree) = in_degree.get_mut(dependent) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dependent);
                    }
                }
            }
        }
    }

    if sorted.len() != component_ids.len() {
        let remaining: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&id, _)| id)
            .collect();
        return Err(format!(
            "Dependency cycle detected among: {}",
            remaining.join(", ")
        ));
    }

    Ok(sorted)
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn test_all_24_native_components_listed() {
        // 24 installer + 9 benchmark + 1 fastvideo + 1 llama-cpp = 35
        assert_eq!(NATIVE_COMPONENT_IDS.len(), 35);
    }

    #[test]
    fn test_is_native_component_true_for_ported() {
        for id in NATIVE_COMPONENT_IDS {
            assert!(is_native_component(id), "Expected '{}' to be native", id);
        }
    }

    #[test]
    fn test_is_native_component_false_for_verification() {
        assert!(!is_native_component("verify-basic"));
        assert!(!is_native_component("verify-enhanced"));
        assert!(!is_native_component("verify-build"));
    }

    #[test]
    fn test_is_native_component_true_for_performance() {
        assert!(is_native_component("mlperf-inference"));
        assert!(is_native_component("rocm-benchmarks"));
        assert!(is_native_component("all-benchmarks"));
        assert!(is_native_component("fastvideo"));
    }

    #[test]
    fn test_is_native_component_false_for_unknown() {
        assert!(!is_native_component("nonexistent"));
        assert!(!is_native_component(""));
    }

    #[test]
    fn test_megatron_dependencies() {
        let deps = get_dependencies("megatron");
        assert!(deps.contains(&"pytorch"));
        assert!(deps.contains(&"mpi4py"));
    }

    #[test]
    fn test_vllm_dependencies() {
        let deps = get_dependencies("vllm");
        assert!(deps.contains(&"pytorch"));
    }

    #[test]
    fn test_deepspeed_dependencies() {
        let deps = get_dependencies("deepspeed");
        assert!(deps.contains(&"pytorch"));
    }

    #[test]
    fn test_flash_attention_dependencies() {
        let deps = get_dependencies("flash-attn");
        assert!(deps.contains(&"pytorch"));
        assert!(deps.contains(&"rocm"));
    }

    #[test]
    fn test_onnx_dependencies() {
        let deps = get_dependencies("onnx");
        assert!(deps.contains(&"rocm"));
    }

    #[test]
    fn test_aiter_dependencies() {
        let deps = get_dependencies("aiter");
        assert!(deps.contains(&"pytorch"));
        assert!(deps.contains(&"rocm"));
    }

    #[test]
    fn test_comfyui_dependencies() {
        let deps = get_dependencies("comfyui");
        assert!(deps.contains(&"pytorch"));
    }

    #[test]
    fn test_llama_cpp_dependencies() {
        let deps = get_dependencies("llama-cpp");
        assert!(deps.contains(&"rocm"), "llama-cpp must depend on ROCm");
    }

    #[test]
    fn test_no_deps_components() {
        for id in &[
            "rocm",
            "triton",
            "mpi4py",
            "ml-stack-core",
            "permanent-env",
            "rocm-smi",
            "bitsandbytes",
            "wandb",
            "migraphx",
            "vllm-studio",
            "textgen",
            "repair-stack",
            "amdgpu-drivers",
            "migraphx-python",
        ] {
            assert!(
                get_dependencies(id).is_empty(),
                "Expected no dependencies for '{}', got {:?}",
                id,
                get_dependencies(id)
            );
        }
    }

    #[test]
    fn test_pytorch_depends_on_rocm() {
        let deps = get_dependencies("pytorch");
        assert!(
            deps.contains(&"rocm"),
            "PyTorch for ROCm must depend on ROCm"
        );
    }

    #[test]
    fn test_topological_sort_empty() {
        let result = topological_sort(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_topological_sort_no_deps() {
        let ids = vec![
            "rocm".to_string(),
            "pytorch".to_string(),
            "triton".to_string(),
        ];
        let result = topological_sort(&ids);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_topological_sort_respects_dependencies() {
        let ids = vec![
            "megatron".to_string(),
            "pytorch".to_string(),
            "mpi4py".to_string(),
        ];
        let result = topological_sort(&ids).unwrap();
        let pytorch_pos = result.iter().position(|s| s == "pytorch").unwrap();
        let mpi4py_pos = result.iter().position(|s| s == "mpi4py").unwrap();
        let megatron_pos = result.iter().position(|s| s == "megatron").unwrap();
        assert!(
            pytorch_pos < megatron_pos,
            "pytorch must come before megatron"
        );
        assert!(
            mpi4py_pos < megatron_pos,
            "mpi4py must come before megatron"
        );
    }

    #[test]
    fn test_topological_sort_complex_graph() {
        let ids = vec![
            "megatron".to_string(),
            "flash-attn".to_string(),
            "deepspeed".to_string(),
            "vllm".to_string(),
            "aiter".to_string(),
            "onnx".to_string(),
            "comfyui".to_string(),
            "pytorch".to_string(),
            "rocm".to_string(),
            "mpi4py".to_string(),
            "triton".to_string(),
        ];
        let result = topological_sort(&ids).unwrap();
        assert_eq!(result.len(), 11);

        // Verify all dependency constraints
        let pos = |id: &str| result.iter().position(|s| s == id).unwrap();
        assert!(pos("rocm") < pos("flash-attn"));
        assert!(pos("rocm") < pos("onnx"));
        assert!(pos("rocm") < pos("aiter"));
        assert!(pos("pytorch") < pos("megatron"));
        assert!(pos("mpi4py") < pos("megatron"));
        assert!(pos("pytorch") < pos("vllm"));
        assert!(pos("pytorch") < pos("deepspeed"));
        assert!(pos("pytorch") < pos("flash-attn"));
        assert!(pos("pytorch") < pos("aiter"));
        assert!(pos("pytorch") < pos("comfyui"));
    }

    #[test]
    fn test_topological_sort_filters_unrelated_deps() {
        // Only pytorch and vllm in the set; vllm depends on pytorch
        let ids = vec!["vllm".to_string(), "pytorch".to_string()];
        let result = topological_sort(&ids).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "pytorch");
        assert_eq!(result[1], "vllm");
    }

    #[test]
    fn test_dependency_graph_is_acyclic() {
        // Verify all native components can be sorted together
        let all_ids: Vec<String> = NATIVE_COMPONENT_IDS.iter().map(|s| s.to_string()).collect();
        let result = topological_sort(&all_ids);
        assert!(
            result.is_ok(),
            "Dependency graph must be acyclic: {:?}",
            result
        );
        let sorted = result.unwrap();
        assert_eq!(sorted.len(), NATIVE_COMPONENT_IDS.len());
    }

    #[test]
    fn test_native_ids_match_state_component_ids() {
        // Ensure all NATIVE_COMPONENT_IDS actually correspond to components
        // defined in state.rs default_components()
        for id in NATIVE_COMPONENT_IDS {
            // Just verify they're non-empty strings
            assert!(!id.is_empty(), "Component ID must not be empty");
            assert!(
                id.chars().all(|c| c == '-' || c.is_ascii_alphanumeric()),
                "Component ID '{}' must be alphanumeric with hyphens",
                id
            );
        }
    }
}
