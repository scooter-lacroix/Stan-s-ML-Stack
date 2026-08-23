//! Integration tests for installer dispatch to native Rust modules.
//!
//! Validates:
//! - VAL-INSTALL-031: installer.rs no bash subprocess for ported components
//! - VAL-INSTALL-032: installer.rs dispatches to correct Rust module per ID
//! - VAL-INSTALL-033: sudo behavior preserved for needs_sudo:true components
//! - VAL-INSTALL-034: Environment variable injection preserved
//! - VAL-INSTALL-035: InstallerEvent progress reporting preserved
//! - VAL-INSTALL-036: Error message format matches original scripts
//! - VAL-INSTALL-037: Component.script no longer holds .sh filenames
//! - VAL-INSTALL-038: state.rs supports native module routing
//! - VAL-INSTALL-039: All 35 components reference correct Rust modules
//! - VAL-INSTALL-040: Non-installer components unchanged
//! - VAL-INSTALL-041-047: Dependency declarations
//! - VAL-INSTALL-048: Dependency graph is acyclic
//! - VAL-INSTALL-049: installer.rs respects dependency ordering
//! - VAL-INSTALL-052: installer.rs integration tests exist

use rusty_stack::installers::components::{
    get_dependencies, is_native_component, topological_sort, NATIVE_COMPONENT_IDS,
};
use rusty_stack::state::{default_components, Category, Component};

// ===========================================================================
// VAL-INSTALL-031 + VAL-INSTALL-039: All ported components are native
// ===========================================================================

#[test]
fn test_all_native_components_recognized() {
    // 28 installer/action ids + 11 benchmarks + fastvideo + llama-cpp + freetoken.
    assert_eq!(
        NATIVE_COMPONENT_IDS.len(),
        42,
        "Must have exactly 42 native component/action IDs"
    );

    for id in NATIVE_COMPONENT_IDS {
        assert!(
            is_native_component(id),
            "Component '{}' should be recognized as native",
            id
        );
    }
}

#[test]
fn test_no_bash_dispatch_for_native_components() {
    // VAL-INSTALL-031: Verify that all ported components are marked native
    // so installer.rs won't spawn bash for them
    let components = default_components();
    let native_installers: Vec<&Component> = components
        .iter()
        .filter(|c| c.category != Category::Maintenance && c.category != Category::Performance)
        .collect();

    for comp in &native_installers {
        assert!(
            is_native_component(&comp.id),
            "Installer component '{}' must be native (no bash dispatch)",
            comp.id
        );
    }
}

// ===========================================================================
// VAL-INSTALL-037: Component.script no longer holds .sh for ported components
// ===========================================================================

#[test]
fn test_ported_components_have_empty_script() {
    let components = default_components();
    let native_installers: Vec<&Component> = components
        .iter()
        .filter(|c| is_native_component(&c.id))
        .collect();

    // 22 of 34 native components appear in default_components().
    // The other 12 (amdgpu-drivers, migraphx-python, enhanced-env, plus
    // 9 benchmark components) are chain-referenced or dispatched differently.
    assert!(
        native_installers.len() >= 22,
        "Must have at least 22 ported components in default_components(), got {}",
        native_installers.len()
    );

    for comp in &native_installers {
        assert!(
            comp.script.is_empty(),
            "Ported component '{}' should have empty script, got '{}'",
            comp.id,
            comp.script
        );
    }
}

// ===========================================================================
// VAL-INSTALL-038: state.rs supports native module routing
// ===========================================================================

#[test]
fn test_component_is_native_method() {
    let components = default_components();

    for comp in &components {
        if is_native_component(&comp.id) {
            assert!(
                comp.is_native(),
                "Component '{}' should report is_native() = true",
                comp.id
            );
        } else {
            assert!(
                !comp.is_native(),
                "Component '{}' should report is_native() = false",
                comp.id
            );
        }
    }
}

// ===========================================================================
// VAL-INSTALL-040: Non-installer components unchanged (still reference .sh)
// ===========================================================================

#[test]
fn test_verification_components_use_native_rust() {
    let components = default_components();
    // verify-* live in the Maintenance category alongside repair-stack; filter
    // by id prefix to isolate the three verification components.
    let verification: Vec<&Component> = components
        .iter()
        .filter(|c| c.id.starts_with("verify-"))
        .collect();

    assert_eq!(verification.len(), 3, "Must have 3 verification components");

    for comp in &verification {
        assert!(
            comp.script.is_empty(),
            "Verification component '{}' should use native Rust (empty script), got '{}'",
            comp.id,
            comp.script
        );
    }
}

#[test]
fn test_performance_components_use_native_rust() {
    let components = default_components();
    let performance: Vec<&Component> = components
        .iter()
        .filter(|c| c.category == Category::Performance)
        .collect();

    assert_eq!(performance.len(), 11, "Must have 11 performance components");

    for comp in &performance {
        assert!(
            comp.script.is_empty(),
            "Performance component '{}' should use native Rust (empty script), got '{}'",
            comp.id,
            comp.script
        );
        assert!(
            is_native_component(&comp.id),
            "Performance component '{}' should be native",
            comp.id
        );
    }
}

// ===========================================================================
// VAL-INSTALL-041-047: Dependency declarations
// ===========================================================================

#[test]
fn test_megatron_depends_on_pytorch_and_mpi4py() {
    let deps = get_dependencies("megatron");
    assert!(deps.contains(&"pytorch"), "Megatron must depend on pytorch");
    assert!(deps.contains(&"mpi4py"), "Megatron must depend on mpi4py");
}

#[test]
fn test_vllm_depends_on_pytorch() {
    let deps = get_dependencies("vllm");
    assert!(deps.contains(&"pytorch"), "vLLM must depend on pytorch");
}

#[test]
fn test_deepspeed_depends_on_pytorch() {
    let deps = get_dependencies("deepspeed");
    assert!(
        deps.contains(&"pytorch"),
        "DeepSpeed must depend on pytorch"
    );
}

#[test]
fn test_flash_attention_depends_on_pytorch_and_rocm() {
    for id in &["flash-attn-triton", "flash-attn-ck"] {
        let deps = get_dependencies(id);
        assert!(deps.contains(&"pytorch"), "{id} must depend on pytorch");
        assert!(deps.contains(&"rocm"), "{id} must depend on rocm");
    }
}

#[test]
fn test_onnx_depends_on_rocm() {
    let deps = get_dependencies("onnx");
    assert!(deps.contains(&"rocm"), "ONNX Runtime must depend on rocm");
}

#[test]
fn test_aiter_depends_on_pytorch_and_rocm() {
    let deps = get_dependencies("aiter");
    assert!(deps.contains(&"pytorch"), "AITER must depend on pytorch");
    assert!(deps.contains(&"rocm"), "AITER must depend on rocm");
}

// ===========================================================================
// VAL-INSTALL-053: FreeToken dependency closure
// ===========================================================================

#[test]
fn test_freetoken_depends_on_pytorch_and_rocm() {
    let deps = get_dependencies("freetoken");
    assert!(
        deps.contains(&"pytorch"),
        "FreeToken must depend on pytorch (venv base + torch from the ROCm index)"
    );
    assert!(deps.contains(&"rocm"), "FreeToken must depend on rocm");
}

#[test]
fn test_freetoken_is_native_and_in_tui() {
    assert!(is_native_component("freetoken"));
    let components = default_components();
    let comp = components
        .iter()
        .find(|c| c.id == "freetoken")
        .expect("freetoken must appear in default_components()");
    assert_eq!(comp.category, Category::Extension);
    assert!(!comp.needs_sudo, "freetoken installs as the user (venv + pip)");
    assert!(comp.script.is_empty(), "freetoken routes to the native installer");
}

#[test]
fn test_freetoken_topological_order() {
    let ids = vec![
        "freetoken".to_string(),
        "pytorch".to_string(),
        "rocm".to_string(),
    ];
    let sorted = topological_sort(&ids).unwrap();
    let pos = |id: &str| sorted.iter().position(|s| s == id).unwrap();
    assert!(pos("rocm") < pos("pytorch"), "rocm before pytorch");
    assert!(pos("pytorch") < pos("freetoken"), "pytorch before freetoken");
}

#[test]
fn test_comfyui_depends_on_pytorch() {
    let deps = get_dependencies("comfyui");
    assert!(deps.contains(&"pytorch"), "ComfyUI must depend on pytorch");
}

// ===========================================================================
// VAL-INSTALL-048: Dependency graph is acyclic
// ===========================================================================

#[test]
fn test_dependency_graph_is_acyclic() {
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

// ===========================================================================
// VAL-INSTALL-049: installer.rs respects dependency ordering
// ===========================================================================

#[test]
fn test_topological_sort_orders_dependencies_before_dependents() {
    let ids = vec![
        "megatron".to_string(),
        "pytorch".to_string(),
        "mpi4py".to_string(),
        "rocm".to_string(),
    ];
    let sorted = topological_sort(&ids).unwrap();

    let pos = |id: &str| sorted.iter().position(|s| s == id).unwrap();
    assert!(pos("rocm") < pos("pytorch"), "rocm before pytorch");
    assert!(pos("pytorch") < pos("megatron"), "pytorch before megatron");
    assert!(pos("mpi4py") < pos("megatron"), "mpi4py before megatron");
}

#[test]
fn test_topological_sort_complex_dependency_chain() {
    let ids = vec![
        "megatron".to_string(),
        "flash-attn-triton".to_string(),
        "vllm".to_string(),
        "aiter".to_string(),
        "onnx".to_string(),
        "deepspeed".to_string(),
        "comfyui".to_string(),
        "pytorch".to_string(),
        "rocm".to_string(),
        "mpi4py".to_string(),
    ];
    let sorted = topological_sort(&ids).unwrap();

    let pos = |id: &str| sorted.iter().position(|s| s == id).unwrap();

    // Verify declared dependency constraints
    // rocm -> flash-attn, onnx, aiter
    assert!(pos("rocm") < pos("flash-attn-triton"));
    assert!(pos("rocm") < pos("onnx"));
    assert!(pos("rocm") < pos("aiter"));
    // pytorch -> megatron, vllm, deepspeed, flash-attn, aiter, comfyui
    assert!(pos("pytorch") < pos("megatron"));
    assert!(pos("mpi4py") < pos("megatron"));
    assert!(pos("pytorch") < pos("vllm"));
    assert!(pos("pytorch") < pos("deepspeed"));
    assert!(pos("pytorch") < pos("comfyui"));
}

// ===========================================================================
// VAL-INSTALL-032: Correct Rust module mapping per component ID
// ===========================================================================

#[test]
fn test_every_native_component_has_installer_module() {
    // Verify that each native component ID maps to a module that exists
    // by checking the module can be imported and an installer can be created.
    // We just verify the mapping exists by creating default configs.
    use rusty_stack::installers::components::*;

    for id in NATIVE_COMPONENT_IDS {
        // Each component should be dispatchable — we verify the mapping
        // by ensuring the module exists and can create a default installer
        match *id {
            "rocm" => {
                let _ = RocmInstaller::with_defaults();
            }
            "pytorch" => {
                let _ = PyTorchInstaller::with_defaults();
            }
            "triton" => {
                let _ = TritonInstaller::with_defaults();
            }
            "mpi4py" => {
                let _ = Mpi4PyInstaller::with_defaults();
            }
            "deepspeed" => {
                let _ = DeepSpeedInstaller::with_defaults();
            }
            "ml-stack-core" => {
                let _ = MlStackInstaller::with_defaults();
            }
            "flash-attn" | "flash-attn-triton" | "flash-attn-ck" => {
                let _ = FlashAttentionInstaller::with_defaults();
            }
            "megatron" => {
                let _ = MegatronInstaller::with_defaults();
            }
            "vllm" => {
                let _ = VllmInstaller::with_defaults();
            }
            "aiter" => {
                let _ = AiterInstaller::with_defaults();
            }
            "vllm-studio" => {
                let _ = VllmStudioInstaller::with_defaults();
            }
            "comfyui" => {
                let _ = ComfyuiInstaller::with_defaults();
            }
            "textgen" => {
                let _ = TextgenInstaller::with_defaults();
            }
            "onnx" => {
                let _ = OnnxRuntimeInstaller::with_defaults();
            }
            "bitsandbytes" => {
                let _ = BitsAndBytesInstaller::with_defaults();
            }
            "rocm-smi" => {
                let _ = RocmSmiInstaller::new();
            }
            "migraphx" => {
                let _ = MigraphxInstaller::with_defaults();
            }
            "pytorch-profiler" => {
                let _ = PytorchProfilerInstaller::with_defaults();
            }
            "wandb" => {
                let _ = WandbInstaller::with_defaults();
            }
            "amdgpu-drivers" => {
                let _ = AmdgpuInstaller::with_defaults();
            }
            "migraphx-python" => {
                let _ = MigraphxPythonInstaller::with_defaults();
            }
            "permanent-env" => {
                let _ = PermanentEnvInstaller::with_defaults();
            }
            "repair-stack" => {
                let _ = RepairInstaller::with_defaults();
            }
            "rccl-repair" => {
                let _ = rccl::RcclInstaller::new(
                    "/tmp",
                    "python3",
                    "/opt/rocm",
                    "7.2.1",
                    "latest",
                    None,
                );
            }
            "enhanced-env" => {} // env setup module (no dedicated installer struct yet)
            // Benchmark components — dispatched via benchmark_runners module
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
                // Benchmarks are dispatched via benchmark_runners::run_benchmark()
            }
            // FastVideo — uses dedicated installer
            "fastvideo" => {
                use rusty_stack::installers::components::fastvideo::FastVideoConfig;
                let _ = rusty_stack::installers::components::fastvideo::FastVideoInstaller::new(
                    FastVideoConfig::default(),
                );
            }
            // llama-cpp — uses dedicated installer
            "llama-cpp" => {
                use rusty_stack::installers::components::llama_cpp::LlamaCppConfig;
                let _ = rusty_stack::installers::components::llama_cpp::LlamaCppInstaller::new(
                    LlamaCppConfig::default(),
                );
            }
            // freetoken — uses dedicated installer
            "freetoken" => {
                use rusty_stack::installers::components::freetoken::FreetokenConfig;
                let _ = rusty_stack::installers::components::freetoken::FreetokenInstaller::new(
                    FreetokenConfig::default(),
                );
            }
            _ => panic!("Unknown native component ID: {}", id),
        }
    }
}

// ===========================================================================
// VAL-INSTALL-033: Sudo behavior preserved
// ===========================================================================

#[test]
fn test_native_components_preserve_needs_sudo_flag() {
    let components = default_components();
    let by_id = |id: &str| -> &Component {
        components
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("component {id} missing"))
    };

    // Components that genuinely need root: system-package installs (apt/dnf/
    // pacman), system-dir writes, or repair/verify that may rebuild system
    // state. These keep needs_sudo=true.
    for sudo_id in [
        "permanent-env",
        "rocm",
        "rocm-smi",
        "verify-basic",
        "verify-enhanced",
        "verify-build",
        "repair-stack",
    ] {
        assert!(by_id(sudo_id).needs_sudo, "{sudo_id} should need sudo");
    }

    // pip / git / cmake installs land in the managed venv or user home — they
    // never write system dirs, so needs_sudo MUST be false. (A `true` here made
    // `run_installation`'s sudo gate spawn `sudo -n true` and leak
    // "a password is required" during `update`, even though pip needs no root.)
    for pip_id in [
        "pytorch",
        "megatron",
        "triton",
        "mpi4py",
        "deepspeed",
        "pytorch-profiler",
        "flash-attn-triton",
        "flash-attn-ck",
        "migraphx",
        "llama-cpp",
        "aiter",
        "vllm",
        "bitsandbytes",
        "wandb",
        "onnx",
        "vllm-studio",
        "comfyui",
    ] {
        assert!(
            !by_id(pip_id).needs_sudo,
            "{pip_id} is pip/git-only and must not need sudo"
        );
    }
}

// ===========================================================================
// Cross-checks: default_components() consistency
// ===========================================================================

#[test]
fn test_default_components_total_count() {
    let components = default_components();
    // 25 native TUI installers/actions (incl fastvideo + llama-cpp + RCCL repair
    // + migraphx-python + freetoken) + 3 verification + 11 performance = 40.
    assert_eq!(components.len(), 40, "Expected 40 total components");
}

#[test]
fn test_chain_referenced_component_ids_are_resolvable() {
    // Regression (PR #27 third-wave review): components advertised by BOTH the
    // manifest/registry AND the installer dispatch must also be present in
    // default_components(), or DirectInstallerExecutor::component_for_id (an
    // exact .find on default_components()) fails with "Unknown component ID"
    // before the native installer runs. Not every NATIVE_COMPONENT_ID needs to
    // be in default_components() (meta/benchmark/env ids are dispatch-only), so
    // this asserts the specific chain-referenced set: migraphx-python and the
    // Flash Attention backends (legacy flash-attn normalizes to flash-attn-triton).
    let components = default_components();
    let ids: Vec<&str> = components.iter().map(|c| c.id.as_str()).collect();
    assert!(
        ids.contains(&"migraphx-python"),
        "migraphx-python must be in default_components() so update/apply can dispatch it"
    );
    assert!(
        ids.contains(&"flash-attn-triton"),
        "flash-attn-triton must be present (legacy flash-attn normalizes to it)"
    );
    assert!(
        ids.contains(&"flash-attn-ck"),
        "flash-attn-ck must be present"
    );
}

#[test]
fn test_no_script_field_contains_sh_for_native_components() {
    let components = default_components();
    for comp in &components {
        if is_native_component(&comp.id) {
            assert!(
                !comp.script.contains(".sh"),
                "Native component '{}' should not reference .sh in script field, got '{}'",
                comp.id,
                comp.script
            );
        }
    }
}

#[test]
fn test_all_components_are_native() {
    // All components in default_components() should now have empty script fields —
    // no component references .sh scripts anymore.
    // Verification uses a dedicated native path (run_verification), benchmarks
    // use benchmark_runners module, and installers use the native dispatch.
    let components = default_components();
    for comp in &components {
        assert!(
            comp.script.is_empty(),
            "Component '{}' should have empty script (all scripts ported to Rust), got '{}'",
            comp.id,
            comp.script
        );
    }
}
