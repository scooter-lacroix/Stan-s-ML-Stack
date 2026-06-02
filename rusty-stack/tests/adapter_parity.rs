//! Integration tests for adapter parity — behavior comparison between Rust and shell.
//!
//! These tests verify that:
//! - The adapter registry correctly routes to Rust or legacy adapters
//! - Output format matches between Rust and shell for migrated components
//! - Exit codes match between Rust and shell for success/failure/partial cases
//! - Unknown components are properly rejected
//!
//! # Validation Assertions
//!
//! - **VAL-MIGR-009**: Rust-registered adapter routes to Rust impl, no shell subprocess
//! - **VAL-MIGR-010**: No Rust adapter → falls back to install_*.sh with correct args
//! - **VAL-MIGR-011**: Unknown component ID returns Err(AdapterNotFound)
//! - **VAL-MIGR-012**: Output format matches shell for migrated components
//! - **VAL-MIGR-013**: Exit codes match shell for success/failure/partial cases

use rusty_stack::adapter::legacy_adapter::LegacyAdapter;
use rusty_stack::adapter::rust_adapter::{RustAdapter, ShellOutputFormat};
use rusty_stack::adapter::RegistryExecutor;
use rusty_stack::adapter::{Adapter, AdapterError, AdapterOutput, AdapterRegistry};
use rusty_stack::core::types::ExecutorKind;
use rusty_stack::orchestrator::apply::{ApplyEngine, ApplyExecutor};

use std::path::PathBuf;

// ===========================================================================
// Test Helpers
// ===========================================================================

/// Create a temp directory with mock install scripts for testing.
fn create_test_scripts_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    // Create mock install scripts that simulate real shell behavior
    std::fs::write(
        dir.path().join("install_pytorch_rocm.sh"),
        "#!/bin/bash\necho \"PyTorch $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_triton_multi.sh"),
        "#!/bin/bash\necho \"Triton $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_flash_attention_ck.sh"),
        "#!/bin/bash\necho \"Flash Attention $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_deepspeed.sh"),
        "#!/bin/bash\necho \"Error: DeepSpeed $1 installation failed\" >&2\nexit 1\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_vllm_multi.sh"),
        "#!/bin/bash\necho \"vLLM $1 partially installed\"\necho \"Warning: some deps missing\" >&2\nexit 2\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_migraphx_multi.sh"),
        "#!/bin/bash\necho \"MIGraphX $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_bitsandbytes_multi.sh"),
        "#!/bin/bash\necho \"bitsandbytes $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_mpi4py.sh"),
        "#!/bin/bash\necho \"MPI4Py $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_aiter.sh"),
        "#!/bin/bash\necho \"AITER $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_wandb.sh"),
        "#!/bin/bash\necho \"Weights & Biases $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_rocm.sh"),
        "#!/bin/bash\necho \"ROCm $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_onnxruntime_multi.sh"),
        // Note: the registry uses "build_onnxruntime_multi.sh" but we also create this one
        "#!/bin/bash\necho \"ONNX Runtime $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("build_onnxruntime_multi.sh"),
        "#!/bin/bash\necho \"ONNX Runtime $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_comfyui.sh"),
        "#!/bin/bash\necho \"ComfyUI $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_vllm_studio.sh"),
        "#!/bin/bash\necho \"vLLM Studio $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    std::fs::write(
        dir.path().join("install_textgen.sh"),
        "#!/bin/bash\necho \"text-generation-webui $1 installed successfully\"\nexit 0\n",
    )
    .unwrap();

    dir
}

/// Create a registry with both Rust (for pytorch) and legacy adapters.
fn create_hybrid_registry(scripts_dir: &PathBuf) -> AdapterRegistry {
    let mut registry = AdapterRegistry::with_legacy(scripts_dir);

    // Register a Rust adapter for pytorch that mimics shell output
    registry.register_rust(
        "pytorch",
        Box::new(RustAdapter::new(Box::new(|_id, version| {
            let output = ShellOutputFormat::success("pytorch", version);
            Ok((output, String::new()))
        }))),
    );

    // Register a Rust adapter for triton that mimics shell output
    registry.register_rust(
        "triton",
        Box::new(RustAdapter::new(Box::new(|_id, version| {
            let output = ShellOutputFormat::success("triton", version);
            Ok((output, String::new()))
        }))),
    );

    registry
}

// ===========================================================================
// VAL-MIGR-009: Rust-registered adapter routes to Rust impl
// ===========================================================================

#[test]
fn test_integration_rust_adapter_dispatches_without_subprocess() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // pytorch has a Rust adapter
    let result = registry.execute("pytorch", "2.5.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::Rust);
    assert_eq!(result.exit_code, 0);
    // The Rust adapter output should contain the display name
    assert!(result.stdout.contains("PyTorch"));
    assert!(result.stdout.contains("2.5.0"));
}

#[test]
fn test_integration_rust_adapter_output_matches_shell_format() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Execute pytorch via Rust adapter
    let rust_result = registry.execute("pytorch", "2.5.0").unwrap();
    assert_eq!(rust_result.executor_kind, ExecutorKind::Rust);

    // Execute pytorch directly via legacy adapter for comparison
    let legacy = LegacyAdapter::new(scripts_dir.path().to_path_buf());
    let legacy_result = legacy.execute("pytorch", "2.5.0").unwrap();

    // Both should report success
    assert_eq!(rust_result.exit_code, legacy_result.exit_code);
    assert!(rust_result.is_success());
    assert!(legacy_result.is_success());

    // Both should mention the component and version
    assert!(rust_result.stdout.contains("2.5.0"));
    assert!(legacy_result.stdout.contains("2.5.0"));
}

#[test]
fn test_integration_triton_rust_adapter() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("triton", "3.1.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::Rust);
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("Triton"));
    assert!(result.stdout.contains("3.1.0"));
}

// ===========================================================================
// VAL-MIGR-010: No Rust adapter → falls back to install_*.sh
// ===========================================================================

#[test]
fn test_integration_legacy_fallback_for_non_rust_component() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // flash-attn has no Rust adapter, should fall back to legacy
    let result = registry.execute("flash-attn", "2.6.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::LegacyScript);
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("Flash Attention"));
    assert!(result.stdout.contains("2.6.0"));
}

#[test]
fn test_integration_legacy_fallback_for_migraphx() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("migraphx", "1.2.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::LegacyScript);
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("MIGraphX"));
}

#[test]
fn test_integration_legacy_fallback_for_deepspeed() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("deepspeed", "0.14.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::LegacyScript);
    assert_eq!(result.exit_code, 1); // DeepSpeed script fails in our mock
    assert!(!result.is_success());
}

#[test]
fn test_integration_legacy_fallback_for_git_components() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // ComfyUI is a git-based component
    let result = registry.execute("comfyui", "latest").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::LegacyScript);
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("ComfyUI"));
}

// ===========================================================================
// VAL-MIGR-011: Unknown component ID returns Err(AdapterNotFound)
// ===========================================================================

#[test]
fn test_integration_unknown_component_returns_not_found() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("nonexistent-tool", "1.0.0");
    assert!(matches!(result, Err(AdapterError::AdapterNotFound { .. })));
}

#[test]
fn test_integration_multiple_unknown_components_rejected() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let unknown_ids = [
        "fake-component",
        "does-not-exist",
        "imaginary-tool",
        "xyz-123",
        "not-real",
    ];

    for id in &unknown_ids {
        let result = registry.execute(id, "1.0.0");
        assert!(
            matches!(result, Err(AdapterError::AdapterNotFound { .. })),
            "Expected AdapterNotFound for '{id}'"
        );
    }
}

#[test]
fn test_integration_empty_id_rejected() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("", "1.0.0");
    assert!(matches!(result, Err(AdapterError::AdapterNotFound { .. })));
}

// ===========================================================================
// VAL-MIGR-012: Output format matches shell for migrated components
// ===========================================================================

#[test]
fn test_integration_output_format_parity_pytorch() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Execute via registry (Rust adapter)
    let rust_result = registry.execute("pytorch", "2.5.0").unwrap();

    // Execute directly via legacy
    let legacy = LegacyAdapter::new(scripts_dir.path().to_path_buf());
    let legacy_result = legacy.execute("pytorch", "2.5.0").unwrap();

    // Both should succeed
    assert!(rust_result.is_success());
    assert!(legacy_result.is_success());

    // Both should contain the component name and version
    assert!(rust_result.stdout.contains("PyTorch"));
    assert!(legacy_result.stdout.contains("PyTorch"));
    assert!(rust_result.stdout.contains("2.5.0"));
    assert!(legacy_result.stdout.contains("2.5.0"));
}

#[test]
fn test_integration_output_format_parity_triton() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let rust_result = registry.execute("triton", "3.1.0").unwrap();

    let legacy = LegacyAdapter::new(scripts_dir.path().to_path_buf());
    let legacy_result = legacy.execute("triton", "3.1.0").unwrap();

    assert!(rust_result.is_success());
    assert!(legacy_result.is_success());
    assert!(rust_result.stdout.contains("Triton"));
    assert!(legacy_result.stdout.contains("Triton"));
    assert!(rust_result.stdout.contains("3.1.0"));
    assert!(legacy_result.stdout.contains("3.1.0"));
}

#[test]
fn test_integration_output_format_parity_flash_attn() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // flash-attn uses legacy adapter
    let result = registry.execute("flash-attn", "2.6.0").unwrap();
    assert_eq!(result.executor_kind, ExecutorKind::LegacyScript);
    assert!(result.stdout.contains("Flash Attention"));
    assert!(result.stdout.contains("2.6.0"));
}

// ===========================================================================
// VAL-MIGR-013: Exit codes match shell for success/failure/partial
// ===========================================================================

#[test]
fn test_integration_exit_code_parity_success() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Success case: pytorch (Rust) and flash-attn (legacy)
    let rust_result = registry.execute("pytorch", "2.5.0").unwrap();
    let legacy_result = registry.execute("flash-attn", "2.6.0").unwrap();

    assert_eq!(rust_result.exit_code, 0);
    assert_eq!(legacy_result.exit_code, 0);
    assert!(rust_result.is_success());
    assert!(legacy_result.is_success());
}

#[test]
fn test_integration_exit_code_parity_failure() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Register a failing Rust adapter
    let mut registry = registry;
    registry.register_rust(
        "fail-rust",
        Box::new(RustAdapter::new(Box::new(|_id, _version| {
            Err((1, String::new(), "installation failed".to_string()))
        }))),
    );

    // Rust failure
    let rust_result = registry.execute("fail-rust", "1.0.0").unwrap();
    assert_eq!(rust_result.exit_code, 1);

    // Legacy failure (deepspeed mock script exits with 1)
    let legacy_result = registry.execute("deepspeed", "0.14.0").unwrap();
    assert_eq!(legacy_result.exit_code, 1);
}

#[test]
fn test_integration_exit_code_parity_partial() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Register a partial-failure Rust adapter
    let mut registry = registry;
    registry.register_rust(
        "partial-rust",
        Box::new(RustAdapter::new(Box::new(|_id, _version| {
            Err((2, "partial output".to_string(), "some warnings".to_string()))
        }))),
    );

    // Rust partial failure
    let rust_result = registry.execute("partial-rust", "1.0.0").unwrap();
    assert_eq!(rust_result.exit_code, 2);
    assert!(!rust_result.is_success());

    // Legacy partial failure (vllm mock script exits with 2)
    let legacy_result = registry.execute("vllm", "0.6.0").unwrap();
    assert_eq!(legacy_result.exit_code, 2);
    assert!(!legacy_result.is_success());
}

// ===========================================================================
// RegistryExecutor integration with ApplyEngine
// ===========================================================================

#[test]
fn test_integration_registry_executor_with_apply_engine() {
    let scripts_dir = create_test_scripts_dir();
    let mut registry = AdapterRegistry::with_legacy(scripts_dir.path().to_owned());

    // Register Rust adapters
    registry.register_rust(
        "pytorch",
        Box::new(RustAdapter::simple_success("rust-pytorch")),
    );
    registry.register_rust(
        "triton",
        Box::new(RustAdapter::simple_success("rust-triton")),
    );

    let executor = RegistryExecutor::new(registry);
    let engine = ApplyEngine::new(executor);

    // The engine should be able to use the registry executor
    assert!(engine.apply(&[], &Default::default()).success.is_empty());
}

#[test]
fn test_integration_registry_executor_reports_failure() {
    let mut registry = AdapterRegistry::new();
    registry.register_rust(
        "fail-component",
        Box::new(RustAdapter::always_fails(1, "test failure")),
    );

    let executor = RegistryExecutor::new(registry);

    let result = executor.apply_component("fail-component", "1.0.0");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exit code 1"));
}

// ===========================================================================
// Full registry with all 17 known components
// ===========================================================================

#[test]
fn test_integration_all_known_components_route_correctly() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // Components with Rust adapters
    assert_eq!(
        registry.executor_kind("pytorch").unwrap(),
        ExecutorKind::Rust
    );
    assert_eq!(
        registry.executor_kind("triton").unwrap(),
        ExecutorKind::Rust
    );

    // Components with legacy adapters (no Rust adapter)
    assert_eq!(
        registry.executor_kind("flash-attn").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("deepspeed").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("migraphx").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("onnx").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("bitsandbytes").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("mpi4py").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("aiter").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("wandb").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("vllm").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("comfyui").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("vllm-studio").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("textgen").unwrap(),
        ExecutorKind::LegacyScript
    );
    assert_eq!(
        registry.executor_kind("rocm").unwrap(),
        ExecutorKind::LegacyScript
    );
}

#[test]
fn test_integration_components_without_installer_scripts() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    // rocm-smi and permanent-env don't have installer scripts
    assert!(registry.executor_kind("rocm-smi").is_err());
    assert!(registry.executor_kind("permanent-env").is_err());
}

// ===========================================================================
// AdapterOutput serialization for machine-readable output
// ===========================================================================

#[test]
fn test_integration_adapter_output_json_serialization() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("pytorch", "2.5.0").unwrap();

    // AdapterOutput should serialize to JSON
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("pytorch"));
    assert!(json.contains("Rust"));

    // Should roundtrip
    let back: AdapterOutput = serde_json::from_str(&json).unwrap();
    assert_eq!(result, back);
}

#[test]
fn test_integration_adapter_error_json_serialization() {
    let scripts_dir = create_test_scripts_dir();
    let registry = create_hybrid_registry(&scripts_dir.path().to_path_buf());

    let result = registry.execute("nonexistent", "1.0.0");
    let err = result.unwrap_err();

    // AdapterError should serialize to JSON
    let json = serde_json::to_string(&err).unwrap();
    assert!(json.contains("nonexistent"));

    // Should roundtrip
    let back: AdapterError = serde_json::from_str(&json).unwrap();
    assert_eq!(err, back);
}

// ===========================================================================
// Shell script argument forwarding verification
// ===========================================================================

#[test]
fn test_integration_legacy_adapter_forwards_version_to_script() {
    let dir = tempfile::tempdir().unwrap();

    // Create a script that echoes its arguments
    std::fs::write(
        dir.path().join("install_pytorch_rocm.sh"),
        "#!/bin/bash\necho \"args: $@\"\nexit 0\n",
    )
    .unwrap();

    let registry = AdapterRegistry::with_legacy(dir.path().to_owned());
    let result = registry.execute("pytorch", "2.5.0").unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("2.5.0"));
    // The version should be forwarded as the first argument
    assert!(result.stdout.contains("args: 2.5.0"));
}

#[test]
fn test_integration_legacy_adapter_empty_version_no_extra_arg() {
    let dir = tempfile::tempdir().unwrap();

    // Create a script that echoes its argument count
    std::fs::write(
        dir.path().join("install_pytorch_rocm.sh"),
        "#!/bin/bash\necho \"argc: $#\"\nexit 0\n",
    )
    .unwrap();

    let registry = AdapterRegistry::with_legacy(dir.path().to_owned());
    let result = registry.execute("pytorch", "").unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("argc: 0"));
}
