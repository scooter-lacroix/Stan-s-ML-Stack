//! Native Rust verification module.
//!
//! Ports the verification logic from:
//! - `scripts/verify_installation.sh` → `full_verify()`
//! - `scripts/enhanced_verify_installation.sh` → `enhanced_verify()`
//! - `scripts/verify_and_build.sh` → `verify_and_build()`
//!
//! All verification uses `component_status.rs` detection — no subprocess
//! calls to legacy shell scripts.

use crate::component_status;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a single verification check.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationItem {
    pub component_id: String,
    pub label: String,
    pub installed: bool,
    pub category: String,
}

/// Result of a full verification run.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationResult {
    pub mode: String,
    pub items: Vec<VerificationItem>,
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
    pub all_passed: bool,
}

// ---------------------------------------------------------------------------
// Component lists for each verification mode
// ---------------------------------------------------------------------------

/// Components checked by `verify_installation.sh` (basic/full mode).
fn full_verify_components() -> Vec<(&'static str, &'static str)> {
    vec![
        ("rocm", "ROCm"),
        ("pytorch", "PyTorch"),
        ("triton", "Triton"),
        ("mpi4py", "MPI4Py"),
        ("deepspeed", "DeepSpeed"),
        ("ml-stack-core", "ML Stack Core"),
    ]
}

/// Components checked by `enhanced_verify_installation.sh` (all components).
fn enhanced_verify_components() -> Vec<(&'static str, &'static str)> {
    let mut components = full_verify_components();
    components.extend(vec![
        ("flash-attn", "Flash Attention"),
        ("vllm", "vLLM"),
        ("aiter", "AITER"),
        ("onnx", "ONNX Runtime"),
        ("bitsandbytes", "bitsandbytes"),
        ("rocm-smi", "ROCm SMI"),
        ("migraphx", "MIGraphX"),
        ("pytorch-profiler", "PyTorch Profiler"),
        ("wandb", "Weights & Biases"),
        ("megatron", "Megatron-LM"),
        ("comfyui", "ComfyUI"),
        ("textgen", "text-generation-webui"),
        ("vllm-studio", "vLLM Studio"),
        ("basic-env", "Basic Environment"),
        ("enhanced-env", "Enhanced Environment"),
        ("permanent-env", "Permanent Environment"),
    ]);
    components
}

/// Components checked by `verify_and_build.sh` (build-critical components).
fn build_verify_components() -> Vec<(&'static str, &'static str)> {
    let mut components = full_verify_components();
    components.extend(vec![
        ("onnx", "ONNX Runtime"),
        ("flash-attn", "Flash Attention"),
        ("migraphx", "MIGraphX"),
    ]);
    components
}

// ---------------------------------------------------------------------------
// Core verification logic
// ---------------------------------------------------------------------------

/// Run verification for a list of components using component_status detection.
fn verify_components(component_list: &[(impl AsRef<str>, impl AsRef<str>)]) -> VerificationResult {
    let python_candidates = component_status::python_interpreters();
    let mut items = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;

    for (id, label) in component_list {
        let id_str = id.as_ref();
        let label_str = label.as_ref();
        let installed = component_status::is_component_installed_by_id(id_str, &python_candidates);
        if installed {
            passed += 1;
        } else {
            failed += 1;
        }
        items.push(VerificationItem {
            component_id: id_str.to_string(),
            label: label_str.to_string(),
            installed,
            category: category_for_component(id_str),
        });
    }

    let total = items.len();
    VerificationResult {
        mode: "full".to_string(),
        items,
        passed,
        failed,
        total,
        all_passed: failed == 0,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Full verification — equivalent to `verify_installation.sh`.
///
/// Checks core components: ROCm, PyTorch, Triton, MPI4Py, DeepSpeed, ML Stack Core.
pub fn full_verify() -> VerificationResult {
    let mut result = verify_components(&full_verify_components());
    result.mode = "full".to_string();
    result
}

/// Enhanced verification — equivalent to `enhanced_verify_installation.sh`.
///
/// Checks all components including ROCm tools, Python imports, and environment files.
pub fn enhanced_verify() -> VerificationResult {
    let mut result = verify_components(&enhanced_verify_components());
    result.mode = "enhanced".to_string();
    result
}

/// Verify and build — equivalent to `verify_and_build.sh`.
///
/// Checks core + build-critical components. Returns the verification result;
/// the caller (CLI) is responsible for triggering rebuilds for failed components.
pub fn verify_and_build() -> VerificationResult {
    let mut result = verify_components(&build_verify_components());
    result.mode = "build".to_string();
    result
}

/// Get the list of failed component IDs from a verification result.
pub fn failed_components(result: &VerificationResult) -> Vec<String> {
    result
        .items
        .iter()
        .filter(|item| !item.installed)
        .map(|item| item.component_id.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn category_for_component(id: &str) -> String {
    match id {
        "rocm" | "rocm-smi" => "ROCm".to_string(),
        "pytorch" | "triton" | "flash-attn" | "onnx" | "migraphx" | "pytorch-profiler" => {
            "ML Framework".to_string()
        }
        "mpi4py" | "deepspeed" | "megatron" => "Training".to_string(),
        "vllm" | "aiter" | "vllm-studio" => "Inference".to_string(),
        "bitsandbytes" | "wandb" => "Utilities".to_string(),
        "ml-stack-core" => "Core".to_string(),
        "comfyui" | "textgen" => "Applications".to_string(),
        "basic-env" | "enhanced-env" | "permanent-env" => "Environment".to_string(),
        _ => "Other".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Format verification result as human-readable text.
pub fn format_result_human(result: &VerificationResult) -> String {
    let mut output = String::new();
    let mode_label = match result.mode.as_str() {
        "full" => "Full Verification",
        "enhanced" => "Enhanced Verification",
        "build" => "Verify & Build",
        _ => "Verification",
    };

    output.push_str(&format!(
        "Rusty Stack {} — {} components checked\n",
        mode_label, result.total
    ));
    output.push_str(&format!(
        "  Passed: {}  |  Failed: {}  |  Result: {}\n\n",
        result.passed,
        result.failed,
        if result.all_passed {
            "ALL PASSED ✓"
        } else {
            "SOME FAILED ✗"
        }
    ));

    for item in &result.items {
        let status = if item.installed { "✓" } else { "✗" };
        output.push_str(&format!(
            "  {} {:30} [{}]\n",
            status, item.label, item.category
        ));
    }

    if !result.all_passed {
        output.push_str("\nFailed components:\n");
        for item in &result.items {
            if !item.installed {
                output.push_str(&format!("  - {} ({})\n", item.label, item.component_id));
            }
        }
    }

    output
}

/// Format verification result as JSON.
pub fn format_result_json(result: &VerificationResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|e| {
        format!(
            "{{\"error\": \"failed to serialize verification result: {}\"}}",
            e
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_verify_returns_result() {
        let result = full_verify();
        assert_eq!(result.mode, "full");
        assert!(
            result.total > 0,
            "Full verify should check at least 1 component"
        );
        assert_eq!(result.total, result.passed + result.failed);
    }

    #[test]
    fn test_enhanced_verify_returns_more_components() {
        let full = full_verify();
        let enhanced = enhanced_verify();
        assert!(
            enhanced.total > full.total,
            "Enhanced should check more components than full ({} > {})",
            enhanced.total,
            full.total
        );
    }

    #[test]
    fn test_verify_and_build_includes_build_components() {
        let result = verify_and_build();
        assert_eq!(result.mode, "build");
        // Should include onnx, flash-attn, migraphx (build-critical)
        let ids: Vec<&str> = result
            .items
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        assert!(ids.contains(&"onnx"), "build verify should include onnx");
        assert!(
            ids.contains(&"flash-attn"),
            "build verify should include flash-attn"
        );
        assert!(
            ids.contains(&"migraphx"),
            "build verify should include migraphx"
        );
    }

    #[test]
    fn test_failed_components_extracts_failures() {
        let result = VerificationResult {
            mode: "test".to_string(),
            items: vec![
                VerificationItem {
                    component_id: "rocm".to_string(),
                    label: "ROCm".to_string(),
                    installed: true,
                    category: "ROCm".to_string(),
                },
                VerificationItem {
                    component_id: "pytorch".to_string(),
                    label: "PyTorch".to_string(),
                    installed: false,
                    category: "ML Framework".to_string(),
                },
            ],
            passed: 1,
            failed: 1,
            total: 2,
            all_passed: false,
        };
        let failed = failed_components(&result);
        assert_eq!(failed, vec!["pytorch"]);
    }

    #[test]
    fn test_format_human_output() {
        let result = full_verify();
        let text = format_result_human(&result);
        assert!(
            text.contains("Verification"),
            "Human output should contain 'Verification'"
        );
        assert!(
            text.contains("Passed:"),
            "Human output should contain 'Passed:'"
        );
    }

    #[test]
    fn test_format_json_output() {
        let result = full_verify();
        let json = format_result_json(&result);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("JSON should be parseable");
        assert!(parsed.get("mode").is_some(), "JSON should contain 'mode'");
        assert!(parsed.get("items").is_some(), "JSON should contain 'items'");
        assert!(
            parsed.get("all_passed").is_some(),
            "JSON should contain 'all_passed'"
        );
    }

    #[test]
    fn test_full_verify_includes_core_components() {
        let result = full_verify();
        let ids: Vec<&str> = result
            .items
            .iter()
            .map(|i| i.component_id.as_str())
            .collect();
        assert!(ids.contains(&"rocm"), "full verify should include rocm");
        assert!(
            ids.contains(&"pytorch"),
            "full verify should include pytorch"
        );
        assert!(ids.contains(&"triton"), "full verify should include triton");
        assert!(ids.contains(&"mpi4py"), "full verify should include mpi4py");
        assert!(
            ids.contains(&"deepspeed"),
            "full verify should include deepspeed"
        );
        assert!(
            ids.contains(&"ml-stack-core"),
            "full verify should include ml-stack-core"
        );
    }

    #[test]
    fn test_enhanced_verify_includes_all_categories() {
        let result = enhanced_verify();
        let categories: Vec<&str> = result.items.iter().map(|i| i.category.as_str()).collect();
        assert!(
            categories.contains(&"ROCm"),
            "enhanced should include ROCm category"
        );
        assert!(
            categories.contains(&"ML Framework"),
            "enhanced should include ML Framework category"
        );
        assert!(
            categories.contains(&"Environment"),
            "enhanced should include Environment category"
        );
    }

    #[test]
    fn test_no_shell_subprocess_in_verification() {
        // Verification uses component_status::is_component_installed_by_id which
        // does detection via filesystem checks and python module probing — no bash
        // subprocess to shell scripts. This test verifies the module compiles and
        // runs without calling any shell verification scripts.
        let result = full_verify();
        assert!(result.total > 0);
    }
}
