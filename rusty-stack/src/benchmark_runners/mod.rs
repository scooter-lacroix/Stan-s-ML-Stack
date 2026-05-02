//! Native Rust benchmark runner module.
//!
//! Dispatches to the existing `benchmarks/` module functions.
//! Provides the CLI-facing interface for running individual benchmarks,
//! the full suite, and JSON output.
//!
//! Ports functionality from:
//! - `scripts/run_rocm_benchmarks.sh` → ROCm benchmarks
//! - `scripts/run_all_benchmarks_suite.sh` → full suite
//! - `scripts/run_benchmarks.sh` → individual benchmark dispatch

use crate::benchmarks;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Combined benchmark result for CLI output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkOutput {
    pub name: String,
    pub success: bool,
    pub execution_time_ms: u128,
    pub results: serde_json::Value,
    pub errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Benchmark name registry
// ---------------------------------------------------------------------------

/// Returns all available benchmark names in canonical order.
pub fn available_benchmarks() -> Vec<&'static str> {
    vec![
        "gpu-capability",
        "memory-bandwidth",
        "tensor-core",
        "gemm",
        "pytorch",
        "flash-attention",
        "vllm",
        "deepspeed",
        "megatron",
        "all-pre",
        "all",
    ]
}

/// Returns true if the benchmark name is recognized.
pub fn is_valid_benchmark(name: &str) -> bool {
    available_benchmarks().contains(&name)
}

// ---------------------------------------------------------------------------
// Benchmark dispatch
// ---------------------------------------------------------------------------

/// Run a single named benchmark, dispatching to the correct benchmarks:: function.
pub fn run_benchmark(name: &str) -> Result<BenchmarkOutput, String> {
    match name {
        "gpu-capability" => Ok(convert_result(
            name,
            benchmarks::run_gpu_capability_benchmark(),
        )),
        "memory-bandwidth" => Ok(convert_result(
            name,
            benchmarks::run_memory_bandwidth_benchmark(),
        )),
        "tensor-core" => Ok(convert_result(
            name,
            benchmarks::run_tensor_core_benchmark(),
        )),
        "gemm" => Ok(convert_result(name, benchmarks::run_gemm_benchmark())),
        "pytorch" => Ok(convert_result(name, benchmarks::run_pytorch_benchmark())),
        "flash-attention" => Ok(convert_result(
            name,
            benchmarks::run_flash_attention_benchmark(),
        )),
        "vllm" => Ok(convert_result(name, benchmarks::run_vllm_benchmark())),
        "deepspeed" => Ok(convert_result(name, benchmarks::run_deepspeed_benchmark())),
        "megatron" => Ok(convert_result(name, benchmarks::run_megatron_benchmark())),
        "all-pre" => Ok(run_all_pre()),
        "all" => Ok(run_all()),
        _ => Err(format!("Unknown benchmark: {}", name)),
    }
}

/// Run all pre-installation benchmarks (gpu-capability, memory-bandwidth, tensor-core).
fn run_all_pre() -> BenchmarkOutput {
    let mut output = BenchmarkOutput {
        name: "all-pre".to_string(),
        success: true,
        execution_time_ms: 0,
        results: serde_json::Value::Object(serde_json::Map::new()),
        errors: Vec::new(),
    };

    let mut combined = serde_json::Value::Object(serde_json::Map::new());

    collect_benchmark(
        &mut combined,
        "gpu_capability",
        &mut output,
        benchmarks::run_gpu_capability_benchmark(),
        true,
    );
    collect_benchmark(
        &mut combined,
        "memory_bandwidth",
        &mut output,
        benchmarks::run_memory_bandwidth_benchmark(),
        true,
    );
    collect_benchmark(
        &mut combined,
        "tensor_core",
        &mut output,
        benchmarks::run_tensor_core_benchmark(),
        true,
    );

    output.results = combined;
    output
}

/// Run the full benchmark suite (all benchmarks).
fn run_all() -> BenchmarkOutput {
    let mut output = BenchmarkOutput {
        name: "all".to_string(),
        success: true,
        execution_time_ms: 0,
        results: serde_json::Value::Object(serde_json::Map::new()),
        errors: Vec::new(),
    };

    let mut combined = serde_json::Value::Object(serde_json::Map::new());

    // Pre-installation benchmarks (required)
    collect_benchmark(
        &mut combined,
        "gpu_capability",
        &mut output,
        benchmarks::run_gpu_capability_benchmark(),
        true,
    );
    collect_benchmark(
        &mut combined,
        "memory_bandwidth",
        &mut output,
        benchmarks::run_memory_bandwidth_benchmark(),
        true,
    );
    collect_benchmark(
        &mut combined,
        "tensor_core",
        &mut output,
        benchmarks::run_tensor_core_benchmark(),
        true,
    );

    // GPU performance (required)
    collect_benchmark(
        &mut combined,
        "gemm",
        &mut output,
        benchmarks::run_gemm_benchmark(),
        true,
    );

    // Component benchmarks (optional — failures don't fail the suite)
    collect_benchmark(
        &mut combined,
        "pytorch",
        &mut output,
        benchmarks::run_pytorch_benchmark(),
        false,
    );
    collect_benchmark(
        &mut combined,
        "flash_attention",
        &mut output,
        benchmarks::run_flash_attention_benchmark(),
        false,
    );
    collect_benchmark(
        &mut combined,
        "vllm",
        &mut output,
        benchmarks::run_vllm_benchmark(),
        false,
    );
    collect_benchmark(
        &mut combined,
        "deepspeed",
        &mut output,
        benchmarks::run_deepspeed_benchmark(),
        false,
    );
    collect_benchmark(
        &mut combined,
        "megatron",
        &mut output,
        benchmarks::run_megatron_benchmark(),
        false,
    );

    output.results = combined;
    output
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn convert_result(name: &str, result: benchmarks::BenchmarkResult) -> BenchmarkOutput {
    BenchmarkOutput {
        name: name.to_string(),
        success: result.success,
        execution_time_ms: result.execution_time_ms,
        results: serde_json::to_value(&result).unwrap_or_default(),
        errors: result.errors.clone(),
    }
}

fn collect_benchmark(
    combined: &mut serde_json::Value,
    key: &str,
    output: &mut BenchmarkOutput,
    result: benchmarks::BenchmarkResult,
    required: bool,
) {
    if !result.success {
        if required {
            output.success = false;
        }
        if result.errors.is_empty() {
            output.errors.push(format!(
                "{}: benchmark failed ({})",
                key,
                if required { "required" } else { "optional" }
            ));
        }
    }
    output.execution_time_ms += result.execution_time_ms;
    for err in &result.errors {
        output.errors.push(format!(
            "{} ({}): {}",
            key,
            if required { "required" } else { "optional" },
            err
        ));
    }

    if let serde_json::Value::Object(map) = combined {
        map.insert(
            key.to_string(),
            serde_json::to_value(&result).unwrap_or_default(),
        );
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Format benchmark output as human-readable text.
pub fn format_output_human(output: &BenchmarkOutput) -> String {
    let mut text = String::new();
    text.push_str(&format!("Benchmark: {}\n", output.name));
    text.push_str(&format!(
        "Status: {}\n",
        if output.success { "SUCCESS" } else { "FAILED" }
    ));
    text.push_str(&format!("Time: {} ms\n", output.execution_time_ms));

    if !output.errors.is_empty() {
        text.push_str("Errors:\n");
        for e in &output.errors {
            text.push_str(&format!("  - {}\n", e));
        }
    }

    if output.success {
        if let Some(map) = output.results.as_object() {
            text.push_str("\nResults:\n");
            for (key, value) in map {
                if let Some(obj) = value.as_object() {
                    let status = obj
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .map(|ok| if ok { "SUCCESS" } else { "FAILED" })
                        .unwrap_or("UNKNOWN");
                    let time_ms = obj
                        .get("execution_time_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    text.push_str(&format!("  {} [{} | {} ms]\n", key, status, time_ms));
                    if let Some(metrics) = obj.get("metrics").and_then(|m| m.as_object()) {
                        for (metric_key, metric_val) in metrics.iter().take(4) {
                            text.push_str(&format!("    - {}: {}\n", metric_key, metric_val));
                        }
                        if metrics.len() > 4 {
                            text.push_str(&format!(
                                "    - ... {} more metrics\n",
                                metrics.len() - 4
                            ));
                        }
                    }
                } else {
                    text.push_str(&format!("  {}: {}\n", key, value));
                }
            }
        }
    }

    text
}

/// Format benchmark output as JSON.
pub fn format_output_json(output: &BenchmarkOutput) -> String {
    serde_json::to_string_pretty(output)
        .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {}\"}}", e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_benchmarks_contains_known_names() {
        let names = available_benchmarks();
        assert!(names.contains(&"gpu-capability"));
        assert!(names.contains(&"memory-bandwidth"));
        assert!(names.contains(&"tensor-core"));
        assert!(names.contains(&"gemm"));
        assert!(names.contains(&"pytorch"));
        assert!(names.contains(&"flash-attention"));
        assert!(names.contains(&"vllm"));
        assert!(names.contains(&"deepspeed"));
        assert!(names.contains(&"megatron"));
        assert!(names.contains(&"all-pre"));
        assert!(names.contains(&"all"));
    }

    #[test]
    fn test_is_valid_benchmark() {
        assert!(is_valid_benchmark("gpu-capability"));
        assert!(is_valid_benchmark("all"));
        assert!(!is_valid_benchmark("nonexistent"));
        assert!(!is_valid_benchmark(""));
    }

    #[test]
    fn test_run_unknown_benchmark_returns_error() {
        let result = run_benchmark("nonexistent-xyz");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("Unknown benchmark"),
            "Error should mention 'Unknown benchmark'"
        );
    }

    #[test]
    fn test_run_gpu_capability_dispatches() {
        // This will likely fail on a system without ROCm, but should not panic
        let result = run_benchmark("gpu-capability");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.name, "gpu-capability");
    }

    #[test]
    fn test_run_all_pre_dispatches() {
        let result = run_benchmark("all-pre");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.name, "all-pre");
    }

    #[test]
    fn test_format_human_output() {
        let output = BenchmarkOutput {
            name: "test".to_string(),
            success: true,
            execution_time_ms: 100,
            results: serde_json::Value::Object(serde_json::Map::new()),
            errors: vec![],
        };
        let text = format_output_human(&output);
        assert!(text.contains("test"));
        assert!(text.contains("SUCCESS"));
    }

    #[test]
    fn test_format_json_output() {
        let output = BenchmarkOutput {
            name: "test".to_string(),
            success: true,
            execution_time_ms: 100,
            results: serde_json::Value::Object(serde_json::Map::new()),
            errors: vec![],
        };
        let json = format_output_json(&output);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should be valid JSON");
        assert_eq!(parsed["name"], "test");
        assert_eq!(parsed["success"], true);
    }

    #[test]
    fn test_benchmark_output_serialization_roundtrip() {
        let output = BenchmarkOutput {
            name: "gpu-capability".to_string(),
            success: false,
            execution_time_ms: 42,
            results: serde_json::json!({"key": "value"}),
            errors: vec!["test error".to_string()],
        };
        let json = serde_json::to_string(&output).unwrap();
        let parsed: BenchmarkOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "gpu-capability");
        assert!(!parsed.success);
        assert_eq!(parsed.execution_time_ms, 42);
        assert_eq!(parsed.errors.len(), 1);
    }
}
