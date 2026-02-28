//! Benchmark Runner Binary
//! Provides CLI access to all benchmarks and JSON output for scripts/TUI.

use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::env;
use std::process;

use rusty_stack::benchmarks;

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkOutput {
    name: String,
    success: bool,
    execution_time_ms: u128,
    results: Value,
    errors: Vec<String>,
}

fn assign(output: &mut BenchmarkOutput, result: benchmarks::BenchmarkResult) {
    output.success = result.success;
    output.execution_time_ms = result.execution_time_ms;
    output.errors = result.errors.clone();
    output.results = serde_json::to_value(result).unwrap_or_default();
}

fn assign_value(target: &mut Value, key: &str, result: benchmarks::BenchmarkResult) {
    if let Value::Object(map) = target {
        map.insert(
            key.to_string(),
            serde_json::to_value(result).unwrap_or_default(),
        );
    }
}

fn run_and_collect(
    all_results: &mut Value,
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
    assign_value(all_results, key, result);
}

fn run_benchmark(name: &str) -> BenchmarkOutput {
    let mut output = BenchmarkOutput {
        name: name.to_string(),
        success: false,
        execution_time_ms: 0,
        results: Value::Object(serde_json::Map::new()),
        errors: Vec::new(),
    };

    match name {
        "gpu-capability" => assign(&mut output, benchmarks::run_gpu_capability_benchmark()),
        "memory-bandwidth" => assign(&mut output, benchmarks::run_memory_bandwidth_benchmark()),
        "tensor-core" => assign(&mut output, benchmarks::run_tensor_core_benchmark()),
        "gemm" => assign(&mut output, benchmarks::run_gemm_benchmark()),
        "pytorch" => assign(&mut output, benchmarks::run_pytorch_benchmark()),
        "flash-attention" => assign(&mut output, benchmarks::run_flash_attention_benchmark()),
        "vllm" => assign(&mut output, benchmarks::run_vllm_benchmark()),
        "deepspeed" => assign(&mut output, benchmarks::run_deepspeed_benchmark()),
        "megatron" => assign(&mut output, benchmarks::run_megatron_benchmark()),
        "all-pre" => {
            let mut all_results = Value::Object(serde_json::Map::new());
            output.success = true;
            run_and_collect(
                &mut all_results,
                "gpu_capability",
                &mut output,
                benchmarks::run_gpu_capability_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "memory_bandwidth",
                &mut output,
                benchmarks::run_memory_bandwidth_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "tensor_core",
                &mut output,
                benchmarks::run_tensor_core_benchmark(),
                true,
            );
            output.results = all_results;
        }
        "all" => {
            let mut all_results = Value::Object(serde_json::Map::new());
            output.success = true;
            run_and_collect(
                &mut all_results,
                "gpu_capability",
                &mut output,
                benchmarks::run_gpu_capability_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "memory_bandwidth",
                &mut output,
                benchmarks::run_memory_bandwidth_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "tensor_core",
                &mut output,
                benchmarks::run_tensor_core_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "gemm",
                &mut output,
                benchmarks::run_gemm_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "pytorch",
                &mut output,
                benchmarks::run_pytorch_benchmark(),
                true,
            );
            run_and_collect(
                &mut all_results,
                "flash_attention",
                &mut output,
                benchmarks::run_flash_attention_benchmark(),
                false,
            );
            run_and_collect(
                &mut all_results,
                "vllm",
                &mut output,
                benchmarks::run_vllm_benchmark(),
                false,
            );
            run_and_collect(
                &mut all_results,
                "deepspeed",
                &mut output,
                benchmarks::run_deepspeed_benchmark(),
                false,
            );
            run_and_collect(
                &mut all_results,
                "megatron",
                &mut output,
                benchmarks::run_megatron_benchmark(),
                false,
            );
            output.results = all_results;
        }
        _ => {
            output.errors.push(format!("Unknown benchmark: {}", name));
        }
    }

    output
}

fn print_help() {
    println!("Rusty-Stack Benchmark Runner");
    println!();
    println!("Usage: rusty-stack-bench <benchmark> [options]");
    println!();
    println!("Pre-installation Benchmarks:");
    println!("  gpu-capability       - GPU capability detection");
    println!("  memory-bandwidth     - HBM memory bandwidth test");
    println!("  tensor-core          - Tensor Core performance");
    println!();
    println!("GPU Performance:");
    println!("  gemm                 - Matrix multiplication benchmark");
    println!();
    println!("Component Benchmarks:");
    println!("  pytorch              - PyTorch performance tests");
    println!("  flash-attention      - Flash Attention vs standard");
    println!("  vllm                 - vLLM throughput benchmark");
    println!("  deepspeed            - DeepSpeed ZeRO performance");
    println!("  megatron             - Megatron-LM throughput/import benchmark");
    println!();
    println!("Combined Benchmarks:");
    println!("  all-pre              - All pre-installation benchmarks");
    println!("  all                  - All benchmarks (optional component failures are non-fatal)");
    println!();
    println!("Options:");
    println!("  --json               - Output in JSON format");
    println!("  --help               - Show this help message");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" {
        print_help();
        process::exit(0);
    }

    let benchmark = &args[1];
    let json_output = args.contains(&"--json".to_string());

    let result = run_benchmark(benchmark);

    if json_output {
        let json = serde_json::to_string_pretty(&result).unwrap_or_default();
        println!("{}", json);
    } else {
        println!("Benchmark: {}", result.name);
        println!(
            "Status: {}",
            if result.success { "SUCCESS" } else { "FAILED" }
        );
        println!("Time: {} ms", result.execution_time_ms);

        if !result.errors.is_empty() {
            println!("Errors:");
            for e in &result.errors {
                println!("  - {}", e);
            }
        }

        if result.success {
            if let Some(map) = result.results.as_object() {
                println!("\nResults:");
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
                        println!("  {} [{} | {} ms]", key, status, time_ms);
                        if let Some(metrics) = obj.get("metrics").and_then(|m| m.as_object()) {
                            for (metric_key, metric_val) in metrics.iter().take(4) {
                                println!("    - {}: {}", metric_key, metric_val);
                            }
                            if metrics.len() > 4 {
                                println!("    - ... {} more metrics", metrics.len() - 4);
                            }
                        }
                    } else {
                        println!("  {}: {}", key, value);
                    }
                }
            }
        }
    }

    if !result.success {
        process::exit(1);
    }
}
