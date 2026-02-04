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
        map.insert(key.to_string(), serde_json::to_value(result).unwrap_or_default());
    }
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
        "all-pre" => {
            let mut all_results = Value::Object(serde_json::Map::new());
            assign_value(
                &mut all_results,
                "gpu_capability",
                benchmarks::run_gpu_capability_benchmark(),
            );
            assign_value(
                &mut all_results,
                "memory_bandwidth",
                benchmarks::run_memory_bandwidth_benchmark(),
            );
            assign_value(
                &mut all_results,
                "tensor_core",
                benchmarks::run_tensor_core_benchmark(),
            );
            output.success = true;
            output.results = all_results;
        }
        "all" => {
            let mut all_results = Value::Object(serde_json::Map::new());
            assign_value(
                &mut all_results,
                "gpu_capability",
                benchmarks::run_gpu_capability_benchmark(),
            );
            assign_value(
                &mut all_results,
                "memory_bandwidth",
                benchmarks::run_memory_bandwidth_benchmark(),
            );
            assign_value(
                &mut all_results,
                "tensor_core",
                benchmarks::run_tensor_core_benchmark(),
            );
            assign_value(&mut all_results, "gemm", benchmarks::run_gemm_benchmark());
            assign_value(&mut all_results, "pytorch", benchmarks::run_pytorch_benchmark());
            assign_value(
                &mut all_results,
                "flash_attention",
                benchmarks::run_flash_attention_benchmark(),
            );
            assign_value(&mut all_results, "vllm", benchmarks::run_vllm_benchmark());
            assign_value(&mut all_results, "deepspeed", benchmarks::run_deepspeed_benchmark());
            output.success = true;
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
    println!();
    println!("Combined Benchmarks:");
    println!("  all-pre              - All pre-installation benchmarks");
    println!("  all                  - All benchmarks");
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
                    println!("  {}: {}", key, value);
                }
            }
        }
    }

    if !result.success {
        process::exit(1);
    }
}
