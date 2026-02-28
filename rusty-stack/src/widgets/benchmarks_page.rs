//! Benchmark performance page for Rusty-Stack TUI
//! Renders real benchmark results with ratatui charts and summaries.

use crate::benchmark_logs::{
    benchmark_log_directories, collect_matching_logs, extract_benchmark_json_string,
    find_latest_log_in_dirs, is_json_log,
};
use chrono::Local;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Tabs},
    Frame,
};
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const STALE_FULL_BENCHMARK_WINDOW: Duration = Duration::from_secs(15 * 60);
const STALE_ERROR_WINDOW: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub gpus: Vec<GpuInfo>,
    pub memory_bandwidth: Option<MemoryBandwidthData>,
    pub tensor_core: Option<TensorCoreData>,
    pub pytorch: Option<PyTorchData>,
    pub flash_attention: Option<FlashAttentionData>,
    pub vllm: Option<VllmData>,
    pub deepspeed: Option<DeepspeedData>,
    pub megatron: Option<MegatronData>,
    pub errors: Vec<String>,
    pub baseline: Option<Box<BenchmarkResults>>,
}

#[derive(Debug, Clone)]
pub struct DeepspeedData {
    pub throughput_samples_per_sec: f64,
    pub avg_latency_ms: f64,
    pub stage: u32,
    pub accelerator: String,
    pub samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MegatronData {
    pub throughput_samples_per_sec: f64,
    pub avg_latency_ms: f64,
    pub backend: String,
    pub samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: usize,
    pub model: String,
    pub vram_gb: f64,
    pub compute_units: u32,
    pub tensor_cores: bool,
    pub temperature_c: Option<f64>,
    pub power_watts: Option<f64>,
    pub utilization_percent: Option<f64>,
    pub memory_percent: Option<f64>,
    pub sclk_mhz: Option<u32>,
    pub mclk_mhz: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct MemoryBandwidthData {
    pub hbm_peak_gb_s: f64,
    pub system_peak_gb_s: f64,
    pub hbm_ratio: f64,
    pub hbm_samples: Vec<f64>,
    pub system_samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TensorCoreData {
    pub fp16_tflops: f64,
    pub bf16_tflops: f64,
    pub fp32_tflops: f64,
    pub fp16_samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PyTorchData {
    pub gemm_gflops: f64,
    pub convolution_gflops: f64,
    pub autograd_overhead_percent: f64,
    pub gemm_samples: Vec<f64>,
    pub conv_samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FlashAttentionData {
    pub standard_attention_speed: f64,
    pub flash_attention_speed: f64,
    pub speedup: f64,
    pub memory_savings_gb: f64,
    pub standard_samples: Vec<f64>,
    pub flash_samples: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VllmData {
    pub throughput_tokens_per_sec: f64,
    pub latency_ms: f64,
    pub model: String,
    pub model_format: String,
    pub throughput_samples: Vec<f64>,
}

/// Load benchmark results from all available log files
pub fn load_benchmark_results() -> BenchmarkResults {
    let log_dirs = benchmark_log_directories();

    if log_dirs.is_empty() {
        return BenchmarkResults::empty();
    }

    let mut results = BenchmarkResults::empty();
    let mut selected_logs: Vec<(&str, PathBuf, SystemTime)> = Vec::new();

    // primary JSON sources written by the wrapper scripts
    for pattern in &[
        "rocm_benchmarks",
        "gpu_memory_bandwidth",
        "pytorch_performance",
        "mlperf_inference",
        "vllm_benchmarks",
        "deepspeed_benchmarks",
        "megatron_benchmarks",
        "full_benchmarks",
    ] {
        if let Some(log_path) = find_latest_log_in_dirs(&log_dirs, pattern) {
            let modified = fs::metadata(&log_path)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            selected_logs.push((pattern, log_path, modified));
        }
    }

    let newest_component_log = selected_logs
        .iter()
        .filter(|(pattern, _, _)| *pattern != "full_benchmarks")
        .map(|(_, _, modified)| *modified)
        .max();

    for (pattern, log_path, modified) in selected_logs {
        if pattern == "full_benchmarks" {
            if let Some(component_mtime) = newest_component_log {
                let is_stale = component_mtime
                    .duration_since(modified)
                    .map(|delta| delta > STALE_FULL_BENCHMARK_WINDOW)
                    .unwrap_or(false);
                if is_stale {
                    continue;
                }
            }
        }

        let include_errors = newest_component_log
            .and_then(|latest| latest.duration_since(modified).ok())
            .map(|delta| delta <= STALE_ERROR_WINDOW)
            .unwrap_or(true);

        if let Ok(contents) = std::fs::read_to_string(&log_path) {
            let json = extract_json(&contents);
            if !json.is_empty() {
                parse_benchmark_json(&json, &mut results, include_errors);
            }
        }
    }

    if results.gpus.is_empty() {
        results.gpus.push(GpuInfo {
            index: 0,
            model: "No GPU detected".to_string(),
            vram_gb: 0.0,
            compute_units: 0,
            tensor_cores: false,
            temperature_c: None,
            power_watts: None,
            utilization_percent: None,
            memory_percent: None,
            sclk_mhz: None,
            mclk_mhz: None,
        });
    }

    // Try to load baseline (first log file that isn't the current one)
    results.baseline = load_baseline(&log_dirs);
    dedupe_errors_in_place(&mut results.errors);

    results
}

fn load_baseline(log_dirs: &[PathBuf]) -> Option<Box<BenchmarkResults>> {
    let baseline_patterns = [
        "rocm_benchmarks",
        "vllm_benchmarks",
        "deepspeed_benchmarks",
        "megatron_benchmarks",
        "full_benchmarks",
    ];
    let mut log_files = collect_matching_logs(log_dirs, &baseline_patterns);
    let json_files: Vec<PathBuf> = log_files
        .iter()
        .filter(|path| is_json_log(path))
        .cloned()
        .collect();
    if !json_files.is_empty() {
        log_files = json_files;
    }

    if log_files.len() > 1 {
        for file in log_files {
            if let Ok(contents) = std::fs::read_to_string(file) {
                let json = extract_json(&contents);
                if json.is_empty() {
                    continue;
                }
                let mut baseline_results = BenchmarkResults {
                    gpus: Vec::new(),
                    memory_bandwidth: None,
                    tensor_core: None,
                    pytorch: None,
                    flash_attention: None,
                    vllm: None,
                    deepspeed: None,
                    megatron: None,
                    errors: Vec::new(),
                    baseline: None,
                };
                parse_benchmark_json(&json, &mut baseline_results, false);
                if !baseline_results.gpus.is_empty()
                    || baseline_results.memory_bandwidth.is_some()
                    || baseline_results.tensor_core.is_some()
                    || baseline_results.pytorch.is_some()
                    || baseline_results.flash_attention.is_some()
                    || baseline_results.vllm.is_some()
                    || baseline_results.deepspeed.is_some()
                    || baseline_results.megatron.is_some()
                {
                    return Some(Box::new(baseline_results));
                }
            }
        }
    }
    None
}

fn push_error_unique(results: &mut BenchmarkResults, message: impl Into<String>) {
    let message = message.into();
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return;
    }
    if results.errors.iter().any(|existing| existing.trim() == trimmed) {
        return;
    }
    results.errors.push(trimmed.to_string());
}

fn dedupe_errors_in_place(errors: &mut Vec<String>) {
    let mut deduped: Vec<String> = Vec::new();
    for err in errors.drain(..) {
        let trimmed = err.trim();
        if trimmed.is_empty() {
            continue;
        }
        if deduped.iter().any(|existing| existing == trimmed) {
            continue;
        }
        deduped.push(trimmed.to_string());
    }
    *errors = deduped;
}

fn parse_benchmark_json(json: &str, results: &mut BenchmarkResults, include_errors: bool) {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
        if let Some(obj) = value.as_object() {
            if let Some(res) = obj.get("results") {
                if res.is_object() {
                    apply_metrics(res, results, include_errors);
                }
                if include_errors {
                    if let Some(errors) = obj.get("errors").and_then(|v| v.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    }) {
                        for err in errors {
                            push_error_unique(results, err);
                        }
                    }
                }
            } else {
                for (_, val) in obj {
                    apply_metrics(val, results, include_errors);
                }
            }
        }
    }
}

fn to_f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
        .unwrap_or_default()
}

fn is_integrated_gpu_model(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    if lowered.contains("ryzen") {
        return true;
    }
    if lowered.contains("integrated") || lowered.contains("igpu") || lowered.contains("apu") {
        return true;
    }
    if lowered.contains("radeon graphics")
        && !lowered.contains("rx")
        && !lowered.contains("pro")
        && !lowered.contains("instinct")
    {
        return true;
    }
    [
        "raphael",
        "phoenix",
        "rembrandt",
        "cezanne",
        "barcelo",
        "yellow carp",
        "green sardine",
        "pink sardine",
        "hawk point",
    ]
    .iter()
    .any(|token| lowered.contains(token))
}

fn apply_metrics(val: &serde_json::Value, results: &mut BenchmarkResults, include_errors: bool) {
    if let Some(obj) = val.as_object() {
        // If this object contains recognized benchmark fields, process it
        if obj.contains_key("metrics")
            || obj.contains_key("results")
            || obj.contains_key("gpus")
            || obj.contains_key("gpu_model")
            || obj.contains_key("hbm_peak_gb_s")
            || obj.contains_key("fp16_tflops")
            || obj.contains_key("gemm_gflops")
            || obj.contains_key("flash_attention_speed")
            || obj.contains_key("throughput_tokens_per_sec")
            || obj.contains_key("throughput_samples_per_sec")
            || obj.contains_key("megatron_throughput_samples_per_sec")
        {
            apply_metrics_internal(val, results, include_errors);
        } else {
            // Otherwise, iterate through the object as it might be a container of results
            for (_, nested_val) in obj {
                apply_metrics(nested_val, results, include_errors);
            }
        }
    }
}

fn apply_metrics_internal(val: &serde_json::Value, results: &mut BenchmarkResults, include_errors: bool) {
    if include_errors {
        if let Some(root_obj) = val.as_object() {
            if let Some(errors) = root_obj
                .get("errors")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| e.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
            {
                for err in errors {
                    push_error_unique(results, err);
                }
            }
        }
    }

    let metrics = val
        .get("metrics")
        .or_else(|| val.get("results"))
        .unwrap_or(val);

    if let Some(obj) = metrics.as_object() {
        if include_errors {
            if let Some(errors) = obj
                .get("errors")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| e.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
            {
                for err in errors {
                    push_error_unique(results, err);
                }
            }
        }

        if obj.contains_key("gpus") || obj.contains_key("gpu_model") {
            let gpus = if let Some(arr) = obj.get("gpus").and_then(|v| v.as_array()) {
                arr.iter()
                    .filter_map(|g| g.as_object())
                    .enumerate()
                    .filter_map(|(idx, g)| {
                        let model = g
                            .get("gpu_model")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown");
                        if is_integrated_gpu_model(model) {
                            return None;
                        }
                        let gpu_index = g
                            .get("index")
                            .and_then(|v| v.as_u64())
                            .or_else(|| g.get("local_index").and_then(|v| v.as_u64()))
                            .unwrap_or(idx as u64) as usize;
                        Some(GpuInfo {
                            index: gpu_index,
                            model: model.into(),
                            vram_gb: g.get("vram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            compute_units: g
                                .get("compute_units")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            tensor_cores: g
                                .get("tensor_cores")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false),
                            temperature_c: g.get("temperature_c").and_then(|v| v.as_f64()),
                            power_watts: g.get("power_watts").and_then(|v| v.as_f64()),
                            utilization_percent: g
                                .get("utilization_percent")
                                .and_then(|v| v.as_f64()),
                            memory_percent: g.get("memory_percent").and_then(|v| v.as_f64()),
                            sclk_mhz: g.get("sclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                            mclk_mhz: g.get("mclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                        })
                    })
                    .collect()
            } else {
                let model = obj
                    .get("gpu_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                if is_integrated_gpu_model(model) {
                    Vec::new()
                } else {
                    vec![GpuInfo {
                        index: obj.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        model: model.into(),
                        vram_gb: obj.get("vram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        compute_units: obj
                            .get("compute_units")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32,
                        tensor_cores: obj
                            .get("tensor_cores")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        temperature_c: obj.get("temperature_c").and_then(|v| v.as_f64()),
                        power_watts: obj.get("power_watts").and_then(|v| v.as_f64()),
                        utilization_percent: obj.get("utilization_percent").and_then(|v| v.as_f64()),
                        memory_percent: obj.get("memory_percent").and_then(|v| v.as_f64()),
                        sclk_mhz: obj
                            .get("sclk_mhz")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                        mclk_mhz: obj
                            .get("mclk_mhz")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                    }]
                }
            };
            results.gpus = gpus;
        }

        if obj.contains_key("hbm_peak_gb_s") {
            results.memory_bandwidth = Some(MemoryBandwidthData {
                hbm_peak_gb_s: obj
                    .get("hbm_peak_gb_s")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                system_peak_gb_s: obj
                    .get("system_peak_gb_s")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                hbm_ratio: obj.get("hbm_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0),
                hbm_samples: obj
                    .get("hbm_samples_gbps")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
                system_samples: obj
                    .get("system_samples_gbps")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }

        if obj.contains_key("fp16_tflops") {
            results.tensor_core = Some(TensorCoreData {
                fp16_tflops: obj
                    .get("fp16_tflops")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                bf16_tflops: obj
                    .get("bf16_tflops")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                fp32_tflops: obj
                    .get("fp32_tflops")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                fp16_samples: obj
                    .get("fp16_samples_tflops")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }

        if obj.contains_key("gemm_gflops") || obj.contains_key("convolution_gflops") {
            results.pytorch = Some(PyTorchData {
                gemm_gflops: obj
                    .get("gemm_gflops")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                convolution_gflops: obj
                    .get("convolution_gflops")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                autograd_overhead_percent: obj
                    .get("autograd_overhead_percent")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                gemm_samples: obj
                    .get("gemm_samples_gflops")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
                conv_samples: obj
                    .get("conv_samples_gflops")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }

        if obj.contains_key("flash_attention_speed") || obj.contains_key("standard_attention_speed")
        {
            results.flash_attention = Some(FlashAttentionData {
                standard_attention_speed: obj
                    .get("standard_attention_speed")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                flash_attention_speed: obj
                    .get("flash_attention_speed")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                speedup: obj.get("speedup").and_then(|v| v.as_f64()).unwrap_or(0.0),
                memory_savings_gb: obj
                    .get("memory_savings_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                standard_samples: obj
                    .get("standard_samples_tok_s")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
                flash_samples: obj
                    .get("flash_samples_tok_s")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }

        if obj.contains_key("throughput_tokens_per_sec") {
            let model_name = obj
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();
            let model_format = obj
                .get("model_format")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    if model_name.to_ascii_lowercase().ends_with(".gguf") {
                        "gguf".to_string()
                    } else {
                        "safetensors".to_string()
                    }
                });
            results.vllm = Some(VllmData {
                throughput_tokens_per_sec: obj
                    .get("throughput_tokens_per_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                latency_ms: obj
                    .get("latency_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                model: model_name,
                model_format,
                throughput_samples: obj
                    .get("throughput_samples")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
            let throughput = obj
                .get("throughput_tokens_per_sec")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if throughput <= 0.0 {
                if let Some(reason) = obj
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    push_error_unique(results, format!("vLLM benchmark degraded: {reason}"));
                } else {
                    push_error_unique(results, "vLLM benchmark degraded: zero throughput");
                }
            }
        }

        if obj.contains_key("throughput_samples_per_sec") {
            results.deepspeed = Some(DeepspeedData {
                throughput_samples_per_sec: obj
                    .get("throughput_samples_per_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                avg_latency_ms: obj
                    .get("avg_latency_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                stage: obj.get("stage").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
                accelerator: obj
                    .get("accelerator")
                    .and_then(|v| v.as_str())
                    .unwrap_or("rocm")
                    .to_string(),
                samples: obj.get("samples").map(to_f64_vec).unwrap_or_default(),
            });
        }

        if obj.contains_key("megatron_throughput_samples_per_sec") {
            results.megatron = Some(MegatronData {
                throughput_samples_per_sec: obj
                    .get("megatron_throughput_samples_per_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                avg_latency_ms: obj
                    .get("megatron_avg_latency_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                backend: obj
                    .get("megatron_backend")
                    .and_then(|v| v.as_str())
                    .unwrap_or("megatron")
                    .to_string(),
                samples: obj
                    .get("megatron_samples")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }
    }
}

fn extract_json(content: &str) -> String {
    extract_benchmark_json_string(content).unwrap_or_default()
}

impl BenchmarkResults {
    fn empty() -> Self {
        BenchmarkResults {
            gpus: Vec::new(),
            memory_bandwidth: None,
            tensor_core: None,
            pytorch: None,
            flash_attention: None,
            vllm: None,
            deepspeed: None,
            megatron: None,
            errors: Vec::new(),
            baseline: None,
        }
    }
}

/// Render the benchmark results page
pub fn render_benchmark_page(
    frame: &mut Frame,
    area: Rect,
    results: &BenchmarkResults,
    tab_index: usize,
) {
    let tabs = Tabs::new(vec![
        Line::from("GPU"),
        Line::from("Memory"),
        Line::from("Tensor Core"),
        Line::from("PyTorch"),
        Line::from("Flash Attn"),
        Line::from("vLLM"),
        Line::from("DeepSpeed"),
        Line::from("Megatron"),
    ])
    .select(tab_index)
    .style(Style::default().fg(Color::Cyan))
    .highlight_style(
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Yellow),
    );

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(area);

    frame.render_widget(tabs, chunks[0]);

    match tab_index {
        0 => render_gpu_tab(frame, chunks[1], results),
        1 => render_memory_tab(frame, chunks[1], results),
        2 => render_tensor_core_tab(frame, chunks[1], results),
        3 => render_pytorch_tab(frame, chunks[1], results),
        4 => render_flash_attention_tab(frame, chunks[1], results),
        5 => render_vllm_tab(frame, chunks[1], results),
        6 => render_deepspeed_tab(frame, chunks[1], results),
        7 => render_megatron_tab(frame, chunks[1], results),
        _ => render_gpu_tab(frame, chunks[1], results),
    }
}

fn render_gpu_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    let mut content = String::from("GPU Information & Performance Summary\n\n");

    if results.gpus.is_empty() {
        content.push_str("No GPU data available\n\nRun benchmarks to see GPU information");
    } else {
        for gpu in &results.gpus {
            content.push_str(&format!(
                "GPU {}: {}\nVRAM: {:.1} GB\nCompute Units: {}\nTensor Cores: {}\n",
                gpu.index,
                gpu.model,
                gpu.vram_gb,
                gpu.compute_units,
                if gpu.tensor_cores { "Yes" } else { "No" }
            ));

            if let (Some(temp), Some(power)) = (gpu.temperature_c, gpu.power_watts) {
                content.push_str(&format!(
                    "Temperature: {:.1}°C | Power: {:.1}W\n",
                    temp, power
                ));
            }

            if let (Some(util), Some(mem)) = (gpu.utilization_percent, gpu.memory_percent) {
                content.push_str(&format!(
                    "GPU Utilization: {:.1}% | Memory Usage: {:.1}%\n",
                    util, mem
                ));
            }

            if let (Some(sclk), Some(mclk)) = (gpu.sclk_mhz, gpu.mclk_mhz) {
                content.push_str(&format!(
                    "GFX Clock: {} MHz | Memory Clock: {} MHz\n",
                    sclk, mclk
                ));
            }

            content.push_str("---\n\n");
        }

        content.push_str("Best Observed Performance:\n");
        let mut has_perf = false;

        if let Some(mem) = &results.memory_bandwidth {
            content.push_str(&format!(
                "  • Memory Bandwidth: {:.1} GB/s\n",
                mem.hbm_peak_gb_s
            ));
            has_perf = true;
        }
        if let Some(tc) = &results.tensor_core {
            content.push_str(&format!(
                "  • Tensor Cores (FP16): {:.1} TFLOPS\n",
                tc.fp16_tflops
            ));
            has_perf = true;
        }
        if let Some(torch) = &results.pytorch {
            content.push_str(&format!(
                "  • PyTorch GEMM: {:.1} GFLOPS\n",
                torch.gemm_gflops
            ));
            has_perf = true;
        }
        if let Some(fa) = &results.flash_attention {
            content.push_str(&format!(
                "  • Flash Attention: {:.1} tok/s ({:.2}x speedup)\n",
                fa.flash_attention_speed, fa.speedup
            ));
            has_perf = true;
        }
        if let Some(v) = &results.vllm {
            if v.throughput_tokens_per_sec > 0.0 {
                content.push_str(&format!(
                    "  • vLLM Throughput: {:.1} tok/s\n",
                    v.throughput_tokens_per_sec
                ));
                has_perf = true;
            } else {
                content.push_str("  • vLLM: Benchmark degraded (see Recent errors)\n");
            }
        }
        if let Some(ds) = &results.deepspeed {
            if ds.throughput_samples_per_sec > 0.0 {
                content.push_str(&format!(
                    "  • DeepSpeed Throughput: {:.1} samples/s\n",
                    ds.throughput_samples_per_sec
                ));
                has_perf = true;
            }
        }
        if let Some(meg) = &results.megatron {
            if meg.throughput_samples_per_sec > 0.0 {
                content.push_str(&format!(
                    "  • Megatron Throughput: {:.1} samples/s\n",
                    meg.throughput_samples_per_sec
                ));
                has_perf = true;
            }
        }

        if !has_perf {
            content.push_str("  (No performance benchmarks run yet)\n");
        }
        content.push_str("\n");
    }

    if !results.errors.is_empty() {
        content.push_str("Recent errors:\n");
        for err in &results.errors {
            content.push_str(&format!("- {}\n", err));
        }
    }

    let paragraph = Paragraph::new(content)
        .alignment(Alignment::Left)
        .block(Block::default().title("GPU Details").borders(Borders::ALL));

    frame.render_widget(paragraph, area);
}

fn render_memory_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(mem) = &results.memory_bandwidth {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(7)])
            .split(area);

        let mut summary = format!(
            "HBM Peak: {:.1} GB/s | System Peak: {:.1} GB/s | Ratio: {:.2}x",
            mem.hbm_peak_gb_s, mem.system_peak_gb_s, mem.hbm_ratio
        );

        if let Some(baseline) = &results.baseline {
            if let Some(b_mem) = &baseline.memory_bandwidth {
                let diff =
                    ((mem.hbm_peak_gb_s - b_mem.hbm_peak_gb_s) / b_mem.hbm_peak_gb_s) * 100.0;
                summary.push_str(&format!(
                    " (Baseline: {:.1} GB/s | {:+.1}%)",
                    b_mem.hbm_peak_gb_s, diff
                ));
            }
        }

        frame.render_widget(
            Paragraph::new(summary)
                .alignment(Alignment::Left)
                .style(Style::default().fg(if results.baseline.is_some() {
                    Color::Green
                } else {
                    Color::White
                }))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Summary & Comparison"),
                ),
            chunks[0],
        );

        let hbm_data = prepare_data(&mem.hbm_samples);
        let system_data = prepare_data(&mem.system_samples);
        let mut datasets = Vec::new();
        if !hbm_data.is_empty() {
            datasets.push(series_dataset("HBM", &hbm_data, Color::Cyan));
        }
        if !system_data.is_empty() {
            datasets.push(series_dataset("System", &system_data, Color::Yellow));
        }
        let y_values: Vec<f64> = mem
            .hbm_samples
            .iter()
            .chain(mem.system_samples.iter())
            .cloned()
            .collect();
        let (y0, y1) = y_bounds(&y_values);
        let x_bounds = [0.0, (mem.hbm_samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Memory Bandwidth Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Buffer Size (MB)")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec!["64".into(), "128".into(), "256".into(), "512".into()]),
                )
                .y_axis(
                    Axis::default()
                        .title("GB/s")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No memory data available\n\nRun benchmarks to see memory bandwidth")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("Memory Bandwidth")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_tensor_core_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(tc) = &results.tensor_core {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(7)])
            .split(area);

        let summary = format!(
            "FP16: {:.1} | BF16: {:.1} | FP32: {:.1} TFLOPS",
            tc.fp16_tflops, tc.bf16_tflops, tc.fp32_tflops
        );
        frame.render_widget(
            Paragraph::new(summary).block(Block::default().borders(Borders::ALL).title("Summary")),
            chunks[0],
        );

        let fp16_data = prepare_data(&tc.fp16_samples);
        let mut datasets = Vec::new();
        if !fp16_data.is_empty() {
            datasets.push(series_dataset("FP16 sweep", &fp16_data, Color::Magenta));
        }
        let (y0, y1) = y_bounds(&tc.fp16_samples);
        let x_bounds = [0.0, (tc.fp16_samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Tensor Core Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Matrix Size (N x N)")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec!["512".into(), "1024".into(), "2048".into()]),
                )
                .y_axis(
                    Axis::default()
                        .title("TFLOPS")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No Tensor Core data available\n\nRun benchmarks to see performance")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("Tensor Core Performance")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_pytorch_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(torch) = &results.pytorch {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(7)])
            .split(area);

        let summary = format!(
            "GEMM: {:.1} GFLOPS | Conv: {:.1} GFLOPS | Autograd overhead: {:.1}%",
            torch.gemm_gflops, torch.convolution_gflops, torch.autograd_overhead_percent
        );
        frame.render_widget(
            Paragraph::new(summary).block(Block::default().borders(Borders::ALL).title("Summary")),
            chunks[0],
        );

        let gemm_data = prepare_data(&torch.gemm_samples);
        let conv_data = prepare_data(&torch.conv_samples);
        let mut datasets = Vec::new();
        if !gemm_data.is_empty() {
            datasets.push(series_dataset("GEMM", &gemm_data, Color::Cyan));
        }
        if !conv_data.is_empty() {
            datasets.push(series_dataset("Conv", &conv_data, Color::Yellow));
        }
        let y_values: Vec<f64> = torch
            .gemm_samples
            .iter()
            .chain(torch.conv_samples.iter())
            .cloned()
            .collect();
        let (y0, y1) = y_bounds(&y_values);
        let x_bounds = [0.0, (torch.gemm_samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("PyTorch Kernel Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Benchmark Trial")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec!["Trial 1".into(), "Trial 2".into(), "Trial 3".into()]),
                )
                .y_axis(
                    Axis::default()
                        .title("GFLOPS")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No PyTorch data available\n\nRun benchmarks to see performance")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("PyTorch Performance")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_flash_attention_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(fa) = &results.flash_attention {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(7)])
            .split(area);

        let summary = format!(
            "Standard: {:.1} tok/s | Flash: {:.1} tok/s | Speedup: {:.2}x | Saved: {:.2} GB",
            fa.standard_attention_speed, fa.flash_attention_speed, fa.speedup, fa.memory_savings_gb
        );
        frame.render_widget(
            Paragraph::new(summary).block(Block::default().borders(Borders::ALL).title("Summary")),
            chunks[0],
        );

        let standard_data = prepare_data(&fa.standard_samples);
        let flash_data = prepare_data(&fa.flash_samples);
        let mut datasets = Vec::new();
        if !standard_data.is_empty() {
            datasets.push(series_dataset("Standard", &standard_data, Color::Yellow));
        }
        if !flash_data.is_empty() {
            datasets.push(series_dataset("Flash", &flash_data, Color::Green));
        }
        let y_values: Vec<f64> = fa
            .standard_samples
            .iter()
            .chain(fa.flash_samples.iter())
            .cloned()
            .collect();
        let (y0, y1) = y_bounds(&y_values);
        let x_bounds = [0.0, (fa.flash_samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Flash Attention Speedup"),
                )
                .x_axis(
                    Axis::default()
                        .title("Sequence Length")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "128".into(),
                            "256".into(),
                            "512".into(),
                            "1024".into(),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Tokens/s")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No Flash Attention data available\n\nRun benchmarks to see comparison")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("Flash Attention vs Standard")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_vllm_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(v) = &results.vllm {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(7)])
            .split(area);

        let mut summary = format!(
            "Model: {} ({}) | Throughput: {:.1} tok/s | Latency: {:.1} ms",
            v.model, v.model_format, v.throughput_tokens_per_sec, v.latency_ms
        );

        if let Some(baseline) = &results.baseline {
            if let Some(b_vllm) = &baseline.vllm {
                if b_vllm.throughput_tokens_per_sec > 0.0 && v.throughput_tokens_per_sec > 0.0 {
                    let diff = ((v.throughput_tokens_per_sec - b_vllm.throughput_tokens_per_sec)
                        / b_vllm.throughput_tokens_per_sec)
                        * 100.0;
                    summary.push_str(&format!(
                        " (Baseline: {:.1} | {:+.1}%)",
                        b_vllm.throughput_tokens_per_sec, diff
                    ));
                }
            }
        }

        let mut summary_text = Text::from(summary);
        if v.throughput_tokens_per_sec == 0.0 {
            summary_text.lines.push(Line::from(""));
            summary_text.lines.push(Line::from(vec![
                Span::styled("⚠ WARNING: ", Style::default().fg(Color::Yellow)),
                Span::raw(
                    "Benchmark produced no results. Ensure a tiny safetensors model or local tiny GGUF model is configured.",
                ),
            ]));

            // Check for vLLM specific errors
            for err in &results.errors {
                let lowered = err.to_ascii_lowercase();
                if lowered.contains("vllm")
                    || lowered.contains("safetensors model")
                    || lowered.contains("torch.cuda")
                    || lowered.contains("hip gpu")
                    || lowered.contains("engine core initialization failed")
                {
                    summary_text.lines.push(Line::from(vec![
                        Span::styled("ERROR: ", Style::default().fg(Color::Red)),
                        Span::raw(err),
                    ]));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(summary_text).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Summary & Comparison"),
            ),
            chunks[0],
        );

        let throughput_data = prepare_data(&v.throughput_samples);
        let mut datasets = Vec::new();
        if !throughput_data.is_empty() {
            datasets.push(series_dataset("Throughput", &throughput_data, Color::Cyan));
        }
        let (y0, y1) = y_bounds(&v.throughput_samples);
        let x_bounds = [0.0, (v.throughput_samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("vLLM Throughput Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Benchmark Trial")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "T1".into(),
                            format!("T{}", v.throughput_samples.len() / 2 + 1).into(),
                            format!("T{}", v.throughput_samples.len()).into(),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Tokens/s")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No vLLM data available\n\nInstall vLLM and rerun benchmarks")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("vLLM Performance")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_deepspeed_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(ds) = &results.deepspeed {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(7)])
            .split(area);

        let mut summary = format!(
            "Accelerator: {} | ZeRO Stage: {} | Throughput: {:.1} samples/s | Latency: {:.1} ms",
            ds.accelerator, ds.stage, ds.throughput_samples_per_sec, ds.avg_latency_ms
        );

        if let Some(baseline) = &results.baseline {
            if let Some(b_ds) = &baseline.deepspeed {
                if b_ds.throughput_samples_per_sec > 0.0 {
                    let diff = ((ds.throughput_samples_per_sec - b_ds.throughput_samples_per_sec)
                        / b_ds.throughput_samples_per_sec)
                        * 100.0;
                    summary.push_str(&format!(
                        " (Baseline: {:.1} | {:+.1}%)",
                        b_ds.throughput_samples_per_sec, diff
                    ));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(summary).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Summary & Comparison"),
            ),
            chunks[0],
        );

        let ds_data = prepare_data(&ds.samples);
        let mut datasets = Vec::new();
        if !ds_data.is_empty() {
            datasets.push(series_dataset("DeepSpeed", &ds_data, Color::White));
        }
        let (y0, y1) = y_bounds(&ds.samples);
        let x_bounds = [0.0, (ds.samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("DeepSpeed Throughput Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Benchmark Trial")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "T1".into(),
                            format!("T{}", ds.samples.len() / 2 + 1).into(),
                            format!("T{}", ds.samples.len()).into(),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Samples/s")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No DeepSpeed data available\n\nRun DeepSpeed Performance benchmark")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("DeepSpeed Performance")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn render_megatron_tab(frame: &mut Frame, area: Rect, results: &BenchmarkResults) {
    if let Some(meg) = &results.megatron {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(7)])
            .split(area);

        let mut summary = format!(
            "Backend: {} | Throughput: {:.1} samples/s | Latency: {:.1} ms",
            meg.backend, meg.throughput_samples_per_sec, meg.avg_latency_ms
        );

        if let Some(baseline) = &results.baseline {
            if let Some(b_meg) = &baseline.megatron {
                if b_meg.throughput_samples_per_sec > 0.0 {
                    let diff = ((meg.throughput_samples_per_sec - b_meg.throughput_samples_per_sec)
                        / b_meg.throughput_samples_per_sec)
                        * 100.0;
                    summary.push_str(&format!(
                        " (Baseline: {:.1} | {:+.1}%)",
                        b_meg.throughput_samples_per_sec, diff
                    ));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(summary).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Summary & Comparison"),
            ),
            chunks[0],
        );

        let series = prepare_data(&meg.samples);
        let mut datasets = Vec::new();
        if !series.is_empty() {
            datasets.push(series_dataset("Megatron", &series, Color::LightBlue));
        }
        let (y0, y1) = y_bounds(&meg.samples);
        let x_bounds = [0.0, (meg.samples.len().max(1) - 1) as f64];

        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Megatron Throughput Performance"),
                )
                .x_axis(
                    Axis::default()
                        .title("Benchmark Trial")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "T1".into(),
                            format!("T{}", meg.samples.len() / 2 + 1).into(),
                            format!("T{}", meg.samples.len()).into(),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Samples/s")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([y0, y1])
                        .labels(vec![
                            format!("{:.0}", y0).into(),
                            format!("{:.0}", (y0 + y1) / 2.0).into(),
                            format!("{:.0}", y1).into(),
                        ]),
                ),
            chunks[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new("No Megatron data available\n\nRun Megatron-LM Performance benchmark")
                .alignment(Alignment::Left)
                .block(
                    Block::default()
                        .title("Megatron-LM Performance")
                        .borders(Borders::ALL),
                ),
            area,
        );
    }
}

fn series_dataset<'a>(name: &'a str, data: &'a [(f64, f64)], color: Color) -> Dataset<'a> {
    Dataset::default()
        .name(name)
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(color))
        .graph_type(GraphType::Line)
        .data(data)
}

fn prepare_data(values: &[f64]) -> Vec<(f64, f64)> {
    values
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v))
        .collect()
}

fn y_bounds(values: &[f64]) -> (f64, f64) {
    let max = values.iter().cloned().fold(0.0_f64, f64::max);
    if max == 0.0 {
        (0.0, 1.0)
    } else {
        (0.0, (max * 1.2).ceil())
    }
}

pub fn export_benchmark_report_html(
    results: &BenchmarkResults,
    output_path: Option<&Path>,
) -> Result<PathBuf, String> {
    let report_path = resolve_report_path(output_path)?;
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create report directory {}: {}", parent.display(), e))?;
    }

    let generated_at = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let report_data = serde_json::json!({
        "generated_at": generated_at,
        "summary": build_summary_rows(results),
        "metric_rows": build_metric_rows(results),
        "sample_rows": build_sample_rows(results),
        "errors": results.errors,
        "gpus": results.gpus.iter().map(|g| serde_json::json!({
            "index": g.index,
            "model": g.model,
            "vram_gb": g.vram_gb,
            "compute_units": g.compute_units,
            "tensor_cores": g.tensor_cores,
            "temperature_c": g.temperature_c,
            "power_watts": g.power_watts,
            "utilization_percent": g.utilization_percent,
            "memory_percent": g.memory_percent
        })).collect::<Vec<_>>(),
        "charts": {
            "memory_hbm": results.memory_bandwidth.as_ref().map(|m| m.hbm_samples.clone()).unwrap_or_default(),
            "memory_system": results.memory_bandwidth.as_ref().map(|m| m.system_samples.clone()).unwrap_or_default(),
            "tensor_fp16": results.tensor_core.as_ref().map(|t| t.fp16_samples.clone()).unwrap_or_default(),
            "pytorch_gemm": results.pytorch.as_ref().map(|p| p.gemm_samples.clone()).unwrap_or_default(),
            "pytorch_conv": results.pytorch.as_ref().map(|p| p.conv_samples.clone()).unwrap_or_default(),
            "flash_standard": results.flash_attention.as_ref().map(|f| f.standard_samples.clone()).unwrap_or_default(),
            "flash_optimized": results.flash_attention.as_ref().map(|f| f.flash_samples.clone()).unwrap_or_default(),
            "vllm": results.vllm.as_ref().map(|v| v.throughput_samples.clone()).unwrap_or_default(),
            "deepspeed": results.deepspeed.as_ref().map(|d| d.samples.clone()).unwrap_or_default(),
            "megatron": results.megatron.as_ref().map(|m| m.samples.clone()).unwrap_or_default()
        },
        "key_metrics": {
            "memory_hbm_peak": results.memory_bandwidth.as_ref().map(|m| m.hbm_peak_gb_s).unwrap_or(0.0),
            "tensor_fp16_tflops": results.tensor_core.as_ref().map(|t| t.fp16_tflops).unwrap_or(0.0),
            "pytorch_gemm_gflops": results.pytorch.as_ref().map(|p| p.gemm_gflops).unwrap_or(0.0),
            "flash_speedup": results.flash_attention.as_ref().map(|f| f.speedup).unwrap_or(0.0),
            "vllm_tps": results.vllm.as_ref().map(|v| v.throughput_tokens_per_sec).unwrap_or(0.0),
            "deepspeed_sps": results.deepspeed.as_ref().map(|d| d.throughput_samples_per_sec).unwrap_or(0.0),
            "megatron_sps": results.megatron.as_ref().map(|m| m.throughput_samples_per_sec).unwrap_or(0.0)
        }
    });
    let report_json =
        serde_json::to_string_pretty(&report_data).map_err(|e| format!("json serialization failed: {}", e))?;

    let html = format!(
        r###"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rusty-Stack Benchmark Report</title>
  <style>
    :root {{
      --bg-0: #0b1220;
      --bg-1: #111a2d;
      --bg-2: #18243a;
      --card: #0f1728;
      --text: #e8edf9;
      --muted: #9fb0d0;
      --accent-a: #27d7c6;
      --accent-b: #74a7ff;
      --accent-c: #ffa24d;
      --ok: #3ddc84;
      --warn: #ffd166;
      --err: #ff6b6b;
      --line: #2a3a57;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Sora", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 10% -10%, #1e365f 0%, transparent 60%),
        radial-gradient(700px 380px at 100% 0%, #23334f 0%, transparent 60%),
        linear-gradient(160deg, var(--bg-0), var(--bg-1), var(--bg-2));
      min-height: 100vh;
      padding: 24px;
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      animation: fadeIn 420ms ease-out;
    }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 24px;
      background: linear-gradient(160deg, rgba(39, 215, 198, 0.08), rgba(116, 167, 255, 0.06), rgba(255, 162, 77, 0.08));
      backdrop-filter: blur(4px);
    }}
    .hero h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.2px;
    }}
    .meta {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      margin-top: 20px;
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(8, 14, 24, 0.75);
      padding: 14px;
      transform: translateY(10px);
      opacity: 0;
      animation: rise 400ms ease forwards;
    }}
    .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.7px;
    }}
    .v {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      color: var(--accent-a);
    }}
    .section {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(8, 14, 24, 0.72);
      padding: 14px;
    }}
    h2 {{
      margin: 0 0 10px 0;
      font-size: 20px;
      color: #f3f7ff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #243452;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #bfccdf;
      font-weight: 600;
    }}
    .charts {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    .plot {{
      border: 1px solid #253659;
      border-radius: 12px;
      padding: 10px;
      background: #0a1322;
    }}
    .plot h3 {{
      margin: 0 0 8px 0;
      font-size: 15px;
      color: #dce8ff;
    }}
    canvas {{
      width: 100%;
      height: 220px;
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(26,41,66,0.35), rgba(11,18,32,0.65));
    }}
    .chip {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid #38517a;
      color: #b8c9e7;
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .err li {{
      margin-bottom: 6px;
      color: #ffd4d4;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; }}
      to {{ opacity: 1; }}
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 900px) {{
      .charts {{ grid-template-columns: 1fr; }}
      body {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Rusty-Stack Performance Report</h1>
      <div class="meta">Generated: <span id="generatedAt"></span></div>
      <div class="meta">Profile: Tiny safetensors/gguf benchmark policy enabled for model-based tests.</div>
    </div>

    <div class="grid" id="metricCards"></div>

    <div class="section">
      <h2>Summary Table</h2>
      <table id="summaryTable">
        <thead><tr><th>Component</th><th>Status</th><th>Primary Metric</th><th>Detail</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="section">
      <h2>Detailed Metrics</h2>
      <table id="metricsTable">
        <thead><tr><th>Component</th><th>Metric</th><th>Value</th><th>Unit</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="section">
      <h2>Sample Data (Text Summary)</h2>
      <table id="samplesTable">
        <thead><tr><th>Benchmark</th><th>Series</th><th>Samples</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="section">
      <h2>GPU Inventory</h2>
      <div id="gpuChips"></div>
      <table id="gpuTable">
        <thead><tr><th>GPU</th><th>VRAM (GB)</th><th>CUs</th><th>Tensor Cores</th><th>Utilization (%)</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="section">
      <h2>Benchmark Graphs</h2>
      <div class="charts">
        <div class="plot"><h3>Memory Bandwidth</h3><canvas id="chartMemory"></canvas></div>
        <div class="plot"><h3>Tensor Core FP16</h3><canvas id="chartTensor"></canvas></div>
        <div class="plot"><h3>PyTorch (GEMM/Conv)</h3><canvas id="chartPytorch"></canvas></div>
        <div class="plot"><h3>Flash Attention</h3><canvas id="chartFlash"></canvas></div>
        <div class="plot"><h3>vLLM Throughput</h3><canvas id="chartVllm"></canvas></div>
        <div class="plot"><h3>DeepSpeed Throughput</h3><canvas id="chartDeepspeed"></canvas></div>
        <div class="plot"><h3>Megatron Throughput</h3><canvas id="chartMegatron"></canvas></div>
      </div>
    </div>

    <div class="section">
      <h2>Errors / Warnings</h2>
      <ul class="err" id="errors"></ul>
    </div>
  </div>

  <script>
    const reportData = {report_json};
    document.getElementById("generatedAt").textContent = reportData.generated_at;

    const cards = [
      ["HBM Peak (GB/s)", reportData.key_metrics.memory_hbm_peak],
      ["FP16 TFLOPS", reportData.key_metrics.tensor_fp16_tflops],
      ["PyTorch GEMM (GFLOPS)", reportData.key_metrics.pytorch_gemm_gflops],
      ["Flash Speedup (x)", reportData.key_metrics.flash_speedup],
      ["vLLM tok/s", reportData.key_metrics.vllm_tps],
      ["DeepSpeed samp/s", reportData.key_metrics.deepspeed_sps],
      ["Megatron samp/s", reportData.key_metrics.megatron_sps]
    ];
    const metricCards = document.getElementById("metricCards");
    cards.forEach((entry, idx) => {{
      const c = document.createElement("div");
      c.className = "card";
      c.style.animationDelay = `${{idx * 35}}ms`;
      c.innerHTML = `<div class="k">${{entry[0]}}</div><div class="v">${{Number(entry[1] || 0).toFixed(2)}}</div>`;
      metricCards.appendChild(c);
    }});

    const tbody = document.querySelector("#summaryTable tbody");
    (reportData.summary || []).forEach(row => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${{row.component}}</td><td>${{row.status}}</td><td>${{row.metric}}</td><td>${{row.detail}}</td>`;
      tbody.appendChild(tr);
    }});

    const metricsTbody = document.querySelector("#metricsTable tbody");
    (reportData.metric_rows || []).forEach(row => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${{row.component}}</td><td>${{row.metric}}</td><td>${{Number(row.value || 0).toFixed(2)}}</td><td>${{row.unit}}</td>`;
      metricsTbody.appendChild(tr);
    }});

    const samplesTbody = document.querySelector("#samplesTable tbody");
    (reportData.sample_rows || []).forEach(row => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${{row.benchmark}}</td><td>${{row.series}}</td><td>${{(row.samples || []).map(v => Number(v).toFixed(2)).join(", ") || "-"}}</td>`;
      samplesTbody.appendChild(tr);
    }});

    const gpuChips = document.getElementById("gpuChips");
    const gpuTbody = document.querySelector("#gpuTable tbody");
    (reportData.gpus || []).forEach(g => {{
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `GPU ${{g.index}} • ${{g.model}}`;
      gpuChips.appendChild(chip);

      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${{g.model}}</td><td>${{Number(g.vram_gb||0).toFixed(2)}}</td><td>${{g.compute_units}}</td><td>${{g.tensor_cores ? "yes" : "no"}}</td><td>${{g.utilization_percent ?? "-"}}</td>`;
      gpuTbody.appendChild(tr);
    }});

    const errNode = document.getElementById("errors");
    if ((reportData.errors || []).length === 0) {{
      const li = document.createElement("li");
      li.textContent = "No benchmark errors were recorded.";
      errNode.appendChild(li);
    }} else {{
      reportData.errors.forEach(e => {{
        const li = document.createElement("li");
        li.textContent = e;
        errNode.appendChild(li);
      }});
    }}

    function drawMultiLine(canvasId, series) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      const all = series.flatMap(s => s.values || []);
      if (all.length === 0) {{
        ctx.fillStyle = "#9fb0d0";
        ctx.font = "14px sans-serif";
        ctx.fillText("No data", 16, 24);
        return;
      }}
      const maxV = Math.max(...all, 1);
      const minV = Math.min(...all, 0);
      const pad = 34;
      const x0 = pad, y0 = h - pad, x1 = w - pad, y1 = pad;
      const yRange = Math.max(maxV - minV, 1e-9);
      const yTicks = 4;
      const plotY = (v, t=1) => y0 - ((v - minV) / yRange) * (y0 - y1) * t;

      let start = null;
      function frame(ts) {{
        if (!start) start = ts;
        const t = Math.min(1, (ts - start) / 900);
        ctx.clearRect(0, 0, w, h);

        // Grid and axes with numeric labels
        ctx.font = "11px Space Grotesk, Segoe UI, sans-serif";
        ctx.fillStyle = "#9fb0d0";
        ctx.strokeStyle = "rgba(115,140,180,0.22)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= yTicks; i++) {{
          const yv = minV + (yRange * i / yTicks);
          const py = plotY(yv);
          ctx.beginPath();
          ctx.moveTo(x0, py);
          ctx.lineTo(x1, py);
          ctx.stroke();
          ctx.fillText(Number(yv).toFixed(2), 4, py + 3);
        }}

        const maxLen = Math.max(1, ...series.map(s => (s.values || []).length));
        const xTicks = Math.min(4, Math.max(1, maxLen - 1));
        for (let i = 0; i <= xTicks; i++) {{
          const xi = maxLen === 1 ? 0 : Math.round(((maxLen - 1) * i) / xTicks);
          const px = maxLen === 1 ? (x0 + x1) / 2 : x0 + (xi / (maxLen - 1)) * (x1 - x0);
          ctx.beginPath();
          ctx.moveTo(px, y0);
          ctx.lineTo(px, y0 + 4);
          ctx.stroke();
          ctx.fillText(`S${{xi + 1}}`, px - 9, y0 + 16);
        }}

        ctx.strokeStyle = "rgba(115,140,180,0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y1); ctx.stroke();

        series.forEach((s, si) => {{
          const vals = s.values || [];
          if (vals.length < 1) return;
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          vals.forEach((v, i) => {{
            const px = vals.length === 1 ? (x0 + x1) / 2 : x0 + (i / (vals.length - 1)) * (x1 - x0);
            const py = plotY(v, t);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }});
          ctx.stroke();

          // Data points
          ctx.fillStyle = s.color;
          vals.forEach((v, i) => {{
            const px = vals.length === 1 ? (x0 + x1) / 2 : x0 + (i / (vals.length - 1)) * (x1 - x0);
            const py = plotY(v, t);
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }});

          // Simple legend
          const ly = y1 + 14 + si * 14;
          ctx.fillStyle = s.color;
          ctx.fillRect(x1 - 120, ly - 8, 10, 10);
          ctx.fillStyle = "#dce8ff";
          ctx.fillText(s.name, x1 - 105, ly);
        }});
        if (t < 1) requestAnimationFrame(frame);
      }}
      requestAnimationFrame(frame);
    }}

    drawMultiLine("chartMemory", [
      {{name: "HBM", color: "#27d7c6", values: reportData.charts.memory_hbm}},
      {{name: "System", color: "#ffd166", values: reportData.charts.memory_system}}
    ]);
    drawMultiLine("chartTensor", [{{name: "FP16", color: "#74a7ff", values: reportData.charts.tensor_fp16}}]);
    drawMultiLine("chartPytorch", [
      {{name: "GEMM", color: "#3ddc84", values: reportData.charts.pytorch_gemm}},
      {{name: "Conv", color: "#ffa24d", values: reportData.charts.pytorch_conv}}
    ]);
    drawMultiLine("chartFlash", [
      {{name: "Standard", color: "#8a99b6", values: reportData.charts.flash_standard}},
      {{name: "Optimized", color: "#27d7c6", values: reportData.charts.flash_optimized}}
    ]);
    drawMultiLine("chartVllm", [{{name: "vLLM", color: "#74a7ff", values: reportData.charts.vllm}}]);
    drawMultiLine("chartDeepspeed", [{{name: "DeepSpeed", color: "#ffd166", values: reportData.charts.deepspeed}}]);
    drawMultiLine("chartMegatron", [{{name: "Megatron", color: "#ff8fab", values: reportData.charts.megatron}}]);
  </script>
</body>
</html>"###
    );

    fs::write(&report_path, html)
        .map_err(|e| format!("failed to write report {}: {}", report_path.display(), e))?;
    Ok(report_path)
}

fn resolve_report_path(output_path: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = output_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MLSTACK_BENCHMARK_REPORT_PATH") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    let base = std::env::var("HOME")
        .map(|h| Path::new(&h).join(".mlstack").join("reports"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/mlstack/reports"));
    let file = format!(
        "benchmark_report_{}.html",
        Local::now().format("%Y%m%d_%H%M%S")
    );
    Ok(base.join(file))
}

fn build_summary_rows(results: &BenchmarkResults) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();

    rows.push(serde_json::json!({
        "component": "GPU",
        "status": if results.gpus.is_empty() { "missing" } else { "ok" },
        "metric": format!("{} GPU(s)", results.gpus.len()),
        "detail": results.gpus.iter().map(|g| g.model.clone()).collect::<Vec<_>>().join(" | ")
    }));

    if let Some(m) = &results.memory_bandwidth {
        rows.push(serde_json::json!({
            "component": "Memory Bandwidth",
            "status": "ok",
            "metric": format!("{:.2} GB/s", m.hbm_peak_gb_s),
            "detail": format!("System {:.2} GB/s, ratio {:.2}x", m.system_peak_gb_s, m.hbm_ratio)
        }));
    }
    if let Some(t) = &results.tensor_core {
        rows.push(serde_json::json!({
            "component": "Tensor Core",
            "status": "ok",
            "metric": format!("{:.2} TFLOPS", t.fp16_tflops),
            "detail": format!("BF16 {:.2} TFLOPS, FP32 {:.2} TFLOPS", t.bf16_tflops, t.fp32_tflops)
        }));
    }
    if let Some(p) = &results.pytorch {
        rows.push(serde_json::json!({
            "component": "PyTorch",
            "status": "ok",
            "metric": format!("{:.2} GFLOPS", p.gemm_gflops),
            "detail": format!("Conv {:.2} GFLOPS, autograd overhead {:.2}%", p.convolution_gflops, p.autograd_overhead_percent)
        }));
    }
    if let Some(f) = &results.flash_attention {
        rows.push(serde_json::json!({
            "component": "Flash Attention",
            "status": "ok",
            "metric": format!("{:.2} tok/s", f.flash_attention_speed),
            "detail": format!("Standard {:.2} tok/s, speedup {:.2}x, saved {:.3} GB", f.standard_attention_speed, f.speedup, f.memory_savings_gb)
        }));
    }
    if let Some(v) = &results.vllm {
        rows.push(serde_json::json!({
            "component": "vLLM",
            "status": if v.throughput_tokens_per_sec > 0.0 { "ok" } else { "degraded" },
            "metric": format!("{:.2} tok/s", v.throughput_tokens_per_sec),
            "detail": format!("Model: {} ({}), latency {:.2} ms", v.model, v.model_format, v.latency_ms)
        }));
    }
    if let Some(d) = &results.deepspeed {
        rows.push(serde_json::json!({
            "component": "DeepSpeed",
            "status": if d.throughput_samples_per_sec > 0.0 { "ok" } else { "degraded" },
            "metric": format!("{:.2} samples/s", d.throughput_samples_per_sec),
            "detail": format!("{} stage {}, latency {:.2} ms", d.accelerator, d.stage, d.avg_latency_ms)
        }));
    }
    if let Some(m) = &results.megatron {
        rows.push(serde_json::json!({
            "component": "Megatron-LM",
            "status": if m.throughput_samples_per_sec > 0.0 { "ok" } else { "degraded" },
            "metric": format!("{:.2} samples/s", m.throughput_samples_per_sec),
            "detail": format!("backend {}, latency {:.2} ms", m.backend, m.avg_latency_ms)
        }));
    }

    rows
}

fn build_metric_rows(results: &BenchmarkResults) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();

    if let Some(m) = &results.memory_bandwidth {
        rows.push(serde_json::json!({"component":"Memory","metric":"HBM Peak","value":m.hbm_peak_gb_s,"unit":"GB/s"}));
        rows.push(serde_json::json!({"component":"Memory","metric":"System Peak","value":m.system_peak_gb_s,"unit":"GB/s"}));
        rows.push(serde_json::json!({"component":"Memory","metric":"HBM Ratio","value":m.hbm_ratio,"unit":"x"}));
    }
    if let Some(t) = &results.tensor_core {
        rows.push(serde_json::json!({"component":"Tensor Core","metric":"FP16","value":t.fp16_tflops,"unit":"TFLOPS"}));
        rows.push(serde_json::json!({"component":"Tensor Core","metric":"BF16","value":t.bf16_tflops,"unit":"TFLOPS"}));
        rows.push(serde_json::json!({"component":"Tensor Core","metric":"FP32","value":t.fp32_tflops,"unit":"TFLOPS"}));
    }
    if let Some(p) = &results.pytorch {
        rows.push(serde_json::json!({"component":"PyTorch","metric":"GEMM","value":p.gemm_gflops,"unit":"GFLOPS"}));
        rows.push(serde_json::json!({"component":"PyTorch","metric":"Convolution","value":p.convolution_gflops,"unit":"GFLOPS"}));
        rows.push(serde_json::json!({"component":"PyTorch","metric":"Autograd Overhead","value":p.autograd_overhead_percent,"unit":"%"}));
    }
    if let Some(f) = &results.flash_attention {
        rows.push(serde_json::json!({"component":"Flash Attention","metric":"Standard","value":f.standard_attention_speed,"unit":"tok/s"}));
        rows.push(serde_json::json!({"component":"Flash Attention","metric":"Flash","value":f.flash_attention_speed,"unit":"tok/s"}));
        rows.push(serde_json::json!({"component":"Flash Attention","metric":"Speedup","value":f.speedup,"unit":"x"}));
    }
    if let Some(v) = &results.vllm {
        rows.push(serde_json::json!({"component":"vLLM","metric":"Throughput","value":v.throughput_tokens_per_sec,"unit":"tok/s"}));
        rows.push(serde_json::json!({"component":"vLLM","metric":"Latency","value":v.latency_ms,"unit":"ms"}));
    }
    if let Some(d) = &results.deepspeed {
        rows.push(serde_json::json!({"component":"DeepSpeed","metric":"Throughput","value":d.throughput_samples_per_sec,"unit":"samples/s"}));
        rows.push(serde_json::json!({"component":"DeepSpeed","metric":"Latency","value":d.avg_latency_ms,"unit":"ms"}));
    }
    if let Some(m) = &results.megatron {
        rows.push(serde_json::json!({"component":"Megatron","metric":"Throughput","value":m.throughput_samples_per_sec,"unit":"samples/s"}));
        rows.push(serde_json::json!({"component":"Megatron","metric":"Latency","value":m.avg_latency_ms,"unit":"ms"}));
    }

    rows
}

fn build_sample_rows(results: &BenchmarkResults) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();

    if let Some(m) = &results.memory_bandwidth {
        rows.push(serde_json::json!({"benchmark":"Memory","series":"HBM GB/s","samples":m.hbm_samples}));
        rows.push(serde_json::json!({"benchmark":"Memory","series":"System GB/s","samples":m.system_samples}));
    }
    if let Some(t) = &results.tensor_core {
        rows.push(serde_json::json!({"benchmark":"Tensor Core","series":"FP16 TFLOPS","samples":t.fp16_samples}));
    }
    if let Some(p) = &results.pytorch {
        rows.push(serde_json::json!({"benchmark":"PyTorch","series":"GEMM GFLOPS","samples":p.gemm_samples}));
        rows.push(serde_json::json!({"benchmark":"PyTorch","series":"Conv GFLOPS","samples":p.conv_samples}));
    }
    if let Some(f) = &results.flash_attention {
        rows.push(serde_json::json!({"benchmark":"Flash Attention","series":"Standard tok/s","samples":f.standard_samples}));
        rows.push(serde_json::json!({"benchmark":"Flash Attention","series":"Flash tok/s","samples":f.flash_samples}));
    }
    if let Some(v) = &results.vllm {
        rows.push(serde_json::json!({"benchmark":"vLLM","series":"Throughput tok/s","samples":v.throughput_samples}));
    }
    if let Some(d) = &results.deepspeed {
        rows.push(serde_json::json!({"benchmark":"DeepSpeed","series":"Throughput samples/s","samples":d.samples}));
    }
    if let Some(m) = &results.megatron {
        rows.push(serde_json::json!({"benchmark":"Megatron","series":"Throughput samples/s","samples":m.samples}));
    }

    rows
}
