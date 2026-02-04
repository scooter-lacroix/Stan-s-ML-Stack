//! Benchmark performance page for Rusty-Stack TUI
//! Renders real benchmark results with ratatui charts and summaries.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Tabs},
    Frame,
};
use serde_json;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub gpus: Vec<GpuInfo>,
    pub memory_bandwidth: Option<MemoryBandwidthData>,
    pub tensor_core: Option<TensorCoreData>,
    pub pytorch: Option<PyTorchData>,
    pub flash_attention: Option<FlashAttentionData>,
    pub vllm: Option<VllmData>,
    pub deepspeed: Option<DeepspeedData>,
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
    pub throughput_samples: Vec<f64>,
}

/// Load benchmark results from all available log files
pub fn load_benchmark_results() -> BenchmarkResults {
    let log_dir = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".rusty-stack")
        .join("logs");

    if !log_dir.exists() {
        return BenchmarkResults::empty();
    }

    let mut results = BenchmarkResults::empty();

    // primary JSON sources written by the wrapper scripts
    for pattern in &[
        "rocm_benchmarks",
        "gpu_memory_bandwidth",
        "pytorch_performance",
        "mlperf_inference",
        "vllm_benchmarks",
        "deepspeed_benchmarks",
        "full_benchmarks",
    ] {
        if let Some(log_path) = find_latest_log(&log_dir, pattern) {
            if let Ok(contents) = std::fs::read_to_string(&log_path) {
                let json = extract_json(&contents);
                if !json.is_empty() {
                    parse_benchmark_json(&json, &mut results);
                }
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
    results.baseline = load_baseline(&log_dir);

    results
}

fn load_baseline(log_dir: &PathBuf) -> Option<Box<BenchmarkResults>> {
    let mut log_files: Vec<_> = std::fs::read_dir(log_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.contains("rocm_benchmarks") || name.contains("vllm_benchmarks") || name.contains("deepspeed_benchmarks") || name.contains("full_benchmarks")
        })
        .collect();

    log_files.sort_by(|a, b| {
        a.metadata().ok().and_then(|m| m.modified().ok())
            .cmp(&b.metadata().ok().and_then(|m| m.modified().ok()))
    });

    if log_files.len() > 1 {
        // The first one is our baseline
        if let Ok(contents) = std::fs::read_to_string(log_files[0].path()) {
            let json = extract_json(&contents);
            let mut baseline_results = BenchmarkResults {
                gpus: Vec::new(),
                memory_bandwidth: None,
                tensor_core: None,
                pytorch: None,
                flash_attention: None,
                vllm: None,
                deepspeed: None,
                errors: Vec::new(),
                baseline: None,
            };
            parse_benchmark_json(&json, &mut baseline_results);
            return Some(Box::new(baseline_results));
        }
    }
    None
}

fn parse_benchmark_json(json: &str, results: &mut BenchmarkResults) {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
        if let Some(obj) = value.as_object() {
            if let Some(res) = obj.get("results") {
                if res.is_object() {
                    apply_metrics(res, results);
                }
                if let Some(errors) = obj
                    .get("errors")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    })
                {
                    results.errors.extend(errors);
                }
            } else {
                for (_, val) in obj {
                    apply_metrics(val, results);
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

fn apply_metrics(val: &serde_json::Value, results: &mut BenchmarkResults) {
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
        {
            apply_metrics_internal(val, results);
        } else {
            // Otherwise, iterate through the object as it might be a container of results
            for (_, nested_val) in obj {
                apply_metrics(nested_val, results);
            }
        }
    }
}

fn apply_metrics_internal(val: &serde_json::Value, results: &mut BenchmarkResults) {
    let metrics = val
        .get("metrics")
        .or_else(|| val.get("results"))
        .unwrap_or(val);

    if let Some(obj) = metrics.as_object() {
        if obj.contains_key("gpus") || obj.contains_key("gpu_model") {
            let gpus = if let Some(arr) = obj.get("gpus").and_then(|v| v.as_array()) {
                arr.iter()
                    .filter_map(|g| g.as_object())
                    .enumerate()
                    .map(|(idx, g)| GpuInfo {
                        index: idx,
                        model: g
                            .get("gpu_model")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown")
                            .into(),
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
                        utilization_percent: g.get("utilization_percent").and_then(|v| v.as_f64()),
                        memory_percent: g.get("memory_percent").and_then(|v| v.as_f64()),
                        sclk_mhz: g.get("sclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                        mclk_mhz: g.get("mclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                    })
                    .collect()
            } else {
                vec![GpuInfo {
                    index: 0,
                    model: obj
                        .get("gpu_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown")
                        .into(),
                    vram_gb: obj.get("vram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    compute_units: obj
                        .get("compute_units")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32,
                    tensor_cores: obj.get("tensor_cores").and_then(|v| v.as_bool()).unwrap_or(false),
                    temperature_c: obj.get("temperature_c").and_then(|v| v.as_f64()),
                    power_watts: obj.get("power_watts").and_then(|v| v.as_f64()),
                    utilization_percent: obj.get("utilization_percent").and_then(|v| v.as_f64()),
                    memory_percent: obj.get("memory_percent").and_then(|v| v.as_f64()),
                    sclk_mhz: obj.get("sclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                    mclk_mhz: obj.get("mclk_mhz").and_then(|v| v.as_u64()).map(|v| v as u32),
                }]
            };
            results.gpus = gpus;
        }

        if obj.contains_key("hbm_peak_gb_s") {
            results.memory_bandwidth = Some(MemoryBandwidthData {
                hbm_peak_gb_s: obj.get("hbm_peak_gb_s").and_then(|v| v.as_f64()).unwrap_or(0.0),
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
                fp16_tflops: obj.get("fp16_tflops").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bf16_tflops: obj.get("bf16_tflops").and_then(|v| v.as_f64()).unwrap_or(0.0),
                fp32_tflops: obj.get("fp32_tflops").and_then(|v| v.as_f64()).unwrap_or(0.0),
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
                memory_savings_gb: obj.get("memory_savings_gb").and_then(|v| v.as_f64()).unwrap_or(0.0),
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
            results.vllm = Some(VllmData {
                throughput_tokens_per_sec: obj
                    .get("throughput_tokens_per_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                latency_ms: obj.get("latency_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
                model: obj
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                throughput_samples: obj
                    .get("throughput_samples")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }

        if obj.contains_key("throughput_samples_per_sec") {
            results.deepspeed = Some(DeepspeedData {
                throughput_samples_per_sec: obj
                    .get("throughput_samples_per_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                avg_latency_ms: obj.get("avg_latency_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
                stage: obj.get("stage").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
                accelerator: obj
                    .get("accelerator")
                    .and_then(|v| v.as_str())
                    .unwrap_or("rocm")
                    .to_string(),
                samples: obj
                    .get("samples")
                    .map(to_f64_vec)
                    .unwrap_or_default(),
            });
        }
    }
}

fn find_latest_log(log_dir: &PathBuf, pattern: &str) -> Option<PathBuf> {
    let mut log_files: Vec<_> = std::fs::read_dir(log_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.contains(pattern))
                .unwrap_or(false)
        })
        .collect();

    log_files.sort_by(|a, b| {
        a.metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .cmp(&b.metadata().ok().and_then(|m| m.modified().ok()))
    });

    log_files.last().map(|e| e.path())
}

fn extract_json(content: &str) -> String {
    // If we have our clear marker, use everything after it
    if let Some(marker_pos) = content.find("---BENCHMARK_RESULTS_START---") {
        let json_part = &content[marker_pos + "---BENCHMARK_RESULTS_START---".len()..];
        if let Some(start) = json_part.find('{') {
            if let Some(end) = json_part.rfind('}') {
                return json_part[start..=end].to_string();
            }
        }
    }

    let trimmed = content.trim();
    if trimmed.starts_with('{') {
        return trimmed.to_string();
    }
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }
    String::new()
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
        _ => {}
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
            content.push_str(&format!("  • Memory Bandwidth: {:.1} GB/s\n", mem.hbm_peak_gb_s));
            has_perf = true;
        }
        if let Some(tc) = &results.tensor_core {
            content.push_str(&format!("  • Tensor Cores (FP16): {:.1} TFLOPS\n", tc.fp16_tflops));
            has_perf = true;
        }
        if let Some(torch) = &results.pytorch {
            content.push_str(&format!("  • PyTorch GEMM: {:.1} GFLOPS\n", torch.gemm_gflops));
            has_perf = true;
        }
        if let Some(fa) = &results.flash_attention {
            content.push_str(&format!("  • Flash Attention: {:.1} tok/s ({:.2}x speedup)\n", fa.flash_attention_speed, fa.speedup));
            has_perf = true;
        }
        if let Some(v) = &results.vllm {
            if v.throughput_tokens_per_sec > 0.0 {
                content.push_str(&format!("  • vLLM Throughput: {:.1} tok/s\n", v.throughput_tokens_per_sec));
                has_perf = true;
            } else if !results.errors.iter().any(|e| e.contains("vLLM")) {
                 // If no explicit error but throughput is 0, it's likely just not run or missing model
                 content.push_str("  • vLLM: Detected (Run vLLM Performance benchmark)\n");
            }
        }
        if let Some(ds) = &results.deepspeed {
            if ds.throughput_samples_per_sec > 0.0 {
                content.push_str(&format!("  • DeepSpeed Throughput: {:.1} samples/s\n", ds.throughput_samples_per_sec));
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
                let diff = ((mem.hbm_peak_gb_s - b_mem.hbm_peak_gb_s) / b_mem.hbm_peak_gb_s) * 100.0;
                summary.push_str(&format!(" (Baseline: {:.1} GB/s | {:+.1}%)", b_mem.hbm_peak_gb_s, diff));
            }
        }

        frame.render_widget(
            Paragraph::new(summary)
                .alignment(Alignment::Left)
                .style(Style::default().fg(if results.baseline.is_some() { Color::Green } else { Color::White }))
                .block(Block::default().borders(Borders::ALL).title("Summary & Comparison")),
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
                .block(Block::default().borders(Borders::ALL).title("Memory Bandwidth Performance"))
                .x_axis(
                    Axis::default()
                        .title("Buffer Size (MB)")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "64".into(),
                            "128".into(),
                            "256".into(),
                            "512".into(),
                        ]),
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
                .block(Block::default().title("Memory Bandwidth").borders(Borders::ALL)),
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
            Paragraph::new(summary)
                .block(Block::default().borders(Borders::ALL).title("Summary")),
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
                .block(Block::default().borders(Borders::ALL).title("Tensor Core Performance"))
                .x_axis(
                    Axis::default()
                        .title("Matrix Size (N x N)")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "512".into(),
                            "1024".into(),
                            "2048".into(),
                        ]),
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
                .block(Block::default().title("Tensor Core Performance").borders(Borders::ALL)),
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
            Paragraph::new(summary)
                .block(Block::default().borders(Borders::ALL).title("Summary")),
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
                .block(Block::default().borders(Borders::ALL).title("PyTorch Kernel Performance"))
                .x_axis(
                    Axis::default()
                        .title("Benchmark Trial")
                        .style(Style::default().fg(Color::Gray))
                        .bounds(x_bounds)
                        .labels(vec![
                            "Trial 1".into(),
                            "Trial 2".into(),
                            "Trial 3".into(),
                        ]),
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
                .block(Block::default().title("PyTorch Performance").borders(Borders::ALL)),
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
            Paragraph::new(summary)
                .block(Block::default().borders(Borders::ALL).title("Summary")),
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
                .block(Block::default().borders(Borders::ALL).title("Flash Attention Speedup"))
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
                .block(Block::default().title("Flash Attention vs Standard").borders(Borders::ALL)),
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
            "Model: {} | Throughput: {:.1} tok/s | Latency: {:.1} ms",
            v.model, v.throughput_tokens_per_sec, v.latency_ms
        );

        if let Some(baseline) = &results.baseline {
            if let Some(b_vllm) = &baseline.vllm {
                if b_vllm.throughput_tokens_per_sec > 0.0 && v.throughput_tokens_per_sec > 0.0 {
                    let diff = ((v.throughput_tokens_per_sec - b_vllm.throughput_tokens_per_sec) / b_vllm.throughput_tokens_per_sec) * 100.0;
                    summary.push_str(&format!(" (Baseline: {:.1} | {:+.1}%)", b_vllm.throughput_tokens_per_sec, diff));
                }
            }
        }

        let mut summary_text = Text::from(summary);
        if v.throughput_tokens_per_sec == 0.0 {
            summary_text.lines.push(Line::from(""));
            summary_text.lines.push(Line::from(vec![
                Span::styled("⚠ WARNING: ", Style::default().fg(Color::Yellow)),
                Span::raw("Benchmark produced no results. Ensure 'facebook/opt-125m' is available."),
            ]));
            
            // Check for vLLM specific errors
            for err in &results.errors {
                if err.contains("vLLM") || err.contains("vllm") {
                    summary_text.lines.push(Line::from(vec![
                        Span::styled("ERROR: ", Style::default().fg(Color::Red)),
                        Span::raw(err),
                    ]));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(summary_text)
                .block(Block::default().borders(Borders::ALL).title("Summary & Comparison")),
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
                .block(Block::default().borders(Borders::ALL).title("vLLM Throughput Performance"))
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
                .block(Block::default().title("vLLM Performance").borders(Borders::ALL)),
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
                    let diff = ((ds.throughput_samples_per_sec - b_ds.throughput_samples_per_sec) / b_ds.throughput_samples_per_sec) * 100.0;
                    summary.push_str(&format!(" (Baseline: {:.1} | {:+.1}%)", b_ds.throughput_samples_per_sec, diff));
                }
            }
        }

        frame.render_widget(
            Paragraph::new(summary)
                .block(Block::default().borders(Borders::ALL).title("Summary & Comparison")),
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
                .block(Block::default().borders(Borders::ALL).title("DeepSpeed Throughput Performance"))
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
                .block(Block::default().title("DeepSpeed Performance").borders(Borders::ALL)),
            area,
        );
    }
}

fn series_dataset<'a>(
    name: &'a str,
    data: &'a [(f64, f64)],
    color: Color,
) -> Dataset<'a> {
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
