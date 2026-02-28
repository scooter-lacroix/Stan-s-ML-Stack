//! Performance benchmark infrastructure for Rusty-Stack.
//! Heavy lifting is delegated to a small embedded Python helper that runs
//! ROCm-enabled PyTorch kernels to exercise the real hardware.

use serde::{Deserialize, Serialize};
use serde_json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub success: bool,
    pub execution_time_ms: u128,
    pub metrics: serde_json::Value,
    pub errors: Vec<String>,
}

fn helper_dir_candidates() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".rusty-stack").join("tmp"));
    }
    dirs.push(PathBuf::from("/tmp").join("rusty-stack-tmp"));
    dirs
}

fn ensure_helper_script() -> Result<PathBuf, String> {
    let body = PY_HELPER.trim_start();
    let mut last_error = String::new();

    for dir in helper_dir_candidates() {
        if let Err(err) = fs::create_dir_all(&dir) {
            last_error = format!("failed to create helper dir {}: {}", dir.display(), err);
            continue;
        }
        let path = dir.join("rusty_bench.py");
        match fs::write(&path, body) {
            Ok(_) => return Ok(path),
            Err(err) => {
                last_error = format!("failed to write helper script {}: {}", path.display(), err);
                continue;
            }
        }
    }

    Err(last_error)
}

fn resolve_benchmark_python() -> String {
    for key in [
        "MLSTACK_BENCHMARK_PYTHON",
        "MLSTACK_PYTHON_BIN",
        "UV_PYTHON",
    ] {
        if let Ok(value) = env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }

    for candidate in ["/usr/local/bin/python3"] {
        let path = Path::new(candidate);
        if path.exists() {
            return candidate.to_string();
        }
    }

    for candidate in ["python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return candidate.to_string();
        }
    }

    "python3".to_string()
}

fn extract_helper_payload(stdout: &str) -> Option<serde_json::Value> {
    let marker = "---BENCHMARK_RESULTS_START---";
    let search = if let Some(pos) = stdout.rfind(marker) {
        &stdout[pos + marker.len()..]
    } else {
        stdout
    };

    if let Some(start) = search.find('{') {
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for (idx, ch) in search[start..].char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                    continue;
                }
                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    if depth == 0 {
                        return None;
                    }
                    depth -= 1;
                    if depth == 0 {
                        let end = start + idx + 1;
                        return serde_json::from_str::<serde_json::Value>(&search[start..end]).ok();
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn run_python_benchmark(name: &str) -> BenchmarkResult {
    let start = Instant::now();
    let python_bin = resolve_benchmark_python();
    let helper = match ensure_helper_script() {
        Ok(path) => path,
        Err(err) => {
            return BenchmarkResult {
                name: name.to_string(),
                success: false,
                execution_time_ms: start.elapsed().as_millis(),
                metrics: serde_json::json!({}),
                errors: vec![format!("Unable to create benchmark helper script: {}", err)],
            };
        }
    };

    let mut command = Command::new(&python_bin);
    command.arg(helper).arg(name).arg("--json");
    if env::var_os("PYTHONHASHSEED").is_none() {
        command.env("PYTHONHASHSEED", "0");
    }
    let output = command.output();

    let parse_payload_result = |parsed: serde_json::Value| BenchmarkResult {
        name: parsed
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(name)
            .to_string(),
        success: parsed
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        execution_time_ms: parsed
            .get("execution_time_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(start.elapsed().as_millis() as u64) as u128,
        metrics: parsed.get("metrics").cloned().unwrap_or_default(),
        errors: parsed
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| e.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
    };

    match output {
        Ok(out) => {
            let stdout_str = String::from_utf8_lossy(&out.stdout);
            if let Some(parsed) = extract_helper_payload(&stdout_str) {
                let mut result = parse_payload_result(parsed);
                if !result.success {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stderr_trim = stderr.trim();
                    if !stderr_trim.is_empty() {
                        let stderr_tail = stderr_trim
                            .lines()
                            .rev()
                            .take(24)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join(" | ");
                        if !stderr_tail.is_empty() {
                            result
                                .errors
                                .push(format!("Helper stderr tail: {}", stderr_tail));
                        }
                    }
                }
                return result;
            }

            if out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                let parse_error = if stderr.is_empty() {
                    format!("Failed to parse helper JSON output using {}", python_bin)
                } else {
                    format!(
                        "Failed to parse helper JSON output using {} (stderr: {})",
                        python_bin, stderr
                    )
                };
                return BenchmarkResult {
                    name: name.to_string(),
                    success: false,
                    execution_time_ms: start.elapsed().as_millis(),
                    metrics: serde_json::json!({}),
                    errors: vec![parse_error],
                };
            }

            BenchmarkResult {
                name: name.to_string(),
                success: false,
                execution_time_ms: start.elapsed().as_millis(),
                metrics: serde_json::json!({}),
                errors: {
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let detail = if !stderr.is_empty() {
                        stderr
                    } else if !stdout.is_empty() {
                        stdout
                    } else {
                        "no output".to_string()
                    };
                    vec![format!(
                        "Helper failed with status {} using {}: {}",
                        out.status, python_bin, detail
                    )]
                },
            }
        }
        Err(err) => BenchmarkResult {
            name: name.to_string(),
            success: false,
            execution_time_ms: start.elapsed().as_millis(),
            metrics: serde_json::json!({}),
            errors: vec![format!("Failed to invoke {}: {}", python_bin, err)],
        },
    }
}

pub fn run_gpu_capability_benchmark() -> BenchmarkResult {
    run_python_benchmark("gpu-info")
}
pub fn run_memory_bandwidth_benchmark() -> BenchmarkResult {
    run_python_benchmark("memory-bandwidth")
}
pub fn run_tensor_core_benchmark() -> BenchmarkResult {
    run_python_benchmark("tensor-core")
}
pub fn run_gemm_benchmark() -> BenchmarkResult {
    run_python_benchmark("gemm")
}
pub fn run_pytorch_benchmark() -> BenchmarkResult {
    run_python_benchmark("pytorch")
}
pub fn run_flash_attention_benchmark() -> BenchmarkResult {
    run_python_benchmark("flash-attention")
}
pub fn run_vllm_benchmark() -> BenchmarkResult {
    run_python_benchmark("vllm")
}
pub fn run_deepspeed_benchmark() -> BenchmarkResult {
    run_python_benchmark("deepspeed")
}
pub fn run_megatron_benchmark() -> BenchmarkResult {
    run_python_benchmark("megatron")
}

// ---------------------------------------------------------------------------
// Embedded Python helper
// ---------------------------------------------------------------------------
const PY_HELPER: &str = r#"
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_GPU_RUNTIME_CACHE = None
_DEFAULT_TINY_SAFETENSORS_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _set_amd_gpu_id_table_env():
    if os.environ.get("AMDGPU_ASIC_ID_TABLE_PATH"):
        return
    candidate = "/usr/share/libdrm/amdgpu.ids"
    if os.path.isfile(candidate):
        os.environ["AMDGPU_ASIC_ID_TABLE_PATH"] = candidate
        os.environ.setdefault("AMDGPU_ASIC_ID_TABLE_PATHS", "/usr/share/libdrm")


def _env_or_default(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _resolve_gguf_model_path(raw):
    raw = (raw or "").strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    if os.path.isfile(expanded):
        return expanded
    if os.path.isdir(expanded):
        for entry in sorted(os.listdir(expanded)):
            if entry.lower().endswith(".gguf"):
                return os.path.join(expanded, entry)
    return ""


def _resolve_vllm_model_candidates():
    safetensors_model = _env_or_default(
        "MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL",
        _DEFAULT_TINY_SAFETENSORS_MODEL,
    )
    gguf_hint = os.environ.get("MLSTACK_BENCH_VLLM_GGUF_MODEL_PATH", "")
    gguf_tokenizer = _env_or_default("MLSTACK_BENCH_VLLM_GGUF_TOKENIZER", safetensors_model)
    gguf_path = _resolve_gguf_model_path(gguf_hint)

    candidates = []
    if gguf_path:
        candidates.append({
            "format": "gguf",
            "model": gguf_path,
            "tokenizer": gguf_tokenizer,
        })
    candidates.append({
        "format": "safetensors",
        "model": safetensors_model,
    })
    return candidates


def _probe_gpu_runtime():
    global _GPU_RUNTIME_CACHE
    if _GPU_RUNTIME_CACHE is not None:
        return _GPU_RUNTIME_CACHE

    _set_amd_gpu_id_table_env()
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        reason = "; ".join(errors) if errors else "unable to import torch"
        _GPU_RUNTIME_CACHE = (False, reason, {"available": False})
        return _GPU_RUNTIME_CACHE

    if not torch.cuda.is_available():
        _GPU_RUNTIME_CACHE = (False, "torch.cuda is not available", {"available": False})
        return _GPU_RUNTIME_CACHE

    arch = "unknown"
    try:
        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        arch = getattr(props, "gcnArchName", "unknown")
    except Exception:
        pass

    probe_code = (
        "import torch;"
        "x=torch.ones(1, device='cuda', dtype=torch.float32);"
        "y=x+1;"
        "torch.cuda.synchronize();"
        "print(float(y.item()))"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if probe.returncode == 0:
            _GPU_RUNTIME_CACHE = (True, "", {"available": True, "gcn_arch": arch})
            return _GPU_RUNTIME_CACHE

        detail = (probe.stderr or probe.stdout or "").strip()
        if not detail:
            detail = f"exit code {probe.returncode}"
        if probe.returncode < 0:
            detail = f"signal {-probe.returncode}: {detail}"
        reason = f"GPU runtime probe failed on {arch}: {detail}"
        _GPU_RUNTIME_CACHE = (False, reason, {"available": True, "gcn_arch": arch})
        return _GPU_RUNTIME_CACHE
    except subprocess.TimeoutExpired:
        reason = f"GPU runtime probe timed out on {arch}"
        _GPU_RUNTIME_CACHE = (False, reason, {"available": True, "gcn_arch": arch})
        return _GPU_RUNTIME_CACHE
    except Exception as exc:
        reason = f"GPU runtime probe error on {arch}: {exc}"
        _GPU_RUNTIME_CACHE = (False, reason, {"available": True, "gcn_arch": arch})
        return _GPU_RUNTIME_CACHE


def _degraded_metrics(name, reason, extra=None):
    payload = {
        "mode": "degraded",
        "component": name,
        "gpu_runtime_stable": False,
        "reason": reason,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # pragma: no cover
        return None, [f"Unable to import torch: {exc}"]


def _parse_visible_gpu_indices():
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue

        values = []
        valid = True
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                valid = False
                break
            values.append(int(token))

        if valid and values:
            return values
    return []


def _coerce_smi_value(entry, key):
    raw = entry.get(key)
    if isinstance(raw, dict):
        value = raw.get("value")
        if value is None:
            return None
        return value
    return raw


def _discrete_hint_from_name(name):
    if not name:
        return None

    lowered = str(name).lower()
    if any(token in lowered for token in ("radeon rx", " rx ", "radeon pro", "instinct", "firepro")):
        return True
    if "ryzen" in lowered:
        return False
    if "radeon graphics" in lowered and "rx" not in lowered and "pro" not in lowered:
        return False
    if "integrated" in lowered or "igpu" in lowered or "apu" in lowered:
        return False
    return None


def _is_integrated_name(name):
    hint = _discrete_hint_from_name(name)
    if hint is not None:
        return not hint
    lowered = str(name or "").lower()
    for token in (
        "raphael",
        "phoenix",
        "rembrandt",
        "cezanne",
        "barcelo",
        "yellow carp",
        "green sardine",
        "pink sardine",
        "hawk point",
    ):
        if token in lowered:
            return True
    return False


def _gpu_info():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda is not available"]

    visible_indices = _parse_visible_gpu_indices()
    smi_available = shutil.which("rocm-smi") is not None
    devices = []
    for idx in range(torch.cuda.device_count()):
        global_idx = visible_indices[idx] if idx < len(visible_indices) else idx
        device = torch.device(f"cuda:{idx}")
        props = torch.cuda.get_device_properties(device)
        info = {
            "index": global_idx,
            "local_index": idx,
            "gpu_model": getattr(props, "name", "Unknown"),
            "gcn_arch": getattr(props, "gcnArchName", "unknown"),
            "vram_gb": round(getattr(props, "total_memory", 0) / 1e9, 2),
            "compute_units": getattr(props, "multi_processor_count", 0),
            "max_clock_mhz": getattr(props, "max_frequency", 0) / 1e6,
            "tensor_cores": bool(getattr(props, "gcnArchName", "").startswith("gfx11")),
        }

        smi_discrete_hint = None
        if smi_available:
            try:
                smi = subprocess.run(
                    [
                        "rocm-smi",
                        "--showproductname",
                        "--showtemp",
                        "--showpower",
                        "--showuse",
                        "--showclocks",
                        "--json",
                        "-d",
                        str(global_idx),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if smi.returncode == 0 and smi.stdout.strip().startswith("{"):
                    data = json.loads(smi.stdout)
                    first_key = next(iter(data.keys()))
                    entry = data.get(first_key, {})

                    smi_name = (
                        _coerce_smi_value(entry, "Card Series")
                        or _coerce_smi_value(entry, "Card series")
                        or _coerce_smi_value(entry, "Card Model")
                        or _coerce_smi_value(entry, "Card model")
                    )
                    if smi_name:
                        info["gpu_model"] = str(smi_name).strip()
                        smi_discrete_hint = _discrete_hint_from_name(smi_name)

                    temp = entry.get("Temperature (Sensor die)", {}).get("value")
                    power = entry.get("Average Graphics Package Power", {}).get("value")
                    if temp is not None:
                        info["temperature_c"] = float(temp)
                    if power is not None:
                        info["power_watts"] = float(power)
                    
                    # capture additional metrics requested
                    util = entry.get("GPU use (%)", {}).get("value")
                    if util is not None:
                        info["utilization_percent"] = float(util)
                    
                    mem_util = entry.get("GPU memory use (%)", {}).get("value")
                    if mem_util is not None:
                        info["memory_percent"] = float(mem_util)
                    
                    sclk = entry.get("GFX Clock (MHz)", {}).get("value")
                    if sclk is not None:
                        info["sclk_mhz"] = int(float(sclk))
                    
                    mclk = entry.get("Memory Clock (MHz)", {}).get("value")
                    if mclk is not None:
                        info["mclk_mhz"] = int(float(mclk))
            except Exception:
                pass

        if smi_discrete_hint is False:
            continue
        if smi_discrete_hint is None and _is_integrated_name(info.get("gpu_model")):
            continue
        devices.append(info)

    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    metrics = {"gpus": devices}
    metrics.update({"gpu_runtime_stable": probe_ok})
    if isinstance(probe_meta, dict):
        metrics.update({f"probe_{k}": v for k, v in probe_meta.items()})
    if probe_ok:
        return True, metrics, []
    return True, metrics, [probe_reason]


def _time_fn(fn, warmup=1, repeat=5):
    import torch

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def _memory_bandwidth():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "hbm_peak_gb_s": 0.0,
            "system_peak_gb_s": 0.0,
            "hbm_ratio": 0.0,
            "hbm_samples_gbps": [],
            "system_samples_gbps": [],
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("memory-bandwidth", probe_reason, extra), [probe_reason]

    device = torch.device("cuda:0")
    sizes_mb = [64, 128, 256, 512]
    hbm_samples = []
    system_samples = []

    for size_mb in sizes_mb:
        numel = size_mb * 1024 * 1024 // 4
        a = torch.empty(numel, device=device, dtype=torch.float32)
        def op():
            a.add_(1.0)
        t = _time_fn(op, warmup=1, repeat=4)
        bytes_moved = a.numel() * a.element_size() * 2  # read + write
        hbm_samples.append(bytes_moved / t / 1e9)

        host = torch.empty_like(a.cpu())
        def h2d():
            _ = host.to(device, non_blocking=True)
        t_h2d = _time_fn(h2d, warmup=1, repeat=3)
        system_samples.append(bytes_moved / t_h2d / 1e9 if t_h2d > 0 else 0)

    hbm_peak = max(hbm_samples) if hbm_samples else 0
    sys_peak = max(system_samples) if system_samples else 0
    hbm_ratio = hbm_peak / sys_peak if sys_peak > 0 else 0

    return True, {
        "hbm_peak_gb_s": round(hbm_peak, 2),
        "system_peak_gb_s": round(sys_peak, 2),
        "hbm_ratio": round(hbm_ratio, 2),
        "hbm_samples_gbps": [round(x, 2) for x in hbm_samples],
        "system_samples_gbps": [round(x, 2) for x in system_samples],
    }, []


def _tensor_core():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "fp16_tflops": 0.0,
            "bf16_tflops": 0.0,
            "tf32_tflops": 0.0,
            "fp32_tflops": 0.0,
            "fp16_samples_tflops": [],
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("tensor-core", probe_reason, extra), [probe_reason]

    device = torch.device("cuda:0")
    sizes = [512, 1024, 2048]
    
    results = {}
    
    # FP16
    fp16_samples = []
    for n in sizes:
        a = torch.randn((n, n), device=device, dtype=torch.float16)
        b = torch.randn((n, n), device=device, dtype=torch.float16)
        t = _time_fn(lambda: torch.matmul(a, b), warmup=2, repeat=3)
        flops = 2 * n * n * n
        fp16_samples.append(flops / t / 1e12)
    results["fp16_tflops"] = round(max(fp16_samples), 2)
    results["fp16_samples"] = [round(x, 2) for x in fp16_samples]

    # BF16
    bf16_samples = []
    try:
        for n in sizes:
            a = torch.randn((n, n), device=device, dtype=torch.bfloat16)
            b = torch.randn((n, n), device=device, dtype=torch.bfloat16)
            t = _time_fn(lambda: torch.matmul(a, b), warmup=2, repeat=3)
            flops = 2 * n * n * n
            bf16_samples.append(flops / t / 1e12)
        results["bf16_tflops"] = round(max(bf16_samples), 2)
    except Exception:
        results["bf16_tflops"] = 0.0

    # FP32
    fp32_samples = []
    for n in sizes:
        a = torch.randn((n, n), device=device, dtype=torch.float32)
        b = torch.randn((n, n), device=device, dtype=torch.float32)
        t = _time_fn(lambda: torch.matmul(a, b), warmup=2, repeat=3)
        flops = 2 * n * n * n
        fp32_samples.append(flops / t / 1e12)
    results["fp32_tflops"] = round(max(fp32_samples), 2)
    
    # TF32 is NVIDIA specific, on AMD we'll report 0 or skip
    results["tf32_tflops"] = 0.0

    return True, {
        "fp16_tflops": results["fp16_tflops"],
        "bf16_tflops": results["bf16_tflops"],
        "tf32_tflops": results["tf32_tflops"],
        "fp32_tflops": results["fp32_tflops"],
        "fp16_samples_tflops": results["fp16_samples"],
    }, []


def _gemm():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "fp16_peak_gflops": 0.0,
            "fp16_samples_gflops": [],
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("gemm", probe_reason, extra), [probe_reason]

    device = torch.device("cuda:0")
    shapes = [(1024, 1024, 1024), (1536, 1536, 1536), (2048, 2048, 2048)]
    samples = []
    for m, k, n in shapes:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        def op():
            return torch.matmul(a, b)
        t = _time_fn(op, warmup=1, repeat=3)
        flops = 2 * m * n * k
        samples.append(flops / t / 1e9)
    peak = max(samples) if samples else 0
    return True, {
        "fp16_peak_gflops": round(peak, 2),
        "fp16_samples_gflops": [round(x, 2) for x in samples],
    }, []


def _pytorch():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "gemm_gflops": 0.0,
            "convolution_gflops": 0.0,
            "gemm_samples_gflops": [],
            "conv_samples_gflops": [],
            "autograd_overhead_percent": 0.0,
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("pytorch", probe_reason, extra), [probe_reason]

    device = torch.device("cuda:0")
    m = k = n = 1024
    a = torch.randn((m, k), device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn((k, n), device=device, dtype=torch.float32, requires_grad=True)
    gemm_samples = []
    for _ in range(3):
        gemm_t = _time_fn(lambda: torch.matmul(a, b), warmup=1, repeat=1)
        gemm_samples.append(2 * m * n * k / gemm_t / 1e9)
    gemm_peak = max(gemm_samples) if gemm_samples else 0

    conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1).to(device)
    x = torch.randn((32, 64, 64, 64), device=device)
    conv_samples = []
    for _ in range(3):
        conv_t = _time_fn(lambda: conv(x), warmup=1, repeat=1)
        conv_ops = 2 * 32 * 64 * 64 * 64 * 64 * 3 * 3
        conv_samples.append(conv_ops / conv_t / 1e9)
    conv_peak = max(conv_samples) if conv_samples else 0

    y = a @ b
    y.sum().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = a @ b
    loss = y.sum()
    loss.backward()
    torch.cuda.synchronize()
    full = time.perf_counter() - start

    start = time.perf_counter()
    _ = a @ b
    torch.cuda.synchronize()
    fwd = time.perf_counter() - start
    overhead = (full - fwd) / full * 100 if full > 0 else 0

    return True, {
        "gemm_gflops": round(gemm_peak, 2),
        "convolution_gflops": round(conv_peak, 2),
        "gemm_samples_gflops": [round(x, 2) for x in gemm_samples],
        "conv_samples_gflops": [round(x, 2) for x in conv_samples],
        "autograd_overhead_percent": round(overhead, 2),
    }, []


def _flash_attention():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "standard_attention_speed": 0.0,
            "flash_attention_speed": 0.0,
            "standard_samples_tok_s": [],
            "flash_samples_tok_s": [],
            "speedup": 0.0,
            "memory_savings_gb": 0.0,
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("flash-attention", probe_reason, extra), [probe_reason]

    try:
        from torch.backends.cuda import sdp_kernel
    except Exception as exc:  # pragma: no cover
        return False, {}, [f"flash attention control unavailable: {exc}"]

    device = torch.device("cuda:0")
    seqs = [128, 256, 512, 1024]
    bsz, heads, dim = 8, 8, 64
    flash_samples = []
    std_samples = []
    mem_std = 0
    mem_flash = 0

    for seqlen in seqs:
        q = torch.randn((bsz, heads, seqlen, dim), device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def run_flash():
            with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

        def run_standard():
            with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

        torch.cuda.reset_peak_memory_stats()
        _ = run_standard()
        mem_std = max(mem_std, torch.cuda.max_memory_allocated() / 1e9)

        torch.cuda.reset_peak_memory_stats()
        _ = run_flash()
        mem_flash = max(mem_flash, torch.cuda.max_memory_allocated() / 1e9)

        flash_t = _time_fn(run_flash, warmup=1, repeat=2)
        std_t = _time_fn(run_standard, warmup=1, repeat=2)

        tokens = bsz * seqlen
        flash_samples.append(tokens / flash_t)
        std_samples.append(tokens / std_t)

    flash_peak = max(flash_samples) if flash_samples else 0
    std_peak = max(std_samples) if std_samples else 0

    return True, {
        "standard_attention_speed": round(std_peak, 2),
        "flash_attention_speed": round(flash_peak, 2),
        "standard_samples_tok_s": [round(x, 2) for x in std_samples],
        "flash_samples_tok_s": [round(x, 2) for x in flash_samples],
        "speedup": round(flash_peak / std_peak, 3) if std_peak > 0 else 0,
        "memory_savings_gb": round(max(0, mem_std - mem_flash), 4),
    }, []


def _vllm():
    # Some shell setups can propagate an empty/whitespace-only target device.
    # vLLM rejects empty device strings, so normalize aggressively before import.
    target_device = _env_or_default("VLLM_TARGET_DEVICE", "rocm")
    if target_device not in {"rocm", "cuda", "cpu"}:
        target_device = "rocm"
    os.environ["VLLM_TARGET_DEVICE"] = target_device

    def _ensure_writable_triton_cache_env():
        import tempfile

        def _writable_dir(path):
            if not path:
                return False
            try:
                os.makedirs(path, exist_ok=True)
                probe = os.path.join(path, f".mlstack_probe_{os.getpid()}")
                with open(probe, "w", encoding="utf-8") as fp:
                    fp.write("ok")
                os.remove(probe)
                return True
            except Exception:
                return False

        triton_cache = (os.environ.get("TRITON_CACHE_DIR") or "").strip()
        if triton_cache and _writable_dir(triton_cache):
            triton_home = (os.environ.get("TRITON_HOME") or os.path.dirname(triton_cache)).strip()
            os.environ.setdefault("TRITON_HOME", triton_home or os.path.expanduser("~/.cache/mlstack/triton"))
            os.environ.setdefault("TRITON_DUMP_DIR", os.path.join(os.environ["TRITON_HOME"], "dump"))
            os.environ.setdefault("TRITON_OVERRIDE_DIR", os.path.join(os.environ["TRITON_HOME"], "override"))
            _writable_dir(os.environ.get("TRITON_DUMP_DIR", ""))
            _writable_dir(os.environ.get("TRITON_OVERRIDE_DIR", ""))
            return

        home = os.path.expanduser("~")
        candidates = [
            os.environ.get("MLSTACK_TRITON_HOME", "").strip(),
            os.environ.get("TRITON_HOME", "").strip(),
            os.path.join(home, ".cache", "mlstack", "triton"),
            os.path.join(tempfile.gettempdir(), f"mlstack-triton-{os.getuid()}"),
        ]
        for root in candidates:
            if not root:
                continue
            cache_dir = os.path.join(root, "cache")
            dump_dir = os.path.join(root, "dump")
            override_dir = os.path.join(root, "override")
            if _writable_dir(cache_dir) and _writable_dir(dump_dir) and _writable_dir(override_dir):
                os.environ["MLSTACK_TRITON_HOME"] = root
                os.environ["TRITON_HOME"] = root
                os.environ["TRITON_CACHE_DIR"] = cache_dir
                os.environ["TRITON_DUMP_DIR"] = dump_dir
                os.environ["TRITON_OVERRIDE_DIR"] = override_dir
                return

    _ensure_writable_triton_cache_env()

    def _normalize_visible_devices():
        visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
        candidates = []
        seen = set()
        for token in visible.split(","):
            item = token.strip()
            if not item.isdigit() or item in seen:
                continue
            seen.add(item)
            candidates.append(item)
        if not candidates:
            candidates = ["0"]

        normalized = ",".join(candidates)
        primary = candidates[0]
        os.environ["HIP_VISIBLE_DEVICES"] = normalized
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized
        if not str(os.environ.get("PYTORCH_ROCM_DEVICE", "")).strip():
            os.environ["PYTORCH_ROCM_DEVICE"] = primary
        return normalized, primary

    # Preserve the full discrete-GPU set. Do not collapse to a single GPU.
    visible_devices, primary_visible = _normalize_visible_devices()
    if target_device == "rocm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    candidates = _resolve_vllm_model_candidates()
    candidate_names = [f"{c.get('format')}:{c.get('model')}" for c in candidates]
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        reason = "; ".join(errors) if errors else "unable to import torch"
        return True, _degraded_metrics("vllm", reason, {
            "model": "vLLM (Unavailable)",
            "model_format": "unavailable",
            "throughput_tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "throughput_samples": [],
            "target_device": target_device,
            "candidate_models": candidate_names,
        }), [reason]
    runtime_probe_hint = ""
    if target_device in {"rocm", "cuda"}:
        try:
            if not torch.cuda.is_available():
                runtime_probe_hint = "ROCm torch reports no available HIP GPUs in benchmark environment"
            else:
                torch.cuda.get_device_properties(0)
        except Exception as exc:
            runtime_probe_hint = f"ROCm torch GPU initialization failed before vLLM load: {exc}"
    def _patch_amdsmi_for_vllm_arch_probe():
        try:
            import sys
            import types

            if os.environ.get("MLSTACK_VLLM_DISABLE_AMDSMI_SHIM", "").strip() in {"1", "true", "TRUE"}:
                return

            arch_hint = (
                os.environ.get("GPU_ARCH")
                or os.environ.get("PYTORCH_ROCM_ARCH")
                or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
                or ""
            ).strip()
            if not arch_hint:
                return
            if not arch_hint.startswith("gfx") and "." in arch_hint:
                parts = [p for p in arch_hint.split(".") if p]
                if len(parts) >= 2:
                    arch_hint = f"gfx{parts[0]}{parts[1]}"

            shim = types.ModuleType("amdsmi")

            class AmdSmiException(Exception):
                pass

            def amdsmi_init():
                return None

            def amdsmi_shut_down():
                return None

            def amdsmi_get_processor_handles():
                return [0]

            def amdsmi_get_gpu_asic_info(_handle):
                return {"target_graphics_version": arch_hint}

            def amdsmi_topo_get_link_type(*_args, **_kwargs):
                return (0, 0)

            shim.AmdSmiException = AmdSmiException
            shim.amdsmi_init = amdsmi_init
            shim.amdsmi_shut_down = amdsmi_shut_down
            shim.amdsmi_get_processor_handles = amdsmi_get_processor_handles
            shim.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info
            shim.amdsmi_topo_get_link_type = amdsmi_topo_get_link_type

            sys.modules["amdsmi"] = shim
        except Exception:
            # Best-effort compatibility shim.
            pass

    _patch_amdsmi_for_vllm_arch_probe()
    try:
        import vllm
        from vllm import LLM, SamplingParams
        import logging
        import importlib
        # Suppress vLLM and related logging to avoid breaking JSON parsing
        logging.getLogger("vllm").setLevel(logging.ERROR)
    except Exception as exc:
        err = f"vLLM not available: {exc}"
        return True, _degraded_metrics("vllm", err, {
            "model": "vLLM (Unavailable)",
            "model_format": "unavailable",
            "throughput_tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "throughput_samples": [],
            "target_device": target_device,
            "candidate_models": candidate_names,
        }), [err]
    abi_errors = []
    for ext_name in ("vllm._C", "vllm._rocm_C"):
        try:
            importlib.import_module(ext_name)
        except Exception as exc:
            abi_errors.append(f"{ext_name} import failed: {exc}")
    if len(abi_errors) == 2:
        reason = "vLLM ROCm native extensions failed to load; " + "; ".join(abi_errors)
        return True, _degraded_metrics("vllm", reason, {
            "model": "vLLM (Unavailable)",
            "model_format": "unavailable",
            "throughput_tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "throughput_samples": [],
            "target_device": target_device,
            "candidate_models": candidate_names,
        }), [reason]
    platform_device_type = ""
    platform_class = "unknown"
    try:
        from vllm.platforms import current_platform
        platform_device_type = str(getattr(current_platform, "device_type", "") or "").strip()
        platform_class = current_platform.__class__.__name__
    except Exception:
        pass
    if not platform_device_type:
        platform_reason = (
            "vLLM platform detection returned an empty device_type (UnspecifiedPlatform). "
            "This commonly means amdsmi is missing, so ROCm platform detection failed. "
            "Install amdsmi in the benchmark/runtime Python environment and rerun."
        )
        return True, _degraded_metrics("vllm", platform_reason, {
            "model": "vLLM (Detected)",
            "model_format": "unavailable",
            "throughput_tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "throughput_samples": [],
            "target_device": target_device,
            "candidate_models": candidate_names,
            "vllm_platform_class": platform_class,
            "vllm_platform_device_type": platform_device_type,
        }), [platform_reason]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        runtime_probe_hint = probe_reason if not runtime_probe_hint else f"{runtime_probe_hint}; {probe_reason}"

    prompts = [
        "The future of AI is",
        "ROCm performance on AMD is",
        "vLLM is a fast",
    ]
    max_tokens = 24
    attempt_errors = []

    def _run_vllm_attempt_subprocess(llm_kwargs, prompts, max_tokens, env_overrides=None):
        payload = {
            "llm_kwargs": llm_kwargs,
            "prompts": prompts,
            "max_tokens": int(max_tokens),
        }
        runner = r"""
import json
import logging
import os
import sys
import time
import traceback

def _emit(obj):
    print(json.dumps(obj))

try:
    payload = json.loads(sys.stdin.read())
    llm_kwargs = payload.get("llm_kwargs", {})
    prompts = payload.get("prompts", [])
    max_tokens = int(payload.get("max_tokens", 24))
    try:
        import types
        if os.environ.get("MLSTACK_VLLM_DISABLE_AMDSMI_SHIM", "").strip() not in {"1", "true", "TRUE"}:
            arch_hint = (
                os.environ.get("GPU_ARCH")
                or os.environ.get("PYTORCH_ROCM_ARCH")
                or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
                or ""
            ).strip()
            if arch_hint and not arch_hint.startswith("gfx") and "." in arch_hint:
                parts = [p for p in arch_hint.split(".") if p]
                if len(parts) >= 2:
                    arch_hint = f"gfx{parts[0]}{parts[1]}"
            if arch_hint:
                shim = types.ModuleType("amdsmi")
                class AmdSmiException(Exception):
                    pass
                def amdsmi_init():
                    return None
                def amdsmi_shut_down():
                    return None
                def amdsmi_get_processor_handles():
                    return [0]
                def amdsmi_get_gpu_asic_info(_handle):
                    return {"target_graphics_version": arch_hint}
                def amdsmi_topo_get_link_type(*_args, **_kwargs):
                    return (0, 0)
                shim.AmdSmiException = AmdSmiException
                shim.amdsmi_init = amdsmi_init
                shim.amdsmi_shut_down = amdsmi_shut_down
                shim.amdsmi_get_processor_handles = amdsmi_get_processor_handles
                shim.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info
                shim.amdsmi_topo_get_link_type = amdsmi_topo_get_link_type
                sys.modules["amdsmi"] = shim
    except Exception:
        pass
    import vllm
    from vllm import LLM, SamplingParams
    logging.getLogger("vllm").setLevel(logging.ERROR)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    llm = LLM(**llm_kwargs)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    throughput_samples = []
    for prompt in prompts:
        sample_start = time.perf_counter()
        sample_out = llm.generate([prompt], sampling_params)
        sample_elapsed = time.perf_counter() - sample_start
        tokens = len(sample_out[0].outputs[0].token_ids)
        throughput_samples.append(tokens / sample_elapsed if sample_elapsed > 0 else 0.0)

    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    _emit({
        "ok": True,
        "throughput_tokens_per_sec": round(throughput, 2),
        "latency_ms": round((elapsed / max(len(prompts), 1)) * 1000, 2),
        "throughput_samples": [round(x, 2) for x in throughput_samples],
    })
except Exception as exc:
    tb = traceback.format_exc().strip()
    tail_lines = [line.strip() for line in tb.splitlines()[-12:] if line.strip()]
    msg = str(exc).strip() or repr(exc)
    if tail_lines:
        msg = f"{msg} :: traceback_tail: {' | '.join(tail_lines)}"
    _emit({"ok": False, "error": msg})
"""
        env = os.environ.copy()
        if isinstance(env_overrides, dict):
            for key, value in env_overrides.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = str(value)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=240,
                env=env,
            )
        except Exception as exc:
            return {"ok": False, "error": f"subprocess launch failed: {exc}"}

        merged = (proc.stdout or "").splitlines()
        if proc.stderr:
            merged.extend(proc.stderr.splitlines())
        for line in reversed(merged):
            text = line.strip()
            if not text.startswith("{") or not text.endswith("}"):
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict) and "ok" in parsed:
                if not parsed.get("ok", False):
                    log_tail = []
                    for raw in merged:
                        msg = raw.strip()
                        if not msg:
                            continue
                        if msg.startswith("{") and msg.endswith("}"):
                            continue
                        lower = msg.lower()
                        if any(
                            marker in lower
                            for marker in (
                                "error",
                                "exception",
                                "traceback",
                                "runtimeerror",
                                "valueerror",
                                "fatal",
                                "enginecore",
                                "failed",
                            )
                        ):
                            log_tail.append(msg)
                    if log_tail:
                        tail = " | ".join(log_tail[-12:])[:3000]
                        base = str(parsed.get("error") or "unknown vLLM subprocess failure")
                        if tail and tail not in base:
                            parsed["error"] = f"{base}; engine_log_tail: {tail}"
                return parsed

        detail = " ".join([line.strip() for line in merged if line.strip()])[:4000]
        if not detail:
            detail = f"exit code {proc.returncode}"
        return {"ok": False, "error": detail}

    for candidate in candidates:
        model_name = candidate.get("model")
        model_format = candidate.get("format", "unknown")
        if not model_name:
            continue

        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.65,
            "disable_log_stats": True,
            "tensor_parallel_size": 1,
            "dtype": "float16",
        }
        if model_format == "gguf":
            tokenizer_name = candidate.get("tokenizer")
            if tokenizer_name:
                llm_kwargs["tokenizer"] = tokenizer_name

        attempt = _run_vllm_attempt_subprocess(llm_kwargs, prompts, max_tokens)
        if attempt.get("ok"):
            return True, {
                "model": model_name,
                "model_format": model_format,
                "target_device": target_device,
                "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", primary_visible),
                "throughput_tokens_per_sec": float(attempt.get("throughput_tokens_per_sec", 0.0)),
                "latency_ms": float(attempt.get("latency_ms", 0.0)),
                "throughput_samples": [
                    float(x) for x in (attempt.get("throughput_samples") or [])
                ],
                "candidate_models": candidate_names,
            }, []
        err_msg = str(attempt.get("error") or "unknown vLLM execution error")
        engine_core_failure = (
            "Engine core initialization failed" in err_msg
            or "Failed core proc" in err_msg
            or "No HIP GPUs are available" in err_msg
            or "torch.cuda is not available" in err_msg
        )
        # Defensive fallback for intermittent vLLM device/runtime failures.
        # Preserve full visible GPU list for multi-GPU hosts.
        if "Device string must not be empty" in err_msg or engine_core_failure:
            retry_profiles = [
                (
                    "spawn-v1-default",
                    {
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                        "VLLM_USE_V1": "1",
                    },
                ),
                (
                    "fork-v1-default",
                    {
                        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                        "VLLM_USE_V1": "1",
                    },
                ),
                (
                    "spawn-v0-singleproc",
                    {
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                        "VLLM_USE_V1": "0",
                    },
                ),
                (
                    "spawn-v0-no-aiter",
                    {
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                        "VLLM_USE_V1": "0",
                        "VLLM_ROCM_USE_AITER": "0",
                    },
                ),
            ]
            retry_failures = []
            for profile_name, profile_overrides in retry_profiles:
                retry_overrides = {
                    "HIP_VISIBLE_DEVICES": visible_devices,
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                    "PYTORCH_ROCM_DEVICE": primary_visible,
                    "VLLM_TARGET_DEVICE": target_device,
                }
                retry_overrides.update(profile_overrides)
                retry = _run_vllm_attempt_subprocess(
                    llm_kwargs,
                    prompts,
                    max_tokens,
                    env_overrides=retry_overrides,
                )
                if retry.get("ok"):
                    return True, {
                        "model": model_name,
                        "model_format": model_format,
                        "target_device": target_device,
                        "throughput_tokens_per_sec": float(retry.get("throughput_tokens_per_sec", 0.0)),
                        "latency_ms": float(retry.get("latency_ms", 0.0)),
                        "throughput_samples": [float(x) for x in (retry.get("throughput_samples") or [])],
                        "candidate_models": candidate_names,
                        "visible_devices_after_retry": visible_devices,
                        "fallback_mode": profile_name,
                    }, []
                retry_failures.append(f"{profile_name}: {retry.get('error', 'unknown retry failure')}")
            if retry_failures:
                err_msg = f"{err_msg}; retry_profiles_failed: {' || '.join(retry_failures)}"
        if "Entry Not Found" in err_msg or "not found on the Hugging Face Hub" in err_msg:
            err_msg = (
                f"Model {model_name} not found. Download/copy it first or override "
                "MLSTACK_BENCH_VLLM_SAFETENSORS_MODEL / MLSTACK_BENCH_VLLM_GGUF_MODEL_PATH"
            )
        elif "authentication" in err_msg.lower():
            err_msg = "Hugging Face authentication required. Run 'huggingface-cli login' first."
        attempt_errors.append(f"{model_format} model {model_name} failed: {err_msg}")

    failure_reason = "No tiny vLLM benchmark model could be loaded"
    if runtime_probe_hint:
        failure_reason = f"{failure_reason}; runtime_probe={runtime_probe_hint}"
    if attempt_errors:
        failure_reason = f"{failure_reason}; {attempt_errors[0]}"

    return True, _degraded_metrics("vllm", failure_reason, {
        "model": "vLLM (Detected)",
        "model_format": "unavailable",
        "target_device": target_device,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", primary_visible),
        "throughput_tokens_per_sec": 0.0,
        "latency_ms": 0.0,
        "throughput_samples": [],
        "candidate_models": candidate_names,
    }), attempt_errors or [failure_reason]


def _deepspeed():
    _set_amd_gpu_id_table_env()
    try:
        import deepspeed
        import torch
        import os
    except Exception as exc:
        err = f"DeepSpeed or Torch not available: {exc}"
        return True, _degraded_metrics("deepspeed", err, {
            "throughput_samples_per_sec": 0.0,
            "avg_latency_ms": 0.0,
            "stage": 1,
            "accelerator": "rocm",
            "samples": [],
        }), [err]
    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "throughput_samples_per_sec": 0.0,
            "avg_latency_ms": 0.0,
            "stage": 1,
            "accelerator": "rocm",
            "samples": [],
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("deepspeed", probe_reason, extra), [probe_reason]

    # Ensure ROCm environment
    os.environ["DS_ACCELERATOR"] = "rocm"
    
    try:
        from deepspeed.accelerator import get_accelerator

        accelerator = get_accelerator()
        device_name = accelerator.device_name()
        device = torch.device(device_name if ":" in device_name else f"{device_name}:0")

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.net(x)

        model = SimpleModel().to(device).half()
        batch_size = 4
        input_data = torch.randn(batch_size, 1024, device=device, dtype=torch.float16)

        ds_config = {
            "train_batch_size": batch_size,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
            "fp16": {"enabled": True},
            # DeepSpeed internally computes `step % steps_per_print`; keep this non-zero.
            "steps_per_print": 1,
            "wall_clock_breakdown": False,
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        try:
            model_engine, _, _, _ = deepspeed.initialize(
                config=ds_config,
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,
            )
            mode = "train"
        except Exception as init_exc:
            init_err = str(init_exc)
            if "fused_adam" in init_err.lower() or "cuda_runtime_api.h" in init_err:
                model_engine = deepspeed.init_inference(
                    model,
                    mp_size=1,
                    dtype=torch.float16,
                    replace_with_kernel_inject=False,
                )
                mode = "inference"
            else:
                raise

        times = []
        if mode == "train":
            for _ in range(2):
                out = model_engine(input_data)
                model_engine.backward(out.sum())
                model_engine.step()

            for _ in range(5):
                accelerator.synchronize()
                start = time.perf_counter()
                out = model_engine(input_data)
                model_engine.backward(out.sum())
                model_engine.step()
                accelerator.synchronize()
                times.append(time.perf_counter() - start)
        else:
            model_engine.eval()
            for _ in range(2):
                with torch.no_grad():
                    _ = model_engine(input_data)
                accelerator.synchronize()

            for _ in range(5):
                accelerator.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model_engine(input_data)
                accelerator.synchronize()
                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time

        return True, {
            "throughput_samples_per_sec": round(throughput, 2),
            "avg_latency_ms": round(avg_time * 1000, 2),
            "stage": 0 if mode == "train" else 0,
            "accelerator": "rocm",
            "mode": mode,
            "samples": [round(batch_size / t, 2) for t in times],
        }, []
    except Exception as exc:
        err = f"DeepSpeed benchmark failed: {exc}"
        return True, _degraded_metrics("deepspeed", err, {
            "throughput_samples_per_sec": 0.0,
            "avg_latency_ms": 0.0,
            "stage": 1,
            "accelerator": "rocm",
            "samples": [],
        }), [err]


def _megatron():
    try:
        import torch
        import megatron
    except Exception as exc:
        err = f"Megatron not available: {exc}"
        return True, _degraded_metrics("megatron", err, {
            "megatron_backend": "unavailable",
            "megatron_throughput_samples_per_sec": 0.0,
            "megatron_avg_latency_ms": 0.0,
            "megatron_samples": [],
        }), [err]

    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "megatron_backend": "megatron-core",
            "megatron_throughput_samples_per_sec": 0.0,
            "megatron_avg_latency_ms": 0.0,
            "megatron_samples": [],
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return True, _degraded_metrics("megatron", probe_reason, extra), [probe_reason]

    try:
        device = torch.device("cuda:0")
        model = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 4096),
        ).to(device).half()

        input_data = torch.randn(8, 4096, device=device, dtype=torch.float16)
        samples = []

        for _ in range(2):
            out = model(input_data)
            _ = out.sum()
            torch.cuda.synchronize()

        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(input_data)
            _ = out.sum()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            samples.append(8.0 / elapsed if elapsed > 0 else 0.0)

        avg_tput = sum(samples) / len(samples) if samples else 0.0
        avg_latency_ms = (1000.0 / avg_tput * 8.0) if avg_tput > 0 else 0.0
        return True, {
            "megatron_backend": getattr(megatron, "__name__", "megatron"),
            "megatron_throughput_samples_per_sec": round(avg_tput, 2),
            "megatron_avg_latency_ms": round(avg_latency_ms, 2),
            "megatron_samples": [round(x, 2) for x in samples],
        }, []
    except Exception as exc:
        err = f"Megatron benchmark failed: {exc}"
        return True, _degraded_metrics("megatron", err, {
            "megatron_backend": "megatron-core",
            "megatron_throughput_samples_per_sec": 0.0,
            "megatron_avg_latency_ms": 0.0,
            "megatron_samples": [],
        }), [err]


BENCHES = {
    "gpu-info": _gpu_info,
    "memory-bandwidth": _memory_bandwidth,
    "tensor-core": _tensor_core,
    "gemm": _gemm,
    "pytorch": _pytorch,
    "flash-attention": _flash_attention,
    "vllm": _vllm,
    "deepspeed": _deepspeed,
    "megatron": _megatron,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", help=f"One of: {', '.join(BENCHES.keys())}")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    fn = BENCHES.get(args.bench)
    if fn is None:
        payload = {
            "name": args.bench,
            "success": False,
            "execution_time_ms": 0,
            "metrics": {},
            "errors": [f"Unknown benchmark {args.bench}"],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(1)

    start = time.perf_counter()
    success, metrics, errors = fn()
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    payload = {
        "name": args.bench,
        "success": success,
        "execution_time_ms": elapsed_ms,
        "metrics": metrics,
        "errors": errors,
    }
    
    # Print a clear marker to help the Rust wrapper find the JSON block
    print("\n---BENCHMARK_RESULTS_START---")
    print(json.dumps(payload, indent=2))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
"#;
