//! Performance benchmark infrastructure for Rusty-Stack.
//! Heavy lifting is delegated to a small embedded Python helper that runs
//! ROCm-enabled PyTorch kernels to exercise the real hardware.

use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use std::path::PathBuf;
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

fn helper_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".rusty-stack")
        .join("tmp")
}

fn ensure_helper_script() -> PathBuf {
    let dir = helper_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("rusty_bench.py");
    let body = PY_HELPER.trim_start();
    let _ = fs::write(&path, body);
    path
}

fn run_python_benchmark(name: &str) -> BenchmarkResult {
    let helper = ensure_helper_script();
    let start = Instant::now();

    let mut cmd = Command::new("python3");
    cmd.arg(helper).arg(name).arg("--json");

    // Special handling for DeepSpeed to work around root-owned site-packages permissions
    // during HIPIFICATION (converting CUDA to HIP code)
    if name == "deepspeed" {
        let shadow_root =
            std::env::temp_dir().join(format!("ds_shadow_{}", unsafe { libc::getuid() }));
        let _ = fs::create_dir_all(&shadow_root);
        let ds_dest = shadow_root.join("deepspeed");

        if !ds_dest.exists() {
            // Find deepspeed path
            if let Ok(output) = Command::new("python3")
                .arg("-c")
                .arg("import deepspeed; import os; print(os.path.dirname(deepspeed.__file__))")
                .output()
            {
                let ds_src = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !ds_src.is_empty() && std::path::Path::new(&ds_src).exists() {
                    let _ = Command::new("cp")
                        .arg("-r")
                        .arg(ds_src)
                        .arg(&ds_dest)
                        .status();
                    let _ = Command::new("chmod")
                        .arg("-R")
                        .arg("u+w")
                        .arg(&ds_dest)
                        .status();
                }
            }
        }

        if shadow_root.exists() {
            cmd.env("PYTHONPATH", shadow_root);
        }
    }

    let output = cmd.output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout_str = String::from_utf8_lossy(&out.stdout);

            // Look for the clear marker first, then fallback to last JSON object
            let mut json_str =
                if let Some(marker_pos) = stdout_str.find("---BENCHMARK_RESULTS_START---") {
                    stdout_str[marker_pos + "---BENCHMARK_RESULTS_START---".len()..]
                        .trim()
                        .to_string()
                } else {
                    stdout_str.trim().to_string()
                };

            if !json_str.starts_with('{') {
                if let Some(start) = json_str.find('{') {
                    if let Some(end) = json_str.rfind('}') {
                        if end > start {
                            json_str = json_str[start..=end].to_string();
                        }
                    }
                }
            }

            let parsed: serde_json::Value =
                serde_json::from_str(&json_str).unwrap_or_else(|e| {
                    serde_json::json!({
                        "name": name,
                        "success": false,
                        "execution_time_ms": 0,
                        "metrics": {},
                        "errors": [format!("Failed to parse helper JSON: {}. Full stdout: {}", e, stdout_str)],
                    })
                });
            BenchmarkResult {
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
                    .unwrap_or(start.elapsed().as_millis() as u64)
                    as u128,
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
            }
        }
        Ok(out) => BenchmarkResult {
            name: name.to_string(),
            success: false,
            execution_time_ms: start.elapsed().as_millis(),
            metrics: serde_json::json!({}),
            errors: vec![format!(
                "Helper failed with status {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )],
        },
        Err(err) => BenchmarkResult {
            name: name.to_string(),
            success: false,
            execution_time_ms: start.elapsed().as_millis(),
            metrics: serde_json::json!({}),
            errors: vec![format!("Failed to invoke python3: {}", err)],
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

// ---------------------------------------------------------------------------
// Embedded Python helper
// ---------------------------------------------------------------------------
const PY_HELPER: &str = r#"
import argparse
import json
import subprocess
import sys
import time


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # pragma: no cover
        return None, [f"Unable to import torch: {exc}"]


def _gpu_info():
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda is not available"]

    devices = []
    for idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{idx}")
        props = torch.cuda.get_device_properties(device)
        info = {
            "index": idx,
            "gpu_model": getattr(props, "name", "Unknown"),
            "gcn_arch": getattr(props, "gcnArchName", "unknown"),
            "vram_gb": round(getattr(props, "total_memory", 0) / 1e9, 2),
            "compute_units": getattr(props, "multi_processor_count", 0),
            "max_clock_mhz": getattr(props, "max_frequency", 0) / 1e6,
            "tensor_cores": bool(getattr(props, "gcnArchName", "").startswith("gfx11")),
        }
        try:
            smi = subprocess.run(
                ["rocm-smi", "--showtemp", "--showpower", "--showuse", "--showclocks", "--json", "-d", str(idx)],
                check=False,
                capture_output=True,
                text=True,
            )
            if smi.returncode == 0 and smi.stdout.strip().startswith("{"):
                data = json.loads(smi.stdout)
                first_key = next(iter(data.keys()))
                entry = data.get(first_key, {})
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
        devices.append(info)

    return True, {"gpus": devices}, []


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
    try:
        import vllm
        from vllm import LLM, SamplingParams
        import logging
        # Suppress vLLM and related logging to avoid breaking JSON parsing
        logging.getLogger("vllm").setLevel(logging.ERROR)
    except Exception as exc:
        return False, {}, [f"vLLM not available: {exc}"]

    try:
        # Use a tiny model for benchmarking
        model_name = "facebook/opt-125m"
        
        # We use enforce_eager to save memory/time during quick benchmark
        llm = LLM(
            model=model_name, 
            trust_remote_code=True, 
            enforce_eager=True, 
            gpu_memory_utilization=0.7,
            disable_log_stats=True
        )
        
        prompts = ["The future of AI is", "ROCm performance on AMD is", "vLLM is a fast"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
        
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        throughput_samples = []
        for p in prompts:
            s = time.perf_counter()
            res = llm.generate([p], sampling_params)
            e = time.perf_counter() - s
            toks = len(res[0].outputs[0].token_ids)
            throughput_samples.append(toks / e if e > 0 else 0)

        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / elapsed
        
        return True, {
            "model": model_name,
            "throughput_tokens_per_sec": round(throughput, 2),
            "latency_ms": round((elapsed / len(prompts)) * 1000, 2),
            "throughput_samples": [round(x, 2) for x in throughput_samples]
        }, []
    except Exception as exc:
        err_msg = str(exc)
        if "Entry Not Found" in err_msg or "not found on the Hugging Face Hub" in err_msg:
             err_msg = f"Model {model_name} not found. Run 'huggingface-cli download {model_name}'"
        elif "authentication" in err_msg.lower():
             err_msg = "Hugging Face authentication required for some models. Run 'huggingface-cli login'"
             
        return True, {
            "model": "vLLM (Detected)",
            "throughput_tokens_per_sec": 0.0,
            "latency_ms": 0.0,
            "throughput_samples": []
        }, [f"vLLM detected but benchmark failed: {err_msg}"]


def _deepspeed():
    import warnings
    import logging
    import sys
    import os
    from io import StringIO
    
    # Suppress all warnings and unnecessary logging
    warnings.filterwarnings("ignore")
    logging.getLogger("deepspeed").setLevel(logging.CRITICAL)
    logging.getLogger("torch.distributed").setLevel(logging.CRITICAL)
    os.environ["DEEPSPEED_LOG_LEVEL"] = "ERROR"
    
    # Ensure ROCm environment BEFORE imports
    os.environ["DS_ACCELERATOR"] = "cuda"
    os.environ["ROCM_HOME"] = "/opt/rocm"
    os.environ["ROCM_PATH"] = "/opt/rocm"
    os.environ["HIP_PATH"] = "/opt/rocm"
    os.environ["CPATH"] = "/opt/rocm/include:/opt/rocm/include/hip:/opt/rocm/llvm/include"
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    os.environ["HIPCC_VERBOSE"] = "1"

    # Writable workspace for JIT builds
    import tempfile
    temp_root = tempfile.mkdtemp(prefix="ds_workspace_")
    build_dir = os.path.join(temp_root, "build")
    os.makedirs(build_dir, exist_ok=True)
    os.environ["DS_BUILD_DIR"] = build_dir

    try:
        import torch
        import torch.utils.cpp_extension as ce
        if not hasattr(ce, 'ROCM_HOME'): ce.ROCM_HOME = "/opt/rocm"
        if not hasattr(ce, 'HIP_HOME'): ce.HIP_HOME = "/opt/rocm"

        # Shadow deepspeed package in a writable directory to allow HIPIFICATION
        import deepspeed
        ds_orig_path = os.path.dirname(deepspeed.__file__)
        ds_shadow_root = os.path.join(temp_root, "shadow")
        ds_shadow_path = os.path.join(ds_shadow_root, "deepspeed")
        
        if not os.path.exists(ds_shadow_path):
            os.makedirs(ds_shadow_root, exist_ok=True)
            subprocess.run(["cp", "-r", ds_orig_path, ds_shadow_path], check=True)
            subprocess.run(["chmod", "-R", "u+w", ds_shadow_path], check=True)
        
        if ds_shadow_root not in sys.path:
            sys.path.insert(0, ds_shadow_root)
            
        import importlib
        importlib.reload(deepspeed)
        from deepspeed.accelerator import get_accelerator
    except Exception as exc:
        import traceback
        return False, {}, [f"DeepSpeed Shadowing/Loading failed: {traceback.format_exc()}"]
    
    try:
        accelerator = get_accelerator()
        device_name = accelerator.device_name(0)
        
        # Simple test: Initialize a small model and wrap it with DeepSpeed
        # We'll measure the time for a forward/backward pass with ZeRO-1
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024)
                )
            def forward(self, x):
                return self.net(x)

        model = SimpleModel().to(device_name)
        ds_config = {
            "train_batch_size": 4,
            "zero_optimization": {
                "stage": 1
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            },
            # Suppress DeepSpeed's own logging
            "steps_per_print": 999999,
            "wall_clock_breakdown": False
        }

        # Capture and suppress stdout/stderr during initialization
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        capture_stdout = StringIO()
        capture_stderr = StringIO()
        sys.stdout = capture_stdout
        sys.stderr = capture_stderr
        
        try:
            # Use dummy data
            model_engine, optimizer, _, _ = deepspeed.initialize(
                config=ds_config,
                model=model,
                model_parameters=model.parameters()
            )
        except Exception as e:
            # Restore streams before bailing
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Re-raise with captured context
            raise RuntimeError(f"{e}\n---STDOUT---\n{capture_stdout.getvalue()}\n---STDERR---\n{capture_stderr.getvalue()}")
        finally:
            if sys.stdout == capture_stdout:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        input_data = torch.randn(4, 1024).to(device_name)
        
        # Warmup
        for _ in range(2):
            output = model_engine(input_data)
            model_engine.backward(output.sum())
            model_engine.step()

        # Benchmark
        times = []
        for _ in range(5):
            accelerator.synchronize()
            start = time.perf_counter()
            output = model_engine(input_data)
            model_engine.backward(output.sum())
            model_engine.step()
            accelerator.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = 4 / avg_time  # samples per second

        # Clean up distributed backend if initialized
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except:
            pass
        
        return True, {
            "throughput_samples_per_sec": round(throughput, 2),
            "avg_latency_ms": round(avg_time * 1000, 2),
            "stage": 1,
            "accelerator": "rocm",
            "samples": [round(4/t, 2) for t in times],
            "version": deepspeed.__version__
        }, []
    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        return False, {}, [f"DeepSpeed benchmark failed: {error_details}"]


BENCHES = {
    "gpu-info": _gpu_info,
    "memory-bandwidth": _memory_bandwidth,
    "tensor-core": _tensor_core,
    "gemm": _gemm,
    "pytorch": _pytorch,
    "flash-attention": _flash_attention,
    "vllm": _vllm,
    "deepspeed": _deepspeed,
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
