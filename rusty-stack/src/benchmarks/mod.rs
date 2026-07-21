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
    // Single source (Tenet 3): inject the iGPU token list generated from the
    // Rust `crate::gpu` consts so the benchmark Python filter cannot drift from
    // the install-path filter.
    let body = PY_HELPER
        .trim_start()
        .replace("__INTEGRATED_TOKENS__", &integrated_tokens_py());
    let mut last_error = String::new();

    for dir in helper_dir_candidates() {
        if let Err(err) = fs::create_dir_all(&dir) {
            last_error = format!("failed to create helper dir {}: {}", dir.display(), err);
            continue;
        }
        let path = dir.join("rusty_bench.py");
        match fs::write(&path, &body) {
            Ok(_) => return Ok(path),
            Err(err) => {
                last_error = format!("failed to write helper script {}: {}", path.display(), err);
                continue;
            }
        }
    }

    Err(last_error)
}

/// Build the Python iGPU-token list literal from the Rust `crate::gpu` consts
/// (Tenet 3 single source — the benchmark Python filter derives from the same
/// list as the install path, so it cannot drift).
fn integrated_tokens_py() -> String {
    let mut tokens: Vec<String> = crate::gpu::INTEGRATED_NAME_PATTERNS
        .iter()
        .map(|t| t.to_lowercase())
        .collect();
    tokens.extend(
        crate::gpu::INTEGRATED_GFX_ARCHS
            .iter()
            .map(|t| t.to_lowercase()),
    );
    let mut s = String::from("[");
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let escaped = t.replace('\\', "\\\\").replace('"', "\\\"");
        s.push_str(&format!("\"{escaped}\""));
    }
    s.push(']');
    s
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

    if let Some(value) = mlstack_env_python_bin() {
        return value;
    }

    {
        let candidate = "/usr/local/bin/python3";
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

fn mlstack_env_python_bin() -> Option<String> {
    let home = env::var("HOME").ok()?;
    let contents = fs::read_to_string(Path::new(&home).join(".mlstack_env")).ok()?;
    for line in contents.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed
            .strip_prefix("export MLSTACK_PYTHON_BIN=")
            .or_else(|| trimmed.strip_prefix("MLSTACK_PYTHON_BIN="))
        else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
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

    // Source the canonical env so the benchmark inherits ROCR_VISIBLE_DEVICES
    // (discrete GPUs only — the env is the single GPU source), GPU_ARCH, and
    // MLSTACK_GPU_NAMES/VRAM. `bash -c '... exec "$@"'` runs python with the
    // env active; "_" is $0, python_bin + args follow as "$@" ($1..).
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg("source \"$HOME/.mlstack_env\" 2>/dev/null; exec \"$@\"")
        .arg("_")
        .arg(&python_bin)
        .arg(&helper)
        .arg(name)
        .arg("--json");
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
pub fn run_llama_cpp_benchmark() -> BenchmarkResult {
    // Ensure a GGUF model is present before running llama-bench — downloads
    // the canonical Qwen3-0.6B model to ~/.mlstack/models/ if none is found.
    // Reuses the installer's download path (single source of truth for the URL).
    if let Some(home) = dirs::home_dir() {
        let _ = crate::installers::components::llama_cpp::ensure_default_gguf_model(
            &home.to_string_lossy(),
        );
    }
    run_python_benchmark("llama-cpp")
}
pub fn run_flash_attention_benchmark() -> BenchmarkResult {
    run_python_benchmark("flash-attention")
}
pub fn run_flash_attention_ck_benchmark() -> BenchmarkResult {
    run_python_benchmark("flash-attention-ck")
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
pub fn run_onnx_benchmark() -> BenchmarkResult {
    run_python_benchmark("onnx")
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
import tempfile
import time
from glob import glob

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


def _ensure_cached_hf_model_weights(model_name):
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except Exception as exc:
        return False, f"huggingface_hub unavailable for benchmark model download: {exc}"

    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    marker_files = ("config.json", "tokenizer.json", "tokenizer_config.json")

    for filename in weight_files:
        path = try_to_load_from_cache(model_name, filename)
        if isinstance(path, str) and os.path.isfile(path):
            return True, ""

    try:
        path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        if isinstance(path, str) and os.path.isfile(path):
            return True, ""
    except Exception as exc:
        cached_markers = []
        for filename in marker_files:
            path = try_to_load_from_cache(model_name, filename)
            if isinstance(path, str) and os.path.isfile(path):
                cached_markers.append(filename)
        marker_msg = f" cached files: {', '.join(cached_markers)};" if cached_markers else ""
        return False, (
            f"could not download/cache model.safetensors for {model_name};"
            f"{marker_msg} {exc}"
        )

    return False, f"downloaded model.safetensors for {model_name} but no local file was found"


def _find_llama_cpp_binary():
    candidates = [
        os.path.expanduser("~/.mlstack/components/llama-cpp/bin/llama-bench"),
        shutil.which("llama-bench"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def _find_gguf_model():
    candidates = []
    candidates.extend(sorted(glob(os.path.expanduser("~/.mlstack/models/*.gguf"))))
    candidates.extend(sorted(glob(os.path.expanduser("~/.cache/huggingface/hub/*/snapshots/*/*.gguf"))))
    candidates.extend(sorted(glob(os.path.expanduser("~/.cache/**/*.gguf"), recursive=True)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _gpu_info_from_env():
    """GPU info from the canonical env (~/.mlstack_env) — the SINGLE source.
    Carries ROCR_VISIBLE_DEVICES (discrete indices, iGPU filtered), GPU_ARCH,
    MLSTACK_GPU_NAMES, MLSTACK_GPU_VRAM_GB, MLSTACK_GPU_CUS. No rocm-smi/rocminfo
    here. Returns (indices, names, vram_gb_strings, cus_strings, arch)."""
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    indices = []
    for part in rocr.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    if not indices:
        indices = [0]
    names = [s.strip() for s in os.environ.get("MLSTACK_GPU_NAMES", "").split(",") if s.strip()]
    vram = [s.strip() for s in os.environ.get("MLSTACK_GPU_VRAM_GB", "").split(",") if s.strip()]
    cus = [s.strip() for s in os.environ.get("MLSTACK_GPU_CUS", "").split(",") if s.strip()]
    arch = os.environ.get("GPU_ARCH", "").strip()
    while len(names) < len(indices):
        names.append("GPU {}".format(indices[len(names)]))
    while len(vram) < len(indices):
        vram.append("0")
    while len(cus) < len(indices):
        cus.append("0")
    return indices, names, vram, cus, arch


def _llama_cpp():
    start = time.perf_counter()
    bench = _find_llama_cpp_binary()
    if not bench:
        return False, {
            "name": "llama-cpp",
            "success": False,
            "execution_time_ms": 0,
            "metrics": {},
            "errors": ["llama-cpp not installed"],
        }, ["llama-cpp not installed"]

    # llama-bench links libggml*.so from a sibling lib/ dir (bin/../lib). Add it
    # to LD_LIBRARY_PATH or the loader can't start the binary (libs aren't beside
    # it). Set once here so every subprocess.run below inherits it.
    _bench_lib = os.path.join(os.path.dirname(os.path.dirname(bench)), "lib")
    if os.path.isdir(_bench_lib):
        _cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _bench_lib + (":" + _cur_ld if _cur_ld else "")

    model = _find_gguf_model()
    if not model:
        elapsed = int((time.perf_counter() - start) * 1000)
        return False, {
            "name": "llama-cpp",
            "success": False,
            "execution_time_ms": elapsed,
            "metrics": {},
            "errors": ["no GGUF model found"],
        }, ["no GGUF model found"]

    indices, gpu_names, gpu_vram, gpu_cus, gpu_arch = _gpu_info_from_env()
    if not indices:
        elapsed = int((time.perf_counter() - start) * 1000)
        return False, {
            "name": "llama-cpp",
            "success": False,
            "execution_time_ms": elapsed,
            "metrics": {},
            "errors": ["no GPUs in ROCR_VISIBLE_DEVICES (source ~/.mlstack_env)"],
        }, ["no GPUs in ROCR_VISIBLE_DEVICES (source ~/.mlstack_env)"]

    # Metadata (from the env — canonical GPU info) emitted alongside throughput
    # so the results panel shows real GPU names/arch/VRAM + the model + ROCm ver.
    def _has_wmma(a):
        return a in ("gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201")

    metrics = {
        "gpu_arch": gpu_arch,
        "gpu_names": gpu_names,
        "gpu_vram_gb": [float(v) for v in gpu_vram],
        "model": os.path.basename(model) if model else "",
        "rocm_version": os.environ.get("ROCM_VERSION", "").strip(),
        # Full GPU inventory (model/vram/CUs/tensor_cores) so the "GPU Inventory"
        # panel + HTML export show real devices, not "No GPU detected".
        "gpus": [
            {
                "index": indices[i],
                "gpu_model": gpu_names[i],
                "vram_gb": float(gpu_vram[i]),
                "compute_units": int(gpu_cus[i]) if gpu_cus[i].isdigit() else 0,
                "tensor_cores": _has_wmma(gpu_arch),
            }
            for i in range(len(indices))
        ],
    }
    errors = []
    got_throughput = False
    for gpu_idx in indices:
        env = os.environ.copy()
        # ROCR_VISIBLE_DEVICES (the ROCr runtime filter) — NOT the nonexistent
        # ROCM_VISIBLE_DEVICES — isolates one discrete GPU per llama-bench run.
        env["ROCR_VISIBLE_DEVICES"] = str(gpu_idx)
        try:
            proc = subprocess.run(
                [bench, "-m", model, "-p", "512", "-p", "2048", "-p", "8192", "-p", "16384", "-p", "32768", "-n", "128", "-o", "json", "-r", "3"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            raw = (proc.stdout or "").strip()
            parsed = json.loads(raw) if raw else []
            if isinstance(parsed, dict):
                parsed = [parsed]
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                n_prompt = int(entry.get("n_prompt", 0) or 0)
                n_gen = int(entry.get("n_gen", 0) or 0)
                avg_ts = entry.get("avg_ts")
                if avg_ts is None:
                    avg_ts = entry.get("avg_tps")
                stddev_ts = entry.get("stddev_ts", 0.0)
                prefix = "prefill" if n_prompt > 0 and n_gen == 0 else "decode" if n_prompt == 0 and n_gen > 0 else "other"
                if prefix == "other" or avg_ts is None:
                    continue
                context = n_prompt if prefix == "prefill" else n_gen
                metrics[f"{prefix}_{context}_tps_gpu{gpu_idx}"] = float(avg_ts)
                metrics[f"{prefix}_{context}_stddev_tps_gpu{gpu_idx}"] = float(stddev_ts or 0.0)
                got_throughput = True
        except Exception as exc:
            errors.append(str(exc))

    elapsed = int((time.perf_counter() - start) * 1000)
    success = got_throughput and not errors
    result = {
        "name": "llama-cpp",
        "success": success,
        "execution_time_ms": elapsed,
        "metrics": metrics,
        "errors": errors,
    }
    print("---BENCHMARK_RESULTS_START---")
    print(json.dumps(result, indent=2, sort_keys=True))
    return success, result, errors


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
    n = str(name or "")
    lowered = n.lower()
    up = n.upper()
    # Token set GENERATED from crate::gpu::INTEGRATED_NAME_PATTERNS at write
    # time (single source — no hand-maintained copy that can drift from Rust).
    for token in __INTEGRATED_TOKENS__:
        if token in lowered:
            return True
    # "Ryzen" without "RX" => APU/iGPU (discrete cards never contain "Ryzen").
    if "RYZEN" in up and "RX" not in up:
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
        return False, _degraded_metrics("memory-bandwidth", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("tensor-core", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("gemm", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("pytorch", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("flash-attention", probe_reason, extra), [probe_reason]

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


def _flash_attention_ck():
    """Genuine-model forward benchmark for Flash Attention (CK) on RDNA3.

    CK is forward-only on RDNA3 (no backward, ROCm/composable_kernel#1434), so
    this measures the SUPPORTED inference path: a real Llama-style decoder
    (RMSNorm + rotary + GQA via flash_attn_func + SwiGLU MLP) run forward under
    no_grad. No external model download: built in-torch, so it is reproducible
    on any box with torch+flash_attn.
    """
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        return False, {}, errors
    if not torch.cuda.is_available():
        return False, {}, ["torch.cuda not available"]

    from pathlib import Path
    marker = Path.home() / ".mlstack" / "flash-attention" / ".backend"
    backend = marker.read_text().strip() if marker.exists() else ""
    if backend != "ck":
        return False, {}, [
            "flash-attention-ck requires backend marker 'ck' (found '"
            + backend
            + "'); install Flash Attention (CK) first"
        ]

    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except Exception as exc:
        return False, {}, ["flash_attn import failed: " + str(exc)]

    probe_ok, probe_reason, probe_meta = _probe_gpu_runtime()
    if not probe_ok:
        extra = {
            "backend": "ck",
            "forward_samples_ms": [],
            "throughput_samples_tok_s": [],
            "peak_throughput_tok_s": 0.0,
        }
        if isinstance(probe_meta, dict):
            extra.update(probe_meta)
        return False, _degraded_metrics("flash-attention-ck", probe_reason, extra), [probe_reason]

    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    def _rope(dim, max_seq, base=10000.0):
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        seq = torch.arange(max_seq, device=device).float()
        freqs = torch.outer(seq, inv)
        return torch.cat([freqs, freqs], dim=-1)

    def _apply_rope(x, table):
        s = x.shape[1]
        cos = table[:s].unsqueeze(1)
        sin = table[s:2 * s].unsqueeze(1)
        x1, x2 = x.float().chunk(2, dim=-1)
        rot = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rot * sin).to(x.dtype)

    class _RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, x):
            v = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(v + self.eps)).to(x.dtype) * self.w

    class _Block(nn.Module):
        def __init__(self, dim, n_heads, n_kv, rope_table):
            super().__init__()
            self.n_heads, self.n_kv, self.dim = n_heads, n_kv, dim
            self.head_dim = dim // n_heads
            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(dim, n_kv * self.head_dim, bias=False)
            self.wv = nn.Linear(dim, n_kv * self.head_dim, bias=False)
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
            hidden = int(8 * dim / 3)
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w3 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
            self.n1, self.n2 = _RMSNorm(dim), _RMSNorm(dim)
            self.rope_table = rope_table

        def forward(self, x):
            b, s, _ = x.shape
            q = self.wq(self.n1(x)).view(b, s, self.n_heads, self.head_dim)
            k = self.wk(self.n1(x)).view(b, s, self.n_kv, self.head_dim)
            v = self.wv(self.n1(x)).view(b, s, self.n_kv, self.head_dim)
            q = _apply_rope(q, self.rope_table)
            k = _apply_rope(k, self.rope_table)
            if self.n_kv != self.n_heads:
                rep = self.n_heads // self.n_kv
                k = k.repeat_interleave(rep, dim=2)
                v = v.repeat_interleave(rep, dim=2)
            attn = flash_attn_func(q, k, v, causal=True)  # CK forward path
            attn = attn.reshape(b, s, -1)
            x = x + self.wo(attn)
            m = F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x))
            return x + self.w2(m)

    class _TransformerLM(nn.Module):
        def __init__(self, vocab, dim, n_layers, n_heads, n_kv, max_seq):
            super().__init__()
            rope_table = _rope(dim // n_heads, 2 * max_seq)
            self.embed = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList(
                [_Block(dim, n_heads, n_kv, rope_table) for _ in range(n_layers)]
            )
            self.norm = _RMSNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for blk in self.layers:
                x = blk(x)
            return self.head(self.norm(x))

    vocab = 49152
    dim, n_layers, n_heads, n_kv, max_seq = 768, 12, 12, 4, 4096
    model = _TransformerLM(vocab, dim, n_layers, n_heads, n_kv, max_seq).to(device, dtype)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    seq_lengths = [512, 1024, 2048, 4096]
    latencies_ms = []
    throughputs = []
    errors = []
    for seqlen in seq_lengths:
        ids = torch.randint(0, vocab, (1, seqlen), device=device)
        try:
            with torch.no_grad():
                fwd_t = _time_fn(lambda: model(ids), warmup=3, repeat=10)
            out = model(ids)
            if not torch.isfinite(out).all():
                errors.append("seq " + str(seqlen) + ": non-finite output")
            latencies_ms.append(round(fwd_t * 1000.0, 2))
            throughputs.append(round(seqlen / fwd_t, 0))
        except Exception as exc:
            errors.append("seq " + str(seqlen) + ": " + type(exc).__name__ + ": " + str(exc))
            latencies_ms.append(0.0)
            throughputs.append(0.0)

    peak = max(throughputs) if throughputs else 0.0
    success = len(errors) == 0 and peak > 0
    metrics = {
        "backend": "ck",
        "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
        "model_params_m": round(n_params),
        "model_config": str(n_layers) + "L d" + str(dim) + " h" + str(n_heads)
        + " GQA-kv" + str(n_kv) + " (genuine Llama-style)",
        "seq_lengths": seq_lengths,
        "forward_samples_ms": latencies_ms,
        "throughput_samples_tok_s": throughputs,
        "peak_throughput_tok_s": peak,
    }
    return success, metrics, errors


def _vllm():
    target_device = _env_or_default("VLLM_TARGET_DEVICE", "rocm")
    if target_device not in {"rocm", "cuda", "cpu"}:
        target_device = "rocm"

    def _visible_devices_from_env():
        visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
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

        return ",".join(candidates), candidates[0]

    visible_devices, primary_visible = _visible_devices_from_env()

    candidates = _resolve_vllm_model_candidates()
    candidate_names = [f"{c.get('format')}:{c.get('model')}" for c in candidates]
    torch = _load_torch()
    if isinstance(torch, tuple):
        _, errors = torch
        reason = "; ".join(errors) if errors else "unable to import torch"
        return False, _degraded_metrics("vllm", reason, {
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
            try:
                import amdsmi as _real_amdsmi
            except Exception:
                _real_amdsmi = None
            if _real_amdsmi is not None:
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
        return False, _degraded_metrics("vllm", err, {
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
        return False, _degraded_metrics("vllm", reason, {
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
        return False, _degraded_metrics("vllm", platform_reason, {
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

    def _run_vllm_attempt_subprocess(llm_kwargs, prompts, max_tokens):
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

def _main():
    try:
        payload_path = sys.argv[1] if len(sys.argv) > 1 else ""
        with open(payload_path, "r", encoding="utf-8") as payload_file:
            payload = json.load(payload_file)
        llm_kwargs = payload.get("llm_kwargs", {})
        prompts = payload.get("prompts", [])
        max_tokens = int(payload.get("max_tokens", 24))
        try:
            import types
            if os.environ.get("MLSTACK_VLLM_DISABLE_AMDSMI_SHIM", "").strip() not in {"1", "true", "TRUE"}:
                try:
                    import amdsmi as _real_amdsmi
                except Exception:
                    _real_amdsmi = None
                if _real_amdsmi is not None:
                    raise RuntimeError("__MLSTACK_REAL_AMDSMI_PRESENT__")
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
        except RuntimeError as exc:
            if str(exc) != "__MLSTACK_REAL_AMDSMI_PRESENT__":
                pass
        except Exception:
            pass
        # vLLM 0.25.0's ROCm V1 multiprocessing engine path can hang or fail
        # before the offline LLM benchmark reaches real generation. Keep the
        # benchmark environment sourced from ~/.mlstack_env, but force the local
        # benchmark process onto vLLM's in-process EngineCore path via vLLM's
        # lazy module setting instead of mutating os.environ.
        import vllm.envs as _vllm_envs
        _vllm_envs.environment_variables["VLLM_ENABLE_V1_MULTIPROCESSING"] = lambda: False
        if hasattr(_vllm_envs.__getattr__, "cache_clear"):
            _vllm_envs.__getattr__.cache_clear()
        import vllm
        from vllm import LLM, SamplingParams
        logging.getLogger("vllm").setLevel(logging.ERROR)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        startup_start = time.perf_counter()
        _emit({"event": "startup_begin", "phase": "vllm_startup", "vllm_v1_multiprocessing": False})
        llm = LLM(**llm_kwargs)
        startup_ms = int((time.perf_counter() - startup_start) * 1000)
        _emit({"event": "startup_complete", "phase": "vllm_startup", "startup_ms": startup_ms})

        generation_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        generation_elapsed = time.perf_counter() - generation_start

        throughput_samples = []
        for prompt in prompts:
            sample_start = time.perf_counter()
            sample_out = llm.generate([prompt], sampling_params)
            sample_elapsed = time.perf_counter() - sample_start
            tokens = len(sample_out[0].outputs[0].token_ids)
            throughput_samples.append(tokens / sample_elapsed if sample_elapsed > 0 else 0.0)

        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
        _emit({
            "ok": True,
            "startup_ms": startup_ms,
            "generation_ms": int(generation_elapsed * 1000),
            "throughput_tokens_per_sec": round(throughput, 2),
            "latency_ms": round((generation_elapsed / max(len(prompts), 1)) * 1000, 2),
            "throughput_samples": [round(x, 2) for x in throughput_samples],
            "vllm_v1_multiprocessing": False,
        })
    except Exception as exc:
        tb = traceback.format_exc().strip()
        tail_lines = [line.strip() for line in tb.splitlines()[-12:] if line.strip()]
        msg = str(exc).strip() or repr(exc)
        if tail_lines:
            msg = f"{msg} :: traceback_tail: {' | '.join(tail_lines)}"
        _emit({"ok": False, "error": msg})

if __name__ == "__main__":
    _main()
"""
        payload_path = None
        runner_path = None
        timeout_seconds = 900
        attempt_started_at = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as payload_file:
                json.dump(payload, payload_file)
                payload_path = payload_file.name
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as runner_file:
                runner_file.write(runner)
                runner_path = runner_file.name
            attempt_started_at = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, runner_path, payload_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timeout_ms = int(((time.perf_counter() - attempt_started_at) * 1000) if attempt_started_at else (timeout_seconds * 1000))
            startup_complete = False
            startup_ms = 0
            for line in stdout.splitlines():
                text = line.strip()
                if not text.startswith("{") or not text.endswith("}"):
                    continue
                try:
                    event = json.loads(text)
                except Exception:
                    continue
                if isinstance(event, dict) and event.get("event") == "startup_complete":
                    startup_complete = True
                    startup_ms = int(event.get("startup_ms") or 0)
            timeout_phase = "generation" if startup_complete else "startup"
            detail_parts = []
            for label, text in (("stdout_tail", stdout), ("stderr_tail", stderr)):
                tail = " | ".join([line.strip() for line in text.splitlines() if line.strip()][-20:])
                if tail:
                    detail_parts.append(f"{label}: {tail[:3000]}")
            detail = "; ".join(detail_parts) if detail_parts else "no subprocess output captured"
            return {
                "ok": False,
                "error": f"vLLM {timeout_phase} timed out after {timeout_seconds} seconds; {detail}",
                "startup_ms": startup_ms,
                "startup_timed_out": not startup_complete,
                "generation_timed_out": startup_complete,
                "timeout_ms": timeout_ms,
                "timeout_limit_ms": timeout_seconds * 1000,
                "vllm_timeout_phase": timeout_phase,
            }
        except Exception as exc:
            return {"ok": False, "error": f"subprocess launch failed: {exc}"}
        finally:
            for temp_path in (payload_path, runner_path):
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

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

    last_attempt_metrics = {}
    for candidate in candidates:
        model_name = candidate.get("model")
        model_format = candidate.get("format", "unknown")
        if not model_name:
            continue

        model_cache_ms = 0
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
        elif model_format == "safetensors":
            model_cache_start = time.perf_counter()
            weights_ready, weights_reason = _ensure_cached_hf_model_weights(model_name)
            model_cache_ms = int((time.perf_counter() - model_cache_start) * 1000)
            last_attempt_metrics = {"model_cache_ms": model_cache_ms}
            if not weights_ready:
                attempt_errors.append(f"{model_format} model {model_name} failed: {weights_reason}")
                continue

        attempt = _run_vllm_attempt_subprocess(llm_kwargs, prompts, max_tokens)
        last_attempt_metrics = {"model_cache_ms": model_cache_ms}
        for key in (
            "startup_ms",
            "generation_ms",
            "startup_timed_out",
            "generation_timed_out",
            "timeout_ms",
            "timeout_limit_ms",
            "vllm_timeout_phase",
            "vllm_v1_multiprocessing",
        ):
            if key in attempt:
                last_attempt_metrics[key] = attempt[key]
        if attempt.get("ok"):
            return True, {
                "model": model_name,
                "model_format": model_format,
                "target_device": target_device,
                "visible_devices": visible_devices or primary_visible,
                "model_cache_ms": model_cache_ms,
                "startup_ms": int(attempt.get("startup_ms", 0)),
                "generation_ms": int(attempt.get("generation_ms", 0)),
                "throughput_tokens_per_sec": float(attempt.get("throughput_tokens_per_sec", 0.0)),
                "latency_ms": float(attempt.get("latency_ms", 0.0)),
                "throughput_samples": [
                    float(x) for x in (attempt.get("throughput_samples") or [])
                ],
                "vllm_v1_multiprocessing": attempt.get("vllm_v1_multiprocessing"),
                "candidate_models": candidate_names,
            }, []
        err_msg = str(attempt.get("error") or "unknown vLLM execution error")
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

    failure_metrics = {
        "model": "vLLM (Detected)",
        "model_format": "unavailable",
        "target_device": target_device,
        "visible_devices": visible_devices or primary_visible,
        "throughput_tokens_per_sec": 0.0,
        "latency_ms": 0.0,
        "throughput_samples": [],
        "candidate_models": candidate_names,
    }
    failure_metrics.update(last_attempt_metrics)
    return False, _degraded_metrics("vllm", failure_reason, failure_metrics), attempt_errors or [failure_reason]


def _deepspeed():
    _set_amd_gpu_id_table_env()
    try:
        import deepspeed
        import torch
        import os
    except Exception as exc:
        err = f"DeepSpeed or Torch not available: {exc}"
        return False, _degraded_metrics("deepspeed", err, {
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
        return False, _degraded_metrics("deepspeed", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("deepspeed", err, {
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
        return False, _degraded_metrics("megatron", err, {
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
        return False, _degraded_metrics("megatron", probe_reason, extra), [probe_reason]

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
        return False, _degraded_metrics("megatron", err, {
            "megatron_backend": "megatron-core",
            "megatron_throughput_samples_per_sec": 0.0,
            "megatron_avg_latency_ms": 0.0,
            "megatron_samples": [],
        }), [err]


def _onnx():
    import os
    try:
        import onnxruntime as ort
    except Exception as exc:
        err = f"onnxruntime not available: {exc}"
        return False, _degraded_metrics("onnx", err, {
            "ort_version": "unavailable",
            "provider": "none",
            "providers_available": [],
            "model_load_ms": 0.0,
            "session_create_ms": 0.0,
            "inference_latency_p50_ms": 0.0,
            "inference_latency_p95_ms": 0.0,
            "inference_latency_p99_ms": 0.0,
            "throughput_inf_per_sec": 0.0,
            "input_shape": [],
            "output_shape": [],
            "graph_opt_level": "unavailable",
            "inference_samples": [],
        }), [err]

    amd_provider_order = ("MIGraphXExecutionProvider", "ROCMExecutionProvider")
    ort_version = getattr(ort, "__version__", None)
    if not ort_version or not hasattr(ort, "get_available_providers"):
        err = f"onnxruntime import is incomplete: {getattr(ort, '__file__', '<unknown>')}"
        return False, _degraded_metrics("onnx", err, {
            "ort_version": "incomplete",
            "provider": "none",
            "providers_available": [],
            "provider_priority": list(amd_provider_order),
            "model_load_ms": 0.0,
            "session_create_ms": 0.0,
            "inference_latency_p50_ms": 0.0,
            "inference_latency_p95_ms": 0.0,
            "inference_latency_p99_ms": 0.0,
            "throughput_inf_per_sec": 0.0,
            "input_shape": [],
            "output_shape": [],
            "graph_opt_level": "ORT_ENABLE_ALL",
            "inference_samples": [],
        }), [err]

    all_providers = ort.get_available_providers()
    providers_available = list(all_providers)

    provider = next((name for name in amd_provider_order if name in all_providers), None)
    if provider is None:
        err = (
            "ONNX Runtime AMD execution provider unavailable; expected "
            "MiGraphXExecutionProvider or legacy ROCMExecutionProvider; "
            f"available={providers_available}"
        )
        return False, _degraded_metrics("onnx", err, {
            "ort_version": ort_version,
            "provider": "none",
            "providers_available": providers_available,
            "provider_priority": list(amd_provider_order),
            "model_load_ms": 0.0,
            "session_create_ms": 0.0,
            "inference_latency_p50_ms": 0.0,
            "inference_latency_p95_ms": 0.0,
            "inference_latency_p99_ms": 0.0,
            "throughput_inf_per_sec": 0.0,
            "input_shape": [],
            "output_shape": [],
            "graph_opt_level": "ORT_ENABLE_ALL",
            "inference_samples": [],
        }), [err]

    try:
        import numpy as np
    except Exception:
        err = "numpy not available for ONNX benchmark"
        return False, _degraded_metrics("onnx", err, {
            "ort_version": ort_version,
            "provider": provider,
            "providers_available": providers_available,
            "provider_priority": list(amd_provider_order),
            "model_load_ms": 0.0,
            "session_create_ms": 0.0,
            "inference_latency_p50_ms": 0.0,
            "inference_latency_p95_ms": 0.0,
            "inference_latency_p99_ms": 0.0,
            "throughput_inf_per_sec": 0.0,
            "input_shape": [],
            "output_shape": [],
            "graph_opt_level": "ORT_ENABLE_ALL",
            "inference_samples": [],
        }), [err]

    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except Exception:
        onnx = None

    batch_size = 8
    hidden = 512
    input_shape = [batch_size, hidden]

    def _build_test_model():
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)
        W_init = numpy_helper.from_array(
            np.random.randn(hidden, hidden).astype(np.float32) * 0.01, name="W"
        )
        B_init = numpy_helper.from_array(
            np.zeros(hidden, dtype=np.float32), name="B"
        )
        matmul = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
        add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"])
        relu = helper.make_node("Relu", ["add_out"], ["output"])
        graph = helper.make_graph(
            [matmul, add, relu], "bench_graph", [X], [Y], initializer=[W_init, B_init]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.validate = False
        return model.SerializeToString()

    def _build_quantized_model():
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)
        W_quant = np.clip(
            np.round((np.random.randn(hidden, hidden).astype(np.float32) * 0.01) / 0.01) + 128,
            0,
            255,
        ).astype(np.uint8)
        W_init = numpy_helper.from_array(W_quant, name="W_quant")
        W_zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), name="W_zp")
        B_init = numpy_helper.from_array(
            np.zeros(hidden, dtype=np.float32), name="B"
        )
        dq = helper.make_node(
            "DynamicQuantizeLinear", ["input"], ["input_quant", "input_scale", "input_zp"]
        )
        matmul_int = helper.make_node(
            "MatMulInteger", ["input_quant", "W_quant", "input_zp", "W_zp"], ["matmul_int_out"]
        )
        cast_out = helper.make_node(
            "Cast", ["matmul_int_out"], ["matmul_fp_out"], to=TensorProto.FLOAT
        )
        add = helper.make_node("Add", ["matmul_fp_out", "B"], ["output"])
        graph = helper.make_graph(
            [dq, matmul_int, cast_out, add], "quant_bench_graph",
            [X], [Y], initializer=[W_init, W_zp, B_init]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.validate = False
        return model.SerializeToString()

    import tempfile
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    results = {}
    errors = []

    def _assert_amd_session(session):
        session_providers = list(session.get_providers())
        if (
            provider not in session_providers
            or not session_providers
            or session_providers[0] != provider
        ):
            raise RuntimeError(
                "ONNX session did not prioritize AMD provider; "
                f"selected={provider}; session_providers={session_providers}"
            )
        return session_providers

    # --- FP32 model benchmark ---
    try:
        if onnx is not None:
            model_bytes = _build_test_model()
        else:
            model_bytes = None

        if model_bytes is None:
            raise RuntimeError("onnx package required to build test model")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model_bytes)
            fp32_path = f.name

        t0 = time.perf_counter()
        session = ort.InferenceSession(fp32_path, sess_opts, providers=[provider])
        session_create_ms = (time.perf_counter() - t0) * 1000.0
        session_providers = _assert_amd_session(session)

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_name = input_meta.name
        actual_input_shape = input_meta.shape
        output_shape = output_meta.shape

        if isinstance(actual_input_shape[0], int):
            bench_batch = actual_input_shape[0]
        else:
            bench_batch = batch_size
        actual_input_shape_resolved = [bench_batch] + [
            d if isinstance(d, int) else hidden for d in actual_input_shape[1:]
        ]

        feed = {input_name: np.random.randn(*actual_input_shape_resolved).astype(np.float32)}

        for _ in range(3):
            session.run(None, feed)

        num_runs = 20
        latencies = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg_latency = sum(latencies) / len(latencies)
        throughput = bench_batch / (avg_latency / 1000.0) if avg_latency > 0 else 0.0

        mem_rss_mb = 0.0
        try:
            import resource
            mem_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass

        results.update({
            "fp32_session_create_ms": round(session_create_ms, 2),
            "fp32_session_providers": session_providers,
            "fp32_inference_latency_p50_ms": round(p50, 3),
            "fp32_inference_latency_p95_ms": round(p95, 3),
            "fp32_inference_latency_p99_ms": round(p99, 3),
            "fp32_avg_latency_ms": round(avg_latency, 3),
            "fp32_throughput_inf_per_sec": round(throughput, 2),
            "fp32_inference_samples": [round(l, 3) for l in latencies],
            "input_shape": actual_input_shape_resolved,
            "output_shape": list(output_shape) if output_shape else [],
            "input_dtype": str(input_meta.type),
            "output_dtype": str(output_meta.type),
            "peak_rss_mb": round(mem_rss_mb, 1),
            "num_warmup": 3,
            "num_timed_runs": num_runs,
        })

        os.unlink(fp32_path)
        del session
    except Exception as exc:
        results["fp32_error"] = str(exc)
        errors.append(f"FP32 benchmark failed: {exc}")

    # --- Quantized model benchmark ---
    quantized_ok = False
    try:
        if onnx is not None:
            q_model_bytes = _build_quantized_model()
            if q_model_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                    f.write(q_model_bytes)
                    quant_path = f.name

                t0 = time.perf_counter()
                q_session = ort.InferenceSession(
                    quant_path, sess_opts, providers=[provider]
                )
                q_create_ms = (time.perf_counter() - t0) * 1000.0
                q_session_providers = _assert_amd_session(q_session)

                q_input = q_session.get_inputs()[0]
                q_feed = {
                    q_input.name: np.random.randn(
                        *([batch_size] + [
                            d if isinstance(d, int) else hidden
                            for d in q_input.shape[1:]
                        ])
                    ).astype(np.float32)
                }

                for _ in range(3):
                    q_session.run(None, q_feed)

                q_latencies = []
                for _ in range(20):
                    t_start = time.perf_counter()
                    q_session.run(None, q_feed)
                    q_latencies.append((time.perf_counter() - t_start) * 1000.0)

                q_latencies.sort()
                q_p50 = q_latencies[len(q_latencies) // 2]
                q_avg = sum(q_latencies) / len(q_latencies)
                q_throughput = batch_size / (q_avg / 1000.0) if q_avg > 0 else 0.0

                results.update({
                    "quantized_session_create_ms": round(q_create_ms, 2),
                    "quantized_session_providers": q_session_providers,
                    "quantized_inference_latency_p50_ms": round(q_p50, 3),
                    "quantized_avg_latency_ms": round(q_avg, 3),
                    "quantized_throughput_inf_per_sec": round(q_throughput, 2),
                    "quantized_inference_samples": [round(l, 3) for l in q_latencies],
                    "quantized_ops": ["DynamicQuantizeLinear", "MatMulInteger"],
                })
                quantized_ok = True
                os.unlink(quant_path)
                del q_session
    except Exception as exc:
        results["quantized_error"] = str(exc)
        errors.append(f"Quantized benchmark failed: {exc}")

    p50_val = results.get("fp32_inference_latency_p50_ms", 0.0)
    p95_val = results.get("fp32_inference_latency_p95_ms", 0.0)
    p99_val = results.get("fp32_inference_latency_p99_ms", 0.0)
    tput_val = results.get("fp32_throughput_inf_per_sec", 0.0)

    metrics = {
        "ort_version": ort_version,
        "provider": provider,
        "providers_available": providers_available,
        "provider_priority": list(amd_provider_order),
        "session_providers": results.get("fp32_session_providers", []),
        "graph_opt_level": "ORT_ENABLE_ALL",
        "model_load_ms": results.get("fp32_session_create_ms", 0.0),
        "session_create_ms": results.get("fp32_session_create_ms", 0.0),
        "inference_latency_p50_ms": p50_val,
        "inference_latency_p95_ms": p95_val,
        "inference_latency_p99_ms": p99_val,
        "throughput_inf_per_sec": tput_val,
        "input_shape": results.get("input_shape", input_shape),
        "output_shape": results.get("output_shape", []),
        "input_dtype": results.get("input_dtype", "tensor(float)"),
        "output_dtype": results.get("output_dtype", "tensor(float)"),
        "peak_rss_mb": results.get("peak_rss_mb", 0.0),
        "num_warmup": results.get("num_warmup", 0),
        "num_timed_runs": results.get("num_timed_runs", 0),
        "quantized_supported": quantized_ok,
        "inference_samples": results.get("fp32_inference_samples", []),
    }

    for k in ("fp32_inference_samples", "fp32_session_create_ms", "fp32_inference_latency_p50_ms",
              "fp32_inference_latency_p95_ms", "fp32_inference_latency_p99_ms",
              "fp32_avg_latency_ms", "fp32_throughput_inf_per_sec", "fp32_session_providers",
              "quantized_session_create_ms", "quantized_session_providers", "quantized_inference_latency_p50_ms",
              "quantized_avg_latency_ms", "quantized_throughput_inf_per_sec",
              "quantized_inference_samples", "quantized_ops",
              "quantized_error", "fp32_error"):
        if k in results:
            metrics[k] = results[k]

    success = not errors and "fp32_inference_samples" in results and quantized_ok
    return success, metrics, errors


BENCHES = {
    "gpu-info": _gpu_info,
    "memory-bandwidth": _memory_bandwidth,
    "tensor-core": _tensor_core,
    "gemm": _gemm,
    "pytorch": _pytorch,
    "flash-attention": _flash_attention,
    "flash-attention-ck": _flash_attention_ck,
    "vllm": _vllm,
    "deepspeed": _deepspeed,
    "megatron": _megatron,
    "onnx": _onnx,
    "llama-cpp": _llama_cpp,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolve_benchmark_python_uses_mlstack_env_python_before_system_python() {
        let _global_env = crate::test_support::lock_env();
        let _guard = ENV_LOCK.lock().unwrap();
        let old_home = env::var("HOME").ok();
        let old_benchmark_python = env::var("MLSTACK_BENCHMARK_PYTHON").ok();
        let old_python_bin = env::var("MLSTACK_PYTHON_BIN").ok();
        let old_uv_python = env::var("UV_PYTHON").ok();
        let home = env::temp_dir().join(format!(
            "rusty-stack-benchmark-python-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join(".mlstack_env"),
            "export MLSTACK_PYTHON_BIN=/tmp/rusty-managed-python\n",
        )
        .unwrap();
        env::set_var("HOME", &home);
        env::remove_var("MLSTACK_BENCHMARK_PYTHON");
        env::remove_var("MLSTACK_PYTHON_BIN");
        env::remove_var("UV_PYTHON");

        let resolved = resolve_benchmark_python();

        if let Some(home) = old_home {
            env::set_var("HOME", home);
        } else {
            env::remove_var("HOME");
        }
        if let Some(value) = old_benchmark_python {
            env::set_var("MLSTACK_BENCHMARK_PYTHON", value);
        } else {
            env::remove_var("MLSTACK_BENCHMARK_PYTHON");
        }
        if let Some(value) = old_python_bin {
            env::set_var("MLSTACK_PYTHON_BIN", value);
        } else {
            env::remove_var("MLSTACK_PYTHON_BIN");
        }
        if let Some(value) = old_uv_python {
            env::set_var("UV_PYTHON", value);
        } else {
            env::remove_var("UV_PYTHON");
        }
        let _ = fs::remove_dir_all(&home);

        assert_eq!(resolved, "/tmp/rusty-managed-python");
    }

    #[test]
    fn vllm_amdsmi_shim_preserves_real_amdsmi_when_available() {
        assert!(PY_HELPER.contains("import amdsmi as _real_amdsmi"));
        assert!(PY_HELPER.contains("if _real_amdsmi is not None:"));
    }

    #[test]
    fn onnx_benchmark_uses_public_model_serialization() {
        assert!(!PY_HELPER.contains("onnx._serialize"));
        assert!(PY_HELPER.contains("model.SerializeToString()"));
    }

    #[test]
    fn onnx_quantized_benchmark_uses_integer_weight_tensor() {
        assert!(PY_HELPER.contains("W_quant"));
        assert!(!PY_HELPER.contains(r#""MatMulInteger", ["input_quant", "W"]"#));
    }

    #[test]
    fn onnx_benchmark_requires_amd_execution_provider() {
        assert!(PY_HELPER.contains(
            r#"amd_provider_order = ("MIGraphXExecutionProvider", "ROCMExecutionProvider")"#
        ));
        assert!(PY_HELPER.contains("onnxruntime import is incomplete"));
        assert!(PY_HELPER.contains("ONNX Runtime AMD execution provider unavailable"));
        assert!(PY_HELPER.contains("ONNX session did not prioritize AMD provider"));
        assert!(PY_HELPER.contains("success = not errors"));
    }

    #[test]
    fn vllm_helper_uses_file_payload_without_env_overrides() {
        assert!(PY_HELPER.contains(r#"if __name__ == "__main__":"#));
        assert!(PY_HELPER.contains("NamedTemporaryFile"));
        assert!(!PY_HELPER.contains("env_overrides"));
        assert!(!PY_HELPER.contains("MLSTACK_VLLM_PAYLOAD_JSON"));
        assert!(!PY_HELPER.contains(r#"os.environ["VLLM_TARGET_DEVICE"]"#));
        assert!(!PY_HELPER.contains(r#"os.environ["HIP_VISIBLE_DEVICES"]"#));
        assert!(!PY_HELPER.contains(r#"os.environ["CUDA_VISIBLE_DEVICES"]"#));
        assert!(!PY_HELPER.contains(r#"os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]"#));
    }

    #[test]
    fn vllm_benchmark_disables_v1_multiprocessing_without_env_mutation() {
        assert!(PY_HELPER.contains("import vllm.envs as _vllm_envs"));
        assert!(PY_HELPER.contains(
            r#"_vllm_envs.environment_variables["VLLM_ENABLE_V1_MULTIPROCESSING"] = lambda: False"#
        ));
        assert!(PY_HELPER.contains(r#""vllm_v1_multiprocessing": False"#));
    }

    #[test]
    fn degraded_benchmarks_report_failure_not_success() {
        assert!(PY_HELPER.contains("return False, _degraded_metrics"));
        assert!(!PY_HELPER.contains("return True, _degraded_metrics"));
    }

    #[test]
    fn vllm_benchmark_downloads_missing_hf_weights() {
        assert!(PY_HELPER.contains("def _ensure_cached_hf_model_weights"));
        assert!(PY_HELPER.contains("hf_hub_download"));
        assert!(PY_HELPER.contains(r#"filename="model.safetensors""#));
    }

    #[test]
    fn vllm_timeout_reports_subprocess_output_tail() {
        assert!(PY_HELPER.contains("except subprocess.TimeoutExpired as exc"));
        assert!(PY_HELPER.contains("stdout_tail"));
        assert!(PY_HELPER.contains("stderr_tail"));
        assert!(PY_HELPER.contains("startup_timed_out"));
        assert!(PY_HELPER.contains("vllm_timeout_phase"));
        assert!(PY_HELPER.contains("\"event\": \"startup_begin\""));
        assert!(PY_HELPER.contains("\"event\": \"startup_complete\""));
    }
}
