use crate::benchmark_logs;
use crate::state::{Category, Component};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct VerificationCommand {
    pub label: String,
    pub target_id: String,
    pub program: String,
    pub args: Vec<String>,
    pub modules: Vec<String>,
}

pub fn python_interpreters() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Ok(value) = env::var("MLSTACK_PYTHON_BIN") {
        push_python_candidate(&mut candidates, value);
    }
    if let Ok(value) = env::var("AITER_VENV_PYTHON") {
        push_python_candidate(&mut candidates, value);
    }
    if let Ok(value) = env::var("UV_PYTHON") {
        push_python_candidate(&mut candidates, value);
    }
    for key in [
        "WANDB_VENV_PYTHON",
        "MEGATRON_VENV_PYTHON",
        "MPI4PY_VENV_PYTHON",
        "FLASH_ATTENTION_VENV_PYTHON",
        "PYTORCH_VENV_PYTHON",
        "DEEPSPEED_VENV_PYTHON",
        "VLLM_VENV_PYTHON",
    ] {
        if let Ok(value) = env::var(key) {
            push_python_candidate(&mut candidates, value);
        }
    }
    let home = resolve_component_user_home();
    let venv = Path::new(&home)
        .join("rocm_venv")
        .join("bin")
        .join("python");
    if venv.exists() {
        candidates.push(venv.to_string_lossy().to_string());
    }

    // Anaconda/Miniconda candidates
    let conda_path = Path::new(&home)
        .join("anaconda3")
        .join("bin")
        .join("python");
    if conda_path.exists() {
        candidates.push(conda_path.to_string_lossy().to_string());
    }
    let miniconda_path = Path::new(&home)
        .join("miniconda3")
        .join("bin")
        .join("python");
    if miniconda_path.exists() {
        candidates.push(miniconda_path.to_string_lossy().to_string());
    }

    if Path::new("/usr/local/bin/python3").exists() {
        candidates.push("/usr/local/bin/python3".to_string());
    }
    if command_exists("python3.13") {
        candidates.push("python3.13".to_string());
    }
    if command_exists("python3.12") {
        candidates.push("python3.12".to_string());
    }
    if command_exists("python3") {
        candidates.push("python3".to_string());
    }
    if command_exists("python") {
        candidates.push("python".to_string());
    }

    let mut roots = vec![PathBuf::from(&home)];
    if let Ok(cwd) = env::current_dir() {
        roots.push(cwd.clone());
        roots.push(cwd.join("rusty-stack"));
    }
    for root in roots {
        push_component_venv_candidates(&mut candidates, &root);
    }

    dedupe_python_candidates_in_order(candidates)
}

fn dedupe_python_candidates_in_order(candidates: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) {
            ordered.push(candidate);
        }
    }
    ordered
}

fn push_component_venv_candidates(candidates: &mut Vec<String>, root: &Path) {
    for rel in [
        "wandb_venv/bin/python",
        "megatron_rocm_venv/bin/python",
        "Megatron-LM/megatron_rocm_venv/bin/python",
        "mpi4py_venv/bin/python",
        "flash_attention_venv/bin/python",
        "vllm_venv/bin/python",
        ".mlstack/venvs/aiter/bin/python",
        ".mlstack/venvs/vllm/bin/python",
        "pytorch_rocm_venv/bin/python",
    ] {
        let candidate = root.join(rel);
        if candidate.exists() {
            candidates.push(candidate.to_string_lossy().to_string());
        }
    }
}

fn push_python_candidate(candidates: &mut Vec<String>, value: String) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return;
    }
    let candidate = trimmed.to_string();
    if Path::new(&candidate).exists() || command_exists(&candidate) {
        candidates.push(candidate);
    }
}

pub fn is_component_installed(component: &Component, python_candidates: &[String]) -> bool {
    if component.category == Category::Verification {
        return false;
    }
    is_component_installed_by_id(&component.id, python_candidates)
}

pub fn is_component_installed_by_id(component_id: &str, python_candidates: &[String]) -> bool {
    let home = resolve_component_user_home();
    let has_benchmark_logs = has_benchmark_log_dirs();
    match component_id {
        // ROCm requires BOTH version file AND functional rocminfo (not just binary existence)
        // This prevents false positives from partial downloads
        "rocm" => {
            // Check for version file in multiple locations
            let version_exists = path_exists("/opt/rocm/.info/version")
                || path_exists("/opt/rocm/version")
                || path_exists("/usr/lib/rocm/.info/version");

            // Check if rocminfo actually works (not just exists)
            let rocminfo_functional = if let Ok(output) = std::process::Command::new("rocminfo")
                .arg("--version")
                .output()
            {
                output.status.success()
            } else {
                // Try from ROCm path
                if let Ok(output) = std::process::Command::new("/opt/rocm/bin/rocminfo")
                    .arg("--version")
                    .output()
                {
                    output.status.success()
                } else {
                    false
                }
            };

            // Require both: version file AND functional rocminfo
            // This ensures we don't report partial installs as complete
            version_exists && rocminfo_functional
        }
        "pytorch" => python_any(python_candidates, &["torch"]),
        "triton" => python_any(python_candidates, &["triton"]),
        "mpi4py" => python_any(python_candidates, &["mpi4py"]),
        "deepspeed" => python_any(python_candidates, &["deepspeed"]),
        "ml-stack-core" => {
            python_any(python_candidates, &["stans_ml_stack"]) || repo_has_ml_stack_core()
        }
        "flash-attn" => {
            python_any(python_candidates, &["flash_attn", "flash_attn_2"])
                || path_exists(home_path(&home, &["ml_stack", "flash_attn_amd"]))
                || path_exists(home_path(&home, &["ml_stack", "flash_attn_amd_direct"]))
        }
        "megatron" => {
            path_exists(home_path(&home, &["megatron", "Megatron-LM"]))
                || python_any(python_candidates, &["megatron"])
        }
        "vllm" => {
            python_candidates
                .iter()
                .any(|python| python_exec(python, vllm_runtime_check_snippet()))
                || path_exists(home_path(&home, &["vllm_build"]))
                || path_exists(home_path(&home, &["vllm_py313"]))
        }
        "aiter" => python_any(python_candidates, &["aiter"]),
        "vllm-studio" => {
            command_exists("vllm-studio") || path_exists(home_path(&home, &["vllm-studio"]))
        }
        "onnx" => {
            python_any(python_candidates, &["onnxruntime"])
                || path_exists(home_path(&home, &["onnxruntime_build"]))
        }
        "bitsandbytes" => {
            python_any(python_candidates, &["bitsandbytes"])
                || path_exists(home_path(&home, &["ml_stack", "bitsandbytes"]))
        }
        // rocm-smi - check multiple locations and verify it works
        "rocm-smi" => {
            let rocm_smi_paths = [
                "/opt/rocm/bin/rocm-smi",
                "/usr/bin/rocm-smi",
                "/usr/local/bin/rocm-smi",
            ];
            for path in &rocm_smi_paths {
                if path_exists(path) {
                    // Verify it actually works
                    if let Ok(output) = std::process::Command::new(path).arg("--showuse").output() {
                        if output.status.success() {
                            return true;
                        }
                    }
                }
            }
            // Also check if rocm-smi is in PATH and works
            if command_exists("rocm-smi") {
                if let Ok(output) = std::process::Command::new("rocm-smi")
                    .arg("--showuse")
                    .output()
                {
                    return output.status.success();
                }
            }
            false
        }
        "migraphx" => {
            python_any(python_candidates, &["migraphx"])
                || path_exists(home_path(&home, &["migraphx_build"]))
        }
        "pytorch-profiler" => python_any(python_candidates, &["torch"]),
        "wandb" => python_any(python_candidates, &["wandb"]),
        "permanent-env" => env_file_has_permanent(&home_path(&home, &[".mlstack_env"])),
        "basic-env" => path_exists(home_path(&home, &[".mlstack_env"])),
        "enhanced-env" => env_file_has_enhanced(&home_path(&home, &[".mlstack_env"])),
        "vllm-performance" => has_benchmark_logs,
        "deepspeed-performance" => has_benchmark_logs,
        "megatron-performance" => has_benchmark_logs,
        "all-benchmarks" => has_benchmark_logs,
        "comfyui" => {
            // Check for ComfyUI installation
            // Default location: $HOME/ComfyUI
            // User location: /mnt/e0f7c1a8-b834-4827-b579-0251b006bc1f/ComfyUI/
            // Check for .git folder and main.py
            let default_path = home_path(&home, &["ComfyUI"]);
            path_exists(default_path.join(".git")) || path_exists(default_path.join("main.py"))
        }
        _ => false,
    }
}

pub fn verification_commands(
    component_id: &str,
    python_candidates: &[String],
) -> Vec<VerificationCommand> {
    match component_id {
        "verify-basic" => basic_verification_commands(python_candidates),
        "verify-enhanced" => enhanced_verification_commands(python_candidates),
        "verify-build" => build_verification_commands(python_candidates),
        _ => Vec::new(),
    }
}

pub fn component_verification_commands(
    component_id: &str,
    python_candidates: &[String],
) -> Vec<VerificationCommand> {
    match component_id {
        "rocm" => vec![shell_command("ROCm info", "rocm", "rocminfo", &[])],
        "pytorch" => vec![python_command(
            "PyTorch",
            "pytorch",
            &["torch"],
            python_candidates,
            "import torch, sys; print(torch.__version__); print('hip', getattr(torch.version, 'hip', None)); sys.exit(0 if getattr(torch.version, 'hip', None) else 1)",
        )],
        "triton" => vec![python_command(
            "Triton",
            "triton",
            &["triton"],
            python_candidates,
            "import triton; import triton._C; print(triton.__version__)",
        )],
        "mpi4py" => vec![python_command(
            "MPI4Py",
            "mpi4py",
            &["mpi4py"],
            python_candidates,
            "import importlib.metadata as m; import mpi4py; print(getattr(mpi4py, '__version__', m.version('mpi4py')))",
        )],
        "deepspeed" => vec![python_command(
            "DeepSpeed",
            "deepspeed",
            &["deepspeed"],
            python_candidates,
            "import deepspeed; print(deepspeed.__version__)",
        )],
        "ml-stack-core" => vec![python_command(
            "ML Stack Core",
            "ml-stack-core",
            &["stans_ml_stack"],
            python_candidates,
            "import stans_ml_stack; print(getattr(stans_ml_stack, '__version__', 'ok'))",
        )],
        "flash-attn" => vec![python_command(
            "Flash Attention",
            "flash-attn",
            &["flash_attention_amd", "flash_attn", "flash_attn_2"],
            python_candidates,
            "import importlib, sys;\nmodules=['flash_attention_amd','flash_attn','flash_attn_2'];\nloaded=False;\nerrors=[];\nfor name in modules:\n    try:\n        importlib.import_module(name);\n        print(name);\n        loaded=True;\n        break\n    except Exception as exc:\n        errors.append(f'{name}: {exc}');\nif not loaded:\n    print('Flash Attention import errors:', '; '.join(errors));\n    raise SystemExit(1)",
        )],
        "megatron" => vec![python_command(
            "Megatron-LM",
            "megatron",
            &["megatron"],
            python_candidates,
            "import megatron; print('megatron ok')",
        )],
        "vllm" => vec![python_command(
            "vLLM",
            "vllm",
            &["vllm"],
            python_candidates,
            vllm_runtime_check_snippet(),
        )],
        "aiter" => vec![python_command(
            "AITER",
            "aiter",
            &["aiter"],
            python_candidates,
            "import aiter; import aiter.torch; print(getattr(aiter, '__version__', 'ok'))",
        )],
        "vllm-studio" => vec![shell_command(
            "vLLM Studio",
            "vllm-studio",
            "bun",
            &["--version"],
        )],
        "onnx" => vec![python_command(
            "ONNX Runtime",
            "onnx",
            &["onnxruntime"],
            python_candidates,
            "import onnxruntime as ort; import pathlib; import os; import sys; print('Version:', ort.__version__); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob('libonnxruntime_providers_rocm.so')); [os.environ.update({'ORT_ROCM_EP_PROVIDER_PATH': str(l)}) for l in libs[:1]]; providers=ort.get_available_providers(); print('Providers:', providers); sys.exit(0 if 'ROCMExecutionProvider' in providers else 1)",
        )],
        "bitsandbytes" => vec![python_command(
            "BITSANDBYTES",
            "bitsandbytes",
            &["bitsandbytes"],
            python_candidates,
            "import bitsandbytes as bnb; import pathlib; import sys; print('Version:', getattr(bnb, '__version__', 'unknown')); path=pathlib.Path(bnb.__file__).parent; libs=list(path.glob('libbitsandbytes_rocm*.so')); print('ROCm Libs:', libs); sys.exit(0 if libs else 1)",
        )],
        "rocm-smi" => vec![shell_command(
            "ROCm SMI",
            "rocm-smi",
            "rocm-smi",
            &["--showproductname"],
        )],
        "migraphx" => vec![python_command(
            "MIGraphX",
            "migraphx",
            &["migraphx"],
            python_candidates,
            "import migraphx; print(getattr(migraphx, '__version__', 'ok'))",
        )],
        "pytorch-profiler" => vec![python_command(
            "PyTorch Profiler",
            "pytorch-profiler",
            &["torch"],
            python_candidates,
            "from torch.profiler import profile; print(profile)",
        )],
        "wandb" => vec![python_command(
            "Weights & Biases",
            "wandb",
            &["wandb"],
            python_candidates,
            "import wandb; print(wandb.__version__)",
        )],
        "basic-env" => vec![shell_command(
            "Environment file",
            "basic-env",
            "bash",
            &[
                "-c",
                "test -f \"$HOME/.mlstack_env\" && echo OK || exit 1",
            ],
        )],
        "enhanced-env" => vec![shell_command(
            "Enhanced env",
            "enhanced-env",
            "bash",
            &[
                "-c",
                "grep -q \"Enhanced ML Stack Environment Setup Script\" \"$HOME/.mlstack_env\" && echo OK || exit 1",
            ],
        )],
        "permanent-env" => vec![shell_command(
            "Permanent env",
            "permanent-env",
            "bash",
            &[
                "-c",
                "grep -qiE \"Permanent ROCm Environment|Permanent ROCm Env|MLSTACK_PYTHON_BIN\" \"$HOME/.mlstack_env\" && echo OK || exit 1",
            ],
        )],
        // Benchmark components - verify by checking log files exist
        "mlperf-inference" => vec![benchmark_log_check_command(
            "MLPerf benchmark logs",
            "mlperf-inference",
            "mlperf_inference",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "rocm-benchmarks" => vec![benchmark_log_check_command(
            "ROCm benchmark logs",
            "rocm-benchmarks",
            "rocm_benchmarks",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "gpu-memory-bandwidth" => vec![benchmark_log_check_command(
            "Memory bandwidth logs",
            "gpu-memory-bandwidth",
            "gpu_memory_bandwidth",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "rocm-smi-bench" => vec![benchmark_log_check_command(
            "ROCm SMI logs",
            "rocm-smi-bench",
            "rocm_smi_benchmarks",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "pytorch-performance" => vec![benchmark_log_check_command(
            "PyTorch benchmark logs",
            "pytorch-performance",
            "pytorch_performance",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "vllm-performance" => vec![benchmark_log_check_command(
            "vLLM benchmark logs",
            "vllm-performance",
            "vllm_benchmarks",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "deepspeed-performance" => vec![benchmark_log_check_command(
            "DeepSpeed benchmark logs",
            "deepspeed-performance",
            "deepspeed_benchmarks",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "megatron-performance" => vec![benchmark_log_check_command(
            "Megatron benchmark logs",
            "megatron-performance",
            "megatron_benchmarks",
            "Benchmark logs found",
            "No benchmark logs yet",
        )],
        "all-benchmarks" => vec![benchmark_log_check_command(
            "Full suite logs",
            "all-benchmarks",
            "full_benchmarks",
            "Suite logs found",
            "No suite logs yet",
        )],
        "comfyui" => vec![shell_command(
            "ComfyUI",
            "comfyui",
            "bash",
            &["-c", "test -f \"$HOME/ComfyUI/main.py\" && echo 'ComfyUI installed' || exit 1"],
        )],
        _ => Vec::new(),
    }
}

pub fn modules_available(modules: &[String], python_candidates: &[String]) -> bool {
    if modules.is_empty() {
        return true;
    }
    let module_refs: Vec<&str> = modules.iter().map(|s| s.as_str()).collect();
    python_any(python_candidates, &module_refs)
}

fn basic_verification_commands(python_candidates: &[String]) -> Vec<VerificationCommand> {
    vec![
        shell_command("ROCm info", "rocm", "rocminfo", &[]),
        python_command(
            "PyTorch",
            "pytorch",
            &["torch"],
            python_candidates,
            "import torch, sys; print(torch.__version__); print('hip', getattr(torch.version, 'hip', None)); sys.exit(0 if getattr(torch.version, 'hip', None) else 1)",
        ),
        python_command(
            "Triton",
            "triton",
            &["triton"],
            python_candidates,
            "import triton; import triton._C; print(f'Triton {triton.__version__} ok')",
        ),
        python_command(
            "MPI4Py",
            "mpi4py",
            &["mpi4py"],
            python_candidates,
            "import importlib.metadata as m; import mpi4py; print(getattr(mpi4py, '__version__', m.version('mpi4py')))",
        ),
        python_command(
            "DeepSpeed",
            "deepspeed",
            &["deepspeed", "einops"],
            python_candidates,
            "import deepspeed; import einops; print(deepspeed.__version__)",
        ),
    ]
}

fn enhanced_verification_commands(python_candidates: &[String]) -> Vec<VerificationCommand> {
    let mut steps = basic_verification_commands(python_candidates);
    steps.extend(vec![
        python_command(
            "Flash Attention",
            "flash-attn",
            &["flash_attention_amd", "flash_attn", "flash_attn_2"],
            python_candidates,
            "import importlib, sys;\nmodules=['flash_attention_amd','flash_attn','flash_attn_2'];\nloaded=False;\nerrors=[];\nfor name in modules:\n    try:\n        importlib.import_module(name);\n        print(name);\n        loaded=True;\n        break\n    except Exception as exc:\n        errors.append(f'{name}: {exc}');\nif not loaded:\n    print('Flash Attention import errors:', '; '.join(errors));\n    raise SystemExit(1)",
        ),
        python_command(
            "vLLM",
            "vllm",
            &["vllm"],
            python_candidates,
            vllm_runtime_check_snippet(),
        ),
        python_command(
            "AITER",
            "aiter",
            &["aiter"],
            python_candidates,
            "import aiter; print(getattr(aiter, '__version__', 'ok'))",
        ),
        python_command(
            "ONNX Runtime",
            "onnx",
            &["onnxruntime"],
            python_candidates,
            "import onnxruntime as ort; import pathlib; import os; import sys; print('Version:', ort.__version__); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob('libonnxruntime_providers_rocm.so')); [os.environ.update({'ORT_ROCM_EP_PROVIDER_PATH': str(l)}) for l in libs[:1]]; providers=ort.get_available_providers(); print('Providers:', providers); sys.exit(0 if 'ROCMExecutionProvider' in providers else 1)",
        ),
        python_command(
            "BITSANDBYTES",
            "bitsandbytes",
            &["bitsandbytes"],
            python_candidates,
            "import bitsandbytes as bnb; import pathlib; import sys; print('Version:', getattr(bnb, '__version__', 'unknown')); path=pathlib.Path(bnb.__file__).parent; libs=list(path.glob('libbitsandbytes_rocm*.so')); print('ROCm Libs:', libs); sys.exit(0 if libs else 1)",
        ),
        shell_command("ROCm SMI", "rocm-smi", "rocm-smi", &["--showproductname"]),
        python_command(
            "MIGraphX",
            "migraphx",
            &["migraphx"],
            python_candidates,
            "import migraphx; print(getattr(migraphx, '__version__', 'ok'))",
        ),
        python_command(
            "PyTorch Profiler",
            "pytorch-profiler",
            &["torch"],
            python_candidates,
            "from torch.profiler import profile; print(profile)",
        ),
        python_command(
            "Weights & Biases",
            "wandb",
            &["wandb"],
            python_candidates,
            "import wandb; print(wandb.__version__)",
        ),
        shell_command("vLLM Studio", "vllm-studio", "bun", &["--version"]),
    ]);
    steps
}

fn build_verification_commands(python_candidates: &[String]) -> Vec<VerificationCommand> {
    let mut steps = basic_verification_commands(python_candidates);
    steps.extend(vec![
        python_command(
            "ONNX Runtime",
            "onnx",
            &["onnxruntime"],
            python_candidates,
            "import onnxruntime as ort; import pathlib; import os; import sys; print('Version:', ort.__version__); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob('libonnxruntime_providers_rocm.so')); [os.environ.update({'ORT_ROCM_EP_PROVIDER_PATH': str(l)}) for l in libs[:1]]; providers=ort.get_available_providers(); print('Providers:', providers); sys.exit(0 if 'ROCMExecutionProvider' in providers else 1)",
        ),
        python_command(
            "Flash Attention",
            "flash-attn",
            &["flash_attention_amd", "flash_attn", "flash_attn_2"],
            python_candidates,
            "import importlib, sys;\nmodules=['flash_attention_amd','flash_attn','flash_attn_2'];\nloaded=False;\nerrors=[];\nfor name in modules:\n    try:\n        importlib.import_module(name);\n        print(name);\n        loaded=True;\n        break\n    except Exception as exc:\n        errors.append(f'{name}: {exc}');\nif not loaded:\n    print('Flash Attention import errors:', '; '.join(errors));\n    raise SystemExit(1)",
        ),
        python_command(
            "MIGraphX",
            "migraphx",
            &["migraphx"],
            python_candidates,
            "import migraphx; print(getattr(migraphx, '__version__', 'ok'))",
        ),
    ]);
    steps
}

fn python_command(
    label: &str,
    target_id: &str,
    modules: &[&str],
    python_candidates: &[String],
    code: &str,
) -> VerificationCommand {
    let program = select_python_for_modules(modules, python_candidates)
        .or_else(|| python_candidates.first().cloned())
        .unwrap_or_else(|| "python3".to_string());

    VerificationCommand {
        label: label.to_string(),
        target_id: target_id.to_string(),
        program,
        args: vec!["-c".to_string(), code.to_string()],
        modules: modules.iter().map(|m| m.to_string()).collect(),
    }
}

fn vllm_runtime_check_snippet() -> &'static str {
    r#"import importlib
import vllm
import cachetools
import cbor2
import gguf
import pybase64
import ijson
import mistral_common
import openai_harmony

loaded = []
errs = []
for name in ("vllm._C", "vllm._rocm_C"):
    try:
        importlib.import_module(name)
        loaded.append(name)
    except Exception as exc:
        errs.append(f"{name}: {exc}")

if not loaded:
    raise SystemExit("vLLM native extension load failure: " + " | ".join(errs))

print(vllm.__version__)"#
}

fn shell_command(
    label: &str,
    target_id: &str,
    program: &str,
    args: &[&str],
) -> VerificationCommand {
    VerificationCommand {
        label: label.to_string(),
        target_id: target_id.to_string(),
        program: program.to_string(),
        args: args.iter().map(|s| s.to_string()).collect(),
        modules: Vec::new(),
    }
}

fn has_benchmark_log_dirs() -> bool {
    benchmark_logs::benchmark_log_directories()
        .iter()
        .any(|dir| {
            fs::read_dir(dir)
                .ok()
                .map(|entries| entries.filter_map(|entry| entry.ok()).any(|entry| entry.path().is_file()))
                .unwrap_or(false)
        })
}

fn benchmark_log_check_command(
    label: &str,
    target_id: &str,
    pattern: &str,
    success_message: &str,
    failure_message: &str,
) -> VerificationCommand {
    let script = benchmark_log_check_script(pattern, success_message, failure_message);
    shell_command(label, target_id, "bash", &["-c", &script])
}

fn benchmark_log_check_script(
    pattern: &str,
    success_message: &str,
    failure_message: &str,
) -> String {
    format!(
        "dirs=(\"${{MLSTACK_LOG_DIR:-}}\" \"$HOME/.rusty-stack/logs\" \"${{TMPDIR:-/tmp}}/rusty-stack/logs\")\n\
         found=0\n\
         for dir in \"${{dirs[@]}}\"; do\n\
             if [ -n \"$dir\" ] && [ -d \"$dir\" ]; then\n\
                 if ls \"$dir\" | grep -q \"{}\"; then\n\
                     echo \"{}\"\n\
                     found=1\n\
                     break\n\
                 fi\n\
             fi\n\
         done\n\
         if [ \"$found\" -ne 1 ]; then\n\
             echo \"{}\"\n\
         fi",
        pattern, success_message, failure_message
    )
}

fn python_any(python_candidates: &[String], modules: &[&str]) -> bool {
    python_candidates.iter().any(|python| {
        modules
            .iter()
            .any(|module| python_has_module(python, module))
    })
}

fn python_has_module(python: &str, module: &str) -> bool {
    python_exec(
        python,
        &format!(
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)"
        ),
    )
}

fn python_exec(python: &str, code: &str) -> bool {
    let mut cmd = Command::new(python);
    cmd.arg("-c")
        .arg(code)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if let Ok(existing) = env::var("PYTHONPATH") {
        let mut path = existing;
        let extra_paths = python_search_paths();
        if !extra_paths.is_empty() {
            if !path.is_empty() {
                path.push(':');
            }
            path.push_str(&extra_paths.join(":"));
        }
        cmd.env("PYTHONPATH", path);
    } else {
        let extra_paths = python_search_paths();
        if !extra_paths.is_empty() {
            cmd.env("PYTHONPATH", extra_paths.join(":"));
        }
    }

    let home = resolve_component_user_home();
    let ld_path = component_status_ld_library_path(&home);
    if !ld_path.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld_path);
    }

    cmd.status().map(|status| status.success()).unwrap_or(false)
}

fn push_unique_path(paths: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if paths.iter().any(|existing| existing == value) {
        return;
    }
    paths.push(value.to_string());
}

fn component_user_home_from_passwd(user_name: &str) -> Option<String> {
    let target = user_name.trim();
    if target.is_empty() {
        return None;
    }
    let passwd = fs::read_to_string("/etc/passwd").ok()?;
    passwd.lines().find_map(|line| {
        if line.trim().is_empty() || line.starts_with('#') {
            return None;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 || fields[0] != target {
            return None;
        }
        let home = fields[5].trim();
        if home.is_empty() {
            None
        } else {
            Some(home.to_string())
        }
    })
}

fn component_passwd_homes_with_mlstack() -> Vec<String> {
    let mut homes = Vec::new();
    let Ok(passwd) = fs::read_to_string("/etc/passwd") else {
        return homes;
    };

    for line in passwd.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 {
            continue;
        }
        let home = fields[5].trim();
        if home.is_empty() {
            continue;
        }
        if Path::new(home).join(".mlstack").is_dir() || Path::new(home).join(".mlstack_env").is_file()
        {
            push_unique_path(&mut homes, home);
        }
    }

    homes
}

fn component_candidate_homes(preferred_home: &str) -> Vec<String> {
    let mut homes = Vec::new();
    push_unique_path(&mut homes, preferred_home);

    if let Ok(value) = env::var("MLSTACK_USER_HOME") {
        push_unique_path(&mut homes, &value);
    }
    if let Ok(value) = env::var("HOME") {
        push_unique_path(&mut homes, &value);
    }

    for key in ["SUDO_USER", "USER", "LOGNAME"] {
        if let Ok(user_name) = env::var(key) {
            if let Some(home) = component_user_home_from_passwd(&user_name) {
                push_unique_path(&mut homes, &home);
            }
        }
    }

    for home in component_passwd_homes_with_mlstack() {
        push_unique_path(&mut homes, &home);
    }

    homes
}

fn resolve_component_user_home() -> String {
    let fallback = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let candidates = component_candidate_homes(&fallback);

    for home in &candidates {
        if Path::new(home).join(".mlstack_env").is_file() {
            return home.clone();
        }
    }
    for home in &candidates {
        if Path::new(home).join(".mlstack").is_dir() {
            return home.clone();
        }
    }

    candidates.into_iter().next().unwrap_or(fallback)
}

fn component_status_mpi_compat_dirs(home: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    for home in component_candidate_homes(home) {
        let mlstack_dir = Path::new(&home).join(".mlstack");
        let primary = mlstack_dir.join("libmpi-compat");
        if primary.is_dir() {
            dirs.push(primary.to_string_lossy().to_string());
        }

        if let Ok(entries) = fs::read_dir(&mlstack_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                    continue;
                };
                if name.starts_with("libmpi-compat-user-") {
                    dirs.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    dirs.sort();
    dirs.dedup();
    dirs
}

fn component_status_ld_library_path(home: &str) -> String {
    let mut paths = Vec::new();

    for dir in component_status_mpi_compat_dirs(home) {
        push_unique_path(&mut paths, &dir);
    }

    for rocm_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib", "/opt/rocm/opencl/lib"] {
        if Path::new(rocm_path).exists() {
            push_unique_path(&mut paths, rocm_path);
        }
    }

    if let Ok(existing) = env::var("LD_LIBRARY_PATH") {
        for part in existing.split(':') {
            push_unique_path(&mut paths, part);
        }
    }

    paths.join(":")
}

fn command_exists(command: &str) -> bool {
    Command::new("bash")
        .arg("-c")
        .arg(format!("command -v {}", command))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub fn python_search_paths() -> Vec<String> {
    let home = resolve_component_user_home();
    let mut paths = vec![
        format!("{}/pytorch", home),
        format!("{}/ml_stack/flash_attn_amd_direct", home),
        format!("{}/ml_stack/flash_attn_amd", home),
        format!(
            "{}/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-313",
            home
        ),
        format!(
            "{}/ml_stack/flash_attn_amd/build/lib.linux-x86_64-cpython-312",
            home
        ),
        format!("{}/.local/lib/python3.13/site-packages", home),
        format!("{}/.local/lib/python3.12/site-packages", home),
        format!("{}/rocm_venv/lib/python3.13/site-packages", home),
        format!("{}/rocm_venv/lib/python3.12/site-packages", home),
        format!("{}/megatron/Megatron-LM", home),
        format!("{}/ml_stack/bitsandbytes/bitsandbytes", home),
    ];

    // Dynamically add /opt/rocm/lib/python*/site-packages and /opt/rocm/lib
    let rocm_lib = Path::new("/opt/rocm/lib");
    if rocm_lib.exists() {
        paths.push(rocm_lib.to_string_lossy().to_string());
        if let Ok(entries) = fs::read_dir(rocm_lib) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("python") {
                            let sp = path.join("site-packages");
                            if sp.exists() {
                                paths.push(sp.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    paths
}

fn select_python_for_modules(modules: &[&str], python_candidates: &[String]) -> Option<String> {
    for key in ["MLSTACK_PYTHON_BIN", "UV_PYTHON"] {
        if let Ok(value) = env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                if modules.is_empty() {
                    return Some(value.to_string());
                }
                for module in modules {
                    if python_has_module(value, module) {
                        return Some(value.to_string());
                    }
                }
            }
        }
    }

    for python in python_candidates {
        for module in modules {
            if python_has_module(python, module) {
                return Some(python.clone());
            }
        }
    }
    None
}

fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

fn home_path(home: &str, parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(home);
    for part in parts {
        path.push(part);
    }
    path
}

fn env_file_has_enhanced(path: &Path) -> bool {
    if let Ok(contents) = fs::read_to_string(path) {
        return contents.contains("Enhanced ML Stack Environment Setup Script");
    }
    false
}

fn env_file_has_permanent(path: &Path) -> bool {
    if let Ok(contents) = fs::read_to_string(path) {
        return contents.contains("Permanent ROCm Environment")
            || contents.contains("Permanent ROCm Env")
            || (contents.contains("ML Stack Environment File")
                && contents.contains("MLSTACK_PYTHON_BIN"));
    }
    false
}

fn repo_has_ml_stack_core() -> bool {
    if let Ok(root) = env::var("MLSTACK_REPO_ROOT") {
        return path_exists(Path::new(&root).join("stans_ml_stack"));
    }
    false
}
