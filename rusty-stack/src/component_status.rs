use crate::state::{Category, Component};
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
    let home = env::var("HOME").unwrap_or_default();
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
    candidates.sort();
    candidates.dedup();
    candidates
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
    let home = env::var("HOME").unwrap_or_default();
    match component_id {
        "rocm" => path_exists("/opt/rocm/.info/version") || command_exists("rocminfo"),
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
            python_any(python_candidates, &["vllm"])
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
        "rocm-smi" => path_exists("/opt/rocm/bin/rocm-smi"),
        "migraphx" => {
            python_any(python_candidates, &["migraphx"])
                || path_exists(home_path(&home, &["migraphx_build"]))
        }
        "pytorch-profiler" => python_any(python_candidates, &["torch"]),
        "wandb" => python_any(python_candidates, &["wandb"]),
        "permanent-env" => env_file_has_permanent(&home_path(&home, &[".mlstack_env"])),
        "basic-env" => path_exists(home_path(&home, &[".mlstack_env"])),
        "enhanced-env" => env_file_has_enhanced(&home_path(&home, &[".mlstack_env"])),
        "vllm-performance" => path_exists(home_path(&home, &[".rusty-stack", "logs"])),
        "all-benchmarks" => path_exists(home_path(&home, &[".rusty-stack", "logs"])),
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
            "import torch; print(torch.__version__); print(torch.version.hip)",
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
            "from mpi4py import MPI; print(MPI.Get_version())",
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
            "import vllm; print(vllm.__version__)",
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
            "import bitsandbytes as bnb\nimport pathlib\nimport sys\nprint(f'Version: {getattr(bnb, \"__version__\", \"unknown\")}');\npath=pathlib.Path(bnb.__file__).parent;\nlibs=list(path.glob('libbitsandbytes_rocm*.so'));\nprint(f'ROCm Libs: {libs}');\nsys.exit(0 if libs else 1)",
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
        "mlperf-inference" => vec![shell_command(
            "MLPerf benchmark logs",
            "mlperf-inference",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"mlperf_inference\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "rocm-benchmarks" => vec![shell_command(
            "ROCm benchmark logs",
            "rocm-benchmarks",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"rocm_benchmarks\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "gpu-memory-bandwidth" => vec![shell_command(
            "Memory bandwidth logs",
            "gpu-memory-bandwidth",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"gpu_memory_bandwidth\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "rocm-smi-bench" => vec![shell_command(
            "ROCm SMI logs",
            "rocm-smi-bench",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"rocm_smi_benchmarks\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "pytorch-performance" => vec![shell_command(
            "PyTorch benchmark logs",
            "pytorch-performance",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"pytorch_performance\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "vllm-performance" => vec![shell_command(
            "vLLM benchmark logs",
            "vllm-performance",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"vllm_benchmarks\" && echo \"Benchmark logs found\" || echo \"No benchmark logs yet\"",
            ],
        )],
        "all-benchmarks" => vec![shell_command(
            "Full suite logs",
            "all-benchmarks",
            "bash",
            &[
                "-c",
                "test -d \"$HOME/.rusty-stack/logs\" && ls \"$HOME/.rusty-stack/logs\" | grep -q \"full_benchmarks\" && echo \"Suite logs found\" || echo \"No suite logs yet\"",
            ],
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
            "import torch; print(torch.__version__); print('hip', torch.version.hip); print('cuda', torch.cuda.is_available()); x=torch.rand(1, device='cuda' if torch.cuda.is_available() else 'cpu'); print(x)",
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
            "from mpi4py import MPI; print(MPI.Get_version())",
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
            "import vllm; print(vllm.__version__)",
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
            "import bitsandbytes as bnb\nimport pathlib\nimport sys\nprint(f'Version: {getattr(bnb, \"__version__\", \"unknown\")}');\npath=pathlib.Path(bnb.__file__).parent;\nlibs=list(path.glob('libbitsandbytes_rocm*.so'));\nprint(f'ROCm Libs: {libs}');\nsys.exit(0 if libs else 1)",
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

    cmd.status().map(|status| status.success()).unwrap_or(false)
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
    let home = env::var("HOME").unwrap_or_default();
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
