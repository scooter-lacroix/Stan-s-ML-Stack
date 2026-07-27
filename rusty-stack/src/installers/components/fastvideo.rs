//! FastVideo installer — native Rust build for ROCm gfx11 support.
//!
//! Clones scooter-lacroix/FastVideo and checks out an audited immutable commit,
//! then builds both Python distributions against Rusty's managed ROCm environment.
//!
//! Instead of delegating to `./build.sh --rocm` (which calls `uv pip install`
//! without `--system` and fails outside a venv), this module replicates the
//! build steps individually with proper env var injection and pip prefix
//! handling consistent with the rest of the rusty-stack installer ecosystem.
//!
//! # Validation Assertion
//!
//! - **VAL-INSTALL-050**: FastVideo builds from fork with ROCm gfx11 support

use std::path::{Path, PathBuf};

const FASTVIDEO_REPO: &str = "https://github.com/scooter-lacroix/FastVideo.git";
pub const FASTVIDEO_COMMIT: &str = "22e448771ebf5c81108f1c57b9c7ef4d5c26d182";
const FASTVIDEO_BUILD_DIR: &str = "/tmp/FastVideo_ROCm_build";

/// Leave one CPU free and never run more than four compile jobs.
pub fn max_jobs_for_parallelism(nproc: usize) -> usize {
    nproc.saturating_sub(1).clamp(1, 4)
}

fn is_valid_gpu_arch(arch: &str) -> bool {
    arch.strip_prefix("gfx")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_hexdigit()))
}

fn push_normalized_archs(archs: &mut Vec<String>, value: &str) {
    for arch in value
        .split(|ch: char| ch == ';' || ch == ',' || ch.is_ascii_whitespace())
        .map(str::trim)
        .filter(|arch| is_valid_gpu_arch(arch))
    {
        if !archs.iter().any(|existing| existing == arch) {
            archs.push(arch.to_string());
        }
    }
}

/// Apply the first configured visibility mask to enumerated discrete GPUs.
pub fn select_visible_gpu_archs(
    enumerated_archs: &[String],
    rocr_visible_devices: Option<&str>,
    hip_visible_devices: Option<&str>,
    cuda_visible_devices: Option<&str>,
) -> Vec<String> {
    let mask = [
        rocr_visible_devices,
        hip_visible_devices,
        cuda_visible_devices,
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .find(|value| !value.is_empty());

    let mut selected = Vec::new();
    if let Some(mask) = mask {
        for index in mask
            .split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
            .filter_map(|value| value.trim().parse::<usize>().ok())
        {
            if let Some(arch) = enumerated_archs.get(index) {
                push_normalized_archs(&mut selected, arch);
            }
        }
    } else {
        for arch in enumerated_archs {
            push_normalized_archs(&mut selected, arch);
        }
    }
    selected
}

fn discrete_pci_id_to_gfx(device_id: &str) -> Option<&'static str> {
    match device_id
        .trim()
        .trim_start_matches("0x")
        .to_ascii_lowercase()
        .as_str()
    {
        "7550" | "7551" => Some("gfx1201"),
        "7590" => Some("gfx1200"),
        "744c" | "7448" | "7449" | "744a" | "744b" | "745e" => Some("gfx1100"),
        "747e" | "7470" | "7460" | "7461" => Some("gfx1101"),
        "7480" | "7483" | "7489" | "749f" | "73f0" => Some("gfx1102"),
        "73bf" | "73af" | "73a5" | "73a1" | "73a2" | "73a3" => Some("gfx1030"),
        "73df" | "73c3" => Some("gfx1031"),
        "73ff" | "73ef" | "73e0" | "73e1" | "73e3" => Some("gfx1032"),
        "743f" | "7424" | "7421" | "7422" | "7423" => Some("gfx1034"),
        _ => None,
    }
}

/// Discover known discrete AMD GPUs directly from DRM sysfs in stable card order.
///
/// Matching known discrete PCI IDs excludes APUs without relying on the global
/// HSA override, which can otherwise collapse heterogeneous GPUs to one arch.
pub fn detect_discrete_gpu_archs_from_sysfs(drm_root: &Path) -> Vec<String> {
    let mut cards = match std::fs::read_dir(drm_root) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let index = name.strip_prefix("card")?.parse::<usize>().ok()?;
                Some((index, entry.path()))
            })
            .collect::<Vec<_>>(),
        Err(_) => return Vec::new(),
    };
    cards.sort_by_key(|(index, _)| *index);

    cards
        .into_iter()
        .filter_map(|(_, card)| {
            let device = card.join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).ok()?;
            if vendor.trim() != "0x1002" {
                return None;
            }
            let device_id = std::fs::read_to_string(device.join("device")).ok()?;
            discrete_pci_id_to_gfx(&device_id).map(str::to_string)
        })
        .collect()
}

/// Resolve a canonical, ordered, semicolon-delimited architecture list.
pub fn resolve_gpu_archs(
    sysfs_archs: &[String],
    gpu_archs_env: Option<&str>,
    gpu_arch_env: Option<&str>,
    detected_primary: Option<&str>,
) -> String {
    let mut archs = Vec::new();
    for value in sysfs_archs {
        push_normalized_archs(&mut archs, value);
    }
    if !archs.is_empty() {
        return archs.join(";");
    }

    for source in [gpu_archs_env, gpu_arch_env, detected_primary]
        .into_iter()
        .flatten()
    {
        archs.clear();
        push_normalized_archs(&mut archs, source);
        if !archs.is_empty() {
            return archs.join(";");
        }
    }

    "gfx1100".to_string()
}

/// Select the effective non-root install user.
pub fn resolve_install_user(sudo_user: Option<&str>, user: Option<&str>) -> Option<String> {
    [sudo_user, user]
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_string)
}

/// Explain the bounded cost before kernel compilation begins.
pub fn fastvideo_build_warning(gpu_archs: &str) -> String {
    format!("[FastVideo] Heavy dual-arch HIP compile for {gpu_archs}; MAX_JOBS capped at 4")
}

/// Functional FastVideo verification run through the selected Python interpreter.
///
/// Each physical ROCr visibility token gets its own child process. Removing the
/// inherited HIP/CUDA masks prevents them from filtering the child's renumbered
/// logical device 0.
pub fn fastvideo_verification_snippet() -> &'static str {
    r#"import os
import subprocess
import sys

probe = r"""
import fastvideo
import torch
from fastvideo_kernel import int8_quant

if not torch.cuda.is_available():
    raise SystemExit("FastVideo verification requires an available ROCm GPU")
device_count = torch.cuda.device_count()
if device_count != 1:
    raise SystemExit(f"FastVideo verification expected one isolated ROCm GPU, found {device_count}")

device = "cuda:0"
source = torch.randn(
    (2, 128),
    device=device,
    dtype=torch.float16,
).contiguous()
quantized, scale = int8_quant(source)
torch.cuda.synchronize()

shape_ok = quantized.shape == source.shape
dtype_ok = quantized.dtype == torch.int8
scale_finite = bool(torch.isfinite(scale).all().item())
if not (shape_ok and dtype_ok and scale_finite):
    raise SystemExit(
        "FastVideo int8_quant invariant failed on device 0: "
        f"shape_ok={shape_ok} dtype={quantized.dtype} scale_finite={scale_finite}"
    )

name = torch.cuda.get_device_name(0)
print(
    f"[FastVideo] device=0 name={name} int8_quant=ok "
    f"shape={tuple(quantized.shape)} dtype={quantized.dtype} "
    f"scale_finite={scale_finite}"
)
"""

visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
devices = [device.strip() for device in visible.split(",") if device.strip()]
if not devices:
    env = os.environ.copy()
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("GPU_DEVICE_ORDINAL", None)
    discovered = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    devices = [str(device) for device in range(int(discovered.stdout.strip()))]
if not devices:
    raise SystemExit("FastVideo verification requires at least one visible ROCm GPU")

for device in devices:
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = device
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("GPU_DEVICE_ORDINAL", None)
    subprocess.run([sys.executable, "-c", probe], check=True, env=env)
"#
}

/// Read-only preflights and dependency-free install operations after checkout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastVideoInstallStep {
    Chown,
    ManagedDependenciesPreflight,
    FlashAttentionPreflight,
    SourcePolicyPreflight,
    KernelInstall,
    PackageInstall,
}

impl FastVideoInstallStep {
    pub const fn requires_sudo(self) -> bool {
        matches!(self, Self::Chown)
    }
}

pub const fn post_checkout_plan() -> [FastVideoInstallStep; 6] {
    [
        FastVideoInstallStep::Chown,
        FastVideoInstallStep::ManagedDependenciesPreflight,
        FastVideoInstallStep::FlashAttentionPreflight,
        FastVideoInstallStep::SourcePolicyPreflight,
        FastVideoInstallStep::KernelInstall,
        FastVideoInstallStep::PackageInstall,
    ]
}

/// A constructed shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// The program to run.
    pub program: String,
    /// Arguments to pass.
    pub args: Vec<String>,
    /// Environment variables to set.
    pub env: Vec<(String, String)>,
    /// Working directory for the command.
    pub working_dir: Option<PathBuf>,
}

/// Configuration for the FastVideo installer.
#[derive(Debug, Clone)]
pub struct FastVideoConfig {
    /// Semicolon-delimited GPU architectures (e.g., "gfx1100;gfx1101").
    pub gpu_archs: String,
    /// Python binary to use.
    pub python_bin: String,
}

impl Default for FastVideoConfig {
    fn default() -> Self {
        Self {
            gpu_archs: "gfx1100".to_string(),
            python_bin: "python3".to_string(),
        }
    }
}

/// FastVideo installer.
#[derive(Debug, Clone)]
pub struct FastVideoInstaller {
    repo_url: String,
    commit: String,
    build_dir: String,
    config: FastVideoConfig,
}

impl Default for FastVideoInstaller {
    fn default() -> Self {
        Self::new(FastVideoConfig::default())
    }
}

impl FastVideoInstaller {
    pub fn new(config: FastVideoConfig) -> Self {
        Self {
            repo_url: FASTVIDEO_REPO.to_string(),
            commit: FASTVIDEO_COMMIT.to_string(),
            build_dir: FASTVIDEO_BUILD_DIR.to_string(),
            config,
        }
    }

    pub fn mkdir_build_dir(&self) -> ShellCommand {
        ShellCommand {
            program: "mkdir".to_string(),
            args: vec!["-p".to_string(), self.build_dir.clone()],
            env: vec![],
            working_dir: None,
        }
    }

    pub fn git_clone(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec!["clone".to_string(), self.repo_url.clone(), ".".to_string()],
            env: vec![],
            working_dir: Some(PathBuf::from(&self.build_dir)),
        }
    }

    /// Check out only the audited immutable fork commit in detached-HEAD mode.
    pub fn git_checkout(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "checkout".to_string(),
                "--detach".to_string(),
                self.commit.clone(),
            ],
            env: vec![],
            working_dir: Some(PathBuf::from(&self.build_dir)),
        }
    }

    /// Return ownership of the sudo-created clone to the install user.
    pub fn chown_build_dir(&self, run_user: &str) -> ShellCommand {
        ShellCommand {
            program: "chown".to_string(),
            args: vec![
                "-R".to_string(),
                format!("{run_user}:{run_user}"),
                self.build_dir.clone(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Verify that Rusty's centrally owned dependencies are present at exact pins.
    ///
    /// This preflight is deliberately read-only: FastVideo may consume the managed
    /// environment, but it must never install, upgrade, or resolve dependencies.
    pub fn managed_dependencies_preflight(&self) -> ShellCommand {
        let script = r#"import importlib.metadata
import os
from pathlib import Path
import sys

required = (
    "setuptools==81.0.0",
    "cmake==4.3.4",
    "scikit-build-core==1.0.3",
    "pathspec==1.1.1",
    "wheel==0.47.0",
    "ninja==1.13.0",
    "imageio==2.36.0",
    "diffusers==0.33.1",
    "remote-pdb==2.1.0",
    "ftfy==6.3.1",
    "wcwidth==0.2.13",
)
problems = []
for pin in required:
    name, expected = pin.rsplit("==", 1)
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        problems.append(f"{name}: missing (expected {expected})")
    else:
        if actual != expected:
            problems.append(f"{name}: found {actual}, expected {expected}")

try:
    import torch
except Exception as exc:
    problems.append(f"torch: import failed ({exc})")
else:
    if not getattr(torch.version, "hip", None):
        problems.append("torch: Rusty-managed build is not ROCm (torch.version.hip is empty)")
    torch_root = Path(torch.__file__).resolve().parent
    for relative in ("include/ATen/ATen.h", "include/torch/extension.h"):
        header = torch_root / relative
        if not header.is_file():
            problems.append(f"torch header missing: {header}")

rocm_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
hipcc = Path(os.environ.get("HIPCC", "/opt/rocm/bin/hipcc"))
if not hipcc.is_file() or not os.access(hipcc, os.X_OK):
    problems.append(f"HIP compiler missing or not executable: {hipcc}")
for relative in ("include/hip/hip_runtime.h", "include/hip/hip_fp16.h"):
    header = rocm_root / relative
    if not header.is_file():
        problems.append(f"ROCm header missing: {header}")

scripts_dir = Path(sys.executable).resolve().parent
for executable_name in ("cmake", "ninja"):
    executable = scripts_dir / executable_name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        problems.append(f"managed build executable missing: {executable}")

if problems:
    details = "\\n  - ".join(problems)
    raise SystemExit(
        "FastVideo requires dependencies owned by Rusty Stack; FastVideo will not "
        "modify the managed environment. Repair/reinstall Rusty's managed Python "
        f"dependencies, then retry:\\n  - {details}"
    )
print("[FastVideo] Rusty-managed dependency pins verified")
"#;
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), script.to_string()],
            env: vec![],
            working_dir: None,
        }
    }

    /// Fail closed if the immutable fork violates the audited ROCm-only source policy.
    pub fn source_policy_preflight(&self) -> ShellCommand {
        let script = r#"from pathlib import Path
import os
import re
import subprocess

root = Path(".")
violations = []

expected_commit = os.environ.get("FASTVIDEO_EXPECTED_COMMIT", "")
if not expected_commit:
    violations.append("FASTVIDEO_EXPECTED_COMMIT is missing")
else:
    # git rev-parse proves the checkout is the immutable Rusty pin.
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        violations.append(
            f"commit drift: expected {expected_commit}, found {actual_commit}"
        )

# git status must report no tracked, ignored, or untracked residue.
dirty = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=root,
    check=True,
    text=True,
    capture_output=True,
).stdout
if dirty:
    violations.append("working tree is not pristine:\\n" + dirty.rstrip())

if (root / ".gitmodules").exists():
    violations.append(".gitmodules is forbidden (CUTLASS/TK gitlinks are not allowed)")

staged_records = subprocess.run(
    ["git", "ls-files", "--stage", "-z"],
    cwd=root,
    check=True,
    text=True,
    capture_output=True,
).stdout.split("\0")
tracked_paths = []
gitlinks = []
for record in staged_records:
    if not record:
        continue
    metadata, relative = record.split("\t", 1)
    if metadata.startswith("160000 "):
        gitlinks.append(relative)
    else:
        tracked_paths.append(Path(relative))
if gitlinks:
    violations.append("gitlinks (mode 160000) are forbidden: " + ", ".join(gitlinks))

forbidden_path_terms = ("cutlass", "thunderkittens", "thunder-kittens")
for path in tracked_paths:
    relative = path.as_posix()
    if any(term in relative.lower() for term in forbidden_path_terms):
        violations.append(f"forbidden CUTLASS/TK/ThunderKittens path: {relative}")

expected_native = {
    "fastvideo-kernel/csrc/common_extension.cpp",
    "fastvideo-kernel/csrc/hip_native/gemm_rocm.cpp",
    "fastvideo-kernel/csrc/turbodiffusion/quant/quant.cu",
    "fastvideo-kernel/csrc/turbodiffusion/norm/rmsnorm.cu",
    "fastvideo-kernel/csrc/turbodiffusion/norm/layernorm.cu",
}
native_suffixes = {".cu", ".cpp", ".cc", ".cxx", ".hip"}
actual_native = {
    path.relative_to(root).as_posix()
    for path in (root / "fastvideo-kernel/csrc").rglob("*")
    if path.is_file() and path.suffix.lower() in native_suffixes
}
if actual_native != expected_native:
    violations.append(
        "unexpected native sources: expected "
        + repr(sorted(expected_native))
        + ", found "
        + repr(sorted(actual_native))
    )

dependency_suffixes = {".toml", ".cfg", ".ini", ".txt", ".cmake", ".lock"}
dependency_names = {
    "cmakelists.txt",
    "pip.conf",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "uv.lock",
    "uv.toml",
}
dependency_patterns = (
    re.compile(r"(?im)^\s*(?:dependencies|requires-dist)\s*=.*(?:nvidia|cuda)"),
    re.compile(r"(?im)^\s*(?:--)?(?:extra-index-url|index-url)\b.*(?:nvidia|cuda)"),
    re.compile(r"(?i)\bnvidia[-_.][a-z0-9_.-]+"),
    re.compile(r"(?i)\b(?:cupy-cuda|cuda-python|pytorch-cuda|torch[^\s\"']*\+cu\d|jax[^\s\"']*cuda)"),
)
workflow_root = Path(".github/workflows")
kernel_root = Path("fastvideo-kernel")
for relative_path in tracked_paths:
    path = root / relative_path
    if not path.is_file():
        continue
    relative = relative_path.as_posix()
    name_lower = path.name.lower()
    is_dependency_metadata = (
        path.suffix.lower() in dependency_suffixes
        or name_lower in dependency_names
        or name_lower.startswith("requirements")
    )
    is_kernel_workflow = (
        relative_path.is_relative_to(workflow_root)
        and path.suffix.lower() in {".yml", ".yaml"}
        and path.stem.lower().startswith("fastvideo-kernel")
    )
    is_kernel_template = (
        relative_path.is_relative_to(kernel_root) and "template" in relative.lower()
    )
    if not (is_dependency_metadata or is_kernel_workflow or is_kernel_template):
        continue
    try:
        source_text = path.read_text(errors="strict")
    except (OSError, UnicodeError):
        continue
    if is_dependency_metadata and any(
        pattern.search(source_text) for pattern in dependency_patterns
    ):
        violations.append(f"CUDA/NVIDIA dependency or index URL in {relative}")
    if (is_kernel_workflow or is_kernel_template) and re.search(
        r"(?i)\b(?:cuda|nvidia|cutlass|thunderkittens)\b", source_text
    ):
        violations.append(f"unexpected CUDA/NVIDIA native workflow/template: {relative}")

if violations:
    raise SystemExit("FastVideo ROCm source policy failed:\\n  - " + "\\n  - ".join(violations))
print("[FastVideo] immutable ROCm source policy verified")
"#;
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec![
                "-I".to_string(),
                "-S".to_string(),
                "-c".to_string(),
                script.to_string(),
            ],
            env: vec![("FASTVIDEO_EXPECTED_COMMIT".to_string(), self.commit.clone())],
            working_dir: Some(PathBuf::from(&self.build_dir)),
        }
    }

    /// Verify the selected managed flash-attention backend before compiling FastVideo.
    pub fn flash_attention_preflight(&self) -> ShellCommand {
        let script = r#"from pathlib import Path

marker = Path.home() / ".mlstack/flash-attention/.backend"
try:
    backend = marker.read_text().strip()
except OSError as exc:
    raise SystemExit(f"FastVideo requires managed flash-attention backend marker {marker}: {exc}")

if backend not in {"ck", "triton"}:
    raise SystemExit(
        f"FastVideo requires exact flash-attention backend marker 'ck' or 'triton'; found {backend!r}"
    )

import flash_attn
from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

selected = "CK" if backend == "ck" else "Triton fallback"
print(f"[FastVideo] selected flash_attn backend: {selected}")
"#;
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), script.to_string()],
            env: vec![],
            working_dir: None,
        }
    }

    /// Build and install fastvideo-kernel with ROCm support.
    ///
    /// Sets only the HIP architecture list and selected managed Python interpreter
    /// through CMake. The pinned fork is HIP-only and has no CUDA/TK backend switch.
    ///
    /// Uses `pip install --no-build-isolation --no-deps --no-index .` so the kernel
    /// consumes Rusty's managed build/runtime dependencies without resolving or
    /// modifying them.
    ///
    /// # Pip Target Directory
    ///
    /// The working directory is the clone's `fastvideo-kernel` subdirectory,
    /// whose `pyproject.toml` installs the native `fastvideo_kernel` module.
    pub fn pip_install_kernel(&self) -> ShellCommand {
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--break-system-packages".to_string(),
        ];
        args.extend([
            "-v".to_string(),
            "--no-build-isolation".to_string(),
            "--no-deps".to_string(),
            "--no-index".to_string(),
            ".".to_string(),
        ]);

        let cmake_args = format!(
            "-DCMAKE_HIP_ARCHITECTURES={} -DPython_EXECUTABLE={}",
            self.config.gpu_archs, self.config.python_bin
        );

        let max_jobs = std::thread::available_parallelism()
            .map(|n| max_jobs_for_parallelism(n.get()))
            .unwrap_or(1);

        let env = vec![
            ("CMAKE_ARGS".to_string(), cmake_args),
            ("GPU_ARCHS".to_string(), self.config.gpu_archs.clone()),
            ("MAX_JOBS".to_string(), max_jobs.to_string()),
            (
                "PYTHON_EXECUTABLE".to_string(),
                self.config.python_bin.clone(),
            ),
            ("PIP_NO_INDEX".to_string(), "1".to_string()),
            ("PIP_DISABLE_PIP_VERSION_CHECK".to_string(), "1".to_string()),
            ("UV_OFFLINE".to_string(), "1".to_string()),
        ];

        ShellCommand {
            program: self.config.python_bin.clone(),
            args,
            env,
            working_dir: Some(PathBuf::from(&self.build_dir).join("fastvideo-kernel")),
        }
    }

    /// Install the root FastVideo Python package without resolving CUDA dependencies.
    pub fn pip_install_package(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: [
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "--no-build-isolation",
                "--no-deps",
                "--no-index",
                ".",
            ]
            .map(String::from)
            .to_vec(),
            env: vec![
                ("PIP_NO_INDEX".to_string(), "1".to_string()),
                ("PIP_DISABLE_PIP_VERSION_CHECK".to_string(), "1".to_string()),
                ("UV_OFFLINE".to_string(), "1".to_string()),
            ],
            working_dir: Some(PathBuf::from(&self.build_dir)),
        }
    }

    /// Snapshot all distributions before installation or validate the after-state.
    ///
    /// Only FastVideo's two own distributions may change. CUDA/NVIDIA distributions
    /// fail closed in either phase.
    pub fn distribution_snapshot(&self, phase: &str, runtime_prefixes: &[&str]) -> ShellCommand {
        let script = r#"import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

phase = sys.argv[1]
state_path = Path("/tmp/rusty-fastvideo-distributions.json")
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
current = {
    normalize(dist.metadata["Name"]): dist.version
    for dist in importlib.metadata.distributions()
    if dist.metadata.get("Name")
}
runtime_prefixes = tuple(
    prefix
    for prefix in os.environ["RUSTY_NVIDIA_RUNTIME_PREFIXES"].split(",")
    if prefix
)
forbidden_accelerator = sorted(
    name
    for name in current
    if any(name.startswith(prefix) for prefix in runtime_prefixes) or "cuda" in name
)
if forbidden_accelerator:
    raise SystemExit(
        "CUDA/NVIDIA distributions are forbidden in Rusty's ROCm environment: "
        + ", ".join(forbidden_accelerator)
    )

if phase == "before":
    state_path.write_text(json.dumps(current, sort_keys=True))
elif phase == "after":
    before = json.loads(state_path.read_text())
    changed = {
        name
        for name in set(before) | set(current)
        if before.get(name) != current.get(name)
    }
    unexpected = sorted(changed - {"fastvideo", "fastvideo-kernel"})
    if unexpected:
        raise SystemExit(
            "FastVideo install modified Rusty-managed distributions: " + ", ".join(unexpected)
        )
else:
    raise SystemExit(f"unknown distribution snapshot phase: {phase!r}")
"#;
        ShellCommand {
            program: self.config.python_bin.clone(),
            args: vec!["-c".to_string(), script.to_string(), phase.to_string()],
            env: vec![(
                "RUSTY_NVIDIA_RUNTIME_PREFIXES".to_string(),
                runtime_prefixes.join(","),
            )],
            working_dir: None,
        }
    }

    pub fn cleanup(&self) -> ShellCommand {
        ShellCommand {
            program: "rm".to_string(),
            args: vec!["-rf".to_string(), "--".to_string(), self.build_dir.clone()],
            env: vec![],
            working_dir: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remediation_contract_is_immutable_read_only_and_offline() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100;gfx1101".to_string(),
            python_bin: "/managed/bin/python".to_string(),
        });

        assert_eq!(FASTVIDEO_COMMIT, "22e448771ebf5c81108f1c57b9c7ef4d5c26d182");
        assert_eq!(
            inst.git_checkout().args,
            ["checkout", "--detach", FASTVIDEO_COMMIT].map(String::from)
        );
        assert!(!inst
            .git_clone()
            .args
            .iter()
            .any(|arg| arg.contains("submodule")));

        let managed = inst.managed_dependencies_preflight();
        let managed_script = &managed.args[1];
        for pin in [
            "setuptools==81.0.0",
            "cmake==4.3.4",
            "scikit-build-core==1.0.3",
            "pathspec==1.1.1",
            "wheel==0.47.0",
            "ninja==1.13.0",
            "imageio==2.36.0",
            "diffusers==0.33.1",
            "remote-pdb==2.1.0",
            "ftfy==6.3.1",
            "wcwidth==0.2.13",
        ] {
            assert!(managed_script.contains(pin), "missing managed pin {pin}");
        }
        assert!(managed_script.contains("importlib.metadata"));
        for anchor in [
            "torch.version.hip",
            "/opt/rocm/bin/hipcc",
            "hip/hip_runtime.h",
            "ATen/ATen.h",
            "torch/extension.h",
        ] {
            assert!(
                managed_script.contains(anchor),
                "missing ROCm prerequisite anchor: {anchor}"
            );
        }
        assert!(!managed_script.contains("pip"));
        assert!(managed_script.contains("Rusty"));

        let flash = inst.flash_attention_preflight();
        let flash_script = &flash.args[1];
        assert!(flash_script.contains("from flash_attn import"));
        assert!(flash_script.contains("from flash_attn.flash_attn_interface import"));
        assert!(flash_script.contains("_flash_attn_varlen_forward"));
        assert!(flash_script.contains("_flash_attn_varlen_backward"));

        let source = inst.source_policy_preflight();
        assert_eq!(&source.args[..3], ["-I", "-S", "-c"].map(String::from));
        assert!(source.env.contains(&(
            "FASTVIDEO_EXPECTED_COMMIT".to_string(),
            FASTVIDEO_COMMIT.to_string()
        )));
        let source_script = &source.args[3].to_ascii_lowercase();
        for clean_tree_anchor in [
            "git status",
            "--porcelain=v1",
            "--untracked-files=all",
            "git rev-parse",
            "fastvideo_expected_commit",
        ] {
            assert!(
                source_script.contains(clean_tree_anchor),
                "missing clean-tree proof: {clean_tree_anchor}"
            );
        }
        for forbidden in [
            ".gitmodules",
            "160000",
            "cutlass",
            "thunderkittens",
            "nvidia",
            "cuda",
            "extra-index-url",
            "index-url",
            ".github/workflows",
            "common_extension.cpp",
            "gemm_rocm.cpp",
            "quant.cu",
            "rmsnorm.cu",
            "layernorm.cu",
        ] {
            assert!(
                source_script.contains(forbidden),
                "missing source policy {forbidden}"
            );
        }

        assert!(source_script.contains("is_kernel_workflow"));
        assert!(source_script.contains("path.stem.lower().startswith(\"fastvideo-kernel\")"));
        let dependency_suffixes = source_script
            .lines()
            .find(|line| line.starts_with("dependency_suffixes ="))
            .expect("source policy must define dependency metadata suffixes");
        assert!(!dependency_suffixes.contains(".yml"));
        assert!(!dependency_suffixes.contains(".yaml"));

        for cmd in [inst.pip_install_kernel(), inst.pip_install_package()] {
            assert_eq!(cmd.program, "/managed/bin/python");
            for required in ["--no-deps", "--no-build-isolation", "--no-index"] {
                assert!(
                    cmd.args.contains(&required.to_string()),
                    "missing {required}"
                );
            }
            for forbidden in ["--upgrade", "--extra-index-url", "--index-url"] {
                assert!(!cmd.args.contains(&forbidden.to_string()));
            }
            assert!(cmd
                .env
                .contains(&("PIP_NO_INDEX".to_string(), "1".to_string())));
            assert!(cmd
                .env
                .contains(&("UV_OFFLINE".to_string(), "1".to_string())));
        }

        let runtime_prefixes = [
            "nvidia",
            "cuda",
            "cudnn",
            "cublas",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "cusparselt",
            "nccl",
            "nvtx",
            "nvjitlink",
            "nv-pic",
            "tensorrt",
            "pytorch-cuda",
            "torch-cuda",
        ];
        let snapshot = inst.distribution_snapshot("before", &runtime_prefixes);
        assert_eq!(snapshot.program, "/managed/bin/python");
        assert!(snapshot.args[1].contains("fastvideo-kernel"));
        assert!(snapshot.args[1].contains("RUSTY_NVIDIA_RUNTIME_PREFIXES"));
        let encoded_prefixes = snapshot
            .env
            .iter()
            .find(|(name, _)| name == "RUSTY_NVIDIA_RUNTIME_PREFIXES")
            .map(|(_, value)| value.as_str())
            .expect("snapshot must receive canonical runtime prefixes");
        for forbidden in runtime_prefixes {
            assert!(
                encoded_prefixes
                    .split(',')
                    .any(|prefix| prefix == forbidden),
                "missing canonical runtime distribution pattern {forbidden}"
            );
        }
    }

    /// Source-policy regression guard: the pinned FastVideo fork carries
    /// unrelated upstream CUDA workflows, and the source policy must accept the
    /// pinned fork while still rejecting stale/untracked source.
    ///
    /// Historically this test was gated on a hardcoded personal path and silently
    /// `return`ed when absent — so it never ran in CI or for other contributors.
    /// It now reads the FastVideo fork location from `FASTVIDEO_POLICY_TEST_REPO`
    /// and is **skipped** (via an explicit eprintln + return, since cargo has no
    /// first-class runtime skip without `#[ignore]`) when that var is unset. Set
    /// the var in CI (or locally) to exercise the policy for real.
    #[test]
    fn source_policy_accepts_pinned_fork_with_unrelated_cuda_workflows() {
        use std::process::Command;
        use std::time::{SystemTime, UNIX_EPOCH};

        let repo_str = match std::env::var("FASTVIDEO_POLICY_TEST_REPO") {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                eprintln!(
                    "skip: FASTVIDEO_POLICY_TEST_REPO unset; set it to a FastVideo fork checkout \
                     to exercise source_policy_accepts_pinned_fork_with_unrelated_cuda_workflows"
                );
                return;
            }
        };
        let repo = Path::new(&repo_str);
        if !repo.is_dir() {
            eprintln!(
                "skip: FASTVIDEO_POLICY_TEST_REPO={} is not a directory",
                repo.display()
            );
            return;
        }

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let clone = std::env::temp_dir().join(format!(
            "rusty-fastvideo-policy-{}-{unique}",
            std::process::id()
        ));
        let checkout_status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["worktree", "add", "--detach"])
            .arg(&clone)
            .arg(FASTVIDEO_COMMIT)
            .status()
            .expect("git must create an immutable FastVideo worktree");
        assert!(
            checkout_status.success(),
            "immutable FastVideo worktree must succeed"
        );

        let cmd = FastVideoInstaller::default().source_policy_preflight();
        let run_policy = || {
            Command::new(&cmd.program)
                .args(&cmd.args)
                .envs(cmd.env.iter().cloned())
                .current_dir(&clone)
                .output()
                .expect("configured Python should execute the source policy")
        };
        let clean_output = run_policy();
        assert!(
            clean_output.status.success(),
            "pinned fork must pass source policy despite unrelated upstream workflows: {}",
            String::from_utf8_lossy(&clean_output.stderr)
        );

        std::fs::write(
            clone.join("sitecustomize.py"),
            "raise RuntimeError('must never execute')\n",
        )
        .expect("write stale untracked source");
        let dirty_output = run_policy();
        assert!(
            !dirty_output.status.success(),
            "source policy must reject stale untracked files"
        );
        assert!(
            String::from_utf8_lossy(&dirty_output.stderr).contains("working tree is not pristine"),
            "dirty-tree failure must identify the source residue"
        );

        let remove_status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["worktree", "remove", "--force"])
            .arg(&clone)
            .status()
            .expect("git must remove temporary FastVideo worktree");
        assert!(
            remove_status.success(),
            "temporary worktree removal must succeed"
        );
    }

    #[test]
    fn remediation_plan_has_no_patch_or_dependency_write_phase() {
        use FastVideoInstallStep::*;
        assert_eq!(
            post_checkout_plan(),
            [
                Chown,
                ManagedDependenciesPreflight,
                FlashAttentionPreflight,
                SourcePolicyPreflight,
                KernelInstall,
                PackageInstall,
            ]
        );
    }

    #[test]
    fn test_verification_snippet_isolates_each_visible_gpu_with_rocr_only() {
        use std::process::Command;

        let snippet = fastvideo_verification_snippet();
        for anchor in [
            "import os",
            "import subprocess",
            "import fastvideo",
            "import torch",
            "from fastvideo_kernel import int8_quant",
            "torch.cuda.is_available()",
            "torch.cuda.device_count()",
            "visible = os.environ.get(\"ROCR_VISIBLE_DEVICES\", \"\")",
            "devices = [device.strip() for device in visible.split(\",\") if device.strip()]",
            "for device in devices:",
            "env = os.environ.copy()",
            "env[\"ROCR_VISIBLE_DEVICES\"] = device",
            "env.pop(\"HIP_VISIBLE_DEVICES\", None)",
            "env.pop(\"CUDA_VISIBLE_DEVICES\", None)",
            "env.pop(\"GPU_DEVICE_ORDINAL\", None)",
            "subprocess.run(",
            "device = \"cuda:0\"",
            "(2, 128)",
            "dtype=torch.float16",
            ".contiguous()",
            "int8_quant(source)",
            "torch.cuda.synchronize()",
            "quantized.shape == source.shape",
            "quantized.dtype == torch.int8",
            "torch.isfinite(scale).all()",
            "torch.cuda.get_device_name(0)",
            "raise SystemExit",
        ] {
            assert!(
                snippet.contains(anchor),
                "missing verification anchor: {anchor}"
            );
        }

        for forbidden in ["rocminfo", "/sys/class/drm"] {
            assert!(
                !snippet.contains(forbidden),
                "snippet must use Python/ROCr visibility, found {forbidden}"
            );
        }

        let compile = format!("compile({snippet:?}, '<fastvideo-verification>', 'exec')");
        let output = Command::new(&FastVideoConfig::default().python_bin)
            .args(["-c", &compile])
            .output()
            .expect("configured python3 should run");
        assert!(
            output.status.success(),
            "verification snippet must compile as Python: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn test_git_clone_uses_fork() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let clone_cmd = inst.git_clone();
        assert!(clone_cmd.args.contains(&"clone".into()));
        assert!(
            clone_cmd.args.contains(&FASTVIDEO_REPO.into()),
            "Should clone from scooter-lacroix fork"
        );
    }

    #[test]
    fn test_chown_build_dir_targets_install_user() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cmd = inst.chown_build_dir("ml-user");
        assert_eq!(cmd.program, "chown");
        assert_eq!(
            cmd.args,
            ["-R", "ml-user:ml-user", FASTVIDEO_BUILD_DIR].map(String::from)
        );
    }

    #[test]
    fn test_pip_install_has_multi_arch_rocm_cmake_args() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100;gfx1101".to_string(),
            python_bin: "python3".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        assert!(cmd.args.contains(&"--no-build-isolation".into()));
        assert!(cmd.args.contains(&"--no-deps".into()));
        let cmake_env = cmd.env.iter().find(|(k, _)| k == "CMAKE_ARGS").unwrap();
        assert!(cmake_env
            .1
            .contains("CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101"));
        assert!(!cmake_env.1.contains("GPU_BACKEND"));
        assert!(!cmd.env.iter().any(|(key, _)| key == "GPU_BACKEND"));
        assert!(!cmake_env.1.to_ascii_lowercase().contains("tk"));
    }

    #[test]
    fn test_pip_install_sets_same_gpu_archs() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100;gfx1101".to_string(),
            python_bin: "python3".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        let gpu_env = cmd.env.iter().find(|(k, _)| k == "GPU_ARCHS").unwrap();
        assert_eq!(gpu_env.1, "gfx1100;gfx1101");
    }

    #[test]
    fn test_both_pip_commands_always_allow_selected_interpreter() {
        let inst = FastVideoInstaller::default();
        assert!(inst
            .pip_install_package()
            .args
            .contains(&"--break-system-packages".to_string()));
        assert!(inst
            .pip_install_kernel()
            .args
            .contains(&"--break-system-packages".to_string()));
    }

    #[test]
    fn test_max_jobs_is_cpu_spare_and_capped() {
        assert_eq!(max_jobs_for_parallelism(0), 1);
        assert_eq!(max_jobs_for_parallelism(1), 1);
        assert_eq!(max_jobs_for_parallelism(2), 1);
        assert_eq!(max_jobs_for_parallelism(4), 3);
        assert_eq!(max_jobs_for_parallelism(8), 4);
    }

    #[test]
    fn test_resolve_gpu_archs_prefers_sysfs() {
        let sysfs = vec![" gfx1100 ".to_string(), "gfx1101".to_string()];
        assert_eq!(
            resolve_gpu_archs(&sysfs, Some("gfx1200"), Some("gfx1030"), Some("gfx900")),
            "gfx1100;gfx1101"
        );
    }

    #[test]
    fn test_resolve_gpu_archs_normalizes_and_deduplicates_env() {
        assert_eq!(
            resolve_gpu_archs(&[], Some(" gfx1100, gfx1101;gfx1100\tgfx1101 "), None, None),
            "gfx1100;gfx1101"
        );
        assert_eq!(
            resolve_gpu_archs(&[], None, Some("gfx1100, gfx1101"), None),
            "gfx1100;gfx1101"
        );
    }

    #[test]
    fn test_resolve_gpu_archs_detected_primary_then_default() {
        assert_eq!(
            resolve_gpu_archs(&[], Some(" , ; "), None, Some(" gfx1101 ")),
            "gfx1101"
        );
        assert_eq!(resolve_gpu_archs(&[], None, None, None), "gfx1100");
    }

    #[test]
    fn test_resolve_install_user_precedence_and_blanks() {
        assert_eq!(
            resolve_install_user(Some(" sudo-user "), Some("user")),
            Some("sudo-user".to_string())
        );
        assert_eq!(
            resolve_install_user(Some("   "), Some(" user ")),
            Some("user".to_string())
        );
        assert_eq!(resolve_install_user(None, None), None);
        assert_eq!(resolve_install_user(Some(""), Some(" \t ")), None);
    }

    #[test]
    fn test_heavy_build_warning_is_complete() {
        assert_eq!(
            fastvideo_build_warning("gfx1100;gfx1101"),
            "[FastVideo] Heavy dual-arch HIP compile for gfx1100;gfx1101; MAX_JOBS capped at 4"
        );
    }

    #[test]
    fn test_post_checkout_plan_order_and_privileges() {
        use FastVideoInstallStep::*;

        let plan = post_checkout_plan();
        assert_eq!(
            plan,
            [
                Chown,
                ManagedDependenciesPreflight,
                FlashAttentionPreflight,
                SourcePolicyPreflight,
                KernelInstall,
                PackageInstall,
            ]
        );
        assert!(Chown.requires_sudo());
        for step in [
            ManagedDependenciesPreflight,
            FlashAttentionPreflight,
            SourcePolicyPreflight,
            KernelInstall,
            PackageInstall,
        ] {
            assert!(!step.requires_sudo());
        }
    }

    #[test]
    fn test_flash_attention_preflight_uses_marker_and_required_api() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100".to_string(),
            python_bin: "python3.12".to_string(),
        });
        let cmd = inst.flash_attention_preflight();

        assert_eq!(cmd.program, "python3.12");
        assert_eq!(cmd.args[0], "-c");
        assert_eq!(cmd.working_dir, None);
        let script = &cmd.args[1];
        for anchor in [
            "Path.home()",
            ".mlstack/flash-attention/.backend",
            "ck",
            "triton",
            "from flash_attn import",
            "flash_attn_func",
            "flash_attn_varlen_func",
            "flash_attn_varlen_qkvpacked_func",
            "_flash_attn_varlen_forward",
            "_flash_attn_varlen_backward",
        ] {
            assert!(
                script.contains(anchor),
                "missing preflight anchor: {anchor}"
            );
        }
        assert!(!script.contains("fastvideo_kernel"));
        assert!(!script.contains("common_extension"));
    }

    #[test]
    fn test_root_package_install_is_dependency_free_from_clone_root() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100".to_string(),
            python_bin: "python3.12".to_string(),
        });
        let cmd = inst.pip_install_package();

        assert_eq!(cmd.program, "python3.12");
        assert_eq!(
            cmd.args,
            [
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "--no-build-isolation",
                "--no-deps",
                "--no-index",
                ".",
            ]
            .map(String::from)
        );
        assert_eq!(cmd.working_dir, Some(PathBuf::from(FASTVIDEO_BUILD_DIR)));
        assert!(cmd
            .env
            .contains(&("PIP_NO_INDEX".to_string(), "1".to_string())));
        assert!(cmd
            .env
            .contains(&("UV_OFFLINE".to_string(), "1".to_string())));
        assert!(!cmd.args.iter().any(|arg| {
            let arg = arg.to_ascii_lowercase();
            arg.contains("cuda") || arg.contains("nvidia") || arg == "torch"
        }));
    }

    #[test]
    fn test_visibility_masks_select_discrete_archs() {
        let archs = vec!["gfx1100".to_string(), "gfx1101".to_string()];
        assert_eq!(
            select_visible_gpu_archs(&archs, Some("0,1"), None, None),
            archs
        );
        assert_eq!(
            select_visible_gpu_archs(&archs, Some("1"), Some("0"), Some("0")),
            vec!["gfx1101".to_string()]
        );
        assert!(select_visible_gpu_archs(&archs, Some("bad,9"), None, None).is_empty());
    }

    #[test]
    fn test_gpu_arch_validation_rejects_garbage() {
        assert_eq!(
            resolve_gpu_archs(
                &["not-gfx".to_string(), "gfx1101".to_string()],
                Some("garbage,gfx12zz,gfx1100"),
                None,
                None
            ),
            "gfx1101"
        );
        assert_eq!(
            resolve_gpu_archs(&[], Some("garbage,gfx12zz,gfx1100"), None, None),
            "gfx1100"
        );
    }

    #[test]
    fn test_cleanup_removes_build_dir() {
        let inst = FastVideoInstaller::new(FastVideoConfig::default());
        let cleanup = inst.cleanup();
        assert_eq!(
            cleanup.args,
            ["-rf", "--", FASTVIDEO_BUILD_DIR].map(String::from)
        );
    }

    #[test]
    fn test_sysfs_detection_excludes_apu_and_preserves_discrete_order() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rusty-fastvideo-sysfs-{}-{unique}",
            std::process::id()
        ));

        for (card, device_id) in [
            ("card0", "0x15bf"),
            ("card1", "0x744c"),
            ("card2", "0x747e"),
        ] {
            let device = root.join(card).join("device");
            std::fs::create_dir_all(&device).expect("create fake DRM device");
            std::fs::write(device.join("vendor"), "0x1002\n").expect("write fake vendor");
            std::fs::write(device.join("device"), format!("{device_id}\n"))
                .expect("write fake device id");
        }

        assert_eq!(
            detect_discrete_gpu_archs_from_sysfs(&root),
            vec!["gfx1100".to_string(), "gfx1101".to_string()]
        );
        std::fs::remove_dir_all(&root).expect("remove fake DRM tree");
    }

    #[test]
    fn test_config_propagates_python_bin() {
        let inst = FastVideoInstaller::new(FastVideoConfig {
            gpu_archs: "gfx1100".to_string(),
            python_bin: "python3.12".to_string(),
        });
        let cmd = inst.pip_install_kernel();
        assert_eq!(cmd.program, "python3.12");
    }
}
