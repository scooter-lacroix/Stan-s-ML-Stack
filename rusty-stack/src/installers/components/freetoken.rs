//! FreeToken installer — MoE-offload LLM serving engine (FlashML-org) with the
//! ROCm/HIP feature branch from the scooter-lacroix fork.
//!
//! FreeToken is installed into a DEDICATED venv (`~/.mlstack/venvs/freetoken`),
//! unlike most components that share `~/.mlstack/global`: its pins
//! (transformers>=5.5,<6, numpy<2.5, huggingface_hub>=1.5) are ahead of the
//! co-installed vllm/megatron/textgen set, and a serving engine should not
//! perturb the training stack. The venv interpreter is created from the SAME
//! managed base python, and torch comes from the SAME ROCm wheel index the
//! core pytorch installer uses — only the site-packages differ.
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-050**: FreeToken installs into a dedicated venv under `~/.mlstack/venvs`
//! - **VAL-INSTALL-051**: FreeToken clones the ROCm fork pinned to `feature/rocm`
//! - **VAL-INSTALL-052**: FreeToken installs itself `--no-deps` with a curated dependency list (no CUDA wheels)
//! - **VAL-INSTALL-053**: FreeToken declares dependency on PyTorch and ROCm
//! - **VAL-INSTALL-054**: install-time HIP smoke test proves import + backend resolution + pinned memory + LRU kernel

use crate::installers::common::RocmEnv;
use std::path::PathBuf;

// ===========================================================================
// Types
// ===========================================================================

/// The FreeToken fork the installer builds from.
pub const DEFAULT_REPO_URL: &str = "https://github.com/scooter-lacroix/FreeToken.git";
/// The ROCm/HIP integration branch on the fork.
pub const DEFAULT_BRANCH: &str = "feature/rocm";

/// Configuration for the FreeToken installer.
#[derive(Debug, Clone)]
pub struct FreetokenConfig {
    /// Managed base python the venv is created from (`~/.mlstack/global/bin/python`).
    pub python_bin: String,
    /// venv location; defaults to `~/.mlstack/venvs/freetoken`.
    pub venv_dir: Option<PathBuf>,
    /// Source clone location; defaults to `~/.mlstack/freetoken`.
    pub clone_dir: Option<PathBuf>,
    /// Fork repo URL.
    pub repo_url: String,
    /// Fork branch to track.
    pub branch: String,
    /// Whether to force reinstall (purge venv + clone first).
    pub force_reinstall: bool,
    /// Whether to run in dry-run mode.
    pub dry_run: bool,
    /// ROCm series for the torch wheel index (e.g. "7.2"); resolved from the
    /// installed ROCm when None.
    pub rocm_series: Option<String>,
}

impl Default for FreetokenConfig {
    fn default() -> Self {
        Self {
            python_bin: "python3".to_string(),
            venv_dir: None,
            clone_dir: None,
            repo_url: DEFAULT_REPO_URL.to_string(),
            branch: DEFAULT_BRANCH.to_string(),
            force_reinstall: false,
            dry_run: false,
            rocm_series: None,
        }
    }
}

impl FreetokenConfig {
    /// venv directory (`~/.mlstack/venvs/freetoken`).
    pub fn venv_dir(&self) -> PathBuf {
        self.venv_dir
            .clone()
            .unwrap_or_else(|| dirs_home().join(".mlstack").join("venvs").join("freetoken"))
    }

    /// venv python binary.
    pub fn venv_python(&self) -> PathBuf {
        self.venv_dir().join("bin").join("python")
    }

    /// Source clone directory (`~/.mlstack/freetoken`).
    pub fn clone_dir(&self) -> PathBuf {
        self.clone_dir
            .clone()
            .unwrap_or_else(|| dirs_home().join(".mlstack").join("freetoken"))
    }

    /// ROCm series, resolved from the installed ROCm when unset.
    pub fn resolved_rocm_series(&self) -> String {
        if let Some(s) = &self.rocm_series {
            return s.clone();
        }
        let env = RocmEnv::detect();
        if env.is_detected() && !env.version().is_empty() {
            let series = env.version_major_minor();
            if !series.is_empty() {
                return series;
            }
        }
        "7.2".to_string()
    }

    /// Torch wheel index for the resolved ROCm series.
    pub fn torch_index_url(&self) -> String {
        format!(
            "https://download.pytorch.org/whl/rocm{}",
            self.resolved_rocm_series()
        )
    }
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
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

/// The FreeToken runtime dependencies, installed explicitly.
///
/// FreeToken's own `pyproject.toml` pins these ranges; we curate the same set
/// (minus torch/triton, which the venv gets from the ROCm index, and minus the
/// NVIDIA-only `[fi]`/`[sgl]` extras that have in-repo Triton fallbacks) so the
/// `--no-deps` self-install can never drag a CUDA wheel in transitively.
pub const FREETOKEN_DEPS: &[&str] = &[
    "apache-tvm-ffi==0.1.13.post3",
    "transformers>=5.5,<6",
    "einops>=0.8,<1",
    "fastapi>=0.115,<1",
    "gguf>=0.19,<1",
    "huggingface_hub>=1.5,<2",
    "msgpack>=1.1,<2",
    "modelscope>=1.37,<2",
    "numpy>=2.0,<2.5",
    "openai>=2.0,<3",
    "partial-json-parser>=0.2,<1",
    "prompt_toolkit>=3.0,<4",
    "pydantic>=2.9,<3",
    "pyzmq>=27,<28",
    "safetensors>=0.6,<1",
    "tqdm>=4.66,<5",
    "uvicorn>=0.30,<1",
    // build tools for the two CppExtensions
    "ninja",
    "packaging",
    "setuptools>=77",
    "wheel",
];

/// The FreeToken installer.
pub struct FreetokenInstaller {
    pub config: FreetokenConfig,
}

impl FreetokenInstaller {
    /// Create a new FreeToken installer with the given config.
    pub fn new(config: FreetokenConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(FreetokenConfig::default())
    }

    /// Dependency closure: FreeToken needs the ROCm platform and the managed
    /// python/pytorch base (the venv is created FROM the managed python and
    /// torch from the ROCm index, both owned by the core installers).
    pub fn dependencies(&self) -> &'static [&'static str] {
        &["pytorch", "rocm"]
    }

    /// Build environment for the extension compile: GPU arch identity and the
    /// HIP toolchain prefix. `__HIP_PLATFORM_AMD__` and `USE_ROCM` come from
    /// torch's cpp_extension on a HIP build; these are the stack-side inputs.
    pub fn build_env(&self, gpu_arch: &str, hsa_version: &str) -> Vec<(String, String)> {
        vec![
            ("GPU_ARCH".to_string(), gpu_arch.to_string()),
            ("PYTORCH_ROCM_ARCH".to_string(), gpu_arch.to_string()),
            ("ROCM_PATH".to_string(), "/opt/rocm".to_string()),
            ("ROCM_HOME".to_string(), "/opt/rocm".to_string()),
            ("HSA_OVERRIDE_GFX_VERSION".to_string(), hsa_version.to_string()),
            // The stack env prepends component shims (onnxruntime builds, the
            // RCCL overlay) to PYTHONPATH that must not leak into the venv's
            // import resolution.
            ("PYTHONPATH".to_string(), String::new()),
        ]
    }

    // -----------------------------------------------------------------------
    // Command builders
    // -----------------------------------------------------------------------

    /// Create (or re-use) the dedicated venv. `uv venv` is used because the
    /// managed python is a uv-managed build whose stdlib `venv` emits a broken
    /// `<exec_prefix>` layout; uv handles its own pythons correctly. Falls
    /// back to `python -m venv` when uv is not on PATH.
    pub fn build_venv_create_command(&self, uv_available: bool) -> ShellCommand {
        if uv_available {
            ShellCommand {
                program: "uv".to_string(),
                args: vec![
                    "venv".to_string(),
                    // --seed: uv venvs ship WITHOUT pip; the subsequent
                    // `python -m pip` steps need it seeded.
                    "--seed".to_string(),
                    "--python".to_string(),
                    self.config.python_bin.clone(),
                    self.config.venv_dir().to_string_lossy().to_string(),
                ],
                env: vec![],
                working_dir: None,
            }
        } else {
            ShellCommand {
                program: self.config.python_bin.clone(),
                args: vec![
                    "-m".to_string(),
                    "venv".to_string(),
                    self.config.venv_dir().to_string_lossy().to_string(),
                ],
                env: vec![],
                working_dir: None,
            }
        }
    }

    /// Install ROCm torch into the venv FIRST so every later dependency
    /// resolves against it (VAL-INSTALL-052: no CUDA wheel can satisfy torch).
    pub fn build_torch_install_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.venv_python().to_string_lossy().to_string(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "torch".to_string(),
                "--index-url".to_string(),
                self.config.torch_index_url(),
            ],
            env: vec![("PYTHONPATH".to_string(), String::new())],
            working_dir: None,
        }
    }

    /// Install the curated runtime dependency set (single resolution pass).
    pub fn build_deps_install_command(&self) -> ShellCommand {
        let mut args = vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
        ];
        args.extend(FREETOKEN_DEPS.iter().map(|s| s.to_string()));
        ShellCommand {
            program: self.config.venv_python().to_string_lossy().to_string(),
            args,
            env: vec![("PYTHONPATH".to_string(), String::new())],
            working_dir: None,
        }
    }

    /// Install flashlib with --no-deps (No-CUDA hard tenet). Its metadata
    /// requires nvidia-cutlass-dsl -> cuda-python/cuda-bindings for the CuTe
    /// primitives, none of which FreeToken touches: the slot_cache path
    /// (lru_ensure) is pure Triton, and flashlib loads primitives lazily, so
    /// the CUDA toolchain never loads. torch/triton (its other requirements)
    /// are already provided by the torch step.
    pub fn build_flashlib_install_command(&self) -> ShellCommand {
        ShellCommand {
            program: self.config.venv_python().to_string_lossy().to_string(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--no-deps".to_string(),
                "flashlib==0.3.0".to_string(),
            ],
            env: vec![("PYTHONPATH".to_string(), String::new())],
            working_dir: None,
        }
    }

    /// Clone the ROCm fork (idempotent; run through `git_clone_or_pull`).
    pub fn build_git_clone_command(&self) -> ShellCommand {
        ShellCommand {
            program: "git".to_string(),
            args: vec![
                "clone".to_string(),
                "--branch".to_string(),
                self.config.branch.clone(),
                "--single-branch".to_string(),
                self.config.repo_url.clone(),
                self.config.clone_dir().to_string_lossy().to_string(),
            ],
            env: vec![],
            working_dir: None,
        }
    }

    /// Install FreeToken itself: `--no-deps` scopes pip to the freetoken
    /// package only (its two CppExtensions compile through the HIP shim), and
    /// the curated deps + venv torch make the closure CUDA-free by
    /// construction.
    pub fn build_pip_install_command(&self, gpu_arch: &str, hsa_version: &str) -> ShellCommand {
        ShellCommand {
            program: self.config.venv_python().to_string_lossy().to_string(),
            args: vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "-v".to_string(),
                "--no-build-isolation".to_string(),
                "--no-deps".to_string(),
                ".".to_string(),
            ],
            env: self.build_env(gpu_arch, hsa_version),
            working_dir: Some(self.config.clone_dir()),
        }
    }

    /// Install-time HIP smoke test (VAL-INSTALL-054). Proves on the metal,
    /// inside the venv, that: the torch build is HIP (not CUDA), the package
    /// imports, the engine's auto backend resolution lands on the pure-Triton
    /// path, mapped pinned memory works (the MoE zero-copy substrate), and
    /// flashlib's Triton LRU admission kernel runs. Two-layer hang protection
    /// mirrors the MIGraphX smoke test: an internal SIGALRM plus GNU timeout.
    pub fn build_smoke_test_command(&self) -> ShellCommand {
        let script = r#"
import signal, sys

def _handler(signum, frame):
    raise TimeoutError("FreeToken HIP smoke test stalled")

signal.signal(signal.SIGALRM, _handler)
signal.setitimer(signal.ITIMER_REAL, 420.0)

import torch
assert torch.version.hip, f"expected a HIP torch build, got cuda={torch.version.cuda}"
assert not torch.version.cuda, "CUDA runtime leaked into the freetoken venv"

import freetoken
from freetoken.attention.base import AttnType
import freetoken.engine.engine as ee
backend = ee._resolve_auto_attention_backend(frozenset([AttnType.FULL]), False)
assert backend == "triton", f"auto backend resolved to {backend!r}, expected 'triton'"

from freetoken.kernel import _pinned_tensor
t = _pinned_tensor.alloc_pinned_tensor([4, 8], torch.bfloat16)
assert _pinned_tensor.host_ptr_identity(), "zero-copy pinned identity mapping failed"

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.offload_kernels import ensure_experts
cache = OffloadMoeCache(num_layers=2, num_experts=8, cache_size=12,
                        device=torch.device("cuda"), quant_format="bf16")
ids = torch.tensor([1, 3, 5], dtype=torch.int32, device="cuda")
ensure_experts(cache, 0, ids)
torch.cuda.synchronize()
assert int(cache.num_indices.item()) == 3, "slot cache admitted wrong count"

print("FREETOKEN_HIP_SMOKE_OK")
"#;
        ShellCommand {
            program: "timeout".to_string(),
            args: vec![
                "480".to_string(),
                self.config.venv_python().to_string_lossy().to_string(),
                "-c".to_string(),
                script.trim().to_string(),
            ],
            env: vec![
                ("PYTHONPATH".to_string(), String::new()),
                ("ROCR_VISIBLE_DEVICES".to_string(), "0".to_string()),
            ],
            working_dir: None,
        }
    }

    /// Contents of the `~/.mlstack/bin/ft` launcher shim. Sanitizes PYTHONPATH
    /// by stripping only the onnxruntime-build entry (it shadows the venv's
    /// site-packages), while KEEPING the stack's RCCL overlay shim
    /// (`~/.mlstack/components/rccl/active/python` + its env vars): the stock
    /// system librccl is broken for multi-GPU collectives on this platform,
    /// and `--tensor-parallel-size 2` needs the repaired RCCL preloaded. The
    /// overlay sitecustomize re-execs through ld.so with inhibit-rpath, which
    /// is safe for the venv (verified: TP gloo+RCCL all_reduce on 2 GPUs).
    pub fn launcher_script(&self) -> String {
        format!(
            "#!/bin/sh\n# rusty-stack freetoken launcher: exec the dedicated venv entry point.\n# Strip ONLY the onnxruntime-build PYTHONPATH entry (it shadows venv imports);\n# keep the RCCL overlay shim + vars -- the stock system librccl is broken for\n# multi-GPU collectives, and TP>1 needs the stack's repaired RCCL preloaded.\nMLSTACK_PY=\"$MLSTACK_PYTHON_BIN\"; [ -n \"$MLSTACK_PY\" ] || MLSTACK_PY=\"$HOME/.mlstack/global/bin/python\"\nORPATH=\"$HOME/onnxruntime_build/onnxruntime/build/Linux/Release\"\nNEWPP=\"\"\nOLDIFS=\"$IFS\"; IFS=:\nfor d in \"$PYTHONPATH\"; do\n  [ -z \"$d\" ] && continue\n  [ \"$d\" = \"$ORPATH\" ] && continue\n  if [ -z \"$NEWPP\" ]; then NEWPP=\"$d\"; else NEWPP=\"$NEWPP:$d\"; fi\ndone\nIFS=\"$OLDIFS\"\nPYTHONPATH=\"$NEWPP\" exec {} \"$@\"\n",
            self.config
                .venv_dir()
                .join("bin")
                .join("ft")
                .to_string_lossy()
        )
    }

    /// Scan build output for known failure patterns.
    pub fn check_build_output(&self, output: &str) -> Result<(), String> {
        let patterns = [
            "error: command",
            "FAILED:",
            "undefined reference",
            "ImportError:",
            "ModuleNotFoundError",
            "fatal error:",
            "RuntimeError: Error building extension",
        ];
        for p in patterns {
            if output.contains(p) {
                return Err(format!("FreeToken build failed (pattern {p:?} found)"));
            }
        }
        Ok(())
    }
}

// ===========================================================================
// Tests (VAL-INSTALL-050..054)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FreetokenConfig {
        FreetokenConfig {
            python_bin: "/home/test/.mlstack/global/bin/python".to_string(),
            ..Default::default()
        }
    }

    /// VAL-INSTALL-050: dedicated venv under ~/.mlstack/venvs, separate from
    /// the shared global env.
    #[test]
    fn test_venv_layout() {
        let inst = FreetokenInstaller::new(config());
        let venv = inst.config.venv_dir().to_string_lossy().to_string();
        // HOME-dependent on purpose (stack layout convention); assert structure,
        // not an absolute path, so the test is host-independent.
        assert!(venv.ends_with("/.mlstack/venvs/freetoken"), "{venv}");
        assert!(!venv.contains("/global"), "must NOT be the shared global env");
        assert!(inst
            .config
            .venv_python()
            .to_string_lossy()
            .ends_with("venvs/freetoken/bin/python"));
        assert!(inst
            .config
            .clone_dir()
            .to_string_lossy()
            .ends_with("/.mlstack/freetoken"));
    }

    /// VAL-INSTALL-051: clone from the ROCm fork, pinned to feature/rocm.
    #[test]
    fn test_git_clone_command() {
        let inst = FreetokenInstaller::new(config());
        let cmd = inst.build_git_clone_command();
        assert_eq!(cmd.program, "git");
        assert!(cmd.args.contains(&"--branch".to_string()));
        let branch_idx = cmd.args.iter().position(|a| a == "--branch").unwrap();
        assert_eq!(cmd.args[branch_idx + 1], "feature/rocm");
        assert!(cmd
            .args
            .contains(&"https://github.com/scooter-lacroix/FreeToken.git".to_string()));
    }

    /// VAL-INSTALL-052: self-install is --no-deps; torch comes from the ROCm
    /// index; the curated dep list contains no CUDA/NVIDIA artifacts.
    #[test]
    fn test_dep_closure_is_cuda_free() {
        let inst = FreetokenInstaller::new(config());

        let torch_cmd = inst.build_torch_install_command();
        let torch_idx = torch_cmd
            .args
            .iter()
            .position(|a| a == "--index-url")
            .unwrap();
        assert!(torch_cmd.args[torch_idx + 1].contains("download.pytorch.org/whl/rocm"));
        assert!(!torch_cmd.args[torch_idx + 1].contains("cu1"));

        let pip_cmd = inst.build_pip_install_command("gfx1100", "11.0.0");
        assert!(pip_cmd.args.contains(&"--no-deps".to_string()));
        assert!(pip_cmd.working_dir.is_some());

        for dep in FREETOKEN_DEPS {
            let d = dep.to_lowercase();
            assert!(!d.starts_with("nvidia-"), "CUDA dep in curated list: {dep}");
            assert!(!d.contains("+cu"), "CUDA wheel marker in curated list: {dep}");
            assert!(!d.contains("sglang"), "sgl extra leaked into curated list: {dep}");
            assert!(!d.contains("flashinfer"), "fi extra leaked into curated list: {dep}");
        }
    }

    /// VAL-INSTALL-053: dependency closure declares pytorch + rocm.
    #[test]
    fn test_dependencies() {
        let inst = FreetokenInstaller::new(config());
        assert_eq!(inst.dependencies(), &["pytorch", "rocm"]);
    }

    /// VAL-INSTALL-054: smoke test asserts HIP torch, triton backend,
    /// pinned identity and the LRU kernel; wrapped in timeout.
    #[test]
    fn test_smoke_test_command() {
        let inst = FreetokenInstaller::new(config());
        let cmd = inst.build_smoke_test_command();
        assert_eq!(cmd.program, "timeout");
        assert_eq!(cmd.args[0], "480");
        let script = &cmd.args[cmd.args.len() - 1];
        assert!(script.contains("torch.version.hip"));
        assert!(script.contains("\"triton\""));
        assert!(script.contains("host_ptr_identity"));
        assert!(script.contains("ensure_experts"));
        // PYTHONPATH sanitized in the smoke env
        assert!(cmd
            .env
            .iter()
            .any(|(k, v)| k == "PYTHONPATH" && v.is_empty()));
    }

    /// Launcher shim: sanitized PYTHONPATH, venv ft entry point.
    #[test]
    fn test_launcher_script() {
        let inst = FreetokenInstaller::new(config());
        let script = inst.launcher_script();
        assert!(script.contains("onnxruntime_build"));
        assert!(!script.contains("unset MLSTACK_RCCL"));
        assert!(script.contains(".mlstack/venvs/freetoken/bin/ft"));
    }

    /// flashlib installs --no-deps (its CUDA-target metadata would violate
    /// the No-CUDA tenet; the slot_cache path is pure Triton + lazy).
    #[test]
    fn test_flashlib_no_deps() {
        let inst = FreetokenInstaller::new(config());
        let cmd = inst.build_flashlib_install_command();
        assert!(cmd.args.contains(&"--no-deps".to_string()));
        assert!(cmd.args.contains(&"flashlib==0.3.0".to_string()));
    }

    /// Build output scanning rejects known failure patterns.
    #[test]
    fn test_check_build_output() {
        let inst = FreetokenInstaller::new(config());
        assert!(inst.check_build_output("ok build").is_ok());
        assert!(inst
            .check_build_output("RuntimeError: Error building extension")
            .is_err());
    }

    /// venv creation prefers uv (managed-python correctness) and falls back.
    #[test]
    fn test_venv_create_variants() {
        let inst = FreetokenInstaller::new(config());
        let uv_cmd = inst.build_venv_create_command(true);
        assert_eq!(uv_cmd.program, "uv");
        assert!(uv_cmd.args.contains(&"--seed".to_string()), "pip must be seeded");
        assert!(uv_cmd.args.contains(&"--python".to_string()));
        let plain_cmd = inst.build_venv_create_command(false);
        assert!(plain_cmd.program.contains("python"));
        assert!(plain_cmd.args.contains(&"-m".to_string()));
    }
}
