use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Welcome,
    HardwareDetect,
    Preflight,
    ComponentDetect,
    ComponentSelect,
    Configuration,
    Confirm,
    Installing,
    Complete,
    Benchmarks,
    Recovery,
}

/// What kind of run the selected components constitute. Drives flow verbiage
/// so each screen shows only text/options relevant to the selected action:
/// benchmarks and verification are "run", not "installed".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Install,
    Benchmark,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Foundation,
    Core,
    UiUx,
    Extension,
    Environment,
    Maintenance,
    Performance,
}

#[derive(Debug, Clone)]
pub struct Component {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Script path relative to scripts_dir, or empty string for native Rust modules.
    ///
    /// For native Rust components, this field is empty and the installer
    /// dispatches to the corresponding Rust module via
    /// `installers::components::is_native_component()`.
    ///
    /// # Validation Assertions
    ///
    /// - **VAL-INSTALL-037**: Component.script no longer holds .sh filenames for ported components
    /// - **VAL-INSTALL-038**: state.rs supports native module routing
    /// - **VAL-INSTALL-040**: Verification/performance components use native routing
    pub script: String,
    pub category: Category,
    pub required: bool,
    pub selected: bool,
    pub installed: bool,
    pub progress: f32,
    pub estimate: String,
    /// Whether this component requires sudo to install
    pub needs_sudo: bool,
    /// Experimental / forward-only build (e.g. Flash Attention CK on RDNA3).
    /// Rendered with an EXPERIMENTAL badge + `note`; still selectable +
    /// installable — never greyed out or hidden.
    pub experimental: bool,
    /// Optional detail shown in the component panel (capability caveats,
    /// what works vs. what falls back). Shown only when `experimental` is true.
    pub note: Option<String>,
}

impl Component {
    /// Returns `true` if this component uses a native Rust installer
    /// (no shell script involved).
    ///
    /// # Validation Assertions
    ///
    /// - **VAL-INSTALL-038**: state.rs supports native module routing
    pub fn is_native(&self) -> bool {
        crate::installers::components::is_native_component(&self.id)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GPUInfo {
    pub model: String,
    pub driver: String,
    pub architecture: String,
    pub rocm_version: String,
    pub gpu_count: usize,
    pub memory_gb: f32,
    pub temperature_c: Option<f32>,
    pub power_watts: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub distribution: String,
    pub kernel: String,
    pub cpu_model: String,
    pub memory_gb: f32,
    pub storage_gb: f32,
    pub storage_available_gb: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightStatus {
    Passed,
    Warning,
    Failed,
}

impl PreflightStatus {
    pub fn label(self) -> &'static str {
        match self {
            PreflightStatus::Passed => "passed",
            PreflightStatus::Warning => "warning",
            PreflightStatus::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightType {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone)]
pub struct PreflightCheck {
    pub name: String,
    pub status: PreflightStatus,
    pub check_type: PreflightType,
    pub message: String,
    pub details: String,
    pub score: i32,
}

#[derive(Debug, Clone, Default)]
pub struct PreflightResult {
    pub passed: bool,
    pub summary: String,
    pub checks: Vec<PreflightCheck>,
    pub passed_count: usize,
    pub failed_count: usize,
    pub warning_count: usize,
    pub total_score: i32,
    pub can_continue: bool,
}

#[derive(Debug, Clone, Default)]
pub struct HardwareState {
    pub gpu: GPUInfo,
    pub system: SystemInfo,
    pub status: String,
    pub progress: f32,
}

#[derive(Debug, Clone, Default)]
pub struct InstallStatus {
    pub progress: f32,
    pub message: String,
    pub completed: bool,
}

pub fn default_components() -> Vec<Component> {
    use Category::*;

    vec![
        // ── Environment ──
        Component {
            id: "permanent-env".into(),
            name: "Permanent ROCm Env".into(),
            description: "Unified permanent ROCm environment (Python 3.12, \
                          device-filtered — iGPUs excluded)"
                .into(),
            script: String::new(), // Native Rust installer
            category: Environment,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "1-2 min".into(),
            needs_sudo: true,
            experimental: false,
            note: Some(
                "Auto-sourced into bash/zsh/fish on every shell launch when \
                 installed via the Global option (iGPU filtered, ROCm env always \
                 active). Isolated/named envs are written but NOT auto-sourced \
                 into the global shell — source them manually for that env."
                    .into(),
            ),
        },
        // ── Foundation ──
        Component {
            id: "rocm".into(),
            name: "ROCm Platform".into(),
            description: "AMD ROCm GPU computing platform".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "30-45 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "pytorch".into(),
            name: "PyTorch with ROCm".into(),
            description: "PyTorch optimized for AMD GPUs".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
            // pip-install of the ROCm wheel into the managed venv — no system
            // writes, no sudo. (Was `true`, which made run_installation's sudo
            // gate spawn `sudo -n true` and leak "a password is required" when
            // the sudo timestamp lapsed, even though pip never needs root.)
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "triton".into(),
            name: "Triton".into(),
            description: "Compiler for parallel programming".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
            // pip-install of the triton wheel — no system writes, no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "mpi4py".into(),
            name: "MPI4Py".into(),
            description: "MPI bindings for Python".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
            // pip-install against an EXISTING system openmpi (the installer only
            // reads its path, never apt-installs it). No sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "deepspeed".into(),
            name: "DeepSpeed".into(),
            description: "Deep learning optimization library".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "8-12 min".into(),
            // pip-install of the deepspeed wheel — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "rocm-smi".into(),
            name: "ROCm SMI".into(),
            description: "System monitoring for AMD GPUs".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "2-3 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "pytorch-profiler".into(),
            name: "PyTorch Profiler".into(),
            description: "Performance analysis for PyTorch".into(),
            script: String::new(), // Native Rust installer
            category: Foundation,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
            // pip-install of profiler wheels — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        // ── Core ──
        // Flash Attention ships two backends from ROCm/flash-attention. They
        // install the SAME `flash_attn` package → mutually exclusive; only the
        // last-installed is active (tracked by ~/.mlstack/flash-attention/.backend).
        Component {
            id: "flash-attn-triton".into(),
            name: "Flash Attention (Triton)".into(),
            description: "Flash Attention 2 via the ROCm Triton backend".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "20-30 min".into(),
            // pip build-from-source into the venv — no sudo.
            needs_sudo: false,
            experimental: false,
            note: Some(
                "Full forward + backward pass — the only Flash Attention backend \
                 with a backward pass on RDNA3, so it is the choice for \
                 training/backprop (CDNA + RDNA, fp16/bf16/fp32; causal, MQA/GQA, \
                 rotary, ALiBi, paged, FP8). Mutually exclusive with Flash Attention \
                 (CK): both install the identical flash_attn package, so installing \
                 this replaces CK (the active backend is tracked by the \
                 ~/.mlstack/flash-attention/.backend marker). Source: ROCm/flash-attention \
                 Triton backend."
                    .to_string(),
            ),
        },
        Component {
            id: "flash-attn-ck".into(),
            name: "Flash Attention (CK)".into(),
            description: "Flash Attention 2 via the ROCm composable-kernel backend".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "20-30 min".into(),
            // pip build-from-source into the venv — no sudo.
            needs_sudo: false,
            experimental: true,
            note: Some(
                "EXPERIMENTAL — RDNA3 forward-pass only. The composable-kernel \
                 backward pass is not implemented on RDNA3 \
                 (ROCm/composable_kernel#1434). Inference, generation, and forward \
                 attention are accelerated; training/backprop through this backend \
                 falls back to unfused PyTorch SDPA (higher VRAM, slower). RDNA4 \
                 gains backward with deterministic=False. Mutually exclusive with \
                 the Triton backend — installing this replaces it. \
                 Source: ROCm/flash-attention CK backend."
                    .to_string(),
            ),
        },
        Component {
            id: "migraphx".into(),
            name: "MIGraphX".into(),
            description: "AMD graph inference engine".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
            // prebuilt/wheel install — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            // Standalone AMDMIGraphX Python bindings. The C++ core + onnxruntime
            // MIGraphX EP cover ONNX inference without these bindings, so this is
            // opt-in (experimental). Must be present in default_components() so
            // DirectInstallerExecutor::component_for_id can resolve it — the
            // installer dispatch (installer.rs "migraphx-python" arm) and the
            // registry/manifest both advertise it; without this entry an explicit
            // `rusty-stack update migraphx-python` fails with "Unknown component
            // ID" before reaching the native installer.
            id: "migraphx-python".into(),
            name: "MIGraphX Python".into(),
            description: "AMDMIGraphX Python bindings (source build)".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-25 min".into(),
            // cmake build from source into user space — no sudo.
            needs_sudo: false,
            experimental: true,
            note: Some(
                "Standalone AMDMIGraphX Python bindings built from source \
                 (MLSTACK_MIGRAPHX_BUILD_PYTHON=1). The C++ core + onnxruntime \
                 MIGraphX EP already cover ONNX inference, so these bindings are \
                 only needed for direct Python migraphx API access. Source: \
                 ROCm/AMDMIGraphX."
                    .to_string(),
            ),
        },
        Component {
            id: "llama-cpp".into(),
            name: "Rusty Llama (llama.cpp)".into(),
            description: "Rusty Llama — llama.cpp fork with HIP/ROCm GPU acceleration".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
            // cmake build + git clone into user space — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "megatron".into(),
            name: "Megatron-LM".into(),
            description: "Large-scale training framework".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "25-40 min".into(),
            // git clone + `pip install -e .` into the managed venv — no sudo.
            // Same mis-flag as pytorch; flipping stops the spurious sudo prompt.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "aiter".into(),
            name: "AITER".into(),
            description: "AMD AITER optimization tooling".into(),
            script: String::new(), // Native Rust installer
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            // pip-install of the aiter wheel — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        // ── Extension ──
        Component {
            id: "vllm".into(),
            name: "vLLM".into(),
            description: "High-throughput inference engine".into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-25 min".into(),
            // pip-install of the vllm wheel — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "onnx".into(),
            name: "ONNX Runtime".into(),
            description: "Cross-platform inference accelerator".into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "2-5 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "bitsandbytes".into(),
            name: "BITSANDBYTES".into(),
            description: "Efficient quantization for deep learning".into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
            // pip-install of the bitsandbytes wheel — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "wandb".into(),
            name: "Weights & Biases".into(),
            description: "Experiment tracking and visualization".into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-8 min".into(),
            // pip-install of the wandb wheel — no sudo.
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "fastvideo".into(),
            name: "FastVideo".into(),
            description: "Video generation framework with ROCm gfx11 optimizations".into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        // ── UI/UX ──
        Component {
            id: "vllm-studio".into(),
            name: "vLLM Studio".into(),
            description: "Model lifecycle manager for vLLM/SGLang".into(),
            script: String::new(), // Native Rust installer
            category: UiUx,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "comfyui".into(),
            name: "ComfyUI".into(),
            description: "Node-based AI image generation UI with ROCm support".into(),
            script: String::new(), // Native Rust installer
            category: UiUx,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "textgen".into(),
            name: "text-generation-webui".into(),
            description: "LLM chat/inference web UI with ROCm support (oobabooga)".into(),
            script: String::new(), // Native Rust installer
            category: UiUx,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "freetoken".into(),
            name: "FreeToken".into(),
            description: "MoE-offload LLM serving engine (OpenAI/Anthropic APIs), HIP port"
                .into(),
            script: String::new(), // Native Rust installer
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            // Dedicated venv (~/.mlstack/venvs/freetoken) + torch from the ROCm
            // index; launcher shim at ~/.mlstack/bin/ft. No sudo.
            needs_sudo: false,
            experimental: false,
            note: Some(
                "Serves MoE checkpoints (Qwen3.5-MoE / DeepSeek-V4 / GLM / gpt-oss class) with \
                 experts offloaded to host RAM. Installs into its own venv, so the training \
                 stack is untouched. Attention runs the pure-Triton backend on AMD; fp8/MXFP4/\
                 NVFP4 checkpoints are NOT supported on RDNA consumer GPUs — use bf16 or \
                 Q4_K/Q6_K GGUF. Launcher: ~/.mlstack/bin/ft."
                    .into(),
            ),
        },
        // ── Maintenance (verify + repair) ──
        Component {
            id: "verify-basic".into(),
            name: "Verify Installation".into(),
            description: "Basic verification".into(),
            script: String::new(), // Native Rust verification
            category: Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "2-5 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "verify-enhanced".into(),
            name: "Enhanced Verify Installation".into(),
            description: "Advanced verification".into(),
            script: String::new(), // Native Rust verification
            category: Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "verify-build".into(),
            name: "Verify and Build".into(),
            description: "Verify + build components".into(),
            script: String::new(), // Native Rust verification
            category: Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "repair-stack".into(),
            name: "Repair ML Stack".into(),
            description: "Repair ML Stack installation".into(),
            script: String::new(), // Native Rust installer
            category: Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
            needs_sudo: true,
            experimental: false,
            note: None,
        },
        Component {
            id: "rccl-repair".into(),
            name: "Repair RCCL Multi-GPU".into(),
            description: "Probe and repair ROCm RCCL multi-GPU collectives".into(),
            script: String::new(), // Native Rust repair action
            category: Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "30-90 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        // ── Performance (native Rust via benchmark_runners module) ──
        Component {
            id: "mlperf-inference".into(),
            name: "MLPerf Inference".into(),
            description: "MLPerf benchmark suite for inference performance".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "30-60 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "rocm-benchmarks".into(),
            name: "ROCm Benchmarks".into(),
            description: "ROCm-specific performance benchmarks".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "20-40 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "gpu-memory-bandwidth".into(),
            name: "GPU Memory Bandwidth".into(),
            description: "Memory bandwidth performance testing".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "rocm-smi-bench".into(),
            name: "ROCm SMI Benchmarks".into(),
            description: "ROCm SMI performance monitoring".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-25 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "vllm-performance".into(),
            name: "vLLM Performance".into(),
            description: "High-throughput vLLM inference benchmark".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-25 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "deepspeed-performance".into(),
            name: "DeepSpeed Performance".into(),
            description: "DeepSpeed ZeRO optimization throughput benchmark".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "megatron-performance".into(),
            name: "Megatron-LM Performance".into(),
            description: "Megatron-LM import and throughput benchmark".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "onnx-performance".into(),
            name: "ONNX Runtime Performance".into(),
            description: "ONNX Runtime inference benchmark".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "1-2 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
        Component {
            id: "flash-attention-ck-performance".into(),
            name: "Flash Attention (CK) Performance".into(),
            description: "Genuine-model forward throughput via the CK backend".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "1-2 min".into(),
            needs_sudo: false,
            experimental: true,
            note: Some(
                "RDNA3 forward-only: measures inference throughput of a genuine \
                 Llama-style decoder through the composable-kernel flash_attn_func."
                    .into(),
            ),
        },
        Component {
            id: "rusty-llama-performance".into(),
            name: "Rusty Llama Performance".into(),
            description: "Rusty Llama (llama.cpp) inference benchmark".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "1-2 min".into(),
            needs_sudo: false,
            experimental: false,
            note: Some(
                "Runs llama-bench against the installed Rusty Llama binary — \
                 downloads a small verification model if needed, runs NO installs."
                    .into(),
            ),
        },
        Component {
            id: "all-benchmarks".into(),
            name: "Full Suite Benchmark".into(),
            description: "Run all post-installation performance tests".into(),
            script: String::new(), // Native Rust benchmark
            category: Performance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "45-90 min".into(),
            needs_sudo: false,
            experimental: false,
            note: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_performance_is_exposed_as_frontend_benchmark_option() {
        let component = default_components()
            .into_iter()
            .find(|component| component.id == "onnx-performance")
            .expect("ONNX benchmark should be exposed in component list");
        assert_eq!(component.category, Category::Performance);
        assert_eq!(component.name, "ONNX Runtime Performance");
    }
}
