use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Welcome,
    HardwareDetect,
    Preflight,
    ComponentSelect,
    Configuration,
    Confirm,
    Installing,
    Complete,
    Recovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Foundation,
    Core,
    Extension,
    Environment,
    Verification,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Category::Foundation => "Foundation",
            Category::Core => "Core",
            Category::Extension => "Extensions",
            Category::Environment => "Environment",
            Category::Verification => "Verification",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Component {
    pub id: String,
    pub name: String,
    pub description: String,
    pub script: String,
    pub category: Category,
    pub required: bool,
    pub selected: bool,
    pub installed: bool,
    pub progress: f32,
    pub estimate: String,
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
        Component {
            id: "rocm".into(),
            name: "ROCm Platform".into(),
            description: "AMD ROCm GPU computing platform".into(),
            script: "install_rocm.sh".into(),
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "30-45 min".into(),
        },
        Component {
            id: "pytorch".into(),
            name: "PyTorch with ROCm".into(),
            description: "PyTorch optimized for AMD GPUs".into(),
            script: "install_pytorch_rocm.sh".into(),
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
        },
        Component {
            id: "triton".into(),
            name: "Triton".into(),
            description: "Compiler for parallel programming".into(),
            script: "install_triton_multi.sh".into(),
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
        },
        Component {
            id: "mpi4py".into(),
            name: "MPI4Py".into(),
            description: "MPI bindings for Python".into(),
            script: "install_mpi4py.sh".into(),
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
        },
        Component {
            id: "deepspeed".into(),
            name: "DeepSpeed".into(),
            description: "Deep learning optimization library".into(),
            script: "install_deepspeed.sh".into(),
            category: Foundation,
            required: true,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "8-12 min".into(),
        },
        Component {
            id: "ml-stack-core".into(),
            name: "ML Stack Core".into(),
            description: "Core ML Stack components".into(),
            script: "install_ml_stack.sh".into(),
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-20 min".into(),
        },
        Component {
            id: "flash-attn".into(),
            name: "Flash Attention".into(),
            description: "Efficient attention computation".into(),
            script: "install_flash_attention_ck.sh".into(),
            category: Core,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "20-30 min".into(),
        },
        Component {
            id: "repair-stack".into(),
            name: "Repair ML Stack".into(),
            description: "Repair ML Stack installation".into(),
            script: "repair_ml_stack.sh".into(),
            category: Core,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
        },
        Component {
            id: "megatron".into(),
            name: "Megatron-LM".into(),
            description: "Large-scale training framework".into(),
            script: "install_megatron.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "25-40 min".into(),
        },
        Component {
            id: "vllm".into(),
            name: "vLLM".into(),
            description: "High-throughput inference engine".into(),
            script: "install_vllm_multi.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "15-25 min".into(),
        },
        Component {
            id: "aiter".into(),
            name: "AITER".into(),
            description: "AMD AITER optimization tooling".into(),
            script: "install_aiter.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
        },
        Component {
            id: "vllm-studio".into(),
            name: "vLLM Studio".into(),
            description: "Model lifecycle manager for vLLM/SGLang".into(),
            script: "install_vllm_studio.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-20 min".into(),
        },
        Component {
            id: "onnx".into(),
            name: "ONNX Runtime".into(),
            description: "Cross-platform inference accelerator".into(),
            script: "build_onnxruntime_multi.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "45-90 min".into(),
        },
        Component {
            id: "bitsandbytes".into(),
            name: "BITSANDBYTES".into(),
            description: "Efficient quantization for deep learning".into(),
            script: "install_bitsandbytes_multi.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
        },
        Component {
            id: "rocm-smi".into(),
            name: "ROCm SMI".into(),
            description: "System monitoring for AMD GPUs".into(),
            script: "install_rocm_smi.sh".into(),
            category: Extension,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "2-3 min".into(),
        },
        Component {
            id: "migraphx".into(),
            name: "MIGraphX".into(),
            description: "AMD graph inference engine".into(),
            script: "install_migraphx_multi.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
        },
        Component {
            id: "pytorch-profiler".into(),
            name: "PyTorch Profiler".into(),
            description: "Performance analysis for PyTorch".into(),
            script: "install_pytorch_profiler.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "3-5 min".into(),
        },
        Component {
            id: "wandb".into(),
            name: "Weights & Biases".into(),
            description: "Experiment tracking and visualization".into(),
            script: "install_wandb.sh".into(),
            category: Extension,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-8 min".into(),
        },
        Component {
            id: "permanent-env".into(),
            name: "Permanent ROCm Env".into(),
            description: "Unified permanent environment for Python 3.12".into(),
            script: "setup_permanent_rocm_env.sh".into(),
            category: Environment,
            required: false,
            selected: true,
            installed: false,
            progress: 0.0,
            estimate: "1-2 min".into(),
        },
        Component {
            id: "verify-basic".into(),
            name: "Verify Installation".into(),
            description: "Basic verification".into(),
            script: "verify_installation.sh".into(),
            category: Verification,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "2-5 min".into(),
        },
        Component {
            id: "verify-enhanced".into(),
            name: "Enhanced Verify Installation".into(),
            description: "Advanced verification".into(),
            script: "enhanced_verify_installation.sh".into(),
            category: Verification,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "5-10 min".into(),
        },
        Component {
            id: "verify-build".into(),
            name: "Verify and Build".into(),
            description: "Verify + build components".into(),
            script: "verify_and_build.sh".into(),
            category: Verification,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: "10-15 min".into(),
        },
    ]
}
