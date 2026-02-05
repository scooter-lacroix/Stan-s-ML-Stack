//! ML Stack Installers
//!
//! Provides native Rust installers for ROCm, PyTorch, and ML components
//! with repository configuration, GPG key management, and installation profiles.
//!
//! ## Components
//!
//! ### Foundation
//! - `rocm`: ROCm platform installer
//! - `pytorch`: PyTorch with ROCm support
//! - `triton`: Triton compiler
//!
//! ### Inference & Optimization
//! - `vllm`: High-throughput LLM inference
//! - `onnx`: ONNX Runtime with ROCm
//! - `migraphx`: AMD MIGraphX graph optimizer
//! - `flash_attn`: Flash Attention for efficient attention
//!
//! ### Training & Optimization
//! - `deepspeed`: Distributed training with ZeRO
//! - `megatron`: Large-scale model training
//! - `bitsandbytes`: Model quantization (8-bit/4-bit)
//! - `aiter`: AMD AI Tensor Engine
//!
//! ### Monitoring & Profiling
//! - `rocm_smi`: GPU monitoring and management
//! - `pytorch_profiler`: Performance profiling
//! - `wandb`: Experiment tracking

pub mod builder;
pub mod common;
pub mod flash_attn;
pub mod gpg;
pub mod megatron;
pub mod onnx;
pub mod package_manager;
pub mod profiles;
pub mod pytorch;
pub mod repair;
pub mod repository;
pub mod rocm;
pub mod triton;
pub mod verification;
pub mod vllm;

// New installer modules
pub mod aiter;
pub mod bitsandbytes;
pub mod deepspeed;
pub mod migraphx;
pub mod pytorch_profiler;
pub mod rocm_smi;
pub mod wandb;

// Core exports
pub use builder::{presets, BuildConfig, BuildSystem, SourceBuilder, SourceRepository};
pub use common::{InstallationStatus, Installer, InstallerError, ProgressCallback};
pub use gpg::GpgKeyManager;
pub use package_manager::{
    OutputFormatter, PackageManager as UnifiedPackageManager, PackageRequest, PackageResult,
    PackageType,
};
pub use profiles::{InstallProfile, ProfileConfig};
pub use repair::RepairModule;
pub use repository::{PackageManager as RepoPackageManager, RepositoryConfig};
pub use verification::{VerificationItem, VerificationModule, VerificationStatus};

// Foundation component exports
pub use flash_attn::{FlashAttentionInstaller, FlashAttentionVersion};
pub use pytorch::{CudaBlockConfig, PyTorchInstallMethod, PyTorchInstaller, PyTorchVersion};
pub use rocm::{RocmInstaller, RocmVersion};
pub use triton::{TritonInstaller, TritonSource, TritonVersion};

// Inference & optimization exports
pub use megatron::{MegatronInstaller, MegatronPatchManager, MegatronSource, MegatronVersion};
pub use onnx::{ExecutionProvider, OnnxInstaller, OnnxSource, OnnxVerification, OnnxVersion};
pub use vllm::{VllmInstaller, VllmPatch, VllmPatchManager, VllmSource, VllmVersion};

// New component exports
pub use aiter::{
    AiterBuildConfig, AiterEnvironment, AiterInstaller, AiterSource, AiterVerificationResult,
    AiterVersion, GpuArchitecture,
};
pub use bitsandbytes::{
    BitsAndBytesBuildConfig, BitsAndBytesEnvironment, BitsAndBytesInstaller, BitsAndBytesSource,
    BitsAndBytesVerification, BitsAndBytesVersion, QuantizationBits,
};
pub use deepspeed::{
    generate_ds_config, DeepSpeedBuildConfig, DeepSpeedEnvironment, DeepSpeedInstaller,
    DeepSpeedOps, DeepSpeedSource, DeepSpeedVerification, DeepSpeedVersion, ZeroStage,
};
pub use migraphx::{
    MigraphxBuildConfig, MigraphxEnvironment, MigraphxInstaller, MigraphxSource,
    MigraphxVerification, MigraphxVersion, ModelFormat, OptimizationTarget,
};
pub use pytorch_profiler::{
    ProfileActivity, ProfilerConfig, ProfilerEnvironment, ProfilerOutput, ProfilerVerification,
    PytorchProfilerInstaller,
};
pub use rocm_smi::{GpuInfo, PerformanceLevel, PowerProfile, RocmSmiInstaller, RocmSmiSource};
pub use wandb::{
    log_amd_gpu_metrics, WandbConfig, WandbInstaller, WandbMode, WandbVerification, WandbVersion,
};
