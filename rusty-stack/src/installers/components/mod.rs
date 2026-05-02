//! Native Rust installer modules for major components.
//!
//! Each module ports the corresponding shell installer script, constructing
//! the correct system/pip commands matching the original script behavior.
//! All modules use `installers::common::` for shared operations (package
//! manager, distro detection, ROCm env, etc.).
//!
//! # Validation Assertions
//!
//! - **VAL-INSTALL-001**: ROCm installer correct package commands
//! - **VAL-INSTALL-002**: ROCm channel selection correct version pins
//! - **VAL-INSTALL-003**: PyTorch installer correct wheel selection
//! - **VAL-INSTALL-004**: Triton installer correct pip command
//! - **VAL-INSTALL-005**: MPI4Py installer detects system MPI
//! - **VAL-INSTALL-006**: DeepSpeed installer correct pip command
//! - **VAL-INSTALL-007**: ML Stack Core installer correct pip command
//! - **VAL-INSTALL-008**: Flash Attention correct cmake command
//! - **VAL-INSTALL-009**: Flash Attention correct make command
//! - **VAL-INSTALL-010**: ONNX Runtime correct cmake command
//! - **VAL-INSTALL-011**: ONNX Runtime pip install from build output
//! - **VAL-INSTALL-012**: Megatron correct git clone
//! - **VAL-INSTALL-013**: Megatron correct pip install post-clone
//! - **VAL-INSTALL-015**: vLLM correct git clone and pip install
//! - **VAL-INSTALL-016**: AITER correct git clone and pip install
//! - **VAL-INSTALL-017**: bitsandbytes correct pip command
//! - **VAL-INSTALL-018**: ROCm SMI correct package command
//! - **VAL-INSTALL-019**: MIGraphX correct pip command
//! - **VAL-INSTALL-020**: PyTorch Profiler correct pip command
//! - **VAL-INSTALL-021**: W&B correct pip command
//! - **VAL-INSTALL-022**: vLLM Studio correct git clone
//! - **VAL-INSTALL-023**: ComfyUI correct git clone and pip install
//! - **VAL-INSTALL-024**: text-generation-webui correct git clone
//! - **VAL-INSTALL-025**: App installers correct target directory
//! - **VAL-INSTALL-026**: Repair invokes correct sub-installers in order
//! - **VAL-INSTALL-027**: Repair propagates individual failures
//! - **VAL-INSTALL-028**: Permanent env creates correct venv and symlinks
//! - **VAL-INSTALL-029**: AMD GPU drivers correct package command
//! - **VAL-INSTALL-030**: MIGraphX Python correct pip command
//! - **VAL-INSTALL-047**: ComfyUI declares dependency on PyTorch

pub mod aiter;
pub mod amdgpu_drivers;
pub mod bitsandbytes_multi;
pub mod comfyui;
pub mod deepspeed;
pub mod flash_attention_ck;
pub mod megatron;
pub mod migraphx_multi;
pub mod migraphx_python;
pub mod ml_stack;
pub mod mpi4py;
pub mod onnxruntime;
pub mod permanent_env;
pub mod pytorch;
pub mod pytorch_profiler;
pub mod repair;
pub mod rocm;
pub mod rocm_smi;
pub mod textgen;
pub mod triton;
pub mod vllm_multi;
pub mod vllm_studio;
pub mod wandb;

// Re-export key types for convenience
pub use aiter::{AiterConfig, AiterInstaller};
pub use amdgpu_drivers::{AmdgpuConfig, AmdgpuInstaller};
pub use bitsandbytes_multi::{BitsAndBytesConfig, BitsAndBytesInstaller};
pub use comfyui::{ComfyuiConfig, ComfyuiInstaller};
pub use deepspeed::{DeepSpeedConfig, DeepSpeedInstaller};
pub use flash_attention_ck::{FlashAttentionConfig, FlashAttentionInstaller, GpuArch};
pub use megatron::{MegatronConfig, MegatronInstaller};
pub use migraphx_multi::{MigraphxConfig, MigraphxInstaller};
pub use migraphx_python::{InstallMethod as MigraphxPythonInstallMethod, MigraphxPythonConfig, MigraphxPythonInstaller};
pub use ml_stack::{MlStackConfig, MlStackInstaller};
pub use mpi4py::{Mpi4PyConfig, Mpi4PyInstaller, MpiImplementation};
pub use onnxruntime::{HipArchs, OnnxRuntimeConfig, OnnxRuntimeInstaller};
pub use permanent_env::{PermanentEnvConfig, PermanentEnvInstaller};
pub use pytorch::{PyTorchConfig, PyTorchInstaller, TorchChannel};
pub use pytorch_profiler::{PytorchProfilerConfig, PytorchProfilerInstaller};
pub use repair::{RepairConfig, RepairInstaller, RepairResult, RepairStep};
pub use rocm::{RocmConfig, RocmChannel, RocmInstallType, RocmInstaller};
pub use rocm_smi::{RocmSmiConfig, RocmSmiInstaller};
pub use textgen::{TextgenConfig, TextgenInstaller};
pub use triton::{TritonBranch, TritonConfig, TritonInstaller};
pub use vllm_multi::{VllmConfig, VllmInstaller};
pub use vllm_studio::{VllmStudioConfig, VllmStudioInstaller};
pub use wandb::{WandbConfig, WandbInstaller};
