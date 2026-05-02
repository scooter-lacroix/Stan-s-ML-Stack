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

pub mod aiter;
pub mod bitsandbytes_multi;
pub mod deepspeed;
pub mod flash_attention_ck;
pub mod megatron;
pub mod migraphx_multi;
pub mod ml_stack;
pub mod mpi4py;
pub mod onnxruntime;
pub mod pytorch;
pub mod pytorch_profiler;
pub mod rocm;
pub mod rocm_smi;
pub mod triton;
pub mod vllm_multi;
pub mod wandb;

// Re-export key types for convenience
pub use aiter::{AiterConfig, AiterInstaller};
pub use bitsandbytes_multi::{BitsAndBytesConfig, BitsAndBytesInstaller};
pub use deepspeed::{DeepSpeedConfig, DeepSpeedInstaller};
pub use flash_attention_ck::{FlashAttentionConfig, FlashAttentionInstaller, GpuArch};
pub use megatron::{MegatronConfig, MegatronInstaller};
pub use migraphx_multi::{MigraphxConfig, MigraphxInstaller};
pub use ml_stack::{MlStackConfig, MlStackInstaller};
pub use mpi4py::{Mpi4PyConfig, Mpi4PyInstaller, MpiImplementation};
pub use onnxruntime::{HipArchs, OnnxRuntimeConfig, OnnxRuntimeInstaller};
pub use pytorch::{PyTorchConfig, PyTorchInstaller, TorchChannel};
pub use pytorch_profiler::{PytorchProfilerConfig, PytorchProfilerInstaller};
pub use rocm::{RocmConfig, RocmChannel, RocmInstallType, RocmInstaller};
pub use rocm_smi::{RocmSmiConfig, RocmSmiInstaller};
pub use triton::{TritonBranch, TritonConfig, TritonInstaller};
pub use vllm_multi::{VllmConfig, VllmInstaller};
pub use wandb::{WandbConfig, WandbInstaller};
