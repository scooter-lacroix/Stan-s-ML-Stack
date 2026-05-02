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

pub mod deepspeed;
pub mod flash_attention_ck;
pub mod megatron;
pub mod ml_stack;
pub mod mpi4py;
pub mod onnxruntime;
pub mod pytorch;
pub mod rocm;
pub mod triton;

// Re-export key types for convenience
pub use deepspeed::{DeepSpeedConfig, DeepSpeedInstaller};
pub use flash_attention_ck::{FlashAttentionConfig, FlashAttentionInstaller, GpuArch};
pub use megatron::{MegatronConfig, MegatronInstaller};
pub use ml_stack::{MlStackConfig, MlStackInstaller};
pub use mpi4py::{Mpi4PyConfig, Mpi4PyInstaller, MpiImplementation};
pub use onnxruntime::{HipArchs, OnnxRuntimeConfig, OnnxRuntimeInstaller};
pub use pytorch::{PyTorchConfig, PyTorchInstaller, TorchChannel};
pub use rocm::{RocmConfig, RocmChannel, RocmInstallType, RocmInstaller};
pub use triton::{TritonBranch, TritonConfig, TritonInstaller};
