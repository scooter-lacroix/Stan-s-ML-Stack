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

pub mod deepspeed;
pub mod ml_stack;
pub mod mpi4py;
pub mod pytorch;
pub mod rocm;
pub mod triton;

// Re-export key types for convenience
pub use deepspeed::{DeepSpeedInstaller, DeepSpeedConfig};
pub use ml_stack::{MlStackInstaller, MlStackConfig};
pub use mpi4py::{Mpi4PyInstaller, Mpi4PyConfig, MpiImplementation};
pub use pytorch::{PyTorchInstaller, PyTorchConfig, TorchChannel};
pub use rocm::{RocmInstaller, RocmConfig, RocmChannel, RocmInstallType};
pub use triton::{TritonInstaller, TritonConfig, TritonBranch};
