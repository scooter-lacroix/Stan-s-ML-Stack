//! Platform module — hardware detection, environment management, and OS abstraction.
//!
//! This module provides platform-specific functionality including:
//! - GPU detection with fallback chain (rocminfo → lspci → sysfs)
//! - GPU architecture correction from marketing names
//! - ROCm version detection and path search
//! - Linux distribution detection and package manager identification
//! - Environment normalization, home directory resolution, Python interpreter discovery
//! - Component registry, installed component detection, and version querying
//! - Linux-specific platform operations
//! - Windows build support, WSL2 detection, and backend routing
//!
//! # Platform-Specific Modules
//!
//! - `linux` — Linux-native GPU/ROCm detection (Unix-only internals, cfg-gated)
//! - `windows` — Backend routing model (cross-platform)
//! - `wsl` — WSL2 detection, health checks, provisioning guidance (cross-platform)

pub mod detection;
pub mod environment;
#[cfg(unix)]
pub mod linux;
pub mod registry;
pub mod windows;
pub mod wsl;

pub use detection::*;
pub use environment::*;
#[cfg(unix)]
pub use linux::*;
pub use registry::*;
pub use windows::*;
pub use wsl::*;
