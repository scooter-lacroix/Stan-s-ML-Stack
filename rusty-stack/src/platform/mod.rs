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
//! - Windows ↔ WSL2 path translation bridge
//! - Service launch surfaces with localhost-only binding
//! - Windows-native control shell using shared Rust contracts
//!
//! # Platform-Specific Modules
//!
//! - `linux` — Linux-native GPU/ROCm detection (Unix-only internals, cfg-gated)
//! - `windows` — Backend routing model (cross-platform)
//! - `wsl` — WSL2 detection, health checks, provisioning guidance (cross-platform)
//! - `path_bridge` — Windows ↔ WSL2 path translation (cross-platform)
//! - `service` — Service launch with localhost-only binding and health endpoints
//! - `control_shell` — Windows-native control shell using shared Rust contracts

pub mod control_shell;
pub mod detection;
pub mod environment;
#[cfg(unix)]
pub mod linux;
pub mod path_bridge;
pub mod registry;
pub mod service;
pub mod windows;
pub mod wsl;

pub use control_shell::*;
pub use detection::*;
pub use environment::*;
#[cfg(unix)]
pub use linux::*;
pub use path_bridge::*;
pub use registry::*;
pub use service::*;
pub use windows::*;
pub use wsl::*;
