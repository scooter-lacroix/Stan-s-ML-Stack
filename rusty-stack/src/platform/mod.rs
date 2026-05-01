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

pub mod detection;
pub mod environment;
pub mod linux;
pub mod registry;

pub use detection::*;
pub use environment::*;
pub use linux::*;
pub use registry::*;
