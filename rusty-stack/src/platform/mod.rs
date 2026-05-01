//! Platform module — hardware detection, environment management, and OS abstraction.
//!
//! This module provides platform-specific functionality including:
//! - GPU detection with fallback chain (rocminfo → lspci → sysfs)
//! - GPU architecture correction from marketing names
//! - ROCm version detection and path search
//! - Linux-specific platform operations

pub mod linux;

pub use linux::*;
