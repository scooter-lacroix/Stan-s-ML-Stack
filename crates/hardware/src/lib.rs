//! # Hardware Detection Library
//!
//! A comprehensive Rust library for hardware detection, focusing on AMD GPUs and system information.
//! This library provides multi-method detection strategies for maximum compatibility across different
//! Linux systems and hardware configurations.
//!
//! ## Features
//!
//! - **Multi-method GPU Detection**: Uses rocminfo, lspci, and sysfs fallbacks
//! - **ROCm Version Detection**: Detects ROCm installation and version information
//! - **GPU Architecture Identification**: Maps PCI IDs to GPU architectures (GFX9, GFX11, etc.)
//! - **iGPU Filtering**: Excludes integrated GPUs from detection results
//! - **System Information**: Gathers system-level information using sysinfo crate
//!
//! ## Usage
//!
//! ```rust
//! use mlstack_hardware::{HardwareDiscovery, GPUFilter};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let discovery = HardwareDiscovery::new();
//!
//!     // Detect all GPUs
//!     let gpus = discovery.detect_gpus()?;
//!     println!("Found {} GPU(s)", gpus.len());
//!
//!     // Filter out integrated GPUs
//!     let discrete_gpus = GPUFilter::default().filter(gpus);
//!     println!("Found {} discrete GPU(s)", discrete_gpus.len());
//!
//!     // Get system information
//!     let system_info = discovery.detect_system()?;
//!     println!("OS: {} {}", system_info.os_version, system_info.kernel_version);
//!
//!     // Get ROCm information
//!     if let Ok(rocm) = discovery.detect_rocm() {
//!         println!("ROCm version: {}", rocm.version);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Detection Methods
//!
//! GPU detection uses a priority-based fallback system:
//!
//! 1. **rocminfo** (highest priority): Uses AMD's rocminfo tool for accurate GPU detection
//! 2. **lspci**: Queries PCI bus for GPU information
//! 3. **sysfs** (lowest priority): Reads from Linux sysfs filesystem
//!
//! Each method returns a confidence score, and results are merged with deduplication.
//!
//! ## Architecture Support
//!
//! Supported GPU architectures include:
//!
//! - **GFX9 Series**: gfx906 (Vega 20), gfx908 (Vega 10), gfx90a (Aldebaran), gfx942 (MI300X)
//! - **GFX11 Series**: gfx1100 (Navi 31), gfx1101 (Navi 32), gfx1151 (Navi 33)
//! - **GFX12 Series**: gfx1200 (Navi 44), gfx1201 (Navi 48)
//!
//! ## Examples
//!
//! See the `examples/` directory for more usage examples:
//!
//! - `basic_detection.rs`: Basic hardware detection
//! - `rocm_info.rs`: ROCm-specific detection
//! - `gpu_filtering.rs`: Filtering integrated GPUs

pub mod architecture;
pub mod discovery;
pub mod filter;
pub mod gpu;
pub mod rocm;

// Re-export main types for convenient access
pub use architecture::GPUArchitecture;
pub use discovery::HardwareDiscovery;
pub use filter::GPUFilter;
pub use gpu::GPUInfo;
pub use rocm::{Confidence, ROCmDetection, ROCmInfo};

use serde::{Deserialize, Serialize};

/// Represents detailed system information collected from the system.
///
/// Contains operating system, kernel, and CPU information gathered
/// using the sysinfo crate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemInfo {
    /// Operating system name (e.g., "Linux")
    pub os_name: String,
    /// Operating system version (e.g., "Ubuntu 22.04.3 LTS")
    pub os_version: String,
    /// Kernel version (e.g., "5.15.0-91-generic")
    pub kernel_version: String,
    /// CPU model name
    pub cpu_model: String,
    /// Number of physical CPU cores
    pub cpu_cores: usize,
    /// Total system memory in bytes
    pub memory_bytes: u64,
    /// Available system memory in bytes
    pub available_memory_bytes: u64,
}

impl SystemInfo {
    /// Creates a new SystemInfo instance.
    ///
    /// This is typically called internally by [`HardwareDiscovery::detect_system`].
    pub fn new(
        os_name: String,
        os_version: String,
        kernel_version: String,
        cpu_model: String,
        cpu_cores: usize,
        memory_bytes: u64,
        available_memory_bytes: u64,
    ) -> Self {
        Self {
            os_name,
            os_version,
            kernel_version,
            cpu_model,
            cpu_cores,
            memory_bytes,
            available_memory_bytes,
        }
    }

    /// Returns total memory in gigabytes.
    #[must_use]
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / 1_073_741_824.0
    }

    /// Returns available memory in gigabytes.
    #[must_use]
    pub fn available_memory_gb(&self) -> f64 {
        self.available_memory_bytes as f64 / 1_073_741_824.0
    }

    /// Returns the used memory percentage.
    #[must_use]
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_bytes == 0 {
            return 0.0;
        }
        ((self.memory_bytes - self.available_memory_bytes) as f64 / self.memory_bytes as f64)
            * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info_memory_gb() {
        let info = SystemInfo::new(
            "Linux".to_string(),
            "22.04".to_string(),
            "5.15.0".to_string(),
            "AMD Ryzen 9 7950X".to_string(),
            16,
            34_359_738_368, // 32 GB in bytes
            17_179_869_184, // 16 GB in bytes
        );

        assert_eq!(info.memory_gb(), 32.0);
        assert_eq!(info.available_memory_gb(), 16.0);
        assert!((info.memory_usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_system_info_zero_memory() {
        let info = SystemInfo::new(
            "Linux".to_string(),
            "22.04".to_string(),
            "5.15.0".to_string(),
            "AMD Ryzen 9 7950X".to_string(),
            16,
            0,
            0,
        );

        assert_eq!(info.memory_gb(), 0.0);
        assert_eq!(info.available_memory_gb(), 0.0);
        assert_eq!(info.memory_usage_percent(), 0.0);
    }
}
