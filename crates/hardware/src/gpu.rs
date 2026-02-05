//! GPU Information Structures
//!
//! Defines structures for representing GPU information including
//! model, architecture, memory, and thermal/power metrics.

use crate::architecture::GPUArchitecture;
use serde::{Deserialize, Serialize};

/// Represents information about a detected GPU.
///
/// This structure contains all relevant information about a GPU including
/// its model name, architecture, memory capacity, and current metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GPUInfo {
    /// GPU model name (e.g., "AMD Radeon RX 7900 XTX")
    pub model: String,
    /// GPU architecture (e.g., Gfx1100)
    pub architecture: GPUArchitecture,
    /// GPU memory in gigabytes
    pub memory_gb: f32,
    /// Current GPU temperature in Celsius (if available)
    pub temperature_c: Option<f32>,
    /// Current GPU power consumption in watts (if available)
    pub power_watts: Option<f32>,
    /// Whether this is an integrated GPU (iGPU)
    pub is_igpu: bool,
    /// PCI bus ID (e.g., "0000:03:00.0")
    pub pci_bus_id: Option<String>,
    /// GPU device index (for HIP_VISIBLE_DEVICES)
    pub device_index: Option<usize>,
    /// Raw rocminfo output for this GPU (if available)
    pub raw_info: Option<String>,
}

impl GPUInfo {
    /// Creates a new GPUInfo with default values.
    pub fn new(model: String, architecture: GPUArchitecture) -> Self {
        Self {
            model,
            architecture,
            memory_gb: 0.0,
            temperature_c: None,
            power_watts: None,
            is_igpu: false,
            pci_bus_id: None,
            device_index: None,
            raw_info: None,
        }
    }

    /// Sets the GPU memory capacity.
    pub fn with_memory(mut self, memory_gb: f32) -> Self {
        self.memory_gb = memory_gb;
        self
    }

    /// Sets the GPU temperature.
    pub fn with_temperature(mut self, temp_c: f32) -> Self {
        self.temperature_c = Some(temp_c);
        self
    }

    /// Sets the GPU power consumption.
    pub fn with_power(mut self, power_watts: f32) -> Self {
        self.power_watts = Some(power_watts);
        self
    }

    /// Marks this GPU as an integrated GPU.
    pub fn as_igpu(mut self) -> Self {
        self.is_igpu = true;
        self
    }

    /// Sets the PCI bus ID.
    pub fn with_pci_bus_id(mut self, bus_id: String) -> Self {
        self.pci_bus_id = Some(bus_id);
        self
    }

    /// Sets the device index.
    pub fn with_device_index(mut self, index: usize) -> Self {
        self.device_index = Some(index);
        self
    }

    /// Returns true if this GPU supports ROCm.
    ///
    /// All architectures except some very old ones support ROCm.
    pub fn supports_rocm(&self) -> bool {
        !self.architecture.is_legacy()
            || matches!(
                self.architecture,
                crate::architecture::GPUArchitecture::Gfx906
            )
    }

    /// Returns true if this GPU requires HSA_OVERRIDE_GFX_VERSION.
    ///
    /// RDNA 3 and RDNA 4 GPUs require this for PyTorch compatibility.
    pub fn requires_hsa_override(&self) -> bool {
        self.architecture.is_rdna3() || self.architecture.is_rdna4()
    }

    /// Returns the recommended HSA_OVERRIDE_GFX_VERSION value.
    pub fn hsa_override_value(&self) -> Option<&'static str> {
        self.architecture.hsa_override_gfx_version()
    }
}

impl Default for GPUInfo {
    fn default() -> Self {
        Self {
            model: String::new(),
            architecture: GPUArchitecture::Gfx1100,
            memory_gb: 0.0,
            temperature_c: None,
            power_watts: None,
            is_igpu: false,
            pci_bus_id: None,
            device_index: None,
            raw_info: None,
        }
    }
}

/// Represents a collection of GPUs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GPUCollection {
    /// List of detected GPUs
    pub gpus: Vec<GPUInfo>,
    /// Number of discrete GPUs
    pub discrete_count: usize,
    /// Number of integrated GPUs
    pub igpu_count: usize,
}

impl GPUCollection {
    /// Creates a new empty GPU collection.
    pub fn new() -> Self {
        Self {
            gpus: Vec::new(),
            discrete_count: 0,
            igpu_count: 0,
        }
    }

    /// Adds a GPU to the collection.
    pub fn add(&mut self, gpu: GPUInfo) {
        if gpu.is_igpu {
            self.igpu_count += 1;
        } else {
            self.discrete_count += 1;
        }
        self.gpus.push(gpu);
    }

    /// Returns only discrete (non-integrated) GPUs.
    pub fn discrete_gpus(&self) -> Vec<&GPUInfo> {
        self.gpus.iter().filter(|g| !g.is_igpu).collect()
    }

    /// Returns only integrated GPUs.
    pub fn igpus(&self) -> Vec<&GPUInfo> {
        self.gpus.iter().filter(|g| g.is_igpu).collect()
    }

    /// Returns the primary GPU (first discrete GPU, or first GPU if none).
    pub fn primary_gpu(&self) -> Option<&GPUInfo> {
        self.discrete_gpus()
            .into_iter()
            .next()
            .or_else(|| self.gpus.first())
    }

    /// Returns true if any GPU requires HSA_OVERRIDE_GFX_VERSION.
    pub fn requires_hsa_override(&self) -> bool {
        self.gpus.iter().any(|g| g.requires_hsa_override())
    }

    /// Returns the HIP_VISIBLE_DEVICES value for discrete GPUs.
    pub fn hip_visible_devices(&self) -> String {
        self.discrete_gpus()
            .iter()
            .filter_map(|g| g.device_index.map(|i| i.to_string()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl Default for GPUCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::GPUArchitecture;

    #[test]
    fn test_gpu_info_builder() {
        let gpu = GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100)
            .with_memory(24.0)
            .with_temperature(65.0)
            .with_power(350.0)
            .with_pci_bus_id("0000:03:00.0".to_string())
            .with_device_index(0);

        assert_eq!(gpu.model, "RX 7900 XTX");
        assert_eq!(gpu.architecture, GPUArchitecture::Gfx1100);
        assert_eq!(gpu.memory_gb, 24.0);
        assert_eq!(gpu.temperature_c, Some(65.0));
        assert_eq!(gpu.power_watts, Some(350.0));
        assert!(!gpu.is_igpu);
    }

    #[test]
    fn test_gpu_collection() {
        let mut collection = GPUCollection::new();

        let discrete =
            GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100).with_device_index(0);
        let igpu = GPUInfo::new("Raphael".to_string(), GPUArchitecture::Gfx1030)
            .as_igpu()
            .with_device_index(1);

        collection.add(discrete);
        collection.add(igpu);

        assert_eq!(collection.discrete_count, 1);
        assert_eq!(collection.igpu_count, 1);
        assert_eq!(collection.discrete_gpus().len(), 1);
        assert_eq!(collection.igpus().len(), 1);
    }

    #[test]
    fn test_hip_visible_devices() {
        let mut collection = GPUCollection::new();

        collection
            .add(GPUInfo::new("GPU1".to_string(), GPUArchitecture::Gfx1100).with_device_index(0));
        collection
            .add(GPUInfo::new("GPU2".to_string(), GPUArchitecture::Gfx1100).with_device_index(1));
        collection.add(
            GPUInfo::new("iGPU".to_string(), GPUArchitecture::Gfx1030)
                .as_igpu()
                .with_device_index(2),
        );

        assert_eq!(collection.hip_visible_devices(), "0,1");
    }
}
