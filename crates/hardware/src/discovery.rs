//! Hardware Discovery Module
//!
//! Provides comprehensive hardware detection capabilities including
//! GPU detection via multiple methods, system information gathering,
//! and ROCm detection.

use crate::architecture::GPUArchitecture;
use crate::filter::GPUFilter;
use crate::gpu::{GPUCollection, GPUInfo};
use crate::rocm::{ROCmDetection, ROCmInfo};
use crate::SystemInfo;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use sysinfo::System;

/// Trait for GPU detection methods.
pub trait GPUDetector: Send + Sync {
    /// Detects GPUs using this method.
    fn detect(&self) -> Result<Vec<GPUInfo>>;
}

/// Detector using rocminfo.
pub struct RocminfoDetector;
impl GPUDetector for RocminfoDetector {
    fn detect(&self) -> Result<Vec<GPUInfo>> {
        let output = Command::new("rocminfo")
            .output()
            .context("Failed to execute rocminfo")?;

        if !output.status.success() {
            anyhow::bail!("rocminfo returned non-zero exit code");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();
        let mut current_gpu: Option<GPUInfo> = None;
        let mut device_index = 0;

        for line in stdout.lines() {
            if line.contains("Name:") && !line.contains("Marketing Name") {
                let name = line.split(':').nth(1).unwrap_or("").trim().to_string();
                if name.to_lowercase().contains("cpu") {
                    continue;
                }
                current_gpu = Some(GPUInfo::new(name, GPUArchitecture::Gfx1100));
            }

            if line.contains("gfx") && current_gpu.is_some() {
                let gfx_str = line.trim();
                if let Ok(arch) = gfx_str.parse::<GPUArchitecture>() {
                    if let Some(ref mut gpu) = current_gpu {
                        gpu.architecture = arch;
                    }
                }
            }

            if line.contains("******") && current_gpu.is_some() {
                if let Some(mut gpu) = current_gpu.take() {
                    gpu.device_index = Some(device_index);
                    gpus.push(gpu);
                    device_index += 1;
                }
            }
        }

        if let Some(mut gpu) = current_gpu {
            gpu.device_index = Some(device_index);
            gpus.push(gpu);
        }

        Ok(gpus)
    }
}

/// Detector using lspci.
pub struct LspciDetector;
impl GPUDetector for LspciDetector {
    fn detect(&self) -> Result<Vec<GPUInfo>> {
        let output = Command::new("lspci")
            .output()
            .context("Failed to execute lspci")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();
        let mut device_index = 0;

        for line in stdout.lines() {
            let lower = line.to_lowercase();
            if lower.contains("amd") && lower.contains("vga") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let pci_id = parts.first().map(|s| s.to_string());
                let model = if parts.len() > 2 {
                    parts[2..].join(" ")
                } else {
                    "AMD GPU".to_string()
                };

                let mut gpu = GPUInfo::new(model, GPUArchitecture::Gfx1100);
                gpu.pci_bus_id = pci_id;
                gpu.device_index = Some(device_index);
                gpus.push(gpu);
                device_index += 1;
            }
        }
        Ok(gpus)
    }
}

/// Detector using sysfs.
pub struct SysfsDetector;
impl GPUDetector for SysfsDetector {
    fn detect(&self) -> Result<Vec<GPUInfo>> {
        let kfd_path = Path::new("/sys/class/kfd/kfd/topology/nodes/");
        if !kfd_path.exists() {
            anyhow::bail!("Sysfs KFD path not found");
        }

        let mut gpus = Vec::new();
        let mut device_index = 0;

        if let Ok(entries) = fs::read_dir(kfd_path) {
            for entry in entries.flatten() {
                let node_path = entry.path();
                let gpu_id_file = node_path.join("gpu_id");
                if gpu_id_file.exists() {
                    let name_file = node_path.join("name");
                    let model = fs::read_to_string(name_file)
                        .unwrap_or_else(|_| format!("GPU Node {}", device_index))
                        .trim()
                        .to_string();

                    let props_file = node_path.join("properties");
                    let arch = if let Ok(props) = fs::read_to_string(props_file) {
                        HardwareDiscovery::parse_arch_from_properties(&props)
                    } else {
                        GPUArchitecture::Gfx1100
                    };

                    let mut gpu = GPUInfo::new(model, arch);
                    gpu.device_index = Some(device_index);
                    gpus.push(gpu);
                    device_index += 1;
                }
            }
        }
        Ok(gpus)
    }
}

/// Main hardware discovery interface.
pub struct HardwareDiscovery {
    filter: GPUFilter,
    detectors: Vec<Box<dyn GPUDetector>>,
}

impl HardwareDiscovery {
    /// Creates a new HardwareDiscovery with default settings.
    pub fn new() -> Self {
        Self {
            filter: GPUFilter::new(),
            detectors: vec![
                Box::new(RocminfoDetector),
                Box::new(LspciDetector),
                Box::new(SysfsDetector),
            ],
        }
    }

    /// Creates a new HardwareDiscovery with custom detectors (for testing).
    pub fn with_detectors(detectors: Vec<Box<dyn GPUDetector>>) -> Self {
        Self {
            filter: GPUFilter::new(),
            detectors,
        }
    }

    /// Creates a new HardwareDiscovery with a custom GPU filter.
    pub fn with_filter(filter: GPUFilter) -> Self {
        Self {
            filter,
            detectors: vec![
                Box::new(RocminfoDetector),
                Box::new(LspciDetector),
                Box::new(SysfsDetector),
            ],
        }
    }

    /// Detects all GPUs using configured detectors.
    pub fn detect_gpus(&self) -> Result<Vec<GPUInfo>> {
        let mut all_gpus = Vec::new();

        for detector in &self.detectors {
            if let Ok(gpus) = detector.detect() {
                if !gpus.is_empty() {
                    all_gpus = gpus;
                    break;
                }
            }
        }

        // Mark iGPUs and apply filter
        let marked = self.filter.mark_igpus(all_gpus);
        let filtered = self.filter.filter(marked);

        // Try to enrich with SMI if available and further filter out disabled GPUs
        let mut final_gpus = filtered;
        let _ = self.enrich_gpu_info(&mut final_gpus);

        // Additional filter: remove GPUs that don't appear in rocm-smi
        // This catches disabled iGPUs that appear in sysfs but aren't actually accessible
        let accessible = self.filter_accessible_gpus(final_gpus);

        Ok(accessible)
    }

    /// Filters GPUs to only include those accessible via rocm-smi.
    ///
    /// This is important because disabled iGPUs may appear in sysfs/KFD nodes
    /// but aren't actually accessible for computation.
    fn filter_accessible_gpus(&self, gpus: Vec<GPUInfo>) -> Vec<GPUInfo> {
        // Get GPU list from rocm-smi to verify accessibility
        let smi_gpus = self.get_smi_gpu_indices();

        if smi_gpus.is_empty() {
            // If rocm-smi isn't available, return all GPUs
            return gpus;
        }

        // Only include GPUs that appear in rocm-smi output
        gpus.into_iter()
            .filter(|gpu| {
                if let Some(idx) = gpu.device_index {
                    smi_gpus.contains(&idx)
                } else {
                    false
                }
            })
            .collect()
    }

    /// Gets the GPU indices reported by rocm-smi.
    fn get_smi_gpu_indices(&self) -> Vec<usize> {
        let output = Command::new("rocm-smi")
            .args(["--showgpuindex", "--csv"])
            .output();

        let mut indices = Vec::new();

        if let Ok(o) = output {
            if o.status.success() {
                let stdout = String::from_utf8_lossy(&o.stdout);
                for line in stdout.lines().skip(1) { // Skip header
                    let parts: Vec<&str> = line.split(',').collect();
                    if !parts.is_empty() {
                        if let Ok(idx) = parts[0].trim().parse::<usize>() {
                            indices.push(idx);
                        }
                    }
                }
            }
        }

        indices
    }

    /// Parses GPU architecture from sysfs properties.
    fn parse_arch_from_properties(props: &str) -> GPUArchitecture {
        for line in props.lines() {
            // Look for a word that starts with "gfx" followed by digits
            for word in line.split_whitespace() {
                if word.starts_with("gfx") && word.len() > 3 {
                    if let Ok(arch) = word.parse::<GPUArchitecture>() {
                        return arch;
                    }
                }
            }
        }
        GPUArchitecture::Gfx1100
    }

    /// Enriches GPU information with data from rocm-smi.
    fn enrich_gpu_info(&self, gpus: &mut [GPUInfo]) -> Result<()> {
        // Get memory info
        if let Ok(output) = Command::new("rocm-smi")
            .args(["--showmeminfo", "vram", "--csv"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (idx, line) in stdout.lines().skip(1).enumerate() {
                    if idx >= gpus.len() {
                        break;
                    }

                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        if let Ok(memory_mb) = parts[1].trim().parse::<f32>() {
                            gpus[idx].memory_gb = memory_mb / 1024.0;
                        }
                    }
                }
            }
        }

        // Get temperature
        if let Ok(output) = Command::new("rocm-smi")
            .args(["--showtemp", "--csv"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (idx, line) in stdout.lines().skip(1).enumerate() {
                    if idx >= gpus.len() {
                        break;
                    }

                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        if let Ok(temp) = parts[1].trim().parse::<f32>() {
                            gpus[idx].temperature_c = Some(temp);
                        }
                    }
                }
            }
        }

        // Get power
        if let Ok(output) = Command::new("rocm-smi")
            .args(["--showpower", "--csv"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (idx, line) in stdout.lines().skip(1).enumerate() {
                    if idx >= gpus.len() {
                        break;
                    }

                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        if let Ok(power) = parts[1].trim().parse::<f32>() {
                            gpus[idx].power_watts = Some(power);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Detects system information.
    pub fn detect_system(&self) -> Result<SystemInfo> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let os_name = System::name().unwrap_or_else(|| "Unknown".to_string());
        let os_version = System::long_os_version().unwrap_or_else(|| "Unknown".to_string());
        let kernel_version = System::kernel_version().unwrap_or_else(|| "Unknown".to_string());

        let cpu_model = sys.global_cpu_info().brand().to_string();
        let cpu_cores = sys.cpus().len();

        let memory_bytes = sys.total_memory() * 1024; // Convert from KB to bytes
        let available_memory_bytes = sys.available_memory() * 1024;

        Ok(SystemInfo::new(
            os_name,
            os_version,
            kernel_version,
            cpu_model,
            cpu_cores,
            memory_bytes,
            available_memory_bytes,
        ))
    }

    /// Detects ROCm installation.
    pub fn detect_rocm(&self) -> Result<ROCmInfo> {
        let detection = ROCmDetection::new();
        let info = detection.detect();

        if info.confidence.is_sufficient() {
            Ok(info)
        } else {
            anyhow::bail!("ROCm not detected or insufficient confidence")
        }
    }

    /// Returns the HIP_VISIBLE_DEVICES string for detected GPUs.
    pub fn hip_visible_devices(&self) -> Result<String> {
        let gpus = self.detect_gpus()?;
        let collection = GPUCollection {
            gpus,
            discrete_count: 0,
            igpu_count: 0,
        };
        Ok(collection.hip_visible_devices())
    }
}

impl Default for HardwareDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_discovery_creation() {
        let discovery = HardwareDiscovery::new();
        assert!(discovery.filter.exclude_igpus);
    }

    #[test]
    fn test_parse_arch_from_properties() {
        let props = "device_type GPU\ngfx_version gfx1100\n";
        let arch = HardwareDiscovery::parse_arch_from_properties(props);
        assert_eq!(arch, GPUArchitecture::Gfx1100);

        let props2 = "device_type GPU\ngfx_version gfx1201\n";
        let arch2 = HardwareDiscovery::parse_arch_from_properties(props2);
        assert_eq!(arch2, GPUArchitecture::Gfx1201);
    }

    /// Mock detector for testing.
    struct MockDetector(Vec<GPUInfo>);
    impl GPUDetector for MockDetector {
        fn detect(&self) -> Result<Vec<GPUInfo>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn test_mock_hardware_discovery() {
        let mock_gpu = GPUInfo::new("Mock GPU".to_string(), GPUArchitecture::Gfx1100);
        let detectors: Vec<Box<dyn GPUDetector>> = vec![Box::new(MockDetector(vec![mock_gpu]))];
        let discovery = HardwareDiscovery::with_detectors(detectors);

        let gpus = discovery.detect_gpus().unwrap();
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].model, "Mock GPU");
    }
}
