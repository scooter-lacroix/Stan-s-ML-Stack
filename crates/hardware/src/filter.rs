//! iGPU Filtering Module
//!
//! Provides functionality to filter out integrated GPUs (iGPUs) from detection results.
//! This is critical for ROCm-only environments where iGPUs can interfere with HIP device selection.

use crate::gpu::GPUInfo;
use serde::{Deserialize, Serialize};

/// Patterns used to identify integrated GPUs.
///
/// These patterns are matched against GPU model names to detect iGPUs.
/// All patterns are case-insensitive.
pub const IGPU_PATTERNS: &[&str] = &[
    "raphael",             // Ryzen 7000 series iGPUs
    "integrated",          // Generic integrated graphics
    "igpu",                // Explicit iGPU naming
    "amd radeon graphics", // Generic Radeon iGPU pattern
    "gfx1030",             // Often used for Ryzen iGPUs
    "gfx1100",             // Can appear for some Ryzen iGPUs in older ROCm
    "unknown",             // "Unknown" GPU type is often the iGPU
    "family 17h",          // AMD GPU family for Ryzen iGPUs
    " Picasso",             // Ryzen 3000 series APUs
    "Raven",               // Ryzen 2000 series APUs
    "Cezanne",             // Ryzen 5000/6000 series APUs
    "Rembrandt",           // Ryzen 6000/7000 series mobile APUs
];

/// Represents a filter for excluding integrated GPUs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GPUFilter {
    /// Whether to exclude integrated GPUs
    pub exclude_igpus: bool,
    /// Additional custom patterns to match
    pub custom_patterns: Vec<String>,
    /// Minimum VRAM in GB to consider a GPU (filters out low-memory iGPUs)
    pub min_vram_gb: Option<f32>,
}

impl GPUFilter {
    /// Creates a new GPU filter with default settings.
    ///
    /// By default, iGPUs are excluded and GPUs with less than 4GB VRAM are filtered out.
    pub fn new() -> Self {
        Self {
            exclude_igpus: true,
            custom_patterns: Vec::new(),
            min_vram_gb: Some(4.0), // Filter out low-memory iGPUs by default
        }
    }

    /// Creates a filter that includes all GPUs (no filtering).
    pub fn include_all() -> Self {
        Self {
            exclude_igpus: false,
            custom_patterns: Vec::new(),
            min_vram_gb: None,
        }
    }

    /// Sets whether to exclude integrated GPUs.
    pub fn exclude_igpus(mut self, exclude: bool) -> Self {
        self.exclude_igpus = exclude;
        self
    }

    /// Adds a custom pattern to filter GPUs.
    pub fn with_pattern(mut self, pattern: String) -> Self {
        self.custom_patterns.push(pattern);
        self
    }

    /// Sets minimum VRAM requirement.
    pub fn with_min_vram(mut self, min_gb: f32) -> Self {
        self.min_vram_gb = Some(min_gb);
        self
    }

    /// Checks if a GPU model name matches iGPU patterns.
    ///
    /// Returns true if the GPU appears to be an integrated GPU.
    pub fn is_igpu(&self, model: &str) -> bool {
        let model_lower = model.to_lowercase();

        // Check standard iGPU patterns
        for pattern in IGPU_PATTERNS {
            if model_lower.contains(pattern) {
                return true;
            }
        }

        // Check for "AMD Ryzen" + "Graphics" pattern
        if model_lower.contains("amd ryzen") && model_lower.contains("graphics") {
            return true;
        }

        // Check for "AMD Ryzen" + small/low memory indicators
        if model_lower.contains("amd ryzen") && model_lower.contains(" amd ") {
            return true;
        }

        // Check custom patterns
        for pattern in &self.custom_patterns {
            if model_lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        false
    }

    /// Filters a list of GPUs, returning only those that pass the filter.
    pub fn filter(&self, gpus: Vec<GPUInfo>) -> Vec<GPUInfo> {
        gpus.into_iter()
            .filter(|gpu| self.should_include(gpu))
            .collect()
    }

    /// Determines whether a specific GPU should be included.
    pub fn should_include(&self, gpu: &GPUInfo) -> bool {
        // Check if explicitly marked as iGPU
        if self.exclude_igpus && gpu.is_igpu {
            return false;
        }

        // Check model name patterns
        if self.exclude_igpus && self.is_igpu(&gpu.model) {
            return false;
        }

        // Check minimum VRAM requirement
        if let Some(min_gb) = self.min_vram_gb {
            if gpu.memory_gb < min_gb {
                return false;
            }
        }

        true
    }

    /// Returns a list of GPUs with iGPUs marked.
    ///
    /// This updates the `is_igpu` field based on pattern matching.
    pub fn mark_igpus(&self, mut gpus: Vec<GPUInfo>) -> Vec<GPUInfo> {
        for gpu in &mut gpus {
            if self.is_igpu(&gpu.model) {
                gpu.is_igpu = true;
            }
        }
        gpus
    }
}

impl Default for GPUFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Quickly checks if a GPU model appears to be an iGPU.
///
/// This is a convenience function that uses the default filter.
pub fn is_likely_igpu(model: &str) -> bool {
    GPUFilter::new().is_igpu(model)
}

/// Filters out integrated GPUs from a list.
///
/// This is a convenience function that uses the default filter.
pub fn filter_igpus(gpus: Vec<GPUInfo>) -> Vec<GPUInfo> {
    GPUFilter::new().filter(gpus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::GPUArchitecture;

    #[test]
    fn test_igpu_detection_patterns() {
        let filter = GPUFilter::new();

        // Should detect as iGPU
        assert!(filter.is_igpu("AMD Ryzen 9 7950X Raphael"));
        assert!(filter.is_igpu("AMD Radeon Graphics"));
        assert!(filter.is_igpu("Integrated AMD Radeon"));
        assert!(filter.is_igpu("AMD Ryzen 7 7800X3D Graphics"));

        // Should NOT detect as iGPU
        assert!(!filter.is_igpu("AMD Radeon RX 7900 XTX"));
        assert!(!filter.is_igpu("AMD Radeon RX 7800 XT"));
        assert!(!filter.is_igpu("AMD Instinct MI300X"));
    }

    #[test]
    fn test_filter_applies() {
        let filter = GPUFilter::new();

        let gpus = vec![
            GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100),
            GPUInfo::new("Raphael".to_string(), GPUArchitecture::Gfx1030).as_igpu(),
            GPUInfo::new("RX 7800 XT".to_string(), GPUArchitecture::Gfx1101),
        ];

        let filtered = filter.filter(gpus);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|g| !g.is_igpu));
    }

    #[test]
    fn test_min_vram_filter() {
        let filter = GPUFilter::new().with_min_vram(8.0);

        let gpus = vec![
            GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100).with_memory(24.0),
            GPUInfo::new("Low Memory GPU".to_string(), GPUArchitecture::Gfx1030).with_memory(4.0),
        ];

        let filtered = filter.filter(gpus);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].model, "RX 7900 XTX");
    }

    #[test]
    fn test_custom_patterns() {
        let filter = GPUFilter::new()
            .with_pattern("Virtual".to_string())
            .with_pattern("Mock".to_string());

        assert!(filter.is_igpu("Virtual GPU"));
        assert!(filter.is_igpu("Mock AMD GPU"));
        assert!(!filter.is_igpu("Real AMD GPU"));
    }

    #[test]
    fn test_include_all() {
        let filter = GPUFilter::include_all();

        let gpus = vec![
            GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100),
            GPUInfo::new("Raphael".to_string(), GPUArchitecture::Gfx1030).as_igpu(),
        ];

        let filtered = filter.filter(gpus);
        assert_eq!(filtered.len(), 2); // Includes all
    }

    #[test]
    fn test_mark_igpus() {
        let filter = GPUFilter::new();

        let gpus = vec![
            GPUInfo::new("RX 7900 XTX".to_string(), GPUArchitecture::Gfx1100),
            GPUInfo::new("AMD Ryzen Graphics".to_string(), GPUArchitecture::Gfx1030),
        ];

        let marked = filter.mark_igpus(gpus);
        assert!(!marked[0].is_igpu);
        assert!(marked[1].is_igpu);
    }
}
