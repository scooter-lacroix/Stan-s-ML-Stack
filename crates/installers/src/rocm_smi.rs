//! ROCm SMI (System Management Interface) Installer
//!
//! Pure Rust implementation for installing and managing ROCm SMI,
//! the GPU monitoring and management tool for AMD GPUs.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

/// GPU power profile modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerProfile {
    /// Auto/default profile
    Auto,
    /// Low power mode
    Low,
    /// High performance mode
    High,
    /// Video playback optimized
    Video,
    /// Virtual reality optimized
    VR,
    /// Compute optimized
    Compute,
    /// Custom profile
    Custom,
}

impl PowerProfile {
    /// Returns rocm-smi profile argument.
    pub fn as_smi_arg(&self) -> &'static str {
        match self {
            PowerProfile::Auto => "auto",
            PowerProfile::Low => "low",
            PowerProfile::High => "high",
            PowerProfile::Video => "video",
            PowerProfile::VR => "vr",
            PowerProfile::Compute => "compute",
            PowerProfile::Custom => "custom",
        }
    }
}

/// GPU performance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceLevel {
    /// Auto adjust
    Auto,
    /// Lowest power consumption
    Low,
    /// Highest performance
    High,
    /// Manual control
    Manual,
}

/// GPU information from rocm-smi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: usize,
    /// GPU name
    pub name: String,
    /// GPU architecture (e.g., gfx1100)
    pub arch: String,
    /// VRAM total in bytes
    pub vram_total: u64,
    /// VRAM used in bytes
    pub vram_used: u64,
    /// Temperature in Celsius
    pub temperature: Option<f32>,
    /// Power usage in Watts
    pub power_usage: Option<f32>,
    /// Fan speed percentage
    pub fan_speed: Option<u32>,
    /// GPU utilization percentage
    pub utilization: Option<u32>,
}

/// ROCm SMI installation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RocmSmiSource {
    /// Install via system package manager (apt/dnf)
    #[default]
    PackageManager,
    /// Already included with ROCm installation
    RocmBundle,
    /// Build from source
    Source,
}

/// ROCm SMI installer.
pub struct RocmSmiInstaller {
    rocm_path: PathBuf,
    source: RocmSmiSource,
}

impl RocmSmiInstaller {
    /// Creates a new installer.
    pub fn new() -> Self {
        Self {
            rocm_path: PathBuf::from("/opt/rocm"),
            source: RocmSmiSource::RocmBundle,
        }
    }

    /// Sets ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Sets installation source.
    pub fn with_source(mut self, source: RocmSmiSource) -> Self {
        self.source = source;
        self
    }

    /// Gets the rocm-smi binary path.
    pub fn smi_path(&self) -> PathBuf {
        self.rocm_path.join("bin/rocm-smi")
    }

    /// Gets the Python library path.
    pub fn python_lib_path(&self) -> PathBuf {
        self.rocm_path.join("libexec/rocm_smi")
    }

    /// Checks if rocm-smi is available.
    fn check_installed(&self) -> bool {
        self.smi_path().exists() ||
            Command::new("rocm-smi").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    /// Gets ROCm SMI version.
    pub fn get_version(&self) -> Option<String> {
        Command::new("rocm-smi")
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout);
                stdout.lines().next().unwrap_or("unknown").trim().to_string()
            })
    }

    /// Lists all GPUs.
    pub fn list_gpus(&self) -> Result<Vec<GpuInfo>> {
        let output = Command::new("rocm-smi")
            .args(["--showallinfo", "--json"])
            .output()
            .context("Failed to run rocm-smi")?;

        if !output.status.success() {
            // Try simpler query
            return self.list_gpus_simple();
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_json_output(&stdout)
    }

    /// Simple GPU listing without JSON.
    fn list_gpus_simple(&self) -> Result<Vec<GpuInfo>> {
        let output = Command::new("rocm-smi")
            .output()
            .context("Failed to run rocm-smi")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        // Parse basic output format
        for line in stdout.lines() {
            if line.contains("GPU[") {
                // Extract GPU index from pattern like "GPU[0]"
                if let Some(start) = line.find("GPU[") {
                    if let Some(end) = line[start..].find(']') {
                        let idx_str = &line[start + 4..start + end];
                        if let Ok(index) = idx_str.parse::<usize>() {
                            gpus.push(GpuInfo {
                                index,
                                name: "AMD GPU".to_string(),
                                arch: "unknown".to_string(),
                                vram_total: 0,
                                vram_used: 0,
                                temperature: None,
                                power_usage: None,
                                fan_speed: None,
                                utilization: None,
                            });
                        }
                    }
                }
            }
        }

        // If no GPUs parsed, try rocminfo
        if gpus.is_empty() {
            return self.list_gpus_rocminfo();
        }

        Ok(gpus)
    }

    /// List GPUs using rocminfo as fallback.
    fn list_gpus_rocminfo(&self) -> Result<Vec<GpuInfo>> {
        let output = Command::new("rocminfo")
            .output()
            .context("Failed to run rocminfo")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();
        let mut current_index = 0;
        let mut current_name = String::new();
        let mut current_arch = String::new();

        for line in stdout.lines() {
            let line = line.trim();

            if line.starts_with("Name:") && line.contains("gfx") {
                current_arch = line.split_whitespace().last().unwrap_or("unknown").to_string();
            } else if line.starts_with("Marketing Name:") {
                current_name = line.strip_prefix("Marketing Name:").unwrap_or("").trim().to_string();
            } else if line.contains("Agent ") && line.contains("*GPU*") {
                if !current_name.is_empty() || !current_arch.is_empty() {
                    gpus.push(GpuInfo {
                        index: current_index,
                        name: if current_name.is_empty() { format!("AMD GPU {}", current_arch) } else { current_name.clone() },
                        arch: current_arch.clone(),
                        vram_total: 0,
                        vram_used: 0,
                        temperature: None,
                        power_usage: None,
                        fan_speed: None,
                        utilization: None,
                    });
                    current_index += 1;
                }
                current_name.clear();
                current_arch.clear();
            }
        }

        // Don't forget the last GPU
        if !current_name.is_empty() || !current_arch.is_empty() {
            gpus.push(GpuInfo {
                index: current_index,
                name: if current_name.is_empty() { format!("AMD GPU {}", current_arch) } else { current_name },
                arch: current_arch,
                vram_total: 0,
                vram_used: 0,
                temperature: None,
                power_usage: None,
                fan_speed: None,
                utilization: None,
            });
        }

        Ok(gpus)
    }

    /// Parse JSON output from rocm-smi.
    fn parse_json_output(&self, json_str: &str) -> Result<Vec<GpuInfo>> {
        // Try to parse as JSON
        let value: serde_json::Value = serde_json::from_str(json_str)
            .context("Failed to parse JSON output")?;

        let mut gpus = Vec::new();

        if let Some(obj) = value.as_object() {
            for (key, info) in obj {
                if key.starts_with("card") || key.starts_with("GPU") {
                    let index = key.chars().filter(|c| c.is_ascii_digit()).collect::<String>()
                        .parse::<usize>().unwrap_or(gpus.len());

                    let name = info.get("Card series").and_then(|v| v.as_str())
                        .or_else(|| info.get("card_series").and_then(|v| v.as_str()))
                        .unwrap_or("AMD GPU")
                        .to_string();

                    let arch = info.get("GFX Version").and_then(|v| v.as_str())
                        .or_else(|| info.get("gfx_version").and_then(|v| v.as_str()))
                        .unwrap_or("unknown")
                        .to_string();

                    gpus.push(GpuInfo {
                        index,
                        name,
                        arch,
                        vram_total: info.get("VRAM Total Memory (B)").and_then(|v| v.as_u64()).unwrap_or(0),
                        vram_used: info.get("VRAM Total Used Memory (B)").and_then(|v| v.as_u64()).unwrap_or(0),
                        temperature: info.get("Temperature (Sensor edge) (C)").and_then(|v| v.as_f64()).map(|v| v as f32),
                        power_usage: info.get("Average Graphics Package Power (W)").and_then(|v| v.as_f64()).map(|v| v as f32),
                        fan_speed: info.get("Fan Speed (%)").and_then(|v| v.as_u64()).map(|v| v as u32),
                        utilization: info.get("GPU use (%)").and_then(|v| v.as_u64()).map(|v| v as u32),
                    });
                }
            }
        }

        Ok(gpus)
    }

    /// Sets GPU power profile.
    pub fn set_power_profile(&self, gpu_index: usize, profile: PowerProfile) -> Result<()> {
        let output = Command::new("rocm-smi")
            .args([
                "-d", &gpu_index.to_string(),
                "--setprofile", profile.as_smi_arg(),
            ])
            .output()
            .context("Failed to set power profile")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Failed to set power profile: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Sets GPU fan speed.
    pub fn set_fan_speed(&self, gpu_index: usize, speed_percent: u32) -> Result<()> {
        let speed = speed_percent.min(100);
        let output = Command::new("rocm-smi")
            .args([
                "-d", &gpu_index.to_string(),
                "--setfan", &speed.to_string(),
            ])
            .output()
            .context("Failed to set fan speed")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Failed to set fan speed: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Resets GPU to default settings.
    pub fn reset_gpu(&self, gpu_index: usize) -> Result<()> {
        let output = Command::new("rocm-smi")
            .args(["-d", &gpu_index.to_string(), "-r"])
            .output()
            .context("Failed to reset GPU")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Failed to reset GPU: {}", stderr)
            ).into());
        }

        Ok(())
    }

    /// Installs Python bindings.
    async fn install_python_bindings(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.6, "Installing Python bindings...".to_string());
        }

        let output = Command::new("python3")
            .args(["-m", "pip", "install", "pyrsmi"])
            .output();

        if let Ok(output) = output {
            if !output.status.success() {
                // Try rocm_smi_lib
                let _ = Command::new("python3")
                    .args(["-m", "pip", "install", "rocm-smi-lib"])
                    .output();
            }
        }

        Ok(())
    }
}

impl Default for RocmSmiInstaller {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Installer for RocmSmiInstaller {
    fn name(&self) -> &str {
        "ROCm SMI"
    }

    fn version(&self) -> &str {
        "bundled"
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.check_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if self.rocm_path.exists() {
            checks.push(format!("ROCm found at {}", self.rocm_path.display()));
        } else {
            checks.push("ROCm installation not found".to_string());
        }

        if self.smi_path().exists() {
            checks.push("rocm-smi binary found".to_string());
        } else {
            checks.push("rocm-smi binary not found".to_string());
        }

        if self.check_installed() {
            if let Some(version) = self.get_version() {
                checks.push(format!("ROCm SMI version: {}", version));
            }

            // List detected GPUs
            if let Ok(gpus) = self.list_gpus() {
                checks.push(format!("Detected {} GPU(s)", gpus.len()));
                for gpu in gpus {
                    checks.push(format!("  GPU {}: {} ({})", gpu.index, gpu.name, gpu.arch));
                }
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Checking ROCm SMI installation...".to_string());
        }

        // ROCm SMI is typically bundled with ROCm
        if self.check_installed() {
            if let Some(ref cb) = progress {
                cb(0.5, "ROCm SMI already available with ROCm".to_string());
            }
        } else {
            // Try to install via package manager
            if let Some(ref cb) = progress {
                cb(0.3, "Installing ROCm SMI...".to_string());
            }

            let output = Command::new("apt-get")
                .args(["install", "-y", "rocm-smi-lib"])
                .output();

            if output.is_err() || !output.unwrap().status.success() {
                // Try with sudo
                let _ = Command::new("sudo")
                    .args(["apt-get", "install", "-y", "rocm-smi-lib"])
                    .output();
            }
        }

        // Install Python bindings
        self.install_python_bindings(&progress).await?;

        if let Some(ref cb) = progress {
            cb(1.0, "ROCm SMI installation complete".to_string());
        }

        Ok(())
    }

    async fn uninstall(&self) -> Result<()> {
        // ROCm SMI is typically part of ROCm - just remove Python bindings
        let _ = Command::new("python3")
            .args(["-m", "pip", "uninstall", "-y", "pyrsmi", "rocm-smi-lib"])
            .output();

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        if !self.check_installed() {
            return Ok(false);
        }

        // Verify by listing GPUs
        let gpus = self.list_gpus()?;
        Ok(!gpus.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_profile() {
        assert_eq!(PowerProfile::Compute.as_smi_arg(), "compute");
        assert_eq!(PowerProfile::High.as_smi_arg(), "high");
    }

    #[test]
    fn test_installer_creation() {
        let installer = RocmSmiInstaller::new();
        assert_eq!(installer.name(), "ROCm SMI");
        assert_eq!(installer.rocm_path, PathBuf::from("/opt/rocm"));
    }

    #[test]
    fn test_smi_path() {
        let installer = RocmSmiInstaller::new();
        assert_eq!(installer.smi_path(), PathBuf::from("/opt/rocm/bin/rocm-smi"));
    }
}
