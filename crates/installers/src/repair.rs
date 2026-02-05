use anyhow::Result;
use crate::package_manager::{PackageManager, PackageRequest, PackageType};
use mlstack_hardware::discovery::HardwareDiscovery;
use mlstack_env::manager::EnvironmentManager;
use mlstack_env::config::EnvConfig;
use std::process::Command;

/// Module for automated repair of the ML stack.
pub struct RepairModule;

impl RepairModule {
    /// Creates a new repair module.
    pub fn new() -> Self {
        Self
    }

    /// Repairs environment variables by detecting discrete GPUs and updating configuration.
    pub async fn repair_environment(&self) -> Result<()> {
        let discovery = HardwareDiscovery::new();
        let hip_devices = discovery.hip_visible_devices()?;
        
        let config = EnvConfig::default();
        let env_manager = EnvironmentManager::new(config);
        
        // Update current process environment
        std::env::set_var("HIP_VISIBLE_DEVICES", &hip_devices);
        std::env::set_var("CUDA_VISIBLE_DEVICES", &hip_devices);
        std::env::set_var("PYTORCH_ROCM_DEVICE", &hip_devices);
        
        // Persist to user environment file
        env_manager.persist_user_env()?;
        Ok(())
    }

    /// Fixes PyTorch by reinstalling ROCm version if CUDA is detected.
    pub async fn fix_pytorch(&self) -> Result<()> {
        // Run a quick Python check
        let output = Command::new("python3")
            .args(["-c", "import torch; print(torch.version.hip)"])
            .output()?;
            
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // If torch.version.hip is None, it's not a ROCm build
        if stdout.trim() == "None" {
            let mut pm = PackageManager::new();
            let request = PackageRequest::new(PackageType::PyTorch, "latest")
                .with_force(true);
            
            pm.install_packages(&[request], &None).await;
        }
        Ok(())
    }

    /// Cleans up broken or partial installations.
    pub async fn clean_broken_installs(&self) -> Result<()> {
        // Placeholder for cleanup logic: remove build directories, clear caches
        Ok(())
    }
}

impl Default for RepairModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_module_creation() {
        let _module = RepairModule::new();
    }
}
