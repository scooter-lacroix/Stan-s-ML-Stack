//! Megatron-LM Installer
//!
//! NVIDIA Megatron-LM installer for ROCm with compatibility patches.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Megatron installation sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MegatronSource {
    /// Install from NVIDIA repository
    #[default]
    Nvidia,
    /// Install from AMD fork with ROCm support
    AmdFork,
    /// Build from source with patches
    Source,
}

/// Megatron version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegatronVersion {
    /// Version or branch name
    pub version: String,
    /// Git reference (tag, branch, or commit)
    pub git_ref: String,
    /// ROCm version compatibility
    pub rocm_version: String,
}

impl MegatronVersion {
    /// Creates a new Megatron version.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: version_str.clone(),
            version: version_str,
            rocm_version: rocm_version.into(),
        }
    }

    /// Sets the git reference.
    pub fn with_git_ref(mut self, git_ref: impl Into<String>) -> Self {
        self.git_ref = git_ref.into();
        self
    }
}

/// Megatron patch for ROCm compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegatronPatch {
    /// Patch name
    pub name: String,
    /// Description
    pub description: String,
    /// Files affected
    pub files: Vec<String>,
    /// Patch content
    pub content: String,
}

/// Megatron patch manager.
pub struct MegatronPatchManager {
    patches: HashMap<String, MegatronPatch>,
}

impl MegatronPatchManager {
    /// Creates a new patch manager with default patches.
    pub fn new() -> Self {
        let mut patches = HashMap::new();
        
        patches.insert(
            "rocm_fused_kernels".to_string(),
            MegatronPatch {
                name: "ROCm Fused Kernels".to_string(),
                description: "Enables fused kernels for ROCm".to_string(),
                files: vec!["megatron/fused_kernels/__init__.py".to_string()],
                content: Self::fused_kernels_patch(),
            },
        );
        
        patches.insert(
            "rocm_initialize".to_string(),
            MegatronPatch {
                name: "ROCm Initialize".to_string(),
                description: "Fixes device initialization for ROCm".to_string(),
                files: vec!["megatron/initialize.py".to_string()],
                content: Self::initialize_patch(),
            },
        );
        
        Self { patches }
    }

    fn fused_kernels_patch() -> String {
        r#"--- a/megatron/fused_kernels/__init__.py
+++ b/megatron/fused_kernels/__init__.py
@@ -10,6 +10,10 @@
 import torch
 
 def load_fused_kernels():
+    # ROCm doesn't require fused kernel compilation
+    if torch.version.hip:
+        return
+    
     # Load CUDA fused kernels
     try:
         from . import fused_mix_prec_layer_norm_cuda
"#
        .to_string()
    }

    fn initialize_patch() -> String {
        r#"--- a/megatron/initialize.py
+++ b/megatron/initialize.py
@@ -50,6 +50,10 @@
     else:
         torch.cuda.set_device(local_rank)
 
+    # ROCm-specific initialization
+    if torch.version.hip:
+        torch.backends.cuda.enable_flash_sdp(False)
+
 def _init_autoresume():
     '''Set up autoresume for checkpointing'''
     pass
"#
        .to_string()
    }

    /// Gets a patch by name.
    pub fn get_patch(&self, name: &str) -> Option<&MegatronPatch> {
        self.patches.get(name)
    }

    /// Applies a patch.
    pub fn apply_patch(&self, name: &str, source_dir: &Path) -> Result<()> {
        let patch = self.get_patch(name)
            .ok_or_else(|| InstallerError::InstallationFailed(format!("Patch {} not found", name)))?;

        let patch_file = std::env::temp_dir().join(format!("megatron_{}.patch", name));
        std::fs::write(&patch_file, &patch.content)?;

        let output = Command::new("git")
            .args(["apply", patch_file.to_str().unwrap()])
            .current_dir(source_dir)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Failed to apply patch {}: {}", name, stderr)
            ).into());
        }

        Ok(())
    }

    /// Applies all patches.
    pub fn apply_all_patches(&self, source_dir: &Path) -> Result<Vec<String>> {
        let mut applied = Vec::new();
        for name in self.patches.keys() {
            self.apply_patch(name, source_dir)?;
            applied.push(name.clone());
        }
        Ok(applied)
    }
}

impl Default for MegatronPatchManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Megatron-LM installer.
pub struct MegatronInstaller {
    version: MegatronVersion,
    source: MegatronSource,
    rocm_path: String,
    patch_manager: MegatronPatchManager,
    apply_patches: bool,
}

impl MegatronInstaller {
    /// Creates a new Megatron installer.
    pub fn new(version: MegatronVersion, source: MegatronSource) -> Self {
        Self {
            version,
            source,
            rocm_path: "/opt/rocm".to_string(),
            patch_manager: MegatronPatchManager::new(),
            apply_patches: true,
        }
    }

    /// Sets the ROCm path.
    pub fn with_rocm_path(mut self, path: impl Into<String>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Sets whether to apply patches.
    pub fn with_patches(mut self, apply: bool) -> Self {
        self.apply_patches = apply;
        self
    }

    /// Checks if Megatron is installed.
    pub fn is_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import megatron; print('OK')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Gets the repository URL based on source.
    fn repo_url(&self) -> &'static str {
        match self.source {
            MegatronSource::Nvidia | MegatronSource::Source => {
                "https://github.com/NVIDIA/Megatron-LM.git"
            }
            MegatronSource::AmdFork => "https://github.com/ROCm/Megatron-LM.git",
        }
    }

    /// Installs Megatron from source.
    async fn install_from_source(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing Megatron-LM...".to_string());
        }

        let repo_url = self.repo_url();
        let clone_dir = std::env::temp_dir().join("megatron-build");

        // Clone
        if let Some(ref cb) = progress {
            cb(0.2, "Cloning Megatron repository...".to_string());
        }

        let output = Command::new("git")
            .args([
                "clone",
                "--branch",
                &self.version.git_ref,
                "--depth",
                "1",
                repo_url,
                clone_dir.to_str().unwrap(),
            ])
            .output()
            .context("Failed to clone Megatron repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        // Apply patches for NVIDIA source
        if self.apply_patches && matches!(self.source, MegatronSource::Nvidia | MegatronSource::Source) {
            if let Some(ref cb) = progress {
                cb(0.4, "Applying ROCm patches...".to_string());
            }

            self.patch_manager.apply_all_patches(&clone_dir)?;
        }

        // Install
        if let Some(ref cb) = progress {
            cb(0.6, "Installing Megatron-LM...".to_string());
        }

        std::env::set_var("ROCM_HOME", &self.rocm_path);

        let output = Command::new("pip")
            .args(["install", "-e", "."])
            .current_dir(&clone_dir)
            .output()
            .context("Failed to install Megatron")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Megatron install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "Megatron-LM installation complete".to_string());
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl Installer for MegatronInstaller {
    fn name(&self) -> &str {
        "Megatron-LM"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 not installed".to_string());
        }

        if Command::new("pip").arg("--version").output().is_err() {
            checks.push("pip not installed".to_string());
        }

        if !Path::new(&self.rocm_path).exists() {
            checks.push(format!("ROCm not found at {}", self.rocm_path));
        }

        if Command::new("python3")
            .args(["-c", "import torch"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            checks.push("PyTorch not installed".to_string());
        }

        if self.is_installed() {
            checks.push("Megatron already installed".to_string());
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        self.install_from_source(&progress).await
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "megatron-lm"])
            .output()
            .context("Failed to uninstall Megatron")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_installed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_megatron_version_creation() {
        let version = MegatronVersion::new("main", "6.4");
        assert_eq!(version.version, "main");
        assert_eq!(version.git_ref, "main");
    }

    #[test]
    fn test_megatron_patch_manager() {
        let manager = MegatronPatchManager::new();
        assert!(!manager.patches.is_empty());
    }

    #[test]
    fn test_megatron_installer_creation() {
        let version = MegatronVersion::new("main", "6.4");
        let installer = MegatronInstaller::new(version, MegatronSource::AmdFork);
        assert_eq!(installer.name(), "Megatron-LM");
    }

    #[test]
    fn test_megatron_repo_url() {
        let version = MegatronVersion::new("main", "6.4");
        
        let nvidia = MegatronInstaller::new(version.clone(), MegatronSource::Nvidia);
        assert!(nvidia.repo_url().contains("NVIDIA"));
        
        let amd = MegatronInstaller::new(version, MegatronSource::AmdFork);
        assert!(amd.repo_url().contains("ROCm"));
    }
}
