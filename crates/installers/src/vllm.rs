//! vLLM Installer
//!
//! vLLM inference engine installer for ROCm with patching support.

use crate::common::{Installer, InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// vLLM installation sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VllmSource {
    /// Install from PyPI
    #[default]
    Pypi,
    /// Build from source
    Source,
    /// Install with ROCm patches
    RocmPatched,
}

/// vLLM version configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmVersion {
    /// Version string
    pub version: String,
    /// Git tag or branch
    pub git_ref: String,
    /// ROCm version compatibility
    pub rocm_version: String,
}

impl VllmVersion {
    /// Creates a new vLLM version.
    pub fn new(version: impl Into<String>, rocm_version: impl Into<String>) -> Self {
        let version_str = version.into();
        Self {
            git_ref: format!("v{}", version_str),
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

/// Patch definition for vLLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmPatch {
    /// Patch name
    pub name: String,
    /// Patch description
    pub description: String,
    /// Files to patch
    pub files: Vec<String>,
    /// Patch content (unified diff format)
    pub content: String,
    /// Whether this patch is required for ROCm
    pub required_for_rocm: bool,
}

/// vLLM patch manager.
pub struct VllmPatchManager {
    patches: HashMap<String, VllmPatch>,
}

impl VllmPatchManager {
    /// Creates a new patch manager with default ROCm patches.
    pub fn new() -> Self {
        let mut patches = HashMap::new();
        
        // Add default ROCm patches
        patches.insert(
            "rocm_gfx_arch".to_string(),
            VllmPatch {
                name: "ROCm GFX Architecture Support".to_string(),
                description: "Adds support for RDNA 3/4 GFX architectures".to_string(),
                files: vec!["setup.py".to_string(), "pyproject.toml".to_string()],
                content: Self::rocm_gfx_patch_content(),
                required_for_rocm: true,
            },
        );
        
        patches.insert(
            "rocm_custom_paged_attn".to_string(),
            VllmPatch {
                name: "ROCm Custom Paged Attention".to_string(),
                description: "Enables custom paged attention kernels for ROCm".to_string(),
                files: vec!["vllm/attention/selector.py".to_string()],
                content: Self::paged_attn_patch_content(),
                required_for_rocm: true,
            },
        );
        
        Self { patches }
    }

    /// Returns the GFX architecture patch content.
    fn rocm_gfx_patch_content() -> String {
        r#"--- a/setup.py
+++ b/setup.py
@@ -100,6 +100,10 @@
     # ROCm support
     if torch.version.hip:
         extra_compile_args["cxx"] = ["-DUSE_ROCM"]
+        # Add RDNA 3/4 support
+        extra_compile_args["cxx"].extend([
+            "-DHIPBLAS_V2",
+        ])
     
     return extra_compile_args
"#
        .to_string()
    }

    /// Returns the paged attention patch content.
    fn paged_attn_patch_content() -> String {
        r#"--- a/vllm/attention/selector.py
+++ b/vllm/attention/selector.py
@@ -50,6 +50,9 @@
     if torch.version.hip:
         # Use ROCm-optimized attention
         from vllm.attention.backends.rocm_flash_attn import ROCmFlashAttentionBackend
+        # Enable custom paged attention for RDNA 3/4
+        if os.environ.get("HIP_VISIBLE_DEVICES"):
+            return ROCmFlashAttentionBackend()
         return ROCmFlashAttentionBackend()
     
     # CUDA path
"#
        .to_string()
    }

    /// Gets a patch by name.
    pub fn get_patch(&self, name: &str) -> Option<&VllmPatch> {
        self.patches.get(name)
    }

    /// Returns all required ROCm patches.
    pub fn required_rocm_patches(&self) -> Vec<&VllmPatch> {
        self.patches
            .values()
            .filter(|p| p.required_for_rocm)
            .collect()
    }

    /// Applies a patch to the source directory.
    pub fn apply_patch(&self, patch_name: &str, source_dir: &Path) -> Result<()> {
        let patch = self
            .get_patch(patch_name)
            .ok_or_else(|| InstallerError::InstallationFailed(format!("Patch {} not found", patch_name)))?;

        // Write patch to temporary file
        let patch_file = std::env::temp_dir().join(format!("{}.patch", patch_name));
        std::fs::write(&patch_file, &patch.content)
            .context("Failed to write patch file")?;

        // Apply patch using git apply
        let output = Command::new("git")
            .args(["apply", patch_file.to_str().unwrap()])
            .current_dir(source_dir)
            .output()
            .context("Failed to apply patch")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Failed to apply patch {}: {}", patch_name, stderr)
            ).into());
        }

        Ok(())
    }

    /// Applies all required ROCm patches.
    pub fn apply_rocm_patches(&self, source_dir: &Path) -> Result<Vec<String>> {
        let mut applied = Vec::new();
        
        for patch in self.required_rocm_patches() {
            self.apply_patch(&patch.name, source_dir)?;
            applied.push(patch.name.clone());
        }
        
        Ok(applied)
    }
}

impl Default for VllmPatchManager {
    fn default() -> Self {
        Self::new()
    }
}

/// vLLM installer for ROCm.
pub struct VllmInstaller {
    version: VllmVersion,
    source: VllmSource,
    rocm_path: String,
    patch_manager: VllmPatchManager,
    apply_patches: bool,
}

impl VllmInstaller {
    /// Creates a new vLLM installer.
    pub fn new(version: VllmVersion, source: VllmSource) -> Self {
        Self {
            version,
            source,
            rocm_path: "/opt/rocm".to_string(),
            patch_manager: VllmPatchManager::new(),
            apply_patches: true,
        }
    }

    /// Sets the ROCm installation path.
    pub fn with_rocm_path(mut self, path: impl Into<String>) -> Self {
        self.rocm_path = path.into();
        self
    }

    /// Sets whether to apply ROCm patches.
    pub fn with_patches(mut self, apply: bool) -> Self {
        self.apply_patches = apply;
        self
    }

    /// Checks if vLLM is installed.
    pub fn is_installed(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import vllm; print(vllm.__version__)"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Checks if vLLM has ROCm support.
    pub fn has_rocm_support(&self) -> bool {
        Command::new("python3")
            .args(["-c", "import vllm; from vllm.utils import is_hip; print(is_hip())"])
            .output()
            .map(|o| {
                if o.status.success() {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    stdout.trim() == "True"
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    /// Installs vLLM from PyPI.
    async fn install_from_pypi(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Installing vLLM from PyPI...".to_string());
        }

        let output = Command::new("pip")
            .args(["install", &format!("vllm=={}", self.version.version)])
            .output()
            .context("Failed to run pip install for vLLM")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("vLLM pip install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "vLLM installation complete".to_string());
        }

        Ok(())
    }

    /// Builds vLLM from source with optional patches.
    async fn install_from_source(&self, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.1, "Building vLLM from source...".to_string());
        }

        // Clone repository
        let repo_url = "https://github.com/vllm-project/vllm.git";
        let clone_dir = std::env::temp_dir().join("vllm-build");

        if let Some(ref cb) = progress {
            cb(0.2, "Cloning vLLM repository...".to_string());
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
            .context("Failed to clone vLLM repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        // Apply patches if enabled
        if self.apply_patches {
            if let Some(ref cb) = progress {
                cb(0.3, "Applying ROCm patches...".to_string());
            }

            let applied = self.patch_manager.apply_rocm_patches(&clone_dir)?;
            
            if let Some(ref cb) = progress {
                cb(0.4, format!("Applied {} patches", applied.len()));
            }
        }

        if let Some(ref cb) = progress {
            cb(0.5, "Building vLLM with ROCm support...".to_string());
        }

        // Set environment variables
        std::env::set_var("ROCM_HOME", &self.rocm_path);
        std::env::set_var("HIP_PATH", format!("{}/hip", self.rocm_path));
        std::env::set_var("VLLM_TARGET_DEVICE", "rocm");

        // Build and install
        let output = Command::new("pip")
            .args(["install", "-e", "."])
            .current_dir(&clone_dir)
            .output()
            .context("Failed to build vLLM from source")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("vLLM build failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, "vLLM source build complete".to_string());
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl Installer for VllmInstaller {
    fn name(&self) -> &str {
        "vLLM"
    }

    fn version(&self) -> &str {
        &self.version.version
    }

    async fn is_installed(&self) -> Result<bool> {
        Ok(self.is_installed())
    }

    async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut checks = Vec::new();

        // Check Python
        if Command::new("python3").arg("--version").output().is_err() {
            checks.push("Python 3 is not installed".to_string());
        }

        // Check pip
        if Command::new("pip").arg("--version").output().is_err() {
            checks.push("pip is not installed".to_string());
        }

        // Check ROCm
        if !Path::new(&self.rocm_path).exists() {
            checks.push(format!("ROCm not found at {}", self.rocm_path));
        }

        // Check PyTorch
        if Command::new("python3")
            .args(["-c", "import torch"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            checks.push("PyTorch is not installed (required for vLLM)".to_string());
        }

        // Check for ROCm PyTorch
        let is_rocm_pytorch = Command::new("python3")
            .args(["-c", "import torch; print(torch.version.hip)"])
            .output()
            .map(|o| {
                if o.status.success() {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    !stdout.trim().is_empty() && stdout.trim() != "None"
                } else {
                    false
                }
            })
            .unwrap_or(false);

        if !is_rocm_pytorch {
            checks.push("WARNING: CUDA PyTorch detected - vLLM requires ROCm PyTorch".to_string());
        }

        // Check existing vLLM
        if self.is_installed() {
            checks.push("vLLM is already installed".to_string());
            if !self.has_rocm_support() {
                checks.push("WARNING: Existing vLLM may not have ROCm support".to_string());
            }
        }

        Ok(checks)
    }

    async fn install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        match self.source {
            VllmSource::Pypi => self.install_from_pypi(&progress).await,
            VllmSource::Source | VllmSource::RocmPatched => {
                self.install_from_source(&progress).await
            }
        }
    }

    async fn uninstall(&self) -> Result<()> {
        let output = Command::new("pip")
            .args(["uninstall", "-y", "vllm"])
            .output()
            .context("Failed to uninstall vLLM")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(stderr.to_string()).into());
        }

        Ok(())
    }

    async fn verify(&self) -> Result<bool> {
        Ok(self.is_installed() && self.has_rocm_support())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_version_creation() {
        let version = VllmVersion::new("0.4.0", "6.4");
        assert_eq!(version.version, "0.4.0");
        assert_eq!(version.git_ref, "v0.4.0");
        assert_eq!(version.rocm_version, "6.4");
    }

    #[test]
    fn test_vllm_version_with_git_ref() {
        let version = VllmVersion::new("0.4.0", "6.4")
            .with_git_ref("main");
        assert_eq!(version.git_ref, "main");
    }

    #[test]
    fn test_vllm_patch_manager_creation() {
        let manager = VllmPatchManager::new();
        assert!(!manager.patches.is_empty());
    }

    #[test]
    fn test_vllm_required_rocm_patches() {
        let manager = VllmPatchManager::new();
        let required = manager.required_rocm_patches();
        assert!(!required.is_empty());
        assert!(required.iter().all(|p| p.required_for_rocm));
    }

    #[test]
    fn test_vllm_installer_creation() {
        let version = VllmVersion::new("0.4.0", "6.4");
        let installer = VllmInstaller::new(version, VllmSource::Pypi);
        assert_eq!(installer.name(), "vLLM");
        assert!(installer.apply_patches);
    }

    #[test]
    fn test_vllm_installer_builder() {
        let version = VllmVersion::new("0.4.0", "6.4");
        let installer = VllmInstaller::new(version, VllmSource::RocmPatched)
            .with_rocm_path("/custom/rocm")
            .with_patches(false);
        
        assert_eq!(installer.rocm_path, "/custom/rocm");
        assert!(!installer.apply_patches);
    }
}
