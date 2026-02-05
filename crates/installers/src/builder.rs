//! Source Builder Framework
//!
//! Generic framework for building ML components from source with ROCm support.

use crate::common::{InstallerError, ProgressCallback};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build system types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildSystem {
    /// CMake-based build
    CMake,
    /// Python setuptools
    Setuptools,
    /// Python poetry
    Poetry,
    /// Make-based build
    Make,
    /// Custom build script
    Custom,
}

/// Source repository information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRepository {
    /// Repository URL
    pub url: String,
    /// Git branch, tag, or commit
    pub git_ref: String,
    /// Clone depth (0 for full clone)
    pub clone_depth: usize,
    /// Whether to clone submodules
    pub recursive: bool,
}

impl SourceRepository {
    /// Creates a new source repository.
    pub fn new(url: impl Into<String>, git_ref: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            git_ref: git_ref.into(),
            clone_depth: 1,
            recursive: false,
        }
    }

    /// Sets clone depth.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.clone_depth = depth;
        self
    }

    /// Sets recursive clone.
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Build configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build system to use
    pub build_system: BuildSystem,
    /// Build directory (relative to source)
    pub build_dir: PathBuf,
    /// CMake arguments (for CMake builds)
    pub cmake_args: Vec<String>,
    /// Make arguments
    pub make_args: Vec<String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Number of parallel jobs
    pub parallel_jobs: usize,
    /// Build type (Release, Debug)
    pub build_type: String,
}

impl BuildConfig {
    /// Creates a new build configuration.
    pub fn new(build_system: BuildSystem) -> Self {
        Self {
            build_system,
            build_dir: PathBuf::from("build"),
            cmake_args: Vec::new(),
            make_args: Vec::new(),
            env_vars: HashMap::new(),
            parallel_jobs: num_cpus::get(),
            build_type: "Release".to_string(),
        }
    }

    /// Adds a CMake argument.
    pub fn with_cmake_arg(mut self, arg: impl Into<String>) -> Self {
        self.cmake_args.push(arg.into());
        self
    }

    /// Adds a make argument.
    pub fn with_make_arg(mut self, arg: impl Into<String>) -> Self {
        self.make_args.push(arg.into());
        self
    }

    /// Sets an environment variable.
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Sets parallel jobs.
    pub fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.parallel_jobs = jobs;
        self
    }

    /// Sets build type.
    pub fn with_build_type(mut self, build_type: impl Into<String>) -> Self {
        self.build_type = build_type.into();
        self
    }
}

/// Source builder for generic component builds.
pub struct SourceBuilder {
    name: String,
    repository: SourceRepository,
    build_config: BuildConfig,
    install_command: Vec<String>,
}

impl SourceBuilder {
    /// Creates a new source builder.
    pub fn new(name: impl Into<String>, repository: SourceRepository) -> Self {
        Self {
            name: name.into(),
            repository,
            build_config: BuildConfig::new(BuildSystem::CMake),
            install_command: vec!["pip".to_string(), "install".to_string(), "-e".to_string(), ".".to_string()],
        }
    }

    /// Sets the build configuration.
    pub fn with_build_config(mut self, config: BuildConfig) -> Self {
        self.build_config = config;
        self
    }

    /// Sets the install command.
    pub fn with_install_command(mut self, command: Vec<String>) -> Self {
        self.install_command = command;
        self
    }

    /// Clones the repository.
    pub async fn clone_repository(&self, progress: &Option<ProgressCallback>) -> Result<PathBuf> {
        if let Some(ref cb) = progress {
            cb(0.1, format!("Cloning {} repository...", self.name));
        }

        let clone_dir = std::env::temp_dir().join(format!("{}-build", self.name.to_lowercase().replace(" ", "-")));

        let mut args = vec![
            "clone".to_string(),
            "--branch".to_string(),
            self.repository.git_ref.clone(),
        ];

        if self.repository.clone_depth > 0 {
            args.push("--depth".to_string());
            args.push(self.repository.clone_depth.to_string());
        }

        if self.repository.recursive {
            args.push("--recursive".to_string());
        }

        args.push(self.repository.url.clone());
        args.push(clone_dir.to_str().unwrap().to_string());

        let output = Command::new("git")
            .args(&args)
            .output()
            .context(format!("Failed to clone {} repository", self.name))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Git clone failed: {}", stderr)
            ).into());
        }

        Ok(clone_dir)
    }

    /// Configures the build.
    pub async fn configure_build(&self, source_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.3, format!("Configuring {} build...", self.name));
        }

        // Set environment variables
        for (key, value) in &self.build_config.env_vars {
            std::env::set_var(key, value);
        }

        match self.build_config.build_system {
            BuildSystem::CMake => {
                let build_dir = source_dir.join(&self.build_config.build_dir);
                std::fs::create_dir_all(&build_dir)?;

                let mut args = vec![
                    "-S".to_string(),
                    source_dir.to_str().unwrap().to_string(),
                    "-B".to_string(),
                    build_dir.to_str().unwrap().to_string(),
                    format!("-DCMAKE_BUILD_TYPE={}", self.build_config.build_type),
                ];
                args.extend(self.build_config.cmake_args.clone());

                let output = Command::new("cmake")
                    .args(&args)
                    .output()
                    .context("Failed to configure CMake build")?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(InstallerError::InstallationFailed(
                        format!("CMake configuration failed: {}", stderr)
                    ).into());
                }
            }
            _ => {
                // Other build systems don't need explicit configuration
            }
        }

        Ok(())
    }

    /// Builds the component.
    pub async fn build(&self, source_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.5, format!("Building {}...", self.name));
        }

        match self.build_config.build_system {
            BuildSystem::CMake => {
                let build_dir = source_dir.join(&self.build_config.build_dir);
                
                let mut args = vec![
                    "--build".to_string(),
                    build_dir.to_str().unwrap().to_string(),
                    "--parallel".to_string(),
                    self.build_config.parallel_jobs.to_string(),
                ];
                args.extend(self.build_config.make_args.clone());

                let output = Command::new("cmake")
                    .args(&args)
                    .output()
                    .context("Failed to build with CMake")?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(InstallerError::InstallationFailed(
                        format!("Build failed: {}", stderr)
                    ).into());
                }
            }
            BuildSystem::Make => {
                let output = Command::new("make")
                    .args(["-j", &self.build_config.parallel_jobs.to_string()])
                    .current_dir(source_dir)
                    .output()
                    .context("Failed to build with Make")?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(InstallerError::InstallationFailed(
                        format!("Make failed: {}", stderr)
                    ).into());
                }
            }
            _ => {
                // Other build systems handle building during install
            }
        }

        Ok(())
    }

    /// Installs the built component.
    pub async fn install(&self, source_dir: &Path, progress: &Option<ProgressCallback>) -> Result<()> {
        if let Some(ref cb) = progress {
            cb(0.8, format!("Installing {}...", self.name));
        }

        let output = Command::new(&self.install_command[0])
            .args(&self.install_command[1..])
            .current_dir(source_dir)
            .output()
            .context("Failed to install component")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(InstallerError::InstallationFailed(
                format!("Install failed: {}", stderr)
            ).into());
        }

        if let Some(ref cb) = progress {
            cb(1.0, format!("{} installation complete", self.name));
        }

        Ok(())
    }

    /// Full build and install process.
    pub async fn build_and_install(&self, progress: Option<ProgressCallback>) -> Result<()> {
        let source_dir = self.clone_repository(&progress).await?;
        self.configure_build(&source_dir, &progress).await?;
        self.build(&source_dir, &progress).await?;
        self.install(&source_dir, &progress).await?;
        Ok(())
    }
}

/// Predefined build configurations for common ML components.
pub mod presets {
    use super::*;

    /// Creates a CMake build config for ROCm components.
    pub fn rocm_cmake_config(rocm_path: &str) -> BuildConfig {
        BuildConfig::new(BuildSystem::CMake)
            .with_cmake_arg(format!("-DROCM_HOME={}", rocm_path))
            .with_cmake_arg("-DCMAKE_PREFIX_PATH=/opt/rocm")
            .with_env_var("ROCM_HOME", rocm_path)
            .with_env_var("HIP_PATH", format!("{}/hip", rocm_path))
    }

    /// Creates a Python setuptools config.
    pub fn python_setuptools_config() -> BuildConfig {
        BuildConfig::new(BuildSystem::Setuptools)
    }

    /// Creates a PyTorch extension config.
    pub fn pytorch_extension_config(rocm_path: &str) -> BuildConfig {
        BuildConfig::new(BuildSystem::Setuptools)
            .with_env_var("ROCM_HOME", rocm_path)
            .with_env_var("TORCH_CUDA_ARCH_LIST", "gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101")
            .with_env_var("MAX_JOBS", num_cpus::get().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_repository_creation() {
        let repo = SourceRepository::new("https://github.com/test/repo.git", "main");
        assert_eq!(repo.url, "https://github.com/test/repo.git");
        assert_eq!(repo.git_ref, "main");
        assert_eq!(repo.clone_depth, 1);
    }

    #[test]
    fn test_source_repository_builder() {
        let repo = SourceRepository::new("https://github.com/test/repo.git", "main")
            .with_depth(0)
            .with_recursive(true);
        assert_eq!(repo.clone_depth, 0);
        assert!(repo.recursive);
    }

    #[test]
    fn test_build_config_creation() {
        let config = BuildConfig::new(BuildSystem::CMake);
        assert_eq!(config.build_system, BuildSystem::CMake);
        assert_eq!(config.build_type, "Release");
        assert!(config.parallel_jobs > 0);
    }

    #[test]
    fn test_build_config_builder() {
        let config = BuildConfig::new(BuildSystem::CMake)
            .with_cmake_arg("-DTEST=ON")
            .with_env_var("KEY", "value")
            .with_parallel_jobs(4);
        
        assert!(config.cmake_args.contains(&"-DTEST=ON".to_string()));
        assert_eq!(config.env_vars.get("KEY"), Some(&"value".to_string()));
        assert_eq!(config.parallel_jobs, 4);
    }

    #[test]
    fn test_source_builder_creation() {
        let repo = SourceRepository::new("https://github.com/test/repo.git", "main");
        let builder = SourceBuilder::new("TestComponent", repo);
        assert_eq!(builder.name, "TestComponent");
    }

    #[test]
    fn test_rocm_cmake_preset() {
        let config = presets::rocm_cmake_config("/opt/rocm");
        assert_eq!(config.build_system, BuildSystem::CMake);
        assert!(config.cmake_args.iter().any(|a| a.contains("ROCM_HOME")));
    }

    #[test]
    fn test_python_setuptools_preset() {
        let config = presets::python_setuptools_config();
        assert_eq!(config.build_system, BuildSystem::Setuptools);
    }
}
